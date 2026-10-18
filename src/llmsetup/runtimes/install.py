"""A user-local CPU fallback. No sudo, shell installers, or PATH changes.

Upstream release b11669, with the SHA256 values published alongside its assets:
https://github.com/ggml-org/llama.cpp/releases/expanded_assets/b11669
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from llmsetup.types import HardwareProfile
from llmsetup.util.errors import ProvisionError
from llmsetup.util.fs import app_dir
from llmsetup.util.log import Console
from llmsetup.util.proc import run

BUILD = "b11669"
ASSETS = {
    ("linux", "x86_64"): ("ubuntu-x64.tar.gz", "cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac"),
    ("linux", "arm64"): ("ubuntu-arm64.tar.gz", "9ac6fda2058d35677608a06f751a50122ddfa4ce0ddb7f084466b8e79b6d4a99"),
    ("darwin", "x86_64"): ("macos-x64.tar.gz", "cab28238d78158109b48deb27b05d50e0dcef56275d2e76da9ca8281526c7532"),
    ("darwin", "arm64"): ("macos-arm64.tar.gz", "dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd"),
    ("windows", "x86_64"): ("win-cpu-x64.zip", "37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b"),
    ("windows", "arm64"): ("win-cpu-arm64.zip", "9cff39ffed17edcd03a7d876073f74d7b48cd43a1e5eafdd59ee094185ad871b"),
}

#: GPU-capable variants, one self-contained archive (the CUDA builds need a
#: separate cudart download and a matching driver; Vulkan ships everything and
#: runs on any recent NVIDIA/AMD/Intel card). Used only when a GPU is present.
GPU_ASSETS = {
    ("windows", "x86_64"): ("vulkan", "win-vulkan-x64.zip",
                            "2c07574e7bde71321cb785282b20cc821e951132149eeceeea32ddebdbf0598e"),
}
_MAX_ARCHIVE = 512 * 1024 ** 2
_MAX_EXTRACTED = 2 * 1024 ** 3


def _root(hardware: HardwareProfile) -> Path:
    return app_dir() / "runtimes" / f"llamacpp-{BUILD}" / f"{hardware.os}-{hardware.arch}"


def desired_flavor(hardware: HardwareProfile) -> str:
    """The managed build this machine should have: GPU-capable when there is a
    GPU and a self-contained GPU build exists for the platform, else CPU."""
    if hardware.gpus and (hardware.os, hardware.arch) in GPU_ASSETS:
        return GPU_ASSETS[(hardware.os, hardware.arch)][0]
    return "cpu"


def managed_flavor(hardware: Optional[HardwareProfile] = None) -> Optional[str]:
    """The flavor of the installed managed build. None when nothing is
    installed; installs predating flavor markers are the CPU build."""
    if hardware is None:
        import platform

        from llmsetup.hardware.cpu import normalized_arch

        hardware = HardwareProfile(os=platform.system().lower(), arch=normalized_arch())
    marker = _root(hardware) / ".flavor"
    try:
        return marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        return "cpu" if managed_binary(hardware) is not None else None


def managed_binary(hardware: Optional[HardwareProfile] = None) -> Optional[Path]:
    if hardware is None:
        import platform

        from llmsetup.hardware.cpu import normalized_arch

        hardware = HardwareProfile(os=platform.system().lower(), arch=normalized_arch())
    root = _root(hardware)
    name = "llama-server.exe" if hardware.os == "windows" else "llama-server"
    return next(root.rglob(name), None) if root.is_dir() else None


def _safe_name(name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ProvisionError(f"unsafe runtime archive path: {name!r}")
    return Path(*path.parts)


def _unpack(archive: Path, dest: Path) -> None:
    """Write regular files only; resolve internal tar links to file contents."""
    total = 0
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > 10000:
                raise ProvisionError("runtime archive has too many entries")
            for member in members:
                target = dest / _safe_name(member.filename)
                total += member.file_size
                if total > _MAX_EXTRACTED or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ProvisionError("unsafe runtime archive contents")
                if member.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
    else:
        with tarfile.open(archive) as bundle:
            tar_entries = bundle.getmembers()
            if len(tar_entries) > 10000:
                raise ProvisionError("runtime archive has too many entries")
            by_name = {entry.name: entry for entry in tar_entries}
            for entry in tar_entries:
                target = dest / _safe_name(entry.name)
                if entry.isdir():
                    continue
                original = entry
                visited = set()
                while entry.issym() or entry.islnk():
                    if entry.name in visited:
                        raise ProvisionError("runtime archive contains a link cycle")
                    visited.add(entry.name)
                    link = (str(PurePosixPath(entry.name).parent / entry.linkname)
                            if entry.issym() else entry.linkname)
                    _safe_name(link)
                    if link not in by_name:
                        raise ProvisionError("runtime archive contains a dangling link")
                    entry = by_name[link]
                if not entry.isfile():
                    raise ProvisionError("runtime archive contains a special file")
                total += entry.size
                if total > _MAX_EXTRACTED:
                    raise ProvisionError("runtime archive is too large")
                target.parent.mkdir(parents=True, exist_ok=True)
                file_source = bundle.extractfile(entry)
                if file_source is None:
                    raise ProvisionError("could not read runtime archive entry")
                with file_source, target.open("wb") as out:
                    shutil.copyfileobj(file_source, out)
                target.chmod(original.mode & 0o777)


def install_llamacpp(hardware: HardwareProfile, console: Console,
                     flavor: Optional[str] = None) -> Path:
    """Fetch a pinned, checksum-verified portable build and check it can execute."""
    flavor = flavor or desired_flavor(hardware)
    gpu_entry = GPU_ASSETS.get((hardware.os, hardware.arch))
    if gpu_entry is not None and flavor == gpu_entry[0]:
        _, suffix, digest = gpu_entry
    else:
        flavor = "cpu"
        asset = ASSETS.get((hardware.os, hardware.arch))
        if asset is None:
            raise ProvisionError(f"no portable llama.cpp build for {hardware.os}/{hardware.arch}")
        suffix, digest = asset
    existing = managed_binary(hardware)
    if existing is not None and managed_flavor(hardware) == flavor:
        return existing
    root = _root(hardware)
    if root.exists():
        # A different flavor than this machine should run (usually the CPU
        # build on a machine with a GPU). Replace it wholesale; the archives
        # are self-contained and nothing else writes here.
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/llama-{BUILD}-bin-{suffix}"
    console.info(f"installing verified llama.cpp {BUILD} in {root}")
    with tempfile.TemporaryDirectory(prefix="install-", dir=root.parent) as tmp:
        archive = Path(tmp) / suffix
        for attempt in range(2):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "llmsetup/0.1.0"})
                with urllib.request.urlopen(request, timeout=30) as response, archive.open("wb") as out:
                    sha = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > _MAX_ARCHIVE:
                            raise ProvisionError("runtime archive exceeds size limit")
                        sha.update(chunk)
                        out.write(chunk)
                if sha.hexdigest() != digest:
                    raise ProvisionError("runtime archive checksum mismatch")
                break
            except (OSError, urllib.error.URLError, ProvisionError) as exc:
                if attempt:
                    raise ProvisionError(f"automatic llama.cpp installation failed: {exc}") from exc
                console.info("runtime download failed validation or connectivity; retrying once")
        dest = Path(tmp) / "unpacked"
        dest.mkdir()
        _unpack(archive, dest)
        name = "llama-server.exe" if hardware.os == "windows" else "llama-server"
        binary = next(dest.rglob(name), None)
        if binary is None:
            raise ProvisionError("runtime archive contains no llama-server")
        binary.chmod(binary.stat().st_mode | 0o700)
        probe = run([str(binary), "--version"], timeout=15)
        if not probe.ok:
            raise ProvisionError(f"portable llama.cpp cannot run on this system: {probe.output[-1000:]}")
        if root.exists():
            raise ProvisionError(f"incomplete runtime installation at {root}")
        (dest / ".flavor").write_text(flavor, encoding="utf-8")
        dest.rename(root)
        return root / binary.relative_to(dest)
