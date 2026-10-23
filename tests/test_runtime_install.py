"""Portable engine installation is pinned, verified, bounded, and user-local."""

import hashlib
import io
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest
from conftest import make_hardware

from llmsetup.runtimes import install
from llmsetup.util.errors import ProvisionError
from llmsetup.util.log import Console
from llmsetup.util.proc import CommandResult


def tar_bytes(entries=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
        for name, body in (entries or [("build/bin/llama-server", b"fake executable")]):
            member = tarfile.TarInfo(name)
            member.size = len(body)
            member.mode = 0o755
            bundle.addfile(member, io.BytesIO(body))
    return stream.getvalue()


def fake_asset(monkeypatch, data, suffix="test.tar.gz"):
    hardware = make_hardware(vram_gb=0)
    monkeypatch.setitem(install.ASSETS, (hardware.os, hardware.arch),
                        (suffix, hashlib.sha256(data).hexdigest()))
    requests = []

    def urlopen(request, **kw):
        requests.append(request.full_url)
        return io.BytesIO(data)

    monkeypatch.setattr(install.urllib.request, "urlopen", urlopen)
    executions = []

    def run(command, **kw):
        executions.append(command)
        assert Path(command[0]).is_file()
        return CommandResult(0, "llama-server fixture", "")

    monkeypatch.setattr(install, "run", run)
    return hardware, requests, executions


@pytest.mark.parametrize("windows", [False, True])
def test_install_checks_digest_before_executing_and_reuses_binary(monkeypatch, windows):
    if windows:
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("llama-server.exe", b"fixture")
        data = stream.getvalue()
    else:
        data = tar_bytes()
    hardware, requests, executions = fake_asset(monkeypatch, data)
    if windows:
        hardware.os = "windows"
        monkeypatch.setitem(install.ASSETS, (hardware.os, hardware.arch),
                            ("test.zip", hashlib.sha256(data).hexdigest()))
    binary = install.install_llamacpp(hardware, Console(quiet=True))
    assert binary.is_file()
    assert "runtimes" in binary.parts
    assert "llamacpp-b11669" in binary.parts
    assert len(requests) == len(executions) == 1
    assert "/download/b11669/" in requests[0]
    assert install.install_llamacpp(hardware, Console(quiet=True)) == binary
    assert len(requests) == 1


def test_bad_digest_retries_once_and_never_executes(monkeypatch):
    hardware, requests, executions = fake_asset(monkeypatch, tar_bytes())
    monkeypatch.setitem(install.ASSETS, (hardware.os, hardware.arch), ("bad.tar.gz", "0" * 64))
    with pytest.raises(ProvisionError, match="checksum mismatch"):
        install.install_llamacpp(hardware, Console(quiet=True))
    assert len(requests) == 2
    assert not executions
    assert install.managed_binary(hardware) is None


def test_transient_network_error_is_retried(monkeypatch):
    hardware, _requests, executions = fake_asset(monkeypatch, tar_bytes())
    original = install.urllib.request.urlopen
    failures = [True]

    def flaky(request, **kw):
        if failures:
            failures.pop()
            raise urllib.error.URLError("connection reset")
        return original(request, **kw)

    monkeypatch.setattr(install.urllib.request, "urlopen", flaky)
    assert install.install_llamacpp(hardware, Console(quiet=True)).is_file()
    assert len(executions) == 1


def test_unsupported_platform_does_not_download(monkeypatch):
    hardware, requests, executions = fake_asset(monkeypatch, tar_bytes())
    hardware.arch = "riscv64"
    with pytest.raises(ProvisionError, match="no portable"):
        install.install_llamacpp(hardware, Console(quiet=True))
    assert not requests and not executions


@pytest.mark.parametrize("path", ["../outside", "/outside", "a/../../outside", "C:/outside", "a\\outside"])
def test_unsafe_archive_paths_cannot_escape(tmp_path, path):
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(tar_bytes([(path, b"unsafe")]))
    with pytest.raises(ProvisionError, match="unsafe runtime archive path"):
        install._unpack(archive, tmp_path / "dest")
    assert not (tmp_path / "outside").exists()


def test_internal_library_link_is_copied_as_regular_file(tmp_path):
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("build/lib/libggml.so.1")
        member.size = 3
        bundle.addfile(member, io.BytesIO(b"lib"))
        link = tarfile.TarInfo("build/lib/libggml.so")
        link.type = tarfile.SYMTYPE
        link.linkname = "libggml.so.1"
        bundle.addfile(link)
    dest = tmp_path / "dest"
    install._unpack(archive, dest)
    result = dest / "build/lib/libggml.so"
    assert result.read_bytes() == b"lib"
    assert not result.is_symlink()


def test_unusable_verified_binary_is_not_committed(monkeypatch):
    hardware, _, _ = fake_asset(monkeypatch, tar_bytes())
    monkeypatch.setattr(install, "run", lambda *a, **kw: CommandResult(1, "", "unsupported CPU"))
    with pytest.raises(ProvisionError, match="unsupported CPU"):
        install.install_llamacpp(hardware, Console(quiet=True))
    assert install.managed_binary(hardware) is None

def gpu_zip(data=b"fixture"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("llama-server.exe", data)
    return stream.getvalue()


def test_desired_flavor_prefers_vulkan_on_windows_with_gpu():
    hardware = make_hardware(vram_gb=8, os_name="windows")
    assert install.desired_flavor(hardware) == "vulkan"


def test_desired_flavor_stays_cpu_without_gpu_or_gpu_build():
    assert install.desired_flavor(make_hardware(vram_gb=0, os_name="windows")) == "cpu"
    assert install.desired_flavor(make_hardware(vram_gb=8, os_name="linux")) == "cpu"


def test_gpu_install_marks_flavor_and_replaces_a_cpu_install(monkeypatch):
    data = gpu_zip()
    hardware = make_hardware(vram_gb=8, os_name="windows")
    monkeypatch.setitem(install.GPU_ASSETS, ("windows", "x86_64"),
                        ("vulkan", "test-gpu.zip", hashlib.sha256(data).hexdigest()))
    monkeypatch.setitem(install.ASSETS, ("windows", "x86_64"),
                        ("test-cpu.zip", hashlib.sha256(data).hexdigest()))
    requested = []

    def urlopen(request, **kw):
        requested.append(request.full_url)
        return io.BytesIO(data)

    monkeypatch.setattr(install.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(install, "run",
                        lambda command, **kw: CommandResult(0, "llama-server fixture", ""))

    binary = install.install_llamacpp(hardware, Console(quiet=True))
    assert binary.is_file()
    assert "test-gpu.zip" in requested[0]
    assert install.managed_flavor(hardware) == "vulkan"

    # Asking for the CPU flavor replaces the GPU install, and vice versa.
    binary = install.install_llamacpp(hardware, Console(quiet=True), flavor="cpu")
    assert "test-cpu.zip" in requested[-1]
    assert install.managed_flavor(hardware) == "cpu"
    assert binary.is_file()


def test_managed_flavor_defaults_to_cpu_for_legacy_installs(monkeypatch, tmp_path):
    hardware = make_hardware(vram_gb=0)
    root = install._root(hardware)
    (root / "bin").mkdir(parents=True)
    (root / "bin" / ("llama-server.exe" if hardware.os == "windows" else "llama-server")).write_bytes(b"old")
    assert install.managed_flavor(hardware) == "cpu"


def test_managed_flavor_is_none_without_an_install():
    assert install.managed_flavor(make_hardware(vram_gb=0)) is None


def test_failed_gpu_upgrade_preserves_working_cpu_build(monkeypatch):
    data = gpu_zip()
    hardware = make_hardware(vram_gb=8, os_name="windows")
    root = install._root(hardware)
    root.mkdir(parents=True)
    binary = root / "llama-server.exe"
    binary.write_bytes(b"working CPU build")
    (root / ".flavor").write_text("cpu", encoding="utf-8")
    monkeypatch.setitem(install.GPU_ASSETS, ("windows", "x86_64"),
                        ("vulkan", "bad-gpu.zip", "0" * 64))
    monkeypatch.setattr(install.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(data))
    monkeypatch.setattr(install, "run", lambda *a, **kw: pytest.fail("executed an unverified file"))
    with pytest.raises(ProvisionError, match="checksum mismatch"):
        install.install_llamacpp(hardware, Console(quiet=True))
    assert binary.read_bytes() == b"working CPU build"
    assert install.managed_flavor(hardware) == "cpu"


def test_runtime_archive_deadline_stops_a_trickling_download(monkeypatch):
    hardware = make_hardware(vram_gb=0)
    clock = [0.0]
    requests = []

    class Response(io.BytesIO):
        def read(self, size):
            clock[0] += 301
            return b"partial archive"

        read1 = read

    def urlopen(request, timeout):
        assert 0 < timeout <= 30
        requests.append(request)
        return Response()

    monkeypatch.setattr(install.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(install.urllib.request, "urlopen", urlopen)
    with pytest.raises(ProvisionError, match="timed out after 300s") as error:
        install.install_llamacpp(hardware, Console(quiet=True))
    assert len(requests) == 1
    assert "llmsetup setup" in error.value.hint
    assert install.managed_binary(hardware) is None
