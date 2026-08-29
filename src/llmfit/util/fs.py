"""Filesystem helpers, including the link strategy used by the interop layer."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Tuple


def app_dir() -> Path:
    """Where llmfit keeps profiles, logs and its shared model store."""
    env = os.environ.get("LLMFIT_HOME")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "llmfit"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "llmfit"


def model_store() -> Path:
    """Single home for downloaded weights; every app is linked against this."""
    env = os.environ.get("LLMFIT_MODEL_STORE")
    return Path(env).expanduser() if env else app_dir() / "models"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def free_bytes(path: Path) -> int:
    """Free space on the filesystem holding ``path`` (walks up if it is missing)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except OSError:
        return 0


#: Characters allowed in a path component derived from catalog data.
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: str, fallback: str = "unknown") -> str:
    """Make a catalog-supplied string safe to use as one path component.

    Model ids, publishers and quant names come from YAML that a user can extend,
    share, or (in future) fetch. Interpolating them straight into a path lets an
    entry like ``id: ../../../../etc/cron.d/x`` place a multi-gigabyte download
    anywhere the process can write. Everything outside a conservative allowlist
    is collapsed to a dash, and the traversal-significant names are rejected
    outright rather than merely escaped.
    """
    cleaned = _SAFE_COMPONENT.sub("-", str(value or "")).strip("-. ")
    if not cleaned or set(cleaned) <= {"."}:
        return fallback
    return cleaned[:128]


def safe_subpath(root: Path, *components: str) -> Path:
    """Join ``components`` under ``root``, guaranteeing the result stays inside.

    Belt and braces: each component is sanitised, and the resolved path is then
    checked against the root. The second check catches anything the first misses
    (symlinked roots, platform-specific path quirks) rather than trusting the
    sanitiser alone.
    """
    candidate = root.joinpath(*(safe_component(c) for c in components))
    root_resolved = root.resolve() if root.exists() else root.absolute()
    resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValueError(
            f"refusing to build a path outside {root}: {'/'.join(map(str, components))}"
        )
    return candidate


def link_or_copy(src: Path, dst: Path) -> Tuple[str, Path]:
    """Materialise ``src`` at ``dst`` as cheaply as the filesystem allows.

    Order is hardlink → symlink → copy. A hardlink costs nothing and looks like a
    real file to apps that stat it; a symlink costs nothing but breaks across
    filesystems and on some Windows setups; a copy always works but doubles
    multi-gigabyte weights. Returns the method used, so the report can tell the
    user whether they just spent 20 GB of disk.
    """
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if _same_file(src, dst):
            return "existing", dst
        if dst.is_dir() and not dst.is_symlink():
            # A directory where a weights file should be is not ours to remove;
            # deleting it could destroy unrelated user data.
            raise IsADirectoryError(
                f"{dst} is a directory; expected a file or nothing. "
                f"Move it aside and retry."
            )
        dst.unlink()

    try:
        os.link(str(src), str(dst))
        return "hardlink", dst
    except OSError:
        pass
    try:
        os.symlink(str(src), str(dst))
        return "symlink", dst
    except OSError:
        pass
    shutil.copy2(str(src), str(dst))
    return "copy", dst


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or os.path.samefile(str(a), str(b))
    except OSError:
        return False


def write_json(path: Path, data: Any) -> Path:
    return write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def write_text(path: Path, text: str) -> Path:
    """Atomic write: temp file in the same directory, then rename.

    Config files this writes are read by other applications (LM Studio, Ollama),
    so a half-written file is worse than no file.
    """
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".llmfit-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def expand(path: str) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()
