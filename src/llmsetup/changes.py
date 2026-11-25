"""What llmsetup changed outside its own data directory, and how to put it back.

A setup or an export reaches into other apps: links in LM Studio's and Jan's model folders, a tag
in Ollama, a client's own config file when `llmsetup connect` writes it. Each of those is recorded
here as it happens, in an append-only journal beside llmsetup's other data, together with what
undoing it takes. `llmsetup undo` reverses the most recent run, `llmsetup undo --all` everything,
and `llmsetup changes` lists what is recorded.

Undo only removes what it can show is still what llmsetup left there. A file is removed when its
size, modification time and inode (and, for small files, its content hash) still match what was
recorded; anything edited since is reported and left alone unless `--force` is given. An edited
config file is restored from the copy taken before llmsetup touched it, which stays in the backups
folder until then. Downloaded weights and llmsetup's own files are not "changes" in this sense:
they live in llmsetup's directory, and removing them is a separate decision.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from llmsetup.util.fs import app_dir, ensure_dir

#: Files up to this size are fingerprinted by content as well as by size, time and inode.
HASH_LIMIT = 8 * 1024 * 1024

_current: Optional[Dict[str, str]] = None


def journal_path() -> Path:
    return app_dir() / "changes.jsonl"


def backups_dir() -> Path:
    return ensure_dir(app_dir() / "backups")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


@contextmanager
def run(command: str) -> Iterator[str]:
    """Group the changes one command makes, so `llmsetup undo` can reverse them together."""
    global _current
    previous = _current
    _current = {"id": uuid.uuid4().hex[:12], "command": command, "at": _now()}
    try:
        yield _current["id"]
    finally:
        _current = previous


def _active_run() -> Dict[str, str]:
    global _current
    if _current is None:
        _current = {"id": uuid.uuid4().hex[:12], "command": "llmsetup", "at": _now()}
    return _current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> Dict[str, Any]:
    """What identifies the file llmsetup left at ``path``: link target, or size, time and inode."""
    if path.is_symlink():
        return {"symlink": os.readlink(path)}
    if path.is_dir():
        return {"dir": True}
    stat = path.stat()
    result: Dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ino": stat.st_ino}
    if stat.st_size <= HASH_LIMIT:
        result["sha256"] = _sha256(path)
    return result


def _unchanged(path: Path, recorded: Optional[Dict[str, Any]]) -> bool:
    if not recorded:
        return False
    try:
        current = fingerprint(path)
    except OSError:
        return False
    if "sha256" in recorded and "sha256" in current:
        # Content is what matters for a small file; an editor that rewrites it in place with the
        # same bytes has not changed anything worth keeping.
        return bool(current["sha256"] == recorded["sha256"])
    return current == recorded


def record(kind: str, target: str, **details: Any) -> Dict[str, Any]:
    run_info = _active_run()
    entry = {"change": uuid.uuid4().hex[:12], "run": run_info["id"], "command": run_info["command"],
             "at": _now(), "kind": kind, "target": target, **details}
    ensure_dir(journal_path().parent)
    with open(journal_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def created(target: str, path: Path) -> Dict[str, Any]:
    """A file or link llmsetup made where nothing was before."""
    return record("created", target, path=str(path), fingerprint=fingerprint(path))


def created_dir(target: str, path: Path) -> Dict[str, Any]:
    """A folder llmsetup made; undo removes it only once it is empty again."""
    return record("created_dir", target, path=str(path))


def backup_before_edit(path: Path) -> Optional[Path]:
    """Copy a file that is about to be edited; None when there is nothing there yet."""
    if not path.exists():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = backups_dir() / f"{stamp}-{uuid.uuid4().hex[:6]}-{path.name}"
    shutil.copy2(path, destination)
    return destination


def modified(target: str, path: Path, backup: Optional[Path]) -> Dict[str, Any]:
    """A file llmsetup rewrote; ``backup`` is its prior content, or None if it did not exist."""
    return record("modified", target, path=str(path), backup=str(backup) if backup else None,
                  fingerprint=fingerprint(path))


def moved_aside(original: Path, backup: Path) -> Dict[str, Any]:
    """A file that was in llmsetup's way, renamed rather than deleted."""
    return record("moved_aside", "filesystem", path=str(original), backup=str(backup))


def ollama_model(name: str) -> Dict[str, Any]:
    return record("ollama_model", "ollama", name=name)


def manual(target: str, note: str, **details: Any) -> Dict[str, Any]:
    """A change llmsetup cannot reverse itself; undo prints how to."""
    return record("manual", target, note=note, **details)


# --------------------------------------------------------------------------- #
# reading the journal
# --------------------------------------------------------------------------- #


def entries() -> List[Dict[str, Any]]:
    try:
        lines = journal_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and "kind" in item:
            result.append(item)
    return result


def pending_changes(items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Recorded changes not yet undone, oldest first."""
    items = entries() if items is None else items
    undone = {item.get("of") for item in items if item["kind"] == "undone"}
    return [item for item in items if item["kind"] != "undone" and item.get("change") not in undone]


def runs(items: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Runs with their changes, newest first: ``{run, command, at, changes, pending}``."""
    items = entries() if items is None else items
    still = {item["change"] for item in pending_changes(items)}
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if item["kind"] == "undone":
            continue
        group = grouped.setdefault(item["run"], {"run": item["run"], "command": item.get("command", ""),
                                                 "at": item["at"], "changes": [], "pending": 0})
        group["changes"].append({**item, "status": "pending" if item["change"] in still else "undone"})
        group["pending"] += item["change"] in still
    # Insertion order is journal order, which is the order runs happened in; timestamps have
    # one-second resolution and tie for runs that close together.
    return list(reversed(grouped.values()))


# --------------------------------------------------------------------------- #
# undo
# --------------------------------------------------------------------------- #


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def undo_change(change: Dict[str, Any], force: bool = False, dry_run: bool = False,
                ollama_rm: Optional[Callable[[str], bool]] = None) -> Dict[str, Any]:
    """Reverse one recorded change. Returns ``{change, status, note}``; status is one of
    ``undone``, ``already`` (nothing left to do), ``kept`` (changed since, left alone),
    ``manual`` or ``failed``."""
    kind = change["kind"]
    path = Path(change["path"]) if change.get("path") else None
    result = {"change": change["change"], "target": change["target"], "kind": kind}

    def done(status: str, note: str) -> Dict[str, Any]:
        return {**result, "status": status, "note": note}

    if kind == "created" and path is not None:
        if not (path.exists() or path.is_symlink()):
            return done("already", f"{path} is already gone")
        if not force and not _unchanged(path, change.get("fingerprint")):
            return done("kept", f"{path} changed since llmsetup created it, so it was left in place")
        if not dry_run:
            _remove(path)
        return done("undone", f"removed {path}")

    if kind == "created_dir" and path is not None:
        if not path.is_dir():
            return done("already", f"{path} is already gone")
        if any(path.iterdir()):
            return done("kept", f"{path} is not empty, so it was left in place")
        if not dry_run:
            path.rmdir()
        return done("undone", f"removed the empty folder {path}")

    if kind == "modified" and path is not None:
        backup = Path(change["backup"]) if change.get("backup") else None
        if not force and path.exists() and not _unchanged(path, change.get("fingerprint")):
            where = f"; the original is kept at {backup}" if backup else ""
            return done("kept", f"{path} was edited after llmsetup wrote it, so it was left alone{where}")
        if backup is not None:
            if not backup.exists():
                return done("failed", f"the backup of {path} ({backup}) is missing")
            if not dry_run:
                shutil.copy2(backup, path)
            return done("undone", f"restored {path} as it was before llmsetup edited it")
        if not path.exists():
            return done("already", f"{path} is already gone")
        if not dry_run:
            _remove(path)
        return done("undone", f"removed {path}, which llmsetup had created")

    if kind == "moved_aside" and path is not None:
        backup = Path(change["backup"])
        if not backup.exists():
            return done("already", f"{backup} is no longer there to restore")
        if path.exists() or path.is_symlink():
            return done("kept", f"something is at {path} again, so {backup} was left where it is")
        if not dry_run:
            backup.rename(path)
        return done("undone", f"put {path} back")

    if kind == "ollama_model":
        name = str(change.get("name"))
        if dry_run:
            return done("undone", f"would remove the Ollama model {name}")
        if ollama_rm is None:
            ollama_rm = _ollama_rm
        if ollama_rm(name):
            return done("undone", f"removed the Ollama model {name}")
        return done("failed", f"`ollama rm {name}` did not succeed; remove it by hand if it is still listed")

    if kind == "manual":
        return done("manual", str(change.get("note", "")))

    return done("failed", f"llmsetup does not know how to undo a {kind!r} change")


def _ollama_rm(name: str) -> bool:
    from llmsetup.util.proc import run as run_command
    from llmsetup.util.proc import which

    if not which("ollama"):
        return False
    result = run_command(["ollama", "rm", name], timeout=120)
    # A model already removed by hand, or by undoing a later run, has nothing left to undo.
    return result.ok or "not found" in result.output.lower()


def undo(changes: List[Dict[str, Any]], force: bool = False, dry_run: bool = False,
         ollama_rm: Optional[Callable[[str], bool]] = None) -> List[Dict[str, Any]]:
    """Reverse ``changes`` newest first, and journal each one that no longer needs undoing."""
    results = []
    for change in reversed(changes):
        outcome = undo_change(change, force=force, dry_run=dry_run, ollama_rm=ollama_rm)
        results.append(outcome)
        if not dry_run and outcome["status"] in ("undone", "already", "manual"):
            record("undone", change["target"], of=change["change"])
    return results
