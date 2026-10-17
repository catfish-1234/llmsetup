"""Subprocess helpers.

Everything that shells out goes through :func:`run`, which never raises on a
non-zero exit. Detection code is expected to probe for tools that may not exist,
so "the command failed" is normal control flow here, not an error.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        return (self.stdout + "\n" + self.stderr).strip()


def which(name: str) -> Optional[str]:
    """Absolute path to an executable, or None."""
    return shutil.which(name)


def have(name: str) -> bool:
    return which(name) is not None


def run(
    cmd: Sequence[str],
    timeout: float = 30.0,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    stdin: Optional[str] = None,
) -> CommandResult:
    """Run a command to completion. Missing binaries return rc=127, not an exception."""
    merged = None
    if env:
        merged = {**os.environ, **env}
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=merged,
            input=stdin,
        )
        return CommandResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            -1,
            _decode(exc.stdout),
            _decode(exc.stderr) or f"timed out after {timeout}s",
            timed_out=True,
        )
    except FileNotFoundError:
        return CommandResult(127, "", f"{cmd[0]}: not found")
    except PermissionError as exc:
        return CommandResult(126, "", str(exc))


def spawn(
    cmd: Sequence[str],
    log_path: Optional[str] = None,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """Start a long-lived background process (an inference server).

    Output is redirected to ``log_path`` when given, because the repair layer
    reads server logs to diagnose failures that the HTTP API cannot surface.
    """
    merged = {**os.environ, **(env or {})}
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        # Deliberately not a context manager: the handle must outlive this
        # function so the spawned server can keep writing to it.
        handle = open(log_path, "ab", buffering=0)  # noqa: SIM115
        return subprocess.Popen(list(cmd), stdout=handle, stderr=handle, env=merged, cwd=cwd)
    return subprocess.Popen(
        list(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=merged, cwd=cwd
    )


def first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _decode(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def parse_csv_table(text: str) -> List[List[str]]:
    """Parse `nvidia-smi --format=csv,noheader` style output."""
    rows = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows
