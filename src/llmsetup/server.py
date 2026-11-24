"""Background serving: `llmsetup start --detach`, `status`, `logs` and `stop`.

`start` runs the verified command in the foreground, which ties the model to a
terminal. Detached, the same command runs in its own session with output in a
log file, and a small state file records which process it is, so `status` can
say whether it is alive and answering and `stop` can end it.

A PID on its own is not proof of identity: once the server exits, the operating
system can hand the same number to an unrelated process. `stop` therefore only
signals a process whose command line still names the executable it started,
and otherwise treats the record as stale and removes it.
"""

from __future__ import annotations

import codecs
import datetime
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO

from llmsetup.util import http
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import app_dir, ensure_dir, read_json, write_json
from llmsetup.util.log import Console
from llmsetup.util.proc import format_command, run

STATE_FILE = "server.json"
LOG_FILE = "server.log"


def state_path() -> Path:
    return app_dir() / STATE_FILE


def log_path() -> Path:
    return app_dir() / LOG_FILE


def read_state() -> Optional[Dict[str, Any]]:
    state = read_json(state_path())
    if not isinstance(state, dict) or not isinstance(state.get("pid"), int):
        return None
    command = state.get("command")
    if not (isinstance(command, list) and command and all(isinstance(p, str) for p in command)):
        return None
    return state


def clear_state() -> None:
    try:
        state_path().unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------- #
# process identity
# --------------------------------------------------------------------------- #


def process_command(pid: int) -> Optional[str]:
    """The command line of a live process, or None when there is no such process."""
    if pid <= 0:
        return None
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip() or None
    if os.name == "nt":
        result = run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], timeout=10)
        line = result.stdout.strip().splitlines()[0] if result.ok and result.stdout.strip() else ""
        if not line.startswith('"'):
            return None
        return line.split('","')[0].strip('"') or None
    result = run(["ps", "-o", "command=", "-p", str(pid)], timeout=10)
    return (result.stdout.strip() or None) if result.ok else None


def _executable_name(command: List[str]) -> str:
    name = Path(command[0]).name.lower()
    return name[:-4] if name.endswith(".exe") else name


def owns(state: Dict[str, Any], running: Optional[str]) -> bool:
    """Whether the live process at the recorded PID is still the server we started."""
    return running is not None and _executable_name(state["command"]) in running.lower()


def server_running(state: Dict[str, Any]) -> bool:
    return owns(state, process_command(state["pid"]))


# --------------------------------------------------------------------------- #
# start --detach
# --------------------------------------------------------------------------- #


def _healthy(base_url: str) -> bool:
    for path in ("/health", "/v1/models"):
        if http.request(base_url.rstrip("/") + path, timeout=2).ok:
            return True
    return False


def _log_tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _popen_detached(command: List[str], log: Path) -> subprocess.Popen:
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = (getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                                   | getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        # Its own session: closing the terminal or pressing Ctrl-C there does not reach it.
        kwargs["start_new_session"] = True
    with open(log, "ab", buffering=0) as handle:
        return subprocess.Popen(command, stdout=handle, stderr=handle, stdin=subprocess.DEVNULL, **kwargs)


def ensure_not_running() -> None:
    """Refuse to start a second background server while the recorded one is alive.

    Checked before the port preflight, whose generic "port in use" would otherwise be the
    message for the most common case: the server llmsetup itself started is still running.
    """
    existing = read_state()
    if existing is not None and server_running(existing):
        raise LlmsetupError(
            f"a server started by llmsetup is already running (pid {existing['pid']}) at {existing.get('base_url')}",
            hint="run `llmsetup status` to check it, or `llmsetup stop` before starting another")


def start_detached(record: Dict[str, Any], console: Console, wait_s: float = 180.0,
                   popen: Callable[[List[str], Path], Any] = _popen_detached) -> int:
    """Start the recorded command in the background and wait until it answers."""
    ensure_not_running()
    clear_state()

    command: List[str] = record["command"]
    base_url = str(record.get("base_url", ""))
    model = str(record.get("served_model") or record.get("model_id") or "")
    log = ensure_dir(app_dir()) / LOG_FILE
    try:
        process = popen(command, log)
    except OSError as exc:
        raise LlmsetupError(f"could not start {command[0]}: {exc}",
                            hint="run `llmsetup doctor` to re-test the setup") from None

    write_json(state_path(), {
        "pid": process.pid, "command": command, "base_url": base_url, "model": model,
        "runtime": record.get("runtime"), "log": str(log),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    })
    console.step(f"Starting {model} in the background (pid {process.pid})")
    console.info(console.dim(f"log: {log}"))

    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            clear_state()
            tail = _log_tail(log)
            raise LlmsetupError(
                f"{Path(command[0]).name} exited with code {process.returncode} before it was ready"
                + (f":\n{tail}" if tail else ""),
                hint=f"read {log}, or run `llmsetup doctor` to re-test and repair the setup")
        if _healthy(base_url):
            console.ok(f"serving {model} at {base_url}/v1")
            console.info("stop it with `llmsetup stop`")
            return 0
        time.sleep(1.0)
    console.warn(f"{model} is still loading after {wait_s:g}s; it keeps running in the background")
    console.info(f"check it with `llmsetup status`, or read {log}")
    return 0


# --------------------------------------------------------------------------- #
# status and stop
# --------------------------------------------------------------------------- #


def uptime_seconds(started_at: Any, now: Optional[datetime.datetime] = None) -> Optional[int]:
    """Seconds since an ISO ``started_at``, or None when it is missing or unreadable."""
    if not isinstance(started_at, str):
        return None
    try:
        started = datetime.datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.timezone.utc)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return max(0, int((now - started).total_seconds()))


def format_duration(seconds: int) -> str:
    """A short human duration: 45s, 12m, 2h 13m, 4d 2h."""
    minutes, hours, days = seconds // 60, seconds // 3600, seconds // 86400
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{minutes}m"
    if seconds < 86400:
        return f"{hours}h {minutes % 60}m"
    return f"{days}d {hours % 24}h"


def status() -> Dict[str, Any]:
    """Whether the background server is alive and answering."""
    state = read_state()
    if state is None:
        return {"running": False, "managed": False}
    alive = server_running(state)
    report = {
        "running": alive, "managed": True, "pid": state["pid"], "model": state.get("model"),
        "base_url": state.get("base_url"), "runtime": state.get("runtime"),
        "started_at": state.get("started_at"), "log": state.get("log"),
        "command": format_command(state["command"]),
    }
    report["healthy"] = bool(alive and state.get("base_url") and _healthy(str(state["base_url"])))
    if alive:
        report["uptime_s"] = uptime_seconds(state.get("started_at"))
    return report


# sys.platform rather than os.name: mypy narrows on it, so the POSIX-only calls below type-check
# on a Windows machine too (os.killpg, os.getpgid and signal.SIGKILL do not exist there).
def _terminate(pid: int) -> None:
    if sys.platform == "win32":
        run(["taskkill", "/PID", str(pid), "/T"], timeout=15)
    else:
        os.killpg(pid, signal.SIGTERM) if _is_group_leader(pid) else os.kill(pid, signal.SIGTERM)


def _kill(pid: int) -> None:
    if sys.platform == "win32":
        run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=15)
    else:
        os.killpg(pid, signal.SIGKILL) if _is_group_leader(pid) else os.kill(pid, signal.SIGKILL)


def _is_group_leader(pid: int) -> bool:
    if sys.platform == "win32":
        return False
    try:
        return os.getpgid(pid) == pid
    except OSError:
        return False


def stop(console: Console, grace_s: float = 15.0) -> int:
    """Stop the background server, if the recorded PID is still that server."""
    state = read_state()
    if state is None:
        console.info("no server started by `llmsetup start --detach` is recorded")
        return 0
    pid = state["pid"]
    if not server_running(state):
        clear_state()
        console.info(f"the recorded server (pid {pid}) is no longer running; cleared the record")
        return 0
    console.step(f"Stopping {state.get('model') or 'the server'} (pid {pid})")
    try:
        _terminate(pid)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise LlmsetupError(f"could not stop pid {pid}: {exc}",
                            hint="stop it from your process manager, then run `llmsetup stop` again") from None
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and server_running(state):
        time.sleep(0.25)
    if server_running(state):
        console.warn(f"pid {pid} ignored the stop request for {grace_s:g}s; ending it")
        try:
            _kill(pid)
        except ProcessLookupError:
            pass
    clear_state()
    console.ok("stopped")
    return 0


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #

#: ANSI escape sequences, then any other control character but tab and newline. The log is
#: written by a model server that may echo request text, so it is shown inert: nothing in it
#: can move the cursor, retitle the terminal or clear the screen.
_ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _inert(text: str) -> str:
    # A carriage return is dropped rather than shown: CRLF line ends and progress bars that
    # redraw one line are the common sources, and neither should print as "?".
    return _CONTROL.sub("?", _ANSI.sub("", text.replace("\r", "")))


def tail(path: Path, lines: int, block: int = 64 * 1024) -> List[str]:
    """The last ``lines`` lines of a file, read backwards so a large log stays cheap."""
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        data = b""
        while end > 0 and data.count(b"\n") <= lines:
            start = max(0, end - block)
            handle.seek(start)
            data = handle.read(end - start) + data
            end = start
    text = data.decode("utf-8", errors="replace")
    return [_inert(line) for line in text.splitlines()[-lines:]] if lines else []


def current_log() -> Path:
    """The recorded server's log, else the default log location; it must exist."""
    state = read_state()
    path = Path(state["log"]) if state and isinstance(state.get("log"), str) else log_path()
    if not path.is_file():
        raise LlmsetupError("there is no server log yet",
                            hint="start a background server with `llmsetup start --detach`")
    return path


def logs(console: Console, lines: int = 50, follow: bool = False, poll_s: float = 0.5,
         out: Optional[TextIO] = None) -> int:
    """Print the end of the background server's log, and with ``follow`` keep printing."""
    stream = out or sys.stdout
    state = read_state()
    path = current_log()
    for line in tail(path, lines):
        stream.write(line + "\n")
    stream.flush()
    if not follow:
        return 0
    if state is None or not server_running(state):
        console.info("the background server is not running, so there is nothing more to follow")
        return 0
    # Incremental, so a multi-byte character split across two reads still decodes.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        try:
            while True:
                chunk = handle.read()
                if chunk:
                    stream.write(_inert(decoder.decode(chunk)))
                    stream.flush()
                elif not server_running(state):
                    console.info("the background server stopped")
                    return 0
                else:
                    time.sleep(poll_s)
        except KeyboardInterrupt:
            return 0
