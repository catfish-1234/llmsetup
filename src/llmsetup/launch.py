"""`llmsetup start`: run the last verified setup again.

Setup stops the servers it started once verification is done, and until now
getting the model back meant copying the command out of the report. A verified
run now records the exact command it tested, and `start` runs that command,
nothing re-planned and nothing re-tuned, so what starts is what passed.

Only a run whose checks passed gets a launch record. An unverified or failed
setup has nothing `start` should vouch for; `llmsetup doctor` is the way back.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console
from llmsetup.util.proc import format_command, which

#: Runtimes whose server is a shared daemon. The model is served on demand by a
#: process llmsetup neither started nor stops, so there is nothing to launch.
SHARED_DAEMONS = {"ollama", "lmstudio"}

RECORD_VERSION = 1


def launch_record(result: Any) -> Optional[Dict[str, Any]]:
    """The launch record for a finished run, or None when it is not launchable."""
    if not getattr(result, "succeeded", False) or result.plan.best is None:
        return None
    from llmsetup.runtimes.registry import get_runtime

    spec, config = result.plan.best.model, result.config
    try:
        command = [str(part) for part in get_runtime(result.plan.runtime).launch_command(spec, config)]
    except Exception:
        return None
    if not command:
        return None
    return {
        "version": RECORD_VERSION,
        "runtime": result.plan.runtime,
        "model_id": spec.id,
        "served_model": config.served_name or spec.id,
        "base_url": config.base_url,
        "command": command,
        "shared_daemon": result.plan.runtime in SHARED_DAEMONS,
    }


def load_launch(saved: Optional[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    """Validate the launch record inside a saved setup."""
    if not isinstance(saved, dict):
        raise LlmsetupError(f"{path} is not readable",
                            hint="run `llmsetup setup` to record a new setup")
    record = saved.get("launch")
    if record is None:
        if saved.get("succeeded") is False:
            raise LlmsetupError(
                "the last setup did not pass verification, so there is nothing verified to start",
                hint="run `llmsetup doctor` to re-test and repair it")
        raise LlmsetupError(
            f"{path} has no launch record; it was saved by an older llmsetup or by a dry run",
            hint="run `llmsetup doctor` once to re-test the setup and record how to start it")
    command = record.get("command") if isinstance(record, dict) else None
    if not (isinstance(command, list) and command and all(isinstance(p, str) and p for p in command)):
        raise LlmsetupError(f"the launch record in {path} is malformed",
                            hint="run `llmsetup doctor` to re-test and record it again")
    return record


def _port(base_url: str) -> Optional[int]:
    try:
        parts = urlsplit(base_url)
        return parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    except ValueError:
        return None


def preflight(record: Dict[str, Any]) -> None:
    """Refuse to start when the runtime is gone or the port is taken."""
    command: List[str] = record["command"]
    base_url = str(record.get("base_url", ""))
    model = str(record.get("served_model") or record.get("model_id") or "")
    executable = command[0]
    if not (Path(executable).is_file() or which(executable)):
        raise LlmsetupError(
            f"{executable} is no longer installed",
            hint="run `llmsetup doctor` to reinstall the runtime and re-test the setup")

    from llmsetup.util.http import port_is_free

    port = _port(base_url)
    host = urlsplit(base_url).hostname or "127.0.0.1"
    if port is not None and not port_is_free(host, port):
        raise LlmsetupError(
            f"port {port} is already in use, so the model cannot start there",
            hint=(f"if it is already running, check it with `llmsetup status` or "
                  f"`llmsetup verify --base-url {base_url} --model {model}`; "
                  "otherwise stop whatever holds the port and rerun `llmsetup start`"))


def start(record: Dict[str, Any], console: Console, print_only: bool = False,
          runner=subprocess.call) -> int:
    """Print or run the recorded command. Returns the process exit code."""
    command: List[str] = record["command"]
    base_url = str(record.get("base_url", ""))
    model = str(record.get("served_model") or record.get("model_id") or "")

    if print_only:
        print(format_command(command))
        return 0

    if record.get("shared_daemon"):
        console.ok(f"{record.get('runtime')} serves {model} on demand; there is no server to start")
        console.info(f"OpenAI-compatible endpoint: {base_url}/v1  model {model}")
        console.info(f"chat in the terminal with: {format_command(command)}")
        return 0

    executable = command[0]
    preflight(record)

    console.step(f"Starting {model}")
    console.info(f"OpenAI-compatible endpoint: {base_url}/v1  model {model}")
    console.info(console.dim(f"command: {format_command(command)}"))
    console.info(console.dim("press Ctrl-C to stop"))
    try:
        return runner(command)
    except KeyboardInterrupt:
        console.info("stopped")
        return 130
    except OSError as exc:
        raise LlmsetupError(f"could not start {executable}: {exc}",
                            hint="run `llmsetup doctor` to re-test the setup") from None
