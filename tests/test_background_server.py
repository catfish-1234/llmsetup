"""`llmsetup start --detach`, `status` and `stop` against a stand-in model server."""

import json
import os
import socket
import sys
import time

import pytest

from llmsetup import cli
from llmsetup.server import owns, process_command, read_state, start_detached, status, stop
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")

FAKE_SERVER = """
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = b'{"data": [{"id": "fake"}]}' if self.path in ("/health", "/v1/models") else b"{}"
        self.send_response(200 if self.path in ("/health", "/v1/models") else 404)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
print("fake llama-server ready", flush=True)
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
"""


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def record(tmp_path):
    script = tmp_path / "llama_server_fake.py"
    script.write_text(FAKE_SERVER)
    port = free_port()
    return {"runtime": "llamacpp", "served_model": "qwen3-8b", "base_url": f"http://127.0.0.1:{port}",
            "command": [sys.executable, str(script), str(port)], "shared_daemon": False}


def test_detach_status_stop_round_trip(record):
    console = Console(quiet=True)
    try:
        assert start_detached(record, console, wait_s=20) == 0
        report = status()
        assert report["running"] and report["healthy"]
        assert report["model"] == "qwen3-8b"
        with open(report["log"]) as log:
            assert "fake llama-server ready" in log.read()
        with pytest.raises(LlmsetupError, match="already running"):
            start_detached(record, console, wait_s=5)
    finally:
        assert stop(console) == 0
    assert read_state() is None
    assert status() == {"running": False, "managed": False}


def test_a_server_that_dies_while_loading_reports_its_log(record, tmp_path):
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nprint('failed to load model: out of memory', flush=True)\nsys.exit(3)\n")
    record["command"] = [sys.executable, str(crash)]
    with pytest.raises(LlmsetupError, match="out of memory") as error:
        start_detached(record, Console(quiet=True), wait_s=20)
    assert "llmsetup doctor" in error.value.hint
    assert read_state() is None


def test_stop_never_signals_a_reused_pid(tmp_path):
    """A recorded PID that now belongs to another program is a stale record, not a target."""
    from llmsetup.server import state_path
    from llmsetup.util.fs import write_json

    write_json(state_path(), {"pid": os.getpid(), "command": ["/opt/llama/bin/llama-server", "-m", "x"],
                              "base_url": "http://127.0.0.1:1", "model": "m"})
    assert not owns(read_state(), process_command(os.getpid()))
    assert stop(Console(quiet=True)) == 0
    assert read_state() is None  # cleared, and this test process is still alive


def test_cli_status_and_stop_without_a_server(capsys):
    assert cli.main(["status", "--json"]) == 3
    assert json.loads(capsys.readouterr().out) == {"running": False, "managed": False}
    assert cli.main(["stop"]) == 0


def test_cli_start_detach_uses_the_saved_launch_record(record, monkeypatch):
    from llmsetup.util.fs import app_dir, write_json

    write_json(app_dir() / "last-setup.json", {"succeeded": True, "launch": record})
    try:
        assert cli.main(["start", "--detach", "--wait", "20"]) == 0
        assert status()["healthy"]
    finally:
        cli.main(["stop"])
    time.sleep(0.2)
    assert read_state() is None


def test_cli_start_detach_twice_names_the_running_server(record, capsys):
    """The port is busy because of our own server; say that, not just "port in use"."""
    from llmsetup.util.fs import app_dir, write_json

    write_json(app_dir() / "last-setup.json", {"succeeded": True, "launch": record})
    try:
        assert cli.main(["start", "--detach", "--wait", "20"]) == 0
        capsys.readouterr()
        assert cli.main(["start", "--detach", "--wait", "5"]) != 0
        err = capsys.readouterr().err
        assert "already running (pid" in err
        assert "llmsetup stop" in err
    finally:
        cli.main(["stop"])


def test_cli_restart_replaces_the_running_server(record):
    from llmsetup.util.fs import app_dir, write_json

    write_json(app_dir() / "last-setup.json", {"succeeded": True, "launch": record})
    try:
        assert cli.main(["start", "--detach", "--wait", "20"]) == 0
        first = read_state()["pid"]
        assert status()["uptime_s"] >= 0
        assert cli.main(["restart", "--wait", "20"]) == 0
        report = status()
        assert report["running"] and report["healthy"]
        assert report["pid"] != first
    finally:
        cli.main(["stop"])


def test_cli_restart_starts_a_server_when_none_is_running(record, capsys):
    from llmsetup.util.fs import app_dir, write_json

    write_json(app_dir() / "last-setup.json", {"succeeded": True, "launch": record})
    try:
        assert cli.main(["restart", "--wait", "20"]) == 0
        assert "no background server was running" in capsys.readouterr().err
        assert status()["healthy"]
    finally:
        cli.main(["stop"])
