"""`llmsetup logs`: the end of the background server's log, shown inert."""

import io
import threading

import pytest

from llmsetup import cli, server
from llmsetup.server import log_path, logs, state_path, tail
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import ensure_dir, write_json
from llmsetup.util.log import Console


def write_log(text: str, name: str = "server.log"):
    path = ensure_dir(log_path().parent) / name
    path.write_bytes(text.encode("utf-8"))
    return path


def test_tail_returns_the_last_lines_of_a_large_log(tmp_path):
    path = tmp_path / "big.log"
    path.write_text("".join(f"line {i}\n" for i in range(50_000)))
    assert tail(path, 3, block=1024) == ["line 49997", "line 49998", "line 49999"]
    assert tail(path, 0) == []


def test_tail_of_a_short_log_returns_all_of_it(tmp_path):
    path = tmp_path / "short.log"
    path.write_text("only\nthree\nlines")
    assert tail(path, 50) == ["only", "three", "lines"]


def test_logs_prints_the_recorded_log(tmp_path):
    path = write_log("".join(f"loading layer {i}\n" for i in range(10)), "custom.log")
    write_json(state_path(), {"pid": 999_999_999, "command": ["llama-server"], "log": str(path)})
    out = io.StringIO()
    assert logs(Console(quiet=True), lines=2, out=out) == 0
    assert out.getvalue() == "loading layer 8\nloading layer 9\n"


def test_logs_falls_back_to_the_default_log_without_a_record():
    write_log("model loaded\n")
    out = io.StringIO()
    assert logs(Console(quiet=True), out=out) == 0
    assert out.getvalue() == "model loaded\n"


def test_logs_shows_terminal_escapes_inert():
    write_log("\x1b[32mok\x1b[0m\r\n\x1b]0;pwned\x07title\x1b[2Jcleared\x07bell?\n")
    out = io.StringIO()
    logs(Console(quiet=True), out=out)
    assert out.getvalue() == "ok\ntitlecleared?bell?\n"
    assert "\x1b" not in out.getvalue()


def test_logs_without_any_log_explains_how_to_get_one():
    with pytest.raises(LlmsetupError, match="no server log yet") as error:
        logs(Console(quiet=True), out=io.StringIO())
    assert "llmsetup start --detach" in error.value.hint


def test_follow_stops_at_once_when_no_server_is_running():
    write_log("last words\n")
    out = io.StringIO()
    assert logs(Console(quiet=True), follow=True, out=out) == 0
    assert out.getvalue() == "last words\n"


def test_follow_prints_new_output_until_the_server_stops(monkeypatch):
    path = write_log("start\n")
    write_json(state_path(), {"pid": 4242, "command": ["llama-server"], "log": str(path)})
    alive = threading.Event()
    alive.set()
    monkeypatch.setattr(server, "server_running", lambda state: alive.is_set())

    def append_then_stop():
        # "é" split across two writes must still arrive whole.
        with open(path, "ab") as handle:
            handle.write("caf".encode() + "é".encode()[:1])
            handle.flush()
            handle.write("é".encode()[1:] + b" ready\n")
        alive.clear()

    out = io.StringIO()
    timer = threading.Timer(0.2, append_then_stop)
    timer.start()
    try:
        assert logs(Console(quiet=True), follow=True, poll_s=0.05, out=out) == 0
    finally:
        timer.cancel()
    assert out.getvalue() == "start\ncafé ready\n"


def test_cli_logs_lines_flag(capsys):
    write_log("".join(f"{i}\n" for i in range(100)))
    assert cli.main(["logs", "-n", "3"]) == 0
    assert capsys.readouterr().out == "97\n98\n99\n"
    with pytest.raises(SystemExit):
        cli.main(["logs", "-n", "0"])


def test_cli_logs_json(capsys):
    import json

    path = write_log("a\nb\nc\n")
    assert cli.main(["logs", "--json", "-n", "2"]) == 0
    assert json.loads(capsys.readouterr().out) == {"log": str(path), "lines": ["b", "c"]}
