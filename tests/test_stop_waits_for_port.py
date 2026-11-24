"""`llmsetup stop` reports "stopped" only once the server's port no longer answers.

`restart` starts the new server straight after `stop`. The recorded process could be gone while
its port still accepted connections for a moment, and the start's own port check then refused
with "port is already in use" (seen on CI as a failed restart).
"""

from __future__ import annotations

import io

from llmsetup import server
from llmsetup.util.log import Console

STATE = {"pid": 4242, "base_url": "http://127.0.0.1:59999", "model": "m"}


def _stub(monkeypatch, answers):
    running = iter([True, False])
    monkeypatch.setattr(server, "read_state", lambda: dict(STATE))
    monkeypatch.setattr(server, "server_running", lambda state: next(running, False))
    monkeypatch.setattr(server, "_terminate", lambda pid: None)
    monkeypatch.setattr(server, "clear_state", lambda: None)
    monkeypatch.setattr(server.time, "sleep", lambda seconds: None)
    checks = []

    def port_is_free(host, port):
        checks.append((host, port))
        return next(answers)
    monkeypatch.setattr(server.http, "port_is_free", port_is_free)
    return checks


def test_stop_waits_until_the_port_is_released(monkeypatch):
    checks = _stub(monkeypatch, iter([False, False, True]))
    out = io.StringIO()
    assert server.stop(Console(stream=out)) == 0
    assert checks == [("127.0.0.1", 59999)] * 3
    assert "stopped" in out.getvalue() and "still accepting" not in out.getvalue()


def test_stop_says_when_something_else_keeps_the_port(monkeypatch):
    clock = iter(range(0, 1000, 5))
    monkeypatch.setattr(server.time, "monotonic", lambda: float(next(clock)))
    _stub(monkeypatch, iter([False] * 1000))
    out = io.StringIO()
    assert server.stop(Console(stream=out)) == 0
    assert "still accepting connections" in out.getvalue()


def test_a_record_without_a_port_does_not_wait():
    assert server._port_released("") is True
    assert server._port_released("not a url") is True
