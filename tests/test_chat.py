"""`llmsetup chat`: finding the endpoint, streaming a reply, and the conversation loop."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from llmsetup import chat
from llmsetup.cli import main
from llmsetup.runtimes.base import Endpoint
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console


class FakeServer:
    """An OpenAI-compatible server that streams the replies it is given, in order."""

    def __init__(self) -> None:
        self.replies: List[Any] = []
        self.requests: List[Dict[str, Any]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"data": []}')

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"body": body, "auth": self.headers.get("Authorization")})
                reply = server.replies.pop(0)
                if isinstance(reply, tuple):  # (status, json body): an HTTP error
                    self.send_response(reply[0])
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(reply[1]).encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in reply:
                    self.wfile.write(f"{line}\n\n".encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def sse(*pieces: str, thinking: str = "") -> List[str]:
    lines = [": keep-alive"]
    if thinking:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"reasoning_content": thinking}}]}))
    lines += ["data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) for p in pieces]
    return [*lines, "data: not json", "data: [DONE]"]


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


@pytest.fixture
def endpoint(server):
    return Endpoint(base_url=server.url, model="test-model", runtime="external")


def quiet() -> Console:
    return Console(stream=io.StringIO())


def test_streams_a_reply_and_skips_keepalives_and_junk(server, endpoint):
    server.replies.append(sse("Hel", "lo", "!"))
    out = io.StringIO()
    assert chat.run(endpoint, quiet(), message="hi", out=out) == 0
    assert out.getvalue() == "Hello!\n"
    sent = server.requests[0]["body"]
    assert sent["model"] == "test-model" and sent["stream"] is True
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_model_output_is_printed_inert(server, endpoint):
    server.replies.append(sse("safe \x1b]0;pwned\x07title \x1b[2J", "done\x07"))
    out = io.StringIO()
    chat.run(endpoint, quiet(), message="hi", out=out)
    assert "\x1b" not in out.getvalue() and "\x07" not in out.getvalue()
    assert out.getvalue().startswith("safe title ")


def test_thinking_is_hidden_unless_asked_for(server, endpoint):
    server.replies += [sse("42", thinking="six times seven"), sse("42", thinking="six times seven")]
    hidden, shown = io.StringIO(), io.StringIO()
    chat.run(endpoint, quiet(), message="?", out=hidden)
    chat.run(endpoint, quiet(), message="?", out=shown, show_thinking=True)
    assert hidden.getvalue() == "42\n"
    assert "six times seven" in shown.getvalue() and shown.getvalue().endswith("42\n")


def test_json_mode_prints_one_object(server, endpoint, capsys):
    server.replies.append(sse("4"))
    assert chat.run(endpoint, quiet(), message="2+2", as_json=True) == 0
    assert json.loads(capsys.readouterr().out) == {"model": "test-model", "base_url": server.url, "reply": "4"}


def test_conversation_keeps_history_and_reset_clears_it(server, endpoint):
    server.replies += [sse("first"), sse("second"), sse("third")]
    said = iter(["one", "two", "/reset", "three", "/exit"])
    out = io.StringIO()
    assert chat.run(endpoint, quiet(), system="be brief", read=lambda _: next(said), out=out) == 0
    histories = [r["body"]["messages"] for r in server.requests]
    assert histories[1] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
    ]
    assert histories[2] == [{"role": "system", "content": "be brief"}, {"role": "user", "content": "three"}]


def test_end_of_input_leaves_cleanly(server, endpoint):
    def eof(_: str) -> str:
        raise EOFError
    assert chat.run(endpoint, quiet(), read=eof, out=io.StringIO()) == 0
    assert server.requests == []


def test_a_failed_turn_is_reported_and_dropped(server, endpoint):
    server.replies += [(404, {"error": {"message": "model 'test-model' not found"}}), sse("ok")]
    said = iter(["first try", "second try", "/exit"])
    errors = io.StringIO()
    chat.run(endpoint, Console(stream=errors), read=lambda _: next(said), out=io.StringIO())
    assert "HTTP 404" in errors.getvalue() and "not found" in errors.getvalue()
    # The question that failed is not resent with the next one.
    assert server.requests[1]["body"]["messages"] == [{"role": "user", "content": "second try"}]


def test_an_error_inside_the_stream_is_raised(server, endpoint):
    server.replies.append(["data: " + json.dumps({"error": {"message": "context length exceeded"}})])
    with pytest.raises(LlmsetupError, match="context length exceeded"):
        chat.run(endpoint, quiet(), message="hi", out=io.StringIO())


def test_api_key_is_sent_as_a_bearer_token(server, tmp_path):
    server.replies.append(sse("ok"))
    endpoint = chat.resolve_endpoint(server.url, "m", tmp_path / "none.json", api_key="secret-key")
    chat.run(endpoint, quiet(), message="hi", out=io.StringIO())
    assert server.requests[0]["auth"] == "Bearer secret-key"


# --------------------------------------------------------------------------- #
# finding the endpoint
# --------------------------------------------------------------------------- #


def _not_running(monkeypatch):
    monkeypatch.setattr("llmsetup.server.status", lambda: {"running": False, "managed": False})


def test_base_url_needs_a_model(tmp_path):
    with pytest.raises(LlmsetupError, match="needs --model"):
        chat.resolve_endpoint("http://127.0.0.1:1", None, tmp_path / "x.json")


def test_prefers_the_running_background_server(monkeypatch, tmp_path):
    monkeypatch.setattr("llmsetup.server.status", lambda: {
        "running": True, "managed": True, "base_url": "http://127.0.0.1:8081", "model": "served-name",
        "runtime": "llamacpp"})
    endpoint = chat.resolve_endpoint(None, None, tmp_path / "missing.json")
    assert (endpoint.base_url, endpoint.model) == ("http://127.0.0.1:8081", "served-name")


def test_falls_back_to_the_last_verified_setup(monkeypatch, tmp_path):
    _not_running(monkeypatch)
    setup = tmp_path / "last-setup.json"
    setup.write_text(json.dumps({"launch": {
        "version": 1, "runtime": "ollama", "model_id": "qwen3-8b", "served_model": "qwen3:8b",
        "base_url": "http://127.0.0.1:11434/v1", "command": ["ollama", "serve"]}}))
    endpoint = chat.resolve_endpoint(None, None, setup)
    assert (endpoint.base_url, endpoint.model) == ("http://127.0.0.1:11434", "qwen3:8b")
    assert chat.resolve_endpoint(None, "other", setup).model == "other"


def test_says_what_to_do_when_there_is_no_setup(monkeypatch, tmp_path):
    _not_running(monkeypatch)
    with pytest.raises(LlmsetupError, match="no model to talk to") as caught:
        chat.resolve_endpoint(None, None, tmp_path / "missing.json")
    assert "llmsetup setup" in str(caught.value)


def test_reports_a_server_that_is_not_answering():
    with pytest.raises(LlmsetupError, match="nothing is answering") as caught:
        chat.ensure_answering(Endpoint(base_url="http://127.0.0.1:9", model="m", runtime="external"))
    assert "llmsetup start --detach" in str(caught.value)


def test_cli_one_question_against_a_server(server, capsys):
    server.replies.append(sse("pong"))
    assert main(["chat", "--base-url", server.url, "--model", "m", "-m", "ping", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["reply"] == "pong"


def test_cli_refuses_json_without_a_question(server, capsys):
    assert main(["chat", "--base-url", server.url, "--model", "m", "--json"]) == 1
    assert "--json needs a question" in capsys.readouterr().err
