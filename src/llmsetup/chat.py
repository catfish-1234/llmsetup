"""Talk to the verified model from the terminal: ``llmsetup chat``.

Setup ends with a model that has passed every check, and the first thing anyone wants to do with
it is ask it something. Until now that meant finding a client, pasting the base URL into it and
guessing the served model name. This finds the endpoint the same way ``start`` and ``status`` do:
the background server if one is running, otherwise the last verified setup.

Replies stream token by token, so a slow model still shows progress. The model's text is printed
inert, as ``llmsetup logs`` prints the server log: a model repeats whatever it was given, and
nothing it writes can move the cursor, retitle the terminal or clear the screen.
"""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TextIO, Tuple

from llmsetup.runtimes.base import Endpoint
from llmsetup.server import _inert as inert
from llmsetup.util import http
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console, write_console

#: One streamed piece of a reply: ``("answer", text)`` or ``("thinking", text)``.
Piece = Tuple[str, str]
StreamLines = Callable[..., Iterator[str]]


# --------------------------------------------------------------------------- #
# finding the endpoint
# --------------------------------------------------------------------------- #


def resolve_endpoint(base_url: Optional[str], model: Optional[str], setup: Path,
                     api_key: Optional[str] = None) -> Endpoint:
    """The endpoint to talk to: explicit flags, else the background server, else the last setup."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    if base_url:
        if not model:
            raise LlmsetupError("--base-url needs --model too",
                                hint="pass the model name the server expects, e.g. --model llama3.1:8b")
        return Endpoint(base_url=base_url, model=model, runtime="external", headers=headers)

    from llmsetup.server import status

    report = status()
    if report.get("running") and report.get("base_url") and (model or report.get("model")):
        return Endpoint(base_url=str(report["base_url"]), model=str(model or report["model"]),
                        runtime=str(report.get("runtime") or "llmsetup"), headers=headers)

    if not setup.exists():
        raise LlmsetupError(
            "there is no model to talk to yet",
            hint="run `llmsetup setup` first, or point at a server with --base-url and --model")
    from llmsetup.launch import load_launch
    from llmsetup.util.fs import read_json

    record = load_launch(read_json(setup), setup)
    served = model or record.get("served_model") or record.get("model_id")
    if not record.get("base_url") or not served:
        raise LlmsetupError(f"the launch record in {setup} has no endpoint",
                            hint="run `llmsetup doctor` to re-test and record it again")
    return Endpoint(base_url=str(record["base_url"]), model=str(served),
                    runtime=str(record.get("runtime") or "llmsetup"), headers=headers)


def ensure_answering(endpoint: Endpoint) -> None:
    from llmsetup.server import _healthy

    if not _healthy(endpoint.base_url):
        raise LlmsetupError(
            f"nothing is answering at {endpoint.base_url}",
            hint="start the verified setup with `llmsetup start --detach`, "
                 "or point at a running server with --base-url and --model")


# --------------------------------------------------------------------------- #
# one reply
# --------------------------------------------------------------------------- #


def _server_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except OSError:
        body = ""
    try:
        detail: Any = json.loads(body).get("error", body)
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
    except (ValueError, AttributeError):
        detail = body
    text = inert(str(detail).strip())[:300]
    return f"the server answered HTTP {exc.code}" + (f": {text}" if text else "")


def stream_reply(endpoint: Endpoint, messages: List[Dict[str, str]], max_tokens: int,
                 temperature: float, timeout: float = 300.0,
                 lines: StreamLines = http.stream_lines) -> Iterator[Piece]:
    """Stream a chat completion as pieces of answer and, when the server splits it out, thinking."""
    body = {"model": endpoint.model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": temperature}
    try:
        for line in lines(endpoint.chat_url, body, headers=endpoint.headers, timeout=timeout):
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if chunk.get("error"):
                error = chunk["error"]
                message = error.get("message", error) if isinstance(error, dict) else error
                raise LlmsetupError(f"the server reported an error: {inert(str(message))[:300]}")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    continue
                # llama.cpp and vLLM name the split-out thinking reasoning_content; LM Studio
                # and newer Ollama versions call it reasoning.
                thinking = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(thinking, str) and thinking:
                    yield "thinking", thinking
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield "answer", content
    except urllib.error.HTTPError as exc:
        raise LlmsetupError(_server_error(exc),
                            hint=f"check that {endpoint.model} is the name the server serves") from None
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LlmsetupError(f"lost the connection to {endpoint.base_url}: {reason}",
                            hint="`llmsetup status` shows whether the server is still running") from None


def answer(pieces: Iterator[Piece], out: TextIO, console: Console, show_thinking: bool) -> str:
    """Print a streamed reply as it arrives and return the answer text."""
    reply: List[str] = []
    thinking_shown = False
    for kind, text in pieces:
        if kind == "thinking":
            if show_thinking:
                write_console(out, console.dim(inert(text)))
                thinking_shown = True
            continue
        if thinking_shown and not reply:
            write_console(out, "\n\n")
        reply.append(text)
        write_console(out, inert(text))
    write_console(out, "\n")
    return "".join(reply)


# --------------------------------------------------------------------------- #
# the command
# --------------------------------------------------------------------------- #

HELP = "/reset starts over, /exit or Ctrl-D leaves, Ctrl-C stops a reply"


def run(endpoint: Endpoint, console: Console, *, message: Optional[str] = None,
        system: Optional[str] = None, max_tokens: int = 1024, temperature: float = 0.7,
        show_thinking: bool = False, as_json: bool = False,
        read: Callable[[str], str] = input, out: Optional[TextIO] = None,
        lines: StreamLines = http.stream_lines) -> int:
    out = out or sys.stdout
    base: List[Dict[str, str]] = [{"role": "system", "content": system}] if system else []

    def reply_to(messages: List[Dict[str, str]], sink: TextIO) -> str:
        pieces = stream_reply(endpoint, messages, max_tokens, temperature, lines=lines)
        return answer(pieces, sink, console, show_thinking)

    if message is not None:
        conversation = [*base, {"role": "user", "content": message}]
        if as_json:
            text = "".join(t for kind, t in stream_reply(endpoint, conversation, max_tokens, temperature,
                                                         lines=lines) if kind == "answer")
            print(json.dumps({"model": endpoint.model, "base_url": endpoint.base_url, "reply": text}))
            return 0
        reply_to(conversation, out)
        return 0

    console.info(f"talking to {endpoint.model} at {endpoint.base_url}/v1 ({HELP})")
    messages = list(base)
    while True:
        try:
            text = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            write_console(out, "\n")
            return 0
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return 0
        if text == "/reset":
            messages = list(base)
            console.info("conversation cleared")
            continue
        if text.startswith("/") and " " not in text:
            console.warn(f"unknown command {text}; {HELP}")
            continue
        messages.append({"role": "user", "content": text})
        try:
            messages.append({"role": "assistant", "content": reply_to(messages, out)})
        except KeyboardInterrupt:
            # The unanswered question is dropped, so the next one is not asked twice.
            write_console(out, "\n")
            console.info("stopped")
            messages.pop()
        except LlmsetupError as exc:
            console.error(str(exc))
            messages.pop()
