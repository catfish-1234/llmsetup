"""Minimal HTTP client over stdlib urllib.

The verifier only ever talks to a local inference endpoint, so this
deliberately avoids a `requests` dependency. Connection errors are returned as
``HttpResponse`` objects rather than raised: "the server isn't up yet" is an
expected state during startup polling, not an exception.
"""

from __future__ import annotations

import json as _json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional


@dataclass
class HttpResponse:
    status: int
    body: str
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        try:
            return _json.loads(self.body)
        except (ValueError, TypeError):
            return None


def request(
    url: str,
    method: str = "GET",
    json_body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 60.0,
) -> HttpResponse:
    data = None
    hdrs = {"Accept": "application/json", **(headers or {})}
    if json_body is not None:
        data = _json.dumps(json_body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    started = time.monotonic()
    try:
        # Local endpoints only; no proxy handling wanted here.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            return HttpResponse(
                resp.getcode(), body, dict(resp.headers), elapsed_s=time.monotonic() - started
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return HttpResponse(
            exc.code, body, dict(exc.headers or {}), error=str(exc),
            elapsed_s=time.monotonic() - started,
        )
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return HttpResponse(0, "", error=str(exc), elapsed_s=time.monotonic() - started)


def stream_lines(
    url: str,
    json_body: Any,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Yield decoded lines from a streaming (SSE / NDJSON) response.

    Used by the streaming check, which measures time-to-first-token — a number
    you cannot get from a buffered request.
    """
    hdrs = {"Content-Type": "application/json", "Accept": "text/event-stream", **(headers or {})}
    req = urllib.request.Request(
        url, data=_json.dumps(json_body).encode("utf-8"), headers=hdrs, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n\r")
            if line:
                yield line


def wait_for(
    predicate: Callable[[], bool],
    timeout: float = 120.0,
    interval: float = 1.0,
    on_tick: Optional[Callable[[float], None]] = None,
) -> bool:
    """Poll ``predicate`` until true or timeout. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            pass
        if on_tick:
            on_tick(deadline - time.monotonic())
        time.sleep(interval)
    try:
        return predicate()
    except Exception:
        return False


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def find_free_port(host: str, start: int, limit: int = 50) -> int:
    """First free port at or after ``start``. Avoids colliding with a running server."""
    for port in range(start, start + limit):
        if port_is_free(host, port):
            return port
    return start
