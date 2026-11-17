"""Minimal HTTP client over stdlib urllib.

The verifier only ever talks to a local inference endpoint, so this
deliberately avoids a `requests` dependency. Connection errors are returned as
``HttpResponse`` objects rather than raised: "the server isn't up yet" is an
expected state during startup polling, not an exception.
"""

from __future__ import annotations

import http.client
import json as _json
import socket
import time
import urllib.error
import urllib.parse
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
        try:
            body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        except (OSError, http.client.HTTPException) as read_error:
            body = f"could not read error response: {read_error}"
        return HttpResponse(
            exc.code, body, dict(exc.headers or {}), error=str(exc),
            elapsed_s=time.monotonic() - started,
        )
    except (urllib.error.URLError, socket.timeout, OSError, http.client.HTTPException) as exc:
        return HttpResponse(0, "", error=str(exc), elapsed_s=time.monotonic() - started)


def stream_lines(
    url: str,
    json_body: Any,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 120.0,
) -> Iterator[str]:
    """Yield decoded lines from a streaming (SSE / NDJSON) response.

    Used by the streaming check, which measures time-to-first-token, a number
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
    raise_errors: bool = False,
) -> bool:
    """Poll ``predicate`` until true or timeout. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            if raise_errors:
                raise
        if on_tick:
            on_tick(deadline - time.monotonic())
        time.sleep(interval)
    try:
        return predicate()
    except Exception:
        if raise_errors:
            raise
        return False


def _origin(url: str) -> tuple:
    parts = urllib.parse.urlsplit(url)
    default = {"http": 80, "https": 443}.get(parts.scheme)
    try:
        port = parts.port or default
    except ValueError:
        port = None
    return parts.scheme.lower(), (parts.hostname or "").lower(), port


class CredentialSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirects that cannot leak credentials or downgrade the transport.

    The standard handler copies every ordinary header onto the redirected
    request, so a token in ``Authorization`` is replayed to whatever host the
    redirect names. Credentials here travel as unredirected headers, which the
    standard handler drops on every redirect. This handler puts them back only
    when the redirect stays on the same scheme, host and port, which is what a
    renamed Hugging Face repo needs, and refuses an HTTPS to HTTP downgrade.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        old_origin, new_origin = _origin(req.full_url), _origin(new.full_url)
        if old_origin[0] == "https" and new_origin[0] != "https":
            raise urllib.error.HTTPError(
                newurl, code, f"refusing to follow a redirect from HTTPS to {new_origin[0]}",
                headers, fp)
        if old_origin == new_origin:
            for name, value in req.unredirected_hdrs.items():
                if name.lower() in ("authorization", "proxy-authorization"):
                    new.add_unredirected_header(name, value)
        return new


def install_credential_safe_redirects() -> None:
    """Make ``urllib.request.urlopen`` use :class:`CredentialSafeRedirectHandler`.

    Called once by the CLI. Library callers get the safe default regardless:
    credentials sent as unredirected headers never follow a redirect at all.
    """
    urllib.request.install_opener(urllib.request.build_opener(CredentialSafeRedirectHandler))


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def find_free_port(host: str, start: int, limit: int = 50) -> int:
    """First free port at or after ``start``. Avoids colliding with a running server."""
    for port in range(start, start + limit):
        if port_is_free(host, port):
            return port
    from llmsetup.util.errors import ProvisionError

    raise ProvisionError(f"ports {start}-{start + limit - 1} on {host} are all in use",
                         hint="stop an unused local server, then rerun `llmsetup setup`")
