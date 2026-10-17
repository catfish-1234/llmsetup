"""The `llmsetup ui` server.

A deliberate choice of shape: this is a localhost HTTP server built on the
standard library, serving one static page, rather than an Electron app.

The reasoning is that the value of this project is the *reasoning*, the fit
maths, the verification, the repair ladder. Not the chrome. An Electron app
would add a Node toolchain, a build step, per-platform packaging, code signing,
and a self-update mechanism, all of which need maintaining forever and none of
which make a single recommendation better. A local page costs one file, works on
every platform the CLI already works on, ships inside the same `pip install`,
and needs no network (which matters for a tool whose whole point is running
models locally).

The CLI stays the engine. This is a thin client over the same functions, so the
two can never disagree about what the tool does.
"""

from __future__ import annotations

import json
import shlex
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from llmsetup.util.log import Console

STATIC = Path(__file__).parent / "static"

#: Only these Host headers are served. A browser page on another site can make
#: requests to 127.0.0.1, and DNS rebinding can point a hostile domain at the
#: loopback address. Checking Host is what stops a random web page from driving
#: this server and reading the machine's hardware profile.
def _allowed_hosts(port: int) -> set:
    return {
        f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}",
        "127.0.0.1", "localhost",
    }


class UiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, console: Console):
        super().__init__(address, handler)
        self.console = console
        self.allowed_hosts = _allowed_hosts(address[1])
        #: Cached so the page loads instantly; hardware does not change mid-session.
        self.cache: Dict[str, Any] = {}


class Handler(BaseHTTPRequestHandler):
    server: UiServer
    server_version = "llmsetup"
    sys_version = ""

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args: Any) -> None:
        self.server.console.debug(f"ui: {fmt % args}")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in self.server.allowed_hosts

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This page never embeds anything external and is never framed.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; img-src data:",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0 or length > 1024 * 1024:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (ValueError, OSError):
            return {}

    # ------------------------------------------------------------- routing

    def do_GET(self) -> None:
        if not self._host_ok():
            self._json(403, {"error": "host not allowed"})
            return

        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_page()
        elif path == "/api/hardware":
            self._json(200, self._hardware())
        elif path == "/api/catalog":
            self._json(200, self._catalog())
        elif path == "/api/runtimes":
            self._json(200, self._runtimes())
        elif path == "/api/questions":
            self._json(200, self._questions())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_ok():
            self._json(403, {"error": "host not allowed"})
            return

        path = urlparse(self.path).path
        if path == "/api/plan":
            self._json(200, self._plan(self._read_json()))
        else:
            self._json(404, {"error": "not found"})

    # ------------------------------------------------------------- handlers

    def _serve_page(self) -> None:
        try:
            body = (STATIC / "index.html").read_bytes()
        except OSError:
            self._json(500, {"error": "UI assets are missing from the install"})
            return
        self._send(200, body, "text/html; charset=utf-8")

    def _hardware(self) -> Dict[str, Any]:
        cache = self.server.cache
        if "hardware" not in cache:
            from llmsetup.hardware import detect_hardware
            from llmsetup.hardware.bandwidth import describe_bandwidth
            from llmsetup.hardware.detect import summarize

            hardware = detect_hardware()
            cache["hardware_obj"] = hardware
            cache["hardware"] = {
                **hardware.to_dict(),
                "summary": summarize(hardware),
                "bandwidth": describe_bandwidth(hardware),
                "memory_budget": hardware.memory_budget,
                "memory_budget_gb": round(hardware.memory_budget / (1024 ** 3), 1),
            }
        return cache["hardware"]

    def _catalog(self) -> Dict[str, Any]:
        from llmsetup.catalog.catalog import load_catalog

        catalog = load_catalog()
        return {
            "updated": catalog.updated,
            "count": len(catalog),
            "models": [
                {
                    **model.to_dict(),
                    "arch_source": catalog.arch_source(model.id),
                    "is_moe": model.is_moe,
                }
                for model in catalog.models
            ],
        }

    def _runtimes(self) -> Dict[str, Any]:
        from llmsetup.runtimes.registry import all_runtimes

        rows = []
        for runtime in all_runtimes():
            available = runtime.is_available()
            rows.append({
                "name": runtime.name, "label": runtime.label,
                "available": available,
                "install": "" if available else runtime.install_hint(),
            })
        return {"runtimes": rows}

    def _questions(self) -> Dict[str, Any]:
        """The interview, as data, the same question bank the CLI wizard uses."""
        from llmsetup.interview.questions import QUESTIONS

        return {
            "questions": [
                {
                    "key": q.key, "prompt": q.prompt, "kind": q.kind,
                    "why": q.why, "default": q.default,
                    "options": [
                        {"value": o.value, "label": o.label, "hint": o.hint}
                        for o in q.options
                    ],
                }
                for q in QUESTIONS
            ]
        }

    def _plan(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        from llmsetup.catalog.catalog import load_catalog
        from llmsetup.interview.questions import infer_flags
        from llmsetup.interview.wizard import profile_from_answers
        from llmsetup.planner.plan import build_plan, plan_embedding_model
        from llmsetup.planner.score import explain
        from llmsetup.util.errors import LlmsetupError

        self._hardware()          # populate the cache
        hardware = self.server.cache["hardware_obj"]

        try:
            profile = profile_from_answers(infer_flags(answers))
        except LlmsetupError as exc:
            return {"error": exc.message, "hint": exc.hint}

        try:
            catalog = load_catalog()
            plan = build_plan(profile, hardware, catalog)
        except LlmsetupError as exc:
            return {"error": exc.message, "hint": exc.hint}

        embedder = plan_embedding_model(profile, hardware, catalog)
        return {
            "runtime": plan.runtime,
            "runtime_reason": plan.runtime_reason,
            "notes": plan.notes,
            "profile": profile.to_dict(),
            "embedding_model": (
                {"id": embedder.id, "name": embedder.name,
                 "license": embedder.license} if embedder else None
            ),
            "candidates": [
                {
                    "id": c.model.id,
                    "name": c.model.name,
                    "publisher": c.model.publisher,
                    "quant": c.quant,
                    "license": c.model.license,
                    "params_b": c.model.params_b,
                    "active_params_b": c.model.active_params_b,
                    "is_moe": c.model.is_moe,
                    "score": c.score,
                    "why": explain(c, profile),
                    "warnings": c.warnings,
                    "fits_on_gpu": c.fits_on_gpu,
                    "tokens_per_second": c.score_breakdown.get("est_tokens_per_s"),
                    "memory": {
                        "total_gb": round(c.estimate.total_gb, 2),
                        "weights_gb": round(c.estimate.weights / (1024 ** 3), 2),
                        "kv_gb": round(c.estimate.kv_cache / (1024 ** 3), 2),
                        "overhead_gb": round(c.estimate.overhead / (1024 ** 3), 2),
                        "kv_quant": c.estimate.kv_quant,
                    },
                    "command": _setup_command(c, profile),
                }
                for c in plan.candidates
            ],
        }


def _setup_command(candidate, profile) -> str:
    """The exact CLI command for this recommendation.

    The UI never runs a multi-gigabyte download behind a button press. It hands
    over the command instead, so the irreversible step stays an explicit,
    inspectable choice, and so the terminal remains the source of truth.
    """
    parts = ["llmsetup setup", f"--model {shlex.quote(candidate.model.id)}",
             f"--quant {shlex.quote(candidate.quant)}", f"--context {profile.context_tokens}"]
    for task in profile.tasks:
        parts.append(f"--task {task.value}")
    if profile.needs_tools:
        parts.append("--tools")
    if profile.needs_vision:
        parts.append("--vision")
    if profile.needs_embeddings:
        parts.append("--embeddings")
    if profile.commercial_use:
        parts.append("--commercial")
    if profile.concurrent_users > 1:
        parts.append(f"--users {profile.concurrent_users}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def serve(console: Console, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True,
          ready: Optional[Callable[[str], None]] = None) -> None:
    """Run the UI until interrupted.

    Bound to loopback by default and deliberately not configurable to a public
    interface from the CLI: this server reports the machine's hardware and can
    be driven to plan downloads, which is not something to expose on a network.
    """
    from llmsetup.util import http as http_util

    if not http_util.port_is_free(host, port):
        port = http_util.find_free_port(host, port + 1)

    server = UiServer((host, port), Handler, console)
    url = f"http://{host}:{port}"

    console.print()
    console.print(console.bold(f"  llmsetup is running at {url}"))
    console.info(console.dim("  press Ctrl-C to stop"))
    console.print()

    if ready:
        ready(url)
    if open_browser:
        threading.Timer(0.4, lambda: _open(url, console)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print()
        console.info("stopped")
    finally:
        server.server_close()


def _open(url: str, console: Console) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        console.debug("could not open a browser; open the URL yourself")
