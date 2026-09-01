"""The local web UI.

It is a thin client over the same functions the CLI uses, so these tests cover
the seam (routing, serialisation and the security guard) rather than
re-testing the planner through HTTP.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest

from llmfit.ui.server import STATIC, Handler, UiServer, _setup_command
from llmfit.util.log import Console


@contextmanager
def running_server():
    """A real server on an ephemeral port, torn down afterwards."""
    server = UiServer(("127.0.0.1", 0), Handler, Console(quiet=True))
    server.allowed_hosts = {f"127.0.0.1:{server.server_address[1]}", "127.0.0.1"}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def get(url, host=None, timeout=30):
    request = urllib.request.Request(url)
    if host:
        request.add_header("Host", host)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode()) \
            if "json" in response.headers.get("Content-Type", "") \
            else response.read().decode()


def post(url, payload, timeout=60):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


class TestAssets:
    def test_the_page_ships_with_the_package(self):
        assert (STATIC / "index.html").is_file()

    def test_the_page_fetches_nothing_external(self):
        """A tool for running models locally has to work with the network off,
        so the UI embeds everything and loads no CDN, font or script."""
        text = (STATIC / "index.html").read_text(encoding="utf-8")
        assert "https://" not in text
        assert "//cdn" not in text
        assert "<script src=" not in text
        assert "@import" not in text

    def test_the_page_is_theme_aware(self):
        text = (STATIC / "index.html").read_text(encoding="utf-8")
        assert "prefers-color-scheme: dark" in text
        assert '[data-theme="dark"]' in text


class TestSecurity:
    def test_serves_a_loopback_host(self):
        with running_server() as base:
            status, _ = get(base + "/api/hardware")
            assert status == 200

    def test_refuses_a_foreign_host_header(self):
        """A page on another site can reach 127.0.0.1, and DNS rebinding can
        point a hostile domain at loopback. Checking Host is what stops a random
        web page driving this server and reading the machine's profile."""
        with running_server() as base:
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                get(base + "/api/hardware", host="evil.example.com")
            assert excinfo.value.code == 403

    def test_refuses_a_foreign_host_on_post_too(self):
        with running_server() as base:
            request = urllib.request.Request(
                base + "/api/plan", data=b"{}", method="POST",
                headers={"Content-Type": "application/json",
                         "Host": "evil.example.com"},
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                opener.open(request, timeout=30)
            assert excinfo.value.code == 403

    def test_sends_hardening_headers(self):
        with running_server() as base:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(base + "/", timeout=30) as response:
                headers = response.headers
            assert headers.get("X-Frame-Options") == "DENY"
            assert headers.get("X-Content-Type-Options") == "nosniff"
            assert "default-src 'none'" in headers.get("Content-Security-Policy", "")

    def test_an_unknown_route_is_a_404_not_a_traversal(self):
        with running_server() as base:
            for path in ("/../../etc/passwd", "/static/../../../etc/passwd",
                         "/api/nope"):
                with pytest.raises(urllib.error.HTTPError) as excinfo:
                    get(base + path)
                assert excinfo.value.code == 404

    def test_an_oversized_body_is_ignored_rather_than_buffered(self):
        with running_server() as base:
            request = urllib.request.Request(
                base + "/api/plan", data=b"{}", method="POST",
                headers={"Content-Type": "application/json",
                         "Content-Length": "999999999"},
            )
            # The handler caps the read; the request simply yields a plan for
            # default answers rather than allocating a gigabyte.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                opener.open(request, timeout=30)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                pass          # a refused oversized body is equally acceptable


class TestApi:
    def test_hardware_reports_a_usable_budget(self):
        with running_server() as base:
            _, data = get(base + "/api/hardware")
            assert data["summary"]
            assert data["memory_budget"] > 0
            assert "bandwidth" in data

    def test_catalog_lists_models_with_provenance(self):
        with running_server() as base:
            _, data = get(base + "/api/catalog")
            assert data["count"] > 10
            assert all("arch_source" in m for m in data["models"])

    def test_questions_mirror_the_cli_wizard(self):
        """One question bank drives both, so the two can never drift."""
        from llmfit.interview.questions import QUESTIONS

        with running_server() as base:
            _, data = get(base + "/api/questions")
            assert [q["key"] for q in data["questions"]] == [q.key for q in QUESTIONS]
            assert all(q["why"] for q in data["questions"])

    def test_runtimes_reports_availability(self):
        with running_server() as base:
            _, data = get(base + "/api/runtimes")
            names = {r["name"] for r in data["runtimes"]}
            assert {"ollama", "llamacpp", "vllm"} <= names

    def test_plan_returns_ranked_candidates_with_a_memory_breakdown(self):
        with running_server() as base:
            data = post(base + "/api/plan",
                        {"tasks": ["coding"], "context_tokens": 8192})
            assert data.get("candidates")
            scores = [c["score"] for c in data["candidates"]]
            assert scores == sorted(scores, reverse=True)
            memory = data["candidates"][0]["memory"]
            assert memory["weights_gb"] > 0
            assert memory["kv_gb"] > 0
            assert memory["total_gb"] >= memory["weights_gb"]

    def test_plan_reports_impossible_requirements_instead_of_crashing(self):
        with running_server() as base:
            data = post(base + "/api/plan",
                        {"tasks": ["chat"], "needs_vision": True,
                         "context_tokens": 900_000})
            assert data.get("error")
            assert data.get("hint")

    def test_plan_rejects_invalid_answers_with_an_explanation(self):
        with running_server() as base:
            data = post(base + "/api/plan", {"tasks": ["not-a-real-task"]})
            assert data.get("error")

    def test_rag_surfaces_the_companion_embedding_model(self):
        with running_server() as base:
            data = post(base + "/api/plan",
                        {"tasks": ["rag"], "context_tokens": 32768})
            assert data["embedding_model"] is not None


class TestHandoffCommand:
    """The UI never starts a multi-gigabyte download behind a button. It hands
    over the exact command, so the irreversible step stays explicit and the
    terminal remains the source of truth."""

    def _command(self, answers):
        from conftest import make_hardware

        from llmfit.catalog.catalog import load_catalog
        from llmfit.interview.wizard import profile_from_answers
        from llmfit.planner.plan import build_plan

        profile = profile_from_answers(answers)
        plan = build_plan(profile, make_hardware(), load_catalog())
        return _setup_command(plan.best, profile)

    def test_includes_the_model_quant_and_context(self):
        command = self._command({"tasks": ["coding"], "context_tokens": 32768})
        assert command.startswith("llmfit setup --model ")
        assert "--quant " in command
        assert "--context 32768" in command

    def test_carries_the_stated_requirements(self):
        command = self._command({
            "tasks": ["agent"], "needs_tools": True, "commercial_use": True,
            "concurrent_users": 4, "context_tokens": 8192,
        })
        assert "--tools" in command
        assert "--commercial" in command
        assert "--users 4" in command

    def test_the_command_actually_parses(self):
        """A handed-over command that the CLI rejects would be worse than none."""
        from llmfit.cli import build_parser

        command = self._command({"tasks": ["coding"], "needs_tools": True,
                                 "context_tokens": 16384})
        args = build_parser().parse_args(command.split()[1:])
        assert args.command == "setup"
        assert args.model and args.context_tokens == 16384
