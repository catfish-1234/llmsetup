"""Companion embedding model provisioning.

A RAG setup is half-built without a retriever. Before this, `llmfit plan` told
users "a separate embedding model is needed" and `setup` never provided one —
these tests hold that gap shut.
"""

from __future__ import annotations

from typing import List

import pytest
from conftest import make_hardware

from llmfit.pipeline import Pipeline
from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.types import (
    CheckResult,
    CheckStatus,
    RuntimeConfig,
    Severity,
    Task,
    UseCaseProfile,
)
from llmfit.util.log import Console


class FakeRuntime(Runtime):
    """A runtime that records what it was asked to do, without doing it."""

    name = "fake"
    label = "Fake"
    uses_gguf = True

    def __init__(self, fail_on=None):
        self.started: List[str] = []
        self.stopped: List[str] = []
        self.configs: List[RuntimeConfig] = []
        self.fail_on = fail_on

    def is_available(self) -> bool:
        return True

    def ensure_model(self, spec, config, console):
        config.model_path = f"/fake/{spec.id}.gguf"
        return config

    def start(self, spec, config, console) -> Endpoint:
        if self.fail_on and spec.id == self.fail_on:
            raise RuntimeUnavailable(f"cannot start {spec.id}")
        self.started.append(spec.id)
        self.configs.append(config)
        return Endpoint(base_url=f"http://127.0.0.1:{config.port}", model=spec.id,
                        runtime=self.name, started_by_us=True,
                        supports_embeddings=spec.is_embedding)

    def stop(self, endpoint) -> None:
        self.stopped.append(endpoint.model)

    def launch_command(self, spec, config):
        return ["fake", spec.id]

    def local_model_path(self, spec, config):
        return None


def passing(names):
    return [CheckResult(n, CheckStatus.PASS, Severity.CRITICAL, "ok",
                        metrics={"dim": 1024} if n == "embeddings" else {})
            for n in names]


@pytest.fixture
def rag_profile():
    return UseCaseProfile(tasks=[Task.RAG], needs_embeddings=True,
                          context_tokens=32768)


def build(catalog, profile, runtime, monkeypatch, checks=None):
    """A pipeline wired to a fake runtime and scripted check results."""
    monkeypatch.setattr("llmfit.runtimes.registry.get_runtime", lambda name: runtime)
    monkeypatch.setattr("llmfit.pipeline.get_runtime", lambda name: runtime)
    monkeypatch.setattr(
        "llmfit.verify.harness.Harness.run",
        lambda self, names, stop_on_critical=True: (checks or passing)(list(names)),
    )
    return Pipeline(profile, make_hardware(), catalog, Console(quiet=True),
                    runtime_name="fake", register=False)


class TestProvisioning:
    def test_rag_gets_an_embedding_model_set_up(self, catalog, rag_profile,
                                                monkeypatch):
        runtime = FakeRuntime()
        result = build(catalog, rag_profile, runtime, monkeypatch).run()

        assert result.embedding is not None
        assert result.embedding["ok"] is True
        assert result.embedding["dimensions"] == 1024
        # Two distinct models were started: the chat model and the retriever.
        assert len(runtime.started) == 2
        assert result.embedding["model"] in runtime.started

    def test_plain_chat_gets_no_embedding_model(self, catalog, monkeypatch):
        runtime = FakeRuntime()
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        result = build(catalog, profile, runtime, monkeypatch).run()

        assert result.embedding is None
        assert len(runtime.started) == 1

    def test_the_embedding_server_runs_on_its_own_port(self, catalog, rag_profile,
                                                       monkeypatch):
        """It must not collide with the chat model's server."""
        runtime = FakeRuntime()
        build(catalog, rag_profile, runtime, monkeypatch).run()
        ports = [c.port for c in runtime.configs]
        assert len(set(ports)) == len(ports)

    def test_embedding_flags_are_set_for_llama_cpp(self, catalog, rag_profile,
                                                   monkeypatch):
        """llama.cpp refuses to serve embeddings without --embeddings."""
        runtime = FakeRuntime()
        build(catalog, rag_profile, runtime, monkeypatch).run()
        embedding_config = runtime.configs[-1]
        flags = embedding_config.extra.get("llamacpp_flags", [])
        assert "--embeddings" in flags
        assert "--pooling" in flags

    def test_a_server_we_started_is_stopped_after_verification(self, catalog,
                                                               rag_profile, monkeypatch):
        runtime = FakeRuntime()
        result = build(catalog, rag_profile, runtime, monkeypatch).run()
        assert result.embedding["model"] in runtime.stopped
        assert "note" in result.embedding


class TestFailureIsNotFatal:
    def test_a_failed_embedding_setup_does_not_fail_the_run(self, catalog,
                                                            rag_profile, monkeypatch):
        """A working chat model with a broken retriever is still useful, so this
        reports rather than aborts."""
        from llmfit.planner.plan import plan_embedding_model

        embedder = plan_embedding_model(rag_profile, make_hardware(), catalog)
        runtime = FakeRuntime(fail_on=embedder.id)
        result = build(catalog, rag_profile, runtime, monkeypatch).run()

        assert result.embedding["ok"] is False
        assert result.embedding["reason"]
        # The chat model still came up and still passed.
        assert result.succeeded

    def test_a_failing_embedding_check_is_reported_honestly(self, catalog,
                                                            rag_profile, monkeypatch):
        """Vectors that do not separate meaning ruin retrieval silently, so a
        failure here must not be dressed up as success."""
        def checks(names):
            if "embeddings" in names:
                return [CheckResult("embeddings", CheckStatus.FAIL, Severity.REQUIRED,
                                    "vectors do not separate meaning")]
            return passing(names)

        runtime = FakeRuntime()
        result = build(catalog, rag_profile, runtime, monkeypatch, checks=checks).run()
        assert result.embedding["ok"] is False


class TestReporting:
    def test_the_report_states_the_embedding_outcome(self, catalog, rag_profile,
                                                     monkeypatch):
        from llmfit.report.render import render_markdown

        runtime = FakeRuntime()
        result = build(catalog, rag_profile, runtime, monkeypatch).run()
        markdown = render_markdown(result)
        assert "## Retrieval (embeddings)" in markdown
        assert "1024-dimensional" in markdown

    def test_the_report_warns_when_retrieval_is_broken(self, catalog, rag_profile,
                                                       monkeypatch):
        from llmfit.planner.plan import plan_embedding_model
        from llmfit.report.render import render_markdown

        embedder = plan_embedding_model(rag_profile, make_hardware(), catalog)
        runtime = FakeRuntime(fail_on=embedder.id)
        result = build(catalog, rag_profile, runtime, monkeypatch).run()
        markdown = render_markdown(result)
        assert "Not working" in markdown
        assert "poor retriever" in markdown

    def test_the_outcome_survives_json_serialisation(self, catalog, rag_profile,
                                                     monkeypatch):
        import json

        runtime = FakeRuntime()
        result = build(catalog, rag_profile, runtime, monkeypatch).run()
        payload = json.loads(json.dumps(result.to_dict(), default=str))
        assert payload["embedding"]["ok"] is True
