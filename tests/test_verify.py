"""Verification: signature classification, check selection, and check logic
against a fake endpoint."""

from __future__ import annotations

import json

import pytest
from conftest import make_hardware

from llmsetup.types import CheckStatus, Severity, Task, UseCaseProfile
from llmsetup.verify import signatures as sig
from llmsetup.verify.checks import (
    CheckContext,
    _extract_json,
    _repetition_ratio,
    _solid_png,
    _strip_thinking,
    check_completion,
    check_embeddings,
    check_health,
    check_json,
    check_stability,
    check_streaming,
    check_tools,
)
from llmsetup.verify.client import ChatResult
from llmsetup.verify.harness import select_checks


class TestSignatures:
    @pytest.mark.parametrize("text,expected", [
        ("ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory",
         sig.OUT_OF_MEMORY),
        ("MPS backend out of memory (MPS allocated: 9.06 GB)", sig.OUT_OF_MEMORY),
        ("ROCm error: hipMalloc failed", sig.OUT_OF_MEMORY),
        ("the request exceeds the available context size", sig.CONTEXT_EXCEEDED),
        ("input is too long for this model", sig.CONTEXT_EXCEEDED),
        ("<urlopen error [Errno 111] Connection refused>", sig.SERVER_UNREACHABLE),
        ('model "qwen3:8b" not found, try pulling it first', sig.MODEL_NOT_LOADED),
        ("failed to load model from /models/x.gguf", sig.MODEL_NOT_LOADED),
        ("This model does not support tools", sig.TOOLS_UNSUPPORTED),
        ("embeddings are not supported by this model", sig.EMBEDDINGS_UNSUPPORTED),
        ("read operation timed out", sig.REQUEST_TIMEOUT),
    ])
    def test_real_error_strings_classify(self, text, expected):
        assert sig.classify(text) == expected

    def test_unrecognised_text_returns_none_rather_than_guessing(self):
        """A wrong classification is worse than none: it sends repair down the
        wrong branch."""
        assert sig.classify("a completely novel failure") is None
        assert sig.classify("") is None

    def test_specific_patterns_beat_generic_ones(self):
        """An OOM message often also contains the word 'error'."""
        assert sig.classify("CUDA error: out of memory") == sig.OUT_OF_MEMORY


class TestSelection:
    def test_baseline_checks_always_run(self, catalog):
        for task in (Task.CHAT, Task.CODING, Task.RAG):
            chosen = select_checks(UseCaseProfile(tasks=[task]),
                                   catalog.require("qwen3-8b"))
            assert {"health", "completion", "stability"} <= set(chosen)

    def test_stated_needs_become_checks(self, catalog):
        chosen = select_checks(
            UseCaseProfile(tasks=[Task.AGENT], needs_tools=True, needs_json=True,
                           context_tokens=32768),
            catalog.require("qwen3-8b"))
        assert "tool_calling" in chosen and "json_output" in chosen
        assert "context" in chosen

    def test_vision_is_not_tested_when_not_wanted(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.CHAT]),
                               catalog.require("qwen3-8b"))
        assert "vision" not in chosen

    def test_an_embedding_model_only_gets_embedding_checks(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.RAG]),
                               catalog.require("bge-m3"))
        assert chosen == ["health", "embeddings"]

    def test_quick_mode_drops_the_optional_checks(self, catalog):
        full = select_checks(UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768),
                             catalog.require("qwen3-8b"))
        quick = select_checks(UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768),
                              catalog.require("qwen3-8b"), quick=True)
        assert len(quick) < len(full)
        assert "health" in quick

    def test_no_check_is_listed_twice(self, catalog):
        chosen = select_checks(
            UseCaseProfile(tasks=[Task.AGENT, Task.RAG, Task.VISION],
                           needs_tools=True, needs_vision=True,
                           needs_embeddings=True, context_tokens=65536),
            catalog.require("gemma-3-27b"))
        assert len(chosen) == len(set(chosen))


class FakeClient:
    """A scripted endpoint. Each method returns whatever the test queued."""

    def __init__(self, chat=None, models=None, embed=None):
        self._chat = chat
        self._models = models if models is not None else {"ok": True, "ids": ["m"]}
        self._embed = embed
        self.endpoint = type("E", (), {"model": "m"})()

    def chat(self, *args, **kwargs):
        return self._chat(*args, **kwargs) if callable(self._chat) else self._chat

    def list_models(self):
        return self._models

    def embed(self, texts, **kwargs):
        return self._embed


def context(client, catalog, profile=None, model="qwen3-8b"):
    from llmsetup.types import RuntimeConfig

    spec = catalog.require(model)
    return CheckContext(
        client=client, spec=spec,
        config=RuntimeConfig(runtime="llamacpp", model_id=spec.id, quant="Q4_K_M",
                             context_tokens=8192, total_layers=spec.layers,
                             gpu_layers=spec.layers),
        profile=profile or UseCaseProfile(tasks=[Task.CHAT]),
        hardware=make_hardware(),
    )


class TestHealthCheck:
    def test_passes_when_the_model_is_listed(self, catalog):
        ctx = context(FakeClient(models={"ok": True, "ids": ["m"]}), catalog)
        assert check_health(ctx).status is CheckStatus.PASS

    def test_fails_when_unreachable(self, catalog):
        ctx = context(FakeClient(models={"ok": False, "error": "Connection refused"}),
                      catalog)
        result = check_health(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.SERVER_UNREACHABLE

    def test_fails_when_a_different_model_is_loaded(self, catalog):
        """A previous model still resident is a real and confusing failure."""
        ctx = context(FakeClient(models={"ok": True, "ids": ["some-other-model"]}),
                      catalog)
        result = check_health(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.MODEL_NOT_LOADED


class TestCompletionCheck:
    def test_passes_on_correct_answers(self, catalog):
        answers = iter(["Paris", "42"])
        ctx = context(FakeClient(chat=lambda *a, **k: ChatResult(ok=True, text=next(answers))),
                      catalog)
        assert check_completion(ctx).status is CheckStatus.PASS

    def test_fails_on_an_empty_response(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=True, text="   ")), catalog)
        result = check_completion(ctx)
        assert result.failure_signature == sig.EMPTY_RESPONSE

    def test_fails_when_every_answer_is_wrong(self, catalog):
        """Both sanity answers wrong means a broken quant or chat template."""
        ctx = context(FakeClient(chat=ChatResult(ok=True, text="banana banana")), catalog)
        result = check_completion(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.INCOHERENT_OUTPUT

    def test_grades_the_answer_not_the_reasoning_scratchpad(self, catalog):
        """A reasoning model that muses about London before answering Paris
        must still pass; one that muses about Paris and answers London must not."""
        answers = iter([
            "<think>Could be London? No.</think>Paris",
            "<think>seventeen plus twenty five</think>42",
        ])
        ctx = context(FakeClient(chat=lambda *a, **k: ChatResult(ok=True, text=next(answers))),
                      catalog)
        assert check_completion(ctx).status is CheckStatus.PASS

    def test_an_oom_during_completion_is_classified(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=False, error="CUDA out of memory")),
                      catalog)
        assert check_completion(ctx).failure_signature == sig.OUT_OF_MEMORY


class TestToolCheck:
    def _call(self, name="get_current_weather", args='{"city": "Tokyo"}'):
        return [{"function": {"name": name, "arguments": args}}]

    def test_passes_on_a_well_formed_call(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=True, tool_calls=self._call())),
                      catalog)
        assert check_tools(ctx).status is CheckStatus.PASS

    def test_prose_instead_of_a_call_is_diagnosed_as_a_template_problem(self, catalog):
        """The single most common local tool-calling failure."""
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, text="I would call get_current_weather with city Tokyo")), catalog)
        result = check_tools(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.TOOL_CALL_MALFORMED
        assert "template" in result.message

    def test_no_call_at_all_is_a_capability_failure(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=True, text="It is sunny.")), catalog)
        assert check_tools(ctx).failure_signature == sig.TOOLS_UNSUPPORTED

    def test_malformed_arguments_are_caught(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, tool_calls=self._call(args="not json"))), catalog)
        assert check_tools(ctx).failure_signature == sig.TOOL_CALL_MALFORMED

    def test_missing_required_argument_is_caught(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, tool_calls=self._call(args="{}"))), catalog)
        assert check_tools(ctx).failure_signature == sig.TOOL_CALL_MALFORMED

    def test_severity_follows_whether_the_user_asked_for_tools(self, catalog):
        failing = ChatResult(ok=True, text="sunny")
        required = check_tools(context(
            FakeClient(chat=failing), catalog,
            UseCaseProfile(tasks=[Task.AGENT], needs_tools=True)))
        advisory = check_tools(context(
            FakeClient(chat=failing), catalog, UseCaseProfile(tasks=[Task.CHAT])))
        assert required.severity is Severity.REQUIRED and required.blocking
        assert advisory.severity is Severity.ADVISORY and not advisory.blocking


class TestJsonCheck:
    def test_passes_on_valid_schema_conforming_json(self, catalog):
        payload = json.dumps({"name": "Amara Okafor", "age": 34,
                              "languages": ["Igbo", "English", "French"]})
        ctx = context(FakeClient(chat=ChatResult(ok=True, text=payload)), catalog)
        assert check_json(ctx).status is CheckStatus.PASS

    def test_tolerates_a_code_fence(self, catalog):
        payload = ('```json\n{"name": "A", "age": 3, "languages": ["x"]}\n```')
        ctx = context(FakeClient(chat=ChatResult(ok=True, text=payload)), catalog)
        assert check_json(ctx).status is CheckStatus.PASS

    def test_rejects_a_wrong_type_for_a_field(self, catalog):
        payload = json.dumps({"name": "A", "age": "thirty", "languages": []})
        ctx = context(FakeClient(chat=ChatResult(ok=True, text=payload)), catalog)
        assert check_json(ctx).failure_signature == sig.JSON_INVALID

    def test_rejects_missing_required_keys(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=True, text='{"name": "A"}')), catalog)
        assert check_json(ctx).failure_signature == sig.JSON_INVALID


class TestEmbeddingCheck:
    def test_passes_when_related_text_is_closer_than_unrelated(self, catalog):
        vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
        ctx = context(FakeClient(embed={"ok": True, "vectors": vectors, "dim": 3}),
                      catalog, model="bge-m3")
        assert check_embeddings(ctx).status is CheckStatus.PASS

    def test_fails_when_embeddings_do_not_separate_meaning(self, catalog):
        """Near-identical vectors pass a naive 'did I get a vector' check and
        then silently ruin retrieval, so this is worth failing on."""
        vectors = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        ctx = context(FakeClient(embed={"ok": True, "vectors": vectors, "dim": 2}),
                      catalog, model="bge-m3")
        result = check_embeddings(ctx)
        assert result.status is CheckStatus.FAIL
        assert "retrieval" in result.message

    def test_fails_when_the_endpoint_refuses(self, catalog):
        ctx = context(FakeClient(embed={"ok": False, "error": "embeddings are not supported"}),
                      catalog, model="bge-m3")
        assert check_embeddings(ctx).failure_signature == sig.EMBEDDINGS_UNSUPPORTED


class TestHelpers:
    def test_strips_think_blocks(self):
        assert _strip_thinking("<think>hmm</think>Paris") == "Paris"

    def test_handles_an_unterminated_think_block(self):
        """A reply cut off mid-thought should not evaluate to nothing."""
        assert _strip_thinking("Answer: 42<think>wait") == "Answer: 42"

    def test_leaves_plain_text_alone(self):
        assert _strip_thinking("just an answer") == "just an answer"

    def test_repetition_ratio_detects_a_loop(self):
        looping = "the sea is blue and vast. " * 12
        varied = ("The ocean covers most of the planet. Tides follow the moon. "
                  "Deep currents carry heat around the globe, shaping climate "
                  "far inland from any coast at all times.")
        assert _repetition_ratio(looping) > 0.5
        assert _repetition_ratio(varied) < 0.3

    def test_repetition_ratio_ignores_short_text(self):
        assert _repetition_ratio("too short") == 0.0

    def test_extract_json_finds_an_object_in_prose(self):
        parsed, _ = _extract_json('Sure! {"a": 1} hope that helps')
        assert parsed == {"a": 1}

    def test_extract_json_returns_none_on_garbage(self):
        parsed, _ = _extract_json("no json here at all")
        assert parsed is None

    def test_the_generated_test_image_is_a_valid_png(self):
        data = _solid_png(8, 8, (220, 30, 30))
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert b"IHDR" in data and b"IDAT" in data and data.endswith(b"IEND\xae\x42\x60\x82")

class TestReasoningTruncation:
    """A reasoning model spends a small probe budget on the thinking block and
    returns no visible answer. That is a max_tokens choice, not an empty
    response, and it must not send recovery after healthy engines."""

    def test_stability_tolerates_thinking_filling_the_probe_budget(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, text="", reasoning_text="let me think about the number",
            completion_tokens=96, finish_reason="length")), catalog)
        result = check_stability(ctx)
        assert result.status is CheckStatus.PASS
        assert result.metrics["thinking_only_responses"] == 4
        assert "thinking budget" in result.message

    def test_stability_still_fails_on_a_genuinely_empty_response(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, text="", completion_tokens=0, finish_reason="stop")), catalog)
        result = check_stability(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.EMPTY_RESPONSE

    def test_stability_passes_when_thinking_eats_the_budget_even_mislabeled(self, catalog):
        """A card mislabeled as non-reasoning still thinks (MiniCPM5-2B did,
        and the false empty-response verdict switched engines for nothing).
        The response shape, not the card, decides."""
        def chat(messages, **kwargs):
            if kwargs.get("max_tokens", 0) <= 96:
                # Small probe budget: thinking eats it all, like MiniCPM5-2B.
                return ChatResult(ok=True, text="", completion_tokens=96,
                                  finish_reason="length")
            return ChatResult(ok=True, text="A fine number.",
                              completion_tokens=104, finish_reason="stop")

        ctx = context(FakeClient(chat=chat), catalog, model="qwen2.5-coder-7b")
        result = check_stability(ctx)
        assert result.status is CheckStatus.PASS
        assert result.metrics["thinking_only_responses"] == 4

    def test_stability_fails_empty_when_reasoning_is_disabled(self, catalog):
        from types import SimpleNamespace
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, text="", completion_tokens=96, finish_reason="length")), catalog)
        ctx.config.advanced = SimpleNamespace(
            reasoning=SimpleNamespace(enabled=False, token_budget=None))
        result = check_stability(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.EMPTY_RESPONSE

    def test_completion_retries_with_thinking_headroom(self, catalog):
        budgets = []

        def chat(messages, **kwargs):
            budgets.append(kwargs["max_tokens"])
            if len(budgets) % 2 == 1:
                return ChatResult(ok=True, text="", reasoning_text="working",
                                  completion_tokens=kwargs["max_tokens"],
                                  finish_reason="length")
            answer = "Paris" if "France" in messages[0]["content"] else "42"
            return ChatResult(ok=True, text=answer)

        ctx = context(FakeClient(chat=chat), catalog)
        result = check_completion(ctx)
        assert result.status is CheckStatus.PASS
        assert budgets[0] == 512
        assert budgets[1] > 512

    def test_completion_fails_when_the_retry_also_returns_nothing(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(
            ok=True, text="", reasoning_text="working",
            completion_tokens=512, finish_reason="length")), catalog)
        result = check_completion(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.EMPTY_RESPONSE

class TestShapelessTruncation:
    """Some servers report a thinking-truncated reply with no reasoning field,
    no usage and a "stop" finish reason. The stability probe must ask once
    more with room for the thinking budget before calling it empty."""

    def test_stability_retries_shapeless_empty_with_thinking_headroom(self, catalog):
        budgets = []

        def chat(messages, **kwargs):
            budgets.append(kwargs["max_tokens"])
            if kwargs["max_tokens"] <= 96:
                return ChatResult(ok=True, text="", finish_reason="stop")
            return ChatResult(ok=True, text="It is a number.", finish_reason="stop")

        ctx = context(FakeClient(chat=chat), catalog)
        result = check_stability(ctx)
        assert result.status is CheckStatus.PASS
        assert result.metrics["thinking_only_responses"] == 4
        assert budgets[0] == 96 and budgets[1] > 96

    def test_stability_fails_when_the_retry_is_empty_too(self, catalog):
        ctx = context(FakeClient(chat=ChatResult(ok=True, text="", finish_reason="stop")),
                      catalog)
        result = check_stability(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.EMPTY_RESPONSE
        assert "finish_reason" in result.detail

    def test_streaming_counts_thinking_chunks(self, catalog, monkeypatch):
        from llmsetup.runtimes.base import Endpoint
        from llmsetup.verify.client import OpenAIClient

        lines = iter([
            'data: {"choices": [{"delta": {"reasoning_content": "thinking"}}]}',
            'data: {"choices": [{"delta": {"reasoning_content": " more"}}]}',
            'data: {"choices": [{"delta": {"content": "1 2 3"}}]}',
            'data: [DONE]',
        ])
        monkeypatch.setattr("llmsetup.util.http.stream_lines", lambda *a, **kw: lines)
        ctx = context(FakeClient(), catalog)
        ctx.client = OpenAIClient(Endpoint("http://fixture.invalid", "m", "llamacpp"))
        result = check_streaming(ctx)
        assert result.status is CheckStatus.PASS
        assert result.metrics["chunks"] == 3
