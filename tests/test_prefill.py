"""Prompt processing is a different number from decode, and it is the one that
long-prompt work actually waits on.

Decode is bound by memory bandwidth, prefill by compute. A setup can be
perfectly good at one and unusable at the other, so measuring only decode --
which is what a "tokens per second" figure almost always means -- hides the
failure that RAG and agent users hit on every single request.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware
from test_verify import FakeClient

from llmsetup.types import CheckStatus, Priority, RuntimeConfig, Task, UseCaseProfile
from llmsetup.verify import signatures as sig
from llmsetup.verify.checks import CheckContext, check_prefill
from llmsetup.verify.client import ChatResult
from llmsetup.verify.harness import select_checks


def answered(prompt_tokens=2048, latency_s=1.0):
    return ChatResult(ok=True, text=".", prompt_tokens=prompt_tokens,
                      completion_tokens=1, latency_s=latency_s)


def context(catalog, result, tasks=(Task.CHAT,), context_tokens=8192):
    spec = catalog.require("qwen3-8b")
    return CheckContext(
        client=FakeClient(chat=lambda *a, **k: result), spec=spec,
        config=RuntimeConfig(runtime="llamacpp", model_id=spec.id, quant="Q4_K_M",
                             context_tokens=context_tokens, total_layers=spec.layers,
                             gpu_layers=spec.layers),
        profile=UseCaseProfile(tasks=list(tasks)),
        hardware=make_hardware(),
    )


class TestMeasurement:
    def test_rate_comes_from_the_servers_own_token_count(self, catalog):
        """Dividing the server's prompt-token count by the latency of a
        single-token completion *is* the prefill rate -- nothing is inferred."""
        ctx = context(catalog, answered(prompt_tokens=2048, latency_s=2.0))
        result = check_prefill(ctx)
        assert result.metrics["prefill_tokens_per_second"] == pytest.approx(1024.0)

    def test_reports_the_wait_a_user_would_actually_feel(self, catalog):
        """1024 tok/s against a 8192-token context is an 8-second stare at an
        empty screen; that is the number worth printing, not the rate."""
        ctx = context(catalog, answered(prompt_tokens=2048, latency_s=2.0),
                      context_tokens=8192)
        assert check_prefill(ctx).metrics["full_context_wait_s"] == pytest.approx(8.0)

    def test_falls_back_to_an_estimate_when_usage_is_missing(self, catalog):
        """Some servers return no usage block. An approximate rate beats none."""
        ctx = context(catalog, ChatResult(ok=True, text=".", latency_s=1.0))
        result = check_prefill(ctx)
        assert result.metrics["prefill_tokens_per_second"] > 0

    def test_a_failed_request_is_reported_as_a_failure_not_a_speed(self, catalog):
        ctx = context(catalog, ChatResult(ok=False, error="Connection refused"))
        result = check_prefill(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.SERVER_UNREACHABLE

    def test_an_untimeable_response_does_not_divide_by_zero(self, catalog):
        ctx = context(catalog, answered(latency_s=0.0))
        assert check_prefill(ctx).status is CheckStatus.PASS


class TestJudgement:
    def test_a_healthy_setup_passes(self, catalog):
        ctx = context(catalog, answered(prompt_tokens=2048, latency_s=0.5))
        assert check_prefill(ctx).status is CheckStatus.PASS

    def test_a_broken_setup_fails(self, catalog):
        """5 tokens/s is not a slow machine -- it is one that has fallen off its
        accelerator entirely."""
        ctx = context(catalog, answered(prompt_tokens=100, latency_s=20.0))
        result = check_prefill(ctx)
        assert result.status is CheckStatus.FAIL
        assert result.failure_signature == sig.PREFILL_TOO_SLOW

    def test_warns_only_where_long_prompts_are_the_norm(self, catalog):
        """The same speed is a problem for RAG and a non-issue for chat, because
        only one of them fills the context on every request."""
        slow = answered(prompt_tokens=2048, latency_s=25.0)   # ~82 tok/s
        rag = check_prefill(context(catalog, slow, tasks=(Task.RAG,),
                                    context_tokens=32768))
        chat = check_prefill(context(catalog, slow, tasks=(Task.CHAT,),
                                     context_tokens=32768))
        assert rag.status is CheckStatus.WARN
        assert rag.failure_signature == sig.PREFILL_TOO_SLOW
        assert chat.status is CheckStatus.PASS

    def test_a_fast_setup_passes_even_for_long_prompt_work(self, catalog):
        fast = answered(prompt_tokens=4096, latency_s=1.0)
        assert check_prefill(context(catalog, fast, tasks=(Task.RAG,),
                                     context_tokens=32768)).status is CheckStatus.PASS


class TestPromptSizing:
    def test_never_sends_a_prompt_longer_than_the_configured_context(self, catalog):
        """A measurement that overflows the context measures an error message."""
        seen = {}

        def record(messages, **kwargs):
            seen["chars"] = len(messages[0]["content"])
            return answered(latency_s=1.0)

        spec = catalog.require("qwen3-8b")
        ctx = CheckContext(
            client=FakeClient(chat=record), spec=spec,
            config=RuntimeConfig(runtime="llamacpp", model_id=spec.id,
                                 quant="Q4_K_M", context_tokens=2048,
                                 total_layers=spec.layers, gpu_layers=spec.layers),
            profile=UseCaseProfile(tasks=[Task.CHAT]), hardware=make_hardware(),
        )
        check_prefill(ctx)
        # ~4 characters to the token, with headroom for the framing.
        assert seen["chars"] / 4 < 2048

    def test_asks_for_one_token_so_the_latency_is_prefill(self, catalog):
        seen = {}

        def record(messages, **kwargs):
            seen.update(kwargs)
            return answered()

        ctx = context(catalog, None)
        ctx.client = FakeClient(chat=record)
        check_prefill(ctx)
        assert seen["max_tokens"] == 1


class TestWiring:
    def test_prefill_runs_as_part_of_a_standard_verification(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.RAG]),
                               catalog.require("qwen3-8b"))
        assert "prefill" in chosen

    def test_quick_mode_skips_it(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.RAG]),
                               catalog.require("qwen3-8b"), quick=True)
        assert "prefill" not in chosen

    def test_an_embedding_model_is_not_asked_to_prefill(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.EMBEDDING]),
                               catalog.require("nomic-embed-text-v1.5"))
        assert "prefill" not in chosen


class TestRepair:
    def _state(self, catalog, **overrides):
        from llmsetup.planner.plan import build_plan
        from llmsetup.repair.strategies import RepairState
        from llmsetup.tuning.tuner import tune

        hardware = make_hardware()
        profile = UseCaseProfile(tasks=[Task.RAG], priority=Priority.SPEED)
        plan = build_plan(profile, hardware, catalog)
        tuned = tune(plan.best, hardware, profile, "llamacpp")
        for key, value in overrides.items():
            setattr(tuned.config, key, value)
        return RepairState(config=tuned.config, candidate=plan.best,
                           hardware=hardware, profile=profile, catalog=catalog)

    def test_raising_the_micro_batch_is_the_first_answer(self, catalog):
        """It is the flag that moves prompt throughput most, and it costs only
        compute-buffer memory."""
        from llmsetup.repair.strategies import _increase_batch

        state = self._state(catalog, ubatch_size=128, batch_size=256)
        outcome = _increase_batch(state)
        assert outcome is not None
        assert state.config.ubatch_size == 256
        assert state.config.batch_size >= state.config.ubatch_size

    def test_stops_before_the_micro_batch_becomes_its_own_problem(self, catalog):
        from llmsetup.repair.strategies import _MAX_UBATCH, _increase_batch

        state = self._state(catalog, ubatch_size=_MAX_UBATCH)
        assert _increase_batch(state) is None

    def test_slow_prefill_has_somewhere_to_go(self, catalog):
        from llmsetup.repair.strategies import strategies_for

        names = {s.name for s in strategies_for(sig.PREFILL_TOO_SLOW)}
        assert "increase_batch" in names
        assert "increase_gpu_layers" in names

    def test_the_batch_strategies_cannot_fight_each_other(self, catalog):
        """`reduce_batch` answers OOM and `increase_batch` answers slow prefill.
        They must never both be offered for the same signature, or the loop
        would undo its own work."""
        from llmsetup.repair.strategies import strategies_for

        for signature in (sig.OUT_OF_MEMORY, sig.PREFILL_TOO_SLOW):
            names = {s.name for s in strategies_for(signature)}
            assert not {"reduce_batch", "increase_batch"} <= names
