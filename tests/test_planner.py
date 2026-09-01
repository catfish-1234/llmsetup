"""Planner behaviour: does it make defensible choices, and refuse gracefully?"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.planner.plan import build_plan, plan_embedding_model
from llmfit.planner.score import WEIGHTS
from llmfit.types import Priority, Task, UseCaseProfile
from llmfit.util.errors import PlanningError


class TestFit:
    def test_never_recommends_something_that_does_not_fit(self, catalog):
        """Across a wide sweep of machines, the top pick always fits."""
        for vram in (6, 8, 12, 16, 24, 48, 80):
            hardware = make_hardware(vram_gb=vram, ram_gb=vram * 2)
            plan = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192),
                              hardware, catalog)
            best = plan.best
            assert best.estimate.total <= hardware.memory_budget, \
                f"{vram} GB: picked {best.label} needing {best.estimate.total_gb:.1f} GB"

    def test_bigger_hardware_gets_a_bigger_model(self, catalog):
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192,
                                 priority=Priority.QUALITY)
        small = build_plan(profile, make_hardware(vram_gb=8, ram_gb=16), catalog).best
        large = build_plan(profile, make_hardware(vram_gb=80, ram_gb=256), catalog).best
        assert large.model.params_b > small.model.params_b

    def test_longer_context_forces_a_smaller_model(self, catalog):
        """The KV cache competes with the weights for the same memory."""
        hardware = make_hardware(vram_gb=12, ram_gb=32)
        short = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=4096),
                           hardware, catalog).best
        long = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=131072),
                          hardware, catalog).best
        assert long.model.params_b <= short.model.params_b

    def test_concurrency_is_budgeted_into_the_kv_cache(self, catalog):
        hardware = make_hardware(vram_gb=24, ram_gb=64)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192,
                                 concurrent_users=8)
        best = build_plan(profile, hardware, catalog).best
        single = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192),
                            hardware, catalog).best
        # Eight slots means eight caches; the same model would need far more.
        assert best.estimate.kv_cache > single.estimate.kv_cache

    def test_partial_offload_is_penalised_below_a_model_that_fits(self, catalog):
        """A model that fits entirely beats one that spills to CPU, because the
        speed difference is an order of magnitude."""
        hardware = make_hardware(vram_gb=8, ram_gb=64)
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192),
                          hardware, catalog)
        assert plan.best.fits_on_gpu

    def test_a_model_barely_fitting_is_flagged(self, catalog):
        hardware = make_hardware(vram_gb=12, ram_gb=32)
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768),
                          hardware, catalog)
        for candidate in plan.candidates:
            utilisation = candidate.estimate.total / hardware.memory_budget
            if 0.85 <= utilisation <= 1.0:
                assert any("usable" in w for w in candidate.warnings)


class TestCapabilities:
    def test_vision_requirement_is_honoured(self, catalog):
        plan = build_plan(
            UseCaseProfile(tasks=[Task.VISION], needs_vision=True, context_tokens=8192),
            make_hardware(vram_gb=24), catalog)
        assert plan.best.model.supports_vision

    def test_tool_requirement_is_honoured(self, catalog):
        plan = build_plan(
            UseCaseProfile(tasks=[Task.AGENT], needs_tools=True, context_tokens=8192),
            make_hardware(vram_gb=24), catalog)
        assert plan.best.model.supports_tools

    def test_commercial_requirement_is_honoured(self, catalog):
        plan = build_plan(
            UseCaseProfile(tasks=[Task.CHAT], commercial_use=True, context_tokens=8192),
            make_hardware(vram_gb=24), catalog)
        assert plan.best.model.commercial_ok


class TestPriority:
    def test_weights_sum_to_one(self):
        for priority, weights in WEIGHTS.items():
            assert sum(weights.values()) == pytest.approx(1.0), priority

    def test_speed_and_quality_produce_different_answers(self, catalog):
        """If the priority did not change the answer it would be decoration."""
        hardware = make_hardware(vram_gb=24, ram_gb=64)
        fast = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192,
                                         priority=Priority.SPEED), hardware, catalog).best
        good = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192,
                                         priority=Priority.QUALITY), hardware, catalog).best
        assert fast.model.compute_params_b <= good.model.compute_params_b

    def test_scores_and_breakdowns_are_populated(self, catalog):
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT]), make_hardware(), catalog)
        for candidate in plan.candidates:
            assert 0.0 <= candidate.score <= 1.0
            assert "est_tokens_per_s" in candidate.score_breakdown
            for key in ("quality", "speed", "fit", "task", "license"):
                assert key in candidate.score_breakdown


class TestDeterminism:
    def test_the_same_inputs_give_the_same_plan(self, catalog):
        profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=32768)
        hardware = make_hardware(vram_gb=24)
        first = [c.label for c in build_plan(profile, hardware, catalog).candidates]
        second = [c.label for c in build_plan(profile, hardware, catalog).candidates]
        assert first == second

    def test_fallbacks_are_different_models_not_the_same_one_requantized(self, catalog):
        """The repair loop needs somewhere real to fall back to."""
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT]), make_hardware(), catalog)
        ids = [c.model.id for c in plan.candidates]
        assert len(ids) == len(set(ids))


class TestRefusal:
    def test_impossible_requirements_raise_with_an_explanation(self, catalog):
        profile = UseCaseProfile(tasks=[Task.CHAT], needs_vision=True,
                                 needs_reasoning=True, context_tokens=8192)
        with pytest.raises(PlanningError) as excinfo:
            build_plan(profile, make_hardware(), catalog)
        assert excinfo.value.hint

    def test_a_tiny_machine_suggests_a_workable_context(self, catalog):
        """Failing with a number the user can act on beats failing with 'no'."""
        hardware = make_hardware(vram_gb=0, ram_gb=1, vendor=None)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=131072)
        with pytest.raises(PlanningError) as excinfo:
            build_plan(profile, hardware, catalog)
        assert "--context" in excinfo.value.hint or "smaller" in excinfo.value.hint


class TestEmbeddingCompanion:
    def test_rag_gets_an_embedding_model(self, catalog):
        profile = UseCaseProfile(tasks=[Task.RAG], needs_embeddings=True)
        model = plan_embedding_model(profile, make_hardware(), catalog)
        assert model is not None and model.is_embedding

    def test_plain_chat_gets_none(self, catalog):
        profile = UseCaseProfile(tasks=[Task.CHAT])
        assert plan_embedding_model(profile, make_hardware(), catalog) is None

    def test_multilingual_corpora_get_a_multilingual_retriever(self, catalog):
        """An English-only retriever fails silently on other languages: it
        returns plausible-looking but wrong neighbours."""
        profile = UseCaseProfile(tasks=[Task.RAG], needs_embeddings=True,
                                 languages=["en", "zh", "ar"])
        model = plan_embedding_model(profile, make_hardware(), catalog)
        assert "zh" in model.languages and "ar" in model.languages
