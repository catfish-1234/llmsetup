"""Two architecture details that change the answer, and how drafting is chosen.

Sliding-window attention and multi-token prediction are both properties of a
specific model file rather than of a family, and both change the plan by more
than a rounding: one decides whether a 27B model fits a 24 GB card at long
context, the other decides whether a second model has to be downloaded at all.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.gguf import GgufInfo, apply_to_spec
from llmfit.planner.memory import (
    estimate_memory,
    kv_bytes_per_token,
    kv_cache_bytes,
    max_context_for_budget,
)
from llmfit.planner.plan import build_plan
from llmfit.tuning.profiles import SpecType
from llmfit.tuning.tuner import tune
from llmfit.types import GIB, ModelSpec, Priority, Task, UseCaseProfile


def spec(**kwargs) -> ModelSpec:
    base = {"id": "t", "name": "T", "publisher": "p", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072}
    base.update(kwargs)
    return ModelSpec(**base)


def windowed(**kwargs) -> ModelSpec:
    return spec(layers=60, sliding_window=1024, swa_pattern=6, **kwargs)


def _as_candidate(plan, catalog, model_id, context_tokens=8192):
    """The named model as a candidate, whatever the ranking actually preferred.

    Tests here are about one model's architecture rather than about which model
    wins, so they pin the model. The estimate is recomputed for it: carrying
    another model's memory figures over would quietly change what "there is no
    room for a draft" means.
    """
    for candidate in plan.candidates:
        if candidate.model.id == model_id:
            return candidate

    from llmfit.planner.memory import estimate_memory
    from llmfit.types import Candidate

    model = catalog.require(model_id)
    quant = model.recommended_quants[0] if model.recommended_quants else "Q4_K_M"
    estimate = estimate_memory(model, catalog.bpw(quant), context_tokens, quant=quant)
    return Candidate(model=model, quant=quant, estimate=estimate,
                     fits_on_gpu=True, gpu_fraction=1.0)


class TestSlidingWindowShape:
    def test_a_model_needs_both_a_window_and_a_pattern_to_count(self):
        """A window with no interleaving, or a pattern with no window, describes
        a model that attends globally. Treating either as windowed would
        under-count the cache, which is the dangerous direction."""
        assert not spec(sliding_window=1024).uses_swa
        assert not spec(swa_pattern=6).uses_swa
        assert windowed().uses_swa

    def test_the_layers_split_the_way_the_pattern_says(self):
        model = windowed()
        assert model.global_layers == 10
        assert model.local_layers == 50
        assert model.global_layers + model.local_layers == model.layers

    def test_a_plain_model_is_all_global(self):
        model = spec()
        assert model.global_layers == model.layers
        assert model.local_layers == 0

    def test_never_claims_zero_global_layers(self):
        """A pattern larger than the layer count would otherwise divide to zero
        and make the cache look free."""
        assert spec(layers=4, sliding_window=1024, swa_pattern=99).global_layers >= 1


class TestKvMaths:
    def test_below_the_window_nothing_is_saved_yet(self):
        model = windowed()
        short = 512
        assert kv_cache_bytes(model, short) == pytest.approx(
            kv_bytes_per_token(model) * short, rel=1e-9)

    def test_the_saving_converges_on_the_published_ratio(self):
        """Gemma 3's own report puts the reduction at one sixth for a 5:1 layer
        ratio. It is a limit rather than a constant: the windowed layers still
        hold their window, so the ratio climbs towards six as the context grows
        and never quite arrives."""
        model = windowed()
        ratios = [kv_cache_bytes(model, n, swa_full=True) / kv_cache_bytes(model, n)
                  for n in (32768, 131072, 1_048_576)]
        assert ratios == sorted(ratios), "the saving should improve with length"
        assert 5.0 < ratios[1] < 6.0
        assert ratios[-1] == pytest.approx(6.0, rel=0.01)

    def test_asking_for_a_full_cache_costs_the_full_amount(self):
        model = windowed()
        assert kv_cache_bytes(model, 65536, swa_full=True) == pytest.approx(
            kv_bytes_per_token(model) * 65536, rel=1e-9)

    def test_a_plain_model_is_unaffected_by_any_of_this(self):
        model = spec()
        assert kv_cache_bytes(model, 32768) == kv_cache_bytes(
            model, 32768, swa_full=True)

    def test_quantizing_the_cache_still_scales_it(self):
        model = windowed()
        assert kv_cache_bytes(model, 32768, "q8_0") < kv_cache_bytes(model, 32768)


class TestContextInversion:
    """The inverse has to agree with the forward calculation, or the repair loop
    shrinks context to a number that still does not fit."""

    @pytest.mark.parametrize("model", [spec(), windowed()])
    @pytest.mark.parametrize("budget_gb", [8, 16, 24, 48])
    def test_what_it_says_fits_actually_fits(self, model, budget_gb):
        budget = int(budget_gb * GIB)
        fitted = max_context_for_budget(model, 4.85, budget)
        if fitted == 0:
            return
        assert estimate_memory(model, 4.85, fitted).total <= budget

    @pytest.mark.parametrize("model", [spec(), windowed()])
    def test_one_more_step_would_not_fit(self, model):
        """Not just safe, but tight: a needlessly short context is a real cost."""
        budget = int(16 * GIB)
        fitted = max_context_for_budget(model, 4.85, budget, granularity=1024)
        if fitted == 0 or fitted >= model.max_context:
            return
        assert estimate_memory(model, 4.85, fitted + 1024).total > budget

    def test_a_windowed_model_is_told_it_fits_far_more(self):
        """This is the whole point: inverting the flat formula would hand back a
        context several times shorter than the model can really take."""
        model = windowed(max_context=1_048_576)
        budget = int(12 * GIB)
        windowed_fit = max_context_for_budget(model, 4.85, budget)
        flat_fit = max_context_for_budget(model, 4.85, budget, swa_full=True)
        assert 0 < flat_fit < model.max_context, "the comparison must not saturate"
        assert windowed_fit > 3 * flat_fit


class TestReadingItFromTheFile:
    def test_a_window_in_the_header_overrides_the_catalog(self):
        updated, corrections = apply_to_spec(
            spec(), GgufInfo(architecture="gemma3", sliding_window=1024))
        assert updated.sliding_window == 1024
        assert any("sliding-window" in c for c in corrections)

    def test_a_stated_window_with_no_known_pattern_assumes_every_layer(self):
        """The conservative reading. Claiming an interleave the file does not
        state would over-estimate the saving, and over-estimating a saving is
        how a model stops fitting after download."""
        updated, _ = apply_to_spec(
            spec(layers=32), GgufInfo(architecture="x", sliding_window=4096))
        assert updated.swa_pattern == 32
        assert updated.global_layers == 1

    def test_prediction_heads_are_read_from_the_build_not_the_name(self):
        """Publishers ship an MTP build and a plain build of the same model, so
        the model's identity settles nothing."""
        updated, corrections = apply_to_spec(
            spec(), GgufInfo(architecture="qwen3", nextn_predict_layers=1))
        assert updated.mtp_heads is True
        assert any("multi-token" in c for c in corrections)

    def test_a_plain_build_of_an_mtp_model_is_corrected_downwards(self):
        """The direction that matters: believing in heads that are not there
        would emit a draft type the server cannot honour."""
        updated, corrections = apply_to_spec(
            spec(mtp_heads=True), GgufInfo(architecture="qwen3",
                                           nextn_predict_layers=0))
        assert updated.mtp_heads is False
        assert any("absent" in c for c in corrections)

    def test_zero_heads_is_not_mtp(self):
        assert not GgufInfo(nextn_predict_layers=0).has_mtp
        assert not GgufInfo().has_mtp
        assert GgufInfo(nextn_predict_layers=2).has_mtp


class TestDraftingChoice:
    def _plan(self, catalog, model_id, hardware, profile):
        plan = build_plan(profile, hardware, catalog)
        candidate = _as_candidate(plan, catalog, model_id,
                                  profile.context_tokens)
        return tune(candidate, hardware, profile, "llamacpp",
                    catalog=catalog).config.advanced.speculative

    def test_prediction_heads_win_over_a_draft_model(self, catalog):
        """They cost no download and no resident second model, so there is
        nothing for a draft model to be better than."""
        catalog.require("qwen3-32b").mtp_heads = True
        try:
            plan = self._plan(catalog, "qwen3-32b", make_hardware(vram_gb=48, ram_gb=128),
                              UseCaseProfile(tasks=[Task.CODING], context_tokens=8192))
            assert plan.spec_type is SpecType.MTP
            assert plan.draft_model_id is None
            assert not plan.needs_draft_model
        finally:
            catalog.require("qwen3-32b").mtp_heads = False

    def test_prediction_heads_apply_even_to_open_ended_work(self, catalog):
        """A draft model is withheld on open-ended tasks because it costs memory
        that could be context. Heads inside the file cost neither."""
        catalog.require("qwen3-14b").mtp_heads = True
        try:
            plan = self._plan(catalog, "qwen3-14b", make_hardware(vram_gb=48, ram_gb=128),
                              UseCaseProfile(tasks=[Task.CREATIVE], context_tokens=8192))
            assert plan.spec_type is SpecType.MTP
        finally:
            catalog.require("qwen3-14b").mtp_heads = False

    def test_falls_back_to_drafting_from_the_prompt_when_nothing_fits(self, catalog):
        plan = self._plan(catalog, "qwen3-32b", make_hardware(vram_gb=24, ram_gb=64),
                          UseCaseProfile(tasks=[Task.RAG], context_tokens=32768))
        assert plan.spec_type is SpecType.NGRAM
        assert plan.draft_memory_bytes == 0

    def test_does_not_draft_from_the_prompt_for_open_ended_work(self, catalog):
        """It guesses by matching text already in the context, so it only helps
        where the answer quotes the source."""
        plan = self._plan(catalog, "qwen3-32b", make_hardware(vram_gb=24, ram_gb=64),
                          UseCaseProfile(tasks=[Task.CREATIVE], context_tokens=32768))
        assert plan.spec_type is SpecType.NONE

    def test_no_speedup_is_claimed_for_prompt_drafting(self, catalog):
        """Acceptance depends entirely on how much of the answer is quotation,
        which varies per request. A number here would read as a promise."""
        plan = self._plan(catalog, "qwen3-32b", make_hardware(vram_gb=24, ram_gb=64),
                          UseCaseProfile(tasks=[Task.RAG], context_tokens=32768))
        assert plan.expected_speedup == 1.0


class TestEmittedFlags:
    def _command(self, catalog, model_id, hardware, profile, **config_overrides):
        from llmfit.runtimes.llamacpp import LlamaCppRuntime

        plan = build_plan(profile, hardware, catalog)
        candidate = _as_candidate(plan, catalog, model_id,
                                  profile.context_tokens)
        config = tune(candidate, hardware, profile, "llamacpp",
                      catalog=catalog).config
        for key, value in config_overrides.items():
            setattr(config, key, value)
        return LlamaCppRuntime().launch_command(candidate.model, config)

    def test_deprecated_flags_are_gone(self, catalog):
        """llama.cpp marked these deprecated; emitting them earns a warning now
        and a failure later."""
        command = self._command(catalog, "qwen3-8b", make_hardware(vram_gb=24),
                                UseCaseProfile(tasks=[Task.AGENT], context_tokens=8192))
        assert "--defrag-thold" not in command

    def test_the_full_cache_flag_is_emitted_only_when_asked_for(self, catalog):
        profile = UseCaseProfile(tasks=[Task.RAG], context_tokens=8192)
        hardware = make_hardware(vram_gb=24)
        assert "--swa-full" not in self._command(catalog, "gemma-3-12b", hardware, profile)
        assert "--swa-full" in self._command(catalog, "gemma-3-12b", hardware, profile,
                                             swa_full=True)

    def test_prompt_drafting_emits_no_draft_model(self, catalog):
        """There is no second model, so a draft path would be a lie the server
        would reject at load."""
        command = self._command(catalog, "qwen3-32b", make_hardware(vram_gb=24, ram_gb=64),
                                UseCaseProfile(tasks=[Task.RAG], context_tokens=32768))
        assert "--spec-type" in command
        assert command[command.index("--spec-type") + 1] == "ngram-cache"
        assert "--spec-draft-model" not in command

    def test_an_agent_pins_its_system_prompt(self, catalog):
        command = self._command(catalog, "qwen3-8b", make_hardware(vram_gb=24),
                                UseCaseProfile(tasks=[Task.AGENT], context_tokens=8192))
        assert "--keep" in command

    def test_a_reasoning_model_gets_a_thinking_ceiling_for_agent_work(self, catalog):
        command = self._command(catalog, "qwen3-8b", make_hardware(vram_gb=24),
                                UseCaseProfile(tasks=[Task.AGENT], context_tokens=8192))
        assert "--reasoning-budget" in command


class TestSwaRepair:
    def _state(self, catalog, model_id, hardware, **overrides):
        from llmfit.repair.strategies import RepairState

        profile = UseCaseProfile(tasks=[Task.RAG], priority=Priority.QUALITY,
                                 context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        candidate = _as_candidate(plan, catalog, model_id)
        tuned = tune(candidate, hardware, profile, "llamacpp", catalog=catalog)
        for key, value in overrides.items():
            setattr(tuned.config, key, value)
        return RepairState(config=tuned.config, candidate=candidate,
                           hardware=hardware, profile=profile, catalog=catalog)

    def test_recall_failure_on_a_windowed_model_widens_the_cache(self, catalog):
        from llmfit.repair.strategies import _full_swa_cache

        state = self._state(catalog, "gemma-3-4b", make_hardware(vram_gb=48))
        outcome = _full_swa_cache(state)
        assert outcome is not None
        assert state.config.swa_full is True
        assert "whole prompt" in outcome.description

    def test_does_nothing_for_a_model_without_a_window(self, catalog):
        from llmfit.repair.strategies import _full_swa_cache

        state = self._state(catalog, "qwen3-8b", make_hardware(vram_gb=48))
        assert _full_swa_cache(state) is None

    def test_refuses_rather_than_causing_the_next_failure(self, catalog):
        """Turning it on costs the memory the window was saving. Doing that
        without checking would trade a recall failure for an out-of-memory one."""
        from llmfit.repair.strategies import _full_swa_cache

        state = self._state(catalog, "gemma-3-27b", make_hardware(vram_gb=24),
                            context_tokens=131072)
        assert _full_swa_cache(state) is None
        assert state.config.swa_full is False

    def test_it_is_registered_for_the_recall_signature(self):
        from llmfit.repair.strategies import STRATEGIES
        from llmfit.verify import signatures as sig

        named = {s.name: s for s in STRATEGIES}
        assert named["full_swa_cache"].handles(sig.NEEDLE_NOT_FOUND)
