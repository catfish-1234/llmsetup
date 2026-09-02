"""Setups that are not chat.

Two shapes people run that are structurally different from a conversation:
inline completion in an editor, which is a different endpoint with a different
model and a latency budget measured against typing speed, and long-form writing,
where the settings that make a model good at answering questions make it dull.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.planner.plan import build_plan, plan_completion_model
from llmfit.runtimes.registry import choose_runtime
from llmfit.tuning.profiles import ReasoningEffort
from llmfit.tuning.sampling import sampling_for
from llmfit.tuning.tuner import tune
from llmfit.types import Task, UseCaseProfile
from llmfit.verify.harness import select_checks


class TestCompletionModelChoice:
    def test_only_offered_when_asked_for(self, catalog):
        hardware = make_hardware(vram_gb=24)
        assert plan_completion_model(
            UseCaseProfile(tasks=[Task.CHAT]), hardware, catalog) is None

    def test_picks_a_model_trained_to_infill(self, catalog):
        """An instruction-tuned model handed a prefix and a suffix does not do a
        worse job of infilling, it does not do the job at all."""
        chosen = plan_completion_model(
            UseCaseProfile(tasks=[Task.AUTOCOMPLETE]), make_hardware(vram_gb=24),
            catalog)
        assert chosen is not None and chosen.supports_fim

    def test_sized_by_active_parameters_not_total(self, catalog):
        """Decode speed is set by the parameters that run per token, so a 30B
        mixture of experts activating 3B answers at 3B speed and belongs in the
        typing loop even though a dense 30B would not."""
        chosen = plan_completion_model(
            UseCaseProfile(tasks=[Task.AUTOCOMPLETE]), make_hardware(vram_gb=24),
            catalog)
        assert chosen.compute_params_b <= 8.0

    def test_returns_nothing_rather_than_something_unsuitable(self, catalog):
        """With no infill-capable model the honest answer is none. Handing back
        a chat model would produce an editor plugin that silently does nothing
        useful."""
        stripped = catalog.restrict_to("qwen3-8b")
        assert plan_completion_model(
            UseCaseProfile(tasks=[Task.AUTOCOMPLETE]), make_hardware(vram_gb=24),
            stripped) is None


class TestCompletionRuntime:
    def test_routed_to_an_engine_that_has_the_endpoint(self):
        """Ollama and LM Studio serve the same model perfectly well for chat and
        simply do not have an infill route, so an editor pointed at them gets
        nothing back and looks broken."""
        runtime, reason = choose_runtime(
            make_hardware(vram_gb=24), UseCaseProfile(tasks=[Task.AUTOCOMPLETE]))
        assert runtime == "llamacpp"
        assert "infill" in reason

    def test_chat_is_not_forced_down_the_same_path(self):
        """The rule is about needing an endpoint, not a preference for one
        engine, so it must not reach setups that do not need it."""
        _, reason = choose_runtime(make_hardware(vram_gb=24),
                                   UseCaseProfile(tasks=[Task.CHAT]))
        assert "infill" not in reason


class TestCompletionTuning:
    def _config(self, catalog, task=Task.AUTOCOMPLETE, model_id=None):
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[task], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        candidate = next((c for c in plan.candidates if c.model.id == model_id),
                         plan.best) if model_id else plan.best
        return tune(candidate, hardware, profile, "llamacpp", catalog=catalog)

    def test_thinking_is_turned_off(self, catalog):
        """A thinking block spends the entire latency budget before the first
        character of the suggestion appears."""
        config = self._config(catalog).config
        assert config.advanced.reasoning.effort is ReasoningEffort.OFF

    def test_the_prefix_is_cached(self, catalog):
        """Every keystroke resends the file above the cursor."""
        serving = self._config(catalog).config.advanced.serving
        assert serving.prefix_caching and serving.cache_reuse_tokens > 0

    def test_the_seed_is_pinned(self, catalog):
        """A completion that changes between identical cursor positions is
        noise, not variety."""
        assert self._config(catalog).config.seed is not None

    def test_no_repetition_penalties(self, catalog):
        """Completion output repeats surrounding identifiers on purpose."""
        sampler = self._config(catalog).config.advanced.sampler
        assert sampler.dry_multiplier == 0.0
        assert sampler.xtc_probability == 0.0


class TestCompletionSampling:
    def test_infilling_is_treated_as_its_own_mode(self, catalog):
        """It never goes through the chat template: the model gets a prefix and
        a suffix on a separate endpoint. A card's temperature describes the
        conversational mode and does not govern this one."""
        model = catalog.require("qwen3-coder-30b-a3b")
        chat = sampling_for(model, UseCaseProfile(tasks=[Task.CODING]))
        infill = sampling_for(model, UseCaseProfile(tasks=[Task.AUTOCOMPLETE]))
        assert infill.temperature < chat.temperature
        assert "different mode" in infill.rationale

    def test_a_reasoning_model_is_still_not_sampled_tightly(self, catalog):
        """The documented failure is repetition loops, and it does not stop
        being documented because the task wants determinism."""
        model = catalog.require("qwen3-8b")
        assert model.is_reasoning
        infill = sampling_for(model, UseCaseProfile(tasks=[Task.AUTOCOMPLETE]))
        assert infill.temperature >= 0.6

    def test_a_lower_published_value_is_not_raised_to_meet_it(self, catalog):
        model = catalog.require("mistral-small-3.2-24b")
        infill = sampling_for(model, UseCaseProfile(tasks=[Task.AUTOCOMPLETE]))
        assert infill.temperature == pytest.approx(0.15)


class TestCompletionVerification:
    def test_infill_is_checked_when_completion_is_the_point(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.AUTOCOMPLETE]),
                               catalog.require("qwen3-coder-30b-a3b"))
        assert "infill" in chosen

    def test_not_checked_for_a_plain_chat_setup(self, catalog):
        chosen = select_checks(UseCaseProfile(tasks=[Task.CHAT]),
                               catalog.require("qwen3-8b"))
        assert "infill" not in chosen

    def test_a_model_that_can_infill_gets_it_checked_anyway(self, catalog):
        """Advisory rather than required: knowing the capability works is worth
        having even when it was not the reason for the setup."""
        chosen = select_checks(UseCaseProfile(tasks=[Task.CODING]),
                               catalog.require("qwen3-coder-30b-a3b"))
        assert "infill" in chosen


class TestCreativeWriting:
    """The regression this fixes: publisher recommendations became the base for
    every task, and Mistral's card says 0.15. Someone writing fiction got flat,
    repetitive prose with nothing in the setup to point at."""

    @pytest.mark.parametrize("model_id", [
        "mistral-small-3.2-24b", "qwen3-8b", "gemma-3-12b", "llama-3.1-8b",
    ])
    def test_creative_work_gets_a_temperature_that_suits_it(self, catalog, model_id):
        sampling = sampling_for(catalog.require(model_id),
                                UseCaseProfile(tasks=[Task.CREATIVE]))
        assert sampling.temperature >= 0.9

    @pytest.mark.parametrize("model_id", [
        "mistral-small-3.2-24b", "qwen3-8b", "gemma-3-12b", "llama-3.1-8b",
    ])
    def test_min_p_comes_with_it(self, catalog, model_id):
        """The pair is the point. Temperature supplies the variety and min-p
        keeps it coherent; raising one without the other gives noise."""
        sampling = sampling_for(catalog.require(model_id),
                                UseCaseProfile(tasks=[Task.CREATIVE]))
        assert sampling.min_p >= 0.05

    def test_repetition_penalty_is_off(self, catalog):
        """DRY handles repeated phrasing without penalising words a passage
        legitimately reuses, such as a character's name."""
        sampling = sampling_for(catalog.require("qwen3-8b"),
                                UseCaseProfile(tasks=[Task.CREATIVE]))
        assert sampling.repeat_penalty == 1.0

    def test_other_tasks_still_take_the_published_value(self, catalog):
        """The carve-out is for open-ended writing only. Everywhere else the
        publisher's number stands."""
        model = catalog.require("mistral-small-3.2-24b")
        assert sampling_for(model, UseCaseProfile(tasks=[Task.CHAT])
                            ).temperature == pytest.approx(0.15)

    def test_it_says_why_it_departed_from_the_card(self, catalog):
        sampling = sampling_for(catalog.require("mistral-small-3.2-24b"),
                                UseCaseProfile(tasks=[Task.CREATIVE]))
        assert "creative" in sampling.rationale.lower()

    def test_a_family_already_sampled_high_is_left_alone(self, catalog):
        """Gemma publishes temperature 1.0, so there is nothing to raise."""
        sampling = sampling_for(catalog.require("gemma-3-12b"),
                                UseCaseProfile(tasks=[Task.CREATIVE]))
        assert sampling.temperature == pytest.approx(1.0)


def test_every_task_produces_a_working_plan(catalog):
    """A task the planner cannot serve is worse than one that is missing: the
    option appears in the interview and then fails."""
    hardware = make_hardware(vram_gb=24, ram_gb=64)
    for task in Task:
        if task is Task.EMBEDDING:
            continue  # served by the companion model, not the chat plan
        profile = UseCaseProfile(tasks=[task])
        plan = build_plan(profile, hardware, catalog)
        assert plan.best is not None, task
        tuned = tune(plan.best, hardware, profile, "llamacpp", catalog=catalog)
        assert tuned.config.context_tokens > 0, task
