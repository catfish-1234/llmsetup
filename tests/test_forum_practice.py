"""Two things practitioners do by hand that arithmetic does better.

Both come from people posting working configurations and the numbers they got.
The first is an ordering mistake llmsetup was making that they were not: giving up
context, which the user asked for, before giving up prompt-batch size, which
they did not. The second is a trick they do by writing regexes and arriving at
by trial, where the boundary is simply computable.
"""

from __future__ import annotations

import re

import pytest
from conftest import make_hardware

from llmsetup.planner.memory import estimate_memory
from llmsetup.planner.moe import EXPERT_TENSORS, expert_bytes, plan_expert_offload
from llmsetup.planner.plan import build_plan
from llmsetup.tuning.tuner import tune
from llmsetup.types import GIB, ModelSpec, Task, UseCaseProfile


def moe_spec(**kwargs) -> ModelSpec:
    base = {"id": "moe", "name": "MoE", "publisher": "p", "params_b": 30.0,
            "active_params_b": 3.0, "layers": 48, "hidden_size": 2048,
            "kv_heads": 4, "head_dim": 128, "max_context": 262144}
    base.update(kwargs)
    return ModelSpec(**base)


class TestContextIsNotTheFirstThingGivenUp:
    """A practitioner reported a third more usable context by dropping the batch
    from 4096/1024 to 128/512 with no throughput they could measure. llmsetup was
    fixing the batch before it fitted the context, so it cut the context
    instead."""

    def _tuned(self, catalog, vram_gb, context, ram_gb=32):
        hardware = make_hardware(vram_gb=vram_gb, ram_gb=ram_gb)
        profile = UseCaseProfile(tasks=[Task.RAG], context_tokens=context)
        plan = build_plan(profile, hardware, catalog)
        return plan, tune(plan.best, hardware, profile, "llamacpp", catalog=catalog)

    @pytest.mark.parametrize("vram_gb,context", [
        (8, 16384), (12, 32768), (16, 65536), (24, 131072),
    ])
    def test_the_requested_context_survives_where_it_can(self, catalog,
                                                         vram_gb, context):
        _, tuned = self._tuned(catalog, vram_gb, context)
        assert tuned.config.context_tokens == context

    def test_it_shrinks_the_batch_rather_than_the_context(self, catalog):
        """The trade is stated in that direction on purpose: the context was the
        requirement and the batch size was an implementation detail."""
        _, tight = self._tuned(catalog, 8, 16384)
        _, roomy = self._tuned(catalog, 48, 16384)
        assert tight.config.ubatch_size <= roomy.config.ubatch_size

    def test_the_batch_it_reports_is_the_batch_it_sets(self, catalog):
        """A decision that appears in the report and not in the config is worse
        than no decision, because it reads as done."""
        _, tuned = self._tuned(catalog, 8, 16384)
        reported = [d for d in tuned.decisions if d.setting == "batch/ubatch"]
        assert reported, "the batch decision is always recorded"
        batch, _, ubatch = reported[-1].value.partition("/")
        assert int(ubatch) == tuned.config.ubatch_size
        assert int(batch) == tuned.config.batch_size

    def test_the_logical_batch_never_drops_below_the_micro_batch(self, catalog):
        for vram_gb, context in ((8, 16384), (6, 8192), (24, 131072)):
            _, tuned = self._tuned(catalog, vram_gb, context)
            assert tuned.config.batch_size >= tuned.config.ubatch_size

    def test_it_still_fits_what_it_claims(self, catalog):
        """The whole point is a config that runs, so the reduced batch has to be
        the one the memory figures were computed against."""
        plan, tuned = self._tuned(catalog, 8, 16384)
        config = tuned.config
        estimate = estimate_memory(
            plan.best.model, 4.85, config.context_tokens, quant=config.quant,
            kv_quant=config.kv_cache_type, runtime=config.runtime,
            ubatch=config.ubatch_size, parallel=config.parallel,
        )
        assert estimate.total <= make_hardware(vram_gb=8, ram_gb=32).memory_budget

    def test_the_batch_is_not_cut_when_it_would_not_help(self, catalog):
        """When the shortfall is in the weights, a smaller batch gives up speed
        for nothing and the honest answer is a shorter context."""
        _, roomy = self._tuned(catalog, 48, 8192)
        assert roomy.config.ubatch_size >= 512


class TestExpertsMoveByTheTensor:
    """Whole-layer offload rounds up, and the round-up is expert weights sitting
    in RAM that the card had room for. People reclaim it by hand with regexes
    like `-ot 'blk\\.21\\.ffn_down.*=CPU'`; the boundary is arithmetic."""

    def test_a_small_shortfall_moves_part_of_one_layer(self):
        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        plan = plan_expert_offload(model, 4.85, int(per_layer * 0.3))
        assert plan.whole_layers == 0
        assert plan.is_partial
        assert len(plan.partial_tensors) == 1

    def test_it_frees_at_least_what_was_asked_for(self):
        """Under-freeing is the dangerous direction: it means an out-of-memory
        error later rather than a slightly slower model now."""
        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        per_tensor = per_layer / len(EXPERT_TENSORS)
        for fraction in (0.1, 0.3, 0.5, 0.9, 1.4, 2.2, 5.7):
            excess = per_layer * fraction
            plan = plan_expert_offload(model, 4.85, int(excess))
            freed = (plan.whole_layers * per_layer
                     + len(plan.partial_tensors) * per_tensor)
            assert freed >= excess - 1, f"under-freed at {fraction} of a layer"

    def test_it_frees_less_than_whole_layer_rounding_would(self):
        """That is the entire point. Every expert left on the card is one that
        does not cross the bus when it fires."""
        import math

        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        per_tensor = per_layer / len(EXPERT_TENSORS)
        excess = per_layer * 2.2
        plan = plan_expert_offload(model, 4.85, int(excess))
        freed = plan.whole_layers * per_layer + len(plan.partial_tensors) * per_tensor
        rounded_up = math.ceil(excess / per_layer) * per_layer
        assert freed < rounded_up

    def test_a_fraction_worth_a_whole_layer_just_takes_the_layer(self):
        """Three tensors is a layer, and saying so keeps the command line
        simpler and does not depend on this build naming its tensors the way a
        regex expects."""
        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        plan = plan_expert_offload(model, 4.85, int(per_layer * 0.99))
        assert plan.whole_layers == 1
        assert not plan.is_partial

    def test_nothing_moves_when_nothing_needs_to(self):
        assert plan_expert_offload(moe_spec(), 4.85, 0).whole_layers == 0
        assert not plan_expert_offload(moe_spec(), 4.85, -1).is_partial

    def test_a_dense_model_has_no_experts_to_move(self):
        dense = moe_spec(active_params_b=None)
        plan = plan_expert_offload(dense, 4.85, int(2 * GIB))
        assert plan.whole_layers == 0 and not plan.is_partial

    def test_asking_for_more_than_the_experts_are_worth_takes_all_of_them(self):
        model = moe_spec()
        plan = plan_expert_offload(model, 4.85, int(999 * GIB))
        assert plan.whole_layers == model.layers
        assert not plan.is_partial, "there is no fraction left to be clever about"

    def test_the_pattern_is_valid_and_names_only_the_boundary_layer(self):
        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        plan = plan_expert_offload(model, 4.85, int(per_layer * 3.4))
        patterns = plan.override_patterns()
        assert len(patterns) == 1
        # It has to compile, and it has to match the layer it claims to.
        expression = patterns[0].split("=")[0]
        compiled = re.compile(expression)
        assert compiled.match(f"blk.{plan.partial_layer}.{plan.partial_tensors[0]}")
        assert not compiled.match(f"blk.{plan.partial_layer + 1}.{plan.partial_tensors[0]}")

    def test_whole_layers_need_no_pattern(self):
        """They are said more simply with --n-cpu-moe, which does not depend on
        tensor names at all."""
        model = moe_spec()
        per_layer = expert_bytes(model, 4.85) / model.layers
        assert plan_expert_offload(model, 4.85, int(per_layer * 2)).override_patterns() == []


class TestItReachesTheCommandLine:
    def test_a_partial_offload_emits_its_pattern(self, catalog):
        from llmsetup.runtimes.llamacpp import LlamaCppRuntime
        from llmsetup.tuning.profiles import AdvancedProfile, OffloadPlan
        from llmsetup.types import RuntimeConfig

        config = RuntimeConfig(runtime="llamacpp", model_id="m", quant="Q4_K_M",
                               context_tokens=8192, gpu_layers=48, total_layers=48)
        config.advanced = AdvancedProfile(offload=OffloadPlan(
            gpu_layers=48, total_layers=48, n_cpu_moe=5,
            override_tensors=["blk\\.5\\.(ffn_gate_exps)=CPU"],
        ))
        command = LlamaCppRuntime().launch_command(moe_spec(), config)
        assert "--n-cpu-moe" in command
        assert "--override-tensor" in command
        assert command[command.index("--override-tensor") + 1].startswith("blk")


class TestBatchGrowsIntoHeadroom:
    """The case that exposed this: someone ran a 35B mixture of experts on a
    6 GB laptop card at `-ub 2048` and measured 350-400 prompt tokens per
    second. llmsetup sized the batch from the card alone and gave that machine
    256, because a table indexed on VRAM cannot see that most of the model's
    weight is not on the card."""

    def _tuned(self, catalog, model_id, vram_gb, context, ram_gb=64):
        import dataclasses

        hardware = make_hardware(vram_gb=vram_gb, ram_gb=ram_gb)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=context)
        plan = build_plan(profile, hardware, catalog)
        model = catalog.require(model_id)
        candidate = next((c for c in plan.candidates if c.model.id == model_id), None)
        if candidate is None:
            quant = model.recommended_quants[0]
            candidate = dataclasses.replace(
                plan.best, model=model, quant=quant,
                estimate=estimate_memory(model, catalog.bpw(quant), context,
                                         quant=quant))
        return tune(candidate, hardware, profile, "llamacpp", catalog=catalog)

    def test_a_small_card_running_an_moe_gets_a_wide_batch(self, catalog):
        tuned = self._tuned(catalog, "qwen3.6-35b-a3b", vram_gb=6, context=102400)
        assert tuned.config.ubatch_size >= 2048

    def test_a_small_card_running_a_dense_model_does_not(self, catalog):
        """The headroom is real for a mixture of experts because its weight
        leaves the card. A dense model's does not, so there is nothing to grow
        into and claiming otherwise would mean an out-of-memory error."""
        tuned = self._tuned(catalog, "qwen3-8b", vram_gb=6, context=8192)
        assert tuned.config.ubatch_size <= 1024

    def test_it_never_grows_past_where_returns_stop(self, catalog):
        """Larger micro-batches show diminishing returns and long prompts start
        behaving unpredictably, so there is a ceiling."""
        for vram_gb in (24, 48, 96):
            tuned = self._tuned(catalog, "qwen3.6-35b-a3b", vram_gb, context=32768)
            assert tuned.config.ubatch_size <= 2048

    def test_what_it_grows_to_still_fits(self, catalog):
        """A batch grown past the budget is an out-of-memory error dressed up as
        a tuning decision."""
        from llmsetup.planner.memory import overhead_bytes

        hardware = make_hardware(vram_gb=6, ram_gb=64)
        tuned = self._tuned(catalog, "qwen3.6-35b-a3b", vram_gb=6, context=102400)
        config = tuned.config
        model = catalog.require("qwen3.6-35b-a3b")
        buffers = overhead_bytes(model, config.runtime, config.ubatch_size,
                                 config.parallel)
        cache = estimate_memory(model, 4.85, config.context_tokens,
                                kv_quant=config.kv_cache_type).kv_cache
        assert buffers + cache < hardware.memory_budget

    def test_the_reported_batch_matches_the_config(self, catalog):
        tuned = self._tuned(catalog, "qwen3.6-35b-a3b", vram_gb=6, context=102400)
        reported = [d for d in tuned.decisions if d.setting == "batch/ubatch"][-1]
        batch, _, ubatch = reported.value.partition("/")
        assert int(ubatch) == tuned.config.ubatch_size
        assert int(batch) == tuned.config.batch_size

    def test_it_keeps_experts_on_the_card_rather_than_moving_all_of_them(
            self, catalog):
        """The published configuration used --n-cpu-moe 99, which moves every
        expert. Whatever the card can hold is decode speed left on the table."""
        tuned = self._tuned(catalog, "qwen3.6-35b-a3b", vram_gb=6, context=102400)
        model = catalog.require("qwen3.6-35b-a3b")
        assert tuned.config.advanced.offload.n_cpu_moe < model.layers
