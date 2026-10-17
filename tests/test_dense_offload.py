"""Losing memory gently on a dense model.

Until now a dense model that did not fit lost whole layers from the
accelerator, which costs an order of magnitude because a dropped layer takes
its attention to the CPU along with its weights. Attention is the half that
wants to be near the cache. Moving only the feed-forward weights frees
comparable memory for a fraction of the cost, which is the same argument that
makes expert offload worth doing on a mixture of experts, and llmsetup was making
it in only one of the two places.
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import make_hardware

from llmsetup.planner.memory import estimate_memory
from llmsetup.planner.moe import DENSE_FFN_FRACTION, dense_ffn_layers_to_offload
from llmsetup.planner.plan import build_plan
from llmsetup.runtimes.llamacpp import LlamaCppRuntime
from llmsetup.tuning.tuner import tune
from llmsetup.types import GIB, ModelSpec, Task, UseCaseProfile


def dense(**kwargs) -> ModelSpec:
    base = {"id": "d", "name": "D", "publisher": "p", "params_b": 32.0,
            "layers": 64, "hidden_size": 5120, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072}
    base.update(kwargs)
    return ModelSpec(**base)


class TestHowManyLayersMove:
    def test_nothing_moves_when_the_model_fits(self):
        assert dense_ffn_layers_to_offload(dense(), 4.85, 0) == 0
        assert dense_ffn_layers_to_offload(dense(), 4.85, -1) == 0

    def test_more_shortfall_moves_more_layers(self):
        model = dense()
        small = dense_ffn_layers_to_offload(model, 4.85, int(2 * GIB))
        large = dense_ffn_layers_to_offload(model, 4.85, int(8 * GIB))
        assert 0 < small < large

    def test_it_frees_at_least_the_shortfall_or_says_it_cannot(self):
        """Under-freeing means an out-of-memory error later rather than a
        slightly slower model now. The feed-forward weights are finite, so past
        a point the honest answer is every layer, and the caller reads that as
        "this route is not enough" and drops layers instead."""
        model = dense()
        weights = model.params_b * 1e9 * 4.85 / 8
        per_layer = weights * DENSE_FFN_FRACTION / model.layers
        for excess_gb in (1, 3, 6, 12):
            excess = excess_gb * GIB
            layers = dense_ffn_layers_to_offload(model, 4.85, int(excess))
            assert layers * per_layer >= excess - 1 or layers == model.layers

    def test_it_declines_rather_than_pretending_when_it_is_not_enough(self, catalog):
        """Taking every layer's feed-forward and still not fitting is not a
        plan, so the caller has to be able to tell that apart from success."""
        model = dense()
        weights = model.params_b * 1e9 * 4.85 / 8
        every_layer = weights * DENSE_FFN_FRACTION
        assert dense_ffn_layers_to_offload(
            model, 4.85, int(every_layer * 2)) == model.layers

    def test_it_never_asks_for_more_layers_than_exist(self):
        assert dense_ffn_layers_to_offload(dense(), 4.85, int(500 * GIB)) == 64

    def test_a_mixture_of_experts_uses_its_own_route(self):
        """Expert offload moves more bytes for less speed, so it wins where it
        applies and this must not compete with it."""
        moe = dense(active_params_b=3.0)
        assert dense_ffn_layers_to_offload(moe, 4.85, int(8 * GIB)) == 0


class TestItReplacesDroppingLayers:
    def _tuned(self, catalog, model_id, vram_gb, context=8192):
        model = catalog.require(model_id)
        hardware = make_hardware(vram_gb=vram_gb, ram_gb=128)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=context)
        plan = build_plan(profile, hardware, catalog)
        candidate = dataclasses.replace(
            plan.best, model=model, quant="Q4_K_M",
            estimate=estimate_memory(model, catalog.bpw("Q4_K_M"), context,
                                     quant="Q4_K_M"))
        return model, tune(candidate, hardware, profile, "llamacpp",
                           catalog=catalog)

    @pytest.mark.parametrize("model_id,vram_gb", [
        ("qwen3-32b", 12), ("llama-3.3-70b", 24), ("qwen3-14b", 6),
    ])
    def test_a_dense_model_that_does_not_fit_moves_its_feed_forward(
            self, catalog, model_id, vram_gb):
        _, tuned = self._tuned(catalog, model_id, vram_gb)
        assert tuned.config.advanced.offload.n_cpu_ffn > 0

    def test_every_layer_stays_on_the_accelerator(self, catalog):
        """That is the whole point: attention stays near the cache."""
        model, tuned = self._tuned(catalog, "qwen3-32b", 12)
        assert tuned.config.gpu_layers == model.layers

    def test_the_command_does_not_contradict_itself(self, catalog):
        """Asking for 35 layers and for the feed-forward of 48 is not a
        configuration. The layer count has to follow the plan."""
        model, tuned = self._tuned(catalog, "qwen3-32b", 12)
        command = LlamaCppRuntime().launch_command(model, tuned.config)
        layers = int(command[command.index("--n-gpu-layers") + 1])
        ffn = int(command[command.index("--n-cpu-ffn") + 1])
        assert ffn <= layers

    def test_a_model_that_fits_gets_neither(self, catalog):
        _, tuned = self._tuned(catalog, "qwen3-8b", 24)
        assert tuned.config.advanced.offload.n_cpu_ffn == 0

    def test_it_says_what_it_did_instead(self, catalog):
        _, tuned = self._tuned(catalog, "qwen3-32b", 12)
        rationale = tuned.config.advanced.offload.rationale
        assert "instead of dropping" in rationale


class TestThreadsAndNuma:
    def _config(self, catalog, cores, vram_gb, ram_gb=128):
        hardware = make_hardware(vram_gb=vram_gb, ram_gb=ram_gb, cores=cores)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        return tune(plan.best, hardware, profile, "llamacpp",
                    catalog=catalog).config

    def test_prompt_processing_gets_the_hyperthreads(self, catalog):
        """The two phases do not parallelise alike. Reading is compute-bound and
        can use them; writing is memory-bound and is slowed by them."""
        config = self._config(catalog, cores=48, vram_gb=0)
        assert config.threads_batch is not None
        assert config.threads_batch > config.threads

    def test_a_fully_offloaded_model_splits_nothing(self, catalog):
        """Neither phase is doing CPU work worth dividing."""
        config = self._config(catalog, cores=16, vram_gb=48)
        assert config.threads_batch is None

    def test_numa_is_set_where_it_helps(self, catalog):
        assert self._config(catalog, cores=48, vram_gb=0).numa == "distribute"

    def test_numa_is_left_alone_on_a_small_machine(self, catalog):
        """It is worth throughput on a multi-socket box and nothing anywhere
        else, so setting it everywhere would be noise."""
        assert self._config(catalog, cores=8, vram_gb=0).numa is None

    def test_numa_is_never_set_on_apple(self, catalog):
        from llmsetup.types import Accelerator

        hardware = make_hardware(vram_gb=64, ram_gb=64, cores=16,
                                 vendor=Accelerator.APPLE, unified=True)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        config = tune(plan.best, hardware, profile, "llamacpp",
                      catalog=catalog).config
        assert config.numa is None


class TestReasoningEffort:
    """The tier was computed and thrown away, so a model asked to think lightly
    and one asked to think hard were given the same instruction."""

    def _command(self, catalog, task):
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[task], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        model = catalog.require("qwen3-8b")
        candidate = dataclasses.replace(
            plan.best, model=model,
            estimate=estimate_memory(model, catalog.bpw("Q4_K_M"), 8192))
        config = tune(candidate, hardware, profile, "llamacpp",
                      catalog=catalog).config
        return LlamaCppRuntime().launch_command(model, config)

    def test_the_effort_reaches_the_server(self, catalog):
        command = self._command(catalog, Task.CHAT)
        assert "--reasoning-effort" in command

    def test_different_tasks_ask_for_different_effort(self, catalog):
        chat = self._command(catalog, Task.CHAT)
        agent = self._command(catalog, Task.AGENT)
        assert (chat[chat.index("--reasoning-effort") + 1]
                != agent[agent.index("--reasoning-effort") + 1])

    def test_nothing_is_asked_of_a_model_with_thinking_off(self, catalog):
        command = self._command(catalog, Task.EXTRACT)
        assert command[command.index("--reasoning") + 1] == "off"
        assert "--reasoning-effort" not in command
