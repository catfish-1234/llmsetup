"""Tuner: are the derived settings correct and self-consistent?"""

from __future__ import annotations

from conftest import make_hardware

from llmfit.planner.plan import build_plan
from llmfit.tuning.sampling import sampling_for
from llmfit.tuning.tuner import tune
from llmfit.types import Task, UseCaseProfile


def plan_and_tune(catalog, hardware, profile, runtime="llamacpp"):
    plan = build_plan(profile, hardware, catalog)
    return plan.best, tune(plan.best, hardware, profile, runtime)


class TestOffload:
    def test_offloads_everything_when_it_fits(self, catalog):
        hardware = make_hardware(vram_gb=24, ram_gb=64)
        candidate, tuned = plan_and_tune(
            catalog, hardware, UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        assert tuned.config.gpu_layers == candidate.model.layers
        assert tuned.config.full_gpu_offload

    def test_offloads_nothing_without_a_gpu(self, catalog):
        hardware = make_hardware(vram_gb=0, ram_gb=32, vendor=None, cores=8)
        _, tuned = plan_and_tune(
            catalog, hardware, UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        assert tuned.config.gpu_layers == 0

    def test_the_final_estimate_fits_the_budget(self, catalog):
        """Whatever the tuner settles on must actually fit — the whole point."""
        for vram in (8, 12, 16, 24, 48):
            hardware = make_hardware(vram_gb=vram, ram_gb=vram * 2)
            profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=32768)
            _, tuned = plan_and_tune(catalog, hardware, profile)
            if tuned.config.full_gpu_offload:
                assert tuned.estimate.total <= hardware.memory_budget, \
                    f"{vram} GB: {tuned.estimate.total_gb:.1f} GB does not fit"

    def test_context_is_reduced_rather_than_spilling_to_cpu(self, catalog):
        """A shorter context on the GPU beats a longer one on the CPU."""
        hardware = make_hardware(vram_gb=8, ram_gb=32)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=131072)
        _, tuned = plan_and_tune(catalog, hardware, profile)
        assert tuned.config.context_tokens <= 131072
        if tuned.config.context_tokens < 131072:
            assert tuned.config.full_gpu_offload

    def test_context_never_falls_below_the_usable_floor_for_the_task(self, catalog):
        hardware = make_hardware(vram_gb=8, ram_gb=16)
        profile = UseCaseProfile(tasks=[Task.RAG], context_tokens=131072)
        _, tuned = plan_and_tune(catalog, hardware, profile)
        assert tuned.config.context_tokens >= 8192


class TestThreads:
    def test_uses_physical_cores_not_hyperthreads(self, catalog):
        hardware = make_hardware(vram_gb=0, ram_gb=32, vendor=None, cores=8)
        _, tuned = plan_and_tune(
            catalog, hardware, UseCaseProfile(tasks=[Task.CHAT], context_tokens=4096))
        assert tuned.config.threads is not None
        assert tuned.config.threads <= hardware.physical_cores

    def test_a_fully_offloaded_model_needs_few_threads(self, catalog):
        hardware = make_hardware(vram_gb=24, ram_gb=64, cores=16)
        _, tuned = plan_and_tune(
            catalog, hardware, UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        assert tuned.config.threads <= 4


class TestFlashAttention:
    def test_enabled_on_modern_nvidia(self, catalog):
        hardware = make_hardware(vram_gb=24, compute="8.9")
        _, tuned = plan_and_tune(catalog, hardware, UseCaseProfile(tasks=[Task.CHAT]))
        assert tuned.config.flash_attention

    def test_disabled_without_a_gpu(self, catalog):
        hardware = make_hardware(vram_gb=0, ram_gb=32, vendor=None)
        _, tuned = plan_and_tune(catalog, hardware, UseCaseProfile(tasks=[Task.CHAT]))
        assert not tuned.config.flash_attention

    def test_forced_on_whenever_the_kv_cache_is_quantized(self, catalog):
        """llama.cpp backends require it for quantized KV; shipping one without
        the other produces a server that refuses to start."""
        hardware = make_hardware(vram_gb=12, ram_gb=32)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=131072)
        _, tuned = plan_and_tune(catalog, hardware, profile)
        if tuned.config.kv_cache_type != "f16":
            assert tuned.config.flash_attention


class TestSampling:
    def test_code_is_sampled_near_deterministically(self, catalog):
        model = catalog.require("qwen3-coder-30b-a3b")
        sampling = sampling_for(model, UseCaseProfile(tasks=[Task.CODING]))
        assert sampling.temperature <= 0.3

    def test_creative_writing_is_sampled_loosely(self, catalog):
        model = catalog.require("gemma-3-27b")
        sampling = sampling_for(model, UseCaseProfile(tasks=[Task.CREATIVE]))
        assert sampling.temperature >= 0.9

    def test_extraction_is_fully_deterministic_on_a_non_reasoning_model(self, catalog):
        model = catalog.require("mistral-small-3.2-24b")
        assert not model.is_reasoning
        sampling = sampling_for(model, UseCaseProfile(tasks=[Task.EXTRACT]))
        assert sampling.temperature == 0.0

    def test_the_reasoning_floor_overrides_even_deterministic_tasks(self, catalog):
        """Deliberate: a reasoning model at temperature 0 does not produce
        crisper extraction, it produces a repetition loop. The floor wins, and
        the rationale says why."""
        model = catalog.require("qwen3-8b")
        assert model.is_reasoning
        sampling = sampling_for(model, UseCaseProfile(tasks=[Task.EXTRACT]))
        assert sampling.temperature >= 0.6
        assert "reasoning" in sampling.rationale

    def test_reasoning_models_are_lifted_off_greedy_sampling(self, catalog):
        """Reasoning models loop when sampled too tightly, so the task's low
        temperature is raised to the family floor rather than applied blindly."""
        reasoning = catalog.require("deepseek-r1-distill-qwen-32b")
        assert reasoning.is_reasoning
        sampling = sampling_for(reasoning, UseCaseProfile(tasks=[Task.EXTRACT]))
        assert sampling.temperature >= 0.6
        assert sampling.repeat_penalty == 1.0
        assert "reasoning" in sampling.rationale

    def test_rationale_is_always_populated(self, catalog):
        for task in Task:
            sampling = sampling_for(catalog.require("qwen3-8b"),
                                    UseCaseProfile(tasks=[task]))
            assert sampling.rationale


class TestExplanation:
    def test_every_decision_has_a_reason(self, catalog):
        _, tuned = plan_and_tune(catalog, make_hardware(),
                                 UseCaseProfile(tasks=[Task.CHAT]))
        assert tuned.decisions
        for decision in tuned.decisions:
            assert decision.reason, f"{decision.setting} has no reason"
            assert decision.value != ""

    def test_the_most_impactful_decision_is_reported_first(self, catalog):
        _, tuned = plan_and_tune(catalog, make_hardware(),
                                 UseCaseProfile(tasks=[Task.CHAT]))
        assert tuned.decisions[0].setting.startswith("gpu_layers")


class TestRuntimeSpecific:
    def test_vllm_gets_its_memory_fraction_and_tool_parser(self, catalog):
        hardware = make_hardware(vram_gb=80, ram_gb=256)
        profile = UseCaseProfile(tasks=[Task.AGENT], needs_tools=True,
                                 concurrent_users=8)
        _, tuned = plan_and_tune(catalog, hardware, profile, runtime="vllm")
        assert 0 < tuned.config.extra["gpu_memory_utilization"] < 1
        assert tuned.config.extra.get("tool_parser")

    def test_concurrent_users_become_parallel_slots(self, catalog):
        profile = UseCaseProfile(tasks=[Task.CHAT], concurrent_users=4,
                                 context_tokens=8192)
        _, tuned = plan_and_tune(catalog, make_hardware(vram_gb=48, ram_gb=128),
                                 profile)
        assert tuned.config.parallel == 4
