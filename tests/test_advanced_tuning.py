"""Deep configuration: placement, drafting, thinking budget, samplers, RoPE.

These are the settings that separate "it runs" from "it runs well", and several
change what is runnable at all — MoE expert offload most of all.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.planner import moe
from llmfit.planner.plan import build_plan
from llmfit.runtimes.registry import get_runtime
from llmfit.tuning.advanced import (
    plan_reasoning,
    plan_rope,
    plan_sampler,
    plan_serving,
)
from llmfit.tuning.profiles import ReasoningEffort, SamplerChain
from llmfit.tuning.tuner import tune
from llmfit.types import GIB, Task, UseCaseProfile


def tuned(catalog, model_id, hardware, profile, runtime="llamacpp"):
    plan = build_plan(profile, hardware, catalog.restrict_to(model_id), runtime=runtime)
    return tune(plan.best, hardware, profile, runtime, catalog=catalog)


# --------------------------------------------------------------------------- #
# MoE expert offload
# --------------------------------------------------------------------------- #


class TestMoeAccounting:
    def test_experts_dominate_an_moe_model(self, catalog):
        """The whole technique rests on experts being most of the weights."""
        for model_id in ("qwen3-coder-30b-a3b", "qwen3-235b-a22b", "gpt-oss-120b"):
            split = moe.breakdown(catalog.require(model_id))
            assert split is not None
            assert split.expert_fraction > 0.9, model_id

    def test_a_dense_model_has_no_split(self, catalog):
        assert moe.breakdown(catalog.require("qwen3-32b")) is None
        assert moe.expert_bytes(catalog.require("qwen3-32b"), 4.85) == 0

    def test_offloading_frees_memory_proportionally(self, catalog):
        spec = catalog.require("gpt-oss-120b")
        full = moe.resident_bytes_with_offload(spec, 4.85, 0)
        half = moe.resident_bytes_with_offload(spec, 4.85, spec.layers // 2)
        allof = moe.resident_bytes_with_offload(spec, 4.85, spec.layers)
        assert allof < half < full
        # With every expert offloaded, only the dense part remains.
        assert allof < full * 0.15

    def test_offload_count_is_bounded_by_the_layer_count(self, catalog):
        spec = catalog.require("qwen3-coder-30b-a3b")
        assert moe.layers_to_offload(spec, 4.85, 10 ** 15) == spec.layers

    def test_nothing_is_offloaded_when_it_already_fits(self, catalog):
        spec = catalog.require("qwen3-coder-30b-a3b")
        assert moe.layers_to_offload(spec, 4.85, 0) == 0
        assert moe.layers_to_offload(spec, 4.85, -1) == 0

    def test_the_speed_penalty_is_modest_because_few_experts_fire(self, catalog):
        """The point of the technique: offloading experts costs far less than
        offloading whole layers, because only active parameters cross."""
        spec = catalog.require("gpt-oss-120b")
        retained = moe.throughput_penalty(spec, spec.layers)
        assert 0.05 < retained < 0.5

    def test_a_dense_model_is_never_penalised_by_this_path(self, catalog):
        spec = catalog.require("qwen3-32b")
        assert moe.throughput_penalty(spec, 40) == 1.0


class TestOffloadPlanning:
    def test_a_huge_moe_becomes_runnable_on_a_small_card(self, catalog):
        """gpt-oss-120b is ~66 GB at Q4 and cannot fit a 12 GB card as weights.
        Offloading its experts is what makes it runnable at all."""
        hardware = make_hardware(vram_gb=12, ram_gb=64)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        result = tuned(catalog, "gpt-oss-120b", hardware, profile)

        offload = result.config.advanced.offload
        assert offload.moe_offload_enabled
        assert 0 < offload.n_cpu_moe <= catalog.require("gpt-oss-120b").layers
        assert "system RAM" in offload.rationale

    def test_every_layer_stays_on_the_gpu_when_experts_move(self, catalog):
        """Expert offload is not layer offload: attention and the KV cache stay
        resident, which is where the speed comes from."""
        hardware = make_hardware(vram_gb=12, ram_gb=64)
        result = tuned(catalog, "gpt-oss-120b", hardware,
                       UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        offload = result.config.advanced.offload
        assert offload.gpu_layers == offload.total_layers

    def test_a_model_that_fits_gets_no_offload(self, catalog):
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        result = tuned(catalog, "qwen3-coder-30b-a3b", hardware,
                       UseCaseProfile(tasks=[Task.CODING], context_tokens=8192))
        assert not result.config.advanced.offload.moe_offload_enabled

    def test_a_dense_model_never_gets_expert_offload(self, catalog):
        hardware = make_hardware(vram_gb=8, ram_gb=32)
        result = tuned(catalog, "qwen3-32b", hardware,
                       UseCaseProfile(tasks=[Task.CHAT], context_tokens=4096))
        assert result.config.advanced.offload.n_cpu_moe == 0

    def test_multiple_gpus_get_a_proportional_split(self, catalog):
        hardware = make_hardware(vram_gb=24, ram_gb=128)
        from llmfit.types import GPU, Accelerator

        hardware.gpus.append(
            GPU(1, "NVIDIA RTX 4090", Accelerator.NVIDIA, 24 * GIB, compute_capability="8.9")
        )
        result = tuned(catalog, "qwen3-14b", hardware,
                       UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        offload = result.config.advanced.offload
        assert len(offload.tensor_split) == 2
        assert sum(offload.tensor_split) == pytest.approx(1.0, abs=0.01)


# --------------------------------------------------------------------------- #
# speculative decoding
# --------------------------------------------------------------------------- #


class TestSpeculativeDecoding:
    def test_pairs_a_draft_from_the_same_vocabulary_family(self, catalog):
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        result = tuned(catalog, "qwen3-32b", hardware,
                       UseCaseProfile(tasks=[Task.CODING], context_tokens=8192))
        plan = result.config.advanced.speculative
        assert plan.enabled
        draft = catalog.require(plan.draft_model_id)
        assert draft.family == catalog.require("qwen3-32b").family
        assert draft.params_b < 32.8 / 5

    @pytest.mark.parametrize("model_id", [
        "gemma-3-27b", "mistral-small-3.2-24b", "phi-4-14b", "qwen3-32b",
        "llama-3.3-70b", "deepseek-r1-distill-qwen-32b",
    ])
    def test_never_pairs_across_vocabularies(self, catalog, model_id):
        """A draft that tokenizes differently cannot have its guesses verified,
        so a cross-family pairing is worse than none. Tested against the pairing
        function directly, since the surrounding plan may decline for unrelated
        reasons (task, memory) and hide a bad pairing."""
        from llmfit.tuning.advanced import _pick_draft

        target = catalog.require(model_id)
        draft, ratio = _pick_draft(target, catalog)
        if draft is not None:
            assert draft.family == target.family
            assert draft.id != target.id
            assert ratio >= 5.0

    def test_a_model_with_no_small_sibling_gets_no_draft(self, catalog):
        """Mistral has no draft-capable model in its family, so the honest
        answer is none rather than a mismatched one."""
        from llmfit.tuning.advanced import _pick_draft

        draft, _ = _pick_draft(catalog.require("mistral-small-3.2-24b"), catalog)
        assert draft is None

    def test_unverified_model_generations_do_not_claim_a_shared_vocabulary(self, catalog):
        """Asserting tokenizer compatibility for a model whose architecture we
        could not even verify would be a guess, and a wrong one emits garbage."""
        for model_id in ("qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3-coder-next"):
            spec = catalog.get(model_id)
            if spec is None:
                continue
            assert spec.family != catalog.require("qwen3-8b").family, model_id

    def test_declines_when_there_is_no_memory_for_the_draft(self, catalog):
        """Every byte the draft holds is a byte not available for context."""
        hardware = make_hardware(vram_gb=24, ram_gb=64)
        result = tuned(catalog, "qwen3-32b", hardware,
                       UseCaseProfile(tasks=[Task.CODING], context_tokens=32768))
        plan = result.config.advanced.speculative
        assert not plan.enabled
        assert "spare" in plan.rationale

    def test_declines_for_open_ended_tasks(self, catalog):
        """Speculation is a bet on predictable output; creative writing is not."""
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        result = tuned(catalog, "qwen3-14b", hardware,
                       UseCaseProfile(tasks=[Task.CREATIVE], context_tokens=8192))
        assert not result.config.advanced.speculative.enabled

    def test_declines_for_a_model_that_is_already_draft_speed(self, catalog):
        """A 3B-active MoE has little left to win from a draft."""
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        result = tuned(catalog, "qwen3-coder-30b-a3b", hardware,
                       UseCaseProfile(tasks=[Task.CODING], context_tokens=8192))
        plan = result.config.advanced.speculative
        assert not plan.enabled
        assert "activates only" in plan.rationale


# --------------------------------------------------------------------------- #
# reasoning budget
# --------------------------------------------------------------------------- #


class TestReasoningBudget:
    def test_thinking_is_disabled_for_pure_transformations(self, catalog):
        """The largest latency lever available: extraction is a transformation,
        not a deduction."""
        spec = catalog.require("qwen3-8b")
        assert spec.is_reasoning
        plan = plan_reasoning(spec, UseCaseProfile(tasks=[Task.EXTRACT]))
        assert plan.effort is ReasoningEffort.OFF
        assert not plan.enabled

    def test_thinking_stays_on_for_agents(self, catalog):
        plan = plan_reasoning(catalog.require("qwen3-8b"),
                              UseCaseProfile(tasks=[Task.AGENT]))
        assert plan.enabled
        assert plan.effort is ReasoningEffort.MEDIUM

    def test_an_explicit_request_for_reasoning_wins(self, catalog):
        plan = plan_reasoning(catalog.require("qwen3-8b"),
                              UseCaseProfile(tasks=[Task.EXTRACT], needs_reasoning=True))
        assert plan.effort is ReasoningEffort.HIGH

    def test_a_non_reasoning_model_reports_no_thinking_mode(self, catalog):
        spec = catalog.require("mistral-small-3.2-24b")
        plan = plan_reasoning(spec, UseCaseProfile(tasks=[Task.CHAT]))
        assert not plan.supported and not plan.enabled

    def test_the_thinking_block_is_stripped_for_agents(self, catalog):
        """A thinking block in the response confuses tool-call extraction."""
        plan = plan_reasoning(catalog.require("qwen3-8b"),
                              UseCaseProfile(tasks=[Task.AGENT]))
        assert plan.preserve is False


# --------------------------------------------------------------------------- #
# sampler chain
# --------------------------------------------------------------------------- #


class TestSamplerChain:
    def test_creative_writing_gets_dry(self, catalog):
        chain = plan_sampler(catalog.require("mistral-small-3.2-24b"),
                             UseCaseProfile(tasks=[Task.CREATIVE]))
        assert chain.dry_enabled

    def test_code_gets_no_repetition_penalties(self, catalog):
        """Repeated tokens in code are usually correct; penalising them
        corrupts valid output."""
        chain = plan_sampler(catalog.require("qwen3-coder-30b-a3b"),
                             UseCaseProfile(tasks=[Task.CODING]))
        assert not chain.dry_enabled
        assert not chain.xtc_enabled
        assert chain.penalty_last_n == 0

    def test_json_output_disables_the_creative_samplers(self, catalog):
        chain = plan_sampler(catalog.require("mistral-small-3.2-24b"),
                             UseCaseProfile(tasks=[Task.CREATIVE], needs_json=True))
        assert not chain.dry_enabled and not chain.xtc_enabled

    def test_reasoning_models_avoid_aggressive_truncation(self, catalog):
        chain = plan_sampler(catalog.require("qwen3-8b"),
                             UseCaseProfile(tasks=[Task.CREATIVE]))
        assert not chain.xtc_enabled
        assert chain.top_n_sigma >= 0

    def test_disabled_samplers_leave_the_chain(self, catalog):
        chain = SamplerChain()
        order = chain.normalised_order()
        assert "dry" not in order and "xtc" not in order

    def test_mirostat_is_forced_last(self):
        """Mirostat replaces the truncation step and must see the final
        distribution, so temperature cannot follow it."""
        chain = SamplerChain(mirostat=2)
        assert chain.normalised_order()[-1] == "temperature"


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #


class TestRopeScaling:
    def test_no_scaling_within_the_native_window(self, catalog):
        plan = plan_rope(catalog.require("qwen3-8b"), 32768)
        assert not plan.enabled

    def test_yarn_is_planned_beyond_the_native_window(self, catalog):
        spec = catalog.require("phi-4-14b")          # 16K native
        plan = plan_rope(spec, 65536)
        assert plan.enabled
        assert plan.scaling_type == "yarn"
        assert plan.factor == pytest.approx(4.0)
        assert plan.original_context == 16384

    def test_the_quality_cost_is_stated(self, catalog):
        plan = plan_rope(catalog.require("phi-4-14b"), 65536)
        assert "degrades" in plan.rationale


# --------------------------------------------------------------------------- #
# emitted flags
# --------------------------------------------------------------------------- #


class TestEmittedFlags:
    def _command(self, catalog, model_id, hardware, profile, runtime="llamacpp"):
        result = tuned(catalog, model_id, hardware, profile, runtime)
        config = result.config
        config.model_path = "/models/m.gguf"
        if config.advanced.speculative.enabled:
            config.advanced.speculative.draft_path = "/models/draft.gguf"
        spec = catalog.require(model_id)
        return get_runtime(runtime).launch_command(spec, config), config

    def test_moe_offload_reaches_the_command_line(self, catalog):
        command, config = self._command(
            catalog, "gpt-oss-120b", make_hardware(vram_gb=12, ram_gb=64),
            UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192))
        assert "--n-cpu-moe" in command
        index = command.index("--n-cpu-moe")
        assert int(command[index + 1]) == config.advanced.offload.n_cpu_moe

    def test_the_sampler_chain_is_emitted(self, catalog):
        command, config = self._command(
            catalog, "mistral-small-3.2-24b", make_hardware(vram_gb=24),
            UseCaseProfile(tasks=[Task.CREATIVE], context_tokens=8192))
        assert "--samplers" in command
        assert "--dry-multiplier" in command
        emitted = command[command.index("--samplers") + 1]
        assert emitted == ";".join(config.advanced.sampler.normalised_order())

    def test_the_reasoning_flag_is_emitted_for_hybrid_models(self, catalog):
        command, _ = self._command(
            catalog, "qwen3-8b", make_hardware(vram_gb=24),
            UseCaseProfile(tasks=[Task.EXTRACT], context_tokens=8192))
        assert "--reasoning" in command
        assert command[command.index("--reasoning") + 1] == "off"

    def test_speculative_flags_are_emitted_when_planned(self, catalog):
        command, _ = self._command(
            catalog, "qwen3-32b", make_hardware(vram_gb=48, ram_gb=128),
            UseCaseProfile(tasks=[Task.CODING], context_tokens=8192))
        assert "--model-draft" in command
        assert "--draft-max" in command

    def test_nothing_exotic_appears_for_a_plain_setup(self, catalog):
        """A simple setup must still produce a simple command line."""
        command, _ = self._command(
            catalog, "qwen3-4b", make_hardware(vram_gb=24),
            UseCaseProfile(tasks=[Task.CHAT], context_tokens=4096))
        for flag in ("--n-cpu-moe", "--model-draft", "--xtc-probability",
                     "--mirostat", "--rope-scaling"):
            assert flag not in command

    def test_vllm_gets_prefix_caching(self, catalog):
        command, _ = self._command(
            catalog, "qwen3-14b", make_hardware(vram_gb=48, ram_gb=128),
            UseCaseProfile(tasks=[Task.AGENT], needs_tools=True,
                           concurrent_users=4, context_tokens=8192),
            runtime="vllm")
        assert "--enable-prefix-caching" in command

    def test_vllm_never_combines_chunked_prefill_with_draft_speculation(self, catalog):
        """vLLM rejects that combination outright."""
        command, _ = self._command(
            catalog, "qwen3-32b", make_hardware(vram_gb=80, ram_gb=256),
            UseCaseProfile(tasks=[Task.CODING], concurrent_users=4,
                           context_tokens=8192),
            runtime="vllm")
        assert not ("--speculative-config" in command
                    and "--enable-chunked-prefill" in command)


class TestServing:
    def test_agent_workloads_get_generous_prefix_caching(self):
        plan = plan_serving(UseCaseProfile(tasks=[Task.AGENT]), "llamacpp",
                            make_hardware())
        assert plan.prefix_caching and plan.cache_reuse_tokens >= 512

    def test_concurrency_enables_continuous_batching(self):
        plan = plan_serving(UseCaseProfile(tasks=[Task.CHAT], concurrent_users=8),
                            "llamacpp", make_hardware())
        assert plan.continuous_batching


class TestLlamaSwap:
    """One endpoint serving several models on demand — the right shape when
    VRAM fits one model at a time but the user wants a choice."""

    def _entries(self, catalog):
        from llmfit.runtimes.registry import get_runtime

        entries = []
        hardware = make_hardware(vram_gb=24, ram_gb=64)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        for model_id in ("qwen3-8b", "qwen3-14b"):
            result = tuned(catalog, model_id, hardware, profile)
            result.config.model_path = f"/models/{model_id}.gguf"
            spec = catalog.require(model_id)
            command = get_runtime("llamacpp").launch_command(spec, result.config)
            entries.append((spec, result.config, command))
        return entries

    def test_builds_a_config_with_every_model(self, catalog):
        from llmfit.interop.llama_swap import build_config

        document = build_config(self._entries(catalog))
        assert set(document["models"]) == {"qwen3-8b", "qwen3-14b"}
        for entry in document["models"].values():
            assert entry["cmd"].startswith("llama-server")
            assert entry["ttl"] > 0

    def test_models_are_exclusive_so_they_swap(self, catalog):
        """They do not all fit at once; that is the reason to run llama-swap."""
        from llmfit.interop.llama_swap import build_config

        document = build_config(self._entries(catalog))
        assert document["groups"]["default"]["exclusive"] is True
        assert document["groups"]["default"]["swap"] is True

    def test_the_config_is_valid_yaml_on_disk(self, catalog, tmp_path):
        import yaml

        from llmfit.interop.llama_swap import write_config

        record = write_config(self._entries(catalog), path=tmp_path / "swap.yaml")
        assert record["ok"]
        parsed = yaml.safe_load((tmp_path / "swap.yaml").read_text())
        assert len(parsed["models"]) == 2

    def test_each_command_carries_the_tuning(self, catalog):
        """The point of generating this is that llmfit already knows the tuned
        command; a config with untuned commands would be worthless."""
        from llmfit.interop.llama_swap import build_config

        document = build_config(self._entries(catalog))
        for entry in document["models"].values():
            assert "--ctx-size" in entry["cmd"]
            assert "--n-gpu-layers" in entry["cmd"]


class TestReportSurfacesDeepSettings:
    def test_the_markdown_report_explains_each_non_default(self, catalog):
        from llmfit.report.render import render_markdown
        from llmfit.types import SetupResult

        hardware = make_hardware(vram_gb=12, ram_gb=64)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog.restrict_to("gpt-oss-120b"))
        result = tuned(catalog, "gpt-oss-120b", hardware, profile)

        markdown = render_markdown(SetupResult(
            profile=profile, hardware=hardware, plan=plan, config=result.config))
        assert "## Deep configuration" in markdown
        assert "system RAM" in markdown

    def test_a_plain_setup_has_no_wall_of_knobs(self, catalog):
        """Only non-default decisions are reported."""
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        result = tuned(catalog, "qwen3-4b", hardware,
                       UseCaseProfile(tasks=[Task.CHAT], context_tokens=4096))
        explained = " ".join(result.config.advanced.explain())
        assert "n-cpu-moe" not in explained
        assert "YaRN" not in explained
