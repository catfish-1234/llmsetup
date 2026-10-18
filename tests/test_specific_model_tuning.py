"""Native runtimes must apply the memory savings reported for a retained model."""

from dataclasses import replace

from conftest import make_hardware

from llmsetup.catalog.catalog import Catalog
from llmsetup.planner.memory import estimate_memory
from llmsetup.planner.plan import build_adaptive_plan
from llmsetup.runtimes.vllm import VllmRuntime
from llmsetup.tuning.tuner import SAFETY_MARGIN, tune
from llmsetup.types import GIB, Accelerator, ModelSpec, UseCaseProfile


def native_plan(runtime, params, hardware, context=8192):
    model = ModelSpec(
        id="retained", name="Retained", publisher="test", params_b=params,
        max_context=32768, tasks=["chat"], hf_repo="test/retained",
        mlx_repo="test/retained-4bit" if runtime == "mlx" else None,
    )
    catalog = Catalog([model], {})
    profile = UseCaseProfile(requested_model=model.id, context_tokens=context)
    plan = build_adaptive_plan(profile, hardware, catalog, runtime=runtime)
    return plan, catalog


def test_mlx_preserves_supported_cache_precision_under_memory_pressure():
    hardware = make_hardware(vram_gb=16, ram_gb=16, vendor=Accelerator.APPLE,
                             unified=True, os_name="darwin")
    plan, catalog = native_plan("mlx", 16, hardware)
    assert plan.best.estimate.kv_quant == "f16"
    # This fit needs tuning: keeping the original prompt batch consumes the
    # extra margin, which previously made the tuner invent an 8-bit MLX cache.
    assert plan.best.estimate.total > hardware.memory_budget * (1 - SAFETY_MARGIN)
    tuned = tune(plan.best, hardware, plan.profile, "mlx", catalog=catalog)
    assert tuned.config.model_id == "retained"
    assert tuned.config.quant == "MLX4"
    assert tuned.config.kv_cache_type == "f16"
    assert tuned.config.full_gpu_offload
    assert 512 <= tuned.config.context_tokens <= plan.profile.context_tokens
    assert tuned.estimate.total <= hardware.memory_budget


def test_mlx_uses_real_available_memory_when_extra_margin_is_unavailable():
    hardware = make_hardware(vram_gb=16, ram_gb=16, vendor=Accelerator.APPLE,
                             unified=True, os_name="darwin")
    plan, catalog = native_plan("mlx", 18, hardware)
    tuned = tune(plan.best, hardware, plan.profile, "mlx", catalog=catalog)
    assert tuned.config.quant == "MLX4"
    assert tuned.config.kv_cache_type == "f16"
    assert tuned.config.full_gpu_offload
    assert tuned.estimate.total <= hardware.memory_budget
    assert tuned.estimate.total > hardware.memory_budget * (1 - SAFETY_MARGIN)


def test_vllm_tuning_budgets_the_fp8_cache_it_actually_launches():
    hardware = make_hardware()
    plan, catalog = native_plan("vllm", 4, hardware)
    candidate = plan.best
    candidate.estimate = replace(candidate.estimate, kv_quant="q4_0")
    tuned = tune(candidate, hardware, plan.profile, "vllm", catalog=catalog)
    command = VllmRuntime().launch_command(candidate.model, tuned.config)
    assert tuned.config.model_id == "retained"
    assert tuned.config.quant == "F16"
    assert tuned.config.kv_cache_type == "q8_0"
    assert command[command.index("--kv-cache-dtype") + 1] == "fp8"
    actual = estimate_memory(
        candidate.model, 16, tuned.config.context_tokens, quant="F16",
        kv_quant="q8_0", runtime="vllm", ubatch=tuned.config.ubatch_size,
        parallel=tuned.config.parallel,
    )
    assert tuned.estimate.total == actual.total


def test_vllm_reserves_available_vram_and_keeps_every_layer_on_gpu():
    hardware = make_hardware(vram_gb=24)
    hardware.gpus[0].vram_free_bytes = 8 * GIB
    plan, catalog = native_plan("vllm", 2, hardware)
    tuned = tune(plan.best, hardware, plan.profile, "vllm", catalog=catalog)
    allocation = tuned.config.extra["gpu_memory_utilization"] * hardware.total_vram_bytes
    assert allocation <= hardware.memory_budget
    assert tuned.estimate.total <= allocation
    assert tuned.config.full_gpu_offload
    assert tuned.config.quant == "F16"


def test_native_tuning_keeps_an_already_short_requested_context():
    hardware = make_hardware(vram_gb=6)
    plan, catalog = native_plan("vllm", 2, hardware, context=512)
    tuned = tune(plan.best, hardware, plan.profile, "vllm", catalog=catalog)
    assert tuned.config.context_tokens == 512
    assert tuned.config.full_gpu_offload
    assert tuned.estimate.total <= hardware.memory_budget * (1 - SAFETY_MARGIN)
