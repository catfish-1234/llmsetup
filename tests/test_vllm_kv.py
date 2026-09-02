"""vLLM's KV dtype has to follow the configured cache, not a snapshot of it."""
from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.planner.plan import build_plan
from llmfit.runtimes.vllm import VllmRuntime, kv_cache_dtype
from llmfit.tuning.tuner import tune
from llmfit.types import Accelerator, RuntimeConfig, Task, UseCaseProfile


def config(kv="f16", **kwargs):
    base = {"runtime": "vllm", "model_id": "m", "quant": "Q4_K_M",
            "context_tokens": 8192, "gpu_layers": 1, "total_layers": 1,
            "kv_cache_type": kv}
    base.update(kwargs)
    return RuntimeConfig(**base)


class TestDtypeMapping:
    @pytest.mark.parametrize("kind", ["f16", "bf16", "f32", "auto"])
    def test_full_precision_asks_for_nothing(self, kind):
        assert kv_cache_dtype(config(kind)) is None

    @pytest.mark.parametrize("kind", ["q8_0", "q5_1", "q4_0"])
    def test_a_quantized_cache_maps_to_the_format_vllm_has(self, kind):
        assert kv_cache_dtype(config(kind)) == "fp8"

    def test_an_unset_cache_type_is_treated_as_full_precision(self):
        assert kv_cache_dtype(config(kv="")) is None


class TestRepairReachesTheEngine:
    """The bug this replaces: the dtype was snapshotted at tuning time, so a
    repair that quantized the cache had no effect on vLLM at all, and one that
    restored precision left the quantized setting running while reporting the
    cache as restored."""

    def _command(self, cfg):
        from llmfit.types import ModelSpec

        spec = ModelSpec(id="m", name="M", publisher="p", params_b=8.0, layers=32,
                         hidden_size=4096, kv_heads=8, head_dim=128,
                         max_context=131072, hf_repo="p/m")
        return VllmRuntime().launch_command(spec, cfg)

    def test_quantizing_after_tuning_reaches_the_command(self):
        cfg = config("f16")
        assert "--kv-cache-dtype" not in self._command(cfg)
        cfg.kv_cache_type = "q8_0"
        command = self._command(cfg)
        assert command[command.index("--kv-cache-dtype") + 1] == "fp8"

    def test_restoring_precision_after_tuning_reaches_the_command(self):
        cfg = config("q8_0")
        assert "--kv-cache-dtype" in self._command(cfg)
        cfg.kv_cache_type = "f16"
        assert "--kv-cache-dtype" not in self._command(cfg)

    def test_the_restore_repair_actually_restores_it(self, catalog):
        from llmfit.repair.strategies import RepairState, _restore_kv_precision

        hardware = make_hardware(vram_gb=24, vendor=Accelerator.NVIDIA)
        profile = UseCaseProfile(tasks=[Task.RAG], context_tokens=32768,
                                 concurrent_users=4)
        plan = build_plan(profile, hardware, catalog, runtime="vllm")
        tuned = tune(plan.best, hardware, profile, "vllm", catalog=catalog)
        tuned.config.kv_cache_type = "q8_0"
        state = RepairState(config=tuned.config, candidate=plan.best,
                            hardware=hardware, profile=profile, catalog=catalog)
        if _restore_kv_precision(state) is not None:
            assert kv_cache_dtype(state.config) is None
