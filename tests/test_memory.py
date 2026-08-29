"""The memory model is the foundation everything else rests on."""

from __future__ import annotations

import pytest

from llmfit.planner.memory import (
    KV_ELEMENT_BYTES,
    cheaper_kv_quant,
    estimate_memory,
    kv_bytes_per_token,
    layers_that_fit,
    max_context_for_budget,
    weight_bytes,
)
from llmfit.types import GIB, ModelSpec


def spec(**kwargs) -> ModelSpec:
    base = {"id": "t", "name": "T", "publisher": "p", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072}
    base.update(kwargs)
    return ModelSpec(**base)


class TestWeights:
    def test_matches_published_figure_for_a_large_model(self, catalog):
        """Qwen3-235B at Q4_K_M is independently reported at ~143.4 GB.

        That figure is decimal GB (1e9), the convention for published model and
        file sizes. Our internal maths is in bytes and displays GiB, so the
        comparison has to be made in the same unit the source used.
        """
        model = catalog.require("qwen3-235b-a22b")
        decimal_gb = weight_bytes(model, catalog.bpw("Q4_K_M")) / 1e9
        assert 140 < decimal_gb < 146

    def test_moe_uses_total_not_active_parameters(self, catalog):
        """All experts are resident; only compute uses the active count."""
        moe = catalog.require("qwen3-coder-30b-a3b")
        assert moe.is_moe
        assert moe.compute_params_b < moe.params_b
        size = weight_bytes(moe, catalog.bpw("Q4_K_M")) / GIB
        # Sized by 30.5B, not by 3.3B.
        assert size > 15

    def test_scales_linearly_with_bits_per_weight(self):
        model = spec()
        assert weight_bytes(model, 8.0) == pytest.approx(
            weight_bytes(model, 4.0) * 2, rel=1e-6
        )


class TestKvCache:
    def test_uses_kv_heads_not_attention_heads(self):
        """GQA is the whole reason modern models fit; getting this wrong
        overstates the cache by the grouping factor."""
        gqa = spec(kv_heads=8)
        mha = spec(kv_heads=32)
        assert kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(gqa)

    def test_matches_the_textbook_llama_70b_figure(self):
        """2 x 80 x 8 x 128 x 131072 x 2 bytes = 42.9 GB, per published analyses.

        Exact, not approximate: this pins the formula itself. Again decimal GB,
        matching how the published figure is quoted.
        """
        llama = spec(params_b=70.6, layers=80, kv_heads=8, head_dim=128)
        total_bytes = kv_bytes_per_token(llama, "f16") * 131072
        assert total_bytes == 2 * 80 * 8 * 128 * 131072 * 2
        assert 42.5 < total_bytes / 1e9 < 43.5

    def test_quantization_reduces_cache_proportionally(self):
        model = spec()
        f16 = kv_bytes_per_token(model, "f16")
        q8 = kv_bytes_per_token(model, "q8_0")
        assert q8 == pytest.approx(f16 * KV_ELEMENT_BYTES["q8_0"] / 2.0)
        assert q8 < f16

    def test_grows_linearly_with_context(self):
        model = spec()
        at_8k = estimate_memory(model, 4.85, 8192).kv_cache
        at_32k = estimate_memory(model, 4.85, 32768).kv_cache
        assert at_32k == pytest.approx(at_8k * 4, rel=1e-6)

    def test_kv_can_exceed_weights_at_long_context(self):
        """The reason context length decides model size, not the other way round."""
        model = spec(params_b=8.0, layers=32, kv_heads=8, head_dim=128)
        estimate = estimate_memory(model, 4.85, 131072)
        assert estimate.kv_cache > estimate.weights

    def test_concurrent_slots_multiply_the_cache(self):
        model = spec()
        single = estimate_memory(model, 4.85, 8192, parallel=1)
        quad = estimate_memory(model, 4.85, 8192, parallel=4)
        assert quad.kv_cache == pytest.approx(single.kv_cache * 4)


class TestContextCeiling:
    def test_is_the_inverse_of_the_estimate(self):
        model = spec()
        budget = 12 * GIB
        ceiling = max_context_for_budget(model, 4.85, budget)
        assert estimate_memory(model, 4.85, ceiling).total <= budget
        # And one granularity step further does not fit.
        assert estimate_memory(model, 4.85, ceiling + 2048).total > budget

    def test_returns_zero_when_the_weights_alone_do_not_fit(self):
        huge = spec(params_b=70.0)
        assert max_context_for_budget(huge, 4.85, 4 * GIB) == 0

    def test_quantized_kv_buys_more_context(self):
        model = spec()
        f16 = max_context_for_budget(model, 4.85, 12 * GIB, kv_quant="f16")
        q8 = max_context_for_budget(model, 4.85, 12 * GIB, kv_quant="q8_0")
        assert q8 > f16

    def test_never_exceeds_the_model_maximum(self):
        model = spec(max_context=8192)
        assert max_context_for_budget(model, 4.85, 900 * GIB) == 8192


class TestPartialOffload:
    def test_reports_a_fraction_when_the_model_is_too_big(self):
        model = spec(params_b=32.0, layers=64)
        fitted = layers_that_fit(model, 4.85, 12 * GIB, 8192)
        assert 0 < fitted < 64

    def test_reports_all_layers_when_it_fits(self):
        model = spec(params_b=8.0, layers=32)
        assert layers_that_fit(model, 4.85, 40 * GIB, 8192) == 32

    def test_reports_a_useless_fraction_rather_than_lying(self):
        """A 1 GB budget for a 70B model fits a layer or two, and that is the
        honest answer. Rejecting such a split is the planner's job, not this
        primitive's — see test_planner.py."""
        model = spec(params_b=70.0, layers=80)
        fitted = layers_that_fit(model, 4.85, 1 * GIB, 8192)
        assert 0 <= fitted <= 2

    def test_reports_none_when_overhead_alone_exceeds_the_budget(self):
        model = spec(params_b=70.0, layers=80)
        assert layers_that_fit(model, 4.85, 64 * 1024 ** 2, 8192) == 0


def test_kv_quality_ladder_terminates():
    quant = "f16"
    seen = set()
    while quant:
        assert quant not in seen, "the KV quality ladder loops"
        seen.add(quant)
        quant = cheaper_kv_quant(quant)
    assert len(seen) >= 3
