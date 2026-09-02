"""Models where most layers do not keep a KV cache at all.

A hybrid interleaves recurrent layers, which carry a fixed-size state, with
full-attention layers that keep a growing cache. llama.cpp calls the ratio
`full_attention_interval`. Counting every layer as a cache layer overstates the
memory by that interval, several times over, and rules out models that fit
comfortably.
"""

from __future__ import annotations

import pytest

from llmfit.gguf import GgufInfo, apply_to_spec
from llmfit.planner.memory import (
    estimate_memory,
    kv_cache_bytes,
    max_context_for_budget,
)
from llmfit.types import GIB, MIB, ModelSpec


def spec(**kwargs) -> ModelSpec:
    base = {"id": "t", "name": "T", "publisher": "p", "params_b": 35.0,
            "layers": 48, "hidden_size": 2560, "kv_heads": 4, "head_dim": 128,
            "max_context": 262144}
    base.update(kwargs)
    return ModelSpec(**base)


def hybrid(**kwargs) -> ModelSpec:
    return spec(full_attention_interval=4, **kwargs)


class TestLayerAccounting:
    def test_an_interval_of_one_is_not_a_hybrid(self):
        """Every layer attends, which is what a plain transformer is."""
        model = spec(full_attention_interval=1)
        assert not model.is_hybrid
        assert model.attention_layers == model.layers
        assert model.recurrent_layers == 0

    def test_a_quarter_of_the_layers_attend(self):
        model = hybrid()
        assert model.attention_layers == 12
        assert model.recurrent_layers == 36
        assert model.attention_layers + model.recurrent_layers == model.layers

    def test_never_claims_zero_attention_layers(self):
        """An interval larger than the layer count would divide to zero and make
        the cache look free."""
        assert spec(layers=3, full_attention_interval=8).attention_layers >= 1

    def test_windowing_applies_within_the_attention_layers(self):
        """The two are independent: a hybrid can also window the attention
        layers it does have, and the window applies to those, not to all."""
        model = hybrid(sliding_window=1024, swa_pattern=4)
        assert model.global_layers == 3
        assert model.local_layers == 9
        assert model.global_layers + model.local_layers == model.attention_layers


class TestKvMaths:
    def test_the_cache_shrinks_by_the_interval(self):
        dense, model = spec(), hybrid()
        assert kv_cache_bytes(dense, 131072) / kv_cache_bytes(model, 131072) == (
            pytest.approx(4.0, rel=0.01))

    def test_recurrent_state_is_charged_once_not_per_token(self):
        """That it does not grow with the context is the entire point of it."""
        model = hybrid(recurrent_state_bytes=2 * MIB)
        bare = hybrid()
        for context in (4096, 262144):
            extra = kv_cache_bytes(model, context) - kv_cache_bytes(bare, context)
            assert extra == model.recurrent_layers * 2 * MIB

    def test_a_plain_model_is_unaffected(self):
        model = spec()
        assert kv_cache_bytes(model, 32768) == (
            2 * model.layers * model.kv_heads * model.effective_head_dim
            * 2 * 32768)

    def test_quantizing_the_cache_still_scales_the_attention_part(self):
        model = hybrid()
        assert kv_cache_bytes(model, 65536, "q8_0") < kv_cache_bytes(model, 65536)


class TestContextInversion:
    @pytest.mark.parametrize("model", [spec(), hybrid(), hybrid(recurrent_state_bytes=2 * MIB)])
    @pytest.mark.parametrize("budget_gb", [24, 48, 96])
    def test_what_it_says_fits_actually_fits(self, model, budget_gb):
        budget = int(budget_gb * GIB)
        fitted = max_context_for_budget(model, 4.85, budget)
        if fitted == 0:
            return
        assert estimate_memory(model, 4.85, fitted).total <= budget

    def test_one_more_step_would_not_fit(self, ):
        """Tight as well as safe. A needlessly short context is a real cost on a
        model whose whole selling point is long context."""
        model = hybrid()
        budget = int(32 * GIB)
        fitted = max_context_for_budget(model, 4.85, budget, granularity=1024)
        if 0 < fitted < model.max_context:
            assert estimate_memory(model, 4.85, fitted + 1024).total > budget

    def test_a_hybrid_is_told_it_fits_far_more(self):
        model = hybrid(max_context=1_048_576)
        dense = spec(max_context=1_048_576)
        budget = int(32 * GIB)
        hybrid_fit = max_context_for_budget(model, 4.85, budget)
        dense_fit = max_context_for_budget(dense, 4.85, budget)
        assert 0 < hybrid_fit < model.max_context, "the comparison must not saturate"
        assert hybrid_fit > 3 * dense_fit

    def test_a_state_too_large_for_the_budget_fits_nothing(self):
        """Rather than a negative context, or a division that silently returns
        something plausible."""
        model = hybrid(recurrent_state_bytes=8 * GIB)
        assert max_context_for_budget(model, 4.85, int(4 * GIB)) == 0


class TestReadingItFromTheFile:
    def test_the_interval_is_taken_from_the_header(self):
        updated, corrections = apply_to_spec(
            spec(), GgufInfo(architecture="qwen35", full_attention_interval=4))
        assert updated.full_attention_interval == 4
        assert any("hybrid attention" in c for c in corrections)

    def test_state_size_is_derived_only_from_stated_dimensions(self):
        info = GgufInfo(architecture="qwen35", full_attention_interval=4,
                        ssm_inner_size=4096, ssm_state_size=128)
        assert info.recurrent_state_bytes == 4096 * 128 * 4
        updated, _ = apply_to_spec(spec(), info)
        assert updated.recurrent_state_bytes == 4096 * 128 * 4

    def test_a_missing_dimension_yields_no_guess(self):
        """Zero rather than an invented figure. The state is small next to the
        weights and the overhead term already errs high, so a missing number
        costs far less than a wrong one."""
        assert GgufInfo(architecture="x", ssm_inner_size=4096).recurrent_state_bytes == 0
        assert GgufInfo(architecture="x").recurrent_state_bytes == 0

    def test_an_interval_of_one_is_not_reported_as_hybrid(self):
        assert not GgufInfo(architecture="x", full_attention_interval=1).is_hybrid
        assert not GgufInfo(architecture="x").is_hybrid


class TestTheCatalogEntry:
    def test_the_known_hybrid_is_marked_as_one(self, catalog):
        model = catalog.require("qwen3.6-35b-a3b")
        assert model.is_hybrid
        assert model.attention_layers < model.layers

    def test_its_long_context_is_now_affordable(self, catalog):
        """Before this it was charged for every layer, which put a model that
        fits a 48 GB card at 262K well outside it."""
        model = catalog.require("qwen3.6-35b-a3b")
        estimate = estimate_memory(model, catalog.bpw("Q4_K_M"), 262144, quant="Q4_K_M")
        assert estimate.kv_cache < 8 * GIB

    def test_plain_models_in_the_catalog_are_untouched(self, catalog):
        for model_id in ("qwen3-8b", "llama-3.1-8b", "mistral-small-3.2-24b"):
            assert not catalog.require(model_id).is_hybrid
