"""Where the KV cache stops being a free saving.

q8_0 measures as effectively lossless against an f16 reference. Below it the
trade stops being free and stops being predictable: how much quality goes
depends on the model, and past long context the cache is *slower* too, because
dequantizing inside attention costs more than the memory traffic it saves. So
the ordinary paths stop at the floor, and going below it has to be a decision
the user can see.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmfit.planner.memory import (
    KV_LOSSY_CONTEXT_LIMIT,
    KV_QUALITY_ORDER,
    KV_SAFE_FLOOR,
    cheaper_kv_quant,
    kv_is_lossy,
    kv_lossy_allowed,
)
from llmfit.planner.plan import build_plan
from llmfit.repair.strategies import RepairState, _quantize_kv, _quantize_kv_lossy
from llmfit.tuning.tuner import tune
from llmfit.types import GIB, Priority, Task, UseCaseProfile


def state(catalog, hardware, **overrides):
    profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.QUALITY)
    plan = build_plan(profile, hardware, catalog)
    tuned = tune(plan.best, hardware, profile, "llamacpp")
    for key, value in overrides.items():
        setattr(tuned.config, key, value)
    return RepairState(config=tuned.config, candidate=plan.best, hardware=hardware,
                       profile=profile, catalog=catalog)


class TestTheFloorItself:
    def test_the_floor_is_a_real_rung_on_the_ladder(self):
        assert KV_SAFE_FLOOR in KV_QUALITY_ORDER
        assert KV_QUALITY_ORDER.index(KV_SAFE_FLOOR) > 0, "f16 is not a saving"

    def test_only_dtypes_below_the_floor_count_as_lossy(self):
        assert not kv_is_lossy("f16")
        assert not kv_is_lossy(KV_SAFE_FLOOR)
        assert kv_is_lossy("q4_0")
        assert kv_is_lossy("q5_1")

    def test_an_unknown_dtype_is_not_called_lossy(self):
        """Better to say nothing than to warn about a dtype we do not know."""
        assert not kv_is_lossy("some_future_type")

    def test_the_ordinary_ladder_stops_at_the_floor(self):
        assert cheaper_kv_quant("f16") == KV_SAFE_FLOOR
        assert cheaper_kv_quant(KV_SAFE_FLOOR) is None

    def test_the_deliberate_ladder_goes_further(self):
        assert cheaper_kv_quant(KV_SAFE_FLOOR, allow_lossy=True) == "q5_1"
        assert cheaper_kv_quant("q4_0", allow_lossy=True) is None

    def test_a_lossy_cache_is_pointless_at_long_context(self):
        assert kv_lossy_allowed(8192)
        assert kv_lossy_allowed(KV_LOSSY_CONTEXT_LIMIT)
        assert not kv_lossy_allowed(KV_LOSSY_CONTEXT_LIMIT + 1)
        assert not kv_lossy_allowed(131072)


class TestTuner:
    @pytest.mark.parametrize("vram_gb,ram_gb", [(8, 16), (12, 32), (24, 64), (6, 8)])
    def test_never_chooses_a_lossy_cache_on_its_own(self, catalog, vram_gb, ram_gb):
        """Across tight machines the tuner may quantize the cache, shorten the
        context or spill layers -- but it must never quietly pick a dtype that
        changes answers."""
        hardware = make_hardware(vram_gb=vram_gb, ram_gb=ram_gb)
        for task in (Task.CHAT, Task.CODING, Task.RAG, Task.AGENT):
            profile = UseCaseProfile(tasks=[task], priority=Priority.QUALITY)
            plan = build_plan(profile, hardware, catalog)
            config = tune(plan.best, hardware, profile, "llamacpp").config
            assert not kv_is_lossy(config.kv_cache_type), (
                f"{task} on {vram_gb}GB picked {config.kv_cache_type}")

    def test_quantizing_the_cache_turns_flash_attention_on(self, catalog):
        """The backends that take a quantized cache require it."""
        hardware = make_hardware(vram_gb=8, ram_gb=16)
        profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.QUALITY)
        plan = build_plan(profile, hardware, catalog)
        config = tune(plan.best, hardware, profile, "llamacpp").config
        if config.kv_cache_type != "f16":
            assert config.flash_attention


class TestRepair:
    def test_the_cheap_strategy_stops_at_the_floor(self, catalog):
        hardware = make_hardware(vram_gb=24)
        st = state(catalog, hardware, kv_cache_type=KV_SAFE_FLOOR)
        assert _quantize_kv(st) is None
        assert st.config.kv_cache_type == KV_SAFE_FLOOR

    def test_the_cheap_strategy_reaches_the_floor_from_f16(self, catalog):
        st = state(catalog, make_hardware(vram_gb=24), kv_cache_type="f16")
        outcome = _quantize_kv(st)
        assert outcome is not None and outcome.cost == "low"
        assert st.config.kv_cache_type == KV_SAFE_FLOOR
        assert st.config.flash_attention

    def test_the_last_resort_is_priced_as_one(self, catalog):
        st = state(catalog, make_hardware(vram_gb=24),
                   kv_cache_type=KV_SAFE_FLOOR, context_tokens=8192)
        outcome = _quantize_kv_lossy(st)
        assert outcome is not None
        assert outcome.cost == "high"
        assert kv_is_lossy(st.config.kv_cache_type)

    def test_the_last_resort_says_what_it_cost(self, catalog):
        st = state(catalog, make_hardware(vram_gb=24),
                   kv_cache_type=KV_SAFE_FLOOR, context_tokens=8192)
        text = _quantize_kv_lossy(st).description.lower()
        assert "answers to change" in text
        assert "q8_0" in text, "the user needs to know how to undo it"

    def test_declines_at_long_context_where_it_is_slower_too(self, catalog):
        st = state(catalog, make_hardware(vram_gb=24),
                   kv_cache_type=KV_SAFE_FLOOR, context_tokens=131072)
        assert _quantize_kv_lossy(st) is None
        assert st.config.kv_cache_type == KV_SAFE_FLOOR

    def test_does_not_fire_when_the_cache_is_already_at_the_bottom(self, catalog):
        st = state(catalog, make_hardware(vram_gb=24),
                   kv_cache_type="q4_0", context_tokens=8192)
        assert _quantize_kv_lossy(st) is None

    def test_the_two_strategies_are_ordered_by_what_they_cost(self):
        from llmfit.repair.strategies import COST_ORDER, STRATEGIES

        named = {s.name: s for s in STRATEGIES}
        assert (COST_ORDER[named["quantize_kv_cache"].cost]
                < COST_ORDER[named["quantize_kv_cache_lossy"].cost])

    def test_the_loop_exhausts_the_cheap_cache_before_the_lossy_one(self, catalog):
        """Ordering is the whole point: a free saving must be spent first."""
        from llmfit.repair.strategies import COST_ORDER, strategies_for
        from llmfit.verify import signatures as sig

        names = [s.name for s in sorted(strategies_for(sig.OUT_OF_MEMORY),
                                        key=lambda s: COST_ORDER[s.cost])]
        assert names.index("quantize_kv_cache") < names.index("quantize_kv_cache_lossy")
        assert names.index("reduce_context") < names.index("quantize_kv_cache_lossy")


class TestBudgetArithmeticStillHolds:
    def test_a_lossy_cache_really_is_smaller(self):
        """The floor exists because the saving is real -- not because it isn't."""
        from llmfit.planner.memory import kv_bytes_per_token
        from llmfit.types import ModelSpec

        model = ModelSpec(id="t", name="T", publisher="p", params_b=8.0, layers=32,
                          hidden_size=4096, kv_heads=8, head_dim=128,
                          max_context=131072)
        sizes = [kv_bytes_per_token(model, q) for q in KV_QUALITY_ORDER]
        assert sizes == sorted(sizes, reverse=True), "the ladder must be monotone"
        assert sizes[-1] < sizes[0] / 3


def test_a_tight_machine_still_produces_a_working_plan(catalog):
    """Putting a floor under the cache must not make small machines unservable:
    the tuner has to reach for context and layers instead, not give up."""
    hardware = make_hardware(vram_gb=6, ram_gb=16)
    profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.SPEED)
    plan = build_plan(profile, hardware, catalog)
    config = tune(plan.best, hardware, profile, "llamacpp").config
    assert config.context_tokens >= 2048
    assert plan.best.estimate.total <= hardware.memory_budget + 2 * GIB
