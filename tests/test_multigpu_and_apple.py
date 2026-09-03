"""Two machines llmfit was configuring worse than a careful person would.

A rig with two different cards, where dividing the work by capacity hands the
slow card a share of every forward pass it cannot keep up with. And a Mac, where
the engine that is materially faster was ranked third because the catalog it was
judged against barely knew it existed.
"""

from __future__ import annotations

import pytest

from llmfit.catalog.catalog import load_catalog
from llmfit.hardware.bandwidth import estimated_gpu_bandwidth_bytes
from llmfit.planner.plan import build_plan
from llmfit.runtimes.registry import prefer_mlx_for
from llmfit.tuning.advanced import _tensor_split
from llmfit.types import (
    GIB,
    GPU,
    Accelerator,
    HardwareProfile,
    Task,
    UseCaseProfile,
)


def rig(*cards, ram_gb=64) -> HardwareProfile:
    return HardwareProfile(
        os="linux", arch="x86_64", cpu_name="CPU", physical_cores=8,
        logical_cores=16, ram_bytes=int(ram_gb * GIB),
        disk_free_bytes=int(500 * GIB), unified_memory=False,
        gpus=[GPU(i, name, Accelerator.NVIDIA, int(vram * GIB))
              for i, (name, vram) in enumerate(cards)],
    )


def mac(ram_gb=64) -> HardwareProfile:
    return HardwareProfile(
        os="darwin", arch="arm64", cpu_name="Apple M4 Max", physical_cores=14,
        logical_cores=14, ram_bytes=int(ram_gb * GIB),
        disk_free_bytes=int(500 * GIB), unified_memory=True,
        gpus=[GPU(0, "Apple M4 Max", Accelerator.APPLE, int(ram_gb * GIB))],
    )


FAST_SLOW = (("NVIDIA GeForce RTX 3090", 24), ("NVIDIA GeForce RTX 3060", 12))
MATCHED = (("NVIDIA GeForce RTX 3090", 24), ("NVIDIA GeForce RTX 3090", 24))


class TestMismatchedCards:
    """Under a layer split every token walks through each card in turn, so the
    time per token is the sum of the per-card times. A card handed work in
    proportion to its capacity rather than its speed holds up everything behind
    it, and pairing a fast card with a slow one is the common case."""

    def test_the_slow_card_gets_less_than_its_capacity_share(self):
        hardware = rig(*FAST_SLOW)
        split = _tensor_split(hardware, int(20 * GIB))
        capacity_share = 12 / 36
        assert split[1] < capacity_share

    def test_the_fast_card_gets_more(self):
        split = _tensor_split(rig(*FAST_SLOW), int(20 * GIB))
        assert split[0] > split[1]
        assert split[0] > 24 / 36

    def test_matched_cards_split_evenly(self):
        """Nothing to solve when the cards are the same, and the simpler answer
        reads better in the report."""
        assert _tensor_split(rig(*MATCHED), int(30 * GIB)) == [0.5, 0.5]

    @pytest.mark.parametrize("model_gb", [10, 20, 30, 34])
    def test_no_card_is_given_more_than_it_holds(self, model_gb):
        """Speed decides the split and capacity constrains it, not the other way
        round. Getting that backwards is an out-of-memory error."""
        hardware = rig(*FAST_SLOW)
        split = _tensor_split(hardware, int(model_gb * GIB))
        total = sum(split)
        for index, share in enumerate(split):
            assigned = share / total * model_gb
            assert assigned <= hardware.gpus[index].vram_gb + 1e-6

    def test_the_shares_are_rounded_down_not_to_nearest(self):
        """A card held at its ceiling is there because it cannot take more, and
        rounding its share up by a thousandth is how a split that fits on paper
        fails to on the card."""
        hardware = rig(*FAST_SLOW)
        split = _tensor_split(hardware, int(34 * GIB))
        assert split[0] / sum(split) * 34 <= 24.0

    def test_a_single_card_needs_no_split(self):
        assert _tensor_split(rig(("NVIDIA GeForce RTX 4090", 24)), int(10 * GIB))

    def test_no_cards_yields_nothing(self):
        assert _tensor_split(rig(), int(10 * GIB)) == []

    def test_the_bandwidth_gap_is_real_and_not_assumed(self):
        """The whole argument rests on these being different, so it is worth
        checking that the numbers behind it actually differ."""
        fast, slow = rig(*FAST_SLOW).gpus
        assert (estimated_gpu_bandwidth_bytes(fast)
                > estimated_gpu_bandwidth_bytes(slow) * 1.5)


class TestWhatGetsEmittedForASplit:
    def _flags(self, hardware, catalog):
        from llmfit.runtimes.llamacpp import LlamaCppRuntime
        from llmfit.tuning.tuner import tune

        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        config = tune(plan.best, hardware, profile, "llamacpp", catalog=catalog)
        return LlamaCppRuntime().launch_command(plan.best.model, config.config)

    def test_main_gpu_is_not_passed_alongside_a_split(self, catalog):
        """It names the one card to use when the model is not split at all.
        Passing it with a split says nothing: under a layer split the cache for
        a layer lives on whichever card owns that layer."""
        command = self._flags(rig(*FAST_SLOW), catalog)
        if "--tensor-split" in command:
            assert "--main-gpu" not in command

    def test_the_deprecated_split_mode_is_never_emitted(self, catalog):
        """Upstream deprecated row and says tensor supersedes it."""
        command = self._flags(rig(*MATCHED), catalog)
        if "--split-mode" in command:
            assert command[command.index("--split-mode") + 1] != "row"

    def test_the_report_does_not_claim_a_card_holds_the_cache(self, catalog):
        """It used to say the largest card holds the KV cache, which is not how
        a layer split works and would send someone debugging in the wrong
        direction."""
        from llmfit.tuning.tuner import tune

        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
        hardware = rig(*FAST_SLOW)
        plan = build_plan(profile, hardware, catalog)
        tuned = tune(plan.best, hardware, profile, "llamacpp", catalog=catalog)
        rationale = tuned.config.advanced.offload.rationale
        assert "largest and holds the KV cache" not in rationale


class TestAppleSilicon:
    """MLX generates appreciably faster than llama.cpp on this hardware. llmfit
    ranked it third, justified by the GGUF engines having broader coverage,
    which was measured against a catalog that recorded exactly one MLX build."""

    def test_a_model_with_an_mlx_build_is_served_by_mlx(self, catalog):
        restricted = catalog.restrict_to("qwen3-8b")
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192),
                          mac(), restricted)
        assert plan.best.model.mlx_repo, "the fixture depends on this model having one"
        assert plan.runtime == "mlx"

    def test_a_model_without_one_falls_back_without_pretending(self, catalog):
        model = catalog.require("qwen3-14b")
        assert not model.mlx_repo
        assert prefer_mlx_for(model, mac(), "ollama") is None

    def test_the_reason_says_the_fallback_is_a_fallback(self, catalog):
        model = catalog.require("qwen3-8b")
        _, reason = prefer_mlx_for(model, mac(), "ollama")
        assert "fallback" in reason

    def test_it_does_not_fire_on_a_machine_without_unified_memory(self, catalog):
        model = catalog.require("qwen3-8b")
        assert prefer_mlx_for(model, rig(*MATCHED), "ollama") is None

    def test_it_does_not_fire_when_mlx_was_already_chosen(self, catalog):
        model = catalog.require("qwen3-8b")
        assert prefer_mlx_for(model, mac(), "mlx") is None

    def test_an_explicit_runtime_is_never_overridden(self, catalog):
        """Someone who names an engine has a reason, and it is not llmfit's to
        second-guess."""
        restricted = catalog.restrict_to("qwen3-8b")
        plan = build_plan(UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192),
                          mac(), restricted, runtime="llamacpp")
        assert plan.runtime == "llamacpp"

    def test_a_missing_model_is_handled(self):
        assert prefer_mlx_for(None, mac(), "ollama") is None


def test_the_catalog_records_how_few_mlx_builds_it_knows():
    """Guarding the reasoning rather than the number. The old comment claimed
    the GGUF engines had broader coverage; that was true of this catalog and not
    of the world, which is circular. If this ever grows, the Apple default
    should be revisited rather than left as it is."""
    catalog = load_catalog(include_user_dir=False)
    with_mlx = [m for m in catalog.models if m.mlx_repo]
    assert with_mlx, "at least one model must exercise the MLX path"
