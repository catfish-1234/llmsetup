"""Vision models are two files, and the second one is easy to forget.

A vision model served without its multimodal projector does not fail loudly. It
loads, it accepts an image, and it answers from the text alone -- confidently
and wrongly. These tests pin the whole path: finding the projector, keeping it
out of the way of the language model, budgeting for it, and repairing a setup
that is missing it.
"""

from __future__ import annotations

import pytest

from llmfit.download import RemoteFile, is_projector, pick_gguf, pick_mmproj
from llmfit.planner.memory import estimate_memory
from llmfit.types import MIB, ModelSpec


def files(*names) -> list:
    return [RemoteFile(repo="pub/model-GGUF", filename=n, size=1000) for n in names]


def spec(**kwargs) -> ModelSpec:
    base = {"id": "t", "name": "T", "publisher": "p", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072}
    base.update(kwargs)
    return ModelSpec(**base)


class TestProjectorDiscovery:
    @pytest.mark.parametrize("name", [
        "mmproj-F16.gguf",
        "mmproj-model-f16.gguf",
        "mmproj-gemma-3-4b-it-Q4_K_M.gguf",
        "mm-proj-BF16.gguf",
        "model-projector-f16.gguf",
    ])
    def test_recognises_the_naming_repos_actually_use(self, name):
        assert is_projector(name)

    @pytest.mark.parametrize("name", [
        "gemma-3-4b-it-Q4_K_M.gguf",
        "Qwen3-8B-UD-Q4_K_XL.gguf",
        "model-00001-of-00002.gguf",
    ])
    def test_does_not_mistake_a_language_model_for_one(self, name):
        assert not is_projector(name)

    def test_prefers_the_least_quantized_projector(self):
        """Vision towers degrade fast under quantization and the file is small,
        so precision is nearly free here even when the language model is Q4."""
        chosen = pick_mmproj(files(
            "mmproj-Q4_0.gguf", "mmproj-Q8_0.gguf", "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        ))
        assert chosen.filename == "mmproj-BF16.gguf"

    def test_falls_back_through_the_precision_ladder(self):
        chosen = pick_mmproj(files("mmproj-Q4_0.gguf", "mmproj-Q8_0.gguf"))
        assert chosen.filename == "mmproj-Q8_0.gguf"

    def test_returns_none_when_the_repo_publishes_none(self):
        assert pick_mmproj(files("model-Q4_K_M.gguf")) is None

    def test_takes_an_unlabelled_projector_over_nothing(self):
        chosen = pick_mmproj(files("mmproj.gguf"))
        assert chosen.filename == "mmproj.gguf"


class TestProjectorDoesNotShadowTheModel:
    def test_projector_is_never_chosen_as_the_language_model(self):
        """The trap: some repos name the projector after the model's own quant,
        so a naive quant search can return the 800 MB projector as the model."""
        chosen = pick_gguf(files(
            "mmproj-gemma-3-4b-it-Q4_K_M.gguf",
            "gemma-3-4b-it-Q4_K_M.gguf",
        ), "Q4_K_M")
        assert chosen.filename == "gemma-3-4b-it-Q4_K_M.gguf"

    def test_a_repo_with_only_a_projector_has_no_model(self):
        assert pick_gguf(files("mmproj-F16.gguf"), "F16") is None


class TestLocalStore:
    def test_find_local_skips_the_projector(self, tmp_path, monkeypatch):
        """Alphabetical order is not a safe way to pick the model: a projector
        sorts before a model whose name starts late in the alphabet."""
        from llmfit.runtimes import gguf_store

        model = spec(id="zephyr-vision", publisher="pub", supports_vision=True)
        directory = gguf_store.store_path(model, "Q4_K_M")
        directory.mkdir(parents=True)
        (directory / "mmproj-BF16.gguf").write_bytes(b"proj")
        (directory / "zephyr-vision-Q4_K_M.gguf").write_bytes(b"weights")

        assert gguf_store.find_local(model, "Q4_K_M").name == "zephyr-vision-Q4_K_M.gguf"
        assert gguf_store.find_local_projector(model, "Q4_K_M").name == "mmproj-BF16.gguf"


class TestMemoryBudget:
    def test_a_vision_model_is_charged_for_its_projector(self):
        """It is a second network resident in the same memory. Leaving it out of
        the budget is how a vision model fits on paper and OOMs on the first
        image."""
        text_only = estimate_memory(spec(), 4.85, 8192)
        vision = estimate_memory(spec(supports_vision=True), 4.85, 8192)
        extra = vision.overhead - text_only.overhead
        assert 512 * MIB <= extra <= 2048 * MIB
        assert vision.total > text_only.total


class TestRepair:
    def _state(self, catalog, gpu_24gb, **config_overrides):
        from llmfit.planner.plan import build_plan
        from llmfit.repair.strategies import RepairState
        from llmfit.tuning.tuner import tune
        from llmfit.types import Priority, Task, UseCaseProfile

        profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.QUALITY)
        plan = build_plan(profile, gpu_24gb, catalog)
        tuned = tune(plan.best, gpu_24gb, profile, "llamacpp")
        for key, value in config_overrides.items():
            setattr(tuned.config, key, value)
        return RepairState(config=tuned.config, candidate=plan.best,
                           hardware=gpu_24gb, profile=profile, catalog=catalog)

    def test_does_nothing_for_a_text_only_model(self, catalog, gpu_24gb):
        from llmfit.repair.strategies import _fetch_vision_projector

        state = self._state(catalog, gpu_24gb)
        state.candidate.model.supports_vision = False
        assert _fetch_vision_projector(state) is None

    def test_does_nothing_when_the_projector_is_already_loaded(self, catalog, gpu_24gb):
        from llmfit.repair.strategies import _fetch_vision_projector

        state = self._state(catalog, gpu_24gb, mmproj_path="/tmp/mmproj.gguf")
        state.candidate.model.supports_vision = True
        assert _fetch_vision_projector(state) is None

    def test_loads_a_projector_that_is_already_in_the_store(
            self, catalog, gpu_24gb, monkeypatch):
        """No network: the store already has the file, which is also the path a
        second run takes."""
        from llmfit.repair import strategies
        from llmfit.runtimes import gguf_store

        state = self._state(catalog, gpu_24gb)
        model = state.candidate.model
        model.supports_vision = True
        directory = gguf_store.store_path(model, state.config.quant)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "mmproj-BF16.gguf").write_bytes(b"proj")

        outcome = strategies._fetch_vision_projector(state)
        assert outcome is not None
        assert outcome.cost == "low"
        assert state.config.mmproj_path.endswith("mmproj-BF16.gguf")

    def test_is_registered_for_the_vision_failure_signature(self):
        from llmfit.repair.strategies import STRATEGIES
        from llmfit.verify import signatures as sig

        named = {s.name: s for s in STRATEGIES}
        assert named["fetch_vision_projector"].handles(sig.VISION_UNSUPPORTED)


class TestLaunchCommand:
    def test_mmproj_is_passed_to_the_server(self, catalog, gpu_24gb):
        from llmfit.planner.plan import build_plan
        from llmfit.runtimes.llamacpp import LlamaCppRuntime
        from llmfit.tuning.tuner import tune
        from llmfit.types import Priority, Task, UseCaseProfile

        profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.QUALITY)
        plan = build_plan(profile, gpu_24gb, catalog)
        tuned = tune(plan.best, gpu_24gb, profile, "llamacpp")
        tuned.config.mmproj_path = "/models/mmproj-BF16.gguf"

        cmd = LlamaCppRuntime().launch_command(plan.best.model, tuned.config)
        assert "--mmproj" in cmd
        assert cmd[cmd.index("--mmproj") + 1] == "/models/mmproj-BF16.gguf"

    def test_no_flag_when_there_is_no_projector(self, catalog, gpu_24gb):
        from llmfit.planner.plan import build_plan
        from llmfit.runtimes.llamacpp import LlamaCppRuntime
        from llmfit.tuning.tuner import tune
        from llmfit.types import Priority, Task, UseCaseProfile

        profile = UseCaseProfile(tasks=[Task.CHAT], priority=Priority.QUALITY)
        plan = build_plan(profile, gpu_24gb, catalog)
        tuned = tune(plan.best, gpu_24gb, profile, "llamacpp")
        assert "--mmproj" not in LlamaCppRuntime().launch_command(
            plan.best.model, tuned.config)
