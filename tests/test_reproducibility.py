"""Getting the same answer twice.

Most advice stops at "set temperature to zero and fix the seed", which is not
enough and is worth being precise about. A seed removes the sampler as a source
of variation. It does not make the engine deterministic: floating-point
addition is not associative, so a request that happens to share a batch with
three others accumulates in a different order and can come out different. Those
are two different fixes with two different costs.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmsetup.planner.plan import build_plan
from llmsetup.runtimes.llamacpp import LlamaCppRuntime
from llmsetup.runtimes.vllm import VllmRuntime
from llmsetup.tuning.platform_tuning import plan_platform
from llmsetup.tuning.tuner import tune
from llmsetup.types import ModelSpec, RuntimeConfig, Task, UseCaseProfile


def config(**kwargs):
    base = {"runtime": "llamacpp", "model_id": "m", "quant": "Q4_K_M",
            "context_tokens": 8192, "gpu_layers": 32, "total_layers": 32}
    base.update(kwargs)
    return RuntimeConfig(**base)


def spec():
    return ModelSpec(id="m", name="M", publisher="p", params_b=8.0, layers=32,
                     hidden_size=4096, kv_heads=8, head_dim=128,
                     max_context=131072, hf_repo="p/m")


class TestSeed:
    @pytest.mark.parametrize("task", [Task.EXTRACT, Task.CODING, Task.TRANSLATE])
    def test_tasks_that_need_the_same_answer_twice_get_a_seed(self, catalog, task):
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[task], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        assert tune(plan.best, hardware, profile, "llamacpp",
                    catalog=catalog).config.seed is not None

    @pytest.mark.parametrize("task", [Task.CHAT, Task.CREATIVE])
    def test_open_ended_work_is_left_alone(self, catalog, task):
        """Pinning the seed on a chat model makes it repeat itself across
        sessions, which is a defect rather than a feature there."""
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[task], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        assert tune(plan.best, hardware, profile, "llamacpp",
                    catalog=catalog).config.seed is None

    def test_the_seed_is_stable_between_runs(self, catalog):
        """A generated seed would defeat the point: the same setup has to give
        the same answer tomorrow, not just twice in a row today."""
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[Task.EXTRACT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        seeds = {tune(plan.best, hardware, profile, "llamacpp",
                      catalog=catalog).config.seed for _ in range(3)}
        assert len(seeds) == 1

    def test_it_reaches_both_engines(self):
        for runtime, cfg in ((LlamaCppRuntime(), config(seed=7)),
                             (VllmRuntime(), config(runtime="vllm", seed=7))):
            command = runtime.launch_command(spec(), cfg)
            assert command[command.index("--seed") + 1] == "7"

    def test_no_seed_means_no_flag(self):
        assert "--seed" not in LlamaCppRuntime().launch_command(spec(), config())

    def test_the_plan_says_what_a_seed_does_not_buy(self, catalog):
        """Overstating this is the common failure. Someone who believes a seed
        gives determinism under load will chase a phantom bug."""
        hardware = make_hardware(vram_gb=24)
        profile = UseCaseProfile(tasks=[Task.EXTRACT], context_tokens=8192)
        plan = build_plan(profile, hardware, catalog)
        tuned = tune(plan.best, hardware, profile, "llamacpp", catalog=catalog)
        reason = next(d.reason for d in tuned.decisions if d.setting == "seed")
        assert "batch" in reason.lower()


class TestBatchInvariance:
    def _actions(self, hardware, task=Task.EXTRACT, runtime="vllm"):
        return plan_platform(hardware, config(runtime=runtime),
                             profile=UseCaseProfile(tasks=[task]))

    def _titles(self, actions):
        return [a.title for a in actions]

    def test_offered_where_reproducibility_is_the_point(self):
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"))
        assert any("reproducible" in t for t in self._titles(actions))

    def test_not_offered_for_open_ended_work(self):
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"),
                                task=Task.CHAT)
        assert not any("reproducible" in t for t in self._titles(actions))

    def test_not_offered_on_hardware_that_cannot_do_it(self):
        """Batch-invariant kernels need compute capability 8.0 or newer.
        Suggesting it on a Turing card is an instruction that fails."""
        actions = self._actions(make_hardware(vram_gb=24, compute="7.5"))
        assert not any("reproducible" in t for t in self._titles(actions))

    def test_not_offered_on_an_engine_without_it(self):
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"),
                                runtime="llamacpp")
        assert not any("reproducible" in t for t in self._titles(actions))

    def test_it_is_an_environment_variable_not_a_command(self):
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"))
        action = next(a for a in actions if "reproducible" in a.title)
        assert action.env == {"VLLM_BATCH_INVARIANT": "1"}
        assert action.command is None

    def test_the_throughput_cost_is_stated(self):
        """It is a large cost and the reader has to be able to weigh it."""
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"))
        action = next(a for a in actions if "reproducible" in a.title)
        assert "60%" in action.reason

    def test_it_stays_opt_in(self):
        actions = self._actions(make_hardware(vram_gb=48, compute="8.9"))
        assert next(a for a in actions if "reproducible" in a.title).optional

    def test_an_unreadable_compute_capability_is_not_assumed_capable(self):
        hardware = make_hardware(vram_gb=48, compute=None)
        assert not any("reproducible" in t
                       for t in self._titles(self._actions(hardware)))
