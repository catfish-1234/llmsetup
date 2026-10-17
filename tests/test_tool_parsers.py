"""Telling vLLM which shape a model's tool calls arrive in.

A model can emit a perfectly good tool call and still have the server hand back
prose, because the server was looking for a different shape. The shapes are not
interchangeable between families, or even between versions of one family, so a
plausible-looking default is not a safe default: it causes the exact failure it
is meant to prevent.
"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmsetup.planner.plan import build_plan
from llmsetup.repair.strategies import (
    COST_ORDER,
    STRATEGIES,
    RepairState,
    _try_tool_parser,
    strategies_for,
)
from llmsetup.runtimes.vllm import VllmRuntime
from llmsetup.tuning.tuner import tool_parsers_for, tune
from llmsetup.types import ModelSpec, RuntimeConfig, Task, UseCaseProfile
from llmsetup.verify import signatures as sig


def spec(**kwargs) -> ModelSpec:
    base = {"id": "m", "name": "M", "publisher": "p", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072, "supports_tools": True, "hf_repo": "p/m"}
    base.update(kwargs)
    return ModelSpec(**base)


class TestFamilyMapping:
    @pytest.mark.parametrize("model_id,publisher,expected", [
        ("qwen3-8b", "Qwen", "hermes"),
        ("mistral-small-3.2-24b", "mistralai", "mistral"),
        ("llama-3.1-8b", "meta-llama", "llama3_json"),
    ])
    def test_documented_families_get_their_parser(self, model_id, publisher, expected):
        assert tool_parsers_for(spec(id=model_id, publisher=publisher))[0] == expected

    def test_llama_4_is_not_treated_as_llama_3(self):
        """They emit different shapes, and the substring match would otherwise
        hand a Llama 4 model the Llama 3 parser."""
        parsers = tool_parsers_for(spec(id="llama-4-scout", publisher="meta-llama"))
        assert parsers[0] == "llama4_pythonic"
        assert "llama3_json" not in parsers

    @pytest.mark.parametrize("model_id,publisher", [
        ("gpt-oss-20b", "openai"),
        ("gemma-3-12b", "google"),
        ("phi-4-14b", "microsoft"),
    ])
    def test_an_undocumented_family_gets_nothing(self, model_id, publisher):
        """The important case. Handing these the Hermes parser because it is the
        most common one is how a working tool call turns into prose."""
        assert tool_parsers_for(spec(id=model_id, publisher=publisher)) == []

    def test_a_model_matching_two_families_gets_both(self):
        """A DeepSeek distill of Qwen carries Qwen's tokenizer and template,
        which decides the shape, but DeepSeek's own parser is worth a try after
        that rather than being discarded."""
        parsers = tool_parsers_for(
            spec(id="deepseek-r1-distill-qwen-32b", publisher="deepseek-ai"))
        assert "deepseek_v3" in parsers and "hermes" in parsers

    def test_the_order_does_not_depend_on_dictionary_iteration(self):
        model = spec(id="deepseek-r1-distill-qwen-32b", publisher="deepseek-ai")
        assert tool_parsers_for(model) == tool_parsers_for(model)


class TestEmittedFlags:
    def _command(self, model, **extra):
        config = RuntimeConfig(runtime="vllm", model_id=model.id, quant="Q4_K_M",
                               context_tokens=8192, gpu_layers=32, total_layers=32)
        config.extra.update(extra)
        return VllmRuntime().launch_command(model, config)

    def test_a_known_parser_turns_auto_tool_choice_on(self):
        command = self._command(spec(), tool_parser="hermes")
        assert "--enable-auto-tool-choice" in command
        assert command[command.index("--tool-call-parser") + 1] == "hermes"

    def test_no_parser_means_no_flags_at_all(self):
        """Not a default parser. Auto tool choice without a matching parser is
        worse than leaving it off, because the failure looks like the model's
        fault."""
        command = self._command(spec())
        assert "--tool-call-parser" not in command
        assert "--enable-auto-tool-choice" not in command

    def test_a_model_without_tool_support_gets_neither(self):
        command = self._command(spec(supports_tools=False), tool_parser="hermes")
        assert "--enable-auto-tool-choice" not in command


class TestRepair:
    def _state(self, catalog, model_id="qwen3-8b", **extra):
        hardware = make_hardware(vram_gb=48, ram_gb=128)
        profile = UseCaseProfile(tasks=[Task.AGENT], needs_tools=True,
                                 concurrent_users=4, context_tokens=8192)
        plan = build_plan(profile, hardware, catalog, runtime="vllm")
        candidate = next((c for c in plan.candidates if c.model.id == model_id),
                         plan.best)
        tuned = tune(candidate, hardware, profile, "vllm", catalog=catalog)
        tuned.config.extra.update(extra)
        return RepairState(config=tuned.config, candidate=candidate,
                           hardware=hardware, profile=profile, catalog=catalog)

    def test_moves_to_the_next_parser_for_the_family(self, catalog):
        state = self._state(catalog, tool_parser="hermes")
        outcome = _try_tool_parser(state)
        assert outcome is not None
        assert state.config.extra["tool_parser"] != "hermes"

    def test_never_repeats_a_parser_it_already_tried(self, catalog):
        state = self._state(catalog, tool_parser="hermes")
        seen = {"hermes"}
        while _try_tool_parser(state) is not None:
            parser = state.config.extra["tool_parser"]
            assert parser not in seen, "the loop would cycle"
            seen.add(parser)
        assert len(seen) > 1

    def test_gives_up_once_the_family_is_exhausted(self, catalog):
        """This is what makes it safe to run more than once: it consumes a
        finite list and then declines, so the sequence still ends."""
        state = self._state(catalog, tool_parser="hermes")
        for _ in range(20):
            if _try_tool_parser(state) is None:
                break
        else:
            pytest.fail("the strategy never exhausted itself")

    def test_does_nothing_on_a_runtime_without_the_flag(self, catalog):
        state = self._state(catalog)
        state.config.runtime = "llamacpp"
        assert _try_tool_parser(state) is None

    def test_declared_repeatable_so_the_loop_will_run_it_again(self):
        named = {s.name: s for s in STRATEGIES}
        assert named["try_tool_parser"].exhausts_itself

    def test_only_self_exhausting_strategies_are_repeatable(self):
        """Every other strategy relies on the once-per-signature rule for
        termination, so a stray flag here would let the loop cycle."""
        repeatable = {s.name for s in STRATEGIES if s.exhausts_itself}
        assert repeatable == {"try_tool_parser", "reduce_context", "switch_model"}

    def test_it_is_cheap_enough_to_try_before_swapping_the_model(self, catalog):
        names = [s.name for s in sorted(strategies_for(sig.TOOLS_UNSUPPORTED),
                                        key=lambda s: COST_ORDER[s.cost])]
        assert names.index("try_tool_parser") < names.index("switch_model")
