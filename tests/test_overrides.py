"""``setup --set KEY=VALUE`` and ``--engine-arg``: your settings over the tuner's."""

from __future__ import annotations

import pytest

from llmsetup.cli import main
from llmsetup.runtimes.registry import get_runtime
from llmsetup.tuning.overrides import apply_overrides, parse_overrides
from llmsetup.types import RuntimeConfig
from llmsetup.util.errors import LlmsetupError


def config(**changes):
    base = RuntimeConfig(runtime="llamacpp", model_id="qwen3-8b", quant="Q4_K_M", context_tokens=8192,
                         model_path="/models/m.gguf")
    for key, value in changes.items():
        setattr(base, key, value)
    return base


class TestParse:
    def test_values_take_each_settings_type(self):
        parsed = parse_overrides(["context_tokens=32_768", "flash_attention=yes", "temperature=0.2",
                                  "kv-cache-type=q8_0", "threads=auto", "seed=42"])
        assert parsed == {"context_tokens": 32768, "flash_attention": True, "temperature": 0.2,
                          "kv_cache_type": "q8_0", "threads": None, "seed": 42}

    def test_a_misspelled_setting_suggests_the_right_one(self):
        with pytest.raises(LlmsetupError) as caught:
            parse_overrides(["flash_atention=true"])
        assert "flash_attention" in str(caught.value)

    def test_identity_fields_are_not_settings(self):
        with pytest.raises(LlmsetupError, match="its own flag"):
            parse_overrides(["quant=Q8_0"])

    @pytest.mark.parametrize("pair, message", [
        ("gpu_layers=lots", "whole number"), ("mlock=maybe", "true or false"), ("context_tokens", "KEY=VALUE"),
    ])
    def test_a_bad_value_says_what_it_takes(self, pair, message):
        with pytest.raises(LlmsetupError, match=message):
            parse_overrides([pair])


class TestApply:
    def test_only_what_differs_is_changed_and_reported(self):
        updated, notes = apply_overrides(config(), {"context_tokens": 8192, "flash_attention": True})
        assert updated.flash_attention is True
        assert notes == ["flash_attention: False -> True (your setting, over the tuner's)"]

    def test_engine_flags_reach_the_command_line_as_given(self, catalog):
        updated, _ = apply_overrides(config(), {}, ["--no-warmup", "--some-future-flag=1"])
        command = get_runtime("llamacpp").launch_command(catalog.require("qwen3-8b"), updated)
        # Passed through after llmsetup's own compatibility filter, so a flag the installed build
        # does not list still reaches it, and fails where you can see it if it is wrong.
        assert command[-2:] == ["--no-warmup", "--some-future-flag=1"]

    def test_vllm_takes_engine_flags_too(self, catalog):
        updated, _ = apply_overrides(config(runtime="vllm"), {}, ["--max-num-seqs", "4"])
        command = get_runtime("vllm").launch_command(catalog.require("qwen3-8b"), updated)
        assert command[-2:] == ["--max-num-seqs", "4"]

    def test_an_engine_without_flags_refuses_them(self):
        with pytest.raises(LlmsetupError, match="--runtime llamacpp"):
            apply_overrides(config(runtime="ollama"), {}, ["--no-warmup"])

    def test_serving_to_the_network_is_called_out(self):
        _, notes = apply_overrides(config(), {"host": "0.0.0.0"})
        assert any("anyone who can reach this machine" in note for note in notes)
        _, keyed = apply_overrides(config(), {"host": "0.0.0.0"}, ["--api-key=secret"])
        assert not any("anyone who can reach" in note for note in keyed)
        _, local = apply_overrides(config(), {"host": "::1"})
        assert not any("network" in note for note in local)


def test_setup_rejects_a_bad_setting_before_doing_anything(capsys):
    assert main(["setup", "--set", "contxt=4096", "--task", "chat", "--yes", "--dry-run"]) == 1
    assert "context_tokens" in capsys.readouterr().err


def test_settings_lists_what_can_be_overridden(capsys):
    assert main(["settings"]) == 0
    out = capsys.readouterr().out
    assert "flash_attention" in out and "quant" not in out.split()
