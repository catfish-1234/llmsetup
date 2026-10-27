"""Settings taken from publishers and community practice.

Three findings drive these: publishers state sampling parameters that differ
sharply between families and modes; per-layer optimised quantizations beat plain
K-quants at the same size; and several machine-level settings sit outside the
inference engine entirely.
"""

from __future__ import annotations

import pytest

from llmsetup.download import RemoteFile, dynamic_variants, pick_gguf
from llmsetup.tuning import publisher
from llmsetup.tuning.platform_tuning import plan_platform
from llmsetup.tuning.sampling import sampling_for
from llmsetup.types import (
    GIB,
    GPU,
    Accelerator,
    HardwareProfile,
    RuntimeConfig,
    Task,
    UseCaseProfile,
)


def machine(**overrides):
    base = {"os": "linux", "arch": "x86_64", "cpu_name": "CPU",
            "physical_cores": 8, "logical_cores": 16, "ram_bytes": 64 * GIB,
            "disk_free_bytes": 500 * GIB, "gpus": [], "unified_memory": False}
    base.update(overrides)
    return HardwareProfile(**base)


def config(**overrides):
    base = {"runtime": "llamacpp", "model_id": "m", "quant": "Q4_K_M",
            "context_tokens": 8192, "gpu_layers": 0, "total_layers": 40}
    base.update(overrides)
    return RuntimeConfig(**base)


# --------------------------------------------------------------------------- #
# publisher recommendations
# --------------------------------------------------------------------------- #


class TestPublisherRecommendations:
    def test_every_entry_cites_a_source(self):
        """A recommendation without a source is indistinguishable from a guess."""
        for family, modes in publisher.RECOMMENDATIONS.items():
            for mode, rec in modes.items():
                assert rec.source, f"{family}/{mode}"

    @pytest.mark.parametrize("family,mode", [
        ("qwen3", "instruct"), ("qwen3", "thinking"), ("gemma3", "instruct"),
        ("mistral", "instruct"), ("llama3", "instruct"), ("gptoss", "thinking"),
    ])
    def test_values_are_in_plausible_ranges(self, family, mode):
        rec = publisher.RECOMMENDATIONS[family][mode]
        assert 0.0 <= rec.temperature <= 2.0
        assert 0.0 < rec.top_p <= 1.0
        assert 0.0 <= rec.presence_penalty <= 2.0

    def test_families_genuinely_differ(self):
        """If every family wanted the same settings this table would be pointless."""
        temperatures = {
            family: modes["instruct"].temperature
            for family, modes in publisher.RECOMMENDATIONS.items()
            if "instruct" in modes
        }
        assert len(set(temperatures.values())) >= 3

    def test_an_unknown_family_returns_nothing(self):
        """Better no citation than an invented one."""
        assert publisher.lookup("not-a-family", False) is None

    def test_a_family_with_one_mode_falls_back_to_it(self):
        """gpt-oss publishes only thinking values; those beat generic defaults
        for an instruct run."""
        assert publisher.lookup("gptoss", thinking=False) is not None


class TestSamplingFollowsPublishers:
    def test_published_values_beat_task_defaults(self, catalog):
        rec = publisher.RECOMMENDATIONS["gemma3"]["instruct"]
        sampling = sampling_for(catalog.require("gemma-3-27b"),
                                UseCaseProfile(tasks=[Task.CHAT]))
        assert sampling.temperature == rec.temperature
        assert sampling.top_p == rec.top_p
        assert sampling.source == rec.source

    def test_qwens_unusual_presence_penalty_survives(self, catalog):
        """Qwen3 asks for 1.5 in non-thinking mode, where most families want 0.
        Silently resetting it to a common default is the bug this guards."""
        sampling = sampling_for(catalog.require("qwen3-8b"),
                                UseCaseProfile(tasks=[Task.CHAT]), thinking=False)
        assert sampling.presence_penalty == 1.5

    def test_the_publishers_own_carve_out_for_code_applies(self, catalog):
        """Qwen3 drops the presence penalty for code; repeated tokens there are
        legitimate."""
        sampling = sampling_for(catalog.require("qwen3-8b"),
                                UseCaseProfile(tasks=[Task.CODING]), thinking=False)
        assert sampling.presence_penalty == 0.0

    def test_a_greedy_warning_blocks_the_deterministic_override(self, catalog):
        """Qwen3 warns against greedy decoding, so a deterministic task must not
        drag its temperature to near zero."""
        sampling = sampling_for(catalog.require("qwen3-8b"),
                                UseCaseProfile(tasks=[Task.EXTRACT]), thinking=False)
        assert sampling.temperature >= 0.6

    def test_determinism_is_applied_where_the_publisher_permits(self, catalog):
        """Mistral issues no such warning, so extraction gets the low end."""
        sampling = sampling_for(catalog.require("mistral-small-3.2-24b"),
                                UseCaseProfile(tasks=[Task.EXTRACT]))
        assert sampling.temperature <= 0.2

    def test_the_source_is_always_reported(self, catalog):
        """The user should be able to tell a publisher's number from ours."""
        for model_id in ("qwen3-8b", "gemma-3-27b", "qwen3.8-27b"):
            sampling = sampling_for(catalog.require(model_id),
                                    UseCaseProfile(tasks=[Task.CHAT]))
            assert sampling.source

    def test_an_unlisted_family_says_so_rather_than_inventing_a_citation(self, catalog):
        sampling = sampling_for(catalog.require("qwen3.8-27b"),
                                UseCaseProfile(tasks=[Task.CHAT]))
        assert sampling.source == "llmsetup task defaults"


# --------------------------------------------------------------------------- #
# dynamic quantizations
# --------------------------------------------------------------------------- #


class TestDynamicQuants:
    def test_a_dynamic_build_is_preferred_at_the_same_size(self):
        """Per-layer optimisation is free quality: same bytes, closer to the
        original weights."""
        files = [RemoteFile("r", n) for n in
                 ("M-Q4_K_M.gguf", "M-UD-Q4_K_XL.gguf", "M-Q8_0.gguf")]
        assert pick_gguf(files, "Q4_K_M").filename == "M-UD-Q4_K_XL.gguf"

    def test_it_falls_back_cleanly_when_none_exists(self):
        files = [RemoteFile("r", n) for n in ("M-Q4_K_M.gguf", "M-Q8_0.gguf")]
        assert pick_gguf(files, "Q4_K_M").filename == "M-Q4_K_M.gguf"

    def test_an_explicit_quant_is_still_honoured(self):
        files = [RemoteFile("r", n) for n in
                 ("M-Q4_K_M.gguf", "M-UD-Q4_K_XL.gguf", "M-Q8_0.gguf")]
        assert pick_gguf(files, "Q8_0").filename == "M-Q8_0.gguf"

    def test_the_preference_can_be_turned_off(self):
        files = [RemoteFile("r", n) for n in ("M-Q4_K_M.gguf", "M-UD-Q4_K_XL.gguf")]
        assert pick_gguf(files, "Q4_K_M",
                         prefer_dynamic=False).filename == "M-Q4_K_M.gguf"

    def test_an_already_dynamic_request_is_not_rewritten(self):
        assert dynamic_variants("UD-Q4_K_XL") == ["UD-Q4_K_XL"]

    def test_the_arm_repack_trap_still_does_not_fire(self):
        """Q4_0_4_8 is an ARM-specific repack that will not run on x86; the
        dynamic preference must not reintroduce that confusion."""
        files = [RemoteFile("r", n) for n in ("M-Q4_0.gguf", "M-Q4_0_4_8.gguf")]
        assert pick_gguf(files, "Q4_0").filename == "M-Q4_0.gguf"


# --------------------------------------------------------------------------- #
# platform actions
# --------------------------------------------------------------------------- #


class TestPlatformActions:
    def test_apple_gets_a_wired_limit_only_when_it_needs_one(self):
        mac = machine(os="darwin", unified_memory=True, ram_bytes=128 * GIB,
                      gpus=[GPU(0, "Apple M3 Max", Accelerator.APPLE, 128 * GIB)])
        assert plan_platform(mac, config(), estimate_bytes=int(40 * GIB)) == []

        actions = plan_platform(mac, config(), estimate_bytes=int(100 * GIB))
        assert len(actions) == 1
        assert "iogpu.wired_limit_mb" in actions[0].command
        assert actions[0].needs_root and not actions[0].optional

    def test_the_wired_limit_stays_below_a_desktop_breaking_value(self):
        mac = machine(os="darwin", unified_memory=True, ram_bytes=64 * GIB,
                      gpus=[GPU(0, "Apple M4 Pro", Accelerator.APPLE, 64 * GIB)])
        actions = plan_platform(mac, config(), estimate_bytes=int(63 * GIB))
        requested_mb = int(actions[0].command.rsplit("=", 1)[1])
        assert requested_mb <= int(64 * GIB * 0.85 / (1024 ** 2))

    def test_an_older_amd_card_gets_the_architecture_override(self):
        actions = plan_platform(
            machine(gpus=[GPU(0, "AMD Radeon RX 6800 XT", Accelerator.AMD, 16 * GIB)]),
            config())
        assert any(a.env.get("HSA_OVERRIDE_GFX_VERSION") for a in actions)

    def test_a_natively_supported_amd_card_gets_no_override(self):
        """The override used to be needed for these and is now unhelpful."""
        actions = plan_platform(
            machine(gpus=[GPU(0, "AMD Radeon RX 7900 XTX", Accelerator.AMD, 24 * GIB)]),
            config())
        assert not any(a.env.get("HSA_OVERRIDE_GFX_VERSION") for a in actions)

    def test_an_unknown_amd_card_gets_a_diagnostic_rather_than_a_guess(self):
        actions = plan_platform(
            machine(gpus=[GPU(0, "AMD Radeon Pro Vega II", Accelerator.AMD, 32 * GIB)]),
            config())
        assert actions and not actions[0].env
        assert "rocminfo" in actions[0].command

    def test_numa_advice_only_for_multi_socket_cpu_work(self):
        assert plan_platform(machine(physical_cores=8), config()) == []
        assert plan_platform(machine(physical_cores=64), config())

    def test_no_numa_advice_when_the_model_is_fully_on_the_gpu(self):
        actions = plan_platform(
            machine(physical_cores=64,
                    gpus=[GPU(0, "RTX 4090", Accelerator.NVIDIA, 24 * GIB)]),
            config(gpu_layers=40, total_layers=40))
        assert not any("NUMA" in a.title or "memory nodes" in a.title for a in actions)

    def test_apple_silicon_never_gets_numa_advice(self):
        """Unified memory is a single pool; a NUMA policy there is meaningless."""
        mac = machine(os="darwin", unified_memory=True, physical_cores=64,
                      ram_bytes=128 * GIB,
                      gpus=[GPU(0, "Apple M3 Ultra", Accelerator.APPLE, 128 * GIB)])
        actions = plan_platform(mac, config(), estimate_bytes=int(10 * GIB))
        assert not any("memory nodes" in a.title for a in actions)

    def test_multi_gpu_nvidia_gets_a_topology_check(self):
        actions = plan_platform(
            machine(gpus=[GPU(i, "RTX 4090", Accelerator.NVIDIA, 24 * GIB)
                          for i in range(2)]),
            config(gpu_layers=40, total_layers=40))
        assert any("nvidia-smi topo" in (a.command or "") for a in actions)

    def test_every_action_explains_itself_and_names_a_command(self):
        mac = machine(os="darwin", unified_memory=True, ram_bytes=128 * GIB,
                      gpus=[GPU(0, "Apple M3 Max", Accelerator.APPLE, 128 * GIB)])
        for actions in (plan_platform(mac, config(), estimate_bytes=int(100 * GIB)),
                        plan_platform(machine(physical_cores=64), config())):
            for action in actions:
                assert action.title and action.reason
                assert action.command or action.env
