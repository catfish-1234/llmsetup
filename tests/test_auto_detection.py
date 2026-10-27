"""Hardware and runtime discovery never depend on answers from the user."""

import pytest
from conftest import make_hardware

from llmsetup.hardware import detect as detection
from llmsetup.hardware import gpu as gpu_probe
from llmsetup.interview.questions import QUESTIONS
from llmsetup.planner.plan import build_adaptive_plan
from llmsetup.runtimes.registry import scan_runtimes
from llmsetup.types import GIB, GPU, Accelerator, UseCaseProfile
from llmsetup.util.errors import PlanningError


def test_question_bank_collects_goals_only():
    allowed = {"tasks", "clients", "context_tokens", "priority", "concurrent_users",
               "needs_tools", "needs_json", "needs_vision", "needs_embeddings",
               "languages", "commercial_use", "requested_model"}
    assert {question.key for question in QUESTIONS} <= allowed


def test_failed_probes_choose_conservative_defaults(monkeypatch):
    def broken():
        raise OSError("probe unavailable")

    for name in ("core_counts", "memory_bytes", "cpu_name", "normalized_arch", "is_apple_silicon"):
        monkeypatch.setattr(detection.cpu_probe, name, broken)
    monkeypatch.setattr(detection, "detect_gpus", broken)
    monkeypatch.setattr("llmsetup.runtimes.registry.scan_runtimes", lambda: ({"llamacpp": False}, []))
    hardware = detection.detect_hardware()
    assert hardware.ram_bytes == 2 * GIB
    assert hardware.physical_cores == 1
    assert hardware.gpus == []
    assert hardware.runtime_availability == {"llamacpp": False}
    assert any("assuming 2 GB" in note for note in hardware.detection_warnings)
    assert all("override" not in note for note in hardware.detection_warnings)


def test_unknown_vram_uses_cpu_and_unknown_free_memory_reserves_headroom(monkeypatch):
    unknown = GPU(0, "Unknown GPU", Accelerator.AMD, 0)
    known = GPU(1, "Known GPU", Accelerator.NVIDIA, 10 * GIB)
    monkeypatch.setattr(detection, "detect_gpus", lambda: ([unknown, known], []))
    monkeypatch.setattr(detection.cpu_probe, "is_apple_silicon", lambda: False)
    monkeypatch.setattr("llmsetup.runtimes.registry.scan_runtimes", lambda: ({}, []))
    hardware = detection.detect_hardware()
    assert hardware.gpus == [known]
    assert hardware.usable_accel_bytes == 9 * GIB
    assert any("Unknown GPU: VRAM unknown" in note for note in hardware.detection_warnings)


def test_broken_vendor_probe_keeps_other_gpus(monkeypatch):
    monkeypatch.setattr(gpu_probe, "_is_apple_silicon", lambda: False)
    monkeypatch.setattr(gpu_probe, "detect_nvidia", lambda: (_ for _ in ()).throw(ValueError("bad CSV")))
    amd = GPU(0, "AMD", Accelerator.AMD, 8 * GIB)
    intel = GPU(0, "Intel", Accelerator.INTEL, 4 * GIB)
    monkeypatch.setattr(gpu_probe, "detect_amd", lambda: ([amd], []))
    monkeypatch.setattr(gpu_probe, "detect_intel", lambda: ([intel], []))
    found, notes = gpu_probe.detect_gpus()
    assert found == [amd, intel]
    assert [gpu.index for gpu in found] == [0, 1]
    assert "bad CSV" in notes[0]


def test_runtime_scan_checks_all_six_even_if_one_probe_raises(monkeypatch):
    class Probe:
        def __init__(self, name):
            self.name = self.label = name

        def is_available(self):
            if self.name == "vllm":
                raise OSError("broken interpreter")
            return self.name in {"ollama", "llamacpp", "lmstudio"}

    names = ["ollama", "llamacpp", "lmstudio", "vllm", "mlx", "tabby"]
    monkeypatch.setattr("llmsetup.runtimes.registry.all_runtimes", lambda: [Probe(n) for n in names])
    inventory, warnings = scan_runtimes()
    assert inventory == {n: n in {"ollama", "llamacpp", "lmstudio"} for n in names}
    assert "broken interpreter" in warnings[0]


def test_planning_shortens_context_without_dropping_capabilities(catalog):
    hardware = make_hardware(vram_gb=0, ram_gb=8)
    profile = UseCaseProfile(context_tokens=1_000_000, needs_tools=True)
    plan = build_adaptive_plan(profile, hardware, catalog, runtime="llamacpp")
    assert plan.profile.context_tokens < profile.context_tokens
    assert plan.profile.needs_tools
    assert plan.best.model.supports_tools
    assert "requested 1000000 tokens" in plan.notes[-1]
    assert profile.context_tokens == 1_000_000


def test_no_fitting_hardware_exhausts_context_fallback(catalog):
    with pytest.raises(PlanningError, match="exceeds"):
        build_adaptive_plan(UseCaseProfile(), make_hardware(vram_gb=0, ram_gb=0.01),
                            catalog, runtime="llamacpp")


def test_free_memory_does_not_shrink_the_gpu_budget():
    """Momentary free VRAM is usage, not capacity: other apps come and go, so
    the plan is sized from the card with a safety margin instead."""
    hardware = make_hardware(vram_gb=24, ram_gb=32)
    hardware.gpus[0].vram_free_bytes = 4 * GIB
    hardware.ram_available_bytes = 0
    assert hardware.usable_accel_bytes == int(24 * GIB * 0.9)
    assert hardware.usable_cpu_bytes == 0


def test_multi_gpu_split_respects_capacity_on_each_card():
    from llmsetup.tuning.advanced import _tensor_split

    hardware = make_hardware(vram_gb=16)
    hardware.gpus[0].vram_free_bytes = 1 * GIB   # busy card: usage, not capacity
    hardware.gpus.append(GPU(1, "NVIDIA RTX 4090", Accelerator.NVIDIA, 24 * GIB,
                             vram_free_bytes=12 * GIB))
    shares = _tensor_split(hardware, 16 * GIB)
    assert shares == [0.4, 0.6]


def test_apple_gpu_budget_also_respects_available_ram():
    hardware = make_hardware(unified=True, vendor=Accelerator.APPLE, ram_gb=64)
    hardware.ram_available_bytes = 5 * GIB
    assert hardware.usable_accel_bytes == 4 * GIB


def test_managed_cpu_runtime_replans_against_ram(catalog, monkeypatch, tmp_path):
    hardware = make_hardware(vram_gb=24, ram_gb=8)
    monkeypatch.setattr("llmsetup.runtimes.install.managed_binary", lambda *a: tmp_path / "llama-server")
    monkeypatch.setattr("llmsetup.util.proc.which", lambda *a: None)
    plan = build_adaptive_plan(UseCaseProfile(), hardware, catalog, runtime="llamacpp")
    assert not plan.hardware.gpus
    assert plan.best.estimate.total <= plan.hardware.usable_cpu_bytes
    assert hardware.gpus
    assert any("managed portable runtime uses CPU" in note for note in plan.notes)


def test_cli_scans_before_the_goal_interview(monkeypatch, capsys):
    from llmsetup.cli import main

    events = []
    hardware = make_hardware()
    hardware.runtime_availability = {"ollama": False, "llamacpp": False}

    def scan(**kw):
        events.append("scan")
        return hardware

    def interview(*a, **kw):
        events.append("interview")
        return UseCaseProfile()

    monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")
    monkeypatch.setattr("llmsetup.cli.detect_hardware", scan)
    monkeypatch.setattr("llmsetup.cli.run_interview", interview)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert main(["setup", "--dry-run", "--offline", "--no-register"]) == 0
    assert events == ["scan", "interview"]
    assert "Detected:" in capsys.readouterr().err

def test_managed_vulkan_runtime_keeps_the_gpu(catalog, monkeypatch, tmp_path):
    hardware = make_hardware(vram_gb=24, ram_gb=8)
    monkeypatch.setattr("llmsetup.runtimes.install.managed_binary", lambda *a: tmp_path / "llama-server")
    monkeypatch.setattr("llmsetup.runtimes.install.managed_flavor", lambda *a: "vulkan")
    monkeypatch.setattr("llmsetup.util.proc.which", lambda *a: None)
    plan = build_adaptive_plan(UseCaseProfile(), hardware, catalog, runtime="llamacpp")
    assert plan.hardware.gpus
    assert not any("managed portable runtime uses CPU" in note for note in plan.notes)

def test_busy_gpu_warns_instead_of_shrinking_the_plan(monkeypatch):
    card = GPU(0, "Known GPU", Accelerator.NVIDIA, 8 * GIB, vram_free_bytes=1 * GIB)
    monkeypatch.setattr(detection, "detect_gpus", lambda: ([card], []))
    monkeypatch.setattr(detection.cpu_probe, "is_apple_silicon", lambda: False)
    monkeypatch.setattr("llmsetup.runtimes.registry.scan_runtimes", lambda: ({}, []))
    hardware = detection.detect_hardware()
    assert hardware.usable_accel_bytes == int(8 * GIB * 0.9)
    assert any("in use by other apps" in note for note in hardware.detection_warnings)
