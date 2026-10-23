"""Release regressions reproduced with simulated hardware and no inference engine."""

from dataclasses import replace

import pytest
from conftest import make_hardware

from llmsetup.catalog.catalog import Catalog
from llmsetup.hardware.detect import detect_hardware
from llmsetup.pipeline import Pipeline
from llmsetup.planner.plan import build_adaptive_plan
from llmsetup.runtimes import install
from llmsetup.tuning import tune
from llmsetup.types import GIB, ModelSpec, Task, UseCaseProfile
from llmsetup.util.log import Console


def model(model_id="chosen", params=3, **kwargs):
    fields = {"id": model_id, "name": model_id.title(), "publisher": "fixture", "params_b": params,
              "layers": 32, "hidden_size": 4096, "kv_heads": 32, "head_dim": 128,
              "max_context": 131072, "tasks": ["chat"], "gguf_repo": f"fixture/{model_id}",
              "recommended_quants": ["Q4_K_M"]}
    fields.update(kwargs)
    return ModelSpec(**fields)


def hardware(vram=8, ram=32):
    return replace(make_hardware(vram_gb=vram, ram_gb=ram, os_name="windows",
                                 name="NVIDIA GeForce RTX 5060", compute="12.0"),
                   runtime_availability={"llamacpp": True})


@pytest.mark.parametrize("vram,ram,context,on_gpu,shrinks", [
    (8, 32, 8192, True, False),
    (4, 4, 131072, True, True),
    (0.125, 32, 8192, False, False),
    (0, 16, 8192, False, False),
])
def test_memory_matrix(vram, ram, context, on_gpu, shrinks, monkeypatch):
    monkeypatch.setattr(install, "managed_binary", lambda *a: None)
    hw = hardware(vram, ram)
    catalog = Catalog([model()], {"Q4_K_M": 4.85})
    profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=context)
    plan = build_adaptive_plan(profile, hw, catalog, runtime="llamacpp")
    config = tune(plan.best, plan.hardware, plan.profile, plan.runtime, catalog=catalog).config
    assert (config.gpu_layers > 0) is on_gpu
    assert (config.context_tokens < context) is shrinks
    assert plan.best.model.id == "chosen"
    assert 512 <= config.context_tokens <= plan.profile.context_tokens
    if not on_gpu:
        assert plan.best.estimate.total <= plan.hardware.usable_cpu_bytes
    if shrinks:
        assert any("automatically using" in note for note in plan.notes)
    if vram and not on_gpu:
        assert any("no model fits the GPU" in note for note in plan.notes)


@pytest.mark.parametrize("flavor,uses_gpu", [("vulkan", True), ("cpu", False)])
def test_managed_build_replan_keeps_gpu_only_for_capable_build(flavor, uses_gpu,
                                                             monkeypatch, catalog):
    hw = hardware()
    monkeypatch.setattr("llmsetup.util.proc.which", lambda name: None)
    monkeypatch.setattr(install, "managed_binary", lambda *a: "managed/llama-server.exe")
    monkeypatch.setattr(install, "managed_flavor", lambda *a: flavor)
    plan = build_adaptive_plan(UseCaseProfile(context_tokens=8192), hw, catalog,
                               runtime="llamacpp")
    assert bool(plan.hardware.gpus) is uses_gpu
    config = tune(plan.best, plan.hardware, plan.profile, plan.runtime, catalog=catalog).config
    assert (config.gpu_layers > 0) is uses_gpu
    assert bool(hw.gpus)


def test_windows_nvidia_recovery_upgrades_cpu_build_before_replanning(monkeypatch, catalog):
    hw = hardware()
    installed = []
    monkeypatch.setattr("llmsetup.util.proc.which", lambda name: None)
    monkeypatch.setattr(install, "managed_flavor", lambda *a: "cpu")
    monkeypatch.setattr(install, "install_llamacpp", lambda hw, console, flavor:
                        installed.append((hw, flavor)))
    pipeline = Pipeline(UseCaseProfile(), hw, catalog, Console(quiet=True))
    pipeline._ensure_gpu_runtime(hw)
    assert installed == [(hw, "vulkan")]
    assert installed[0][0].gpus[0].name.endswith("RTX 5060")


def test_busy_vram_warns_without_changing_model_or_tuning(monkeypatch, catalog):
    hw = hardware()
    gpu = hw.gpus[0]
    monkeypatch.setattr("llmsetup.hardware.detect.cpu_probe.memory_bytes", lambda: (32 * GIB, None))
    monkeypatch.setattr("llmsetup.hardware.detect.cpu_probe.is_apple_silicon", lambda: False)
    monkeypatch.setattr("llmsetup.runtimes.registry.scan_runtimes", lambda:
                        ({"llamacpp": True}, []))
    monkeypatch.setattr(install, "managed_binary", lambda *a: None)
    choices = []
    for free in (8 * GIB, GIB, 0):
        monkeypatch.setattr("llmsetup.hardware.detect.detect_gpus", lambda free=free:
                            ([replace(gpu, vram_free_bytes=free)], []))
        detected = detect_hardware()
        plan = build_adaptive_plan(UseCaseProfile(), detected, catalog, runtime="llamacpp")
        config = tune(plan.best, plan.hardware, plan.profile, plan.runtime, catalog=catalog).config
        choices.append((plan.best.model.id, config.quant, config.context_tokens, config.gpu_layers))
        assert detected.memory_budget == int(8 * GIB * 0.9)
        assert any("other apps" in warning for warning in detected.detection_warnings) is (free < 4 * GIB)
    assert choices[0] == choices[1] == choices[2]


def test_insisted_model_tunes_context_and_offload_before_alternative(monkeypatch):
    monkeypatch.setattr(install, "managed_binary", lambda *a: None)
    catalog = Catalog([model(params=16), model("alternative", params=1, quality_tier=10)],
                      {"Q4_K_M": 4.85})
    hw = hardware()
    profile = UseCaseProfile(requested_model="chosen", context_tokens=131072)
    plan = build_adaptive_plan(profile, hw, catalog, runtime="llamacpp")
    config = tune(plan.best, plan.hardware, plan.profile, plan.runtime, catalog=catalog).config
    assert config.model_id == "chosen"
    assert config.quant == "Q4_K_M"
    assert config.context_tokens < profile.context_tokens
    assert config.gpu_layers < config.total_layers
    assert plan.fallbacks[0].model.id == "alternative"


@pytest.mark.parametrize("already_installed", [True, False])
def test_automatic_runtime_selection_installs_vulkan_before_planning(already_installed, monkeypatch):
    from llmsetup.runtimes.registry import get_runtime
    from llmsetup.types import CheckResult, CheckStatus, Severity

    hw = hardware()
    hw.runtime_availability = {"llamacpp": already_installed, "ollama": False}
    flavor = ["cpu" if already_installed else None]
    installed = []
    monkeypatch.setattr("llmsetup.util.proc.which", lambda name: None)
    monkeypatch.setattr(install, "managed_binary", lambda *a:
                        "managed/llama-server.exe" if flavor[0] else None)
    monkeypatch.setattr(install, "managed_flavor", lambda *a: flavor[0])

    def upgrade(hw, console, flavor):
        installed.append(flavor)
    monkeypatch.setattr(install, "install_llamacpp", upgrade)
    llama = get_runtime("llamacpp")
    monkeypatch.setattr(llama, "is_available", lambda: True)
    monkeypatch.setattr(get_runtime("ollama"), "is_available", lambda: False)
    catalog = Catalog([model(params=1)], {"Q4_K_M": 4.85})
    pipeline = Pipeline(UseCaseProfile(), hw, catalog, Console(quiet=True), register=False)

    def ensure(hw):
        # Exercise the real upgrade helper, then make the fake install visible.
        Pipeline._ensure_gpu_runtime(pipeline, hw)
        flavor[0] = installed[-1]
    monkeypatch.setattr(pipeline, "_ensure_gpu_runtime", ensure)
    monkeypatch.setattr(pipeline, "_provision", lambda *a: [])
    monkeypatch.setattr(pipeline, "_launch_and_check", lambda *a:
                        [CheckResult("health", CheckStatus.PASS, Severity.CRITICAL, "ready")])
    result = pipeline.run()
    assert result.succeeded
    assert installed == ["vulkan"]
    assert result.plan.runtime == "llamacpp"
    assert result.config.gpu_layers > 0
    assert result.plan.hardware.gpus[0].name.endswith("RTX 5060")
