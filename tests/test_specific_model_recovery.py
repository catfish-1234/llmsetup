"""A retained model stays fixed through repairs, runtime changes, and exhaustion."""

from dataclasses import replace

import pytest
from conftest import make_hardware

from llmsetup.pipeline import Pipeline
from llmsetup.planner.plan import build_plan, plan_embedding_model
from llmsetup.repair.loop import RepairLoop
from llmsetup.repair.strategies import (
    RepairState,
    _drop_quant_tier,
    _quantize_kv,
    _quantize_kv_lossy,
    _switch_model,
    _use_cached_model,
)
from llmsetup.runtimes.registry import _REGISTRY, all_runtimes
from llmsetup.tuning import tune
from llmsetup.types import CheckResult, CheckStatus, Severity, UseCaseProfile
from llmsetup.util.errors import PlanningError, ProvisionError
from llmsetup.util.log import Console
from llmsetup.verify import signatures as sig


def failed(signature):
    return [CheckResult(name="health", status=CheckStatus.FAIL, severity=Severity.CRITICAL,
                        message="fixture failure", failure_signature=signature)]


def passed():
    return [CheckResult(name="health", status=CheckStatus.PASS, severity=Severity.CRITICAL,
                        message="fixture success")]


def state_for(catalog):
    hardware = make_hardware()
    profile = UseCaseProfile(requested_model="qwen3-8b")
    plan = build_plan(profile, hardware, catalog, runtime="llamacpp")
    return RepairState(config=tune(plan.best, hardware, profile, "llamacpp").config,
                       candidate=plan.best, hardware=hardware, profile=profile,
                       catalog=catalog, fallbacks=plan.fallbacks)


def test_requested_model_cannot_be_swapped_by_ordinary_repairs(catalog):
    state = state_for(catalog)
    fallbacks = list(state.fallbacks)
    assert fallbacks
    assert _switch_model(state) is None
    assert state.fallbacks == fallbacks
    loop = RepairLoop(state, lambda config, names: failed(sig.OUT_OF_MEMORY),
                      Console(quiet=True), max_repairs=20)
    results, journal = loop.run(failed(sig.OUT_OF_MEMORY))
    assert results[0].blocking
    assert state.config.model_id == "qwen3-8b"
    assert any(action.strategy == "reduce_context" for action in journal)
    assert any(action.strategy == "drop_quant_tier" for action in journal)
    assert not any(action.strategy == "switch_model" for action in journal)


def test_download_repair_uses_only_requested_models_cached_weights(catalog, monkeypatch):
    state = state_for(catalog)
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.find_local",
                        lambda spec, quant: spec.id != "qwen3-8b")
    assert _use_cached_model(state) is None
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.find_local",
                        lambda spec, quant: spec.id == "qwen3-8b" and quant == "Q4_K_M")
    outcome = _use_cached_model(state)
    assert outcome is not None
    assert outcome.switch_to.model.id == "qwen3-8b"
    assert outcome.switch_to.quant == "Q4_K_M"


def test_requested_quantization_repairs_can_reach_each_smaller_tier(catalog):
    state = state_for(catalog)
    state.config.quant = "Q8_0"
    loop = RepairLoop(state, lambda config, names: failed(sig.OUT_OF_MEMORY),
                      Console(quiet=True), max_repairs=30)
    _, journal = loop.run(failed(sig.OUT_OF_MEMORY))
    quants = [action.after["quant"] for action in journal if action.strategy == "drop_quant_tier"]
    assert len(quants) > 1
    assert state.config.quant == min(state.candidate.model.recommended_quants, key=catalog.bpw)
    assert state.config.model_id == "qwen3-8b"


@pytest.mark.parametrize("runtime", ["vllm", "mlx"])
def test_native_weights_do_not_claim_to_apply_gguf_quantization(catalog, runtime):
    state = state_for(catalog)
    state.config.runtime = runtime
    state.config.quant = "Q8_0"
    assert _drop_quant_tier(state) is None
    assert state.config.quant == "Q8_0"


@pytest.mark.parametrize("runtime", ["vllm", "mlx"])
def test_native_cache_repairs_keep_supported_precision(catalog, runtime):
    state = state_for(catalog)
    state.config.runtime = runtime
    state.config.context_tokens = 1024
    state.config.kv_cache_type = "q8_0"
    assert _quantize_kv_lossy(state) is None
    assert state.config.kv_cache_type == "q8_0"
    if runtime == "mlx":
        state.config.kv_cache_type = "f16"
        assert _quantize_kv(state) is None
        assert state.config.kv_cache_type == "f16"


class FixtureRuntime:
    uses_gguf = False

    def __init__(self, name, *, available=True):
        self.name = self.label = name
        self.available = available
        self.provisions = []

    def is_available(self):
        return self.available

    def install_hint(self):
        return "fixture runtime is unavailable"

    def ensure_model(self, spec, config, console):
        self.provisions.append((spec.id, config.quant))
        config.model_path = f"/fixture/{spec.id}/{config.quant}"
        return config


def wire(monkeypatch, *runtimes):
    all_runtimes()
    for runtime in runtimes:
        monkeypatch.setitem(_REGISTRY, runtime.name, runtime)
    hardware = make_hardware()
    hardware.runtime_availability = {runtime.name: runtime.available for runtime in runtimes}
    return hardware


def test_requested_model_stays_fixed_after_repair_budget_is_exhausted(catalog, monkeypatch):
    runtime = FixtureRuntime("llamacpp")
    hardware = wire(monkeypatch, runtime)
    profile = UseCaseProfile(requested_model="qwen3-8b")
    instance = Pipeline(profile, hardware, catalog, Console(quiet=True),
                        runtime_name="llamacpp", max_repairs=3, register=False)
    attempts = []

    def launch(spec, config, names):
        attempts.append((spec.id, config.runtime, config.gpu_layers))
        return failed(sig.OUT_OF_MEMORY) if spec.id == profile.requested_model else passed()

    monkeypatch.setattr(instance, "_launch_and_check", launch)
    result = instance.run()
    assert not result.succeeded
    assert result.config.model_id == profile.requested_model
    assert all(model == profile.requested_model for model, _, _ in attempts)
    assert any(layers == 0 for _, _, layers in attempts)
    actions = [action for action in result.repairs
               if action.strategy not in ("runtime_fallback", "cpu_runtime", "portable_cpu_runtime")]
    assert len(actions) == 3
    assert not any(action.strategy == "switch_model" for action in actions)
    assert "was kept fixed" in result.checks[0].detail


def test_no_applicable_repairs_reports_requested_model_failure(catalog, monkeypatch):
    runtime = FixtureRuntime("ollama")
    hardware = wire(monkeypatch, runtime)
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="ollama", register=False)
    monkeypatch.setattr(instance, "_launch_and_check",
                        lambda spec, config, names: failed(sig.PROVISION_FAILED)
                        if spec.id == "qwen3-8b" else passed())
    result = instance.run()
    assert not result.succeeded
    assert result.config.model_id == "qwen3-8b"
    assert not any(action.strategy == "switch_model" for action in result.repairs)
    assert "choose a different model explicitly" in result.checks[0].detail


def test_cached_alternative_never_overrides_requested_model(catalog, monkeypatch):
    runtime = FixtureRuntime("llamacpp")
    runtime.uses_gguf = True
    hardware = wire(monkeypatch, runtime)
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="llamacpp", register=False)
    provisions = []

    def ensure(spec, config, console):
        provisions.append(spec.id)
        raise ProvisionError("network blocked")

    monkeypatch.setattr(runtime, "ensure_model", ensure)
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.find_local",
                        lambda spec, quant: spec.id == "qwen3-4b")
    monkeypatch.setattr(instance, "_launch_and_check", lambda spec, config, names: passed())
    result = instance.run()
    assert not result.succeeded
    assert result.config.model_id == "qwen3-8b"
    assert len(provisions) >= 4
    assert set(provisions) == {"qwen3-8b"}
    assert not any(action.strategy == "use_cached_model" for action in result.repairs)


def test_requested_model_recovery_keeps_companion_embedding_models(catalog, monkeypatch):
    runtime = FixtureRuntime("llamacpp")
    hardware = wire(monkeypatch, runtime)
    profile = UseCaseProfile(requested_model="qwen3-8b", needs_embeddings=True)
    instance = Pipeline(profile, hardware, catalog, Console(quiet=True),
                        runtime_name="llamacpp", register=False)
    monkeypatch.setattr(instance, "_launch_and_check",
                        lambda spec, config, names: failed(sig.OUT_OF_MEMORY)
                        if config.gpu_layers else passed())

    def embedding(runtime):
        spec = plan_embedding_model(instance.profile, instance.hardware, instance.catalog)
        return {"ok": spec is not None, "model": spec.id if spec else None}

    monkeypatch.setattr(instance, "_setup_embedding", embedding)
    result = instance.run()
    assert result.succeeded
    assert result.config.model_id == profile.requested_model
    assert result.config.gpu_layers == 0
    assert result.embedding["ok"]
    assert catalog.require(result.embedding["model"]).is_embedding
    assert instance.catalog is catalog


def test_runtime_planning_failure_retries_requested_model_with_compatible_weights(catalog, monkeypatch):
    unavailable = FixtureRuntime("vllm")
    compatible = FixtureRuntime("llamacpp")
    hardware = wire(monkeypatch, unavailable, compatible)
    # Native full-precision weights cannot fit, but a GGUF quantization can.
    limited = make_hardware(vram_gb=4, ram_gb=12)
    hardware = replace(hardware, ram_bytes=limited.ram_bytes, gpus=limited.gpus)
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="vllm", register=False)
    monkeypatch.setattr(instance, "_launch_and_check", lambda spec, config, names: passed())
    result = instance.run()
    assert result.succeeded
    assert result.config.model_id == "qwen3-8b"
    assert result.config.runtime == "llamacpp"
    assert not unavailable.provisions
    assert compatible.provisions
    assert not any(action.strategy == "switch_model" for action in result.repairs)


def test_runtime_unavailability_does_not_release_requested_model(catalog, monkeypatch):
    unavailable = FixtureRuntime("ollama", available=False)
    compatible = FixtureRuntime("llamacpp")
    hardware = wire(monkeypatch, unavailable, compatible)
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="ollama", register=False)
    monkeypatch.setattr(instance, "_launch_and_check", lambda spec, config, names: passed())
    result = instance.run()
    assert result.succeeded
    assert result.config.model_id == "qwen3-8b"
    assert result.config.runtime == "llamacpp"
    assert compatible.provisions[0][0] == "qwen3-8b"


def test_no_repair_never_switches_the_requested_model(catalog, monkeypatch):
    runtime = FixtureRuntime("llamacpp")
    hardware = wire(monkeypatch, runtime)
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="llamacpp",
                        max_repairs=0, register=False)
    monkeypatch.setattr(instance, "_launch_and_check", lambda spec, config, names: failed(sig.OUT_OF_MEMORY))
    result = instance.run()
    assert not result.succeeded
    assert result.config.model_id == "qwen3-8b"
    assert result.repairs == []
    assert len(runtime.provisions) == 1


def test_no_repair_keeps_incompatible_runtime_planning_failure(catalog, monkeypatch):
    runtime = FixtureRuntime("vllm")
    hardware = replace(wire(monkeypatch, runtime), gpus=[])
    instance = Pipeline(UseCaseProfile(requested_model="qwen3-8b"), hardware,
                        catalog, Console(quiet=True), runtime_name="vllm",
                        max_repairs=0, register=False)
    with pytest.raises(PlanningError):
        instance.run()
