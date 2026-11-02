"""A repaired quantization must refer to weights for that quantization."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import make_hardware

from llmsetup.catalog.catalog import Catalog
from llmsetup.cli import build_parser, main
from llmsetup.pipeline import Pipeline
from llmsetup.planner.plan import build_plan
from llmsetup.repair.strategies import Outcome, RepairStrategy
from llmsetup.runtimes.base import Endpoint, RuntimeUnavailable
from llmsetup.runtimes.gguf_store import CachedModelUnavailable, require_cached, store_path
from llmsetup.runtimes.llamacpp import LlamaCppRuntime
from llmsetup.runtimes.ollama import OllamaRuntime
from llmsetup.tuning.profiles import SpecType
from llmsetup.types import CheckResult, CheckStatus, RuntimeConfig, Severity, UseCaseProfile
from llmsetup.util.log import Console
from llmsetup.util.proc import CommandResult


class RecordingRuntime:
    uses_gguf = False

    def __init__(self):
        self.provisioned = []

    def is_available(self):
        return True

    def ensure_model(self, spec, config, console):
        self.provisioned.append(config.quant)
        config.model_path = f"/synthetic/{spec.id}/{config.quant}.gguf"
        return config


def pipeline(catalog, monkeypatch):
    runtime = RecordingRuntime()
    launched = []
    monkeypatch.setattr("llmsetup.pipeline.get_runtime", lambda name: runtime)
    instance = Pipeline(UseCaseProfile(), make_hardware(),
                        catalog.restrict_to("qwen3-8b"), Console(quiet=True),
                        runtime_name="fake", register=False)

    def launch(spec, config, names):
        launched.append((config.quant, config.model_path))
        return []

    monkeypatch.setattr(instance, "_launch_and_check", launch)
    result = instance.run()
    return instance, result.config, runtime, launched


def test_first_quantization_repair_provisions_matching_weights(catalog, monkeypatch):
    instance, config, runtime, launched = pipeline(catalog, monkeypatch)
    original = config.quant
    config.quant = "Q2_K" if original != "Q2_K" else "Q4_K_M"
    instance._relaunch(config, None)
    assert runtime.provisioned == [original, config.quant]
    assert launched[-1] == (config.quant, f"/synthetic/{config.model_id}/{config.quant}.gguf")


def test_setting_only_repair_reuses_the_provisioned_weights(catalog, monkeypatch):
    instance, config, runtime, launched = pipeline(catalog, monkeypatch)
    config.ubatch_size = max(1, config.ubatch_size // 2)
    instance._relaunch(config, None)
    assert runtime.provisioned == [config.quant]
    assert len(launched) == 2


def retain(spec, quant="Q4_K_M", filename="model.gguf"):
    directory = store_path(spec, quant)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_bytes(b"GGUF")
    return path


@pytest.fixture
def no_fetch(monkeypatch):
    calls = []
    for target in (
        "llmsetup.runtimes.ollama.fetch_gguf",
        "llmsetup.runtimes.llamacpp.fetch_gguf",
        "llmsetup.runtimes.llamacpp.fetch_projector",
        "llmsetup.runtimes.gguf_store.fetch_gguf",
        "llmsetup.runtimes.gguf_store.list_repo_files",
        "llmsetup.runtimes.gguf_store.download_file",
    ):
        call = Mock(side_effect=AssertionError(f"unexpected weight/network call: {target}"))
        monkeypatch.setattr(target, call)
        calls.append(call)
    yield calls
    for call in calls:
        call.assert_not_called()


@pytest.fixture
def setup(catalog, monkeypatch, no_fetch):
    spec = replace(catalog.require("qwen3-8b"), recommended_quants=["Q4_K_M"])
    selected = Catalog([spec], catalog.quant_bpw)
    hardware = make_hardware(vram_gb=24)
    hardware.runtime_availability = {"ollama": True}
    runtime = OllamaRuntime()
    monkeypatch.setattr(runtime, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "_ensure_server", Mock())
    create = Mock(return_value=CommandResult(0, "", ""))
    monkeypatch.setattr("llmsetup.runtimes.ollama.run", create)
    start = Mock(return_value=Endpoint("http://fixture.invalid", spec.id, "ollama"))
    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr("llmsetup.pipeline.get_runtime", lambda name: runtime)
    instance = Pipeline(UseCaseProfile(context_tokens=8192), hardware, selected,
                        Console(quiet=True), runtime_name="ollama", register=False,
                        cached_only=True)
    monkeypatch.setattr(instance, "_correct_from_gguf", lambda *a: None)
    monkeypatch.setattr("llmsetup.pipeline.Harness", lambda **kw: SimpleNamespace(
        run=lambda names: [CheckResult("health", CheckStatus.PASS, Severity.CRITICAL)]))
    return instance, runtime, spec, create, start


def test_uncached_initial_plan_stops_before_provision_or_recovery(setup, monkeypatch):
    instance, runtime, spec, create, start = setup
    ensure = Mock(wraps=runtime.ensure_model)
    monkeypatch.setattr(runtime, "ensure_model", ensure)
    result = instance.run()
    assert not result.succeeded
    assert result.checks[0].blocking
    assert result.checks[0].name == "model_cache"
    assert f"{spec.id}:Q4_K_M" in result.checks[0].message
    assert "--cached-only" in result.checks[0].message
    assert result.repairs == []
    ensure.assert_not_called()
    runtime._ensure_server.assert_not_called()
    create.assert_not_called()
    start.assert_not_called()


def test_cached_initial_plan_allows_local_ollama_import(setup):
    instance, _runtime, spec, create, start = setup
    path = retain(spec)
    result = instance.run()
    assert result.succeeded
    assert result.config.model_path == str(path)
    assert result.config.extra["from_gguf"]
    create.assert_called_once()
    command = create.call_args.args[0]
    assert command[:2] == ["ollama", "create"]
    contents = Path(command[-1]).read_text()
    assert f"FROM {path}\n" in contents
    assert f"FROM {spec.ollama_name}\n" not in contents
    start.assert_called_once()


def test_flag_absent_keeps_existing_fetch_path(setup, monkeypatch, tmp_path):
    instance, _runtime, spec, create, _start = setup
    instance.cached_only = False
    path = tmp_path / "downloaded.gguf"
    path.write_bytes(b"GGUF")
    fetch = Mock(return_value=path)
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", fetch)
    result = instance.run()
    assert result.succeeded
    fetch.assert_called_once_with(spec, "Q4_K_M", instance.console)
    create.assert_called_once()


@pytest.mark.parametrize("mismatch", ["quant", "model", "revision", "incomplete-shards"])
def test_cache_identity_must_match_exactly(setup, mismatch):
    instance, _runtime, spec, create, start = setup
    if mismatch == "quant":
        retain(spec, "Q5_K_M")
    elif mismatch == "model":
        retain(replace(spec, id=spec.id + "-other"))
    elif mismatch == "revision":
        retain(replace(spec, gguf_revision="a" * 40))
        spec.gguf_revision = "b" * 40
    else:
        retain(spec, filename="model-00001-of-00002.gguf")
    result = instance.run()
    assert not result.succeeded
    assert f"{spec.id}:Q4_K_M" in result.checks[0].message
    create.assert_not_called()
    start.assert_not_called()


@pytest.mark.parametrize("runtime_type", [OllamaRuntime, LlamaCppRuntime])
def test_adapter_guard_rejects_stale_path_before_any_fetch(catalog, monkeypatch,
                                                         no_fetch, runtime_type):
    spec = catalog.require("qwen3-8b")
    stale = retain(spec, "Q5_K_M")
    runtime = runtime_type()
    server = Mock()
    monkeypatch.setattr(runtime, "_ensure_server", server, raising=False)
    config = RuntimeConfig(runtime.name, spec.id, "Q4_K_M", 8192,
                           model_path=str(stale), extra={"cached_only": True})
    with pytest.raises(CachedModelUnavailable, match=f"{spec.id}:Q4_K_M"):
        runtime.ensure_model(spec, config, Console(quiet=True))
    server.assert_not_called()


def test_cached_llamacpp_uses_exact_local_path(catalog, no_fetch):
    spec = catalog.require("qwen3-8b")
    path = retain(spec)
    config = RuntimeConfig("llamacpp", spec.id, "Q4_K_M", 8192,
                           extra={"cached_only": True})
    assert LlamaCppRuntime().ensure_model(spec, config, Console(quiet=True)) is config
    assert config.model_path == str(path)


def test_cached_shards_import_together(setup):
    instance, _runtime, spec, create, _start = setup
    first = retain(spec, filename="model-00001-of-00002.gguf")
    retain(spec, filename="model-00002-of-00002.gguf")
    assert instance.run().succeeded
    contents = Path(create.call_args.args[0][-1]).read_text()
    assert f"FROM {first.parent / 'model-?????-of-00002.gguf'}\n" in contents


@pytest.mark.parametrize("change", ["quant", "model"])
def test_repair_reprovision_cannot_reuse_stale_weights(setup, catalog, monkeypatch, change):
    instance, runtime, spec, create, start = setup
    retain(spec)
    result = instance.run()
    config = result.config.copy()
    if change == "quant":
        config.quant = "Q5_K_M"
    else:
        other = catalog.require("qwen3-4b")
        instance.catalog = Catalog([spec, other], catalog.quant_bpw)
        config.model_id = other.id
    # Candidate-switch tuning drops extra keys; the pipeline must restore policy.
    config.extra = {}
    ensure = Mock(wraps=runtime.ensure_model)
    monkeypatch.setattr(runtime, "ensure_model", ensure)
    failures = instance._relaunch(config, None)
    assert failures[0].blocking
    assert f"{config.model_id}:{config.quant}" in failures[0].message
    ensure.assert_not_called()
    assert create.call_count == start.call_count == 1


def test_real_repair_loop_fallback_stops_on_uncached_candidate(setup, catalog, monkeypatch):
    instance, _runtime, spec, create, start = setup
    retain(spec)
    other = catalog.require("qwen3-4b")
    instance.catalog = Catalog([spec, other], catalog.quant_bpw)
    fallback = build_plan(instance.profile, instance.hardware, catalog.restrict_to(other.id),
                          runtime="ollama").best
    monkeypatch.setattr("llmsetup.pipeline.Harness", lambda **kw: SimpleNamespace(
        run=lambda names: [CheckResult("health", CheckStatus.FAIL, Severity.CRITICAL,
                                       failure_signature="fixture_failure")]))
    switch = RepairStrategy("fixture_fallback", ["fixture_failure"], "high",
                            lambda state: Outcome("fallback", "high", switch_to=fallback))
    monkeypatch.setattr("llmsetup.repair.loop.strategies_for",
                        lambda signature: [switch] if signature == "fixture_failure" else [])
    result = instance.run()
    assert not result.succeeded
    assert result.checks[0].name == "model_cache"
    assert other.id in result.checks[0].message
    assert len(result.repairs) == 1
    assert result.repairs[0].strategy == "fixture_fallback"
    assert create.call_count == start.call_count == 1


def test_retry_rechecks_cache_if_weights_disappear(setup, monkeypatch):
    instance, runtime, spec, create, start = setup
    path = retain(spec)

    def fail_after_removal(*args):
        path.unlink()
        raise RuntimeUnavailable("fixture import failure")

    ensure = Mock(side_effect=fail_after_removal)
    monkeypatch.setattr(runtime, "ensure_model", ensure)
    result = instance.run()
    assert not result.succeeded
    assert result.checks[0].name == "model_cache"
    ensure.assert_called_once()
    create.assert_not_called()
    start.assert_not_called()


@pytest.mark.parametrize("cached", [False, True])
def test_draft_model_respects_cached_only(setup, cached):
    instance, runtime, spec, _create, _start = setup
    draft = SimpleNamespace(needs_draft_model=True, draft_model_id=spec.id,
                            draft_quant="Q4_K_M", draft_path=None, spec_type=SpecType.DRAFT)
    config = RuntimeConfig("ollama", spec.id, "Q4_K_M", 8192,
                           advanced=SimpleNamespace(speculative=draft))
    if cached:
        path = retain(spec)
        instance._ensure_draft_model(runtime, config)
        assert draft.draft_path == str(path)
    else:
        with pytest.raises(CachedModelUnavailable, match=spec.id):
            instance._ensure_draft_model(runtime, config)


def test_missing_projector_is_blocked(catalog, no_fetch):
    spec = replace(catalog.require("qwen3-8b"), supports_vision=True)
    retain(spec)
    config = RuntimeConfig("llamacpp", spec.id, "Q4_K_M", 8192,
                           extra={"cached_only": True})
    with pytest.raises(CachedModelUnavailable, match="vision projector"):
        LlamaCppRuntime().ensure_model(spec, config, Console(quiet=True))
    projector = retain(spec, filename="mmproj-f16.gguf")
    LlamaCppRuntime().ensure_model(spec, config, Console(quiet=True))
    assert config.mmproj_path == str(projector)


def test_embedding_retry_cannot_fetch_an_uncached_model(setup, monkeypatch):
    instance, runtime, spec, create, start = setup
    monkeypatch.setattr("llmsetup.pipeline.plan_embedding_model", lambda *a: spec)
    monkeypatch.setattr(instance, "_free_port", lambda: 11500)
    ensure = Mock(wraps=runtime.ensure_model)
    monkeypatch.setattr(runtime, "ensure_model", ensure)
    with pytest.raises(CachedModelUnavailable, match=spec.id):
        instance._setup_embedding(runtime)
    ensure.assert_not_called()
    create.assert_not_called()
    start.assert_not_called()


def test_cached_only_never_uses_an_unverified_runtime_cache(setup, monkeypatch):
    instance, _runtime, spec, _create, _start = setup
    retain(spec)
    instance._runtime = SimpleNamespace(name="vllm", ensure_model=Mock())
    config = RuntimeConfig("vllm", spec.id, "Q4_K_M", 8192)
    failures = instance._provision(spec, config)
    assert failures[0].blocking
    assert "cannot verify retained weights" in failures[0].message
    instance._runtime.ensure_model.assert_not_called()


def test_cached_only_prevents_implicit_official_tag_pull(catalog, no_fetch):
    spec = catalog.require("qwen3-8b")
    config = RuntimeConfig("ollama", spec.id, "Q4_K_M", 8192,
                           extra={"cached_only": True})
    with pytest.raises(CachedModelUnavailable, match=spec.id):
        OllamaRuntime().launch_command(spec, config)


def test_cache_lookup_requires_complete_exact_identity(catalog):
    spec = catalog.require("qwen3-8b")
    path = retain(spec)
    assert require_cached(spec, "Q4_K_M") == path
    with pytest.raises(CachedModelUnavailable, match="Q5_K_M"):
        require_cached(spec, "Q5_K_M")


def test_cli_cached_only_is_separate_from_catalog_offline_flag():
    parser = build_parser()
    args = parser.parse_args(["setup", "--cached-only"])
    assert args.cached_only and args.offline_only is None
    args = parser.parse_args(["setup", "--offline"])
    assert args.offline_only and not args.cached_only
    assert parser.parse_args(["doctor", "--cached-only"]).cached_only


@pytest.mark.parametrize("command", ["setup", "doctor"])
def test_cli_passes_guard_and_reports_a_blocking_cache_error(setup, monkeypatch,
                                                           tmp_path, capsys, command):
    instance, _runtime, spec, create, start = setup
    monkeypatch.setattr("llmsetup.cli.detect_hardware", lambda **kw: instance.hardware)
    monkeypatch.setattr("llmsetup.cli.load_catalog", lambda **kw: instance.catalog)
    monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")
    args = [command, "--cached-only"]
    if command == "setup":
        args += ["--runtime", "ollama", "--task", "chat", "--model", spec.id,
                 "--context", "8192", "--yes", "--offline", "--no-register"]
    else:
        path = tmp_path / "saved.json"
        path.write_text(json.dumps({"profile": {"context_tokens": 8192},
                                    "config": {"model_id": spec.id}, "runtime": "ollama"}))
        args += ["--profile", str(path)]
    assert main(args) == 1
    output = capsys.readouterr()
    assert "--cached-only" in output.out + output.err
    assert f"{spec.id}:Q4_K_M" in output.out + output.err
    create.assert_not_called()
    start.assert_not_called()
