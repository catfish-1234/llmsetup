"""Plan explanations and manual commands must match the configuration and lifecycle."""

from dataclasses import replace
from io import StringIO

import pytest
from conftest import make_hardware

from llmsetup.planner.plan import _build_candidate, _plan_notes, build_plan
from llmsetup.report.render import render_markdown, render_terminal
from llmsetup.runtimes.base import RuntimeUnavailable
from llmsetup.runtimes.ollama import OllamaRuntime
from llmsetup.tuning.sampling import sampling_for
from llmsetup.tuning.tuner import _grow_batch_into_headroom, tune
from llmsetup.types import (
    GIB,
    CheckResult,
    CheckStatus,
    RuntimeConfig,
    SetupResult,
    Severity,
    Task,
    UseCaseProfile,
)
from llmsetup.util.log import Console
from llmsetup.util.proc import CommandResult


@pytest.fixture
def laptop_result(catalog):
    hardware = make_hardware(vram_gb=8, ram_gb=31.4, os_name="windows",
                             name="NVIDIA GeForce RTX 5060 Laptop GPU", compute="12.0")
    hardware.gpus[0].vram_free_bytes = 7 * GIB
    hardware.runtime_availability = {"ollama": True, "lmstudio": True}
    profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=32768)
    plan = build_plan(profile, hardware, catalog.restrict_to("qwen2.5-coder-7b"),
                      runtime="ollama")
    tuned = tune(plan.best, hardware, profile, plan.runtime, catalog=catalog)
    return SetupResult(profile, hardware, plan, tuned.config)


def terminal(result):
    stream = StringIO()
    render_terminal(result, Console(stream=stream, color=False))
    return stream.getvalue()


@pytest.mark.parametrize("task", [Task.CODING, Task.AGENT, Task.EXTRACT, Task.AUTOCOMPLETE])
@pytest.mark.parametrize("family", ["qwen2.5", "unpublished-family"])
def test_code_like_tasks_disable_flat_penalty_with_and_without_publisher(catalog, task, family):
    spec = replace(catalog.require("qwen2.5-coder-7b"), tokenizer_family=family)
    sampling = sampling_for(spec, UseCaseProfile(tasks=[task]))
    assert sampling.repeat_penalty == 1.0
    assert "repetition penalty off" in sampling.rationale


def markdown(result):
    return render_markdown(result)


def test_advisory_failure_is_visible_under_success_verdict(laptop_result):
    laptop_result.checks = [
        CheckResult(name="streaming", status=CheckStatus.PASS, severity=Severity.REQUIRED),
        CheckResult(name="tool_calling", status=CheckStatus.FAIL,
                    severity=Severity.ADVISORY,
                    message="model described the tool call in prose"),
    ]
    assert laptop_result.succeeded
    out = markdown(laptop_result)
    assert "all required checks passed; 1 advisory check(s) failed" in out
    assert "every required check passed" not in out
    assert "**tool_calling** (advisory, not blocking): model described the tool call in prose" in out


def test_clean_success_keeps_original_verdict(laptop_result):
    laptop_result.checks = [
        CheckResult(name="streaming", status=CheckStatus.PASS, severity=Severity.REQUIRED),
        CheckResult(name="tool_calling", status=CheckStatus.PASS, severity=Severity.ADVISORY),
    ]
    out = markdown(laptop_result)
    assert "**Verified: yes, every required check passed**" in out
    assert "## Unresolved" not in out


def test_coding_sampler_report_and_modelfile_agree(laptop_result):
    config = laptop_result.config
    sampler = config.advanced.sampler
    output = terminal(laptop_result)
    assert config.repeat_penalty == 1.0
    assert sampler.penalty_last_n == 0
    assert not sampler.dry_enabled and not sampler.xtc_enabled
    assert "repeat penalty 1.0" in output
    assert "token repetition penalty, DRY and XTC off" in output
    assert "PARAMETER repeat_penalty 1.0\n" in OllamaRuntime().modelfile(
        laptop_result.plan.best.model, config)
    tuned = tune(laptop_result.plan.best, laptop_result.hardware,
                 laptop_result.profile, "ollama")
    sampling_reason = next(d.reason for d in tuned.decisions if d.setting == "sampling")
    assert "temperature 0.7" in sampling_reason
    assert "temperature lowered to 0.2 for reproducible output" in sampling_reason
    assert "Plan ready. Nothing was tested" in output


def test_chat_keeps_publisher_repeat_penalty(catalog):
    sampling = sampling_for(catalog.require("qwen2.5-coder-7b"),
                            UseCaseProfile(tasks=[Task.CHAT]))
    assert sampling.repeat_penalty == 1.05


@pytest.mark.parametrize("partial", [False, True])
def test_batch_growth_reason_matches_weight_placement(laptop_result, partial):
    candidate = laptop_result.plan.best
    layers = candidate.model.layers // 2 if partial else candidate.model.layers
    decisions = []
    _, ubatch = _grow_batch_into_headroom(
        candidate, laptop_result.hardware, "ollama", (4 if partial else 7) * GIB,
        8192, "f16", 512, 256, 1, layers, decisions)
    assert ubatch > 256
    assert len(decisions) == 1
    reason = decisions[0].reason
    assert "memory remaining after weight placement" in reason
    assert ("some model weights are held in system RAM" in reason) is partial
    assert "most of this model's weight is not on the card" not in reason


def test_fully_offloaded_tuned_plan_never_claims_weights_are_off_card(laptop_result, catalog):
    tuned = tune(laptop_result.plan.best, laptop_result.hardware,
                 laptop_result.profile, "ollama", catalog=catalog)
    assert tuned.config.gpu_layers == tuned.config.total_layers == 28
    growth = [d.reason for d in tuned.decisions if d.setting == "batch/ubatch"
              and "raised from" in d.reason]
    assert growth
    assert all("not on the card" not in reason and "system RAM" not in reason
               for reason in growth)


@pytest.mark.parametrize("utilisation", [0.8499, 0.85, 0.87, 0.92, 1.0])
@pytest.mark.parametrize("cpu", [False, True])
def test_fit_explanation_and_warning_share_the_85_percent_boundary(catalog, utilisation, cpu):
    hardware = make_hardware(vram_gb=0 if cpu else 8, ram_gb=32)
    profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=8192)
    spec = catalog.require("qwen2.5-coder-7b")
    baseline = _build_candidate(spec, "Q4_K_M", profile, hardware, catalog,
                                "ollama", 100 * GIB, 100 * GIB, 1, False)
    budget = round(baseline.estimate.total / utilisation)
    if cpu:
        hardware.ram_available_bytes = round(budget / 0.75)
    else:
        hardware.gpus[0].vram_bytes = round(budget / 0.9)
    candidate = _build_candidate(spec, "Q4_K_M", profile, hardware, catalog,
                                 "ollama", hardware.memory_budget,
                                 hardware.usable_cpu_bytes, 1, False)
    assert candidate is not None
    # Ensure fit is one of the reasons under test, independent of ranking changes.
    candidate.score_breakdown = {"fit": 1.0}
    note = _plan_notes([candidate], profile, hardware, 1)[0]
    warned = any("of usable" in warning for warning in candidate.warnings)
    assert warned is (utilisation >= 0.85)
    assert ("limited headroom" in note) is warned
    assert ("without running at the edge" in note) is not warned


@pytest.mark.parametrize("model_id,tag", [
    ("qwen2.5-coder-7b", "qwen2.5-coder:7b"),
    ("qwen2.5-coder-14b", "qwen2.5-coder:14b"),
    ("qwen2.5-coder-32b", "qwen2.5-coder:32b"),
    ("qwen2.5-0.5b", "qwen2.5:0.5b"),
    ("qwen2.5-1.5b", "qwen2.5:1.5b"),
    ("qwen2.5-3b", "qwen2.5:3b"),
    ("qwen2.5-7b", "qwen2.5:7b"),
    ("qwen2.5-14b", "qwen2.5:14b"),
    ("qwen2.5-32b", "qwen2.5:32b"),
    ("qwen3-30b-a3b", "qwen3:30b"),
    ("qwen3-235b-a22b", "qwen3:235b"),
    ("qwen3-coder-next", "qwen3-coder-next"),
    ("deepseek-r1-distill-qwen-1.5b", "deepseek-r1:1.5b"),
])
def test_catalog_official_ollama_tags(catalog, model_id, tag):
    assert catalog.require(model_id).ollama_name == tag


@pytest.mark.parametrize("served_name", [None, "catalog-id", "uncreated-tag"])
def test_unprovisioned_commands_only_use_official_tags(catalog, served_name):
    runtime = OllamaRuntime()
    for spec in catalog.models:
        config = RuntimeConfig(runtime="ollama", model_id=spec.id,
                               quant="Q4_K_M", context_tokens=8192,
                               served_name=spec.id if served_name == "catalog-id" else served_name)
        if spec.ollama_name:
            assert runtime.launch_command(spec, config) == ["ollama", "run", spec.ollama_name]
        else:
            with pytest.raises(RuntimeUnavailable, match="has not been created"):
                runtime.launch_command(spec, config)


@pytest.mark.parametrize("official", [False, True])
def test_dry_run_reports_tuned_model_prerequisite(laptop_result, official):
    if not official:
        laptop_result.plan.best.model = replace(laptop_result.plan.best.model, ollama_name=None)
    assert laptop_result.config.served_name is None
    for output in (terminal(laptop_result), render_markdown(laptop_result)):
        assert "has not been created" in output
        assert "without `--dry-run` to create it" in output
        assert "ollama run llmsetup-" not in output
        assert "ollama run qwen2.5-coder-7b" not in output
        assert ("ollama run qwen2.5-coder:7b" in output) is official
        assert ("without these tuned settings" in output) is official
        if not official:
            assert "ollama run" not in output


@pytest.mark.parametrize("created", [False, True])
def test_ollama_only_advertises_tuned_tag_after_successful_create(
        laptop_result, monkeypatch, tmp_path, created):
    runtime = OllamaRuntime()
    spec, config = laptop_result.plan.best.model, laptop_result.config
    monkeypatch.setattr(runtime, "_ensure_server", lambda *args: None)
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *args: gguf)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return CommandResult(0 if created else 1, "", "")

    monkeypatch.setattr("llmsetup.runtimes.ollama.run", run)
    console = Console(stream=StringIO(), color=False)
    if created:
        runtime.ensure_model(spec, config, console)
        tag = runtime.derived_name(spec)
        assert config.served_name == tag
        assert runtime.launch_command(spec, config) == ["ollama", "run", tag]
        assert runtime.start(spec, config, console).model == tag
        for output in (terminal(laptop_result), render_markdown(laptop_result)):
            assert f"ollama run {tag}" in output
            assert "has not been created" not in output
    else:
        with pytest.raises(RuntimeUnavailable, match="failed"):
            runtime.ensure_model(spec, config, console)
        assert config.served_name is None
        assert runtime.launch_command(spec, config) == ["ollama", "run", spec.ollama_name]
    assert commands[0][:3] == ["ollama", "create", runtime.derived_name(spec)]

def test_zero_gpu_layers_reports_cpu_only_not_partial(laptop_result):
    laptop_result.config.gpu_layers = 0
    output = terminal(laptop_result)
    assert "CPU only" in output
    assert "partial. CPU is the bottleneck" not in output


def test_partial_offload_label_kept_when_some_layers_offload(laptop_result):
    config = laptop_result.config
    config.gpu_layers = max(1, config.total_layers - 1)
    assert "partial. CPU is the bottleneck" in terminal(laptop_result)


def test_unprovisionable_models_are_skipped_with_a_note(catalog):
    """A model that failed to provision earlier in the run must not be the
    next pick: retrying the same broken download loops forever."""
    hardware = make_hardware(vram_gb=24, ram_gb=64)
    profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=8192)
    excluded = {catalog.matching(profile)[0].id}
    plan = build_plan(profile, hardware, catalog, runtime="ollama",
                      exclude_models=excluded)
    assert all(c.model.id not in excluded for c in plan.candidates)
    assert any("skipping" in n and "failed earlier" in n for n in plan.notes)


@pytest.mark.parametrize("status,severity", [
    (CheckStatus.FAIL, Severity.ADVISORY),
    (CheckStatus.WARN, Severity.ADVISORY),
    (CheckStatus.WARN, Severity.REQUIRED),
])
def test_completion_headline_names_check_issues(laptop_result, status, severity):
    laptop_result.checks = [
        CheckResult("health", CheckStatus.PASS, Severity.CRITICAL),
        CheckResult("tool_calling", status, severity),
    ]
    assert laptop_result.succeeded  # Exit status and advisory semantics stay intact.
    headline = next(line for line in terminal(laptop_result).splitlines()
                    if "Setup complete" in line)
    assert "Setup complete and verified" not in headline
    assert f"tool_calling: {status.value.upper()}" in headline
    assert ("(advisory)" in headline) is (severity is Severity.ADVISORY)


def test_completion_headline_names_every_advisory_issue(laptop_result):
    laptop_result.checks = [
        CheckResult("tool_calling", CheckStatus.FAIL),
        CheckResult("throughput", CheckStatus.WARN),
    ]
    headline = next(line for line in terminal(laptop_result).splitlines()
                    if "Setup complete" in line)
    assert "tool_calling: FAIL (advisory)" in headline
    assert "throughput: WARN (advisory)" in headline


@pytest.mark.parametrize("checks,headline", [
    ([CheckResult("health", CheckStatus.PASS, Severity.CRITICAL)],
     "Setup complete and verified"),
    ([CheckResult("health", CheckStatus.PASS), CheckResult("vision", CheckStatus.SKIP)],
     "Setup complete and verified"),
    ([], "Plan ready. Nothing was tested"),
    ([CheckResult("vision", CheckStatus.SKIP)], "Plan ready. Nothing was tested"),
    ([CheckResult("health", CheckStatus.FAIL, Severity.CRITICAL)],
     "Setup finished with unresolved problems"),
])
def test_existing_completion_headlines_stay_intact(laptop_result, checks, headline):
    laptop_result.checks = checks
    assert f"  {headline}\n" in terminal(laptop_result)


def test_reports_distinguish_planned_offload_from_observed_residency(laptop_result):
    laptop_result.checks = [CheckResult("health", CheckStatus.PASS)]
    for output in (terminal(laptop_result), render_markdown(laptop_result)):
        assert "Planned GPU layers" in output
        assert "not observed" in output
        assert "fully offloaded" not in output
