"""Offline CLI flows with real planning, verification, recovery, and reporting."""

import json
import socket
from pathlib import Path

import pytest
from conftest import make_hardware

from llmsetup.cli import main
from llmsetup.pipeline import Pipeline
from llmsetup.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmsetup.runtimes.registry import _REGISTRY, all_runtimes
from llmsetup.types import CheckStatus
from llmsetup.util.errors import ProvisionError
from llmsetup.util.fs import model_store
from llmsetup.util.http import HttpResponse


class Engine(Runtime):
    uses_gguf = False

    def __init__(self, name, *, available=True, provision_errors=(), launch_errors=()):
        self.name = self.label = name
        self.available = available
        self.provision_errors = list(provision_errors)
        self.launch_errors = list(launch_errors)
        self.provisions = self.launches = self.stops = 0
        self.configs = []
        self.corrupt = False
        self.cpu_required = False
        self.max_context = None
        self.gpu_error = "CUDA out of memory"

    def is_available(self):
        return self.available

    def ensure_model(self, spec, config, console):
        self.provisions += 1
        if self.provision_errors:
            raise self.provision_errors.pop(0)
        path = model_store() / spec.id / config.quant / "fixture.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"corrupt" if self.corrupt else b"fixture weights")
            self.corrupt = False
        config.model_path = str(path)
        return config

    def start(self, spec, config, console):
        self.launches += 1
        self.configs.append(config.to_dict())
        if self.launch_errors:
            raise self.launch_errors.pop(0)
        if Path(config.model_path).read_bytes() == b"corrupt":
            raise RuntimeUnavailable("invalid GGUF magic")
        if self.cpu_required and config.gpu_layers:
            raise RuntimeUnavailable(self.gpu_error)
        if self.max_context and config.context_tokens > self.max_context:
            raise RuntimeUnavailable("maximum context length exceeded")
        return Endpoint("http://fixture.invalid", spec.id, self.name)

    def stop(self, endpoint):
        self.stops += 1

    def launch_command(self, spec, config):
        return ["fixture-engine", config.model_path]


@pytest.fixture
def flow(monkeypatch):
    monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: pytest.fail("unexpected network"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("unexpected question"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    hardware = make_hardware(vram_gb=24, ram_gb=64)
    monkeypatch.setattr("llmsetup.cli.detect_hardware", lambda **kw: hardware)

    def request(url, **kw):
        if url.endswith("/models"):
            data = {"data": [{"id": "qwen3-8b"}]}
        else:
            prompt = kw["json_body"]["messages"][0]["content"]
            data = {"choices": [{"message": {"content": "Paris" if "France" in prompt else "42"},
                                 "finish_reason": "stop"}]}
        return HttpResponse(200, json.dumps(data))

    monkeypatch.setattr("llmsetup.util.http.request", request)

    def wire(*engines):
        all_runtimes()
        for engine in engines:
            monkeypatch.setitem(_REGISTRY, engine.name, engine)
        hardware.runtime_availability = {engine.name: engine.available for engine in engines}
        return hardware

    return wire


def setup(capsys, *extra, runtime="llamacpp", register=False):
    code = main(["setup", "--model", "qwen3-8b", "--runtime", runtime, "--task", "chat",
                 "--context", "8192", "--offline", "--quick", "--json",
                 *([] if register else ["--no-register"]), *extra])
    captured = capsys.readouterr()
    return code, json.loads(captured.out) if captured.out else None, captured.err


def test_missing_engine_falls_back_to_installed_runtime(flow, capsys):
    missing = Engine("ollama", available=False)
    fallback = Engine("llamacpp")
    flow(missing, fallback)
    code, result, stderr = setup(capsys, runtime="ollama")
    assert code == 0
    assert result["config"]["runtime"] == "llamacpp"
    assert fallback.provisions == fallback.launches == fallback.stops == 1
    assert missing.provisions == 0
    assert "runtime_fallback" in [action["strategy"] for action in result["repairs"]]
    assert "Install llama-server" not in stderr


def test_transient_provision_and_server_errors_recover_without_questions(flow, capsys):
    engine = Engine("llamacpp", provision_errors=[ProvisionError("connection reset")],
                    launch_errors=[RuntimeUnavailable("connection refused")])
    flow(engine)
    code, result, _ = setup(capsys)
    assert code == 0
    assert engine.provisions == 2
    assert engine.launches == 2
    assert result["repairs"][0]["strategy"] == "restart_server"
    assert result["checks"][0]["status"] == "pass"


def test_corrupt_managed_weights_are_quarantined_and_refetched(flow, capsys):
    engine = Engine("llamacpp")
    engine.corrupt = True
    flow(engine)
    code, result, _ = setup(capsys)
    assert code == 0
    assert engine.provisions == 2
    assert list(model_store().rglob("*.corrupt"))
    assert Path(result["config"]["model_path"]).read_bytes() == b"fixture weights"
    assert result["repairs"][0]["strategy"] == "refetch_weights"


def test_gpu_memory_failure_reaches_cpu_automatically(flow, capsys):
    engine = Engine("llamacpp")
    engine.cpu_required = True
    flow(engine)
    code, result, _ = setup(capsys)
    assert code == 0
    assert result["config"]["gpu_layers"] == 0
    assert "cpu_fallback" in [action["strategy"] for action in result["repairs"]]
    assert engine.launches <= 13


def test_repeated_context_failures_shorten_window_until_server_accepts(flow, capsys):
    engine = Engine("llamacpp")
    engine.max_context = 1024
    flow(engine)
    code, result, _ = setup(capsys)
    assert code == 0
    assert result["config"]["context_tokens"] == 1024
    assert sum(action["strategy"] == "reduce_context" for action in result["repairs"]) == 3


def test_network_blocked_exhausts_recovery_before_actionable_error(flow, capsys, monkeypatch):
    engine = Engine("llamacpp", provision_errors=[ProvisionError("network blocked")] * 20)
    flow(engine)
    registrations = []
    monkeypatch.setattr("llmsetup.pipeline.register_everywhere", lambda *a, **kw: registrations.append(kw))
    code, result, stderr = setup(capsys)
    assert code == 1
    assert engine.provisions == 4
    assert engine.launches == 0
    assert not registrations
    assert "Allow HTTPS to huggingface.co and github.com" in stderr
    assert result["checks"][0]["status"] == "fail"
    assert "Automatic recovery exhausted" in result["checks"][0]["detail"]


def test_no_repair_preserves_single_attempt(flow, capsys):
    engine = Engine("llamacpp", provision_errors=[ProvisionError("network blocked")] * 2)
    flow(engine)
    code, result, _ = setup(capsys, "--no-repair")
    assert code == 1
    assert engine.provisions == 1
    assert result["repairs"] == []


def test_missing_runtime_uses_fake_portable_install_and_cpu_plan(flow, capsys, monkeypatch):
    engine = Engine("llamacpp", available=False)
    hardware = flow(engine)
    installations = []

    def install(self):
        installations.append(self.hardware)
        engine.available = True

    monkeypatch.setattr(Pipeline, "_install_cpu_runtime", install)
    code, result, _ = setup(capsys)
    assert code == 0
    assert len(installations) == 1
    assert not installations[0].gpus
    assert hardware.gpus  # The detected hardware was not mutated.
    assert result["config"]["gpu_layers"] == 0
    assert result["config"]["extra"]["portable_runtime"]
    assert result["hardware"]["gpus"][0]["name"] == hardware.gpus[0].name


def test_uninstallable_runtime_has_clear_instructions(flow, capsys):
    flow(Engine("llamacpp", available=False))
    code, result, stderr = setup(capsys)
    assert code == 1 and result is None
    assert "Automatic recovery exhausted" in stderr
    assert "Install llama-server" in stderr
    assert "Traceback" not in stderr


def test_failed_setup_never_registers_apps(flow, capsys, monkeypatch):
    engine = Engine("llamacpp", launch_errors=[RuntimeUnavailable("connection refused")] * 20)
    flow(engine)
    monkeypatch.setattr("llmsetup.pipeline.register_everywhere", lambda *a, **kw: pytest.fail("registered broken setup"))
    code, result, _ = setup(capsys, "--no-repair", register=True)
    assert code == 1
    assert CheckStatus(result["checks"][0]["status"]) is CheckStatus.FAIL


def test_existing_llamacpp_is_replanned_on_cpu_before_installing_another_engine(flow, capsys):
    engine = Engine("llamacpp")
    engine.cpu_required = True
    engine.gpu_error = "unsupported GPU backend"
    flow(engine)
    code, result, stderr = setup(capsys)
    assert code == 0
    assert result["config"]["gpu_layers"] == 0
    assert "cpu_runtime" in [action["strategy"] for action in result["repairs"]]
    assert not result["config"]["extra"].get("portable_runtime")
    assert "runtime downloads disabled" not in stderr


def test_network_failure_tries_a_cached_quantization(flow, catalog, monkeypatch):
    from llmsetup.runtimes.gguf_store import store_path
    from llmsetup.types import UseCaseProfile
    from llmsetup.util.log import Console

    engine = Engine("llamacpp")
    engine.uses_gguf = True
    hardware = flow(engine)
    spec = catalog.require("qwen3-8b")
    cached = store_path(spec, "Q4_K_M") / "cached.gguf"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"fixture cached weights")
    original = engine.ensure_model

    def ensure(spec, config, console):
        if config.quant != "Q4_K_M":
            engine.provisions += 1
            raise ProvisionError("network blocked")
        return original(spec, config, console)

    monkeypatch.setattr(engine, "ensure_model", ensure)
    pipeline = Pipeline(UseCaseProfile(), hardware, catalog.restrict_to(spec.id),
                        Console(quiet=True), runtime_name="llamacpp", quick=True, register=False)
    try:
        result = pipeline.run()
    finally:
        pipeline.shutdown()
    assert result.succeeded
    assert result.config.quant == "Q4_K_M"
    assert "use_cached_model" in [action.strategy for action in result.repairs]


def test_successful_setup_does_not_ask_for_privileged_os_changes(flow, monkeypatch, capsys, tmp_path):
    engine = Engine("llamacpp")
    flow(engine)
    from llmsetup.tuning.platform_tuning import PlatformAction

    monkeypatch.setattr("llmsetup.tuning.platform_tuning.plan_platform", lambda *a, **kw: [
        PlatformAction("Increase an OS limit", "test limit", command="sudo test-command",
                       needs_root=True, optional=False),
    ])
    report = tmp_path / "report.md"
    code, result, stderr = setup(capsys, "--report", str(report))
    assert code == 0
    assert "sudo test-command" not in stderr
    assert "sudo test-command" not in report.read_text()
    assert result["platform_actions"]  # Diagnostics remain inspectable as data.
