"""Exercise real CLI → pipeline → verifier → saved artifacts with fake external I/O."""

import json
import socket
from pathlib import Path

import pytest
from conftest import make_hardware

from llmfit.cli import main
from llmfit.runtimes.base import Endpoint, Runtime
from llmfit.runtimes.registry import _REGISTRY
from llmfit.util.http import HttpResponse


class OfflineRuntime(Runtime):
    name = "offline-test"
    label = "Offline fixture"
    uses_gguf = False

    def __init__(self):
        self.events = []

    def is_available(self):
        return True

    def ensure_model(self, spec, config, console):
        self.events.append("provision")
        config.model_path = "fixture://local-model"
        return config

    def start(self, spec, config, console):
        self.events.append("start")
        return Endpoint("http://fixture.invalid", spec.id, self.name)

    def stop(self, endpoint):
        self.events.append("stop")

    def launch_command(self, spec, config):
        return ["fixture-engine", config.model_path]


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setenv("LLMFIT_AUTO_UPDATE", "0")
    monkeypatch.setattr("llmfit.cli.detect_hardware", lambda **kw: make_hardware())
    monkeypatch.setattr(socket, "socket", lambda *a, **kw: pytest.fail("unexpected network"))
    runtime = OfflineRuntime()
    monkeypatch.setitem(_REGISTRY, runtime.name, runtime)
    return runtime


@pytest.mark.parametrize("healthy", [True, False])
def test_full_setup_verifies_saves_reports_and_cleans_up(offline, healthy, monkeypatch,
                                                        tmp_path, capsys):
    requests = []

    def request(url, **kwargs):
        assert url.startswith("http://fixture.invalid/v1/")
        requests.append(url)
        if url.endswith("/models"):
            ids = ["qwen3-8b"] if healthy else ["wrong-model"]
            data = {"data": [{"id": model} for model in ids]}
        else:
            body = kwargs["json_body"]
            assert body["model"] == "qwen3-8b"
            prompt = body["messages"][0]["content"]
            text = "Paris" if "France" in prompt else "42"
            data = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
        return HttpResponse(200, json.dumps(data))

    monkeypatch.setattr("llmfit.util.http.request", request)
    report = tmp_path / "report.md"
    args = ["setup", "--model", "qwen3-8b", "--runtime", offline.name, "--task", "chat",
            "--context", "8192", "--yes", "--offline", "--quick", "--no-register",
            "--no-repair", "--json", "--report", str(report)]
    assert main(args) == (0 if healthy else 1)
    result = json.loads(capsys.readouterr().out)
    assert offline.events == ["provision", "start", "stop"]
    assert result["checks"][0]["status"] == ("pass" if healthy else "fail")
    assert len(requests) == (7 if healthy else 1)
    saved = json.loads((tmp_path / "home/last-setup.json").read_text())
    assert saved["config"] == result["config"]
    assert report.read_text() == (tmp_path / "home/last-report.md").read_text()
    assert main(["report"]) == 0
    assert "Qwen3 8B" in capsys.readouterr().out


def test_dry_run_runs_real_decisions_without_provisioning(offline, tmp_path, capsys):
    answers = tmp_path / "answers.yaml"
    assert main(["setup", "--runtime", offline.name, "--task", "coding", "--yes", "--offline",
                 "--dry-run", "--json", "--save-profile", str(answers)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model"]["id"]
    assert result["config"]["context_tokens"] > 0
    assert result["checks"] == []
    assert answers.exists()
    assert offline.events == []
    assert "No checks were run" in Path(tmp_path / "home/last-report.md").read_text()
