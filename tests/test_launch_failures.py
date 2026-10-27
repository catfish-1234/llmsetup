"""Command and transport failures stay bounded, actionable, and printable."""

import io
import json
import subprocess
import sys
from http import client as http_client
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmsetup.cli import main
from llmsetup.download import RemoteFile, download_file
from llmsetup.runtimes.base import Endpoint, RuntimeUnavailable
from llmsetup.runtimes.ollama import OllamaRuntime
from llmsetup.types import RuntimeConfig
from llmsetup.util import http, proc
from llmsetup.util.errors import ProvisionError
from llmsetup.util.log import Console
from llmsetup.verify.client import OpenAIClient


@pytest.mark.parametrize("operation", ["ok", "error", "step", "warn"])
def test_status_output_survives_cp1252(operation):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    getattr(Console(stream=stream), operation)("Model ready: \u2713 \u2192 \u6a21\u578b")
    assert "Model ready:" in raw.getvalue().decode("cp1252")


def test_saved_report_survives_cp1252_stdout(monkeypatch, tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Setup: \u6a21\u578b \u2713\n", encoding="utf-8")
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    assert main(["report", str(report)]) == 0
    assert raw.getvalue().startswith(b"# Setup:")


def test_subprocess_uses_utf8_even_with_legacy_console(monkeypatch):
    def run(command, **kwargs):
        assert kwargs["encoding"] == "utf-8"
        assert kwargs["errors"] == "replace"
        assert kwargs["timeout"] == 3
        return SimpleNamespace(returncode=0, stdout="\u6a21\u578b", stderr="")
    monkeypatch.setattr(proc.subprocess, "run", run)
    assert proc.run(["ollama", "--version"], timeout=3).stdout == "\u6a21\u578b"


@pytest.mark.parametrize("failure,code", [
    (FileNotFoundError("missing runtime"), 127),
    (PermissionError("access denied"), 126),
    (OSError("bad executable format"), 126),
    (subprocess.TimeoutExpired(["ollama"], 3, output=b"partial\xff"), -1),
])
def test_command_failures_include_reason_and_quoted_rerun(monkeypatch, failure, code):
    monkeypatch.setattr(proc.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(failure))
    command = ["ollama", "create", "local", "-f", "model files/tuned.Modelfile"]
    result = proc.run(command, timeout=3)
    assert not result.ok and result.returncode == code
    assert proc.format_command(command) in result.failure(command)
    assert ("timed out" in result.failure(command)) is (code == -1)


def test_spawn_failure_closes_log_and_has_rerun(monkeypatch, tmp_path):
    monkeypatch.setattr(proc.subprocess, "Popen", lambda *a, **kw:
                        (_ for _ in ()).throw(FileNotFoundError("missing runtime")))
    log = tmp_path / "engine.log"
    with pytest.raises(ProvisionError, match="could not start") as error:
        proc.spawn(["missing-server", "--port", "8080"], log_path=str(log))
    assert "Rerun: missing-server --port 8080" in error.value.hint
    log.unlink()


@pytest.mark.parametrize("failure", [RuntimeUnavailable("runtime missing", "install engine"),
                                     OSError("address already in use")])
def test_cli_common_failure_prints_original_rerun_without_traceback(monkeypatch, capsys, failure):
    monkeypatch.setattr("llmsetup.cli.cmd_setup", lambda *a: (_ for _ in ()).throw(failure))
    command = ["setup", "--task", "coding", "--context", "8192", "--yes"]
    assert main(command) == 1
    output = capsys.readouterr().err
    assert "Rerun: llmsetup setup --task coding --context 8192 --yes" in output
    assert str(failure) in output and "Traceback" not in output


def test_all_ports_busy_is_an_actionable_failure(monkeypatch):
    monkeypatch.setattr(http, "port_is_free", lambda *a: False)
    with pytest.raises(ProvisionError, match="all in use") as error:
        http.find_free_port("127.0.0.1", 8080, limit=2)
    assert "llmsetup setup" in error.value.hint


def test_ollama_port_conflict_never_spawns(monkeypatch):
    monkeypatch.setattr("llmsetup.runtimes.ollama.which", lambda name: name)
    monkeypatch.setattr(http, "request", lambda *a, **kw: http.HttpResponse(0, ""))
    monkeypatch.setattr(http, "port_is_free", lambda *a: False)
    monkeypatch.setattr("llmsetup.runtimes.ollama.spawn", lambda *a, **kw: pytest.fail("spawned"))
    with pytest.raises(RuntimeUnavailable, match="port 11434 is in use") as error:
        OllamaRuntime()._ensure_server(Console(quiet=True))
    assert "Rerun: ollama serve" in error.value.install_hint


def test_dead_ollama_is_stopped_immediately_with_log_and_rerun(monkeypatch):
    runtime = OllamaRuntime()
    stopped = []
    monkeypatch.setattr(runtime, "stop", lambda endpoint: stopped.append(endpoint))
    monkeypatch.setattr("llmsetup.runtimes.ollama.which", lambda name: name)
    monkeypatch.setattr(http, "request", lambda *a, **kw: http.HttpResponse(0, ""))
    monkeypatch.setattr(http, "port_is_free", lambda *a: True)
    monkeypatch.setattr(http.time, "sleep", lambda *a: pytest.fail("waited for a dead daemon"))

    def spawn(command, log_path, env):
        assert env["OLLAMA_HOST"] == "127.0.0.1:11435"
        Path(log_path).write_text("missing Vulkan runtime")
        return SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr("llmsetup.runtimes.ollama.spawn", spawn)
    with pytest.raises(RuntimeUnavailable, match="missing Vulkan runtime") as error:
        runtime._ensure_server(Console(quiet=True), port=11435)
    assert len(stopped) == 1
    assert "Rerun: ollama serve" in error.value.install_hint


@pytest.mark.parametrize("timed_out", [False, True])
def test_ollama_create_failure_prints_full_modelfile_command(monkeypatch, catalog, tmp_path, timed_out):
    runtime = OllamaRuntime()
    monkeypatch.setattr(runtime, "_ensure_server", lambda *args: None)
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *a: tmp_path / "m.gguf")

    def run(command, timeout, env):
        assert timeout == 3600
        assert env == {"OLLAMA_HOST": "127.0.0.1:11434"}
        return proc.CommandResult(-1 if timed_out else 1, "", "import failed", timed_out=timed_out)
    monkeypatch.setattr("llmsetup.runtimes.ollama.run", run)
    config = RuntimeConfig("ollama", "qwen3-8b", "Q4_K_M", 8192)
    with pytest.raises(RuntimeUnavailable) as error:
        runtime.ensure_model(catalog.require("qwen3-8b"), config, Console(quiet=True))
    assert "Rerun: ollama create llmsetup-qwen3-8b -f " in str(error.value)
    assert ".Modelfile" in str(error.value)
    assert ("timed out" in str(error.value)) is timed_out


@pytest.mark.parametrize("failure", [http_client.IncompleteRead(b"partial", 10),
                                     http_client.RemoteDisconnected("server died")])
def test_buffered_check_handles_server_dying_mid_response(monkeypatch, failure):
    class Response(io.BytesIO):
        def read(self, *args):
            raise failure
    monkeypatch.setattr(http.urllib.request, "build_opener", lambda *a:
                        SimpleNamespace(open=lambda *a, **kw: Response()))
    result = OpenAIClient(Endpoint("http://localhost", "m", "llamacpp")).chat([])
    assert not result.ok and str(failure) in result.error


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking"])
@pytest.mark.parametrize("ending", ["done", "finish", "eof", "disconnect", "error"])
def test_streaming_reasoning_and_failure_matrix(monkeypatch, field, ending):
    def lines(*args, **kwargs):
        assert kwargs["timeout"] == 3
        yield 'data: ' + json.dumps({"choices": [{"delta": {field: "working"}}]})
        yield 'data: {"choices": [{"delta": {"content": "42"}}]}'
        if ending == "done":
            yield "data: [DONE]"
        elif ending == "finish":
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}'
        elif ending == "disconnect":
            raise http_client.RemoteDisconnected("server died mid-check")
        elif ending == "error":
            yield 'data: {"error": {"message": "out of memory"}}'
    monkeypatch.setattr(http, "stream_lines", lines)
    result = OpenAIClient(Endpoint("http://localhost", "m", "llamacpp")).stream_chat([], timeout=3)
    assert result["ok"] is (ending in ("done", "finish"))
    assert result["chunks"] == 2 and result["text"] == "42"
    assert result["ttft_s"] is not None
    if not result["ok"]:
        assert result["error"]


def test_model_download_deadline_keeps_partial_and_prints_resume(monkeypatch, tmp_path):
    clock = [0.0]
    class SlowResponse(io.BytesIO):
        status = 200

        def __init__(self):
            super().__init__()
            self.headers = {}

        def read(self, size):
            clock[0] += 3
            return b"partial"

        read1 = read
    monkeypatch.setattr("llmsetup.download.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("llmsetup.download.urllib.request.urlopen", lambda *a, **kw: SlowResponse())
    dest = tmp_path / "m.gguf"
    with pytest.raises(ProvisionError, match="timed out after 2s") as error:
        download_file(RemoteFile("a/b", "m.gguf", revision="a" * 40), dest, total_timeout=2)
    assert dest.with_suffix(".gguf.part").read_bytes() == b"partial"
    assert not dest.exists()
    assert "llmsetup setup" in error.value.hint


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "thinking", "usage", "shapeless"])
@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_empty_first_completion_retries_once_with_thinking_room(field, retry_succeeds,
                                                               monkeypatch, catalog):
    from conftest import make_hardware

    from llmsetup.types import CheckStatus, UseCaseProfile
    from llmsetup.verify.checks import CheckContext, check_completion

    budgets = []

    def request(url, json_body, **kwargs):
        assert kwargs["timeout"] > 0
        budgets.append(json_body["max_tokens"])
        message = {"content": ""}
        if json_body["max_tokens"] > 512 and retry_succeeds:
            prompt = json_body["messages"][0]["content"]
            message["content"] = "Paris" if "France" in prompt else "42"
        elif field not in ("usage", "shapeless"):
            message[field] = "working"
        return http.HttpResponse(200, json.dumps({
            "choices": [{"message": message,
                         "finish_reason": "stop" if field == "shapeless" else "length"}],
            "usage": {} if field == "shapeless" else {"completion_tokens": json_body["max_tokens"]},
        }))
    monkeypatch.setattr(http, "request", request)
    ctx = CheckContext(OpenAIClient(Endpoint("http://localhost", "m", "ollama")),
                       catalog.require("qwen3-8b"), RuntimeConfig("ollama", "m", "Q4_K_M", 8192),
                       UseCaseProfile(), make_hardware())
    result = check_completion(ctx)
    assert (result.status is CheckStatus.PASS) is retry_succeeds
    assert budgets[0] == 512 and budgets[1] > 512
    assert len(budgets) == (4 if retry_succeeds else 2)
    if not retry_succeeds:
        assert result.failure_signature == "empty_response"


@pytest.mark.parametrize("failure", [http_client.IncompleteRead(b"partial", 10),
                                     ValueError("malformed Content-Length")])
def test_download_common_failure_is_actionable(failure, monkeypatch, tmp_path):
    class Response(io.BytesIO):
        status = 200

        def __init__(self):
            super().__init__()
            self.headers = {"Content-Length": "invalid"} if isinstance(failure, ValueError) else {}

        def read(self, size):
            raise failure

        read1 = read
    monkeypatch.setattr("llmsetup.download.urllib.request.urlopen", lambda *a, **kw: Response())
    dest = tmp_path / "m.gguf"
    with pytest.raises(ProvisionError) as error:
        download_file(RemoteFile("a/b", "m.gguf", revision="a" * 40), dest, retries=0)
    assert not dest.exists()
    assert "llmsetup setup" in error.value.hint


def test_specific_model_review_survives_cp1252(monkeypatch, catalog):
    from dataclasses import replace

    from conftest import make_hardware

    from llmsetup.catalog.catalog import Catalog
    from llmsetup.interview.wizard import review_model_choice
    from llmsetup.types import UseCaseProfile

    spec = replace(catalog.require("qwen3-8b"), name="Qwen \u6a21\u578b")
    selection = Catalog([spec], catalog.quant_bpw)
    hw = replace(make_hardware(), runtime_availability={"llamacpp": True})
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    chosen = review_model_choice(UseCaseProfile(requested_model=spec.id), hw, selection,
                                 Console(stream=stream), runtime="llamacpp", interactive=False,
                                 stream=stream)
    assert chosen.requested_model == spec.id
    assert b"Model review: Qwen" in raw.getvalue()


def test_mlx_download_timeout_has_pinned_rerun(monkeypatch, catalog):
    from llmsetup.runtimes.mlx import MlxRuntime

    monkeypatch.setattr("llmsetup.runtimes.mlx.pin_revision", lambda *a: "a" * 40)

    def run(command, timeout):
        assert timeout == 7200 and command[-1] == "a" * 40
        return proc.CommandResult(-1, "", "download stalled", timed_out=True)
    monkeypatch.setattr("llmsetup.runtimes.mlx.run", run)
    with pytest.raises(RuntimeUnavailable, match="timed out") as error:
        MlxRuntime().ensure_model(catalog.require("qwen3-8b"),
                                  RuntimeConfig("mlx", "qwen3-8b", "Q4_K_M", 8192),
                                  Console(quiet=True))
    assert "Rerun: python3 -c " in str(error.value)
    assert "a" * 40 in str(error.value)


@pytest.mark.parametrize("stage,timeout", [("server", 90), ("load", 900)])
def test_lmstudio_failed_command_does_not_claim_a_running_model(stage, timeout, monkeypatch, catalog):
    from llmsetup.runtimes.lmstudio import LmStudioRuntime

    monkeypatch.setattr("llmsetup.runtimes.lmstudio.which", lambda name: name)
    monkeypatch.setattr(http, "request", lambda *a, **kw:
                        http.HttpResponse(200 if stage == "load" else 0, "{}"))
    monkeypatch.setattr(http, "wait_for", lambda *a, **kw: pytest.fail("polled after failed command"))

    def run(command, **kwargs):
        assert command[1] == stage and kwargs["timeout"] == timeout
        return proc.CommandResult(-1, "", "engine stalled", timed_out=True)
    monkeypatch.setattr("llmsetup.runtimes.lmstudio.run", run)
    with pytest.raises(RuntimeUnavailable, match="timed out") as error:
        LmStudioRuntime().start(catalog.require("qwen3-8b"),
                                RuntimeConfig("lmstudio", "qwen3-8b", "Q4_K_M", 8192),
                                Console(quiet=True))
    assert f"Rerun: lms {stage}" in str(error.value)
