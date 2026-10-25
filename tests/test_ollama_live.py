"""Opt-in, offline checks against a real Ollama binary and existing GGUFs.

Set LLMSETUP_TEST_OLLAMA to the executable, LLMSETUP_TEST_GGUF to a small
unsplit GGUF, and LLMSETUP_TEST_SPLIT_GGUF to the first of a complete shard set.
LLMSETUP_TEST_OLLAMA_BLOBS optionally supplies read-only cached weight blobs.
No models are pulled; each test owns and terminates its temporary daemon.
"""

import os
import signal
import subprocess
from pathlib import Path

import pytest

from llmsetup.runtimes.base import RuntimeUnavailable
from llmsetup.runtimes.ollama import OllamaRuntime
from llmsetup.types import RuntimeConfig
from llmsetup.util import http
from llmsetup.util.log import Console
from llmsetup.util.proc import run


@pytest.fixture
def ollama_server(tmp_path, monkeypatch):
    executable = os.environ.get("LLMSETUP_TEST_OLLAMA")
    if not executable:
        pytest.skip("set LLMSETUP_TEST_OLLAMA to run offline live Ollama checks")
    binary = Path(executable).resolve()
    assert binary.is_file()
    store = tmp_path / "ollama"
    (store / "blobs").mkdir(parents=True)
    cached = os.environ.get("LLMSETUP_TEST_OLLAMA_BLOBS")
    if cached:
        for blob in Path(cached).glob("sha256-*"):
            # Only immutable weight blobs are linked; manifests and parameters
            # are written entirely inside the temporary store. Disable pruning.
            if blob.stat().st_size > 1_000_000:
                (store / "blobs" / blob.name).symlink_to(blob.resolve())
    port = http.find_free_port("127.0.0.1", 11834)
    monkeypatch.setenv("PATH", str(binary.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")

    def bounded_run(command, **kwargs):
        kwargs["timeout"] = min(kwargs.get("timeout", 30), 30)
        return run(command, **kwargs)

    monkeypatch.setattr("llmsetup.runtimes.ollama.run", bounded_run)
    env = {
        **os.environ, "OLLAMA_MODELS": str(store), "OLLAMA_NOPRUNE": "1",
        "OLLAMA_NO_CLOUD": "1", "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_VULKAN": "0", "CUDA_VISIBLE_DEVICES": "-1",
        "OLLAMA_LOAD_TIMEOUT": "30s",
    }
    base = f"http://127.0.0.1:{port}"
    with (tmp_path / "ollama.log").open("w") as log:
        process = subprocess.Popen([str(binary), "serve"], env=env,
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            assert http.wait_for(
                lambda: http.request(base + "/api/tags", timeout=1).ok,
                timeout=10, interval=0.2,
            ), (tmp_path / "ollama.log").read_text()[-2000:]
            yield base, port
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def local_asset(variable):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to an existing local GGUF")
    path = Path(value).resolve()
    assert path.is_file()
    return path


def test_loaded_context_changes_after_repair(ollama_server, catalog, monkeypatch):
    gguf = local_asset("LLMSETUP_TEST_GGUF")
    base, port = ollama_server
    runtime, console = OllamaRuntime(), Console(quiet=True)
    spec = catalog.require("qwen3-4b")
    config = RuntimeConfig("ollama", spec.id, "Q5_K_M", 2048,
                           gpu_layers=0, threads=2, ubatch_size=64,
                           extra={"ollama_port": port})
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *a: gguf)
    runtime.ensure_model(spec, config, console)

    for context in (2048, 1024):
        config.context_tokens = context
        # Change a non-context parameter too, and inspect it independently.
        config.temperature = 0.7 if context == 2048 else 0.2
        endpoint = runtime.start(spec, config, console)
        shown = http.request(base + "/api/show", method="POST",
                             json_body={"model": endpoint.model}, timeout=10)
        assert shown.ok, shown.body
        parameters = dict(line.split() for line in shown.json()["parameters"].splitlines())
        assert int(parameters["num_ctx"]) == context
        assert float(parameters["temperature"]) == config.temperature
        # Empty-prompt preload: no completion or long context prompt needed.
        loaded = http.request(base + "/api/generate", method="POST", timeout=30,
                              json_body={"model": endpoint.model, "stream": False,
                                         "keep_alive": "30s"})
        assert loaded.ok, loaded.body or loaded.error
        running = http.request(base + "/api/ps", timeout=5)
        assert running.ok, running.body
        model = next(m for m in running.json()["models"]
                     if m["name"].removesuffix(":latest") == endpoint.model)
        assert model["context_length"] == context


def test_real_split_import_preserves_failure_and_imports_all_shards(
        ollama_server, catalog, monkeypatch, tmp_path):
    first = local_asset("LLMSETUP_TEST_SPLIT_GGUF")
    base, port = ollama_server
    # Reproduce the old source exactly: the adjacent second shard is ignored.
    old = tmp_path / "first-only.Modelfile"
    old.write_text(f"FROM {first}\n")
    rejected = subprocess.run(["ollama", "create", "batch-a-first-only", "-f", str(old)],
                              capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert "invalid split GGUF" in rejected.stderr
    assert "has 1 shards, expected 2" in rejected.stderr
    runtime, console = OllamaRuntime(), Console(quiet=True)
    spec = catalog.require("qwen2.5-coder-7b")
    config = RuntimeConfig("ollama", spec.id, "Q4_K_M", 1024,
                           extra={"ollama_port": port})
    # A genuinely incomplete set must retain Ollama's failure, never advertise
    # success just because a first shard exists.
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / first.name).symlink_to(first)
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf",
                        lambda *a: incomplete / first.name)
    with pytest.raises(RuntimeUnavailable, match="has 1 shards, expected 2"):
        runtime.ensure_model(spec, config, console)
    assert config.served_name is None
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *a: first)
    runtime.ensure_model(spec, config, console)
    shown = http.request(base + "/api/show", method="POST",
                         json_body={"model": config.served_name}, timeout=10)
    assert shown.ok, shown.body
    assert shown.json()["details"]["family"] == "qwen2"
