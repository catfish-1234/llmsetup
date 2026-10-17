"""A failed child process should reach repair immediately, with useful diagnosis."""

import importlib
from pathlib import Path

import pytest

from llmsetup.runtimes.base import Endpoint, RuntimeUnavailable
from llmsetup.runtimes.registry import get_runtime
from llmsetup.types import RuntimeConfig
from llmsetup.util.log import Console
from llmsetup.verify import signatures


@pytest.mark.parametrize("name", ["llamacpp", "vllm", "mlx", "tabby"])
def test_dead_server_is_stopped_without_waiting_for_timeout(name, catalog, monkeypatch, tmp_path):
    runtime = get_runtime(name)
    module = importlib.import_module(f"llmsetup.runtimes.{name}")
    stopped = []
    monkeypatch.setattr(runtime, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "launch_command", lambda *a: ["fixture-engine"])
    monkeypatch.setattr(runtime, "stop", lambda endpoint: stopped.append(endpoint))
    monkeypatch.setattr("llmsetup.util.http.port_is_free", lambda *a: True)
    monkeypatch.setattr("llmsetup.util.http.time.sleep", lambda *a: pytest.fail("waited for a dead server"))

    class DeadProcess:
        returncode = 1

        def poll(self):
            return 1

    def spawn(command, *, log_path):
        Path(log_path).write_text("CUDA out of memory")
        return DeadProcess()

    monkeypatch.setattr(module, "spawn", spawn)
    config = RuntimeConfig(runtime=name, model_id="qwen3-8b", quant="Q4_K_M",
                           model_path=str(tmp_path / "weights.gguf"), context_tokens=8192)
    with pytest.raises(RuntimeUnavailable, match="CUDA out of memory") as error:
        runtime.start(catalog.require("qwen3-8b"), config, Console(quiet=True))
    assert len(stopped) == 1
    assert signatures.classify(str(error.value)) == signatures.OUT_OF_MEMORY


def test_startup_timeout_stops_owned_server_and_returns_log(monkeypatch, tmp_path):
    runtime = get_runtime("llamacpp")
    stopped = []
    monkeypatch.setattr(runtime, "stop", lambda endpoint: stopped.append(endpoint))
    monkeypatch.setattr("llmsetup.util.http.wait_for", lambda *a, **kw: False)
    log = tmp_path / "engine.log"
    log.write_text("invalid GGUF magic")
    endpoint = Endpoint("http://fixture.invalid", "fixture", "llamacpp", log_path=str(log))
    with pytest.raises(RuntimeUnavailable, match="invalid GGUF magic") as error:
        runtime.wait_until_ready(endpoint, lambda: False, timeout=90, interval=1)
    assert stopped == [endpoint]
    assert signatures.classify(str(error.value)) == signatures.CORRUPT_WEIGHTS
