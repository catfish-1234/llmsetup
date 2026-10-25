"""Ollama must serve repaired settings and import every split GGUF shard."""

from pathlib import Path

import pytest
from conftest import make_hardware

from llmsetup.pipeline import Pipeline
from llmsetup.runtimes.base import RuntimeUnavailable
from llmsetup.runtimes.ollama import OllamaRuntime
from llmsetup.types import RuntimeConfig, UseCaseProfile
from llmsetup.util.log import Console
from llmsetup.util.proc import CommandResult


@pytest.fixture
def created_model(catalog, monkeypatch, tmp_path):
    runtime = OllamaRuntime()
    spec = catalog.require("qwen3-4b")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    config = RuntimeConfig("ollama", spec.id, "Q5_K_M", 32768,
                           threads=4, extra={"ollama_port": 11999})
    monkeypatch.setattr(runtime, "_ensure_server", lambda *args: None)
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *args: gguf)
    applied = []

    def create(command, **kwargs):
        assert command[:3] == ["ollama", "create", runtime.derived_name(spec)]
        applied.append((Path(command[-1]).read_text(), kwargs))
        return CommandResult(0, "", "")

    monkeypatch.setattr("llmsetup.runtimes.ollama.run", create)
    runtime.ensure_model(spec, config, Console(quiet=True))
    return runtime, spec, config, applied


def test_context_repair_reaches_served_model_before_reverify(created_model, catalog, monkeypatch):
    runtime, spec, config, applied = created_model
    pipeline = Pipeline(UseCaseProfile(), make_hardware(), catalog, Console(quiet=True),
                        runtime_name="ollama", register=False)
    pipeline._runtime = runtime
    pipeline._provisioned = (spec.id, config.quant)
    config.context_tokens = 1024
    monkeypatch.setattr(runtime, "ensure_model", lambda *args: pytest.fail("weights refetched"))

    class Harness:
        def __init__(self, **kwargs):
            assert kwargs["client"].endpoint.model == config.served_name

        def run(self, names):
            assert "PARAMETER num_ctx 1024\n" in applied[-1][0]
            return []

    monkeypatch.setattr("llmsetup.pipeline.Harness", Harness)
    assert pipeline._relaunch(config, None) == []
    assert len(applied) == 2
    assert applied[-1][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11999"


@pytest.mark.parametrize(("field", "value", "parameter"), [
    ("gpu_layers", 3, "num_gpu 3"),
    ("ubatch_size", 64, "num_batch 64"),
    ("threads", 2, "num_thread 2"),
    ("temperature", 0.2, "temperature 0.2"),
    ("top_p", 0.5, "top_p 0.5"),
    ("top_k", 10, "top_k 10"),
    ("min_p", 0.1, "min_p 0.1"),
    ("repeat_penalty", 1.2, "repeat_penalty 1.2"),
])
def test_non_context_repairs_reapply_modelfile(created_model, field, value, parameter):
    runtime, spec, config, applied = created_model
    runtime.start(spec, config, Console(quiet=True))
    assert len(applied) == 1
    setattr(config, field, value)
    runtime.start(spec, config, Console(quiet=True))
    assert len(applied) == 2
    assert f"PARAMETER {parameter}\n" in applied[-1][0]
    assert config.context_tokens == 32768


def test_failed_reapply_is_not_served_or_cached(created_model, monkeypatch):
    runtime, spec, config, applied = created_model
    config.context_tokens = 1024
    with monkeypatch.context() as patch:
        patch.setattr("llmsetup.runtimes.ollama.run",
                      lambda *a, **kw: CommandResult(1, "", "create rejected"))
        with pytest.raises(RuntimeUnavailable, match="create rejected"):
            runtime.start(spec, config, Console(quiet=True))
    runtime.start(spec, config, Console(quiet=True))
    assert len(applied) == 2
    assert "PARAMETER num_ctx 1024\n" in applied[-1][0]


def test_optional_settings_are_added_and_removed(created_model):
    runtime, spec, config, applied = created_model
    config.extra.update(stop=["END"], system_prompt="Answer briefly.")
    runtime.start(spec, config, Console(quiet=True))
    assert 'PARAMETER stop "END"\n' in applied[-1][0]
    assert 'SYSTEM """Answer briefly."""' in applied[-1][0]
    config.threads = None
    config.extra.pop("stop")
    config.extra.pop("system_prompt")
    runtime.start(spec, config, Console(quiet=True))
    assert len(applied) == 3
    assert "num_thread" not in applied[-1][0]
    assert "PARAMETER stop" not in applied[-1][0]
    assert "SYSTEM" not in applied[-1][0]


def test_saved_config_is_reapplied_by_a_fresh_adapter(created_model):
    runtime, spec, config, applied = created_model
    config.context_tokens = 1024
    fresh = OllamaRuntime()
    fresh._ensure_server = runtime._ensure_server
    fresh.start(spec, config, Console(quiet=True))
    assert len(applied) == 2
    assert "PARAMETER num_ctx 1024\n" in applied[-1][0]


def test_same_tag_on_another_server_is_reapplied(created_model):
    runtime, spec, config, applied = created_model
    config.extra["ollama_port"] = 11998
    runtime.start(spec, config, Console(quiet=True))
    assert len(applied) == 2
    assert applied[-1][1]["env"]["OLLAMA_HOST"] == "127.0.0.1:11998"


def test_split_source_includes_all_shards(catalog, tmp_path):
    spec = catalog.require("qwen2.5-coder-7b")
    source = tmp_path / "model-00001-of-00002.gguf"
    config = RuntimeConfig("ollama", spec.id, "Q4_K_M", 1024,
                           model_path=str(source), extra={"from_gguf": True})
    for index in (1, 2):
        (tmp_path / f"model-{index:05d}-of-00002.gguf").write_bytes(b"GGUF")
    # The real CLI's glob import is exercised separately in test_ollama_live.py.
    assert f"FROM {tmp_path / 'model-?????-of-00002.gguf'}\n" in OllamaRuntime().modelfile(spec, config)
