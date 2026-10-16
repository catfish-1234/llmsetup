"""Catalog strings must stay data when turned into paths or shell commands."""

import json
import shlex
import subprocess
import sys

import pytest

from llmfit.interop.llama_swap import build_config
from llmfit.interop.openai_clients import snippets
from llmfit.runtimes.base import Endpoint
from llmfit.runtimes.llamacpp import LlamaCppRuntime
from llmfit.runtimes.mlx import MlxRuntime
from llmfit.runtimes.vllm import VllmRuntime
from llmfit.types import RuntimeConfig, UseCaseProfile
from llmfit.ui.server import _setup_command
from llmfit.util.fs import safe_subpath


def test_missing_leaf_cannot_escape_through_symlink(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "publisher").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        safe_subpath(root, "publisher", "new-model", "file.gguf")


def test_symlinked_root_accepts_a_missing_child(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "store"
    link.symlink_to(actual, target_is_directory=True)
    assert safe_subpath(link, "model", "file.gguf").resolve() == actual / "model/file.gguf"


@pytest.mark.parametrize("runtime", [LlamaCppRuntime(), VllmRuntime(), MlxRuntime()])
def test_generated_scripts_preserve_arguments_without_expansion(runtime, catalog,
                                                                monkeypatch, tmp_path):
    marker = tmp_path / "must-not-exist"
    values = ["a path with spaces", "", 'single\' and "double" quotes',
              f"$(touch {marker})", "`uname`", "a\\b", "semi;colon", "$HOME"]
    command = [sys.executable, "-c", "import json,sys; print(json.dumps(sys.argv[1:]))",
               *values]
    monkeypatch.setattr(runtime, "launch_command", lambda spec, config: command)
    spec = catalog.require("qwen3-8b")
    config = RuntimeConfig(runtime.name, spec.id, "Q4_K_M", 8192)
    script = next(iter(runtime.config_files(spec, config).values()))
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == values
    assert not marker.exists()


def test_swap_command_round_trips_catalog_values(catalog):
    spec = catalog.require("qwen3-8b")
    config = RuntimeConfig("llamacpp", spec.id, "Q4_K_M", 8192)
    command = ["llama-server", "--model", "/models/my model's weights.gguf", "--alias", "$(id)"]
    result = build_config([(spec, config, command)])
    assert shlex.split(result["models"][spec.id]["cmd"]) == command


def test_ui_handoff_quotes_catalog_model_id(catalog):
    from llmfit.types import Candidate, MemoryEstimate

    spec = catalog.require("qwen3-8b")
    spec.id = "model $(id); echo injected"
    candidate = Candidate(spec, "Q4_K_M", MemoryEstimate(1, 1, 1, 8192, "Q4_K_M"), True, 1.0)
    command = shlex.split(_setup_command(candidate, UseCaseProfile()))
    assert command[command.index("--model") + 1] == spec.id


def test_shell_snippet_does_not_expand_model_name(catalog):
    spec = catalog.require("qwen3-8b")
    config = RuntimeConfig("external", spec.id, "Q4_K_M", 8192)
    model = "model 'quoted' $(id)"
    snippet = snippets(Endpoint("http://localhost", model, "external"), spec, config)
    content = snippet["custom_openai"]["content"]
    assert shlex.split(content.splitlines()[-1]) == ["export", f"OPENAI_MODEL={model}"]
