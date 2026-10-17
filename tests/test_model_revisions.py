"""Pin at the I/O boundary; changing upstream HEAD must not change a rerun."""

import io
import json
from dataclasses import replace

import pytest

from llmsetup.download import RemoteFile, list_repo_files, pin_revision
from llmsetup.runtimes.gguf_store import store_path
from llmsetup.runtimes.mlx import MlxRuntime
from llmsetup.runtimes.vllm import VllmRuntime
from llmsetup.types import RuntimeConfig
from llmsetup.util.errors import ProvisionError
from llmsetup.util.log import Console
from llmsetup.util.proc import CommandResult

SHA = "a" * 40


def test_listing_and_all_download_urls_use_persisted_commit(monkeypatch):
    urls = []

    def urlopen(request, **kwargs):
        urls.append(request.full_url)
        return io.BytesIO(json.dumps({"sha": SHA, "siblings": [
            {"rfilename": "model-00001-of-00002.gguf"},
            {"rfilename": "model-00002-of-00002.gguf"},
            {"rfilename": "mmproj-f16.gguf"},
        ]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    files = list_repo_files("a/b")
    assert all(f"/resolve/{SHA}/" in file.url for file in files)
    list_repo_files("a/b")
    assert sum(url.endswith("/api/models/a/b") for url in urls) == 1
    assert all(url.endswith(f"/revision/{SHA}") for url in urls[1:])


@pytest.mark.parametrize("revision", [None, "main", "v1", "../bad", "a" * 39])
def test_remote_file_rejects_mutable_or_missing_revision(revision):
    with pytest.raises(ProvisionError, match="commit SHA"):
        _ = RemoteFile("a/b", "model.gguf", revision=revision).url


def test_explicit_pin_needs_no_resolution_and_separates_stores(catalog, monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: pytest.fail("network"))
    assert pin_revision("a/b", SHA) == SHA
    spec = catalog.require("qwen3-8b")
    assert store_path(replace(spec, gguf_revision=SHA), "Q4_K_M") != store_path(
        replace(spec, gguf_revision="b" * 40), "Q4_K_M")


def test_missing_sha_fails_instead_of_downloading_main(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(b'{}'))
    with pytest.raises(ProvisionError, match="commit SHA"):
        list_repo_files("a/b")


def test_vllm_pins_weights_and_tokenizer(catalog):
    spec = replace(catalog.require("qwen3-8b"), hf_revision=SHA)
    runtime = VllmRuntime()
    config = runtime.ensure_model(spec, RuntimeConfig("vllm", spec.id, "BF16", 8192),
                                  Console(quiet=True))
    command = runtime.launch_command(spec, config)
    assert command[command.index("--revision") + 1] == SHA
    assert command[command.index("--tokenizer-revision") + 1] == SHA
    assert config.to_dict()["extra"]["model_revision"] == SHA


def test_mlx_downloads_snapshot_and_serves_local_path(catalog, monkeypatch):
    spec = replace(catalog.require("qwen3-8b"), mlx_revision=SHA)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return CommandResult(0, f"/cache/snapshots/{SHA}\n", "")

    monkeypatch.setattr("llmsetup.runtimes.mlx.run", run)
    runtime = MlxRuntime()
    config = runtime.ensure_model(spec, RuntimeConfig("mlx", spec.id, "Q4_K_M", 8192),
                                  Console(quiet=True))
    assert commands[0][-2:] == [spec.mlx_repo, SHA]
    assert config.model_path == f"/cache/snapshots/{SHA}"
    assert runtime.launch_command(spec, config)[4] == config.model_path


def test_discovery_reads_and_records_the_pinned_config(monkeypatch):
    from llmsetup.catalog.discover import Candidate, fetch_config

    urls = []

    def urlopen(request, **kwargs):
        urls.append(request.full_url)
        payload = {"sha": SHA} if "/api/models/" in request.full_url else {"num_hidden_layers": 32}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert fetch_config("a/b") == {"num_hidden_layers": 32}
    assert urls[-1].endswith(f"/resolve/{SHA}/config.json")
    entry = Candidate("b", "a/b", "a", "B", 8, 32, 4096, 8, 128, 8192).to_entry()
    assert entry["hf_revision"] == SHA


def test_ollama_imports_gguf_without_pulling_a_mutable_tag(catalog, monkeypatch, tmp_path):
    from llmsetup.runtimes.ollama import OllamaRuntime

    runtime = OllamaRuntime()
    spec = catalog.require("qwen3-8b")
    config = RuntimeConfig("ollama", spec.id, "Q4_K_M", 8192)
    model_path = tmp_path / "pinned.gguf"
    commands = []
    monkeypatch.setattr(runtime, "_ensure_server", lambda console: None)
    monkeypatch.setattr("llmsetup.runtimes.ollama.fetch_gguf", lambda *a: model_path)
    monkeypatch.setattr("llmsetup.runtimes.ollama.run", lambda cmd, **kw:
                        commands.append(cmd) or CommandResult(0, "", ""))
    runtime.ensure_model(spec, config, Console(quiet=True))
    assert all("pull" not in command for command in commands)
    assert f"FROM {model_path}" in runtime.modelfile(spec, config)


def test_projector_uses_same_commit_as_downloaded_weights(catalog, monkeypatch):
    from llmsetup.runtimes.gguf_store import fetch_gguf, fetch_projector

    files = [RemoteFile("a/b", "model-Q4_K_M.gguf", 1, SHA),
             RemoteFile("a/b", "mmproj-f16.gguf", 1, SHA)]
    pins = []
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.list_repo_files", lambda repo, **kw:
                        pins.append(kw["revision"]) or files)
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.download_file", lambda file, dest, *a: dest)
    spec = replace(catalog.require("qwen3-8b"), gguf_repo="a/b")
    weights = fetch_gguf(spec, "Q4_K_M", Console(quiet=True))
    projector = fetch_projector(spec, "Q4_K_M", Console(quiet=True))
    assert pins == [None, SHA]
    assert spec.to_dict()["gguf_revision"] == SHA
    assert weights.parent == projector.parent
    assert weights.parent.name == SHA
