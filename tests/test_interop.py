"""Interop: does the model actually become visible in other apps?"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfit.interop import jan, lmstudio
from llmfit.interop.openai_clients import client_ids, snippets
from llmfit.runtimes.base import Endpoint
from llmfit.types import RuntimeConfig
from llmfit.util.fs import link_or_copy


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "weights" / "Qwen3-8B-Q4_K_M.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF" + b"\x00" * 512)
    return path


@pytest.fixture
def config(catalog):
    spec = catalog.require("qwen3-8b")
    return RuntimeConfig(runtime="llamacpp", model_id=spec.id, quant="Q4_K_M",
                         context_tokens=32768, total_layers=spec.layers,
                         gpu_layers=spec.layers, port=11435, served_name=spec.id)


class TestLmStudio:
    def test_uses_the_publisher_model_file_layout(self, catalog, gguf, monkeypatch):
        """LM Studio parses this exact shape to populate its UI; anything else
        leaves the model invisible."""
        monkeypatch.setattr(lmstudio, "which", lambda name: None)
        spec = catalog.require("qwen3-8b")
        record = lmstudio.register_in_lmstudio(spec, gguf, "Q4_K_M")

        assert record["ok"]
        placed = Path(record["path"])
        assert placed.exists()
        relative = placed.relative_to(lmstudio.models_dir())
        assert len(relative.parts) == 3, relative       # publisher/model/file.gguf
        assert relative.parts[0] == "Qwen"
        assert relative.parts[2].endswith(".gguf")

    def test_links_rather_than_copying(self, catalog, gguf, monkeypatch):
        """Weights are tens of gigabytes; a copy per app is not acceptable."""
        monkeypatch.setattr(lmstudio, "which", lambda name: None)
        record = lmstudio.register_in_lmstudio(catalog.require("qwen3-8b"),
                                               gguf, "Q4_K_M")
        assert record["method"] in ("hardlink", "symlink")

    def test_is_idempotent(self, catalog, gguf, monkeypatch):
        monkeypatch.setattr(lmstudio, "which", lambda name: None)
        spec = catalog.require("qwen3-8b")
        first = lmstudio.register_in_lmstudio(spec, gguf, "Q4_K_M")
        second = lmstudio.register_in_lmstudio(spec, gguf, "Q4_K_M")
        assert second["ok"]
        assert Path(first["path"]) == Path(second["path"])

    def test_split_models_are_registered_whole(self, catalog, tmp_path, monkeypatch):
        """A single shard would leave LM Studio with an unloadable model."""
        monkeypatch.setattr(lmstudio, "which", lambda name: None)
        directory = tmp_path / "split"
        directory.mkdir()
        shards = []
        for index in (1, 2, 3):
            shard = directory / f"model-0000{index}-of-00003.gguf"
            shard.write_bytes(b"GGUF")
            shards.append(shard)

        record = lmstudio.register_in_lmstudio(catalog.require("qwen3-8b"),
                                               shards[0], "Q4_K_M")
        assert len(record["files"]) == 3

    def test_publisher_names_are_made_filesystem_safe(self, catalog, gguf, monkeypatch):
        monkeypatch.setattr(lmstudio, "which", lambda name: None)
        spec = catalog.require("qwen3-8b")
        spec.publisher = "some/publisher with spaces"
        record = lmstudio.register_in_lmstudio(spec, gguf, "Q4_K_M")
        relative = Path(record["path"]).relative_to(lmstudio.models_dir())
        assert "/" not in relative.parts[0] and " " not in relative.parts[0]


class TestJan:
    def test_writes_a_valid_descriptor(self, catalog, gguf, tmp_path, monkeypatch):
        data_dir = tmp_path / "jan"
        (data_dir / "models").mkdir(parents=True)
        monkeypatch.setenv("JAN_DATA_DIR", str(data_dir))

        spec = catalog.require("qwen3-8b")
        cfg = RuntimeConfig(runtime="llamacpp", model_id=spec.id, quant="Q4_K_M",
                            context_tokens=16384, gpu_layers=36, total_layers=36)
        record = jan.register_in_jan(spec, gguf, cfg)
        assert record["ok"]

        descriptor = json.loads((Path(record["path"]) / "model.json").read_text())
        assert descriptor["id"] == spec.id
        assert descriptor["settings"]["ctx_len"] == 16384
        assert descriptor["settings"]["ngl"] == 36
        assert descriptor["sources"][0]["filename"] == gguf.name

    def test_skips_cleanly_when_not_installed(self, catalog, gguf, tmp_path, monkeypatch):
        monkeypatch.setenv("JAN_DATA_DIR", str(tmp_path / "absent"))
        record = jan.register_in_jan(catalog.require("qwen3-8b"), gguf,
                                     RuntimeConfig(runtime="x", model_id="y",
                                                   quant="Q4_K_M", context_tokens=8192))
        assert record["ok"] is False


class TestOpenAiSnippets:
    @pytest.fixture
    def emitted(self, catalog, config):
        endpoint = Endpoint(base_url="http://127.0.0.1:11435", model="qwen3-8b",
                            runtime="llamacpp")
        return snippets(endpoint, catalog.require("qwen3-8b"), config)

    def test_covers_every_advertised_client(self, emitted):
        assert set(emitted) == set(client_ids())

    def test_every_snippet_names_where_it_goes(self, emitted):
        for client, snippet in emitted.items():
            assert snippet["path"], client
            assert snippet["content"].strip(), client

    def test_the_endpoint_url_is_openai_shaped(self, emitted):
        for snippet in emitted.values():
            if "127.0.0.1" in snippet["content"]:
                assert "/v1" in snippet["content"]

    def test_the_zed_snippet_is_valid_json(self, emitted):
        parsed = json.loads(emitted["zed"]["content"])
        assert parsed["language_models"]["openai"]["available_models"]

    def test_the_continue_snippet_is_valid_yaml(self, emitted):
        import yaml

        parsed = yaml.safe_load(emitted["continue"]["content"])
        assert parsed["models"][0]["apiBase"].endswith("/v1")

    def test_context_length_is_propagated_to_clients(self, emitted, config):
        import yaml

        parsed = yaml.safe_load(emitted["continue"]["content"])
        options = parsed["models"][0]["defaultCompletionOptions"]
        assert options["contextLength"] == config.context_tokens

    def test_docker_networking_is_called_out_for_open_webui(self, emitted):
        """127.0.0.1 inside a container is the container — the single most
        common reason Open WebUI cannot reach a local model."""
        assert "host.docker.internal" in emitted["open_webui"]["content"]


class TestLinkStrategy:
    def test_prefers_the_cheapest_available_method(self, tmp_path):
        source = tmp_path / "a.bin"
        source.write_bytes(b"x" * 64)
        method, path = link_or_copy(source, tmp_path / "sub" / "b.bin")
        assert method in ("hardlink", "symlink", "copy")
        assert path.exists()

    def test_recognises_an_existing_link(self, tmp_path):
        source = tmp_path / "a.bin"
        source.write_bytes(b"x" * 64)
        destination = tmp_path / "sub" / "b.bin"
        link_or_copy(source, destination)
        method, _ = link_or_copy(source, destination)
        assert method == "existing"

    def test_replaces_an_unrelated_file_at_the_destination(self, tmp_path):
        source = tmp_path / "a.bin"
        source.write_bytes(b"new content")
        destination = tmp_path / "b.bin"
        destination.write_bytes(b"stale")
        link_or_copy(source, destination)
        assert destination.read_bytes() == b"new content"
