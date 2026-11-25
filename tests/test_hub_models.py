"""Any Hugging Face GGUF repo (--model hf:owner/repo) and named variants (--variant)."""

from __future__ import annotations

import pytest
import yaml

from llmsetup.catalog import hub
from llmsetup.cli import main
from llmsetup.util.errors import CatalogError

SEARCH = "https://huggingface.co/api/models?"


def fake_hub(search_results, repos):
    """A stand-in for the Hub API: search listings by URL prefix, repo info by repo id."""
    def get_json(url, timeout=20.0):
        if url.startswith(SEARCH):
            return search_results
        for repo, info in repos.items():
            if url.endswith(f"/api/models/{repo}"):
                return info
        return None
    return get_json


def gguf_files(*names):
    return {"siblings": [{"rfilename": name} for name in names]}


@pytest.mark.parametrize("value, expected", [
    ("hf:bartowski/Model-GGUF", "bartowski/Model-GGUF"),
    ("https://huggingface.co/bartowski/Model-GGUF", "bartowski/Model-GGUF"),
    ("huggingface.co/bartowski/Model-GGUF/tree/main", "bartowski/Model-GGUF"),
    ("qwen3-8b", None),
    ("hf:not-a-repo", None),
])
def test_parse_hub_ref(value, expected):
    assert hub.parse_hub_ref(value) == expected


class TestVariants:
    def test_finds_named_derivatives_of_the_same_model_only(self, catalog):
        spec = catalog.require("qwen3-8b")
        results = [
            {"id": "someone/Qwen3-8B-abliterated-GGUF", "downloads": 500},
            {"id": "popular/Qwen3-8B-heretic-GGUF", "downloads": 900},
            {"id": "other/Qwen3-80B-abliterated-GGUF", "downloads": 9999},  # a different size
            {"id": "plain/Qwen3-8B-GGUF", "downloads": 5000},                 # not a variant
            {"id": "nofiles/Qwen3-8B-abliterated-GGUF", "downloads": 100},    # publishes no GGUF
        ]
        repos = {
            "someone/Qwen3-8B-abliterated-GGUF": gguf_files("m.Q4_K_M.gguf", "m.Q8_0.gguf"),
            "popular/Qwen3-8B-heretic-GGUF": gguf_files("m-Q5_K_M.gguf", "mmproj-m-f16.gguf"),
            "nofiles/Qwen3-8B-abliterated-GGUF": gguf_files("README.md"),
        }
        found = hub.find_variants(spec, "abliterated", get_json=fake_hub(results, repos))
        assert [c["repo"] for c in found] == ["popular/Qwen3-8B-heretic-GGUF", "someone/Qwen3-8B-abliterated-GGUF"]
        assert found[0]["quants"] == ["Q5_K_M"] and found[0]["projector"]

    def test_a_variant_keeps_the_architecture_and_drops_the_original_weights(self, catalog):
        spec = catalog.require("qwen3-8b")
        candidate = {"repo": "someone/Qwen3-8B-abliterated-GGUF", "downloads": 1, "likes": 0,
                     "quants": ["Q4_K_M", "Q8_0"], "projector": False}
        entry = hub.variant_entry(spec, "abliterated", candidate)
        assert entry["id"] == "qwen3-8b-abliterated"
        assert entry["gguf_repo"] == candidate["repo"]
        for field in hub.SOURCE_FIELDS:
            assert field not in entry, field
        assert (entry["layers"], entry["kv_heads"]) == (spec.layers, spec.kv_heads)
        assert entry["recommended_quants"] == [q for q in spec.recommended_quants if q in candidate["quants"]]

    def test_vision_is_kept_only_when_the_variant_ships_a_projector(self, catalog):
        vision = next(m for m in catalog.models if m.supports_vision and m.gguf_repo)
        base = {"repo": "x/v-abliterated-GGUF", "downloads": 1, "likes": 0, "quants": ["Q4_K_M"]}
        assert hub.variant_entry(vision, "abliterated", {**base, "projector": True})["supports_vision"]
        assert not hub.variant_entry(vision, "abliterated", {**base, "projector": False})["supports_vision"]


class TestRepoEntry:
    def test_reads_architecture_from_the_base_model(self):
        repos = {"pub/Thing-7B-GGUF": {**gguf_files("Thing-7B.Q4_K_M.gguf"),
                                        "cardData": {"base_model": "org/Thing-7B"},
                                        "tags": ["license:apache-2.0"]}}
        configs = {"org/Thing-7B": {"num_hidden_layers": 28, "hidden_size": 3584,
                                    "num_attention_heads": 28, "num_key_value_heads": 4,
                                    "max_position_embeddings": 32768}}
        entry = hub.repo_entry("pub/Thing-7B-GGUF", get_json=fake_hub([], repos), config_for=configs.get)
        assert entry["params_b"] == 7.0 and entry["arch_source"] == "measured"
        assert (entry["layers"], entry["kv_heads"], entry["head_dim"]) == (28, 4, 128)
        assert entry["recommended_quants"] == ["Q4_K_M"] and entry["license"] == "apache-2.0"

    def test_without_a_config_the_architecture_is_marked_estimated(self):
        repos = {"pub/Mystery-3B-GGUF": gguf_files("Mystery-3B-Q4_K_M.gguf")}
        entry = hub.repo_entry("pub/Mystery-3B-GGUF", get_json=fake_hub([], repos), config_for=lambda r: None)
        assert entry["arch_source"] == "estimated" and "layers" not in entry

    def test_a_repo_without_gguf_files_is_refused(self):
        repos = {"org/Safetensors-Only": {"siblings": [{"rfilename": "model.safetensors"}]}}
        with pytest.raises(CatalogError, match="no GGUF"):
            hub.repo_entry("org/Safetensors-Only", get_json=fake_hub([], repos), config_for=lambda r: None)

    def test_a_repo_of_unknown_size_says_how_to_add_it(self):
        repos = {"pub/nameless-GGUF": gguf_files("model-Q4_K_M.gguf")}
        with pytest.raises(CatalogError, match="how large") as caught:
            hub.repo_entry("pub/nameless-GGUF", get_json=fake_hub([], repos), config_for=lambda r: None)
        assert "--params" in str(caught.value)


class TestCommand:
    def test_variant_needs_a_base_model(self, capsys):
        assert main(["plan", "--variant", "abliterated", "--task", "chat", "--yes"]) == 1
        assert "--model" in capsys.readouterr().err

    def test_offline_rules_out_a_hub_lookup(self, capsys):
        assert main(["plan", "--model", "hf:pub/Thing-7B-GGUF", "--offline", "--task", "chat", "--yes"]) == 1
        assert "--offline" in capsys.readouterr().err

    def test_an_hf_model_becomes_an_overlay_entry_the_planner_uses(self, monkeypatch, capsys):
        from llmsetup.util.fs import app_dir

        entry = {"id": "thing-7b", "name": "Thing 7B", "publisher": "pub", "params_b": 7.0,
                 "layers": 28, "hidden_size": 3584, "kv_heads": 4, "head_dim": 128, "max_context": 32768,
                 "tasks": ["chat"], "gguf_repo": "pub/Thing-7B-GGUF", "recommended_quants": ["Q4_K_M"],
                 "arch_source": "measured"}
        monkeypatch.setattr(hub, "repo_entry", lambda repo: dict(entry))
        assert main(["plan", "--model", "hf:pub/Thing-7B-GGUF", "--task", "chat", "--yes",
                     "--vram", "12", "--ram", "32", "--json"]) == 0
        saved = yaml.safe_load((app_dir() / "catalog.d" / "hub.yaml").read_text())
        assert [m["id"] for m in saved["models"]] == ["thing-7b"]
        assert "thing-7b" in capsys.readouterr().out
