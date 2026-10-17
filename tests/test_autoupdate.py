"""Automatic catalog updating.

Two mechanisms, tested separately:

* **discovery** turns a Hugging Face listing into catalog entries;
* **auto-update** lets an installed copy pull a newer catalog on its own.

Neither may ever break a run. The network is faked throughout, the suite makes
no real requests.
"""

from __future__ import annotations

import json

import pytest
import yaml

from llmsetup.catalog import autoupdate
from llmsetup.catalog.autoupdate import (
    apply_update,
    disable_persistently,
    enable_persistently,
    enabled,
    is_due,
    maybe_update,
    status,
    validate_catalog_text,
)
from llmsetup.catalog.catalog import load_catalog
from llmsetup.catalog.discover import (
    Candidate,
    architecture_from_config,
    discover,
    params_from_metadata,
    slug_for,
)
from llmsetup.util.log import Console


@pytest.fixture
def console():
    return Console(quiet=True)


def valid_catalog_text(model_id="fetched-model", updated="2099-01-01"):
    return yaml.safe_dump({
        "schema_version": 1,
        "updated": updated,
        "quant_bpw": {"Q4_K_M": 4.85},
        "models": [{
            "id": model_id, "name": "Fetched", "publisher": "acme",
            "params_b": 8.0, "layers": 32, "hidden_size": 4096, "kv_heads": 8,
            "head_dim": 128, "max_context": 32768, "license": "Apache-2.0",
            "tasks": ["chat"], "gguf_repo": "acme/fetched-GGUF",
        }],
    })


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


class TestArchitectureExtraction:
    """Architecture comes from config.json, which is the publisher's own
    statement, so these must match published configs exactly."""

    @pytest.mark.parametrize("config,expected", [
        ({"model_type": "qwen3", "num_hidden_layers": 36, "hidden_size": 4096,
              "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
              "max_position_embeddings": 40960}, (36, 4096, 8, 128)),
        ({"model_type": "llama", "num_hidden_layers": 32, "hidden_size": 4096,
              "num_attention_heads": 32, "num_key_value_heads": 8,
              "max_position_embeddings": 131072}, (32, 4096, 8, 128)),
        ({"model_type": "gpt_oss", "num_hidden_layers": 24, "hidden_size": 2880,
              "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
              "max_position_embeddings": 131072}, (24, 2880, 8, 64)),
    ])
    def test_matches_published_configs(self, config, expected):
        arch = architecture_from_config(config)
        assert (arch["layers"], arch["hidden_size"], arch["kv_heads"],
                arch["head_dim"]) == expected

    def test_reads_nested_multimodal_configs(self):
        """Vision models put the language model under text_config."""
        arch = architecture_from_config({
            "model_type": "gemma3",
            "vision_config": {"hidden_size": 1152},
            "text_config": {"num_hidden_layers": 62, "hidden_size": 5376,
                                "num_attention_heads": 32, "num_key_value_heads": 16,
                                "head_dim": 128, "max_position_embeddings": 131072},
        })
        assert arch["layers"] == 62 and arch["kv_heads"] == 16
        assert arch["vision"] is True

    def test_falls_back_to_multi_head_when_gqa_is_unstated(self):
        """Assuming kv_heads == heads over-estimates the cache, which is the
        safe direction to be wrong in."""
        arch = architecture_from_config({"num_hidden_layers": 32, "hidden_size": 4096,
                                             "num_attention_heads": 32})
        assert arch["kv_heads"] == 32

    @pytest.mark.parametrize("config", [
        {}, {"hidden_size": 4096}, {"num_hidden_layers": 32},
        {"num_hidden_layers": 32, "hidden_size": 100, "num_attention_heads": 32},
    ])
    def test_refuses_to_guess_from_an_incomplete_config(self, config):
        """A guessed layer count is a wrong memory plan and an OOM at inference
        time, so no entry is better than an approximate one."""
        assert architecture_from_config(config) is None


class TestParameterCount:
    def test_prefers_the_exact_safetensors_total(self):
        assert params_from_metadata(
            {"safetensors": {"total": 8_030_000_000}}, "x/y") == pytest.approx(8.03)

    def test_falls_back_to_the_name(self):
        assert params_from_metadata({}, "Qwen/Qwen3-8B") == 8.0
        assert params_from_metadata({}, "Qwen/Qwen3-30B-A3B") == 30.0

    def test_returns_none_when_undeterminable(self):
        assert params_from_metadata({}, "acme/mystery-model") is None


class TestSlugs:
    @pytest.mark.parametrize("repo,slug", [
        ("Qwen/Qwen3-8B-Instruct", "qwen3-8b"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama-3.3-70b"),
        ("openai/gpt-oss-120b", "gpt-oss-120b"),
    ])
    def test_slugs_match_existing_catalog_ids(self, repo, slug, catalog):
        """Dedupe depends on a discovered slug colliding with the curated id."""
        assert slug_for(repo) == slug
        assert catalog.get(slug) is not None


class TestDiscovery:
    def _listing(self, monkeypatch, items, configs):
        monkeypatch.setattr("llmsetup.catalog.discover.list_trending",
                            lambda limit=60, timeout=20.0: items)
        monkeypatch.setattr("llmsetup.catalog.discover.fetch_config",
                            lambda repo, timeout=20.0: configs.get(repo))
        monkeypatch.setattr("llmsetup.catalog.discover.find_gguf_repo",
                            lambda repo, timeout=20.0: f"{repo}-GGUF")

    def test_discovers_a_new_model(self, console, monkeypatch):
        self._listing(monkeypatch,
            [{"id": "acme/NewModel-8B", "downloads": 100_000, "likes": 300,
              "tags": ["license:apache-2.0", "instruct"]}],
            {"acme/NewModel-8B": {"model_type": "llama", "num_hidden_layers": 32,
                                      "hidden_size": 4096, "num_attention_heads": 32,
                                      "num_key_value_heads": 8,
                                      "max_position_embeddings": 131072}})
        found = discover(console, known_ids=[])
        assert len(found) == 1
        entry = found[0].to_entry()
        assert entry["layers"] == 32 and entry["kv_heads"] == 8
        assert entry["license"] == "apache-2.0"
        assert entry["commercial_ok"] is True
        assert entry["arch_source"] == "measured"

    def test_skips_models_already_in_the_catalog(self, console, monkeypatch, catalog):
        self._listing(monkeypatch,
            [{"id": "Qwen/Qwen3-8B", "downloads": 999_999, "likes": 999,
              "tags": ["license:apache-2.0"]}],
            {"Qwen/Qwen3-8B": {"model_type": "qwen3", "num_hidden_layers": 36,
                                   "hidden_size": 4096, "num_attention_heads": 32,
                                   "num_key_value_heads": 8}})
        assert discover(console, known_ids=[m.id for m in catalog.models]) == []

    def test_skips_unpopular_models(self, console, monkeypatch):
        self._listing(monkeypatch,
            [{"id": "acme/Obscure-8B", "downloads": 5, "likes": 0, "tags": []}], {})
        assert discover(console, known_ids=[]) == []

    def test_skips_models_with_an_unreadable_config(self, console, monkeypatch):
        self._listing(monkeypatch,
            [{"id": "acme/Opaque-8B", "downloads": 100_000, "likes": 300,
              "tags": []}], {})
        assert discover(console, known_ids=[]) == []

    def test_skips_junk_by_name(self, console, monkeypatch):
        self._listing(monkeypatch,
            [{"id": "acme/test-random-debug-8B", "downloads": 999_999,
              "likes": 999, "tags": []}], {})
        assert discover(console, known_ids=[]) == []

    def test_an_unreachable_hub_yields_nothing_rather_than_raising(self, console,
                                                                   monkeypatch):
        monkeypatch.setattr("llmsetup.catalog.discover.list_trending",
                            lambda limit=60, timeout=20.0: [])
        assert discover(console, known_ids=[]) == []

    def test_an_unknown_licence_is_not_marked_commercial_ok(self):
        """Wrongly telling someone they may ship commercially is the expensive
        direction to be wrong in."""
        entry = Candidate(id="x", repo="a/x", publisher="a", name="X", params_b=8,
                          layers=32, hidden_size=4096, kv_heads=8, head_dim=128,
                          max_context=8192, license="unknown").to_entry()
        assert entry["commercial_ok"] is False


# --------------------------------------------------------------------------- #
# runtime auto-update
# --------------------------------------------------------------------------- #


class TestValidationOfDownloads:
    def test_accepts_a_well_formed_catalog(self):
        assert validate_catalog_text(valid_catalog_text()) is not None

    @pytest.mark.parametrize("text", [
        "<html>404 Not Found</html>",
        "not: valid: yaml: [",
        yaml.safe_dump({"models": []}),
        yaml.safe_dump({"models": [{"name": "no id"}]}),
        yaml.safe_dump(["a", "list"]),
        "",
    ])
    def test_rejects_anything_malformed(self, text):
        """A truncated download or an HTML error page must never replace a
        working catalog."""
        assert validate_catalog_text(text) is None


class TestApplyUpdate:
    def _serve(self, monkeypatch, text):
        monkeypatch.setattr(autoupdate, "fetch_catalog",
                            lambda url, timeout=6.0: text)

    def test_caches_a_valid_catalog_and_makes_it_loadable(self, console, monkeypatch):
        self._serve(monkeypatch, valid_catalog_text("fetched-model"))
        result = apply_update(console, force=True)
        assert result["updated"] is True
        # The fetched model is now visible to the loader.
        assert load_catalog().get("fetched-model") is not None

    def test_a_failed_fetch_leaves_the_bundled_catalog_intact(self, console,
                                                              monkeypatch, catalog):
        self._serve(monkeypatch, None)
        result = apply_update(console, force=True)
        assert result["updated"] is False
        assert len(load_catalog()) == len(catalog)

    def test_a_malformed_download_is_never_adopted(self, console, monkeypatch,
                                                   catalog):
        self._serve(monkeypatch, "<html>500</html>")
        assert apply_update(console, force=True)["updated"] is False
        assert len(load_catalog()) == len(catalog)

    def test_a_catalog_that_does_not_load_is_rejected(self, console, monkeypatch,
                                                      catalog):
        """Shape-valid but semantically broken: it must fail the real loader."""
        self._serve(monkeypatch, yaml.safe_dump({
            "models": [{"id": "bad", "params_b": "not-a-number", "layers": "x"}]
        }))
        apply_update(console, force=True)
        assert load_catalog().get("bad") is None

    def test_user_entries_are_never_clobbered(self, console, monkeypatch):
        """The auto-update cache is one file; the user's own overlays sit beside
        it and are layered on top."""
        from llmsetup.catalog.refresh import add_model

        add_model(console, "my-own-model", params_b=7.0, gguf_repo="me/mine-GGUF")
        self._serve(monkeypatch, valid_catalog_text("fetched-model"))
        apply_update(console, force=True)

        catalog = load_catalog()
        assert catalog.get("my-own-model") is not None
        assert catalog.get("fetched-model") is not None


class TestScheduling:
    def test_first_run_is_due(self):
        assert is_due() is True

    def test_not_due_immediately_after_a_check(self, console, monkeypatch):
        monkeypatch.setattr(autoupdate, "fetch_catalog",
                            lambda url, timeout=6.0: valid_catalog_text())
        apply_update(console, force=True)
        assert is_due() is False

    def test_a_non_forced_update_respects_the_interval(self, console, monkeypatch):
        calls = {"n": 0}

        def counting(url, timeout=6.0):
            calls["n"] += 1
            return valid_catalog_text()

        monkeypatch.setattr(autoupdate, "fetch_catalog", counting)
        apply_update(console, force=True)
        apply_update(console)          # too soon
        apply_update(console)
        assert calls["n"] == 1


class TestOptOut:
    def test_env_var_disables_it(self, monkeypatch):
        monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")
        assert enabled() is False

    def test_marker_file_disables_it_persistently(self, monkeypatch):
        monkeypatch.delenv("LLMSETUP_AUTO_UPDATE", raising=False)
        assert enabled() is True
        disable_persistently()
        assert enabled() is False
        enable_persistently()
        assert enabled() is True

    def test_disabled_means_no_network_call(self, console, monkeypatch):
        monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")

        def explode(*args, **kwargs):
            raise AssertionError("must not fetch when disabled")

        monkeypatch.setattr(autoupdate, "fetch_catalog", explode)
        assert maybe_update(console)["updated"] is False

    def test_offline_profiles_never_fetch(self, console, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("must not fetch for an offline profile")

        monkeypatch.setattr(autoupdate, "fetch_catalog", explode)
        assert maybe_update(console, offline=True)["updated"] is False


class TestNeverBreaksTheRun:
    def test_an_exploding_fetch_is_swallowed(self, console, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("network on fire")

        monkeypatch.setattr(autoupdate, "fetch_catalog", explode)
        assert maybe_update(console)["updated"] is False

    def test_status_is_reportable_before_any_update(self):
        state = status()
        assert set(state) >= {"enabled", "url", "interval_hours", "due"}
        assert json.dumps(state)          # serialisable for --json


class TestNonHttpSources:
    """An air-gapped install can point LLMSETUP_CATALOG_URL at an internal file
    or server; those responses carry no HTTP status code."""

    def test_a_file_url_is_a_valid_source(self, tmp_path, console, monkeypatch):
        published = tmp_path / "internal.yaml"
        published.write_text(valid_catalog_text("internal-model"))
        result = apply_update(console, force=True, url=published.as_uri())
        assert result["updated"] is True
        assert load_catalog().get("internal-model") is not None


class TestUnattendedCommitGate:
    """The validation that makes the scheduled workflow safe to run unattended.

    Discovered entries are machine-generated, so the gate has to reject a bad
    one before it reaches the catalog. A wrong layer count is a wrong memory
    plan and an out-of-memory error on a user's machine.
    """

    @pytest.fixture
    def validate_fn(self):
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
        from update_catalog import validate

        return validate

    @pytest.fixture
    def base_entry(self):
        return {
            "id": "good-8b", "name": "Good", "publisher": "acme", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 32768, "license": "Apache-2.0", "tasks": ["chat"],
            "gguf_repo": "acme/g-GGUF", "recommended_quants": ["Q4_K_M"],
        }

    def _catalog_with(self, entry):
        from llmsetup.catalog.catalog import BUNDLED

        document = yaml.safe_load(BUNDLED.read_text(encoding="utf-8"))
        document["models"].append(entry)
        return yaml.safe_dump(document, sort_keys=False)

    def test_a_well_formed_entry_passes(self, validate_fn, base_entry, console):
        assert validate_fn(self._catalog_with(base_entry), console, ["good-8b"])

    @pytest.mark.parametrize("field,value", [
        ("layers", 9999),
        ("hidden_size", 0),
        ("params_b", 99999),
        ("kv_heads", 9999),
        ("max_context", 1),
    ])
    def test_implausible_values_are_rejected(self, validate_fn, base_entry, console,
                                             field, value):
        entry = {**base_entry, "id": "suspect", field: value}
        assert validate_fn(self._catalog_with(entry), console, ["suspect"]) is False
