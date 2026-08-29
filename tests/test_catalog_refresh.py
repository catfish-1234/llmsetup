"""`catalog add` and `catalog refresh`.

These exist because the catalog is a dated snapshot of a landscape that turns
over every few weeks. Both were referenced in user-facing error hints before
they existed, which is the specific gap these tests hold shut.
"""

from __future__ import annotations

import struct

import pytest
import yaml

from llmfit.catalog.catalog import load_catalog
from llmfit.catalog.refresh import add_model, refresh, user_catalog_path
from llmfit.util.errors import CatalogError
from llmfit.util.log import Console


@pytest.fixture
def console():
    return Console(quiet=True)


def write_gguf(path, layers=48, kv_heads=8, heads=32, hidden=4096, context=65536,
               template="{% if tools %}x{% endif %}", pad=0):
    def s(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    pairs = [
        ("general.architecture", 8, s("llama")),
        ("llama.block_count", 4, struct.pack("<I", layers)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", kv_heads)),
        ("llama.attention.head_count", 4, struct.pack("<I", heads)),
        ("llama.embedding_length", 4, struct.pack("<I", hidden)),
        ("llama.context_length", 4, struct.pack("<I", context)),
        ("tokenizer.chat_template", 8, s(template)),
    ]
    body = b"".join(s(k) + struct.pack("<I", t) + p for k, t, p in pairs)
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 400)
        + struct.pack("<Q", len(pairs)) + body + b"\x00" * pad
    )
    return path


class TestAdd:
    def test_reads_architecture_from_the_file(self, tmp_path, console):
        """Hand-typed layer and KV-head counts are how people plan a model into
        memory it does not fit, so the file is the source of truth."""
        gguf = write_gguf(tmp_path / "m.gguf", layers=48, kv_heads=8, hidden=4096)
        add_model(console, "my-model", gguf_path=gguf, publisher="acme")

        catalog = load_catalog()
        spec = catalog.require("my-model")
        assert spec.layers == 48
        assert spec.kv_heads == 8
        assert spec.hidden_size == 4096
        assert spec.head_dim == 128
        assert spec.max_context == 65536
        assert catalog.arch_source("my-model") == "measured"

    def test_infers_tool_support_from_the_chat_template(self, tmp_path, console):
        with_tools = write_gguf(tmp_path / "a.gguf", template="{% if tools %}x{% endif %}")
        add_model(console, "has-tools", gguf_path=with_tools)
        assert load_catalog().require("has-tools").supports_tools

        without = write_gguf(tmp_path / "b.gguf", template="{{ messages }}")
        add_model(console, "no-tools", gguf_path=without)
        assert not load_catalog().require("no-tools").supports_tools

    def test_works_without_a_file_when_params_are_given(self, console):
        add_model(console, "remote-only", params_b=7.0,
                  gguf_repo="acme/remote-only-GGUF", tasks=["chat"])
        spec = load_catalog().require("remote-only")
        assert spec.params_b == 7.0
        assert spec.gguf_repo == "acme/remote-only-GGUF"

    def test_refuses_without_enough_information(self, console):
        with pytest.raises(CatalogError) as excinfo:
            add_model(console, "mystery")
        assert "--params" in excinfo.value.hint

    def test_rejects_a_file_that_is_not_gguf(self, tmp_path, console):
        bogus = tmp_path / "not.gguf"
        bogus.write_bytes(b"definitely not a gguf")
        with pytest.raises(CatalogError):
            add_model(console, "bad", gguf_path=bogus)

    def test_rejects_a_missing_file(self, tmp_path, console):
        with pytest.raises(CatalogError):
            add_model(console, "gone", gguf_path=tmp_path / "absent.gguf")

    def test_adding_twice_replaces_rather_than_duplicates(self, tmp_path, console):
        gguf = write_gguf(tmp_path / "m.gguf", layers=48)
        add_model(console, "dupe", gguf_path=gguf)
        add_model(console, "dupe", gguf_path=write_gguf(tmp_path / "m2.gguf", layers=60))

        document = yaml.safe_load(user_catalog_path().read_text())
        ids = [m["id"] for m in document["models"]]
        assert ids.count("dupe") == 1
        assert load_catalog().require("dupe").layers == 60

    def test_the_added_model_is_usable_by_the_planner(self, tmp_path, console):
        """Writing the file is not the deliverable; a plannable model is."""
        from conftest import make_hardware

        from llmfit.planner.plan import build_plan
        from llmfit.types import Task, UseCaseProfile

        gguf = write_gguf(tmp_path / "m.gguf", layers=32, kv_heads=8, hidden=4096)
        add_model(console, "planner-visible", gguf_path=gguf, params_b=8.0,
                  tasks=["chat"], gguf_repo="acme/x-GGUF")

        catalog = load_catalog()
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)

        # It passes the capability filter alongside the bundled models...
        assert "planner-visible" in {m.id for m in catalog.matching(profile)}

        # ...and the planner can size and rank it for real hardware.
        plan = build_plan(profile, make_hardware(), catalog.restrict_to("planner-visible"))
        assert plan.best.model.id == "planner-visible"
        assert plan.best.estimate.total > 0
        assert plan.best.estimate.kv_cache > 0

    def test_the_overlay_is_valid_yaml_with_a_header(self, tmp_path, console):
        add_model(console, "yamltest", params_b=1.0)
        text = user_catalog_path().read_text()
        assert text.startswith("#")
        assert yaml.safe_load(text)["models"]


class TestRefresh:
    def test_reports_clean_when_nothing_has_drifted(self, catalog, console):
        report = refresh(console, catalog, check_remote=False)
        assert report["findings"] == []
        assert report["corrections"] == {}

    def test_detects_drift_against_a_local_file(self, tmp_path, catalog, console,
                                                monkeypatch):
        """The catalog says one thing, the file on disk says another. That is
        exactly the case where the memory plan would be wrong."""
        spec = catalog.require("qwen3-8b")
        store = tmp_path / "store" / spec.publisher / spec.id / "Q4_K_M"
        store.mkdir(parents=True)
        # Same model id, but a file claiming a different architecture.
        write_gguf(store / "w.gguf", layers=99, kv_heads=2, hidden=4096)
        monkeypatch.setenv("LLMFIT_MODEL_STORE", str(tmp_path / "store"))

        report = refresh(console, catalog, check_remote=False)
        drift = [f for f in report["findings"] if f["model"] == "qwen3-8b"]
        assert drift, "drift against a local file was not detected"
        assert "layers" in drift[0]["detail"]
        assert report["corrections"]["qwen3-8b"]["layers"] == 99

    def test_corrections_are_proposed_not_applied(self, tmp_path, catalog, console,
                                                  monkeypatch):
        """Silently rewriting the catalog would make a reproducible setup change
        underneath the user."""
        spec = catalog.require("qwen3-8b")
        store = tmp_path / "store" / spec.publisher / spec.id / "Q4_K_M"
        store.mkdir(parents=True)
        write_gguf(store / "w.gguf", layers=99)
        monkeypatch.setenv("LLMFIT_MODEL_STORE", str(tmp_path / "store"))

        refresh(console, catalog, check_remote=False)
        # The bundled catalog is untouched; the proposal sits beside it.
        assert load_catalog(include_user_dir=False).require("qwen3-8b").layers == 36

    def test_dry_run_writes_nothing(self, tmp_path, catalog, console, monkeypatch):
        spec = catalog.require("qwen3-8b")
        store = tmp_path / "store" / spec.publisher / spec.id / "Q4_K_M"
        store.mkdir(parents=True)
        write_gguf(store / "w.gguf", layers=99)
        monkeypatch.setenv("LLMFIT_MODEL_STORE", str(tmp_path / "store"))

        report = refresh(console, catalog, check_remote=False, write_overlay=False)
        assert report["corrections"]
        assert "overlay" not in report

    def test_offline_refresh_makes_no_network_calls(self, catalog, console, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("refresh --offline must not touch the network")

        monkeypatch.setattr("llmfit.download.list_repo_files", explode)
        refresh(console, catalog, check_remote=False)

    def test_an_unreachable_repo_is_reported_not_raised(self, catalog, console,
                                                        monkeypatch):
        """One bad entry must not abort the whole refresh."""
        def explode(repo, timeout=15):
            raise RuntimeError("network is down")

        monkeypatch.setattr("llmfit.download.list_repo_files", explode)
        report = refresh(console, catalog, check_remote=True)
        assert any(f["kind"] == "unreachable" for f in report["findings"])
