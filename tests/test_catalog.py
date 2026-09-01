"""Catalog integrity and filtering.

The catalog is data, and data rots. These tests are the guard rail that stops a
malformed or implausible entry reaching the planner.
"""

from __future__ import annotations

import pytest

from llmfit.catalog.catalog import load_catalog
from llmfit.types import Task, UseCaseProfile
from llmfit.util.errors import CatalogError


class TestIntegrity:
    def test_loads(self, catalog):
        assert len(catalog) > 15

    def test_ids_are_unique(self, catalog):
        ids = [m.id for m in catalog.models]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("field,low,high", [
        ("params_b", 0.05, 2000),
        ("layers", 1, 200),
        ("kv_heads", 1, 128),
        ("hidden_size", 128, 32768),
        ("max_context", 512, 10_000_000),
    ])
    def test_architecture_values_are_plausible(self, catalog, field, low, high):
        for model in catalog.models:
            value = getattr(model, field)
            assert low <= value <= high, f"{model.id}.{field} = {value}"

    def test_every_model_is_reachable_by_some_runtime(self, catalog):
        for model in catalog.models:
            assert model.gguf_repo or model.ollama_name or model.hf_repo, \
                f"{model.id} has no way to obtain weights"

    def test_every_model_declares_a_quantization(self, catalog):
        for model in catalog.models:
            assert model.recommended_quants, f"{model.id} lists no quants"
            for quant in model.recommended_quants:
                assert catalog.bpw(quant) > 0

    def test_moe_active_params_are_below_total(self, catalog):
        for model in catalog.models:
            if model.active_params_b is not None:
                assert model.active_params_b < model.params_b, model.id

    def test_head_dim_is_consistent_with_hidden_size(self, catalog):
        """A head_dim that does not divide the hidden size is a data-entry slip."""
        for model in catalog.models:
            if model.head_dim:
                assert model.hidden_size % model.head_dim == 0 or \
                    model.head_dim in (64, 128, 256), model.id

    def test_every_chat_model_declares_tasks(self, catalog):
        for model in catalog.chat_models():
            assert model.tasks, f"{model.id} declares no tasks"

    def test_embedding_models_exist_for_rag(self, catalog):
        assert len(catalog.embedding_models()) >= 2

    def test_estimated_architectures_are_marked(self, catalog):
        """Anything not taken from a published config must say so, because the
        planner surfaces that caveat to the user."""
        for model in catalog.models:
            assert catalog.arch_source(model.id) in ("measured", "estimated")
        # At least one entry is genuinely marked estimated, otherwise this test
        # would pass just as happily against a catalog with no provenance at all.
        assert any(catalog.arch_source(m.id) == "estimated" for m in catalog.models)

    def test_provenance_is_per_catalog_not_global(self, tmp_path):
        """Two Catalog objects in one process must not report each other's
        provenance, the reason this moved off a module global."""
        override = tmp_path / "override.yaml"
        override.write_text("models:\n  - id: qwen3.8-27b\n    arch_source: measured\n")
        bundled = load_catalog(include_user_dir=False)
        patched = load_catalog(extra_paths=[override], include_user_dir=False)
        assert bundled.arch_source("qwen3.8-27b") == "estimated"
        assert patched.arch_source("qwen3.8-27b") == "measured"

    def test_restrict_to_returns_a_consistent_view(self, catalog):
        """`--model` and `doctor` narrow the catalog; the index must stay in step."""
        narrowed = catalog.restrict_to("qwen3-8b")
        assert [m.id for m in narrowed.models] == ["qwen3-8b"]
        assert narrowed.require("qwen3-8b") is not None
        assert narrowed.get("gemma-3-27b") is None
        # The original is untouched.
        assert len(catalog) > 1


class TestFiltering:
    def test_vision_requirement_excludes_text_only_models(self, catalog):
        profile = UseCaseProfile(needs_vision=True, tasks=[Task.CHAT])
        matched = catalog.matching(profile)
        assert matched
        assert all(m.supports_vision for m in matched)

    def test_tool_requirement_excludes_models_without_tools(self, catalog):
        profile = UseCaseProfile(needs_tools=True, tasks=[Task.AGENT])
        assert all(m.supports_tools for m in catalog.matching(profile))

    def test_context_requirement_excludes_short_window_models(self, catalog):
        profile = UseCaseProfile(context_tokens=100_000, tasks=[Task.CHAT])
        matched = catalog.matching(profile)
        assert all(m.max_context >= 100_000 for m in matched)
        # Phi-4 has a 16K window and must be excluded.
        assert "phi-4-14b" not in {m.id for m in matched}

    def test_commercial_filter_respects_licences(self, catalog):
        profile = UseCaseProfile(commercial_use=True, tasks=[Task.CHAT])
        assert all(m.commercial_ok for m in catalog.matching(profile))

    def test_embedding_models_never_appear_as_chat_candidates(self, catalog):
        matched = catalog.matching(UseCaseProfile(tasks=[Task.CHAT]))
        assert not any(m.is_embedding for m in matched)

    def test_unmatched_reasons_explain_exclusions(self, catalog):
        profile = UseCaseProfile(needs_vision=True, context_tokens=200_000,
                                 tasks=[Task.CHAT])
        reasons = catalog.unmatched_reasons(profile)
        assert reasons
        assert sum(reasons.values()) == len(catalog.chat_models()) - \
            len(catalog.matching(profile))


class TestLookup:
    def test_bpw_is_case_insensitive(self, catalog):
        assert catalog.bpw("q4_k_m") == catalog.bpw("Q4_K_M")

    def test_unknown_quant_falls_back_rather_than_raising(self, catalog):
        assert catalog.bpw("NOT_A_QUANT") > 0

    def test_unknown_model_raises_with_a_suggestion(self, catalog):
        with pytest.raises(CatalogError) as excinfo:
            catalog.require("qwen3")
        assert excinfo.value.hint


class TestUserExtensions:
    def test_a_user_file_can_add_a_model(self, tmp_path):
        extra = tmp_path / "extra.yaml"
        extra.write_text(
            "models:\n"
            "  - id: my-model\n"
            "    name: My Model\n"
            "    publisher: me\n"
            "    params_b: 7\n"
            "    tasks: [chat]\n"
            "    gguf_repo: me/my-model-GGUF\n"
        )
        catalog = load_catalog(extra_paths=[extra], include_user_dir=False)
        assert catalog.get("my-model") is not None

    def test_a_user_file_can_patch_one_field_of_an_existing_model(self, tmp_path):
        extra = tmp_path / "patch.yaml"
        extra.write_text("models:\n  - id: qwen3-8b\n    ollama_name: my-custom-tag\n")
        catalog = load_catalog(extra_paths=[extra], include_user_dir=False)
        model = catalog.require("qwen3-8b")
        assert model.ollama_name == "my-custom-tag"
        # The rest of the entry survives the patch.
        assert model.layers == 36
        assert model.name == "Qwen3 8B"

    def test_a_malformed_file_reports_which_file(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("models:\n  - name: no id here\n")
        with pytest.raises(CatalogError) as excinfo:
            load_catalog(extra_paths=[bad], include_user_dir=False)
        assert "bad.yaml" in str(excinfo.value)
