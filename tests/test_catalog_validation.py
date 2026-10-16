"""Malformed bundled entries and overlays fail at the catalog boundary."""

import pytest
import yaml

from llmfit.catalog.catalog import load_catalog
from llmfit.util.errors import CatalogError


@pytest.mark.parametrize("patch,field", [
    ({"params_b": float("nan")}, "params_b"),
    ({"params_b": float("inf")}, "params_b"),
    ({"params_b": None}, "params_b"),
    ({"layers": 3.5}, "layers"),
    ({"supports_tools": "false"}, "supports_tools"),
    ({"tasks": ["caht"]}, "tasks"),
    ({"tasks": [None]}, "tasks"),
    ({"quality_tier": 6}, "quality_tier"),
    ({"active_params_b": 100}, "active_params_b"),
    ({"recommended_quants": []}, "recommended_quants"),
    ({"name": None}, "name"),
    ({"hf_revision": "main"}, "hf_revision"),
    ({"hf_repo": "../invalid"}, "hf_repo"),
    ({"gguf_repo": []}, "gguf_repo"),
])
def test_bad_overlay_names_model_field_and_file(tmp_path, patch, field):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"models": [{"id": "qwen3-8b", **patch}]}))
    with pytest.raises(CatalogError) as error:
        load_catalog([path], include_user_dir=False)
    assert all(part in str(error.value) for part in ("qwen3-8b", field, str(path)))


@pytest.mark.parametrize("data", [
    [], {"schema_version": 2}, {"schema_version": True},
    {"models": {}}, {"quant_bpw": []}, {"quant_bpw": {"Q4": -1}},
    {"models": [{"id": []}]},
    {"models": [{"id": "qwen3-8b"}, {"id": "qwen3-8b"}]},
])
def test_invalid_catalog_structure_is_a_catalog_error(tmp_path, data):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(CatalogError, match=r"bad\.yaml"):
        load_catalog([path], include_user_dir=False)


def test_partial_overlay_and_numeric_strings_still_work(tmp_path):
    path = tmp_path / "overlay.yaml"
    path.write_text('models:\n- id: qwen3-8b\n  params_b: "8.2"\n  supports_tools: false\n')
    model = load_catalog([path], include_user_dir=False).require("qwen3-8b")
    assert model.params_b == 8.2
    assert model.layers == 36
    assert model.supports_tools is False
