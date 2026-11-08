"""Check installed wheel resources from outside the source checkout."""

import importlib.metadata
from importlib.resources import files

from llmsetup.catalog.catalog import load_catalog

metadata = importlib.metadata.metadata("llmsetup")
distribution = importlib.metadata.distribution("llmsetup")
assert metadata["Name"] == "llmsetup"
assert metadata["Version"] == "0.1.1"
assert distribution.read_text("top_level.txt").strip() == "llmsetup"
assert {entry.name: entry.value for entry in distribution.entry_points
        if entry.group == "console_scripts"} == {"llmsetup": "llmsetup.cli:main"}
assert metadata.get("License-Expression") == "Apache-2.0" or \
    metadata.get("License") == "Apache-2.0"
assert len(load_catalog(include_user_dir=False)) > 10
assert "<html" in files("llmsetup.ui").joinpath("static/index.html").read_text().lower()
assert any(str(path).endswith("/LICENSE") for path in importlib.metadata.files("llmsetup"))
print("Installed wheel metadata, catalog, UI, and license are present.")
