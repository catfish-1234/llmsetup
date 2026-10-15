"""Check installed wheel resources from outside the source checkout."""

import importlib.metadata
from importlib.resources import files

from llmfit.catalog.catalog import load_catalog

metadata = importlib.metadata.metadata("llmfit")
assert metadata["Name"] == "llmfit"
assert metadata["Version"] == "0.1.0"
assert metadata.get("License-Expression") == "Apache-2.0" or \
    metadata.get("License") == "Apache-2.0"
assert len(load_catalog(include_user_dir=False)) > 10
assert "<html" in files("llmfit.ui").joinpath("static/index.html").read_text().lower()
assert any(str(path).endswith("/LICENSE") for path in importlib.metadata.files("llmfit"))
print("Installed wheel metadata, catalog, UI, and license are present.")
