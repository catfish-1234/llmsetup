"""Shared GGUF store.

Every GGUF-based runtime resolves weights through here, so a model is downloaded
once and then *linked* into each application's directory. That single decision is
what makes "I can still see it in LM Studio" true without doubling disk usage per
app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llmfit.download import (
    download_file,
    list_repo_files,
    make_progress_printer,
    pick_gguf,
    sibling_shards,
)
from llmfit.types import ModelSpec
from llmfit.util.errors import ProvisionError
from llmfit.util.fs import ensure_dir, model_store
from llmfit.util.log import Console


def store_path(spec: ModelSpec, quant: str) -> Path:
    """Where a model's weights live. Mirrors publisher/model so it reads like HF."""
    return model_store() / spec.publisher / spec.id / quant


def find_local(spec: ModelSpec, quant: str) -> Optional[Path]:
    """An already-downloaded GGUF for this (model, quant), if any."""
    directory = store_path(spec, quant)
    if not directory.is_dir():
        return None
    ggufs = sorted(directory.glob("*.gguf"))
    if not ggufs:
        return None
    # For a split model, the first shard is the one runtimes are pointed at.
    return ggufs[0]


def fetch_gguf(spec: ModelSpec, quant: str, console: Console) -> Path:
    """Ensure the GGUF is in the shared store and return its path."""
    existing = find_local(spec, quant)
    if existing:
        console.ok(f"weights already present: {existing}")
        return existing

    if not spec.gguf_repo:
        raise ProvisionError(
            f"{spec.name} has no GGUF repo in the catalog",
            hint="pick a different runtime, or add `gguf_repo:` to the catalog entry",
        )

    console.info(f"listing {spec.gguf_repo} …")
    files = list_repo_files(spec.gguf_repo)
    chosen = pick_gguf(files, quant)
    if chosen is None:
        available = sorted({
            part for f in files if f.filename.lower().endswith(".gguf")
            for part in [f.filename]
        })[:8]
        raise ProvisionError(
            f"{spec.gguf_repo} has no {quant} build",
            hint=f"available files include: {', '.join(available)}",
        )

    shards = sibling_shards(files, chosen)
    directory = ensure_dir(store_path(spec, quant))
    if len(shards) > 1:
        console.info(f"{spec.name} {quant} is split across {len(shards)} files")

    first: Optional[Path] = None
    for shard in shards:
        dest = directory / Path(shard.filename).name
        download_file(shard, dest, make_progress_printer(console, Path(shard.filename).name))
        first = first or dest

    assert first is not None
    console.ok(f"weights ready: {first}")
    return first
