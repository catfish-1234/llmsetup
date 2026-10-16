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
    is_projector,
    list_repo_files,
    make_progress_printer,
    pick_gguf,
    pick_mmproj,
    saved_revision,
    sibling_shards,
)
from llmfit.types import ModelSpec
from llmfit.util.errors import ProvisionError
from llmfit.util.fs import ensure_dir, model_store, safe_subpath
from llmfit.util.log import Console


def store_path(spec: ModelSpec, quant: str) -> Path:
    """Where a model's weights live. Mirrors publisher/model so it reads like HF.

    Every component is sanitised: these values come from catalog YAML that users
    can extend or share, and an unsanitised id would let an entry place a
    multi-gigabyte download anywhere on disk.
    """
    directory = safe_subpath(model_store(), spec.publisher, spec.id, quant)
    revision = spec.gguf_revision or (saved_revision(spec.gguf_repo) if spec.gguf_repo else None)
    return safe_subpath(directory, revision) if revision else directory


def find_local(spec: ModelSpec, quant: str) -> Optional[Path]:
    """An already-downloaded GGUF for this (model, quant), if any."""
    directory = store_path(spec, quant)
    if not directory.is_dir():
        return None
    ggufs = sorted(p for p in directory.glob("*.gguf") if not is_projector(p.name))
    if not ggufs:
        return None
    # For a split model, the first shard is the one runtimes are pointed at.
    return ggufs[0]


def find_local_projector(spec: ModelSpec, quant: str) -> Optional[Path]:
    """An already-downloaded multimodal projector for this (model, quant)."""
    directory = store_path(spec, quant)
    if not directory.is_dir():
        return None
    found = sorted(p for p in directory.glob("*.gguf") if is_projector(p.name))
    return found[0] if found else None


def fetch_projector(spec: ModelSpec, quant: str, console: Console) -> Optional[Path]:
    """Fetch the vision projector next to the weights, if the repo has one.

    A vision model is two files. Loading only the language model gives an
    endpoint that accepts an image and then describes something that is not in
    it -- the failure looks like a bad model rather than a missing file, which
    is exactly why it is worth fetching up front rather than diagnosing later.

    Returns ``None`` when the repo publishes no projector; the caller decides
    whether that is fatal, because it only matters for a model that claims
    vision.
    """
    existing = find_local_projector(spec, quant)
    if existing:
        console.ok(f"vision projector already present: {existing}")
        return existing
    if not spec.gguf_repo:
        return None

    try:
        files = list_repo_files(spec.gguf_repo, revision=spec.gguf_revision)
    except Exception as exc:  # the projector is never worth aborting the run for
        console.warn(f"could not list {spec.gguf_repo} for a vision projector: {exc}")
        return None

    chosen = pick_mmproj(files)
    if chosen is None:
        return None

    directory = ensure_dir(store_path(spec, quant))
    dest = safe_subpath(directory, Path(chosen.filename).name)
    download_file(chosen, dest, make_progress_printer(console, Path(chosen.filename).name))
    console.ok(f"vision projector ready: {dest}")
    return dest


def fetch_gguf(spec: ModelSpec, quant: str, console: Console) -> Path:
    """Ensure the GGUF is in the shared store and return its path."""
    if spec.gguf_repo and not spec.gguf_revision:
        spec.gguf_revision = saved_revision(spec.gguf_repo)
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
    files = list_repo_files(spec.gguf_repo, revision=spec.gguf_revision)
    if files:
        # Keep the resolved pin in the saved setup as well as the local lock.
        spec.gguf_revision = files[0].revision
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
        dest = safe_subpath(directory, Path(shard.filename).name)
        download_file(shard, dest, make_progress_printer(console, Path(shard.filename).name))
        first = first or dest

    assert first is not None
    console.ok(f"weights ready: {first}")
    return first
