"""Keeping the catalog current.

The open-weight landscape turns over every few weeks, so a shipped catalog is
stale the moment it is written. Two ways to deal with that, both implemented
here because both are things a user actually needs:

* ``add``. Record a model you already know about, with its architecture read
  from a local GGUF where possible so the memory maths is right.
* ``refresh``. Check the catalog's entries against their upstream repos and
  report what has drifted.

Refresh deliberately **reports** rather than silently rewriting: a model catalog
that changes underneath you turns a reproducible setup into a moving target. It
writes a proposed overlay you can inspect and then apply.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llmfit.catalog.catalog import Catalog
from llmfit.gguf import read_gguf_info
from llmfit.types import ModelSpec
from llmfit.util.errors import CatalogError
from llmfit.util.fs import app_dir, ensure_dir, write_text
from llmfit.util.log import Console


def user_catalog_dir() -> Path:
    return ensure_dir(app_dir() / "catalog.d")


def user_catalog_path(name: str = "local.yaml") -> Path:
    return user_catalog_dir() / name


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


def add_model(
    console: Console,
    model_id: str,
    *,
    gguf_path: Optional[Path] = None,
    gguf_repo: Optional[str] = None,
    ollama_name: Optional[str] = None,
    hf_repo: Optional[str] = None,
    params_b: Optional[float] = None,
    publisher: Optional[str] = None,
    name: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    license_name: str = "unknown",
    context: Optional[int] = None,
    supports_tools: bool = False,
    supports_vision: bool = False,
    quants: Optional[List[str]] = None,
) -> Path:
    """Append a model to the user's overlay catalog.

    When a local GGUF is given, its header supplies the architecture. That
    matters more than convenience: layer count and KV head count drive the whole
    memory model, and a hand-typed guess is how people end up planning a model
    into memory it does not fit.
    """
    entry: Dict[str, Any] = {
        "id": model_id,
        "name": name or model_id,
        "publisher": publisher or (gguf_repo or hf_repo or "local").split("/")[0],
        "license": license_name,
        "tasks": tasks or ["chat"],
        "supports_tools": supports_tools,
        "supports_vision": supports_vision,
        "recommended_quants": quants or ["Q4_K_M"],
        "arch_source": "estimated",
    }

    if gguf_path:
        path = Path(gguf_path).expanduser()
        if not path.exists():
            raise CatalogError(f"{path} does not exist")
        info = read_gguf_info(path)
        if info is None:
            raise CatalogError(
                f"{path} is not a readable GGUF file",
                hint="pass --params to add the model without reading its header",
            )
        entry.update({
            "layers": info.layers,
            "hidden_size": info.embedding_length,
            "kv_heads": info.kv_heads,
            "head_dim": info.head_dim,
            "max_context": info.context_length or 8192,
            "arch_source": "measured",
        })
        if info.supports_tools is not None:
            entry["supports_tools"] = bool(info.supports_tools)
        # Recover the parameter count from the file size and its quantization.
        if params_b is None:
            params_b = _params_from_file(path, entry.get("layers"))
        console.ok(
            f"read architecture from the file: {info.layers} layers, "
            f"{info.kv_heads} KV heads, {info.head_dim} head dim"
        )
        entry["gguf_path"] = str(path)

    if params_b is None:
        raise CatalogError(
            "cannot determine the parameter count",
            hint=("pass --params (in billions). If you gave --gguf, the file is "
                  "too small for its size to imply a plausible parameter count."),
        )
    entry["params_b"] = round(float(params_b), 3)

    for key, value in (("gguf_repo", gguf_repo), ("ollama_name", ollama_name),
                       ("hf_repo", hf_repo)):
        if value:
            entry[key] = value
    if context:
        entry["max_context"] = int(context)
    entry.setdefault("max_context", 8192)

    return _append_entry(entry, console)


#: Smallest parameter count that could plausibly be a real model, in billions.
#: Below this the file-size estimate is noise (a truncated download, a test
#: fixture, a stub), and a bogus figure is worse than asking the user for one.
_MIN_PLAUSIBLE_PARAMS_B = 0.05


def _params_from_file(path: Path, layers: Optional[int]) -> Optional[float]:
    """Estimate parameter count from file size, assuming ~4.85 bits/weight.

    Rough, and labelled as such. A wrong-by-10% parameter count is far less
    harmful than no entry, and the architecture fields that actually drive the
    KV maths were read from the header exactly. But an estimate below
    :data:`_MIN_PLAUSIBLE_PARAMS_B` means the file is not a real model, so this
    returns None and the caller asks for --params rather than inventing a
    number the planner would then size memory against.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    estimate = round(size * 8 / 4.85 / 1e9, 2)
    return estimate if estimate >= _MIN_PLAUSIBLE_PARAMS_B else None


def _append_entry(entry: Dict[str, Any], console: Console) -> Path:
    path = user_catalog_path()
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise CatalogError(f"{path} is not valid YAML: {exc}") from None

    models = existing.get("models") or []
    models = [m for m in models if m.get("id") != entry["id"]]
    models.append(entry)
    existing["models"] = models
    existing.setdefault("schema_version", 1)
    existing["updated"] = _today()

    header = (
        "# llmfit user catalog overlay.\n"
        "# Entries here are layered on top of the bundled catalog; repeating an\n"
        "# id from the bundled catalog patches just the fields you list.\n\n"
    )
    write_text(path, header + yaml.safe_dump(existing, sort_keys=False))
    console.ok(f"added {entry['id']} to {path}")
    return path


# --------------------------------------------------------------------------- #
# refresh
# --------------------------------------------------------------------------- #


def refresh(console: Console, catalog: Catalog, check_remote: bool = True,
            write_overlay: bool = True) -> Dict[str, Any]:
    """Check catalog entries against reality and report drift.

    Two sources of truth are consulted, cheapest first:

    1. **Local GGUF files** already in the shared store, free, offline, and
       authoritative for architecture.
    2. **Upstream repos**, confirms an entry still resolves, which is what
       catches a renamed or withdrawn model before a user hits it mid-setup.
    """
    from llmfit.download import list_repo_files
    from llmfit.runtimes.gguf_store import find_local

    findings: List[Dict[str, Any]] = []
    corrections: Dict[str, Dict[str, Any]] = {}
    checked = 0

    for spec in catalog.models:
        local = _first_local(spec, find_local)
        if local is not None:
            checked += 1
            fixes = _diff_against_file(spec, local)
            if fixes:
                corrections[spec.id] = {**fixes, "arch_source": "measured"}
                findings.append({
                    "model": spec.id, "kind": "architecture",
                    "detail": ", ".join(f"{k}: {getattr(spec, k)} -> {v}"
                                        for k, v in fixes.items()),
                    "source": str(local),
                })

    if check_remote:
        for spec in catalog.models:
            if not spec.gguf_repo:
                continue
            checked += 1
            try:
                files = list_repo_files(spec.gguf_repo, timeout=15)
            except Exception as exc:  # a refresh must not abort on one bad entry
                findings.append({"model": spec.id, "kind": "unreachable",
                                 "detail": str(exc)[:200], "source": spec.gguf_repo})
                continue
            missing = _missing_quants(spec, files)
            if missing:
                findings.append({
                    "model": spec.id, "kind": "missing_quant",
                    "detail": f"catalog lists {', '.join(missing)} but the repo has none",
                    "source": spec.gguf_repo,
                })

    report = {
        "checked": checked,
        "catalog_updated": catalog.updated,
        "findings": findings,
        "corrections": corrections,
    }

    for finding in findings:
        console.warn(f"{finding['model']}: {finding['detail']}")
    if not findings:
        console.ok(f"checked {checked} source(s); the catalog matches reality")

    if corrections and write_overlay:
        path = _write_corrections(corrections, console)
        report["overlay"] = str(path)
    return report


def _first_local(spec: ModelSpec, find_local) -> Optional[Path]:
    for quant in spec.recommended_quants:
        path = find_local(spec, quant)
        if path is not None:
            return path
    return None


def _diff_against_file(spec: ModelSpec, path: Path) -> Dict[str, Any]:
    info = read_gguf_info(path)
    if info is None:
        return {}
    fixes: Dict[str, Any] = {}
    for field, value in (("layers", info.layers), ("kv_heads", info.kv_heads),
                         ("hidden_size", info.embedding_length),
                         ("head_dim", info.head_dim)):
        if value and getattr(spec, field) != value:
            fixes[field] = value
    return fixes


def _missing_quants(spec: ModelSpec, files) -> List[str]:
    from llmfit.download import pick_gguf

    return [q for q in spec.recommended_quants if pick_gguf(files, q) is None]


def _write_corrections(corrections: Dict[str, Dict[str, Any]],
                       console: Console) -> Path:
    """Write a proposed overlay rather than editing the catalog in place.

    Silently rewriting the catalog would make a previously reproducible setup
    change underneath the user. They apply this when they choose to.
    """
    path = user_catalog_dir() / "refresh-proposed.yaml"
    document = {
        "schema_version": 1,
        "updated": _today(),
        "models": [{"id": model_id, **fields} for model_id, fields in corrections.items()],
    }
    header = (
        "# Proposed catalog corrections from `llmfit catalog refresh`.\n"
        "# These were read from GGUF files on this machine and are NOT applied\n"
        "# automatically. To adopt them, rename this file to something ending in\n"
        f"# .yaml inside {user_catalog_dir()} (it already is), and they will be\n"
        "# layered over the bundled catalog on the next run. Delete it to ignore.\n\n"
    )
    write_text(path, header + yaml.safe_dump(document, sort_keys=False))
    console.info(f"proposed corrections written to {path}")
    return path


def _today() -> str:
    return datetime.date.today().isoformat()
