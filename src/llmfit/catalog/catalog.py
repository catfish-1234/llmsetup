"""Loading and querying the model catalog.

The catalog is data with a schema, not a Python literal, so a user can extend it
without forking the tool. Layering is: bundled catalog → files in
``$LLMFIT_HOME/catalog.d/*.yaml`` → an explicit ``--catalog`` path. Later layers
override earlier entries by ``id`` and may add new ones.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from llmfit.types import ModelSpec, Task, UseCaseProfile
from llmfit.util.errors import CatalogError
from llmfit.util.fs import app_dir

BUNDLED = Path(__file__).parent / "data" / "models.yaml"

# Used when a catalog entry names a quant we have no bits-per-weight figure for.
DEFAULT_BPW = 4.85

_SPEC_FIELDS = set(ModelSpec.__dataclass_fields__)


class Catalog:
    """A loaded catalog: model specs plus the quantization size table."""

    def __init__(self, models: List[ModelSpec], quant_bpw: Dict[str, float],
                 updated: str = "", sources: Optional[List[str]] = None):
        self.models = models
        self.quant_bpw = quant_bpw
        self.updated = updated
        self.sources = sources or []
        self._by_id = {m.id: m for m in models}

    def __len__(self) -> int:
        return len(self.models)

    def get(self, model_id: str) -> Optional[ModelSpec]:
        return self._by_id.get(model_id)

    def require(self, model_id: str) -> ModelSpec:
        spec = self.get(model_id)
        if spec is None:
            close = [m for m in self._by_id if model_id.lower() in m.lower()]
            raise CatalogError(
                f"unknown model id {model_id!r}",
                hint=(f"did you mean: {', '.join(close[:3])}?" if close
                      else "run `llmfit catalog list` to see available models"),
            )
        return spec

    def bpw(self, quant: str) -> float:
        """Bits per weight for a quantization, case-insensitively."""
        if quant in self.quant_bpw:
            return self.quant_bpw[quant]
        upper = quant.upper()
        for key, value in self.quant_bpw.items():
            if key.upper() == upper:
                return value
        return DEFAULT_BPW

    # ----------------------------------------------------------------- queries

    def chat_models(self) -> List[ModelSpec]:
        return [m for m in self.models if not m.is_embedding]

    def embedding_models(self) -> List[ModelSpec]:
        return [m for m in self.models if m.is_embedding]

    def matching(self, profile: UseCaseProfile) -> List[ModelSpec]:
        """Hard capability filter.

        Everything here is a *requirement the user stated*, so failing it removes
        the model outright rather than costing it points. Soft preferences
        (quality, speed, size) are handled by scoring instead.
        """
        out = []
        for spec in self.chat_models():
            if not self.satisfies(spec, profile):
                continue
            out.append(spec)
        return out

    def satisfies(self, spec: ModelSpec, profile: UseCaseProfile) -> bool:
        if profile.needs_vision and not spec.supports_vision:
            return False
        if profile.needs_tools and not spec.supports_tools:
            return False
        if profile.needs_json and not spec.supports_json:
            return False
        if profile.needs_reasoning and not spec.is_reasoning:
            return False
        if profile.commercial_use and not spec.commercial_ok:
            return False
        # A model whose maximum context is below what the user asked for cannot
        # do the job, regardless of how good it is.
        if profile.context_tokens > spec.max_context:
            return False
        if spec.tasks and profile.tasks:
            wanted = {t.value for t in profile.tasks}
            if not wanted & set(spec.tasks):
                return False
        return True

    def unmatched_reasons(self, profile: UseCaseProfile) -> Dict[str, int]:
        """Why models were filtered out — turns 'nothing fits' into an explanation."""
        reasons: Dict[str, int] = {}

        def bump(key: str) -> None:
            reasons[key] = reasons.get(key, 0) + 1

        for spec in self.chat_models():
            if self.satisfies(spec, profile):
                continue
            if profile.needs_vision and not spec.supports_vision:
                bump("no vision support")
            elif profile.needs_tools and not spec.supports_tools:
                bump("no tool-calling support")
            elif profile.needs_json and not spec.supports_json:
                bump("no structured-output support")
            elif profile.needs_reasoning and not spec.is_reasoning:
                bump("not a reasoning model")
            elif profile.commercial_use and not spec.commercial_ok:
                bump("licence disallows commercial use")
            elif profile.context_tokens > spec.max_context:
                bump(f"max context below {profile.context_tokens}")
            else:
                bump("task mismatch")
        return reasons

    def for_task(self, task: Task) -> List[ModelSpec]:
        return [m for m in self.models if task.value in m.tasks]


def load_catalog(extra_paths: Optional[Iterable[Path]] = None,
                 include_user_dir: bool = True) -> Catalog:
    """Load the bundled catalog, then layer user overrides on top."""
    paths: List[Path] = [BUNDLED]

    if include_user_dir:
        user_dir = app_dir() / "catalog.d"
        if user_dir.is_dir():
            paths.extend(sorted(p for p in user_dir.iterdir()
                                if p.suffix in (".yaml", ".yml")))
    env_path = os.environ.get("LLMFIT_CATALOG")
    if env_path:
        paths.append(Path(env_path).expanduser())
    if extra_paths:
        paths.extend(Path(p).expanduser() for p in extra_paths)

    merged: Dict[str, ModelSpec] = {}
    quant_bpw: Dict[str, float] = {}
    updated = ""
    used: List[str] = []

    for path in paths:
        if not path.exists():
            if path is BUNDLED:
                raise CatalogError(f"bundled catalog missing at {path}",
                                   hint="reinstall llmfit")
            continue
        data = _read_yaml(path)
        quant_bpw.update({str(k): float(v) for k, v in (data.get("quant_bpw") or {}).items()})
        updated = data.get("updated") or updated
        for raw in data.get("models") or []:
            if not isinstance(raw, dict) or "id" not in raw:
                raise CatalogError(f"catalog entry without an 'id' in {path}")
            model_id = raw["id"]
            # A later layer that repeats an id patches the earlier entry field by
            # field, so a user can override just `ollama_name` without restating
            # the whole model. Patching must not require a complete entry, so the
            # existence check comes before we try to build a full ModelSpec.
            if model_id in merged:
                merged[model_id] = _patch(merged[model_id], raw)
                _record_extras(model_id, raw)
            else:
                merged[model_id] = _to_spec(raw, path)
        used.append(str(path))

    if not merged:
        raise CatalogError("catalog contains no models", hint="check your catalog YAML files")
    return Catalog(list(merged.values()), quant_bpw, updated, used)


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not read catalog {path}: {exc}",
                           hint="the file must be valid YAML") from None
    if not isinstance(data, dict):
        raise CatalogError(f"catalog {path} must be a YAML mapping")
    return data


def _to_spec(raw: dict, path: Path) -> ModelSpec:
    """Build a full spec from a complete entry.

    Keys outside the ModelSpec schema (``arch_source`` and other provenance
    notes) are preserved out-of-band by :func:`_record_extras` rather than
    dropped, so the planner can flag estimated architectures.
    """
    if not isinstance(raw, dict) or "id" not in raw:
        raise CatalogError(f"catalog entry without an 'id' in {path}")
    fields = {k: v for k, v in raw.items() if k in _SPEC_FIELDS}
    fields.setdefault("name", raw["id"])
    fields.setdefault("publisher", "unknown")
    try:
        spec = ModelSpec(**fields)
    except TypeError as exc:
        raise CatalogError(f"bad catalog entry {raw.get('id')!r} in {path}: {exc}") from None
    _record_extras(spec.id, raw)
    return spec


def _record_extras(model_id: str, raw: dict) -> None:
    extras = {k: v for k, v in raw.items() if k not in _SPEC_FIELDS}
    if extras:
        _EXTRAS.setdefault(model_id, {}).update(extras)


def _patch(base: ModelSpec, raw: dict) -> ModelSpec:
    changes = {k: v for k, v in raw.items() if k in _SPEC_FIELDS and k != "id"}
    return replace(base, **changes)


# Side-table for non-schema keys (arch_source, notes on provenance).
_EXTRAS: Dict[str, dict] = {}


def arch_source(model_id: str) -> str:
    """'measured' or 'estimated' — whether the architecture figures are trustworthy."""
    return str(_EXTRAS.get(model_id, {}).get("arch_source", "measured"))
