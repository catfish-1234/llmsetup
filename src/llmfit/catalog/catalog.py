"""Loading and querying the model catalog.

The catalog is data with a schema, not a Python literal, so a user can extend it
without forking the tool. Layering is: bundled catalog → files in
``$LLMFIT_HOME/catalog.d/*.yaml`` → an explicit ``--catalog`` path. Later layers
override earlier entries by ``id`` and may add new ones.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
                 updated: str = "", sources: Optional[List[str]] = None,
                 extras: Optional[Dict[str, dict]] = None):
        self.models = models
        self.quant_bpw = quant_bpw
        self.updated = updated
        self.sources = sources or []
        # Non-schema keys (arch_source and other provenance) live on the instance,
        # not in a module global: two Catalog objects in one process must not
        # report each other's provenance.
        self.extras = extras or {}
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {m.id: m for m in self.models}

    def restrict_to(self, model_id: str) -> "Catalog":
        """A view containing only one model, for `--model` and `doctor`.

        Returns a new Catalog rather than mutating this one, so callers never
        have to reach into private indexes to keep them consistent.
        """
        spec = self.require(model_id)
        return Catalog([spec], dict(self.quant_bpw), self.updated, list(self.sources),
                       {spec.id: dict(self.extras.get(spec.id, {}))})

    def arch_source(self, model_id: str) -> str:
        """'measured' or 'estimated', whether the architecture figures are trustworthy."""
        return str(self.extras.get(model_id, {}).get("arch_source", "measured"))

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

    def completion_models(self) -> List[ModelSpec]:
        """Models that can fill a gap between a prefix and a suffix."""
        return [m for m in self.models if m.supports_fim and not m.is_embedding]

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
        """Why models were filtered out, which turns 'nothing fits' into an explanation."""
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
    extras: Dict[str, dict] = {}
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
        version = data.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise CatalogError(f"unsupported schema_version {version!r} in {path}; expected 1")
        table = data.get("quant_bpw", {})
        if not isinstance(table, dict):
            raise CatalogError(f"quant_bpw must be a mapping in {path}")
        for key, value in table.items():
            converted = _coerce_types({"params_b": value}, f"quant_bpw[{key}]", path)
            quant_bpw[str(key)] = converted["params_b"]
        updated = data.get("updated") or updated
        entries = data.get("models", [])
        if not isinstance(entries, list):
            raise CatalogError(f"models must be a list in {path}")
        seen = set()
        for raw in entries:
            if not isinstance(raw, dict) or "id" not in raw:
                raise CatalogError(f"catalog entry without an 'id' in {path}")
            model_id = raw["id"]
            if not isinstance(model_id, str) or not model_id.strip():
                raise CatalogError(f"model id must be a nonempty string in {path}: {model_id!r}")
            if model_id in seen:
                raise CatalogError(f"duplicate model id {model_id!r} in {path}")
            seen.add(model_id)
            # A later layer that repeats an id patches the earlier entry field by
            # field, so a user can override just `ollama_name` without restating
            # the whole model. Patching must not require a complete entry, so the
            # existence check comes before we try to build a full ModelSpec.
            if model_id in merged:
                merged[model_id] = _patch(merged[model_id], raw, path)
            else:
                merged[model_id] = _to_spec(raw, path)
            _record_extras(extras, model_id, raw)
        used.append(str(path))

    if not merged:
        raise CatalogError("catalog contains no models", hint="check your catalog YAML files")
    return Catalog(list(merged.values()), quant_bpw, updated, used, extras)


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogError(f"could not read catalog {path}: {exc}",
                           hint="the file must be valid YAML") from None
    if data is None:
        data = {}
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
    fields = _coerce_types(fields, raw["id"], path)
    try:
        spec = ModelSpec(**fields)
    except TypeError as exc:
        raise CatalogError(f"bad catalog entry {raw.get('id')!r} in {path}: {exc}") from None
    _validate_spec(spec, path)
    return spec


def _record_extras(store: Dict[str, dict], model_id: str, raw: dict) -> None:
    found = {k: v for k, v in raw.items() if k not in _SPEC_FIELDS}
    if found:
        store.setdefault(model_id, {}).update(found)


def _patch(base: ModelSpec, raw: dict, path: Path) -> ModelSpec:
    changes = {k: v for k, v in raw.items() if k in _SPEC_FIELDS and k != "id"}
    changes = _coerce_types(changes, base.id, path)
    spec = replace(base, **changes)
    _validate_spec(spec, path)
    return spec


#: Fields the memory model does arithmetic on. A string here does not fail at
#: load time. It fails several layers down inside the fit maths, with an error
#: that says nothing about the catalog. Types are enforced at the boundary.
_NUMERIC_FIELDS = {
    "params_b": float, "active_params_b": float, "layers": int,
    "hidden_size": int, "kv_heads": int, "head_dim": int, "max_context": int,
    "quality_tier": int, "max_disk_gb": float,
    "sliding_window": int, "swa_pattern": int,
    "full_attention_interval": int, "recurrent_state_bytes": int,
}
_BOOL_FIELDS = (
    "commercial_ok", "supports_tools", "supports_vision", "supports_json",
    "is_reasoning", "is_embedding", "draft_capable", "mtp_heads",
    "supports_fim",
)
_LIST_FIELDS = ("tasks", "languages", "recommended_quants")


def _coerce_types(fields: Dict[str, Any], model_id: str, path: Path) -> Dict[str, Any]:
    """Validate and normalise an entry's field types.

    Catalogs are extended by users and fetched automatically, so an entry can
    carry anything YAML can express. Rejecting a bad value here, naming the file
    and the field, beats a TypeError deep in the planner.
    """
    out = dict(fields)

    for name, caster in _NUMERIC_FIELDS.items():
        if name not in out:
            continue
        value = out[name]
        if value is None and name in {"active_params_b", "head_dim", "max_disk_gb", "sliding_window"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise CatalogError(
                f"{model_id}.{name} must be a number, got {value!r} in {path}"
            )
        try:
            number = float(value)
            if not math.isfinite(number) or (caster is int and not number.is_integer()):
                raise ValueError
            out[name] = caster(value)
        except (TypeError, ValueError, OverflowError):
            raise CatalogError(
                f"{model_id}.{name} must be a number, got {value!r} in {path}"
            ) from None
        if out[name] < 0 or (out[name] == 0 and name != "recurrent_state_bytes"):
            raise CatalogError(
                f"{model_id}.{name} must be positive, got {out[name]} in {path}"
            )

    for name in _BOOL_FIELDS:
        if name in out and not isinstance(out[name], bool):
            raise CatalogError(f"{model_id}.{name} must be a boolean in {path}")

    for name in _LIST_FIELDS:
        if name in out:
            value = out[name]
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, (list, tuple)):
                raise CatalogError(
                    f"{model_id}.{name} must be a list, got {value!r} in {path}"
                )
            if not all(isinstance(item, str) and item.strip() for item in value):
                raise CatalogError(f"{model_id}.{name} must contain nonempty strings in {path}")
            out[name] = list(value)

    for name in ("id", "name", "publisher", "license", "notes"):
        if name in out and (not isinstance(out[name], str) or
                            (name != "notes" and not out[name].strip())):
            raise CatalogError(f"{model_id}.{name} must be a string in {path}")

    return out


def _validate_spec(spec: ModelSpec, path: Path) -> None:
    from llmfit.download import validate_repo, validate_revision
    from llmfit.util.errors import ProvisionError

    for field in ("gguf_repo", "hf_repo", "mlx_repo", "gguf_revision", "hf_revision", "mlx_revision"):
        value = getattr(spec, field)
        if value is not None:
            try:
                if not isinstance(value, str):
                    raise ProvisionError("must be a string")
                if field.endswith("_repo"):
                    setattr(spec, field, validate_repo(value))
                else:
                    setattr(spec, field, validate_revision(value))
            except ProvisionError as exc:
                raise CatalogError(f"{spec.id}.{field}: {exc.message} in {path}") from None
    if not 1 <= spec.quality_tier <= 5:
        raise CatalogError(f"{spec.id}.quality_tier must be between 1 and 5 in {path}")
    if spec.active_params_b is not None and spec.active_params_b > spec.params_b:
        raise CatalogError(f"{spec.id}.active_params_b exceeds params_b in {path}")
    unknown = set(spec.tasks) - {task.value for task in Task}
    if unknown:
        raise CatalogError(f"{spec.id}.tasks has unknown values {sorted(unknown)!r} in {path}")
    if not spec.recommended_quants:
        raise CatalogError(f"{spec.id}.recommended_quants must not be empty in {path}")

