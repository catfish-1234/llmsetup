"""Your settings over the tuner's: ``--set key=value`` and ``--engine-arg``.

The tuner chooses every launch setting from the hardware and the use case, and for most people
that is the point. When you know better, or just want something specific, ``--set`` overrides any
field of the runtime configuration (context, GPU layers, KV cache type, flash attention, batch
sizes, sampling defaults, host and port, ...), and ``--engine-arg`` passes a flag straight to the
engine for anything llmsetup has no field for. Overrides are applied after tuning and before
verification, so the setup that is verified, saved and restarted is the one you asked for. With
repair on, a setting that fails can still be repaired, and the report says so; ``--no-repair``
keeps your values exactly and shows the failure instead.
"""

from __future__ import annotations

import difflib
import ipaddress
import typing
from dataclasses import fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmsetup.types import RuntimeConfig
from llmsetup.util.errors import LlmsetupError

#: Fields that identify the model or are filled in by provisioning, not settings you tune.
FIXED = {"runtime", "model_id", "quant", "model_path", "mmproj_path", "extra", "total_layers"}

#: Engines that take command-line flags llmsetup can pass through.
FLAG_ENGINES = ("llamacpp", "vllm", "mlx")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def settable() -> Dict[str, Any]:
    """Field name to type for everything ``--set`` accepts."""
    hints = typing.get_type_hints(RuntimeConfig)
    return {f.name: hints[f.name] for f in fields(RuntimeConfig) if f.name not in FIXED}


def _coerce(key: str, raw: str, kind: Any) -> Any:
    optional = typing.get_origin(kind) is typing.Union and type(None) in typing.get_args(kind)
    if optional:
        if raw.strip().lower() in ("", "none", "auto", "default"):
            return None
        kind = next(arg for arg in typing.get_args(kind) if arg is not type(None))
    text = raw.strip()
    try:
        if kind is bool:
            if text.lower() in _TRUE:
                return True
            if text.lower() in _FALSE:
                return False
            raise ValueError
        if kind is int:
            return int(text.replace("_", ""))
        if kind is float:
            return float(text)
    except ValueError:
        expected = {bool: "true or false", int: "a whole number", float: "a number"}[kind]
        raise LlmsetupError(f"--set {key}={raw}: {key} takes {expected}") from None
    return text


def parse_overrides(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``["context_tokens=32768", "flash_attention=true"]`` to typed values, checked by name."""
    allowed = settable()
    result: Dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or not key:
            raise LlmsetupError(f"--set {pair}: expected KEY=VALUE", hint="for example: --set flash_attention=true")
        if key not in allowed:
            close = difflib.get_close_matches(key, allowed, n=3)
            hint = (f"did you mean {', '.join(close)}? " if close else "") + "settable: " + ", ".join(sorted(allowed))
            if key in FIXED:
                hint = f"{key} is chosen with its own flag or by provisioning, not --set"
            raise LlmsetupError(f"--set {pair}: unknown setting {key}", hint=hint)
        result[key] = _coerce(key, raw, allowed[key])
    return result


def apply_overrides(config: RuntimeConfig, overrides: Dict[str, Any],
                    engine_args: Optional[Sequence[str]] = None) -> Tuple[RuntimeConfig, List[str]]:
    """The config with your settings applied, and a note for each change and each risk."""
    notes: List[str] = []
    changed = {k: v for k, v in overrides.items() if getattr(config, k) != v}
    for key, value in changed.items():
        notes.append(f"{key}: {getattr(config, key)!r} -> {value!r} (your setting, over the tuner's)")
    updated = replace(config, **changed) if changed else config
    if engine_args:
        if updated.runtime not in FLAG_ENGINES:
            raise LlmsetupError(
                f"--engine-arg passes flags to an engine's command line, and {updated.runtime} does not take any",
                hint="use --runtime " + " or --runtime ".join(FLAG_ENGINES) + ", or a --set setting instead")
        updated.extra = {**updated.extra, "engine_args": [str(a) for a in engine_args]}
        notes.append("engine flags added as given: " + " ".join(str(a) for a in engine_args))
    if not _loopback(updated.host):
        keyed = any("api-key" in str(a) for a in updated.extra.get("engine_args", []))
        notes.append(f"host {updated.host} serves the model to your network"
                     + ("" if keyed else "; anyone who can reach this machine can use it. On llama.cpp, "
                        "--engine-arg=--api-key=<key> requires a key"))
    return updated, notes


def _loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False
