"""LM Studio registration.

LM Studio derives its UI from the directory layout, not from a database:

    ~/.lmstudio/models/<publisher>/<model>/<file>.gguf
                       ^^^^^^^^^^^ ^^^^^^^
                       group in the dropdown, then the model name

so placing a file in that shape is all it takes for a model downloaded
elsewhere to appear in the app. We link rather than copy, so LM Studio and every
other app share one copy of the weights.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util.fs import link_or_copy, safe_component, safe_subpath
from llmfit.util.proc import run, which


def models_dir() -> Path:
    override = os.environ.get("LMSTUDIO_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".lmstudio" / "models"


def is_installed() -> bool:
    return which("lms") is not None or models_dir().parent.is_dir()


def _sanitize(value: str) -> str:
    """LM Studio shows these as folder names, so keep them filesystem-clean."""
    return safe_component(value)


def target_path(spec: ModelSpec, quant: str, filename: str) -> Path:
    # Path.name on the filename too: it originates from a remote repo listing.
    return safe_subpath(
        models_dir(), spec.publisher, f"{spec.id}-{quant}", Path(filename).name
    )


def register_in_lmstudio(spec: ModelSpec, gguf_path: Path, quant: str,
                         shards: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Place the weights where LM Studio will find them.

    ``lms import`` is preferred when the CLI exists because it also refreshes
    LM Studio's index; the directory layout is the fallback and works with the
    app closed.
    """
    files = shards or _shards_beside(gguf_path)

    if which("lms"):
        result = run(["lms", "import", str(gguf_path), "--yes"], timeout=900)
        if result.ok:
            return {"target": "lm_studio", "ok": True, "method": "lms import",
                    "path": str(gguf_path),
                    "note": "visible in LM Studio's My Models list"}

    linked, method = [], "none"
    for shard in files:
        destination = target_path(spec, quant, shard.name)
        method, path = link_or_copy(shard, destination)
        linked.append(str(path))

    return {
        "target": "lm_studio",
        "ok": bool(linked),
        "method": method,
        "path": linked[0] if linked else None,
        "files": linked,
        "note": (f"placed under {models_dir()} as publisher/model/file.gguf; "
                 f"restart LM Studio if it is already running"),
    }


def _shards_beside(gguf_path: Path) -> List[Path]:
    """All shards of a split GGUF, so LM Studio gets a complete model."""
    match = re.match(r"(.+)-\d{5}-of-\d{5}\.gguf$", gguf_path.name)
    if not match:
        return [gguf_path]
    return sorted(gguf_path.parent.glob(f"{match.group(1)}-*-of-*.gguf"))


def describe(spec: ModelSpec, config: RuntimeConfig) -> str:
    return (f"Open LM Studio → My Models. {spec.name} ({config.quant}) appears "
            f"under the '{_sanitize(spec.publisher)}' publisher.")
