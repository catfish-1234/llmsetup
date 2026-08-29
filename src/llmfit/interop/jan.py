"""Jan registration.

Jan reads a per-model folder containing the weights and a ``model.json``
descriptor. Same principle as LM Studio: link the shared weights in, write the
descriptor, do not copy gigabytes.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Dict

from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util.fs import link_or_copy, safe_subpath, write_json


def data_dir() -> Path:
    override = os.environ.get("JAN_DATA_DIR")
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "Jan" / "data"
    if system == "windows":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Jan" / "data"
    return Path.home() / ".config" / "Jan" / "data"


def is_installed() -> bool:
    return data_dir().is_dir()


def register_in_jan(spec: ModelSpec, gguf_path: Path,
                    config: RuntimeConfig) -> Dict[str, Any]:
    if not is_installed():
        return {"target": "jan", "ok": False, "note": "Jan is not installed; skipped"}

    folder = safe_subpath(data_dir() / "models", spec.id)
    method, linked = link_or_copy(gguf_path, folder / gguf_path.name)

    descriptor = {
        "object": "model",
        "version": 1,
        "format": "gguf",
        "id": spec.id,
        "name": f"{spec.name} ({config.quant}, tuned by llmfit)",
        "sources": [{"filename": gguf_path.name, "url": str(linked)}],
        "settings": {
            "ctx_len": config.context_tokens,
            "ngl": config.gpu_layers,
            "prompt_template": spec.chat_template or "",
        },
        "parameters": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_tokens": config.extra.get("max_tokens", 2048),
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": config.extra.get("stop", []),
        },
        "metadata": {
            "author": spec.publisher,
            "tags": list(spec.tasks),
            "size": gguf_path.stat().st_size if gguf_path.exists() else 0,
        },
        "engine": "nitro",
    }
    write_json(folder / "model.json", descriptor)
    return {"target": "jan", "ok": True, "method": method, "path": str(folder),
            "note": "restart Jan to see it in the model list"}
