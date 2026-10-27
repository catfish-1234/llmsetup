"""LM Studio adapter.

LM Studio matters twice over: some people want it as their runtime, and almost
everyone wants to *see* their model there afterwards. This adapter covers the
first case; :mod:`llmsetup.interop.lmstudio` covers the second, and both agree on
the ``publisher/model/file.gguf`` layout LM Studio parses to populate its UI.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from llmsetup.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmsetup.runtimes.gguf_store import fetch_gguf
from llmsetup.runtimes.registry import register
from llmsetup.types import ModelSpec, RuntimeConfig
from llmsetup.util import http
from llmsetup.util.log import Console
from llmsetup.util.proc import first_line, format_command, run, which

DEFAULT_PORT = 1234


def lmstudio_models_dir() -> Path:
    """LM Studio's model root, honouring an explicit override."""
    override = os.environ.get("LMSTUDIO_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".lmstudio" / "models"


class LmStudioRuntime(Runtime):
    name = "lmstudio"
    label = "LM Studio"
    uses_gguf = True

    def is_available(self) -> bool:
        if which("lms"):
            return True
        return http.request(f"http://127.0.0.1:{DEFAULT_PORT}/v1/models", timeout=2).ok

    def install_hint(self) -> str:
        return (
            "install LM Studio from https://lmstudio.ai, then enable its CLI "
            "with `npx lmstudio install-cli` (or Settings → Developer → Install `lms`)"
        )

    def version(self) -> Optional[str]:
        result = run(["lms", "version"], timeout=10)
        return first_line(result.output) if result.ok else None

    def ensure_model(self, spec: ModelSpec, config: RuntimeConfig,
                     console: Console) -> RuntimeConfig:
        path = fetch_gguf(spec, config.quant, console)
        config.model_path = str(path)

        # Put it where LM Studio looks. `lms import` is preferred because it also
        # updates LM Studio's index; the manual layout is the fallback.
        if which("lms"):
            command = ["lms", "import", str(path), "--yes"]
            result = run(command, timeout=600)
            if result.ok:
                console.ok("imported into LM Studio via `lms import`")
                config.served_name = spec.id
                return config
            console.warn(f"`lms import` failed: {result.failure(command)}")

        from llmsetup.interop.lmstudio import register_in_lmstudio

        record = register_in_lmstudio(spec, Path(path), config.quant)
        console.ok(f"placed in LM Studio's model folder ({record['method']})")
        config.served_name = spec.id
        return config

    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        port = int(config.extra.get("lmstudio_port", DEFAULT_PORT))
        base = f"http://127.0.0.1:{port}"

        if not http.request(f"{base}/v1/models", timeout=2).ok:
            if not which("lms"):
                raise RuntimeUnavailable(
                    "LM Studio's local server is not running",
                    "open LM Studio → Developer → Start Server, or install the `lms` CLI",
                )
            console.info("starting LM Studio's server …")
            command = ["lms", "server", "start", "--port", str(port)]
            result = run(command, timeout=90)
            if not result.ok:
                raise RuntimeUnavailable(result.failure(command))
            if not http.wait_for(
                lambda: http.request(f"{base}/v1/models", timeout=2).ok, timeout=60
            ):
                raise RuntimeUnavailable("LM Studio's server did not come up within 60s",
                                         f"Rerun: {format_command(command)}")

        # Load the model with our tuned context length. LM Studio keeps models
        # resident, so an already-loaded copy is reused.
        if which("lms"):
            load = ["lms", "load", config.served_name or spec.id,
                    "--context-length", str(config.context_tokens), "--yes"]
            if config.gpu_layers:
                load += ["--gpu", "max" if config.full_gpu_offload else str(config.gpu_layers)]
            result = run(load, timeout=900)
            if not result.ok:
                raise RuntimeUnavailable(f"`lms load` failed: {result.failure(load)}")

        served = self._resolve_served_name(base, spec, config)
        return Endpoint(
            base_url=base,
            model=served,
            runtime=self.name,
            supports_embeddings=spec.is_embedding,
            started_by_us=False,
        )

    def _resolve_served_name(self, base: str, spec: ModelSpec,
                             config: RuntimeConfig) -> str:
        """LM Studio names models by its own identifier; ask the server."""
        response = http.request(f"{base}/v1/models", timeout=10)
        data = response.json() or {}
        ids = [m.get("id", "") for m in data.get("data", [])]
        wanted = (config.served_name or spec.id).lower()
        for identifier in ids:
            if wanted in identifier.lower():
                return identifier
        return ids[0] if ids else (config.served_name or spec.id)

    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        return ["lms", "load", config.served_name or spec.id,
                "--context-length", str(config.context_tokens)]

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        return {}


register(LmStudioRuntime())
