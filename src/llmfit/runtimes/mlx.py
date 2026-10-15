"""MLX adapter (Apple Silicon).

MLX is the fastest path for dense models on Apple hardware because it targets
the unified memory architecture directly. Its model coverage is narrower than
llama.cpp's, so the registry ranks it below Ollama by default and picks it when
the catalog has an ``mlx_repo`` and the user asked for speed.
"""

from __future__ import annotations

import platform
import shlex
from typing import Dict, List, Optional

from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.runtimes.registry import register
from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util import http
from llmfit.util.fs import app_dir, ensure_dir
from llmfit.util.log import Console
from llmfit.util.proc import run, spawn

DEFAULT_PORT = 8080


class MlxRuntime(Runtime):
    name = "mlx"
    label = "MLX"
    uses_gguf = False

    def is_available(self) -> bool:
        if platform.system().lower() != "darwin" or platform.machine().lower() != "arm64":
            return False
        return run(["python3", "-c", "import mlx_lm"], timeout=30).ok

    def install_hint(self) -> str:
        return "`pip install mlx-lm` (Apple Silicon only)"

    def version(self) -> Optional[str]:
        result = run(
            ["python3", "-c", "import mlx_lm,sys; sys.stdout.write(mlx_lm.__version__)"],
            timeout=30,
        )
        return result.stdout.strip() if result.ok else None

    def supports(self, spec: ModelSpec) -> bool:
        return bool(spec.mlx_repo or spec.hf_repo)

    def ensure_model(self, spec: ModelSpec, config: RuntimeConfig,
                     console: Console) -> RuntimeConfig:
        repo = spec.mlx_repo or spec.hf_repo
        if not repo:
            raise RuntimeUnavailable(f"{spec.name} has no MLX or HF repo in the catalog")
        if not spec.mlx_repo:
            console.warn(
                f"no MLX-converted build of {spec.name} in the catalog; using "
                f"{spec.hf_repo}, which MLX will convert on load"
            )
        config.model_path = repo
        return config

    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        return [
            "python3", "-m", "mlx_lm.server",
            "--model", config.model_path or spec.mlx_repo or spec.id,
            "--host", config.host,
            "--port", str(config.port),
            "--max-tokens", str(config.extra.get("max_tokens", 2048)),
        ]

    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        if not self.is_available():
            raise RuntimeUnavailable("mlx-lm is not installed", self.install_hint())
        if not http.port_is_free(config.host, config.port):
            config.port = http.find_free_port(config.host, config.port + 1)

        log_path = str(ensure_dir(app_dir() / "logs") / f"mlx-{config.port}.log")
        process = spawn(self.launch_command(spec, config), log_path=log_path)
        endpoint = Endpoint(
            base_url=f"http://{config.host}:{config.port}",
            model=config.model_path or spec.id,
            runtime=self.name,
            started_by_us=True,
            log_path=log_path,
            process=process,
        )

        def ready() -> bool:
            if process.poll() is not None:
                raise RuntimeUnavailable(f"mlx_lm.server exited; see {log_path}")
            return http.request(endpoint.models_url, timeout=3).ok

        if not http.wait_for(ready, timeout=900, interval=2.0):
            self.stop(endpoint)
            raise RuntimeUnavailable("mlx_lm.server did not become healthy", f"see {log_path}")
        return endpoint

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        cmd = self.launch_command(spec, config)
        return {f"start-{spec.id}-mlx.sh":
                "#!/usr/bin/env bash\nset -euo pipefail\n\nexec " + shlex.join(cmd) + "\n"}


register(MlxRuntime())
