"""llama.cpp (`llama-server`) adapter.

The reference target: it exposes every knob the tuner computes, so it is the
runtime where the tuning is most fully expressed. Other adapters approximate
these flags with whatever their engine offers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.runtimes.gguf_store import fetch_gguf
from llmfit.runtimes.registry import register
from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util import http
from llmfit.util.fs import app_dir, ensure_dir
from llmfit.util.log import Console
from llmfit.util.proc import first_line, run, spawn, which


class LlamaCppRuntime(Runtime):
    name = "llamacpp"
    label = "llama.cpp"
    uses_gguf = True

    binary = "llama-server"

    def is_available(self) -> bool:
        return which(self.binary) is not None

    def install_hint(self) -> str:
        return (
            "macOS/Linux: `brew install llama.cpp`  |  "
            "Linux: build from https://github.com/ggml-org/llama.cpp  |  "
            "Windows: download a release binary from that repo"
        )

    def version(self) -> Optional[str]:
        result = run([self.binary, "--version"], timeout=15)
        return first_line(result.output) if result.output else None

    def ensure_model(self, spec: ModelSpec, config: RuntimeConfig,
                     console: Console) -> RuntimeConfig:
        if config.model_path and Path(config.model_path).exists():
            return config
        path = fetch_gguf(spec, config.quant, console)
        config.model_path = str(path)
        return config

    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        cmd = [
            self.binary,
            "--model", config.model_path or f"<{spec.id}.gguf>",
            "--host", config.host,
            "--port", str(config.port),
            "--ctx-size", str(config.context_tokens * config.parallel),
            "--n-gpu-layers", str(config.gpu_layers),
            "--batch-size", str(config.batch_size),
            "--ubatch-size", str(config.ubatch_size),
            "--parallel", str(config.parallel),
            # Serve the model's own chat template, which is what makes tool
            # calling and multi-turn formatting behave like the model expects.
            "--jinja",
            "--alias", config.served_name or spec.id,
        ]
        if config.threads:
            cmd += ["--threads", str(config.threads)]
        # Flash attention is an explicit on/off in current llama-server; leaving
        # it unset lets it auto-select, which we do not want when the repair loop
        # has deliberately turned it off.
        cmd += ["--flash-attn", "on" if config.flash_attention else "off"]
        if config.kv_cache_type != "f16":
            cmd += ["--cache-type-k", config.kv_cache_type,
                    "--cache-type-v", config.kv_cache_type]
        if config.mlock:
            cmd.append("--mlock")
        if not config.mmap:
            cmd.append("--no-mmap")
        if spec.is_embedding:
            cmd.append("--embeddings")
        for flag in config.extra.get("llamacpp_flags", []):
            cmd.append(str(flag))
        return cmd

    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        if not self.is_available():
            raise RuntimeUnavailable("llama-server not found", self.install_hint())

        if not http.port_is_free(config.host, config.port):
            config.port = http.find_free_port(config.host, config.port + 1)
            console.info(f"port in use; serving on {config.port} instead")

        log_path = str(ensure_dir(app_dir() / "logs") / f"llamacpp-{config.port}.log")
        cmd = self.launch_command(spec, config)
        console.debug(" ".join(cmd))
        process = spawn(cmd, log_path=log_path)

        endpoint = Endpoint(
            base_url=f"http://{config.host}:{config.port}",
            model=config.served_name or spec.id,
            runtime=self.name,
            supports_embeddings=spec.is_embedding,
            started_by_us=True,
            log_path=log_path,
            process=process,
        )

        def ready() -> bool:
            if process.poll() is not None:
                raise RuntimeUnavailable(
                    f"llama-server exited with code {process.returncode}; see {log_path}"
                )
            return http.request(f"{endpoint.base_url}/health", timeout=3).ok

        # Large models take a while to load from disk; the wait scales with size.
        timeout = 90 + (Path(config.model_path).stat().st_size / (1024 ** 3) * 12
                        if config.model_path and Path(config.model_path).exists() else 0)
        if not http.wait_for(ready, timeout=timeout, interval=1.5):
            self.stop(endpoint)
            raise RuntimeUnavailable(
                f"llama-server did not become healthy within {timeout:.0f}s",
                f"check the log at {log_path}",
            )
        return endpoint

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        """A runnable start script, so the user is never dependent on llmfit."""
        cmd = self.launch_command(spec, config)
        script = "#!/usr/bin/env bash\n# Generated by llmfit. Edit freely.\nset -euo pipefail\n\nexec "
        script += " \\\n  ".join(_quote(part) for part in cmd) + "\n"
        return {f"start-{spec.id}.sh": script}


def _quote(value: str) -> str:
    return value if all(c.isalnum() or c in "-_./:=" for c in value) else f'"{value}"'


register(LlamaCppRuntime())
