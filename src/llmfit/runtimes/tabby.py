"""TabbyAPI / ExLlamaV2 adapter.

ExLlamaV2 is the fastest path for INT4 on a single modern NVIDIA card — its
EXL2 format is measurement-based and mixed-bit-width, and on one 3090/4090/5090
nothing else matches it for batch-size-1 generation. TabbyAPI is the
OpenAI-compatible server around it.

It is deliberately ranked below llama.cpp and Ollama in the registry despite
being faster: it is CUDA-only, needs EXL2 quantizations rather than GGUF, and
has far narrower model coverage. It is the right answer for a specific machine
and a specific model, not the general default.
"""

from __future__ import annotations

from typing import Dict, List

from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.runtimes.registry import register
from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util import http
from llmfit.util.fs import app_dir, ensure_dir
from llmfit.util.log import Console
from llmfit.util.proc import run, spawn, which

DEFAULT_PORT = 5000


class TabbyRuntime(Runtime):
    name = "tabby"
    label = "TabbyAPI (ExLlamaV2)"
    uses_gguf = False

    def is_available(self) -> bool:
        if which("tabbyapi"):
            return True
        if http.request(f"http://127.0.0.1:{DEFAULT_PORT}/v1/models", timeout=2).ok:
            return True
        return run(["python3", "-c", "import exllamav2"], timeout=30).ok

    def install_hint(self) -> str:
        return ("clone https://github.com/theroyallab/tabbyAPI and run its start "
                "script (NVIDIA only; needs EXL2 quantizations, not GGUF)")

    def supports(self, spec: ModelSpec) -> bool:
        # EXL2 conversions are a smaller world than GGUF. The catalog does not
        # track them yet, so this reports what could plausibly load and the
        # adapter warns at provision time if no EXL2 build exists.
        return bool(spec.hf_repo)

    def ensure_model(self, spec: ModelSpec, config: RuntimeConfig,
                     console: Console) -> RuntimeConfig:
        if not spec.hf_repo:
            raise RuntimeUnavailable(
                f"{spec.name} has no repo TabbyAPI could load",
                "use llama.cpp or Ollama for this model",
            )
        config.model_path = spec.hf_repo
        console.warn(
            "TabbyAPI needs an EXL2 quantization; if one does not exist for this "
            "model, llama.cpp with GGUF is the working path"
        )
        return config

    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        cmd = [
            "python3", "-m", "tabbyAPI.main",
            "--model-name", config.model_path or spec.hf_repo or spec.id,
            "--host", config.host,
            "--port", str(config.port),
            "--max-seq-len", str(config.context_tokens),
        ]
        # ExLlamaV2's Q4 cache is its equivalent of a quantized KV cache and is
        # the main way to buy context on a fixed card.
        if config.kv_cache_type != "f16":
            cmd += ["--cache-mode", "Q4" if config.kv_cache_type.startswith("q4") else "Q8"]
        advanced = config.advanced
        if advanced is not None and advanced.offload.tensor_split:
            cmd += ["--gpu-split"] + [str(v) for v in advanced.offload.tensor_split]
        if advanced is not None and advanced.speculative.enabled:
            cmd += ["--draft-model-name", advanced.speculative.draft_model_id or ""]
        return cmd

    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        if not self.is_available():
            raise RuntimeUnavailable("TabbyAPI is not installed", self.install_hint())
        if not http.port_is_free(config.host, config.port):
            config.port = http.find_free_port(config.host, config.port + 1)

        log_path = str(ensure_dir(app_dir() / "logs") / f"tabby-{config.port}.log")
        process = spawn(self.launch_command(spec, config), log_path=log_path)
        endpoint = Endpoint(
            base_url=f"http://{config.host}:{config.port}", model=spec.id,
            runtime=self.name, started_by_us=True, log_path=log_path, process=process,
        )

        def ready() -> bool:
            if process.poll() is not None:
                raise RuntimeUnavailable(f"TabbyAPI exited; see {log_path}")
            return http.request(endpoint.models_url, timeout=3).ok

        if not http.wait_for(ready, timeout=900, interval=2.0):
            self.stop(endpoint)
            raise RuntimeUnavailable("TabbyAPI did not become healthy", f"see {log_path}")
        return endpoint

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        return {}


register(TabbyRuntime())
