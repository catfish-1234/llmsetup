"""vLLM adapter.

Chosen when several people (or several agents) hit the endpoint at once: vLLM's
paged attention and continuous batching hold throughput as concurrency rises,
where single-stream engines collapse. It serves safetensors, not GGUF, so the
shared-GGUF story does not apply, which is called out in the report rather than
quietly ignored.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Optional

from llmsetup.download import pin_revision
from llmsetup.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmsetup.runtimes.registry import register
from llmsetup.types import ModelSpec, RuntimeConfig
from llmsetup.util import http
from llmsetup.util.fs import app_dir, ensure_dir
from llmsetup.util.log import Console
from llmsetup.util.proc import first_line, run, spawn, which

DEFAULT_PORT = 8000


def kv_cache_dtype(config: RuntimeConfig) -> Optional[str]:
    """The vLLM KV dtype implied by the configured cache type.

    Derived at launch rather than stored, because the repair loop changes
    `kv_cache_type` and a value snapshotted at tuning time would not follow it.
    That cuts both ways: a repair that quantizes the cache would have had no
    effect here, and one that restores precision would have left the quantized
    setting in place while reporting the cache as restored.

    vLLM names its own formats, so the llama.cpp-style names map rather than
    pass through. Anything at or above 8-bit maps to fp8; there is no vLLM
    equivalent of a 4-bit cache, so those do not map to one.
    """
    kind = (config.kv_cache_type or "f16").lower()
    if kind in ("f16", "bf16", "f32", "auto"):
        return None
    return "fp8"


class VllmRuntime(Runtime):
    name = "vllm"
    label = "vLLM"
    uses_gguf = False

    def is_available(self) -> bool:
        if which("vllm"):
            return True
        return run(["python3", "-c", "import vllm"], timeout=30).ok

    def install_hint(self) -> str:
        return "`pip install vllm` (Linux + NVIDIA; needs a matching CUDA toolkit)"

    def version(self) -> Optional[str]:
        result = run(["vllm", "--version"], timeout=30)
        return first_line(result.output) if result.ok else None

    def supports(self, spec: ModelSpec) -> bool:
        return bool(spec.hf_repo)

    def ensure_model(self, spec: ModelSpec, config: RuntimeConfig,
                     console: Console) -> RuntimeConfig:
        if not spec.hf_repo:
            raise RuntimeUnavailable(
                f"{spec.name} has no safetensors repo, which vLLM requires",
                "use llama.cpp or Ollama for this model",
            )
        # vLLM downloads through huggingface_hub on first launch; doing it
        # ourselves would just duplicate its cache.
        config.model_path = spec.hf_repo
        config.extra["model_revision"] = pin_revision(spec.hf_repo, spec.hf_revision)
        console.info(f"vLLM will fetch {spec.hf_repo} on first start")
        return config

    def launch_command(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        cmd = [
            "vllm", "serve", config.model_path or spec.hf_repo or spec.id,
            "--host", config.host,
            "--port", str(config.port),
            "--max-model-len", str(config.context_tokens),
            "--served-model-name", config.served_name or spec.id,
            # vLLM pre-allocates this fraction of VRAM for weights + KV cache.
            "--gpu-memory-utilization", str(config.extra.get("gpu_memory_utilization", 0.90)),
            "--max-num-seqs", str(max(config.parallel, 1)),
        ]
        revision = config.extra.get("model_revision") or spec.hf_revision
        if revision:
            cmd += ["--revision", revision, "--tokenizer-revision", revision]
        tensor_parallel = int(config.extra.get("tensor_parallel_size", 1))
        if tensor_parallel > 1:
            cmd += ["--tensor-parallel-size", str(tensor_parallel)]
        quantization = config.extra.get("vllm_quantization")
        if quantization:
            cmd += ["--quantization", str(quantization)]
        if config.seed is not None:
            cmd += ["--seed", str(config.seed)]
        kv_dtype = kv_cache_dtype(config)
        if kv_dtype:
            cmd += ["--kv-cache-dtype", kv_dtype]
        parser = config.extra.get("tool_parser")
        if spec.supports_tools and parser:
            # Without these, tool calls come back as prose instead of structured
            # calls. With the wrong one, so do they, which is why an unknown
            # family gets no parser rather than a plausible-looking default.
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", str(parser)]
        cmd += self._advanced_flags(config)
        return cmd

    def _advanced_flags(self, config: RuntimeConfig) -> List[str]:
        """vLLM's own names for the deep settings."""
        advanced = config.advanced
        if advanced is None:
            return []
        import json

        flags: List[str] = []

        serving = advanced.serving
        if serving.prefix_caching:
            # Prefix reuse is the biggest single win for agent and RAG traffic,
            # where a long system prompt or document repeats every turn.
            flags.append("--enable-prefix-caching")
        if serving.chunked_prefill:
            flags.append("--enable-chunked-prefill")
            if serving.max_batched_tokens:
                flags += ["--max-num-batched-tokens", str(serving.max_batched_tokens)]

        rope = advanced.rope
        if rope.enabled:
            flags += ["--rope-scaling", json.dumps({
                "rope_type": rope.scaling_type,
                "factor": rope.factor,
                "original_max_position_embeddings": rope.original_context,
            })]

        spec_plan = advanced.speculative
        if spec_plan.enabled and spec_plan.draft_path:
            # vLLM cannot combine draft-model speculation with chunked prefill,
            # so the two are never emitted together.
            if "--enable-chunked-prefill" in flags:
                flags.remove("--enable-chunked-prefill")
                if "--max-num-batched-tokens" in flags:
                    index = flags.index("--max-num-batched-tokens")
                    del flags[index:index + 2]
            flags += ["--speculative-config", json.dumps({
                "model": spec_plan.draft_path,
                "num_speculative_tokens": spec_plan.max_draft_tokens,
            })]
        return flags

    def start(self, spec: ModelSpec, config: RuntimeConfig, console: Console) -> Endpoint:
        if not self.is_available():
            raise RuntimeUnavailable("vLLM is not installed", self.install_hint())

        if not http.port_is_free(config.host, config.port):
            config.port = http.find_free_port(config.host, config.port + 1)

        log_path = str(ensure_dir(app_dir() / "logs") / f"vllm-{config.port}.log")
        cmd = self.launch_command(spec, config)
        console.debug(" ".join(cmd))
        console.info("starting vLLM (first launch downloads weights and can take a while) …")
        Path(log_path).write_text("", encoding="utf-8")
        process = spawn(cmd, log_path=log_path)

        endpoint = Endpoint(
            base_url=f"http://{config.host}:{config.port}",
            model=config.served_name or spec.id,
            runtime=self.name,
            supports_embeddings=spec.is_embedding,
            started_by_us=True,
            log_path=log_path,
            process=process,
            command=cmd,
        )

        def ready() -> bool:
            if process.poll() is not None:
                raise RuntimeUnavailable(
                    f"vLLM exited with code {process.returncode}; see {log_path}"
                )
            return http.request(f"{endpoint.base_url}/health", timeout=3).ok

        self.wait_until_ready(endpoint, ready, timeout=1800, interval=3.0)
        return endpoint

    def config_files(self, spec: ModelSpec, config: RuntimeConfig) -> Dict[str, str]:
        cmd = self.launch_command(spec, config)
        script = ("#!/usr/bin/env bash\n# Generated by llmsetup.\nset -euo pipefail\n\nexec "
                  + " \\\n  ".join(shlex.quote(part) for part in cmd) + "\n")
        return {f"start-{spec.id}-vllm.sh": script}


register(VllmRuntime())
