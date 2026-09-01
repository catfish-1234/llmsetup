"""llama.cpp (`llama-server`) adapter.

The reference target: it exposes every knob the tuner computes, so it is the
runtime where the tuning is most fully expressed. Other adapters approximate
these flags with whatever their engine offers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.runtimes.gguf_store import fetch_gguf, fetch_projector
from llmfit.runtimes.registry import register
from llmfit.types import ModelSpec, RuntimeConfig
from llmfit.util import http
from llmfit.util.fs import app_dir, ensure_dir
from llmfit.util.log import Console
from llmfit.util.proc import first_line, run, spawn, which

#: Flags that stand alone, so filtering must not swallow the token after them.
_VALUELESS_FLAGS = {
    "--jinja", "--mlock", "--no-mmap", "--embeddings", "--cont-batching",
    "--reasoning-preserve", "--swa-full", "--no-context-shift", "--context-shift",
}


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
        if not (config.model_path and Path(config.model_path).exists()):
            config.model_path = str(fetch_gguf(spec, config.quant, console))
        # A vision model without its projector loads and serves happily; it just
        # cannot see. Fetch it here so the failure never reaches the user.
        if spec.supports_vision and not config.mmproj_path:
            projector = fetch_projector(spec, config.quant, console)
            if projector is not None:
                config.mmproj_path = str(projector)
            else:
                console.warn(
                    f"{spec.gguf_repo} publishes no multimodal projector, so "
                    f"{spec.name} will run as a text-only model")
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
        if config.mmproj_path:
            cmd += ["--mmproj", config.mmproj_path]
        if config.swa_full:
            cmd.append("--swa-full")
        if config.threads:
            cmd += ["--threads", str(config.threads)]
        # Flash attention is an explicit on/off in current llama-server; leaving
        # it unset lets it auto-select, which we do not want when the repair loop
        # has deliberately turned it off.
        cmd += ["--flash-attn", "on" if config.flash_attention else "off"]
        if config.kv_cache_type != "f16":
            cmd += ["--cache-type-k", config.kv_cache_type,
                    "--cache-type-v", config.kv_cache_type]
        cmd += self._load_mode_flags(config)
        if spec.is_embedding:
            cmd.append("--embeddings")

        cmd += self._advanced_flags(spec, config)

        for flag in config.extra.get("llamacpp_flags", []):
            cmd.append(str(flag))
        return self._drop_unsupported(cmd)

    def _load_mode_flags(self, config: RuntimeConfig) -> List[str]:
        """How the weights get into memory.

        Current llama.cpp folds mmap and mlock into one `--load-mode`; older
        builds have the two separate booleans. Emit whichever the installed
        binary understands, because getting this wrong is not cosmetic: without
        mlock, part of a model that only just fits can be paged out mid
        generation and the speed collapses.
        """
        if "--load-mode" in self._flags():
            if config.mlock:
                return ["--load-mode", "mmap+mlock" if config.mmap else "mlock"]
            return [] if config.mmap else ["--load-mode", "none"]
        flags = []
        if config.mlock:
            flags.append("--mlock")
        if not config.mmap:
            flags.append("--no-mmap")
        return flags

    def _flags(self):
        """Long flags the installed binary accepts; empty when it cannot be asked."""
        from llmfit.runtimes.capabilities import supported_flags

        return supported_flags(self.binary)

    def _drop_unsupported(self, cmd: List[str]) -> List[str]:
        """Strip anything this build does not know about.

        llama.cpp renames and retires flags between releases and people run
        whatever their package manager last shipped. Asking the binary what it
        accepts turns "the server refused to start and the log is 400 lines of
        usage text" into a command that simply runs with one feature less.
        """
        from llmfit.runtimes.capabilities import filter_flags

        return filter_flags(cmd, self._flags(), drop_value=_VALUELESS_FLAGS)

    def _speculative_flags(self, plan) -> List[str]:
        """Flags for whichever kind of drafting was planned.

        The way these are spelled changed: the draft-model options moved under a
        `--spec-` prefix and gained a `--spec-type` that selects between a
        separate draft model, the model's own prediction heads, purpose-trained
        draft heads, and drafting straight from the prompt. On a build that
        predates all that, the unsupported flags are dropped and the setup runs
        without drafting rather than failing to start.
        """
        if not plan.enabled:
            return []
        flags = ["--spec-type", plan.spec_type.value]
        if plan.needs_draft_model:
            if not plan.draft_path:
                # Planned but never provisioned. Emitting a type with no model
                # behind it would fail at load, so say nothing.
                return []
            flags += ["--spec-draft-model", plan.draft_path,
                      "--spec-draft-ngl", str(plan.draft_gpu_layers)]
        flags += ["--spec-draft-n-max", str(plan.max_draft_tokens),
                  "--spec-draft-n-min", str(plan.min_draft_tokens)]
        if plan.p_min > 0:
            # Reject a guess the drafter is not confident about. Verifying a bad
            # guess costs a full forward pass, so a low bar loses time rather
            # than saving it.
            flags += ["--spec-draft-p-min", str(plan.p_min)]
        return flags

    def _advanced_flags(self, spec: ModelSpec, config: RuntimeConfig) -> List[str]:
        """Deep settings: placement, drafting, thinking budget, sampler chain.

        Everything here is off unless the tuner decided it was worth turning on,
        so a simple setup still produces a simple command line.
        """
        advanced = config.advanced
        if advanced is None:
            return []
        flags: List[str] = []

        # --- placement ---------------------------------------------------- #
        offload = advanced.offload
        if offload.moe_offload_enabled:
            # Moves only routed expert weights to system RAM; attention, the
            # router and the KV cache stay on the GPU.
            flags += ["--n-cpu-moe", str(offload.n_cpu_moe)]
        if offload.tensor_split:
            flags += ["--tensor-split", ",".join(str(v) for v in offload.tensor_split)]
            flags += ["--main-gpu", str(offload.main_gpu)]
        if offload.split_mode != "layer":
            flags += ["--split-mode", offload.split_mode]
        for pattern in offload.override_tensors:
            flags += ["--override-tensor", pattern]

        # --- speculative decoding ----------------------------------------- #
        flags += self._speculative_flags(advanced.speculative)

        # --- reasoning budget --------------------------------------------- #
        reasoning = advanced.reasoning
        if reasoning.supported:
            flags += ["--reasoning", "on" if reasoning.enabled else "off"]
            if reasoning.enabled and reasoning.token_budget is not None:
                # A hard ceiling on thinking. Without one, a reasoning model
                # asked something ambiguous can spend its whole output budget
                # deliberating and never answer.
                flags += ["--reasoning-budget", str(reasoning.token_budget)]
            if reasoning.enabled and reasoning.preserve:
                flags.append("--reasoning-preserve")

        # --- RoPE / context extension ------------------------------------- #
        rope = advanced.rope
        if rope.enabled:
            flags += ["--rope-scaling", rope.scaling_type,
                      "--rope-scale", str(rope.factor)]
            if rope.original_context:
                flags += ["--yarn-orig-ctx", str(rope.original_context)]
            if rope.yarn_ext_factor is not None:
                flags += ["--yarn-ext-factor", str(rope.yarn_ext_factor)]
            if rope.yarn_attn_factor is not None:
                flags += ["--yarn-attn-factor", str(rope.yarn_attn_factor)]
            if rope.yarn_beta_fast is not None:
                flags += ["--yarn-beta-fast", str(rope.yarn_beta_fast)]
            if rope.yarn_beta_slow is not None:
                flags += ["--yarn-beta-slow", str(rope.yarn_beta_slow)]
            if rope.freq_base:
                flags += ["--rope-freq-base", str(rope.freq_base)]

        # --- sampler chain ------------------------------------------------- #
        sampler = advanced.sampler
        if sampler.dry_enabled:
            flags += ["--dry-multiplier", str(sampler.dry_multiplier),
                      "--dry-base", str(sampler.dry_base),
                      "--dry-allowed-length", str(sampler.dry_allowed_length),
                      "--dry-penalty-last-n", str(sampler.dry_penalty_last_n)]
            for breaker in sampler.dry_sequence_breakers:
                # Without these, DRY penalises a repeated newline as heavily as
                # a repeated phrase, which mangles anything structured.
                flags += ["--dry-sequence-breaker", breaker]
        if sampler.xtc_enabled:
            flags += ["--xtc-probability", str(sampler.xtc_probability),
                      "--xtc-threshold", str(sampler.xtc_threshold)]
        if sampler.top_n_sigma >= 0:
            flags += ["--top-nsigma", str(sampler.top_n_sigma)]
        if sampler.typical_p < 1.0:
            flags += ["--typical", str(sampler.typical_p)]
        if sampler.mirostat_enabled:
            flags += ["--mirostat", str(sampler.mirostat),
                      "--mirostat-lr", str(sampler.mirostat_eta),
                      "--mirostat-ent", str(sampler.mirostat_tau)]
        flags += ["--repeat-last-n", str(sampler.penalty_last_n)]
        if sampler.frequency_penalty:
            flags += ["--frequency-penalty", str(sampler.frequency_penalty)]
        if sampler.presence_penalty:
            flags += ["--presence-penalty", str(sampler.presence_penalty)]
        flags += ["--samplers", ";".join(sampler.normalised_order())]

        # --- serving ------------------------------------------------------- #
        serving = advanced.serving
        if serving.continuous_batching:
            flags.append("--cont-batching")
        if serving.prefix_caching and serving.cache_reuse_tokens:
            flags += ["--cache-reuse", str(serving.cache_reuse_tokens)]
        if serving.cache_ram_mib is not None:
            # Prompt cache held in system RAM across requests, so a returning
            # conversation skips its prefill instead of paying for it again.
            flags += ["--cache-ram", str(serving.cache_ram_mib)]
        if serving.keep_tokens:
            # A long-running agent has a fixed system prompt at the front. Pin
            # it so a context shift evicts the conversation and not the rules.
            flags += ["--keep", str(serving.keep_tokens)]
        if serving.api_key:
            flags += ["--api-key", serving.api_key]
        for adapter in serving.lora_adapters:
            flags += ["--lora", adapter]
        return flags

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
