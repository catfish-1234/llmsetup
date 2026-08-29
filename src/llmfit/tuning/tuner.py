"""The auto-tuner.

Takes the winning candidate and the real machine, and produces the concrete
settings the runtime will be launched with. Every value here is derived from a
measured or estimated quantity — none of it is a fixed default — and the
reasoning for each is recorded so the report can justify the numbers.

The order matters: GPU offload is decided first because it determines how much
memory the rest of the settings have to work with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from llmfit.planner.memory import (
    estimate_memory,
    layers_that_fit,
    max_context_for_budget,
)
from llmfit.tuning.sampling import sampling_for
from llmfit.types import (
    Accelerator,
    Candidate,
    HardwareProfile,
    RuntimeConfig,
    Task,
    UseCaseProfile,
)

# Fraction of the memory budget left unallocated. Fragmentation, a longer prompt
# than planned, and the compositor all live in this margin.
SAFETY_MARGIN = 0.08

# Prompt-processing batch. Bigger is faster to first token but costs memory;
# these are the values where the curve flattens on consumer and server cards.
_BATCH_BY_VRAM_GB = ((48, 4096, 1024), (20, 2048, 512), (10, 1024, 512), (0, 512, 256))


@dataclass
class TuningDecision:
    setting: str
    value: str
    reason: str


def tune(
    candidate: Candidate,
    hardware: HardwareProfile,
    profile: UseCaseProfile,
    runtime: str,
    port: int = 11435,
) -> "TunedConfig":
    """Produce a launch-ready :class:`RuntimeConfig` plus the reasoning behind it."""
    spec = candidate.model
    decisions: List[TuningDecision] = []

    budget = int(hardware.memory_budget * (1 - SAFETY_MARGIN))
    parallel = _parallel(profile, decisions)
    batch, ubatch = _batch_sizes(hardware, profile, decisions)
    threads = _threads(hardware, candidate, decisions)
    kv_type = candidate.estimate.kv_quant
    context = profile.context_tokens

    # 1. GPU offload. This is the single highest-impact setting: a model fully on
    #    the GPU is often 10-30x faster than one that is 90% there.
    gpu_layers, context, kv_type = _offload(
        candidate, hardware, profile, runtime, budget,
        parallel, ubatch, context, kv_type, decisions,
    )

    # 2. Flash attention: faster prompt processing and a smaller KV cache, but
    #    only where the backend has kernels for it.
    flash = _flash_attention(hardware, kv_type, decisions)

    # 3. Memory locking / mapping.
    mlock, mmap = _memory_mapping(hardware, candidate, gpu_layers, decisions)

    sampling = sampling_for(spec, profile)
    decisions.append(TuningDecision(
        "sampling",
        f"temp={sampling.temperature} top_p={sampling.top_p} top_k={sampling.top_k}",
        sampling.rationale,
    ))

    config = RuntimeConfig(
        runtime=runtime,
        model_id=spec.id,
        quant=candidate.quant,
        context_tokens=context,
        gpu_layers=gpu_layers,
        total_layers=spec.layers,
        batch_size=batch,
        ubatch_size=ubatch,
        threads=threads,
        parallel=parallel,
        kv_cache_type=kv_type,
        flash_attention=flash,
        mmap=mmap,
        mlock=mlock,
        port=port,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        repeat_penalty=sampling.repeat_penalty,
        served_name=spec.id,
        extra={"stop": sampling.stop},
    )
    _runtime_specific(config, candidate, hardware, profile, decisions)

    final = estimate_memory(
        spec, (candidate.estimate.weights * 8) / (spec.params_b * 1e9),
        context, quant=candidate.quant, kv_quant=kv_type,
        runtime=runtime, ubatch=ubatch, parallel=parallel,
    )
    return TunedConfig(config=config, decisions=_ordered(decisions), estimate=final)


# Decisions are appended in dependency order (batch size is needed to compute
# offload), but read best in impact order.
_REPORT_ORDER = [
    "gpu_layers", "context", "kv_cache_type", "flash_attention", "parallel",
    "batch/ubatch", "threads", "mlock", "gpu_memory_utilization",
    "tensor_parallel_size", "sampling",
]


def _ordered(decisions: List[TuningDecision]) -> List[TuningDecision]:
    def key(decision: TuningDecision) -> int:
        for index, name in enumerate(_REPORT_ORDER):
            if decision.setting.startswith(name):
                return index
        return len(_REPORT_ORDER)

    return sorted(decisions, key=key)


@dataclass
class TunedConfig:
    config: RuntimeConfig
    decisions: List[TuningDecision]
    estimate: object

    def explain(self) -> List[str]:
        return [f"{d.setting} = {d.value} — {d.reason}" for d in self.decisions]


# --------------------------------------------------------------------------- #
# individual settings
# --------------------------------------------------------------------------- #


def _offload(candidate, hardware, profile, runtime, budget, parallel, ubatch,
             context, kv_type, decisions):
    """Decide GPU layers, and shrink context if that is what makes it fit.

    The preference order when things do not fit is deliberate: quantize the KV
    cache first (cheap), then reduce context (visible but honest), then offload
    fewer layers (expensive, so it is the last resort).
    """
    spec = candidate.model

    if not hardware.gpus:
        decisions.append(TuningDecision(
            "gpu_layers", "0", "no GPU detected; all layers run on CPU"))
        max_ctx = max_context_for_budget(
            spec, _bpw(candidate), hardware.usable_cpu_bytes, kv_quant=kv_type,
            runtime=runtime, ubatch=ubatch, parallel=parallel,
        )
        if max_ctx and max_ctx < context:
            decisions.append(TuningDecision(
                "context", f"{max_ctx}",
                f"reduced from {context}: system RAM cannot hold a larger KV cache",
            ))
            context = max_ctx
        return 0, context, kv_type

    bpw = _bpw(candidate)
    estimate = estimate_memory(spec, bpw, context, quant=candidate.quant,
                               kv_quant=kv_type, runtime=runtime,
                               ubatch=ubatch, parallel=parallel)

    if estimate.total <= budget:
        decisions.append(TuningDecision(
            "gpu_layers", f"{spec.layers} (all)",
            f"the whole model plus its KV cache fits in {budget / 1024 ** 3:.1f} GB",
        ))
        return spec.layers, context, kv_type

    # Try a cheaper KV cache before giving up any context.
    from llmfit.planner.memory import KV_QUALITY_ORDER

    for candidate_kv in KV_QUALITY_ORDER[KV_QUALITY_ORDER.index(kv_type) + 1:]:
        trial = estimate_memory(spec, bpw, context, quant=candidate.quant,
                                kv_quant=candidate_kv, runtime=runtime,
                                ubatch=ubatch, parallel=parallel)
        if trial.total <= budget:
            decisions.append(TuningDecision(
                "kv_cache_type", candidate_kv,
                f"quantizing the KV cache saves "
                f"{(estimate.kv_cache - trial.kv_cache) / 1024 ** 3:.1f} GB and keeps "
                f"the whole model on the GPU — worth far more than the small "
                f"quality cost",
            ))
            decisions.append(TuningDecision(
                "gpu_layers", f"{spec.layers} (all)", "fits once the KV cache is quantized"))
            return spec.layers, context, candidate_kv

    # Next: shrink context to whatever genuinely fits.
    kv_type = KV_QUALITY_ORDER[-1] if kv_type == "f16" else kv_type
    fitted_ctx = max_context_for_budget(
        spec, bpw, budget, kv_quant=kv_type, runtime=runtime,
        ubatch=ubatch, parallel=parallel,
    )
    min_useful = _minimum_context(profile)
    if fitted_ctx >= min_useful:
        decisions.append(TuningDecision(
            "context", f"{fitted_ctx}",
            f"reduced from {context} so every layer stays on the GPU; a shorter "
            f"context on the GPU beats a longer one spilling to CPU",
        ))
        return spec.layers, fitted_ctx, kv_type

    # Last resort: partial offload, which costs an order of magnitude in speed.
    layers = layers_that_fit(spec, bpw, budget, context, kv_quant=kv_type,
                             runtime=runtime, ubatch=ubatch, parallel=parallel)
    decisions.append(TuningDecision(
        "gpu_layers", f"{layers} of {spec.layers}",
        f"the model cannot fit even at {min_useful} tokens of context, so "
        f"{spec.layers - layers} layers run on CPU — expect a large slowdown",
    ))
    return layers, context, kv_type


def _minimum_context(profile: UseCaseProfile) -> int:
    """Below this the use case stops working, so we stop trading context away."""
    if profile.primary_task in (Task.RAG, Task.SUMMARIZE):
        return 8192
    if profile.primary_task in (Task.CODING, Task.AGENT):
        return 8192
    return 2048


def _parallel(profile: UseCaseProfile, decisions) -> int:
    slots = max(1, min(profile.concurrent_users, 16))
    if slots > 1:
        decisions.append(TuningDecision(
            "parallel", str(slots),
            f"{profile.concurrent_users} concurrent users; each needs its own KV "
            f"cache slot",
        ))
    return slots


def _batch_sizes(hardware: HardwareProfile, profile: UseCaseProfile, decisions):
    vram_gb = (hardware.memory_budget) / 1024 ** 3
    batch, ubatch = next(
        (b, u) for threshold, b, u in _BATCH_BY_VRAM_GB if vram_gb >= threshold
    )
    reason = f"sized for {vram_gb:.0f} GB of usable memory"

    # RAG and summarization push long prompts through; prompt processing is the
    # dominant cost there, and it scales with batch size.
    if profile.primary_task in (Task.RAG, Task.SUMMARIZE) and vram_gb >= 10:
        batch = min(batch * 2, 8192)
        reason += "; doubled because long-prompt workloads are dominated by prefill"

    decisions.append(TuningDecision("batch/ubatch", f"{batch}/{ubatch}", reason))
    return batch, ubatch


def _threads(hardware: HardwareProfile, candidate: Candidate, decisions) -> Optional[int]:
    """Thread count for CPU work.

    Physical cores, not logical: hyperthreads contend for the same vector units,
    and on memory-bound GEMM they cost more in cache thrash than they return.
    """
    cores = hardware.physical_cores or hardware.logical_cores
    if not cores:
        return None
    if candidate.fits_on_gpu and hardware.gpus:
        # Fully offloaded: threads only drive sampling and tokenization.
        threads = min(4, cores)
        decisions.append(TuningDecision(
            "threads", str(threads),
            "model is fully on the GPU, so only a few CPU threads are needed",
        ))
        return threads
    threads = max(1, cores - 1)
    decisions.append(TuningDecision(
        "threads", str(threads),
        f"{cores} physical cores, leaving one for the OS; hyperthreads are "
        f"excluded because they contend for the same vector units",
    ))
    return threads


def _flash_attention(hardware: HardwareProfile, kv_type: str, decisions) -> bool:
    supported = bool(hardware.gpus) and hardware.gpus[0].supports_flash_attention
    if kv_type != "f16" and not supported:
        # Quantized KV needs flash attention on llama.cpp backends.
        decisions.append(TuningDecision(
            "flash_attention", "on (forced)",
            "a quantized KV cache requires flash attention on this backend",
        ))
        return True
    if supported:
        decisions.append(TuningDecision(
            "flash_attention", "on",
            "supported by this backend; reduces KV cache memory and speeds up prefill",
        ))
        return True
    decisions.append(TuningDecision(
        "flash_attention", "off",
        f"no flash-attention kernels for {hardware.accelerator.value} on this backend",
    ))
    return False


def _memory_mapping(hardware: HardwareProfile, candidate: Candidate,
                    gpu_layers: int, decisions):
    """mlock/mmap.

    mmap lets the OS page weights in lazily, which is what makes a partially
    offloaded model start fast. mlock pins them in RAM to stop the OS swapping
    them back out mid-generation — worth it only when there is genuinely spare RAM,
    since pinning on a tight machine causes the exact swapping it prevents.
    """
    weights = candidate.estimate.weights
    ram = hardware.ram_available_bytes or hardware.ram_bytes
    cpu_resident = gpu_layers < candidate.model.layers

    if cpu_resident and weights < ram * 0.5:
        decisions.append(TuningDecision(
            "mlock", "on",
            "part of the model stays in RAM and there is room to pin it, which "
            "prevents the OS swapping weights out mid-generation",
        ))
        return True, True
    return False, True


def _runtime_specific(config: RuntimeConfig, candidate: Candidate,
                      hardware: HardwareProfile, profile: UseCaseProfile, decisions):
    """Settings only one engine understands."""
    spec = candidate.model

    if config.runtime == "vllm":
        utilisation = round(min(0.95, 1.0 - SAFETY_MARGIN), 2)
        config.extra["gpu_memory_utilization"] = utilisation
        decisions.append(TuningDecision(
            "gpu_memory_utilization", str(utilisation),
            "vLLM pre-allocates this share of VRAM for weights and paged KV cache",
        ))
        if len(hardware.gpus) > 1:
            # Tensor parallelism requires the head count to divide evenly.
            usable = max(n for n in (1, 2, 4, 8)
                         if n <= len(hardware.gpus) and spec.kv_heads % n == 0)
            config.extra["tensor_parallel_size"] = usable
            decisions.append(TuningDecision(
                "tensor_parallel_size", str(usable),
                f"{len(hardware.gpus)} GPUs present; {usable} divides the model's "
                f"{spec.kv_heads} KV heads evenly",
            ))
        if spec.supports_tools:
            config.extra["tool_parser"] = _tool_parser(spec)
        if config.kv_cache_type != "f16":
            config.extra["kv_cache_dtype"] = "fp8"

    if config.runtime == "ollama":
        config.extra["ollama_port"] = 11434

    if hardware.accelerator is Accelerator.AMD:
        # llama.cpp's ROCm/HIP path needs the target architecture when the card
        # is not one it recognises.
        config.extra.setdefault("hint", "if offload fails, set HSA_OVERRIDE_GFX_VERSION")


def _tool_parser(spec) -> str:
    """vLLM needs to be told how this family emits tool calls."""
    publisher = spec.publisher.lower()
    if "qwen" in publisher or "qwen" in spec.id:
        return "hermes"
    if "mistral" in publisher:
        return "mistral"
    if "meta-llama" in publisher or "llama" in spec.id:
        return "llama3_json"
    return "hermes"


def _bpw(candidate: Candidate) -> float:
    params = candidate.model.params_b * 1e9
    return (candidate.estimate.weights * 8) / params if params else 4.85
