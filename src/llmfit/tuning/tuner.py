"""The auto-tuner.

Takes the winning candidate and the real machine, and produces the concrete
settings the runtime will be launched with. Every value here is derived from a
measured or estimated quantity, none of it a fixed default, and the
reasoning for each is recorded so the report can justify the numbers.

The order matters: GPU offload is decided first because it determines how much
memory the rest of the settings have to work with.
"""

from __future__ import annotations

import re
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
    catalog=None,
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

    # Sampling depends on whether thinking is actually on: the publishers that
    # state both modes give different values for each, and a thinking-mode
    # temperature applied to a run with thinking disabled is simply the wrong
    # setting. So the reasoning decision is made first and fed in.
    from llmfit.tuning.advanced import plan_reasoning

    reasoning_plan = plan_reasoning(spec, profile)
    sampling = sampling_for(spec, profile, thinking=reasoning_plan.enabled)
    decisions.append(TuningDecision(
        "sampling",
        f"temp={sampling.temperature} top_p={sampling.top_p} top_k={sampling.top_k}"
        + (f" presence={sampling.presence_penalty}" if sampling.presence_penalty else ""),
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
        extra={"stop": sampling.stop,
               "presence_penalty": sampling.presence_penalty,
               "sampling_source": sampling.source},
    )
    _fix_seed(config, profile, decisions)
    _runtime_specific(config, candidate, hardware, profile, decisions)

    # Deep settings last: they depend on the offload decision above, and the
    # speculative planner needs to know what memory is left over.
    if catalog is not None:
        from llmfit.tuning.advanced import plan_advanced

        candidate._presence_penalty = sampling.presence_penalty
        config.advanced = plan_advanced(
            candidate, hardware, profile, catalog, context,
            gpu_layers, _bpw(candidate), runtime,
        )
        # MoE expert offload can keep every layer on the GPU where partial layer
        # offload would otherwise have been forced.
        if config.advanced.offload.moe_offload_enabled:
            config.gpu_layers = config.advanced.offload.gpu_layers
        for line in config.advanced.explain():
            setting, _, reason = line.partition(": ")
            decisions.append(TuningDecision(setting, "planned", reason))

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
        return [f"{d.setting} = {d.value} - {d.reason}" for d in self.decisions]


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

    # Try a cheaper KV cache before giving up any context -- but only down to
    # the rung where that is still a free saving. Below it, a quantized cache
    # changes answers unpredictably and, on long context, is slower too; a
    # shorter context is the honest trade at that point.
    from llmfit.planner.memory import KV_QUALITY_ORDER, KV_SAFE_FLOOR

    floor = KV_QUALITY_ORDER.index(KV_SAFE_FLOOR)
    for candidate_kv in KV_QUALITY_ORDER[
            KV_QUALITY_ORDER.index(kv_type) + 1:floor + 1]:
        trial = estimate_memory(spec, bpw, context, quant=candidate.quant,
                                kv_quant=candidate_kv, runtime=runtime,
                                ubatch=ubatch, parallel=parallel)
        if trial.total <= budget:
            decisions.append(TuningDecision(
                "kv_cache_type", candidate_kv,
                f"quantizing the KV cache saves "
                f"{(estimate.kv_cache - trial.kv_cache) / 1024 ** 3:.1f} GB and keeps "
                f"the whole model on the GPU, worth far more than the small "
                f"quality cost",
            ))
            decisions.append(TuningDecision(
                "gpu_layers", f"{spec.layers} (all)", "fits once the KV cache is quantized"))
            return spec.layers, context, candidate_kv

    # Next: shrink context to whatever genuinely fits, with the cache at the
    # cheapest dtype that costs nothing.
    kv_type = KV_SAFE_FLOOR if kv_type == "f16" else kv_type
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
        f"{spec.layers - layers} layers run on CPU, expect a large slowdown",
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
    them back out mid-generation, worth it only when there is genuinely spare RAM,
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
            parser = _tool_parser(spec)
            if parser:
                config.extra["tool_parser"] = parser
            else:
                decisions.append(TuningDecision(
                    "tool_parser", "none",
                    f"vLLM has no documented tool-call parser for "
                    f"{spec.name}'s family, and guessing one is how a good tool "
                    f"call comes back as prose. Auto tool choice stays off and "
                    f"the tool check will report what the endpoint really does",
                ))
        # The KV dtype is derived from config.kv_cache_type at launch rather
        # than stored here, so the repair loop's changes follow through.
        if config.kv_cache_type != "f16" and config.context_tokens > 32768 \
                and profile.primary_task in (Task.RAG, Task.AGENT, Task.SUMMARIZE):
            # An fp8 cache is a large, usually safe saving, but it is the one
            # setting that has been measured collapsing on long-context recall
            # specifically, which is exactly what this workload depends on. Say
            # so rather than either refusing the memory or staying quiet; the
            # needle check tests it, and the repair restores precision if it
            # fails.
            decisions.append(TuningDecision(
                "kv_cache_type", "fp8",
                "worth watching: an 8-bit cache saves a lot of memory and is "
                "usually indistinguishable, but it is long-context recall that "
                "it degrades first, and that is what this use case rests on. "
                "The needle check tests it and the repair restores precision "
                "if it fails",
            ))

    if config.runtime == "ollama":
        config.extra["ollama_port"] = 11434

    if hardware.accelerator is Accelerator.AMD:
        # llama.cpp's ROCm/HIP path needs the target architecture when the card
        # is not one it recognises.
        config.extra.setdefault("hint", "if offload fails, set HSA_OVERRIDE_GFX_VERSION")


#: How each family emits tool calls, as vLLM names the parsers. Only families
#: with a documented parser appear. The order matters for the fallbacks the
#: repair loop tries: the first entry is the default and the rest are the
#: plausible alternatives for that family.
_TOOL_PARSERS = {
    "qwen": ["hermes", "qwen3_xml"],
    "mistral": ["mistral"],
    "llama3": ["llama3_json"],
    "llama4": ["llama4_pythonic", "pythonic"],
    "deepseek": ["deepseek_v3"],
    "granite": ["granite", "granite4"],
    "glm": ["glm45"],
    "kimi": ["kimi_k2"],
    "internlm": ["internlm"],
}


def tool_parsers_for(spec) -> List[str]:
    """Parsers worth trying for this model, best first.

    Empty when the family has no documented parser. That is the useful answer:
    passing a parser that does not match the family is not a harmless default,
    it is the exact cause of the failure it is meant to prevent, where the model
    emits a perfectly good tool call and the server hands back prose because it
    was looking for another shape. Better to leave auto tool choice off and let
    the tool check report honestly what the endpoint does.
    """
    # Separators are stripped so a key matches however the name is punctuated:
    # "llama-3.1-8b", "Llama_3.1" and "llama3" all have to reach the same entry,
    # and a catalog entry added by hand will not have a tokenizer family set.
    haystack = re.sub(r"[^a-z0-9]", "",
                      f"{spec.publisher} {spec.id} {spec.family}".lower())
    if "llama4" in haystack:
        return list(_TOOL_PARSERS["llama4"])

    # A model can match two families: a DeepSeek distill of Qwen carries Qwen's
    # tokenizer and chat template, which is what decides the tool format, but
    # DeepSeek's own parser is still worth trying if that fails. Collect every
    # match rather than taking whichever the dictionary happened to yield first.
    found: List[str] = []
    for key in sorted(_TOOL_PARSERS, key=lambda k: -len(k)):
        if key == "llama4" or key not in haystack:
            continue
        for parser in _TOOL_PARSERS[key]:
            if parser not in found:
                found.append(parser)
    return found


#: Chosen once and written down rather than generated, because the point of a
#: seed is that the same setup gives the same answer tomorrow as today.
_DEFAULT_SEED = 1337


def _fix_seed(config, profile, decisions) -> None:
    """Pin the sampler's seed where the task wants the same answer every time.

    Worth being precise about what this buys, because most advice overstates it.
    A fixed seed removes the sampler as a source of variation, which is what
    makes a low-temperature run repeatable on an idle machine. It does not make
    the engine deterministic: floating-point addition is not associative, so a
    request batched with three others can accumulate in a different order and
    come out different. That part is a platform setting, not a sampler one, and
    is reported separately.
    """
    if config.seed is not None:
        return
    if profile.primary_task not in _DETERMINISTIC_TASKS:
        return
    config.seed = _DEFAULT_SEED
    decisions.append(TuningDecision(
        "seed", str(_DEFAULT_SEED),
        "fixed so the same prompt gives the same answer between runs, which "
        "this task depends on. Note that it removes the sampler as a source of "
        "variation but not the arithmetic: under concurrent load the same "
        "prompt can still batch differently and come out different",
    ))


#: Tasks where a repeated run producing a different answer is a defect rather
#: than variety. Kept in step with the sampling module's own list.
_DETERMINISTIC_TASKS = {Task.EXTRACT, Task.CODING, Task.TRANSLATE,
                        Task.AUTOCOMPLETE}


def _tool_parser(spec) -> Optional[str]:
    """vLLM needs to be told how this family emits tool calls, or nothing."""
    parsers = tool_parsers_for(spec)
    return parsers[0] if parsers else None


def _bpw(candidate: Candidate) -> float:
    params = candidate.model.params_b * 1e9
    return (candidate.estimate.weights * 8) / params if params else 4.85
