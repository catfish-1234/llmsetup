"""Remediation strategies.

Each strategy declares which failure signatures it answers, what it costs the
user, and whether it needs a server restart. The loop picks the *cheapest
applicable untried* strategy, so the setup degrades gracefully: it gives up a
quantized KV cache before it gives up context, and context before it gives up
the model.

Cost is a real ordering, not a label:

    low     no meaningful quality or speed loss (batch size, flash attention)
    medium  a visible but acceptable trade (shorter context, quantized KV)
    high    a different experience (smaller quantization, a different model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from llmfit.catalog.catalog import Catalog
from llmfit.planner.memory import KV_QUALITY_ORDER, max_context_for_budget
from llmfit.types import (
    Candidate,
    HardwareProfile,
    RuntimeConfig,
    Task,
    UseCaseProfile,
)
from llmfit.verify import signatures as sig

COST_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class RepairState:
    """Everything a strategy may read or change."""

    config: RuntimeConfig
    candidate: Candidate
    hardware: HardwareProfile
    profile: UseCaseProfile
    catalog: Catalog
    fallbacks: List[Candidate] = field(default_factory=list)
    #: Shrinking multiplier applied to the memory budget after each OOM. Reality
    #: has beaten our estimate, so the next attempt assumes less room.
    budget_factor: float = 0.92

    @property
    def budget(self) -> int:
        return int(self.hardware.memory_budget * self.budget_factor)


@dataclass
class Outcome:
    """What a strategy did. ``None`` from a strategy means "not applicable"."""

    description: str
    cost: str
    requires_restart: bool = True
    recheck: Optional[Sequence[str]] = None      # None means re-run everything
    before: Dict = field(default_factory=dict)
    after: Dict = field(default_factory=dict)
    switch_to: Optional[Candidate] = None        # set by the model-swap strategy


@dataclass
class RepairStrategy:
    name: str
    triggers: Sequence[str]
    cost: str
    apply: Callable[[RepairState], Optional[Outcome]]
    description: str = ""

    def handles(self, signature: str) -> bool:
        return signature in self.triggers


# --------------------------------------------------------------------------- #
# memory pressure
# --------------------------------------------------------------------------- #


def _enable_flash_attention(state: RepairState) -> Optional[Outcome]:
    config = state.config
    if config.flash_attention:
        return None
    if not state.hardware.gpus or not state.hardware.gpus[0].supports_flash_attention:
        return None
    before = {"flash_attention": False}
    config.flash_attention = True
    return Outcome(
        "enabled flash attention, which shrinks the KV cache and speeds up prefill "
        "at no quality cost",
        cost="low", before=before, after={"flash_attention": True},
    )


def _reduce_batch(state: RepairState) -> Optional[Outcome]:
    config = state.config
    if config.ubatch_size <= 128:
        return None
    before = {"batch_size": config.batch_size, "ubatch_size": config.ubatch_size}
    config.batch_size = max(256, config.batch_size // 2)
    config.ubatch_size = max(128, config.ubatch_size // 2)
    return Outcome(
        f"halved the prompt batch to {config.batch_size}/{config.ubatch_size}, "
        f"freeing compute-buffer memory; long prompts process a little slower",
        cost="low", before=before,
        after={"batch_size": config.batch_size, "ubatch_size": config.ubatch_size},
    )


def _quantize_kv(state: RepairState) -> Optional[Outcome]:
    config = state.config
    current = config.kv_cache_type.lower()
    try:
        index = KV_QUALITY_ORDER.index(current)
    except ValueError:
        index = 0
    if index + 1 >= len(KV_QUALITY_ORDER):
        return None
    nxt = KV_QUALITY_ORDER[index + 1]

    before = {"kv_cache_type": current}
    config.kv_cache_type = nxt
    # Quantized KV requires flash attention on llama.cpp-family backends.
    config.flash_attention = True
    cost = "low" if nxt == "q8_0" else "medium"
    note = ("q8_0 is effectively lossless" if nxt == "q8_0"
            else "this costs some recall over long context")
    return Outcome(
        f"quantized the KV cache to {nxt} to free memory ({note})",
        cost=cost, before=before, after={"kv_cache_type": nxt},
    )


def _reduce_context(state: RepairState) -> Optional[Outcome]:
    """Shrink context to what genuinely fits, not by a blind halving."""
    config = state.config
    spec = state.candidate.model
    bpw = (state.candidate.estimate.weights * 8) / (spec.params_b * 1e9)

    fitted = max_context_for_budget(
        spec, bpw, state.budget, kv_quant=config.kv_cache_type,
        runtime=config.runtime, ubatch=config.ubatch_size, parallel=config.parallel,
    )
    floor = _context_floor(state.profile)
    # Always make real progress: if the computed fit is not smaller, step down.
    target = min(fitted, int(config.context_tokens * 0.75))
    target = (target // 1024) * 1024
    if target < floor or target >= config.context_tokens:
        return None

    before = {"context_tokens": config.context_tokens}
    config.context_tokens = target
    return Outcome(
        f"reduced context from {before['context_tokens']:,} to {target:,} tokens, "
        f"which is what actually fits in the memory available",
        cost="medium", before=before, after={"context_tokens": target},
    )


def _context_floor(profile: UseCaseProfile) -> int:
    if profile.primary_task in (Task.RAG, Task.SUMMARIZE, Task.CODING, Task.AGENT):
        return 8192
    return 2048


def _reduce_gpu_layers(state: RepairState) -> Optional[Outcome]:
    config = state.config
    if config.gpu_layers <= 0:
        return None
    step = max(1, config.total_layers // 8)
    target = max(0, config.gpu_layers - step)
    before = {"gpu_layers": config.gpu_layers}
    config.gpu_layers = target
    return Outcome(
        f"moved {step} more layers off the GPU ({target}/{config.total_layers} "
        f"remain offloaded); this frees VRAM but costs a lot of speed",
        cost="high", before=before, after={"gpu_layers": target},
    )


def _drop_quant_tier(state: RepairState) -> Optional[Outcome]:
    """Move to a smaller quantization of the same model."""
    spec = state.candidate.model
    quants = list(spec.recommended_quants)
    current = state.config.quant
    if current not in quants:
        return None
    # recommended_quants is ordered smallest-first by convention in the catalog.
    smaller = [q for q in quants
               if state.catalog.bpw(q) < state.catalog.bpw(current)]
    if not smaller:
        return None
    target = max(smaller, key=state.catalog.bpw)   # the largest of the smaller ones

    before = {"quant": current}
    state.config.quant = target
    state.config.model_path = None                 # must be re-fetched
    saved = (state.catalog.bpw(current) - state.catalog.bpw(target)) / 8 * spec.params_b
    return Outcome(
        f"dropped from {current} to {target}, saving about {saved:.1f} GB of weights "
        f"at a real but modest quality cost",
        cost="high", before=before, after={"quant": target},
    )


def _switch_model(state: RepairState) -> Optional[Outcome]:
    """Last resort: the next-best model from the plan."""
    if not state.fallbacks:
        return None
    nxt = state.fallbacks.pop(0)
    before = {"model": state.candidate.label}
    return Outcome(
        f"switched from {state.candidate.model.name} to {nxt.model.name} "
        f"({nxt.quant}), the next-best fit for this machine",
        cost="high", before=before, after={"model": nxt.label},
        switch_to=nxt,
    )


def _reduce_parallel(state: RepairState) -> Optional[Outcome]:
    config = state.config
    if config.parallel <= 1:
        return None
    before = {"parallel": config.parallel}
    config.parallel = max(1, config.parallel // 2)
    return Outcome(
        f"reduced concurrent slots from {before['parallel']} to {config.parallel}; "
        f"each slot needs its own KV cache, so this frees memory proportionally",
        cost="medium", before=before, after={"parallel": config.parallel},
    )


# --------------------------------------------------------------------------- #
# correctness, not capacity
# --------------------------------------------------------------------------- #


def _restore_kv_precision(state: RepairState) -> Optional[Outcome]:
    """Long-context recall failed — undo the KV quantization that caused it.

    The needle check failing while everything else passes is the signature of an
    over-quantized KV cache. The fix trades context back for precision, which is
    the right direction: a shorter context that answers correctly beats a long
    one that does not.
    """
    config = state.config
    current = config.kv_cache_type.lower()
    if current == "f16":
        return None
    index = KV_QUALITY_ORDER.index(current) if current in KV_QUALITY_ORDER else 1
    better = KV_QUALITY_ORDER[max(0, index - 1)]

    spec = state.candidate.model
    bpw = (state.candidate.estimate.weights * 8) / (spec.params_b * 1e9)
    fitted = max_context_for_budget(
        spec, bpw, state.budget, kv_quant=better, runtime=config.runtime,
        ubatch=config.ubatch_size, parallel=config.parallel,
    )
    target = min(config.context_tokens, (fitted // 1024) * 1024)
    if target < _context_floor(state.profile):
        return None

    before = {"kv_cache_type": current, "context_tokens": config.context_tokens}
    config.kv_cache_type = better
    config.context_tokens = target
    return Outcome(
        f"restored KV cache precision to {better} (context now {target:,} tokens) — "
        f"aggressive KV quantization was costing long-context recall",
        cost="medium", before=before,
        after={"kv_cache_type": better, "context_tokens": target},
    )


def _disable_flash_attention(state: RepairState) -> Optional[Outcome]:
    """Some backend/driver combinations produce garbage with flash attention on."""
    config = state.config
    if not config.flash_attention:
        return None
    if config.kv_cache_type != "f16":
        return None   # quantized KV needs it; turning it off would break worse
    before = {"flash_attention": True}
    config.flash_attention = False
    return Outcome(
        "disabled flash attention; some driver and backend combinations produce "
        "incorrect output with it enabled",
        cost="low", before=before, after={"flash_attention": False},
    )


def _fix_repetition_sampling(state: RepairState) -> Optional[Outcome]:
    """Break a repetition loop by loosening sampling."""
    config = state.config
    if config.temperature >= 0.9 and config.repeat_penalty >= 1.15:
        return None
    before = {"temperature": config.temperature, "repeat_penalty": config.repeat_penalty,
              "min_p": config.min_p}
    config.temperature = round(min(0.9, max(config.temperature + 0.25, 0.6)), 2)
    config.repeat_penalty = round(min(1.15, config.repeat_penalty + 0.05), 3)
    config.min_p = max(config.min_p, 0.05)
    return Outcome(
        f"raised temperature to {config.temperature} and repeat penalty to "
        f"{config.repeat_penalty} to break the repetition loop",
        cost="low", requires_restart=False,
        recheck=["repetition", "completion"],
        before=before,
        after={"temperature": config.temperature,
               "repeat_penalty": config.repeat_penalty, "min_p": config.min_p},
    )


def _enable_chat_template(state: RepairState) -> Optional[Outcome]:
    """Tool calls came back as prose: the model's chat template is not applied."""
    config = state.config
    flags = config.extra.setdefault("llamacpp_flags", [])
    if config.runtime == "llamacpp":
        # --jinja is already in the base command; if we are here it did not help,
        # so try an explicit template for the family.
        if config.extra.get("chat_template_forced"):
            return None
        config.extra["chat_template_forced"] = True
        template = _template_name(state.candidate.model)
        flags.extend(["--chat-template", template])
        return Outcome(
            f"forced the {template} chat template; without the right template the "
            f"model emits tool calls as plain text",
            cost="low", before={}, after={"chat_template": template},
            recheck=["tool_calling", "completion"],
        )

    if config.runtime == "vllm":
        parsers = ["hermes", "llama3_json", "mistral", "pythonic"]
        current = config.extra.get("tool_parser", "hermes")
        remaining = [p for p in parsers if p != current
                     and p not in config.extra.get("tried_parsers", [])]
        if not remaining:
            return None
        config.extra.setdefault("tried_parsers", []).append(current)
        config.extra["tool_parser"] = remaining[0]
        return Outcome(
            f"switched vLLM's tool-call parser from {current} to {remaining[0]}",
            cost="low", before={"tool_parser": current},
            after={"tool_parser": remaining[0]}, recheck=["tool_calling"],
        )
    return None


def _template_name(spec) -> str:
    identifier = f"{spec.publisher} {spec.id}".lower()
    if "qwen" in identifier:
        return "chatml"
    if "llama" in identifier:
        return "llama3"
    if "gemma" in identifier:
        return "gemma"
    if "mistral" in identifier:
        return "mistral"
    return "chatml"


def _enable_embeddings_flag(state: RepairState) -> Optional[Outcome]:
    """llama.cpp will not serve embeddings unless started with --embeddings."""
    config = state.config
    if config.runtime != "llamacpp":
        return None
    flags = config.extra.setdefault("llamacpp_flags", [])
    if "--embeddings" in flags:
        return None
    flags.extend(["--embeddings", "--pooling", "mean"])
    return Outcome(
        "restarted the server with --embeddings and mean pooling, which "
        "llama.cpp requires before it will serve the embeddings endpoint",
        cost="low", before={}, after={"embeddings_flag": True},
        recheck=["embeddings"],
    )


def _restart_server(state: RepairState) -> Optional[Outcome]:
    """A crashed or unreachable server gets exactly one free restart."""
    if state.config.extra.get("restarted_once"):
        return None
    state.config.extra["restarted_once"] = True
    return Outcome(
        "restarted the inference server", cost="low", before={}, after={},
    )


def _increase_gpu_layers(state: RepairState) -> Optional[Outcome]:
    """Throughput far below prediction: layers may have silently stayed on CPU."""
    config = state.config
    if config.gpu_layers >= config.total_layers or not state.hardware.gpus:
        return None
    before = {"gpu_layers": config.gpu_layers}
    config.gpu_layers = config.total_layers
    return Outcome(
        f"raised GPU offload to all {config.total_layers} layers; throughput was "
        f"far below what this hardware should manage, which points at layers "
        f"silently running on CPU",
        cost="low", before=before, after={"gpu_layers": config.total_layers},
        recheck=["throughput", "completion"],
    )


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

_MEMORY_SIGNATURES = (sig.OUT_OF_MEMORY, sig.CRASH_DURING_LOAD, sig.KV_CACHE_FULL)

STRATEGIES: List[RepairStrategy] = [
    # Cheap wins first.
    RepairStrategy("enable_flash_attention", _MEMORY_SIGNATURES, "low",
                   _enable_flash_attention),
    RepairStrategy("reduce_batch", _MEMORY_SIGNATURES, "low", _reduce_batch),
    RepairStrategy("quantize_kv_cache", (*_MEMORY_SIGNATURES, sig.CONTEXT_EXCEEDED),
                   "low", _quantize_kv),
    RepairStrategy("restart_server",
                   (sig.SERVER_UNREACHABLE, sig.CRASH_DURING_LOAD, sig.MODEL_NOT_LOADED),
                   "low", _restart_server),
    RepairStrategy("enable_chat_template",
                   (sig.TOOLS_UNSUPPORTED, sig.TOOL_CALL_MALFORMED), "low",
                   _enable_chat_template),
    RepairStrategy("enable_embeddings_flag", (sig.EMBEDDINGS_UNSUPPORTED,), "low",
                   _enable_embeddings_flag),
    RepairStrategy("fix_repetition_sampling", (sig.REPETITION_LOOP,), "low",
                   _fix_repetition_sampling),
    RepairStrategy("disable_flash_attention",
                   (sig.INCOHERENT_OUTPUT, sig.EMPTY_RESPONSE), "low",
                   _disable_flash_attention),
    RepairStrategy("increase_gpu_layers", (sig.THROUGHPUT_TOO_LOW,), "low",
                   _increase_gpu_layers),

    # Visible trades.
    RepairStrategy("reduce_parallel", _MEMORY_SIGNATURES, "medium", _reduce_parallel),
    RepairStrategy("reduce_context",
                   (*_MEMORY_SIGNATURES, sig.CONTEXT_EXCEEDED, sig.REQUEST_TIMEOUT),
                   "medium", _reduce_context),
    RepairStrategy("restore_kv_precision", (sig.NEEDLE_NOT_FOUND,), "medium",
                   _restore_kv_precision),

    # Real sacrifices.
    RepairStrategy("reduce_gpu_layers", _MEMORY_SIGNATURES, "high", _reduce_gpu_layers),
    RepairStrategy("drop_quant_tier", _MEMORY_SIGNATURES, "high", _drop_quant_tier),
    RepairStrategy("switch_model",
                   (*_MEMORY_SIGNATURES,
                    sig.INCOHERENT_OUTPUT, sig.TOOLS_UNSUPPORTED,
                    sig.VISION_UNSUPPORTED, sig.EMBEDDINGS_UNSUPPORTED,
                    sig.THROUGHPUT_TOO_LOW, sig.NEEDLE_NOT_FOUND),
                   "high", _switch_model),
]


def strategies_for(signature: str) -> List[RepairStrategy]:
    """Strategies that answer this signature, cheapest first."""
    matching = [s for s in STRATEGIES if s.handles(signature)]
    return sorted(matching, key=lambda s: COST_ORDER.get(s.cost, 9))
