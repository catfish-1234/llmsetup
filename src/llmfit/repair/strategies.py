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
from llmfit.planner.memory import (
    KV_QUALITY_ORDER,
    cheaper_kv_quant,
    kv_is_lossy,
    kv_lossy_allowed,
    max_context_for_budget,
)
from llmfit.types import (
    Candidate,
    HardwareProfile,
    RuntimeConfig,
    Task,
    UseCaseProfile,
)
from llmfit.util.log import Console
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
    #: Strategies that touch the network or the disk report through this. It is
    #: optional so the pure ones stay constructible in a test without any I/O.
    console: Optional[Console] = None
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
    #: Whether the loop may run this more than once for the same signature.
    #:
    #: Normally it may not, and that is what makes termination structural. A
    #: strategy setting this takes on the obligation itself: it must consume
    #: something finite on every run and return None once that is gone, so the
    #: sequence still ends. Only set it where the strategy walks a fixed list
    #: it records progress through, never where it recomputes a value that
    #: could oscillate.
    exhausts_itself: bool = False

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
    """Take the KV cache down to the cheapest dtype that costs nothing.

    Stops at the safe floor. Going below it is a real quality decision, not a
    free saving, so it lives in its own strategy that the loop only reaches
    after everything cheaper has been tried.
    """
    config = state.config
    nxt = cheaper_kv_quant(config.kv_cache_type)
    if nxt is None:
        return None

    before = {"kv_cache_type": config.kv_cache_type.lower()}
    config.kv_cache_type = nxt
    # Quantized KV requires flash attention on llama.cpp-family backends.
    config.flash_attention = True
    return Outcome(
        f"quantized the KV cache to {nxt} to free memory "
        f"(measures as effectively lossless against f16)",
        cost="low", before=before, after={"kv_cache_type": nxt},
    )


def _quantize_kv_lossy(state: RepairState) -> Optional[Outcome]:
    """Last-resort KV quantization below the safe floor.

    Below q8_0 the saving stops being free: how much quality goes with it
    depends on the model, and at long context the cache is slower as well as
    worse, because dequantizing inside attention costs more than the memory
    traffic it saves. So this only applies where it can still help -- short
    enough context to be a genuine trade -- and it says plainly what it cost.
    """
    config = state.config
    if not kv_lossy_allowed(config.context_tokens):
        return None
    nxt = cheaper_kv_quant(config.kv_cache_type, allow_lossy=True)
    if nxt is None or not kv_is_lossy(nxt):
        return None

    before = {"kv_cache_type": config.kv_cache_type.lower()}
    config.kv_cache_type = nxt
    config.flash_attention = True
    return Outcome(
        f"took the KV cache down to {nxt}. This is past the point of being "
        f"free: expect some answers to change, and raise the cache back to "
        f"q8_0 with a shorter context if quality matters more than length",
        cost="high", before=before, after={"kv_cache_type": nxt},
    )


def _increase_batch(state: RepairState) -> Optional[Outcome]:
    """Give prompt processing a bigger micro-batch.

    Prefill is compute-bound and batched, so the physical micro-batch is the
    single flag that moves it most: on the same hardware, prompt throughput can
    differ by an order of magnitude between a small ubatch and a large one. The
    cost is compute-buffer memory, which is why this is only reached after
    prefill has actually been measured and found wanting -- not guessed at.
    """
    config = state.config
    if config.ubatch_size >= _MAX_UBATCH:
        return None
    before = {"batch_size": config.batch_size, "ubatch_size": config.ubatch_size}
    config.ubatch_size = min(_MAX_UBATCH, config.ubatch_size * 2)
    config.batch_size = max(config.batch_size, config.ubatch_size)
    return Outcome(
        f"raised the prompt micro-batch to {config.ubatch_size} "
        f"(batch {config.batch_size}) so long prompts process in fewer, larger "
        f"passes; this costs some compute-buffer memory",
        cost="low", before=before,
        after={"batch_size": config.batch_size, "ubatch_size": config.ubatch_size},
    )


#: Past this the compute buffer grows faster than prefill improves, and on
#: smaller cards it is a reliable way to turn a working setup into an OOM.
_MAX_UBATCH = 2048


def _reduce_context(state: RepairState) -> Optional[Outcome]:
    """Shrink context to what genuinely fits, not by a blind halving."""
    config = state.config
    spec = state.candidate.model
    bpw = (state.candidate.estimate.weights * 8) / (spec.params_b * 1e9)

    fitted = max_context_for_budget(
        spec, bpw, state.budget, kv_quant=config.kv_cache_type,
        runtime=config.runtime, ubatch=config.ubatch_size, parallel=config.parallel,
        swa_full=config.swa_full,
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


def _full_swa_cache(state: RepairState) -> Optional[Outcome]:
    """Stop windowing the cache on a model whose recall just failed.

    Sliding-window layers only keep the last N tokens, which is where most of a
    model like this saves its memory. The saving is real, but it makes some
    cache operations impossible, and the symptom is exactly what the needle
    check measures: everything works except finding something stated earlier in
    a long prompt.

    Tried before giving up context, because keeping the whole prompt visible is
    usually worth more than keeping it long.
    """
    config, spec = state.config, state.candidate.model
    if not spec.uses_swa or config.swa_full:
        return None
    if config.runtime != "llamacpp":
        return None

    from llmfit.planner.memory import estimate_memory

    bpw = (state.candidate.estimate.weights * 8) / (spec.params_b * 1e9)
    cost = estimate_memory(spec, bpw, config.context_tokens,
                           kv_quant=config.kv_cache_type, runtime=config.runtime,
                           ubatch=config.ubatch_size, parallel=config.parallel,
                           swa_full=True)
    if cost.total > state.budget:
        # Honest refusal rather than a change that will OOM on restart. The
        # context reduction that follows is the workable version of this fix.
        return None

    before = {"swa_full": False}
    config.swa_full = True
    return Outcome(
        f"kept a full-size attention cache instead of a sliding window, so the "
        f"model can see the whole prompt rather than the last "
        f"{spec.sliding_window:,} tokens on most of its layers. This costs "
        f"{(cost.kv_cache - state.candidate.estimate.kv_cache) / (1024 ** 3):.1f} "
        f"GB more",
        cost="medium", before=before, after={"swa_full": True},
    )


def _restore_kv_precision(state: RepairState) -> Optional[Outcome]:
    """Long-context recall failed, undo the KV quantization that caused it.

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
        swa_full=config.swa_full,
    )
    target = min(config.context_tokens, (fitted // 1024) * 1024)
    if target < _context_floor(state.profile):
        return None

    before = {"kv_cache_type": current, "context_tokens": config.context_tokens}
    config.kv_cache_type = better
    config.context_tokens = target
    return Outcome(
        f"restored KV cache precision to {better} (context now {target:,} tokens): "
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


def _try_tool_parser(state: RepairState) -> Optional[Outcome]:
    """Try the next tool-call parser this family might be using.

    A model emitting a perfectly good tool call still comes back as prose when
    the server is looking for another shape, and the shapes are not
    interchangeable between families or even between versions of one. Which
    parser a given build wants is not something to be reasoned about from the
    outside, so try the plausible ones and let the check say which works.
    """
    config, spec = state.config, state.candidate.model
    if config.runtime != "vllm":
        return None

    from llmfit.tuning.tuner import tool_parsers_for

    candidates = tool_parsers_for(spec)
    if not candidates:
        return None
    tried = config.extra.setdefault("tool_parsers_tried", [])
    current = config.extra.get("tool_parser")
    if current and current not in tried:
        tried.append(current)
    remaining = [p for p in candidates if p not in tried]
    if not remaining:
        return None

    nxt = remaining[0]
    before = {"tool_parser": current}
    config.extra["tool_parser"] = nxt
    tried.append(nxt)
    return Outcome(
        f"switched the tool-call parser to {nxt}, because this family emits "
        f"calls in more than one shape and the server has to be told which",
        cost="low", before=before, after={"tool_parser": nxt},
    )


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


def _disable_speculation(state: RepairState) -> Optional[Outcome]:
    """Drop the draft model when it turns out to be incompatible.

    Speculative decoding is an optimisation. A vocabulary mismatch between draft
    and target is fatal to it but irrelevant to the model itself, so the right
    move is to lose the speedup and keep the setup.
    """
    advanced = getattr(state.config, "advanced", None)
    plan = getattr(advanced, "speculative", None)
    if plan is None or not plan.enabled:
        return None
    before = {"draft_model": plan.draft_model_id}
    plan.draft_model_id = None
    plan.draft_path = None
    plan.rationale = "disabled: the draft model was not compatible with this target"
    return Outcome(
        "dropped the speculative draft model; it was not compatible with the "
        "target's vocabulary, and speculation is an optimisation rather than a "
        "requirement",
        cost="low", before=before, after={"draft_model": None},
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


def _fetch_vision_projector(state: RepairState) -> Optional[Outcome]:
    """Fetch the multimodal projector a vision model is missing.

    This is the one repair that adds a file rather than giving something up: a
    vision model served without its projector answers image questions from the
    text alone, confidently and wrongly. Nothing about the settings can fix
    that -- the model is simply only half loaded.
    """
    config, spec = state.config, state.candidate.model
    if not spec.supports_vision or config.mmproj_path:
        return None
    if config.runtime != "llamacpp":
        # Other runtimes bundle the projector with the model; a loose file has
        # nowhere to go, so pretending otherwise would waste a repair round.
        return None

    from llmfit.runtimes.gguf_store import fetch_projector

    console = state.console or Console(quiet=True)
    try:
        path = fetch_projector(spec, config.quant, console)
    except Exception as exc:  # a failed download is a failed repair, not a crash
        console.warn(f"could not fetch the vision projector: {exc}")
        return None
    if path is None:
        return None

    before = {"mmproj_path": None}
    config.mmproj_path = str(path)
    return Outcome(
        description=f"loaded the vision projector ({path.name}) so images work",
        cost="low", before=before, after={"mmproj_path": str(path)},
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
    RepairStrategy("disable_speculation",
                   (sig.DRAFT_INCOMPATIBLE, sig.CRASH_DURING_LOAD), "low",
                   _disable_speculation),
    RepairStrategy("restart_server",
                   (sig.SERVER_UNREACHABLE, sig.CRASH_DURING_LOAD, sig.MODEL_NOT_LOADED),
                   "low", _restart_server),
    RepairStrategy("enable_chat_template",
                   (sig.TOOLS_UNSUPPORTED, sig.TOOL_CALL_MALFORMED), "low",
                   _enable_chat_template),
    # Repeatable: it walks a recorded list of parsers and returns None once
    # every one has been tried, so the sequence is finite either way.
    RepairStrategy("try_tool_parser",
                   (sig.TOOLS_UNSUPPORTED, sig.TOOL_CALL_MALFORMED), "low",
                   _try_tool_parser, exhausts_itself=True),
    RepairStrategy("enable_embeddings_flag", (sig.EMBEDDINGS_UNSUPPORTED,), "low",
                   _enable_embeddings_flag),
    RepairStrategy("fetch_vision_projector", (sig.VISION_UNSUPPORTED,), "low",
                   _fetch_vision_projector),
    RepairStrategy("fix_repetition_sampling", (sig.REPETITION_LOOP,), "low",
                   _fix_repetition_sampling),
    RepairStrategy("disable_flash_attention",
                   (sig.INCOHERENT_OUTPUT, sig.EMPTY_RESPONSE), "low",
                   _disable_flash_attention),
    RepairStrategy("increase_gpu_layers",
                   (sig.THROUGHPUT_TOO_LOW, sig.PREFILL_TOO_SLOW), "low",
                   _increase_gpu_layers),
    RepairStrategy("increase_batch", (sig.PREFILL_TOO_SLOW,), "low",
                   _increase_batch),

    # Visible trades.
    RepairStrategy("reduce_parallel", _MEMORY_SIGNATURES, "medium", _reduce_parallel),
    RepairStrategy("reduce_context",
                   (*_MEMORY_SIGNATURES, sig.CONTEXT_EXCEEDED, sig.REQUEST_TIMEOUT),
                   "medium", _reduce_context),
    RepairStrategy("full_swa_cache", (sig.NEEDLE_NOT_FOUND,), "medium",
                   _full_swa_cache),
    RepairStrategy("restore_kv_precision", (sig.NEEDLE_NOT_FOUND,), "medium",
                   _restore_kv_precision),

    # Real sacrifices.
    RepairStrategy("quantize_kv_cache_lossy",
                   (*_MEMORY_SIGNATURES, sig.CONTEXT_EXCEEDED), "high",
                   _quantize_kv_lossy),
    RepairStrategy("reduce_gpu_layers", _MEMORY_SIGNATURES, "high", _reduce_gpu_layers),
    RepairStrategy("drop_quant_tier", _MEMORY_SIGNATURES, "high", _drop_quant_tier),
    RepairStrategy("switch_model",
                   (*_MEMORY_SIGNATURES,
                    sig.INCOHERENT_OUTPUT, sig.TOOLS_UNSUPPORTED,
                    sig.VISION_UNSUPPORTED, sig.EMBEDDINGS_UNSUPPORTED,
                    sig.THROUGHPUT_TOO_LOW, sig.PREFILL_TOO_SLOW,
                    sig.NEEDLE_NOT_FOUND),
                   "high", _switch_model),
]


def strategies_for(signature: str) -> List[RepairStrategy]:
    """Strategies that answer this signature, cheapest first."""
    matching = [s for s in STRATEGIES if s.handles(signature)]
    return sorted(matching, key=lambda s: COST_ORDER.get(s.cost, 9))
