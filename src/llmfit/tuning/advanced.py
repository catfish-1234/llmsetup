"""Deep tuning: the settings past load and basic sampling.

Each planner here answers one question, takes the machine and the use case into
account, and records why. The ordering in :func:`plan_advanced` is deliberate —
offload decides how much memory is left, which is what speculative decoding then
has to fit a draft model into.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from llmfit.catalog.catalog import Catalog
from llmfit.planner import moe
from llmfit.tuning.profiles import (
    AdvancedProfile,
    OffloadPlan,
    ReasoningEffort,
    ReasoningPlan,
    RopePlan,
    SamplerChain,
    ServingPlan,
    SpeculativePlan,
)
from llmfit.types import GIB, Candidate, HardwareProfile, ModelSpec, Task, UseCaseProfile

#: Tasks whose output is predictable enough for a draft model to be worth it.
#: Speculative decoding is a bet on guessability: code and structured text pay,
#: open-ended prose does not.
_SPECULATIVE_TASKS = {Task.CODING, Task.AGENT, Task.EXTRACT, Task.TRANSLATE,
                      Task.SUMMARIZE, Task.RAG}

#: Smallest size ratio worth drafting with. A draft model that is not much
#: cheaper than the target cannot win back the cost of verifying its guesses.
_MIN_DRAFT_RATIO = 5.0


def plan_advanced(
    candidate: Candidate,
    hardware: HardwareProfile,
    profile: UseCaseProfile,
    catalog: Catalog,
    context_tokens: int,
    gpu_layers: int,
    bpw: float,
    runtime: str = "llamacpp",
) -> AdvancedProfile:
    """Derive every deep setting for this model on this machine."""
    spec = candidate.model
    offload = plan_offload(candidate, hardware, context_tokens, gpu_layers, bpw)
    return AdvancedProfile(
        offload=offload,
        speculative=plan_speculative(candidate, hardware, profile, catalog,
                                     offload, context_tokens, bpw),
        reasoning=plan_reasoning(spec, profile),
        rope=plan_rope(spec, context_tokens),
        sampler=plan_sampler(spec, profile),
        serving=plan_serving(profile, runtime, hardware),
    )


# --------------------------------------------------------------------------- #
# offload
# --------------------------------------------------------------------------- #


def plan_offload(
    candidate: Candidate,
    hardware: HardwareProfile,
    context_tokens: int,
    gpu_layers: int,
    bpw: float,
) -> OffloadPlan:
    """Decide where tensors live, including MoE expert offload.

    For a mixture-of-experts model that does not fit, moving expert weights to
    RAM is tried *before* dropping layers: the experts are most of the bytes but
    only a fraction of the per-token work, so the same memory saving costs far
    less speed than partial layer offload would.
    """
    spec = candidate.model
    plan = OffloadPlan(gpu_layers=gpu_layers, total_layers=spec.layers,
                       main_gpu=0, split_mode="layer")

    if len(hardware.gpus) > 1:
        plan.tensor_split = _tensor_split(hardware)
        # The largest card holds the KV cache and the non-split tensors.
        plan.main_gpu = max(range(len(hardware.gpus)),
                            key=lambda i: hardware.gpus[i].vram_bytes)
        plan.rationale = (
            f"split across {len(hardware.gpus)} GPUs in proportion to their VRAM; "
            f"GPU {plan.main_gpu} is largest and holds the KV cache"
        )

    if not spec.is_moe or not hardware.gpus:
        if not plan.rationale:
            plan.rationale = (
                f"all {gpu_layers} layers on the GPU" if plan.full_gpu
                else f"{gpu_layers} of {spec.layers} layers on the GPU"
            )
        return plan

    budget = hardware.usable_accel_bytes
    weights = int(spec.params_b * 1e9 * bpw / 8)
    from llmfit.planner.memory import kv_bytes_per_token, overhead_bytes

    non_weight = int(kv_bytes_per_token(spec) * context_tokens) + overhead_bytes(spec)
    excess = (weights + non_weight) - budget

    if excess <= 0:
        plan.rationale = plan.rationale or "the whole model fits on the GPU"
        return plan

    layers = moe.layers_to_offload(spec, bpw, excess)
    if layers <= 0:
        return plan

    plan.n_cpu_moe = layers
    plan.gpu_layers = spec.layers        # every layer stays; only experts move
    retained = moe.throughput_penalty(spec, layers)
    resident = moe.resident_bytes_with_offload(spec, bpw, layers)
    plan.rationale = (
        f"routed experts for {layers} of {spec.layers} layers moved to system RAM, "
        f"leaving {resident / GIB:.1f} GB of weights on the GPU. Attention and the "
        f"KV cache stay resident, and only {spec.compute_params_b:g}B parameters "
        f"are active per token, so this costs far less speed than moving whole "
        f"layers would — expect roughly {retained:.0%} of full-GPU throughput"
    )
    return plan


def _tensor_split(hardware: HardwareProfile) -> List[float]:
    total = sum(g.vram_bytes for g in hardware.gpus)
    if total <= 0:
        return []
    return [round(g.vram_bytes / total, 3) for g in hardware.gpus]


# --------------------------------------------------------------------------- #
# speculative decoding
# --------------------------------------------------------------------------- #


def plan_speculative(
    candidate: Candidate,
    hardware: HardwareProfile,
    profile: UseCaseProfile,
    catalog: Catalog,
    offload: OffloadPlan,
    context_tokens: int,
    bpw: float,
) -> SpeculativePlan:
    """Pair a small draft model with the target, when it is likely to pay.

    Three conditions, all required: the task has predictable output, a draft
    model sharing the target's vocabulary exists, and there is memory left over
    for it. Skipping any one of these is why the plan usually stays off.
    """
    spec = candidate.model
    plan = SpeculativePlan()

    if not any(t in _SPECULATIVE_TASKS for t in profile.tasks):
        plan.rationale = (
            "off: speculative decoding is a bet on predictable output, and this "
            "use case is open-ended enough that the verification cost would "
            "likely outweigh the gain"
        )
        return plan

    if spec.is_moe and spec.compute_params_b < 6:
        plan.rationale = (
            f"off: {spec.name} already activates only {spec.compute_params_b:g}B "
            f"parameters per token, so it is close to draft speed already and "
            f"there is little left to win"
        )
        return plan

    draft, ratio = _pick_draft(spec, catalog)
    if draft is None:
        plan.rationale = (
            f"off: no model in the catalog shares {spec.name}'s vocabulary and is "
            f"small enough to draft for it — a draft model must tokenize "
            f"identically or its guesses cannot be verified"
        )
        return plan

    quant = draft.recommended_quants[0] if draft.recommended_quants else "Q4_K_M"
    draft_bytes = int(draft.params_b * 1e9 * catalog.bpw(quant) / 8)
    from llmfit.planner.memory import kv_bytes_per_token

    draft_bytes += int(kv_bytes_per_token(draft) * min(context_tokens, 8192))

    headroom = hardware.usable_accel_bytes - candidate.estimate.total
    if hardware.gpus and draft_bytes > headroom:
        plan.rationale = (
            f"off: {draft.name} would draft well for this model but needs "
            f"{draft_bytes / GIB:.1f} GB and only {max(0, headroom) / GIB:.1f} GB "
            f"is spare — the memory is better spent on context"
        )
        return plan

    plan.draft_model_id = draft.id
    plan.draft_quant = quant
    plan.draft_gpu_layers = draft.layers if hardware.gpus else 0
    plan.draft_memory_bytes = draft_bytes
    plan.max_draft_tokens = 16 if Task.CODING in profile.tasks else 8
    plan.p_min = 0.75
    # Acceptance rates on predictable text run high; the win is bounded by how
    # much cheaper the draft is, so a conservative multiple is quoted.
    plan.expected_speedup = round(min(2.0, 1.0 + 0.55 * min(1.0, ratio / 20)), 2)
    plan.rationale = (
        f"{draft.name} drafts for {spec.name} — same vocabulary, {ratio:.0f}x "
        f"smaller, {draft_bytes / GIB:.1f} GB. It guesses "
        f"{plan.max_draft_tokens} tokens ahead and the target verifies them in "
        f"one pass, which is worth roughly {plan.expected_speedup:.2f}x on "
        f"predictable output and costs nothing when guesses miss"
    )
    return plan


def _pick_draft(spec: ModelSpec, catalog: Catalog) -> Tuple[Optional[ModelSpec], float]:
    """Smallest catalog model that shares the target's vocabulary.

    Smallest, not merely small: the draft only has to be *plausible*, and every
    parameter it carries is memory taken from the target's context.
    """
    best: Optional[ModelSpec] = None
    best_ratio = 0.0
    for other in catalog.chat_models():
        if other.id == spec.id or not other.draft_capable:
            continue
        if other.family != spec.family:
            continue
        ratio = spec.compute_params_b / other.params_b if other.params_b else 0
        if ratio < _MIN_DRAFT_RATIO:
            continue
        if best is None or other.params_b < best.params_b:
            best, best_ratio = other, ratio
    return best, best_ratio


# --------------------------------------------------------------------------- #
# reasoning
# --------------------------------------------------------------------------- #


def plan_reasoning(spec: ModelSpec, profile: UseCaseProfile) -> ReasoningPlan:
    """How hard a hybrid-reasoning model should think.

    The largest latency lever there is: suppressing the thinking block takes
    time-to-first-token from seconds to a fraction of one. Worth it for
    extraction and translation, where the answer is a transformation rather than
    a deduction; not worth it for agents, where the reasoning is the point.
    """
    plan = ReasoningPlan(supported=spec.is_reasoning)
    if not spec.is_reasoning:
        plan.effort = ReasoningEffort.OFF
        plan.rationale = f"{spec.name} has no separate thinking mode"
        return plan

    task = profile.primary_task
    if profile.needs_reasoning:
        plan.effort = ReasoningEffort.HIGH
        plan.rationale = "you asked for a reasoning model, so thinking is left on high"
    elif task in (Task.EXTRACT, Task.TRANSLATE):
        plan.effort = ReasoningEffort.OFF
        plan.rationale = (
            "thinking disabled: this task is a transformation, not a deduction, "
            "and suppressing the thinking block takes time-to-first-token from "
            "seconds to a fraction of one"
        )
    elif task in (Task.CHAT, Task.SUMMARIZE, Task.CREATIVE):
        plan.effort = ReasoningEffort.LOW
        plan.rationale = (
            "thinking kept low: it helps a little here and costs latency on every "
            "turn, which is felt in conversation"
        )
    elif task in (Task.CODING, Task.AGENT):
        plan.effort = ReasoningEffort.MEDIUM
        plan.rationale = (
            "medium thinking: enough to plan a change or a tool call, without the "
            "full deliberation budget on every request"
        )
    else:
        plan.effort = ReasoningEffort.MEDIUM
        plan.rationale = "balanced thinking budget"

    # Agents parse structured output; a thinking block in the response confuses
    # tool-call extraction, so it is stripped unless a human is reading it.
    plan.preserve = task not in (Task.AGENT, Task.EXTRACT)
    return plan


# --------------------------------------------------------------------------- #
# RoPE / context extension
# --------------------------------------------------------------------------- #


def plan_rope(spec: ModelSpec, context_tokens: int) -> RopePlan:
    """Extend context past the trained window, only when asked to."""
    plan = RopePlan()
    if context_tokens <= spec.max_context:
        plan.rationale = (
            f"native context is {spec.max_context:,} tokens, which covers the "
            f"{context_tokens:,} requested — no scaling needed"
        )
        return plan

    factor = context_tokens / spec.max_context
    plan.scaling_type = "yarn"
    plan.factor = round(factor, 3)
    plan.original_context = spec.max_context
    plan.yarn_ext_factor = 1.0
    plan.rationale = (
        f"YaRN scaling by {factor:.2f}x to reach {context_tokens:,} tokens from a "
        f"{spec.max_context:,}-token model. This works, but quality degrades with "
        f"the scale factor — if answers get vaguer at long context, this is why, "
        f"and a model with a natively longer window is the better fix"
    )
    return plan


# --------------------------------------------------------------------------- #
# samplers
# --------------------------------------------------------------------------- #


def plan_sampler(spec: ModelSpec, profile: UseCaseProfile) -> SamplerChain:
    """The full decoding stack, chosen per task and model family."""
    task = profile.primary_task
    chain = SamplerChain()
    notes: List[str] = []

    if task in (Task.CODING, Task.AGENT, Task.EXTRACT):
        # Deterministic work: no truncation games, and repetition in code is
        # usually legitimate (boilerplate, repeated identifiers), so penalties
        # stay off rather than mangling valid output.
        chain.dry_multiplier = 0.0
        chain.xtc_probability = 0.0
        chain.top_n_sigma = -1.0
        chain.penalty_last_n = 0
        notes.append(
            "no repetition penalties: repeated tokens in code and structured "
            "output are usually correct, and penalising them corrupts valid text"
        )
    elif task in (Task.CREATIVE,):
        # Variety is the goal. DRY stops phrase loops without banning necessary
        # words the way a flat repeat penalty does; XTC removes the most obvious
        # continuation to break clichés.
        chain.dry_multiplier = 0.8
        chain.dry_base = 1.75
        chain.dry_allowed_length = 2
        chain.dry_penalty_last_n = -1
        chain.xtc_probability = 0.5
        chain.xtc_threshold = 0.1
        notes.append(
            "DRY penalises repeated phrases rather than repeated tokens, and XTC "
            "drops the most predictable continuation — together they break loops "
            "and clichés without banning words the passage needs"
        )
    else:
        # General use: DRY at low strength, nothing exotic.
        chain.dry_multiplier = 0.6
        chain.dry_allowed_length = 3
        notes.append("mild DRY to stop phrase loops without constraining word choice")

    if spec.is_reasoning:
        # Reasoning models degenerate under aggressive truncation mid-thought.
        chain.top_n_sigma = 1.0
        chain.xtc_probability = 0.0
        notes.append(
            "top-n-sigma instead of hard truncation, and XTC off: reasoning models "
            "collapse mid-thought when the tail is cut too aggressively"
        )

    if profile.needs_json:
        chain.dry_multiplier = 0.0
        chain.xtc_probability = 0.0
        notes.append("all creative samplers off so JSON structure stays intact")

    chain.rationale = "; ".join(notes)
    return chain


# --------------------------------------------------------------------------- #
# serving
# --------------------------------------------------------------------------- #


def plan_serving(profile: UseCaseProfile, runtime: str,
                 hardware: HardwareProfile) -> ServingPlan:
    """Batching and caching behaviour for the endpoint."""
    plan = ServingPlan()
    notes: List[str] = []

    plan.continuous_batching = profile.concurrent_users > 1 or runtime == "vllm"
    if plan.continuous_batching:
        notes.append("continuous batching so concurrent requests interleave")

    # Prefix caching is the single biggest win for agents and RAG, where a long
    # system prompt or document is resent on every turn.
    if profile.primary_task in (Task.AGENT, Task.RAG, Task.CODING):
        plan.prefix_caching = True
        plan.cache_reuse_tokens = 512
        notes.append(
            "prefix caching with generous reuse: agent and retrieval workloads "
            "resend a long prefix every turn, and caching it removes most of the "
            "prompt-processing cost"
        )

    if runtime == "vllm":
        plan.chunked_prefill = True
        # Lower budgets interleave more and keep decode latency fair under load;
        # higher ones favour raw throughput.
        plan.max_batched_tokens = 2048 if profile.concurrent_users >= 4 else 8192
        notes.append(
            f"chunked prefill with a {plan.max_batched_tokens}-token budget, "
            f"{'favouring fair latency under concurrency' if profile.concurrent_users >= 4 else 'favouring throughput'}"
        )

    plan.rationale = "; ".join(notes)
    return plan
