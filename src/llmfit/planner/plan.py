"""Turn (use case + hardware + catalog) into a ranked, explained plan."""

from __future__ import annotations

from typing import List, Optional

from llmfit.catalog.catalog import Catalog, arch_source
from llmfit.planner.memory import (
    KV_QUALITY_ORDER,
    estimate_memory,
    layers_that_fit,
    max_context_for_budget,
)
from llmfit.planner.score import explain, score_candidate
from llmfit.runtimes.registry import choose_runtime
from llmfit.types import (
    GIB,
    Candidate,
    HardwareProfile,
    MemoryEstimate,
    ModelSpec,
    Plan,
    Task,
    UseCaseProfile,
)
from llmfit.util.errors import PlanningError

# Keep more than one candidate per model only when the quants differ meaningfully.
MAX_CANDIDATES = 8


def build_plan(
    profile: UseCaseProfile,
    hardware: HardwareProfile,
    catalog: Catalog,
    runtime: Optional[str] = None,
    allow_partial_offload: bool = True,
) -> Plan:
    """Rank every viable (model, quant) pairing for this user and this machine."""
    chosen_runtime, runtime_reason = (
        (runtime, "explicitly requested") if runtime else choose_runtime(hardware, profile)
    )

    viable = catalog.matching(profile)
    notes: List[str] = []
    if not viable:
        reasons = catalog.unmatched_reasons(profile)
        detail = "; ".join(f"{count} model(s): {reason}" for reason, count in
                           sorted(reasons.items(), key=lambda kv: -kv[1]))
        raise PlanningError(
            "no model in the catalog satisfies your requirements",
            hint=(f"filtered out — {detail}. "
                  "Relax a requirement (shorter context, drop vision) "
                  "or add a model with `llmfit catalog add`."),
        )

    budget = hardware.memory_budget
    cpu_budget = hardware.usable_cpu_bytes
    parallel = _parallel_slots(profile)

    candidates: List[Candidate] = []
    for spec in viable:
        for quant in spec.recommended_quants:
            candidate = _build_candidate(
                spec, quant, profile, hardware, catalog, chosen_runtime,
                budget, cpu_budget, parallel, allow_partial_offload,
            )
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise PlanningError(
            "every candidate model exceeds this machine's memory",
            hint=_too_big_hint(viable, profile, hardware, catalog),
        )

    for candidate in candidates:
        score_candidate(candidate, profile, hardware)

    # Deterministic ordering: score first, then prefer the higher-quality and
    # (at equal quality) smaller model, so equal scores never reorder run to run.
    candidates.sort(
        key=lambda c: (c.score, c.model.quality_tier, -c.model.params_b, c.model.id),
        reverse=True,
    )
    candidates = _dedupe_by_model(candidates)[:MAX_CANDIDATES]

    notes.extend(_plan_notes(candidates, profile, hardware, parallel))

    return Plan(
        candidates=candidates,
        runtime=chosen_runtime,
        runtime_reason=runtime_reason,
        profile=profile,
        hardware=hardware,
        notes=notes,
    )


def _build_candidate(
    spec: ModelSpec,
    quant: str,
    profile: UseCaseProfile,
    hardware: HardwareProfile,
    catalog: Catalog,
    runtime: str,
    budget: int,
    cpu_budget: int,
    parallel: int,
    allow_partial_offload: bool,
) -> Optional[Candidate]:
    """One (model, quant) pairing, with KV quantization applied only if needed."""
    bpw = catalog.bpw(quant)
    warnings: List[str] = []

    # Try KV cache dtypes from best quality down, stopping at the first that fits.
    # This means a KV quantization only ever appears in a plan because it was
    # *required*, never as a default that silently costs quality.
    estimate: Optional[MemoryEstimate] = None
    for kv_quant in KV_QUALITY_ORDER:
        candidate_estimate = estimate_memory(
            spec, bpw, profile.context_tokens, quant=quant, kv_quant=kv_quant,
            runtime=runtime, parallel=parallel,
        )
        if estimate is None:
            estimate = candidate_estimate      # keep the f16 baseline as a fallback
        if candidate_estimate.total <= budget:
            estimate = candidate_estimate
            if kv_quant != "f16":
                cost = ("negligible quality cost" if kv_quant == "q8_0" else
                        "measurably weaker recall over long context — if answers "
                        "start missing details far back in the prompt, this is why")
                warnings.append(
                    f"KV cache quantized to {kv_quant} to fit "
                    f"{profile.context_tokens:,} tokens ({cost})"
                )
            break
    assert estimate is not None

    fits_on_gpu = bool(hardware.gpus) and estimate.total <= budget
    gpu_fraction = 1.0

    if not fits_on_gpu:
        if not hardware.gpus:
            # CPU-only: system RAM is the budget, and everything is "offloaded" to it.
            if estimate.total > cpu_budget:
                return None
            gpu_fraction = 0.0
        elif allow_partial_offload:
            fitted_layers = layers_that_fit(
                spec, bpw, budget, profile.context_tokens,
                kv_quant=estimate.kv_quant, runtime=runtime, parallel=parallel,
            )
            gpu_fraction = fitted_layers / spec.layers if spec.layers else 0.0
            if gpu_fraction < 0.10:
                return None        # so little on the GPU it isn't worth the complexity
            if estimate.total > cpu_budget + budget:
                return None        # doesn't fit even across both tiers
            warnings.append(
                f"only {fitted_layers}/{spec.layers} layers fit in VRAM; "
                "the rest run on CPU and dominate latency"
            )
        else:
            return None

    utilisation = estimate.total / budget if budget else 1.0
    if 0.85 <= utilisation <= 1.0:
        where = "system RAM" if not hardware.gpus else "VRAM"
        warnings.append(
            f"uses {utilisation:.0%} of usable {where}; a long prompt or another "
            "application could push this into swap or an out-of-memory error"
        )

    if profile.max_disk_gb and estimate.weights / GIB > profile.max_disk_gb:
        return None
    if hardware.disk_free_bytes and estimate.weights > hardware.disk_free_bytes * 0.9:
        warnings.append(
            f"download needs ~{estimate.weights / GIB:.1f} GB but only "
            f"{hardware.disk_free_bytes / GIB:.1f} GB is free"
        )

    if arch_source(spec.id) == "estimated":
        warnings.append(
            "architecture figures are estimated; memory numbers will be "
            "recomputed from the real file after download"
        )

    return Candidate(
        model=spec,
        quant=quant,
        estimate=estimate,
        fits_on_gpu=fits_on_gpu,
        gpu_fraction=gpu_fraction,
        warnings=warnings,
    )


def _dedupe_by_model(candidates: List[Candidate]) -> List[Candidate]:
    """Keep the best-scoring quant per model, then one alternate quant at most.

    Without this the top of the list is the same model at five quant levels,
    which is useless as a fallback chain — the repair loop needs *different
    models* to fall back to, not the same one slightly smaller.
    """
    best_per_model: List[Candidate] = []
    seen = set()
    for candidate in candidates:
        if candidate.model.id in seen:
            continue
        seen.add(candidate.model.id)
        best_per_model.append(candidate)
    return best_per_model


def _parallel_slots(profile: UseCaseProfile) -> int:
    """KV cache slots to budget for. Serving several users multiplies the cache."""
    return max(1, min(profile.concurrent_users, 16))


def _plan_notes(
    candidates: List[Candidate],
    profile: UseCaseProfile,
    hardware: HardwareProfile,
    parallel: int,
) -> List[str]:
    notes: List[str] = []
    best = candidates[0]

    notes.append(
        f"{best.model.name} at {best.quant}: {explain(best, profile)}."
    )
    if parallel > 1:
        notes.append(
            f"budgeting {parallel} concurrent slots, so the KV cache is sized "
            f"for {profile.context_tokens * parallel:,} tokens in total"
        )
    if best.model.is_moe:
        notes.append(
            f"{best.model.name} is a mixture-of-experts model: all "
            f"{best.model.params_b:g}B parameters must be resident, but only "
            f"{best.model.compute_params_b:g}B run per token — it is much faster "
            "than a dense model of the same size"
        )
    if best.model.is_reasoning:
        notes.append(
            "this is a reasoning model and emits long thinking blocks; expect "
            "higher latency to first useful token and budget output tokens accordingly"
        )
    if not hardware.gpus:
        notes.append("no GPU available, so throughput is bounded by system RAM bandwidth")
    if profile.needs_embeddings:
        notes.append("a separate embedding model is needed for RAG; see the embedding plan")
    return notes


def _too_big_hint(
    viable: List[ModelSpec], profile: UseCaseProfile, hardware: HardwareProfile,
    catalog: Catalog,
) -> str:
    """When nothing fits, say what *would* — a number the user can act on."""
    budget = hardware.memory_budget
    smallest = min(viable, key=lambda s: s.params_b)
    quant = smallest.recommended_quants[-1] if smallest.recommended_quants else "Q4_K_M"
    reachable = max_context_for_budget(
        smallest, catalog.bpw(quant), budget, kv_quant="q8_0",
    )
    if reachable > 0:
        return (
            f"the smallest match ({smallest.name}) fits at about "
            f"{reachable:,} tokens of context, not {profile.context_tokens:,}. "
            f"Retry with --context {reachable}."
        )
    return (
        f"even {smallest.name} needs more than the {budget / GIB:.1f} GB available. "
        "Add a smaller model to the catalog, or run with --allow-cpu."
    )


def plan_embedding_model(
    profile: UseCaseProfile, hardware: HardwareProfile, catalog: Catalog
) -> Optional[ModelSpec]:
    """Pick a companion embedding model for RAG.

    Multilingual corpora need a multilingual retriever; an English-only model
    will quietly return poor neighbours rather than fail loudly.
    """
    if not (profile.needs_embeddings or Task.RAG in profile.tasks):
        return None

    embedders = catalog.embedding_models()
    if not embedders:
        return None

    non_english = [lang for lang in profile.languages if lang != "en"]

    def rank(spec: ModelSpec) -> tuple:
        multilingual_fit = 1 if (not non_english or
                                 all(lang in spec.languages for lang in non_english)) else 0
        return (multilingual_fit, spec.quality_tier, -spec.params_b)

    return sorted(embedders, key=rank, reverse=True)[0]
