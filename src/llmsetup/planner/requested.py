"""Plan around a retained model, including its available weight formats."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional, Tuple

from llmsetup.catalog.catalog import Catalog
from llmsetup.planner.memory import KV_QUALITY_ORDER, kv_is_lossy, kv_lossy_allowed, kv_options
from llmsetup.planner.score import score_candidate
from llmsetup.runtimes.registry import choose_runtime, recovery_order
from llmsetup.types import (
    Accelerator,
    Candidate,
    HardwareProfile,
    ModelSpec,
    Plan,
    Task,
    UseCaseProfile,
)
from llmsetup.util.errors import PlanningError

_GGUF_RUNTIMES = {"llamacpp", "ollama", "lmstudio"}


def build_requested_plan(
    profile: UseCaseProfile,
    hardware: HardwareProfile,
    catalog: Catalog,
    runtime: Optional[str] = None,
    allow_partial_offload: bool = True,
    _include_fallbacks: bool = True,
    runtime_reason: Optional[str] = None,
) -> Plan:
    """Keep the requested model first while tuning runtime, weights and placement.

    Other models remain available to the repair loop after its same-model budget
    is exhausted. They never compete with the requested model for first place.
    """
    from llmsetup.planner.model_fit import resolve_requested_model
    from llmsetup.planner.plan import (
        MAX_CANDIDATES,
        _build_candidate,
        _parallel_slots,
        _plan_notes,
        _too_big_hint,
    )

    spec = resolve_requested_model(profile.requested_model or "", catalog)
    profile = replace(profile, requested_model=spec.id)
    if Task.AUTOCOMPLETE in profile.tasks and not spec.supports_fim:
        raise PlanningError(
            f"{spec.name} is not cataloged for fill-in-the-middle completion",
            hint="Choose a model with fill-in-the-middle support for inline completion.",
        )
    # Task suitability is a soft tradeoff for an explicit choice. Required
    # capabilities, licence and context limits are still enforced.
    if spec.is_embedding or not catalog.satisfies(replace(spec, tasks=[]), profile):
        reasons = catalog.restrict_to(spec.id).unmatched_reasons(profile)
        detail = "; ".join(reasons) or "this is an embedding model, not a chat model"
        raise PlanningError(
            f"{spec.name} cannot meet the required capabilities or context: {detail}",
            hint="Choose a compatible model or change the stated requirements.",
        )

    options = _runtime_options(spec, profile, hardware, runtime, explicit_reason=runtime_reason)
    ranked: List[Tuple[Candidate, str, str]] = []
    for engine, reason in options:
        engine_budget = (int(hardware.memory_budget * 0.92)
                         if engine == "vllm" else hardware.memory_budget)
        quants = _runtime_quants(spec, engine)
        kv_quants = kv_options(engine, bool(profile.context_floor))
        if engine == "vllm":
            kv_quants = ["f16", "q8_0"]
        for quant in quants:
            for kv_quant in kv_quants:
                if kv_is_lossy(kv_quant) and not kv_lossy_allowed(profile.context_tokens):
                    continue
                candidate = _build_candidate(
                    spec, quant, profile, hardware, catalog, engine,
                    engine_budget, hardware.usable_cpu_bytes,
                    _parallel_slots(profile),
                    allow_partial_offload=(allow_partial_offload
                                           and engine in _GGUF_RUNTIMES
                                           and not hardware.unified_memory),
                    kv_quant_override=kv_quant,
                    allow_cpu_fallback=engine in _GGUF_RUNTIMES,
                    bpw_override=_runtime_bpw(quant, catalog),
                )
                if candidate is not None:
                    score_candidate(candidate, profile, hardware)
                    ranked.append((candidate, engine, reason))

    if not ranked:
        raise PlanningError(
            f"{spec.name} does not fit the available memory with a compatible runtime",
            hint=_too_big_hint([spec], profile, hardware, catalog),
        )

    # Preserve cache precision when the model can already fit without a lossy
    # cache. Otherwise a low-quality cache can win merely by creating headroom.
    comfortable = [row for row in ranked if not kv_is_lossy(row[0].estimate.kv_quant)
                   and (row[0].fits_on_gpu or not hardware.gpus)]
    if comfortable:
        ranked = comfortable

    def rank(row):
        candidate, engine, _ = row
        # Precision is a small preference, below the measured fit/speed terms.
        precision = min(1.0, _runtime_bpw(candidate.quant, catalog) / 8.5)
        native_mlx = engine == "mlx" and bool(spec.mlx_repo)
        return (candidate.score + precision * 0.02, native_mlx,
                -KV_QUALITY_ORDER.index(candidate.estimate.kv_quant))

    ranked.sort(key=rank, reverse=True)
    best, chosen_runtime, runtime_reason = ranked[0]
    if chosen_runtime == "mlx":
        best.warnings.append(
            f"The MLX adapter does not enforce a server context limit; keep prompts within "
            f"the planned {profile.context_tokens:,}-token window."
        )
    candidates = [best]
    if _include_fallbacks:
        alternatives = []
        # Validate every strict capability/task match before limiting the list.
        # A general GGUF ranking can otherwise fill its first eight positions
        # with models whose native weights cannot fit this runtime.
        for alternative_spec in catalog.matching(profile):
            if alternative_spec.id == spec.id:
                continue
            try:
                alternate = build_requested_plan(
                    replace(profile, requested_model=alternative_spec.id), hardware, catalog,
                    runtime=chosen_runtime, allow_partial_offload=allow_partial_offload,
                    _include_fallbacks=False,
                )
            except PlanningError:
                continue
            if alternate.best is not None:
                alternatives.append(alternate.best)
        alternatives.sort(key=lambda c: (c.score, c.model.quality_tier,
                                         -c.model.params_b, c.model.id), reverse=True)
        candidates.extend(alternatives[:MAX_CANDIDATES - 1])

    notes = _plan_notes(candidates, profile, hardware, _parallel_slots(profile))
    # Ranking language from the general planner would claim the retained model
    # beat other models, which it was deliberately not asked to do.
    notes[0] = f"Keeping requested model {spec.name}; using {best.quant} with {chosen_runtime}."
    notes.append("Other models are reserved for recovery after same-model repairs are exhausted.")
    return Plan(candidates=candidates, runtime=chosen_runtime, runtime_reason=runtime_reason,
                profile=profile, hardware=hardware, notes=notes)


def _runtime_options(
    spec: ModelSpec, profile: UseCaseProfile, hardware: HardwareProfile,
    explicit: Optional[str], explicit_reason: Optional[str] = None,
) -> List[Tuple[str, str]]:
    if explicit is not None:
        names = [(explicit, explicit_reason or "explicitly requested")]
    else:
        chosen, reason = choose_runtime(hardware, profile)
        installed = recovery_order(hardware, profile)
        names = [(name, f"{name} is installed and can load {spec.name}") for name in installed]
        names.append((chosen, reason))
        names.append(("llamacpp", "llama.cpp can load this model's GGUF quantizations"))

    out = []
    seen = set()
    for name, reason in names:
        if name in seen:
            continue
        seen.add(name)
        if name in _GGUF_RUNTIMES:
            compatible = bool(spec.gguf_repo)
        elif name == "mlx":
            compatible = hardware.unified_memory and bool(spec.mlx_repo or spec.hf_repo)
        elif name == "vllm":
            compatible = (hardware.os == "linux"
                          and hardware.accelerator is Accelerator.NVIDIA and bool(spec.hf_repo))
        else:
            # The catalog does not identify EXL2 builds, so TabbyAPI cannot yet
            # supply a reproducible quantization for this flow.
            compatible = False
        if compatible:
            out.append((name, reason))
    if not out:
        raise PlanningError(
            f"no compatible weight format for {spec.name} with "
            f"{explicit or 'the runtimes available on this machine'}",
            hint="Use a runtime with a cataloged weight format or add that model format to the catalog.",
        )
    return out


def _runtime_quants(spec: ModelSpec, runtime: str) -> List[str]:
    if runtime in _GGUF_RUNTIMES:
        return list(spec.recommended_quants)
    if runtime == "mlx" and spec.mlx_repo:
        match = re.search(r"(?:-|_)([248])bit(?:$|[-_])", spec.mlx_repo, re.IGNORECASE)
        if match:
            return [f"MLX{match.group(1)}"]
    # Native HF weights are not the catalog's GGUF conversions. The adapters
    # do not request weight quantization, so budget their full precision.
    return ["F16"]


def _runtime_bpw(quant: str, catalog: Catalog) -> float:
    return catalog.bpw(quant)
