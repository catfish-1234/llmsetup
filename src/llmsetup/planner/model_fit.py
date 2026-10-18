"""Factual model-fit assessment shared by the interview and noninteractive CLI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from llmsetup.catalog.catalog import Catalog
from llmsetup.planner.plan import build_adaptive_plan
from llmsetup.runtimes.registry import scan_runtimes
from llmsetup.types import GIB, Candidate, HardwareProfile, ModelSpec, Plan, UseCaseProfile
from llmsetup.util.errors import CatalogError, PlanningError


@dataclass
class ModelAssessment:
    selected_model: ModelSpec
    plan: Optional[Plan] = None
    alternatives: List[Candidate] = field(default_factory=list)
    alternative_runtime: Optional[str] = None
    alternative_runtimes: Dict[str, str] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    poor_fit: bool = False


def resolve_requested_model(name: str, catalog: Catalog) -> ModelSpec:
    """Resolve exact IDs, display names, tags and source repos without guessing."""
    value = name.strip().casefold()
    exact_ids = [model for model in catalog.models if model.id.casefold() == value]
    if len(exact_ids) == 1:
        return exact_ids[0]
    matches = [model for model in catalog.models if value and value in {
        alias.casefold() for alias in (model.name, model.ollama_name, model.gguf_repo,
                                      model.hf_repo, model.mlx_repo) if alias
    }]
    if len(matches) == 1:
        return matches[0]
    if matches or exact_ids:
        choices = ", ".join(model.id for model in matches or exact_ids)
        raise CatalogError(f"model name {name!r} matches more than one catalog entry",
                           hint=f"Use one of these model IDs: {choices}.")
    raise CatalogError(f"unknown model {name!r}",
                       hint="Run `llmsetup catalog list` and use an exact model ID or name.")


def assess_requested_model(
    profile: UseCaseProfile, hardware: HardwareProfile, catalog: Catalog,
    runtime: Optional[str] = None,
) -> ModelAssessment:
    """Describe the retained model and offer feasible alternatives when needed.

    Estimates are the planner's own memory and bandwidth estimates, not a
    benchmark of the user's machine or a claim about measured model quality.
    """
    if not profile.requested_model:
        raise PlanningError("no specific model was requested")
    spec = resolve_requested_model(profile.requested_model, catalog)
    profile = replace(profile, requested_model=spec.id)
    assessment = ModelAssessment(selected_model=spec)
    messages = assessment.messages
    covered = [task.value for task in profile.tasks if task.value in spec.tasks]
    uncovered = [task.value for task in profile.tasks if task.value not in spec.tasks]
    if covered:
        messages.append(f"{spec.name} is cataloged for: {', '.join(covered)}.")
    if uncovered:
        messages.append(f"The catalog does not list {spec.name} for: {', '.join(uncovered)}; "
                        "quality for those tasks is uncertain.")
        assessment.poor_fit = True

    messages.append(
        f"Detected memory budgets: {hardware.usable_cpu_bytes / GIB:.1f} GiB of system RAM"
        + (f" and {hardware.memory_budget / GIB:.1f} GiB for the GPU."
           if hardware.gpus else ".")
    )
    if hardware.unified_memory:
        messages.append("The GPU and CPU share the same memory pool.")
    try:
        assessment.plan = build_adaptive_plan(profile, hardware, catalog, runtime=runtime)
    except PlanningError as exc:
        messages.append(f"{exc.message}." + (f" {exc.hint}" if exc.hint else ""))
        assessment.poor_fit = True
    else:
        best = assessment.plan.best
        assert best is not None
        messages.append(
            f"Estimated fit: {best.quant} weights, {best.estimate.kv_quant} KV cache, "
            f"{best.estimate.context_tokens:,} tokens of context, "
            f"{best.estimate.total_gb:.1f} GiB total, using {assessment.plan.runtime}."
        )
        messages.append("Lower weight precision saves memory and can reduce output quality.")
        if best.estimate.context_tokens < profile.context_tokens:
            messages.append(
                f"The requested {profile.context_tokens:,}-token context does not fit "
                f"the memory budget or the model's {spec.max_context:,}-token limit; "
                f"using {best.estimate.context_tokens:,} tokens."
            )
            assessment.poor_fit = True
        if best.gpu_fraction:
            layers = round(best.gpu_fraction * spec.layers)
            messages.append(f"Estimated GPU offload: {layers} of {spec.layers} layers.")
        else:
            messages.append("All layers run on CPU; speed depends on system memory bandwidth.")
        tps = best.score_breakdown.get("est_tokens_per_s", 0.0)
        messages.append(f"Estimated generation speed: {tps:g} tokens/s; actual speed may differ.")
        if (hardware.gpus and not best.fits_on_gpu) or tps < 10:
            assessment.poor_fit = True
        headroom_budget = (hardware.memory_budget if best.fits_on_gpu or not hardware.gpus
                           else hardware.usable_cpu_bytes)
        if best.estimate.total > headroom_budget * 0.92:
            messages.append("This configuration has little memory headroom.")
            assessment.poor_fit = True
        messages.extend(best.warnings)

    if assessment.poor_fit:
        assessment.alternatives = _alternatives(profile, hardware, catalog, assessment, runtime)
        for candidate in assessment.alternatives:
            tasks = ", ".join(task.value for task in profile.tasks
                              if task.value in candidate.model.tasks)
            messages.append(
                f"Alternative: {candidate.model.name} ({candidate.model.id}) at {candidate.quant}, "
                f"{candidate.estimate.context_tokens:,} tokens of context, estimated "
                f"{candidate.estimate.total_gb:.1f} GiB total and "
                f"{candidate.score_breakdown.get('est_tokens_per_s', 0):g} tokens/s, "
                f"using {assessment.alternative_runtimes[candidate.model.id]}"
                + (f"; cataloged for {tasks}." if tasks else ".")
            )
    return assessment


def _alternatives(
    profile: UseCaseProfile, hardware: HardwareProfile, catalog: Catalog,
    assessment: ModelAssessment, runtime: Optional[str],
) -> List[Candidate]:
    selected = assessment.selected_model
    # Each model gets the original requested context and the same adaptive
    # reductions down to 512. Seed every hard-capability/task match that could
    # satisfy that floor, rather than a truncated GGUF-ranked candidate list.
    minimum_profile = replace(profile, requested_model=None,
                              context_tokens=min(profile.context_tokens, 512))
    pool = [spec for spec in catalog.matching(minimum_profile) if spec.id != selected.id]
    if pool and runtime is None and not hardware.runtime_availability:
        inventory, _ = scan_runtimes()
        hardware = replace(hardware, runtime_availability=inventory)
    alternatives = []
    for spec in pool:
        # Reassess the weight format too: a GGUF memory estimate cannot describe
        # an unquantized native HF download in vLLM or MLX.
        try:
            plan = build_adaptive_plan(replace(profile, requested_model=spec.id), hardware,
                                       catalog.restrict_to(spec.id), runtime=runtime)
        except PlanningError:
            continue
        best = plan.best
        if best is None:
            continue
        original = assessment.plan.best if assessment.plan else None
        if original is not None and not _improves_fit(best, original, profile, hardware):
            continue
        alternatives.append(best)
        assessment.alternative_runtimes[best.model.id] = plan.runtime
    # Sort and cap only after actual native weight formats and context windows
    # have been checked. Task coverage and family identify close substitutes.
    alternatives.sort(key=lambda c: (
        sum(task.value in c.model.tasks for task in profile.tasks),
        c.model.family == selected.family,
        c.fits_on_gpu, c.estimate.context_tokens, c.score, -c.model.params_b, c.model.id,
    ), reverse=True)
    alternatives = alternatives[:3]
    assessment.alternative_runtimes = {c.model.id: assessment.alternative_runtimes[c.model.id]
                                      for c in alternatives}
    if alternatives:
        assessment.alternative_runtime = assessment.alternative_runtimes[alternatives[0].model.id]
    return alternatives


def _improves_fit(candidate, original, profile, hardware) -> bool:
    task_fit = sum(task.value in candidate.model.tasks for task in profile.tasks)
    original_fit = sum(task.value in original.model.tasks for task in profile.tasks)
    return (
        task_fit > original_fit
        or candidate.estimate.context_tokens > original.estimate.context_tokens
        or (hardware.gpus and candidate.fits_on_gpu and not original.fits_on_gpu)
        or candidate.score_breakdown.get("est_tokens_per_s", 0)
        > original.score_breakdown.get("est_tokens_per_s", 0) * 1.2
        or candidate.estimate.total < original.estimate.total * 0.8
    )
