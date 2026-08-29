"""Candidate scoring.

Hard requirements are already gone by the time we get here (the catalog filter
removed them), so scoring only weighs *soft* preferences. Every component is
normalised to 0..1 and the weights shift with the user's stated priority, so
"I care about speed" and "I care about quality" genuinely produce different
answers rather than reordering the same list.

The breakdown is retained on each candidate so the report can explain the pick
instead of asserting it.
"""

from __future__ import annotations

from typing import Dict

from llmfit.hardware.bandwidth import estimate_tokens_per_second
from llmfit.types import Candidate, HardwareProfile, ModelSpec, Priority, Task, UseCaseProfile

# Weight profiles per priority. Columns must sum to 1.0.
WEIGHTS: Dict[Priority, Dict[str, float]] = {
    Priority.SPEED:    {"quality": 0.18, "speed": 0.42, "fit": 0.25, "task": 0.10, "license": 0.05},
    Priority.BALANCED: {"quality": 0.30, "speed": 0.25, "fit": 0.28, "task": 0.12, "license": 0.05},
    Priority.QUALITY:  {"quality": 0.45, "speed": 0.10, "fit": 0.28, "task": 0.12, "license": 0.05},
}

# Throughput past which extra speed stops being worth quality. People read at
# roughly 10 tok/s; 30 is comfortably faster than anyone consumes text
# interactively, and the sqrt curve below compresses everything above it so a
# 200 tok/s model is never ranked over a better one merely for being faster.
SPEED_SATURATION = 30.0

# Quantization levels below this lose enough quality to be worth a penalty.
_QUANT_PENALTY = {
    "Q2_K": 0.30, "Q3_K_S": 0.20, "Q3_K_M": 0.12, "Q3_K_L": 0.08, "IQ4_XS": 0.05,
}

_PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc0-1.0"}


def score_candidate(
    candidate: Candidate,
    profile: UseCaseProfile,
    hardware: HardwareProfile,
) -> Candidate:
    """Attach a 0..1 score plus its breakdown. Mutates and returns the candidate."""
    weights = WEIGHTS[profile.priority]
    spec = candidate.model

    # _speed_score records the predicted throughput into the breakdown, so build
    # the breakdown first and add to it rather than replacing it afterwards.
    components = {
        "quality": _quality_score(spec, candidate.quant),
        "speed": _speed_score(candidate, hardware),
        "fit": _fit_score(candidate, hardware),
        "task": _task_score(spec, profile),
        "license": 1.0 if spec.license.lower() in _PERMISSIVE else 0.6,
    }

    score = sum(components[k] * weights[k] for k in weights)

    # A model that does not fit on the accelerator is not merely lower-scoring —
    # it is a different experience. Apply a multiplicative penalty so a partially
    # offloaded large model never edges out a smaller one that fits.
    if not candidate.fits_on_gpu and hardware.gpus:
        score *= 0.35 + 0.45 * candidate.gpu_fraction

    candidate.score = round(score, 4)
    candidate.score_breakdown.update({k: round(v, 3) for k, v in components.items()})
    candidate.score_breakdown["weighted_total"] = round(score, 4)
    return candidate


def _quality_score(spec: ModelSpec, quant: str) -> float:
    """Model capability, discounted by how hard it has been quantized."""
    base = (spec.quality_tier - 1) / 4.0            # tiers 1..5 -> 0..1

    # Bigger models retain more of their quality when quantized; a 2-bit 70B is
    # still often better than an 8-bit 3B, but 2-bit hurts a small model badly.
    penalty = _QUANT_PENALTY.get(quant.upper(), 0.0)
    if penalty and spec.params_b < 15:
        penalty *= 1.5
    scale = 1.0 - min(0.45, penalty)

    # Parameter count is a weak independent signal on top of the curated tier.
    size_bonus = min(0.12, (spec.params_b ** 0.5) / 100.0)
    return max(0.0, min(1.0, base * scale + size_bonus))


def _speed_score(candidate: Candidate, hardware: HardwareProfile) -> float:
    """Predicted throughput, saturating so that "fast enough" stops earning points."""
    active_bytes = int(
        candidate.model.compute_params_b * 1e9 * _bpw_of(candidate) / 8
    )
    tps = estimate_tokens_per_second(hardware, active_bytes, candidate.gpu_fraction)
    candidate.score_breakdown["est_tokens_per_s"] = round(tps, 1)
    # sqrt so the first 30 tok/s are worth far more than the next 170.
    return min(1.0, (tps / SPEED_SATURATION) ** 0.5)


def _fit_score(candidate: Candidate, hardware: HardwareProfile) -> float:
    """Reward comfortable headroom; punish running right at the edge.

    Utilisation above ~92% is where fragmentation, a longer prompt, or another
    application opening a browser tab turns into an OOM, so the curve drops off
    there rather than peaking at 100%.
    """
    budget = hardware.memory_budget
    if budget <= 0:
        return 0.0
    utilisation = candidate.estimate.total / budget

    if utilisation > 1.0:
        return max(0.0, 0.25 / utilisation)
    if utilisation > 0.92:
        return 0.45                     # fits, but no room to breathe
    if utilisation >= 0.55:
        return 1.0                      # the sweet spot: uses the hardware, leaves slack
    # Comfortably under-using the machine: fine, but a bigger model was available.
    return 0.55 + 0.45 * (utilisation / 0.55)


def _task_score(spec: ModelSpec, profile: UseCaseProfile) -> float:
    """How much of what the user wants to do this model is actually built for."""
    if not profile.tasks:
        return 0.7
    wanted = list(profile.tasks)
    covered = sum(1 for t in wanted if t.value in spec.tasks)
    coverage = covered / len(wanted)

    # The first task named is the one they care about most.
    primary_bonus = 0.15 if profile.primary_task.value in spec.tasks else 0.0
    # Specialist models earn their keep on their specialty.
    if (profile.primary_task in (Task.CODING, Task.AGENT)
            and "coding" in spec.tasks
            and ("coder" in spec.id or "coder" in spec.name.lower())):
        primary_bonus += 0.15
    return min(1.0, coverage * 0.85 + primary_bonus)


def _bpw_of(candidate: Candidate) -> float:
    """Recover bits-per-weight from the already-computed estimate."""
    params = candidate.model.params_b * 1e9
    return (candidate.estimate.weights * 8) / params if params else 4.85


def explain(candidate: Candidate, profile: UseCaseProfile) -> str:
    """One-sentence justification, built from the winning score components."""
    weights = WEIGHTS[profile.priority]
    contributions = {
        k: candidate.score_breakdown.get(k, 0) * w
        for k, w in weights.items()
    }
    top = sorted(contributions, key=contributions.get, reverse=True)[:2]
    phrases = {
        "quality": "strongest output quality in what fits",
        "speed": "fastest of the viable options",
        "fit": "uses the hardware without running at the edge",
        "task": f"built for {profile.primary_task.value}",
        "license": "permissively licensed",
    }
    return " and ".join(phrases[k] for k in top)
