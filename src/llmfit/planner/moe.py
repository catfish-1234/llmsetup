"""Mixture-of-experts memory accounting.

An MoE model is mostly routed expert weights, for a 30B-A3B model the experts
are around 95% of the parameters, and only a handful of experts fire per token.
That asymmetry is exploitable: the expert stacks can sit in system RAM while
attention, the router, the shared experts and the KV cache stay on the GPU.

The trade is far cheaper than moving whole layers off a dense model, because the
per-token work that crosses to the CPU is proportional to *active* parameters,
not total ones. It is what makes a 235B model runnable on a consumer card, and
it is why the planner has to model expert weights separately rather than
treating an MoE model as one undifferentiated blob.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llmfit.types import ModelSpec

#: Vocabulary size assumed when computing embedding parameters. Modern models
#: cluster around 128-152k; the term is small next to the expert stacks, so the
#: approximation costs little.
_ASSUMED_VOCAB = 152_000


@dataclass
class MoeBreakdown:
    """Parameters split into what must stay on the accelerator and what need not."""

    dense_params: float          # attention, embeddings, router, norms (billions)
    expert_params: float         # routed expert stacks (billions)
    total_params: float

    @property
    def expert_fraction(self) -> float:
        return self.expert_params / self.total_params if self.total_params else 0.0


def dense_params_b(spec: ModelSpec) -> float:
    """Non-expert parameters, in billions, from the architecture.

    Attention projections plus embeddings. Norms and the router are rounding
    error at this scale and are ignored rather than guessed at.
    """
    hidden = spec.hidden_size
    head_dim = spec.effective_head_dim
    # q and o are hidden x hidden; k and v are hidden x (kv_heads * head_dim).
    per_layer = 2 * hidden * hidden + 2 * hidden * spec.kv_heads * head_dim
    attention = per_layer * spec.layers
    # Input and output embeddings; tied weights make this an over-estimate,
    # which is the safe direction.
    embeddings = 2 * _ASSUMED_VOCAB * hidden
    return (attention + embeddings) / 1e9


def breakdown(spec: ModelSpec) -> Optional[MoeBreakdown]:
    """Split an MoE model's parameters. None for a dense model."""
    if not spec.is_moe:
        return None
    dense = min(dense_params_b(spec), spec.params_b * 0.5)
    return MoeBreakdown(
        dense_params=dense,
        expert_params=max(0.0, spec.params_b - dense),
        total_params=spec.params_b,
    )


def expert_bytes(spec: ModelSpec, bpw: float) -> int:
    """Bytes occupied by routed expert weights."""
    split = breakdown(spec)
    if split is None:
        return 0
    return int(split.expert_params * 1e9 * bpw / 8)


def layers_to_offload(
    spec: ModelSpec,
    bpw: float,
    excess_bytes: int,
) -> int:
    """How many layers' experts must move to RAM to free ``excess_bytes``.

    Returns 0 when the model already fits, and is capped at the layer count,
    a caller asking to free more than the experts are worth gets every layer,
    and must find the rest elsewhere.
    """
    if excess_bytes <= 0 or not spec.is_moe or spec.layers <= 0:
        return 0
    per_layer = expert_bytes(spec, bpw) / spec.layers
    if per_layer <= 0:
        return 0
    import math

    return min(spec.layers, math.ceil(excess_bytes / per_layer))


def resident_bytes_with_offload(
    spec: ModelSpec, bpw: float, n_cpu_moe: int
) -> int:
    """Weight bytes still on the accelerator after offloading N layers' experts."""
    total = int(spec.params_b * 1e9 * bpw / 8)
    if not spec.is_moe or n_cpu_moe <= 0 or spec.layers <= 0:
        return total
    freed = expert_bytes(spec, bpw) * min(n_cpu_moe, spec.layers) / spec.layers
    return int(total - freed)


def throughput_penalty(spec: ModelSpec, n_cpu_moe: int) -> float:
    """Fraction of original speed retained after offloading experts.

    Modelled on what actually crosses to the CPU: the *active* expert weights
    for the offloaded layers, not the whole expert stack. A dense model moved
    off the GPU by the same fraction would lose far more, which is exactly why
    this technique is MoE-specific.
    """
    if not spec.is_moe or n_cpu_moe <= 0 or spec.layers <= 0:
        return 1.0
    offloaded_fraction = min(n_cpu_moe, spec.layers) / spec.layers
    split = breakdown(spec)
    if split is None:
        return 1.0

    # Share of per-token compute that is routed-expert work.
    active_expert_share = max(
        0.0, (spec.compute_params_b - split.dense_params) / spec.compute_params_b
    ) if spec.compute_params_b else 0.0

    # System RAM is roughly an order of magnitude slower than VRAM for this
    # access pattern; the crossing fraction pays that multiple.
    crossing = offloaded_fraction * active_expert_share
    return 1.0 / (1.0 + crossing * 9.0)
