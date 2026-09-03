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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    On a hybrid model the recurrent layers are counted as though they were
    attention layers, because their projections are the same shape of thing at
    roughly the same scale and the file does not state their dimensions in a
    form this can use. It is an approximation, and the one place it matters is
    the expert split for a hybrid mixture of experts; the figures are replaced
    by the file's own on download, and an under-estimate here surfaces as an
    out-of-memory the repair loop handles rather than as a wrong answer.
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


#: The routed expert tensors in one layer, as llama.cpp names them. Each is a
#: separate matrix of similar size, which is what makes a fractional layer
#: expressible at all: a third of a layer's experts can move while the rest stay.
EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


@dataclass
class ExpertOffload:
    """Which experts move to system RAM, down to the individual tensor.

    ``--n-cpu-moe`` works in whole layers, so freeing one byte more than a
    layer boundary moves an entire layer's experts and leaves hundreds of
    megabytes of VRAM holding nothing. Naming tensors instead reclaims that.
    People do this by hand with regexes and arrive at it by trial; the boundary
    is arithmetic, so it can simply be computed.
    """

    whole_layers: int = 0
    #: Tensors of the next layer that also move, when whole layers overshoot.
    partial_layer: Optional[int] = None
    partial_tensors: Tuple[str, ...] = ()

    @property
    def is_partial(self) -> bool:
        return self.partial_layer is not None and bool(self.partial_tensors)

    def override_patterns(self) -> List[str]:
        """`--override-tensor` patterns expressing exactly this split.

        Only the fractional part needs a pattern. The whole layers are said more
        simply and more robustly with ``--n-cpu-moe``, which does not depend on
        this build naming its tensors the way the regex expects.
        """
        if not self.is_partial:
            return []
        names = "|".join(self.partial_tensors)
        return [f"blk\\.{self.partial_layer}\\.({names})=CPU"]


def plan_expert_offload(
    spec: ModelSpec,
    bpw: float,
    excess_bytes: int,
) -> ExpertOffload:
    """Free ``excess_bytes`` by moving as few expert weights as will do it.

    Every expert left on the GPU is one that does not have to cross the bus when
    it fires, so rounding up to a whole layer is not a rounding error, it is
    speed given away. The remainder after the whole layers is expressed as
    individual tensors of the next layer.
    """
    plan = ExpertOffload()
    if excess_bytes <= 0 or not spec.is_moe or spec.layers <= 0:
        return plan
    per_layer = expert_bytes(spec, bpw) / spec.layers
    if per_layer <= 0:
        return plan

    whole = int(excess_bytes // per_layer)
    if whole >= spec.layers:
        # Even every expert is not enough; the caller has to find the rest
        # elsewhere, and there is no fraction left to be clever about.
        return ExpertOffload(whole_layers=spec.layers)
    plan.whole_layers = whole

    remainder = excess_bytes - whole * per_layer
    if remainder <= 0:
        return plan

    per_tensor = per_layer / len(EXPERT_TENSORS)
    needed = min(len(EXPERT_TENSORS), math.ceil(remainder / per_tensor))
    if needed >= len(EXPERT_TENSORS):
        # The fraction is most of a layer, so take the layer and keep the
        # command line simple.
        plan.whole_layers = whole + 1
        return plan
    plan.partial_layer = whole
    plan.partial_tensors = EXPERT_TENSORS[:needed]
    return plan


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
