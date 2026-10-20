"""The memory model.

Three terms decide whether a model runs:

    total = weights + kv_cache + overhead

**weights** ``params * bits_per_weight / 8``. For MoE models this uses *total*
parameters, not active ones: every expert has to be resident even though only a
few fire per token. Conflating the two is the classic way to plan a 30B-A3B
model into 6 GB of VRAM and get OOM-killed.

**kv_cache** ``2 * layers * kv_heads * head_dim * tokens * bytes_per_element``.
The leading 2 is K and V. Using ``kv_heads`` (not attention heads) is what makes
this correct for GQA models, which is all of them now. This term is linear in
context length and is usually what actually decides your context ceiling: on a
24 GB card the KV cache at 128K can cost more than the weights.

A note on units, because published figures mix them: everything here is in
**bytes**. Model and file sizes are quoted in the wild in decimal GB (1e9),
while GPU capacity is in binary GiB (2^30), a 24 GB card really holds 24 GiB.
Comparing the two directly is a 7% error in the dangerous direction, so the code
never carries a unit-less "GB" number; it carries bytes and converts at display.

**overhead** compute buffers, activations and the runtime's own allocation. This
is an *empirical* model calibrated against observed llama.cpp/Ollama residency,
not a derivation. It is documented as such and deliberately errs high, because
underestimating here means an OOM at inference time instead of a smaller model
at plan time.
"""

from __future__ import annotations

import math
from typing import Optional

from llmsetup.types import MemoryEstimate, ModelSpec

# Bytes per element for KV cache dtypes. llama.cpp's quantized KV types carry a
# small block scale, hence the non-integer values.
KV_ELEMENT_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "iq4_nl": 0.5625,
    "q4_1": 0.5625,
    "q4_0": 0.5625,
}

# Asymmetric cache: keys at q8_0, values at q4_0 ("K/V"). Measured perplexity
# on llama.cpp shows the key cache is the sensitive half (q4_0 keys collapse the
# model) while the value cache tolerates q4_0 almost for free, so this saves
# about 40% of the cache for a small cost where symmetric q4_0 is unusable.
# Only llama.cpp can set the two types separately.
KV_ASYMMETRIC = "q8_0/q4_0"

# Quality cost of quantizing the KV cache, cheapest first.
KV_QUALITY_ORDER = ["f16", "q8_0", KV_ASYMMETRIC, "q5_1", "q4_0"]


def kv_element_bytes(kv_quant: str) -> float:
    """Average bytes per cache element; K and V are the same size, so a
    ``K/V`` pair averages."""
    parts = kv_quant.lower().split("/")
    return sum(KV_ELEMENT_BYTES.get(part, 2.0) for part in parts) / len(parts)


def kv_split(kv_quant: str) -> tuple:
    """``(key_type, value_type)`` for a plain or ``K/V`` cache type."""
    parts = kv_quant.lower().split("/")
    return (parts[0], parts[-1])


def kv_options(runtime: str, precision_sensitive: bool = False) -> list:
    """Cache types worth trying on this runtime, best quality first.

    Asymmetric needs llama.cpp. For precision-sensitive work (code, tool calls,
    retrieval) a symmetric type below q8_0 is skipped: q4_0 keys are the part
    that breaks, and a shorter context is the honest trade there.
    """
    if runtime == "mlx":
        return ["f16"]
    options = []
    for kv in KV_QUALITY_ORDER:
        if kv == KV_ASYMMETRIC and runtime != "llamacpp":
            continue
        if precision_sensitive and kv in ("q5_1", "q4_0"):
            continue
        options.append(kv)
    return options

# The rung below which quantizing the KV cache stops being a cheap win.
#
# q8_0 is as close to free as a memory saving gets: measured against an f16
# reference it changes a handful of answers in five hundred, and none at all
# across deep-context comparisons out to 41K tokens. Below it the picture
# changes and stops being predictable -- q4_0 is strongly model-dependent,
# ranging from a couple of changed answers on one model to 375 in 500 on
# another, and at very long context it is also *slower* than f16, because
# dequantizing during attention costs more than the memory traffic it saves
# (~37% slower generation at ~110K tokens).
#
# So: q8_0 is chosen freely, and anything below it is a deliberate last resort
# that the user is told about, never a silent default.
KV_SAFE_FLOOR = "q8_0"

# Above this, a sub-q8_0 KV cache is not a trade at all -- it is slower *and*
# worse, so there is nothing to weigh.
KV_LOSSY_CONTEXT_LIMIT = 32768


def kv_is_lossy(kv_quant: str) -> bool:
    """Whether this KV dtype is past the point of being a free saving."""
    try:
        return (KV_QUALITY_ORDER.index(kv_quant.lower())
                > KV_QUALITY_ORDER.index(KV_SAFE_FLOOR))
    except ValueError:
        return False


def kv_lossy_allowed(context_tokens: int) -> bool:
    """Whether dropping below the safe floor could still help at this context."""
    return context_tokens <= KV_LOSSY_CONTEXT_LIMIT

# --- empirical overhead constants (bytes) ---------------------------------- #
# Fixed cost of bringing up a runtime + its GPU context.
_RUNTIME_FIXED = {
    "llamacpp": 200 * 1024 ** 2,
    "ollama": 350 * 1024 ** 2,
    "lmstudio": 400 * 1024 ** 2,
    "vllm": 1200 * 1024 ** 2,   # CUDA graphs + a much heavier Python stack
    "mlx": 250 * 1024 ** 2,
}
_DEFAULT_FIXED = 300 * 1024 ** 2

# Number of hidden-size-wide fp16 activation tensors alive at once during a
# forward pass. Fitted against observed compute-buffer sizes; not derived.
_ACTIVATION_TENSORS = 18

# Output projection buffer. Modern models have 100-250K vocabularies, and the
# logits buffer is large enough to matter on small cards.
_ASSUMED_VOCAB = 152_000

# A vision model loads a second network: the projector (a SigLIP/CLIP tower plus
# the layer mapping its output into token space), kept at f16 or bf16 because
# vision quality degrades quickly under quantization. The towers in circulation
# run roughly 400M-1B parameters, so the file alone is most of a gigabyte, and
# image activations sit on top. Counting it is the difference between a vision
# model that fits and one that OOMs the moment it is sent a picture.
_VISION_PROJECTOR_BYTES = 1024 * 1024 ** 2


def kv_bytes_per_token(spec: ModelSpec, kv_quant: str = "f16") -> float:
    """Bytes of KV cache one token costs, if every layer attends globally.

    GQA-aware: the cache is sized by key/value heads, not query heads, which is
    the single most common way a memory estimate comes out several times too
    large.

    For a model with sliding-window layers this is the worst case rather than
    the real one, because those layers stop caching once the window is full.
    Use :func:`kv_cache_bytes`, which knows the difference.
    """
    element = kv_element_bytes(kv_quant)
    return 2 * spec.layers * spec.kv_heads * spec.effective_head_dim * element


def kv_cache_bytes(spec: ModelSpec, context_tokens: int, kv_quant: str = "f16",
                   swa_full: bool = False) -> int:
    """KV cache for a whole context, accounting for sliding-window layers.

    Not every layer costs the same, and on some models most of them cost nothing
    per token at all:

    * Recurrent layers, as in the Gated DeltaNet blocks that make up three
      quarters of a hybrid model, carry a fixed-size state instead of a cache.
    * Windowed attention layers cache the window and no more, however long the
      context is.
    * Only globally-attending layers grow with the whole context.

    Treating every layer as the third kind is not conservative in a useful way.
    It just makes models look impossible that in fact fit comfortably.

    ``swa_full`` covers the case where the engine has been told to keep a
    full-size cache for the windowed layers anyway. That costs the full amount
    again, and it is sometimes worth it, so the maths has to be able to say what
    it would cost.
    """
    context_tokens = max(1, int(context_tokens))
    element = kv_element_bytes(kv_quant)
    per_layer_token = 2 * spec.kv_heads * spec.effective_head_dim * element

    # Recurrent layers hold a state whose size does not depend on how long the
    # context is. It is part of the footprint but not part of the per-token
    # cost, so it is added once rather than multiplied by anything.
    recurrent = spec.recurrent_layers * spec.recurrent_state_bytes

    if not spec.uses_swa or swa_full:
        return int(per_layer_token * spec.attention_layers * context_tokens
                   + recurrent)

    windowed = min(context_tokens, int(spec.sliding_window or context_tokens))
    return int(per_layer_token * (spec.global_layers * context_tokens
                                  + spec.local_layers * windowed)
               + recurrent)


def weight_bytes(spec: ModelSpec, bpw: float) -> int:
    """Resident weight bytes. Uses total parameters. MoE experts are all resident."""
    return int(spec.params_b * 1e9 * bpw / 8)


def overhead_bytes(
    spec: ModelSpec,
    runtime: str = "llamacpp",
    ubatch: int = 512,
    parallel: int = 1,
) -> int:
    """Empirical non-weight, non-KV memory. Errs high on purpose."""
    fixed = _RUNTIME_FIXED.get(runtime, _DEFAULT_FIXED)
    activations = ubatch * spec.hidden_size * 2 * _ACTIVATION_TENSORS
    logits = _ASSUMED_VOCAB * 4 * max(1, parallel)
    # A small proportional term: bigger graphs need bigger scratch space.
    graph = int(spec.layers * spec.hidden_size * 2 * 64)
    vision = _VISION_PROJECTOR_BYTES if spec.supports_vision else 0
    return int(fixed + activations + logits + graph + vision)


def estimate_memory(
    spec: ModelSpec,
    bpw: float,
    context_tokens: int,
    quant: str = "Q4_K_M",
    kv_quant: str = "f16",
    runtime: str = "llamacpp",
    ubatch: int = 512,
    parallel: int = 1,
    swa_full: bool = False,
) -> MemoryEstimate:
    """Full memory breakdown for one (model, quant, context) combination.

    ``parallel`` multiplies the KV cache: each concurrent slot needs its own
    cache, which is why a 4-user endpoint needs four times the context memory of
    a single-user chat at the same context length.
    """
    context_tokens = max(1, int(context_tokens))
    parallel = max(1, int(parallel))
    kv_total_tokens = context_tokens * parallel

    return MemoryEstimate(
        weights=weight_bytes(spec, bpw),
        kv_cache=kv_cache_bytes(spec, kv_total_tokens, kv_quant, swa_full=swa_full),
        overhead=overhead_bytes(spec, runtime, ubatch, parallel),
        context_tokens=context_tokens,
        quant=quant,
        kv_quant=kv_quant,
    )


def max_context_for_budget(
    spec: ModelSpec,
    bpw: float,
    budget_bytes: int,
    kv_quant: str = "f16",
    runtime: str = "llamacpp",
    ubatch: int = 512,
    parallel: int = 1,
    granularity: int = 1024,
    swa_full: bool = False,
) -> int:
    """Largest context that fits in ``budget_bytes``. Zero if the weights alone don't.

    This is the inverse of :func:`estimate_memory` and is what the repair loop
    uses to shrink context to a value that will actually fit, rather than
    halving blindly.

    For a sliding-window model the relationship is not a straight line. Up to
    the window every attention layer caches; past it only the globally-attending
    layers keep growing while the rest hold steady. Recurrent layers never grow
    at all, so on a hybrid model they come off the budget once and then stop
    mattering. Inverting the flat formula would hand back a context several
    times shorter than the model can really take.
    """
    parallel = max(1, parallel)
    fixed = weight_bytes(spec, bpw) + overhead_bytes(spec, runtime, ubatch, parallel)
    remaining = budget_bytes - fixed
    if remaining <= 0:
        return 0

    element = kv_element_bytes(kv_quant)
    per_layer_token = 2 * spec.kv_heads * spec.effective_head_dim * element * parallel
    if per_layer_token <= 0:
        return spec.max_context

    remaining -= spec.recurrent_layers * spec.recurrent_state_bytes
    if remaining <= 0:
        return 0

    if not spec.uses_swa or swa_full:
        tokens = int(remaining / (per_layer_token * spec.attention_layers))
    else:
        window = int(spec.sliding_window or 0)
        # Below the window nothing is capped yet, so every layer still counts.
        flat = int(remaining / (per_layer_token * spec.attention_layers))
        if flat <= window:
            tokens = flat
        else:
            # Past the window the local layers are a fixed cost and only the
            # global ones keep growing.
            local_cost = per_layer_token * spec.local_layers * window
            growth = per_layer_token * spec.global_layers
            tokens = int((remaining - local_cost) / growth) if growth > 0 else window
            tokens = max(window, tokens)

    tokens = (tokens // granularity) * granularity          # round down to a clean value
    return max(0, min(tokens, spec.max_context))


def layers_that_fit(
    spec: ModelSpec,
    bpw: float,
    budget_bytes: int,
    context_tokens: int,
    kv_quant: str = "f16",
    runtime: str = "llamacpp",
    ubatch: int = 512,
    parallel: int = 1,
) -> int:
    """How many transformer layers fit on the accelerator (partial offload).

    llama.cpp offloads whole layers, so when a model does not fit entirely we
    compute how many of its layers do. The KV cache for offloaded layers lives on
    the GPU too, so both scale together.
    """
    if spec.layers <= 0:
        return 0
    overhead = overhead_bytes(spec, runtime, ubatch, parallel)
    available = budget_bytes - overhead
    if available <= 0:
        return 0

    per_layer_weights = weight_bytes(spec, bpw) / spec.layers
    # Averaged across layers. On a sliding-window model the global and local
    # layers cost different amounts, but offload takes a contiguous run of them
    # in roughly the published proportion, so the average is the right estimate
    # to use here and the exact split is not knowable in advance anyway.
    per_layer_kv = (kv_cache_bytes(spec, context_tokens * max(1, parallel), kv_quant)
                    / spec.layers)
    per_layer = per_layer_weights + per_layer_kv
    if per_layer <= 0:
        return spec.layers

    fit = math.floor(available / per_layer)
    return max(0, min(fit, spec.layers))


def fits(estimate: MemoryEstimate, budget_bytes: int, headroom: float = 0.0) -> bool:
    """Whether an estimate fits, optionally reserving a fraction as headroom."""
    return estimate.total <= budget_bytes * (1.0 - headroom)


# The repair ladder is runtime-agnostic, so it never steps onto the llama.cpp-only
# asymmetric type; the planner offers that one directly.
_SYMMETRIC_LADDER = [kv for kv in KV_QUALITY_ORDER if kv != KV_ASYMMETRIC]


def cheaper_kv_quant(current: str, allow_lossy: bool = False) -> Optional[str]:
    """Next symmetric KV cache dtype down the quality ladder, or None at the bottom.

    Stops at :data:`KV_SAFE_FLOOR` unless ``allow_lossy`` is set, so the
    ordinary path can never quietly hand back a dtype that changes answers.
    """
    try:
        idx = _SYMMETRIC_LADDER.index(current.lower())
    except ValueError:
        return KV_SAFE_FLOOR
    nxt = idx + 1
    if nxt >= len(_SYMMETRIC_LADDER):
        return None
    if not allow_lossy and nxt > _SYMMETRIC_LADDER.index(KV_SAFE_FLOOR):
        return None
    return _SYMMETRIC_LADDER[nxt]
