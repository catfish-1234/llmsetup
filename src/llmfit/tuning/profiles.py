"""The deep configuration surface.

Everything a serious local setup can be tuned on, as typed data rather than a
bag of strings. These objects are what separate "it runs" from "it runs well":

* :class:`SamplerChain` — the full decoding stack (DRY, XTC, top-n-sigma,
  typical-p, mirostat), including the *order* samplers run in, which changes the
  output as much as the values do.
* :class:`ReasoningPlan` — whether a hybrid model thinks, and how hard. Turning
  thinking off takes time-to-first-token from seconds to fractions of one, so on
  a latency-sensitive task this is the single largest lever available.
* :class:`SpeculativePlan` — a small draft model guessing ahead for a large one.
* :class:`OffloadPlan` — where tensors live: GPU layers, per-GPU split, and the
  MoE expert offload that decides whether a 235B model runs on a 24 GB card.
* :class:`RopePlan` — context extension past what the model was trained on.

Each carries its own ``rationale`` so the report can explain the setting rather
than just print it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ReasoningEffort(str, Enum):
    """How long a hybrid-reasoning model is allowed to think.

    ``OFF`` suppresses the thinking block entirely; the tiers map to the
    ``reasoning_effort`` levels these models expose.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


@dataclass
class SamplerChain:
    """The decoding stack beyond temperature and top-p.

    ``order`` matters and is not cosmetic: penalties applied before or after
    truncation produce different distributions, and mirostat only behaves if it
    runs last. The default mirrors llama.cpp's own chain.
    """

    # --- repetition control ------------------------------------------------ #
    #: DRY penalises repeated *sequences* rather than repeated tokens, which is
    #: what actually stops a model looping on a phrase. 0 disables it.
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = -1          # -1 = the whole context
    dry_sequence_breakers: List[str] = field(
        default_factory=lambda: ["\n", ":", "\"", "*"]
    )
    penalty_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # --- truncation -------------------------------------------------------- #
    #: XTC removes the most obvious continuations to break clichés. Only
    #: appropriate where surprise is wanted; it is actively harmful for code.
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    typical_p: float = 1.0                # 1.0 = disabled
    #: top-n-sigma truncates by distance from the logit mean, which adapts to
    #: how confident the model is instead of using a fixed cutoff.
    top_n_sigma: float = -1.0             # < 0 = disabled

    # --- adaptive ---------------------------------------------------------- #
    mirostat: int = 0                     # 0 off, 1 v1, 2 v2
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    #: Sampler order. llama.cpp's documented default; mirostat, when enabled,
    #: must be last, which :meth:`normalised_order` enforces.
    order: List[str] = field(default_factory=lambda: [
        "penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p",
        "min_p", "xtc", "temperature",
    ])
    rationale: str = ""

    @property
    def dry_enabled(self) -> bool:
        return self.dry_multiplier > 0

    @property
    def xtc_enabled(self) -> bool:
        return self.xtc_probability > 0

    @property
    def mirostat_enabled(self) -> bool:
        return self.mirostat > 0

    def normalised_order(self) -> List[str]:
        """The order actually passed to the engine.

        Disabled samplers are dropped so the chain says what it does, and
        temperature is forced last when mirostat is on because mirostat
        replaces the truncation step and must see the final distribution.
        """
        chain = [s for s in self.order if not (
            (s == "dry" and not self.dry_enabled)
            or (s == "xtc" and not self.xtc_enabled)
            or (s == "typ_p" and self.typical_p >= 1.0)
            or (s == "top_n_sigma" and self.top_n_sigma < 0)
        )]
        if self.mirostat_enabled:
            chain = [s for s in chain if s != "temperature"] + ["temperature"]
        return chain

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReasoningPlan:
    """Thinking budget for a hybrid-reasoning model."""

    effort: ReasoningEffort = ReasoningEffort.MEDIUM
    #: Keep the thinking block in the response. Agents parsing structured output
    #: usually want it stripped; a human debugging a wrong answer wants it kept.
    preserve: bool = False
    supported: bool = False
    rationale: str = ""

    @property
    def enabled(self) -> bool:
        return self.supported and self.effort is not ReasoningEffort.OFF

    @property
    def notable(self) -> bool:
        return self.supported

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effort"] = self.effort.value
        d["enabled"] = self.enabled
        return d


@dataclass
class SpeculativePlan:
    """A small draft model guessing ahead for a large one.

    Pays off in proportion to how predictable the output is — strong on code and
    structured text, weak on open-ended prose — so it is enabled per task rather
    than globally.
    """

    draft_model_id: Optional[str] = None
    draft_quant: str = "Q4_K_M"
    draft_path: Optional[str] = None
    draft_gpu_layers: int = 0
    draft_memory_bytes: int = 0
    max_draft_tokens: int = 16
    min_draft_tokens: int = 0
    #: Only accept a draft when the draft model is at least this confident;
    #: raising it trades a little speed for fewer wasted verification passes.
    p_min: float = 0.75
    expected_speedup: float = 1.0
    rationale: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.draft_model_id)

    @property
    def notable(self) -> bool:
        """Report when it is on, and when only memory stood in the way — that
        second case tells the user what more VRAM would buy them."""
        return self.enabled or "spare" in self.rationale

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["enabled"] = self.enabled
        return d


@dataclass
class OffloadPlan:
    """Where the tensors live.

    ``n_cpu_moe`` is the interesting one. A mixture-of-experts model is mostly
    routed expert weights, and only a few experts fire per token — so those
    weights can sit in system RAM while attention and the KV cache stay on the
    GPU. That trade is far cheaper for MoE than moving whole layers off a dense
    model, and it is what makes a 235B model runnable on a consumer card.
    """

    gpu_layers: int = 0
    total_layers: int = 0
    #: Move routed expert weights for the first N layers to system RAM.
    n_cpu_moe: int = 0
    #: Fraction of VRAM per GPU on a multi-card machine.
    tensor_split: List[float] = field(default_factory=list)
    main_gpu: int = 0
    split_mode: str = "layer"             # layer | row | none
    #: Raw --override-tensor patterns for cases the simple knobs cannot express.
    override_tensors: List[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def moe_offload_enabled(self) -> bool:
        return self.n_cpu_moe > 0

    @property
    def notable(self) -> bool:
        return self.moe_offload_enabled or bool(self.tensor_split) \
            or bool(self.override_tensors)

    @property
    def full_gpu(self) -> bool:
        return self.total_layers > 0 and self.gpu_layers >= self.total_layers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RopePlan:
    """Context extension beyond the trained window.

    YaRN lets a model address more context than it was trained on, at a real
    quality cost that grows with the scale factor — so this is only planned when
    the user asked for more context than the model natively has.
    """

    scaling_type: str = "none"            # none | linear | yarn
    factor: float = 1.0
    freq_base: Optional[float] = None
    original_context: Optional[int] = None
    yarn_ext_factor: Optional[float] = None
    yarn_attn_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    rationale: str = ""

    @property
    def enabled(self) -> bool:
        return self.scaling_type != "none" and self.factor > 1.0

    @property
    def notable(self) -> bool:
        return self.enabled

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ServingPlan:
    """Endpoint behaviour: batching, caching and reuse."""

    continuous_batching: bool = True
    #: Reuse a cached prefix when a new prompt shares one. Enormous for agents
    #: and RAG, where a long system prompt or document repeats every turn.
    prefix_caching: bool = True
    cache_reuse_tokens: int = 256
    chunked_prefill: bool = False
    max_batched_tokens: Optional[int] = None
    defrag_threshold: float = 0.1
    api_key: Optional[str] = None
    lora_adapters: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdvancedProfile:
    """Every deep setting, together, with the reasoning behind each."""

    sampler: SamplerChain = field(default_factory=SamplerChain)
    reasoning: ReasoningPlan = field(default_factory=ReasoningPlan)
    speculative: SpeculativePlan = field(default_factory=SpeculativePlan)
    offload: OffloadPlan = field(default_factory=OffloadPlan)
    rope: RopePlan = field(default_factory=RopePlan)
    serving: ServingPlan = field(default_factory=ServingPlan)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampler": self.sampler.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "speculative": self.speculative.to_dict(),
            "offload": self.offload.to_dict(),
            "rope": self.rope.to_dict(),
            "serving": self.serving.to_dict(),
        }

    def explain(self) -> List[str]:
        """Decisions worth reporting, as lines of prose.

        A plan that did nothing is not news: reporting "no scaling needed" on
        every run buries the settings that did change.
        """
        lines = []
        for name in ("offload", "speculative", "reasoning", "rope", "sampler",
                     "serving"):
            plan = getattr(self, name)
            notable = getattr(plan, "notable", bool(plan.rationale))
            if plan.rationale and notable:
                lines.append(f"{name}: {plan.rationale}")
        return lines
