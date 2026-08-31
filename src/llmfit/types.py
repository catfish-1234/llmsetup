"""Core data model.

Every pipeline stage is a transformation between the dataclasses defined here.
Keeping them plain (no behaviour beyond derived properties, no I/O) is what lets
the planner, tuner and repair logic be tested without a GPU or a network.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

GIB = 1024 ** 3
MIB = 1024 ** 2


# --------------------------------------------------------------------------- #
# Use case
# --------------------------------------------------------------------------- #


class Task(str, enum.Enum):
    """What the user actually wants the model to do."""

    CHAT = "chat"
    CODING = "coding"
    AGENT = "agent"
    RAG = "rag"
    SUMMARIZE = "summarize"
    CREATIVE = "creative"
    TRANSLATE = "translate"
    EXTRACT = "extract"
    VISION = "vision"
    EMBEDDING = "embedding"


class Priority(str, enum.Enum):
    """Where the user wants to sit on the quality/speed curve."""

    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class Client(str, enum.Enum):
    """Apps the user wants to drive the model from."""

    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPEN_WEBUI = "open_webui"
    JAN = "jan"
    CONTINUE = "continue"
    AIDER = "aider"
    ZED = "zed"
    CURSOR = "cursor"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    CUSTOM_OPENAI = "custom_openai"


@dataclass
class UseCaseProfile:
    """The answers to 'what are you going to use it for'.

    ``needs_*`` fields do double duty: the planner uses them as hard filters over
    the catalog, and the verifier uses them to decide which checks must pass.
    """

    tasks: List[Task] = field(default_factory=lambda: [Task.CHAT])
    clients: List[Client] = field(default_factory=list)
    priority: Priority = Priority.BALANCED
    context_tokens: int = 8192
    concurrent_users: int = 1
    needs_tools: bool = False
    needs_vision: bool = False
    needs_json: bool = False
    needs_embeddings: bool = False
    needs_reasoning: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    commercial_use: bool = False
    offline_only: bool = False
    max_disk_gb: Optional[float] = None
    notes: str = ""

    @property
    def primary_task(self) -> Task:
        return self.tasks[0] if self.tasks else Task.CHAT

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tasks"] = [t.value for t in self.tasks]
        d["clients"] = [c.value for c in self.clients]
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UseCaseProfile":
        d = dict(d or {})
        if "tasks" in d:
            d["tasks"] = [Task(t) for t in d["tasks"]]
        if "clients" in d:
            d["clients"] = [Client(c) for c in d["clients"]]
        if "priority" in d:
            d["priority"] = Priority(d["priority"])
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


class Accelerator(str, enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    INTEL = "intel"
    CPU = "cpu"


@dataclass
class GPU:
    index: int
    name: str
    vendor: Accelerator
    vram_bytes: int
    vram_free_bytes: Optional[int] = None
    compute_capability: Optional[str] = None
    driver: Optional[str] = None

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / GIB

    @property
    def supports_flash_attention(self) -> bool:
        """CUDA (SM 7.5+) and Metal have usable flash-attention kernels."""
        if self.vendor is Accelerator.APPLE:
            return True
        if self.vendor is Accelerator.NVIDIA:
            if not self.compute_capability:
                return True  # assume a modern card; repair can turn it off
            try:
                return float(self.compute_capability) >= 7.5
            except ValueError:
                return True
        return False


@dataclass
class HardwareProfile:
    """Best-effort description of the machine. Every field may be absent."""

    os: str = "unknown"                 # linux | darwin | windows
    arch: str = "unknown"               # x86_64 | arm64
    cpu_name: str = "unknown"
    physical_cores: Optional[int] = None
    logical_cores: Optional[int] = None
    ram_bytes: int = 0
    ram_available_bytes: Optional[int] = None
    unified_memory: bool = False        # Apple Silicon: RAM and VRAM are the same pool
    gpus: List[GPU] = field(default_factory=list)
    disk_free_bytes: int = 0
    detection_warnings: List[str] = field(default_factory=list)

    @property
    def accelerator(self) -> Accelerator:
        return self.gpus[0].vendor if self.gpus else Accelerator.CPU

    @property
    def total_vram_bytes(self) -> int:
        return sum(g.vram_bytes for g in self.gpus)

    @property
    def ram_gb(self) -> float:
        return self.ram_bytes / GIB

    @property
    def vram_gb(self) -> float:
        return self.total_vram_bytes / GIB

    @property
    def usable_accel_bytes(self) -> int:
        """Memory a model may actually occupy on the accelerator.

        Apple Silicon is the special case: there is no separate VRAM pool, and
        the OS caps GPU-resident working set well below installed RAM. We use a
        conservative fraction so the desktop stays alive.
        """
        if self.unified_memory:
            return int(self.ram_bytes * 0.70)
        return self.total_vram_bytes

    @property
    def usable_cpu_bytes(self) -> int:
        """Memory a CPU-resident model may occupy.

        Never the full installed RAM: the OS, the page cache and whatever else
        the user has open all need room, and an OOM killer that takes out the
        desktop is a worse outcome than a smaller model.
        """
        base = self.ram_available_bytes or self.ram_bytes
        return int(base * 0.75)

    @property
    def memory_budget(self) -> int:
        """The budget a model is planned against on this machine."""
        return self.usable_accel_bytes if self.gpus else self.usable_cpu_bytes

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["gpus"] = [{**asdict(g), "vendor": g.vendor.value} for g in self.gpus]
        d["accelerator"] = self.accelerator.value
        return d


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


@dataclass
class ModelSpec:
    """A catalog entry. Populated from ``catalog/data/models.yaml``."""

    id: str
    name: str
    publisher: str
    params_b: float                      # total parameters, in billions
    active_params_b: Optional[float] = None  # MoE: params active per token
    layers: int = 32
    hidden_size: int = 4096
    kv_heads: int = 8                    # GQA key/value heads (not query heads)
    head_dim: Optional[int] = None
    max_context: int = 8192
    license: str = "unknown"
    commercial_ok: bool = True
    tasks: List[str] = field(default_factory=list)
    quality_tier: int = 3                # 1 (weak) .. 5 (frontier-ish), within its size class
    supports_tools: bool = False
    supports_vision: bool = False
    supports_json: bool = False
    is_reasoning: bool = False
    is_embedding: bool = False
    languages: List[str] = field(default_factory=lambda: ["en"])
    gguf_repo: Optional[str] = None      # HF repo carrying GGUF quants
    ollama_name: Optional[str] = None    # tag pullable via `ollama pull`
    hf_repo: Optional[str] = None        # safetensors repo (vLLM / MLX)
    mlx_repo: Optional[str] = None
    chat_template: Optional[str] = None
    recommended_quants: List[str] = field(default_factory=lambda: ["Q4_K_M"])
    #: Models sharing a vocabulary. Speculative decoding requires the draft and
    #: target to tokenize identically, which publisher alone does not imply —
    #: a distill, for instance, inherits its base model's tokenizer.
    tokenizer_family: Optional[str] = None
    #: True for small models worth using as a speculative draft.
    draft_capable: bool = False
    notes: str = ""

    @property
    def family(self) -> str:
        """Vocabulary family, explicit where stated and derived otherwise."""
        if self.tokenizer_family:
            return self.tokenizer_family
        return self.id.split("-")[0]

    @property
    def effective_head_dim(self) -> int:
        return self.head_dim or max(1, self.hidden_size // 32)

    @property
    def is_moe(self) -> bool:
        return self.active_params_b is not None and self.active_params_b < self.params_b

    @property
    def compute_params_b(self) -> float:
        """Parameters that actually do work per token — drives speed, not memory."""
        return self.active_params_b or self.params_b

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


@dataclass
class MemoryEstimate:
    """Where the bytes go. All values in bytes."""

    weights: int
    kv_cache: int
    overhead: int
    context_tokens: int
    quant: str
    kv_quant: str = "f16"

    @property
    def total(self) -> int:
        return self.weights + self.kv_cache + self.overhead

    @property
    def total_gb(self) -> float:
        return self.total / GIB

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total"] = self.total
        return d


@dataclass
class Candidate:
    """A scored (model, quant) pairing with its memory budget worked out."""

    model: ModelSpec
    quant: str
    estimate: MemoryEstimate
    fits_on_gpu: bool
    gpu_fraction: float                  # 0.0-1.0 of layers that fit in VRAM
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.model.id}:{self.quant}"


@dataclass
class Plan:
    """Ranked candidates plus the runtime chosen to serve them."""

    candidates: List[Candidate]
    runtime: str
    runtime_reason: str = ""
    profile: Optional[UseCaseProfile] = None
    hardware: Optional[HardwareProfile] = None
    notes: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def fallbacks(self) -> List[Candidate]:
        return self.candidates[1:]


# --------------------------------------------------------------------------- #
# Runtime configuration
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeConfig:
    """Everything needed to actually launch the thing.

    ``extra`` holds runtime-specific knobs; the tuner fills it and the repair
    strategies mutate it. Kept as a dict so adding a runtime does not require
    changing this class.
    """

    runtime: str
    model_id: str
    quant: str
    context_tokens: int
    gpu_layers: int = 0
    total_layers: int = 32
    batch_size: int = 2048
    ubatch_size: int = 512
    threads: Optional[int] = None
    parallel: int = 1
    kv_cache_type: str = "f16"
    flash_attention: bool = False
    mmap: bool = True
    mlock: bool = False
    host: str = "127.0.0.1"
    port: int = 11435
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    min_p: float = 0.05
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    model_path: Optional[str] = None
    #: Vision models are two files; this is the multimodal projector that turns
    #: image embeddings into tokens the language model understands. Without it
    #: the endpoint accepts images and quietly ignores them.
    mmproj_path: Optional[str] = None
    served_name: Optional[str] = None
    #: Deep settings: sampler chain, reasoning budget, speculative decoding,
    #: tensor placement, RoPE scaling and serving behaviour. Typed rather than
    #: stuffed into `extra` so they are testable and can be reported on.
    advanced: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def full_gpu_offload(self) -> bool:
        return self.gpu_layers >= self.total_layers

    def copy(self) -> "RuntimeConfig":
        import copy as _copy

        return _copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.advanced is not None and hasattr(self.advanced, "to_dict"):
            d["advanced"] = self.advanced.to_dict()
        return d


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


class Severity(str, enum.Enum):
    """Whether a failing check should block the setup."""

    CRITICAL = "critical"   # setup is not usable
    REQUIRED = "required"   # the user explicitly asked for this capability
    ADVISORY = "advisory"   # nice to have; report but do not block


class CheckStatus(str, enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


@dataclass
class CheckResult:
    """Outcome of one verification check.

    ``failure_signature`` is the join key into the repair strategy table; it must
    be a stable, machine-readable token, never a human sentence.
    """

    name: str
    status: CheckStatus
    severity: Severity = Severity.ADVISORY
    message: str = ""
    duration_s: float = 0.0
    failure_signature: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (CheckStatus.PASS, CheckStatus.SKIP, CheckStatus.WARN)

    @property
    def blocking(self) -> bool:
        return self.status is CheckStatus.FAIL and self.severity in (
            Severity.CRITICAL,
            Severity.REQUIRED,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        return d


@dataclass
class RepairAction:
    """One entry in the repair journal — what was changed and what it cost."""

    strategy: str
    trigger: str
    description: str
    cost: str = "low"                  # low | medium | high quality/speed sacrifice
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    resolved: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SetupResult:
    """The end-to-end outcome, and the object the report is rendered from."""

    profile: UseCaseProfile
    hardware: HardwareProfile
    plan: Plan
    config: RuntimeConfig
    checks: List[CheckResult] = field(default_factory=list)
    repairs: List[RepairAction] = field(default_factory=list)
    registrations: List[Dict[str, Any]] = field(default_factory=list)
    #: Companion embedding model for RAG: what was set up, and whether it works.
    embedding: Optional[Dict[str, Any]] = None
    #: Machine-level adjustments (an OS limit, an env var) that llmfit will not
    #: apply on the user's behalf but that this setup wants.
    platform_actions: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @property
    def checks_ran(self) -> bool:
        """Whether anything was actually tested."""
        return any(c.status is not CheckStatus.SKIP for c in self.checks)

    @property
    def succeeded(self) -> bool:
        """Verified working.

        Requires that checks ran *and* none of them blocks. An empty check list
        is not success — it is an untested plan. The tool's central promise is
        that it never claims something works without a check backing it, and
        treating "nothing was tested" as a pass would break exactly that.
        """
        return self.checks_ran and not any(c.blocking for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "hardware": self.hardware.to_dict(),
            "runtime": self.plan.runtime,
            "runtime_reason": self.plan.runtime_reason,
            "model": self.plan.best.model.to_dict() if self.plan.best else None,
            "config": self.config.to_dict(),
            "estimate": self.plan.best.estimate.to_dict() if self.plan.best else None,
            "fallbacks": [c.label for c in self.plan.fallbacks],
            "checks": [c.to_dict() for c in self.checks],
            "repairs": [r.to_dict() for r in self.repairs],
            "registrations": self.registrations,
            "embedding": self.embedding,
            "platform_actions": self.platform_actions,
            "succeeded": self.succeeded,
            "verified": self.checks_ran,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
