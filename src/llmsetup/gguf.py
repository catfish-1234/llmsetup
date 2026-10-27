"""Minimal GGUF metadata reader.

Catalog architecture figures are curated, and for brand-new models they are
estimated. A GGUF file states the truth in its header, so once weights are on
disk we read the real block count, KV head count and embedding width and
recompute the memory plan against those. This is what stops a wrong catalog
guess from turning into an OOM at inference time.

Only the header is read, never the tensor data, so this is fast even on a
70 GB file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Dict, Optional

MAGIC = b"GGUF"

# GGUF metadata value types.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

# Arrays longer than this are skipped without materialising them: the token
# vocabulary is ~150k strings and nothing here needs it.
_MAX_ARRAY_MATERIALISE = 64


@dataclass
class GgufInfo:
    """Architecture facts read out of a GGUF header."""

    architecture: str = "unknown"
    layers: Optional[int] = None
    kv_heads: Optional[int] = None
    attention_heads: Optional[int] = None
    embedding_length: Optional[int] = None
    key_length: Optional[int] = None          # explicit head_dim when present
    context_length: Optional[int] = None
    rope_freq_base: Optional[float] = None
    file_type: Optional[int] = None
    expert_count: Optional[int] = None
    chat_template: Optional[str] = None
    #: Sliding-window span in tokens, when the architecture has one.
    sliding_window: Optional[int] = None
    #: One full-attention layer every N, the rest recurrent. This is how
    #: llama.cpp itself describes a hybrid model, so it is the right thing to
    #: read rather than inferring the shape from the model's name.
    full_attention_interval: Optional[int] = None
    #: State-space dimensions for the recurrent layers, when present.
    ssm_inner_size: Optional[int] = None
    ssm_state_size: Optional[int] = None
    #: Layers of multi-token prediction heads built into this file. These ship
    #: in some builds of a model and not others, so the model's name settles
    #: nothing and only the file can answer.
    nextn_predict_layers: Optional[int] = None
    tensor_count: int = 0
    metadata: Dict[str, Any] = dataclass_field(default_factory=dict)

    @property
    def head_dim(self) -> Optional[int]:
        if self.key_length:
            return self.key_length
        if self.embedding_length and self.attention_heads:
            return self.embedding_length // self.attention_heads
        return None

    @property
    def is_moe(self) -> bool:
        return bool(self.expert_count and self.expert_count > 1)

    @property
    def has_mtp(self) -> bool:
        return bool(self.nextn_predict_layers and self.nextn_predict_layers > 0)

    @property
    def is_hybrid(self) -> bool:
        return bool(self.full_attention_interval and self.full_attention_interval > 1)

    @property
    def recurrent_state_bytes(self) -> int:
        """Bytes of state one recurrent layer holds, when the file says enough.

        Derived only from dimensions the file actually states. Where it does not
        state them this returns 0 rather than a guess: the state is small next
        to the weights and constant in the context length, and the overhead term
        already errs high, so a missing figure costs far less than an invented
        one would.
        """
        if not (self.ssm_inner_size and self.ssm_state_size):
            return 0
        # Recurrent state is kept at full precision by every implementation that
        # ships one, because it is small and it accumulates.
        return int(self.ssm_inner_size * self.ssm_state_size * 4)

    @property
    def supports_tools(self) -> Optional[bool]:
        """Infer tool support from the chat template, which is where it lives."""
        if not self.chat_template:
            return None
        lowered = self.chat_template.lower()
        return "tool" in lowered or "function" in lowered


class _Reader:
    """Bounds-checked reader over a GGUF header.

    Every length in the header is an attacker-controlled uint64. Reading one
    without checking it against the real file size lets a 58-byte file ask for a
    2^61-byte allocation, which raises OverflowError or MemoryError rather than
    the ValueError the caller is prepared for. So sizes are validated against
    the bytes that actually exist before any allocation happens.
    """

    def __init__(self, handle, file_size: int):
        self.handle = handle
        self.file_size = file_size

    def _check(self, size: int) -> None:
        if size < 0 or size > self.file_size:
            raise ValueError(
                f"GGUF header declares {size} bytes but the file is {self.file_size}"
            )

    def raw(self, size: int) -> bytes:
        self._check(size)
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("unexpected end of GGUF header")
        return data

    def skip(self, size: int) -> None:
        self._check(size)
        self.handle.seek(size, 1)

    def scalar(self, vtype: int) -> Any:
        fmt, size = _FIXED[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self.raw(8))[0]
        if length > 32 * 1024 * 1024:
            raise ValueError("implausible string length in GGUF header")
        return self.raw(length).decode("utf-8", "replace")

    def _checked_count(self, count: int, item_bytes: int) -> int:
        """Reject an array whose declared size cannot fit in the file."""
        if count < 0 or (item_bytes and count > self.file_size // max(1, item_bytes)):
            raise ValueError(
                f"GGUF array declares {count} items, more than the file can hold"
            )
        return count

    def value(self, vtype: int) -> Any:
        if vtype in _FIXED:
            return self.scalar(vtype)
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            return self.array()
        raise ValueError(f"unknown GGUF value type {vtype}")

    def array(self) -> Any:
        item_type = struct.unpack("<I", self.raw(4))[0]
        count = struct.unpack("<Q", self.raw(8))[0]

        if item_type in _FIXED:
            _, size = _FIXED[item_type]
            self._checked_count(count, size)
            data = self.raw(size * count)
            if count <= _MAX_ARRAY_MATERIALISE:
                fmt, _ = _FIXED[item_type]
                return list(struct.unpack("<" + fmt[1] * count, data))
            return f"<{count} values>"

        if item_type == STRING:
            # Each string costs at least its 8-byte length prefix, so the count
            # is bounded by the file size even before the contents.
            self._checked_count(count, 8)
            if count <= _MAX_ARRAY_MATERIALISE:
                return [self.string() for _ in range(count)]
            # Skip a large string array (the vocabulary) without decoding it.
            for _ in range(count):
                length = struct.unpack("<Q", self.raw(8))[0]
                self.skip(length)
            return f"<{count} strings>"

        raise ValueError(f"unsupported GGUF array item type {item_type}")


def read_gguf_info(path: Path) -> Optional[GgufInfo]:
    """Read a GGUF header. Returns None if the file is not GGUF or is unreadable.

    Never raises on a malformed file: a corrupt download should degrade to
    "use the catalog estimate", not crash the setup.
    """
    try:
        file_size = path.stat().st_size
        with open(path, "rb", buffering=1024 * 256) as handle:
            if handle.read(4) != MAGIC:
                return None
            struct.unpack("<I", handle.read(4))[0]                  # version
            tensor_count = struct.unpack("<Q", handle.read(8))[0]
            kv_count = struct.unpack("<Q", handle.read(8))[0]
            if kv_count > 100_000:
                return None

            reader = _Reader(handle, file_size)
            metadata: Dict[str, Any] = {}
            for _ in range(kv_count):
                key = reader.string()
                vtype = struct.unpack("<I", reader.raw(4))[0]
                metadata[key] = reader.value(vtype)
    except (OSError, ValueError, struct.error, OverflowError, MemoryError):
        # A corrupt or hostile file must degrade to "use the catalog estimate",
        # never take down the setup run.
        return None

    arch = str(metadata.get("general.architecture", "unknown"))

    def get(suffix: str):
        return metadata.get(f"{arch}.{suffix}")

    info = GgufInfo(
        architecture=arch,
        layers=_as_int(get("block_count")),
        kv_heads=_as_int(get("attention.head_count_kv")),
        attention_heads=_as_int(get("attention.head_count")),
        embedding_length=_as_int(get("embedding_length")),
        key_length=_as_int(get("attention.key_length")),
        context_length=_as_int(get("context_length")),
        rope_freq_base=_as_float(get("rope.freq_base")),
        file_type=_as_int(metadata.get("general.file_type")),
        expert_count=_as_int(get("expert_count")),
        chat_template=_as_str(metadata.get("tokenizer.chat_template")),
        sliding_window=_as_int(get("attention.sliding_window")),
        full_attention_interval=_as_int(get("full_attention_interval")),
        ssm_inner_size=_as_int(get("ssm.inner_size")),
        ssm_state_size=_as_int(get("ssm.state_size")),
        nextn_predict_layers=_as_int(get("nextn_predict_layers")),
        tensor_count=tensor_count,
        metadata=metadata,
    )
    return info


def apply_to_spec(spec, info: Optional[GgufInfo]):
    """Correct a :class:`ModelSpec` with facts from the real file.

    Returns ``(updated_spec, [corrections])``. Only fields the file actually
    states are touched, and each change is reported so the user can see the plan
    was revised against ground truth.
    """
    from dataclasses import replace

    if info is None:
        return spec, []

    changes: Dict[str, Any] = {}
    corrections = []

    for field, value, label in (
        ("layers", info.layers, "layer count"),
        ("kv_heads", info.kv_heads, "KV head count"),
        ("hidden_size", info.embedding_length, "hidden size"),
        ("head_dim", info.head_dim, "head dimension"),
    ):
        if value and getattr(spec, field) != value:
            corrections.append(f"{label}: catalog said {getattr(spec, field)}, file says {value}")
            changes[field] = value

    # Multi-token prediction heads are a property of the build, not the model:
    # publishers ship an MTP variant alongside a plain one under names that
    # differ by a suffix. Only the file settles it, in both directions.
    if info.has_mtp != spec.mtp_heads:
        corrections.append(
            "multi-token prediction heads: "
            + ("present in this build, so the model can draft for itself"
               if info.has_mtp else
               "absent from this build, so a separate draft model is needed")
        )
        changes["mtp_heads"] = info.has_mtp

    # Most of a hybrid model's layers are recurrent and hold a fixed state
    # rather than a cache that grows. Missing this over-states the memory by
    # the interval, several times over, and rules the model out of plans it
    # fits comfortably.
    if info.is_hybrid and info.full_attention_interval != spec.full_attention_interval:
        corrections.append(
            f"hybrid attention: one full-attention layer every "
            f"{info.full_attention_interval}, the rest recurrent"
        )
        changes["full_attention_interval"] = info.full_attention_interval
    state_bytes = info.recurrent_state_bytes
    if state_bytes and state_bytes != spec.recurrent_state_bytes:
        changes["recurrent_state_bytes"] = state_bytes

    # A sliding window changes what the cache costs by several times over, so
    # reading it from the file rather than inferring it from the family is the
    # difference between a context that fits and one that does not.
    if info.sliding_window and info.sliding_window != spec.sliding_window:
        corrections.append(
            f"sliding-window attention: {info.sliding_window} tokens, "
            f"read from the file"
        )
        changes["sliding_window"] = info.sliding_window
        if spec.swa_pattern <= 1:
            # The file states the window but not the interleave. Assume every
            # layer is windowed, which is the conservative reading: it claims
            # the smallest cache saving consistent with what the file says.
            changes["swa_pattern"] = spec.layers

    # A GGUF's stated context length is the trained maximum; trust it over the
    # catalog only when it is *lower*, since repos often omit RoPE extension.
    if info.context_length and info.context_length < spec.max_context:
        corrections.append(
            f"max context: catalog said {spec.max_context}, file says {info.context_length}"
        )
        changes["max_context"] = info.context_length

    return (replace(spec, **changes) if changes else spec), corrections


def _as_int(value) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _as_str(value) -> Optional[str]:
    return value if isinstance(value, str) else None
