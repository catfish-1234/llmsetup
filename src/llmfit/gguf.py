"""Minimal GGUF metadata reader.

Catalog architecture figures are curated, and for brand-new models they are
estimated. A GGUF file states the truth in its header, so once weights are on
disk we read the real block count, KV head count and embedding width and
recompute the memory plan against those. This is what stops a wrong catalog
guess from turning into an OOM at inference time.

Only the header is read — never the tensor data — so this is fast even on a
70 GB file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
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
    tensor_count: int = 0
    metadata: Dict[str, Any] = None

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
    def supports_tools(self) -> Optional[bool]:
        """Infer tool support from the chat template, which is where it lives."""
        if not self.chat_template:
            return None
        lowered = self.chat_template.lower()
        return "tool" in lowered or "function" in lowered


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def raw(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("unexpected end of GGUF header")
        return data

    def scalar(self, vtype: int) -> Any:
        fmt, size = _FIXED[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        length = struct.unpack("<Q", self.raw(8))[0]
        if length > 32 * 1024 * 1024:
            raise ValueError("implausible string length in GGUF header")
        return self.raw(length).decode("utf-8", "replace")

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
            data = self.raw(size * count)
            if count <= _MAX_ARRAY_MATERIALISE:
                fmt, _ = _FIXED[item_type]
                return list(struct.unpack("<" + fmt[1] * count, data))
            return f"<{count} values>"

        if item_type == STRING:
            if count <= _MAX_ARRAY_MATERIALISE:
                return [self.string() for _ in range(count)]
            # Skip a large string array (the vocabulary) without decoding it.
            for _ in range(count):
                length = struct.unpack("<Q", self.raw(8))[0]
                self.handle.seek(length, 1)
            return f"<{count} strings>"

        raise ValueError(f"unsupported GGUF array item type {item_type}")


def read_gguf_info(path: Path) -> Optional[GgufInfo]:
    """Read a GGUF header. Returns None if the file is not GGUF or is unreadable.

    Never raises on a malformed file: a corrupt download should degrade to
    "use the catalog estimate", not crash the setup.
    """
    try:
        with open(path, "rb", buffering=1024 * 256) as handle:
            if handle.read(4) != MAGIC:
                return None
            struct.unpack("<I", handle.read(4))[0]                  # version
            tensor_count = struct.unpack("<Q", handle.read(8))[0]
            kv_count = struct.unpack("<Q", handle.read(8))[0]
            if kv_count > 100_000:
                return None

            reader = _Reader(handle)
            metadata: Dict[str, Any] = {}
            for _ in range(kv_count):
                key = reader.string()
                vtype = struct.unpack("<I", reader.raw(4))[0]
                metadata[key] = reader.value(vtype)
    except (OSError, ValueError, struct.error):
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
