"""Byte/number formatting and parsing."""

from __future__ import annotations

import re
from typing import Optional

KIB = 1024
MIB = 1024 ** 2
GIB = 1024 ** 3

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgt]i?b?)?\s*$", re.IGNORECASE)
_MULTIPLIERS = {
    "": 1, "b": 1,
    "k": KIB, "kb": KIB, "kib": KIB,
    "m": MIB, "mb": MIB, "mib": MIB,
    "g": GIB, "gb": GIB, "gib": GIB,
    "t": GIB * 1024, "tb": GIB * 1024, "tib": GIB * 1024,
}


def human_bytes(n: Optional[float]) -> str:
    """Format bytes for humans. 0 and None are distinguishable."""
    if n is None:
        return "unknown"
    n = float(n)
    for unit, size in (("TiB", GIB * 1024), ("GiB", GIB), ("MiB", MIB), ("KiB", KIB)):
        if abs(n) >= size:
            return f"{n / size:.1f} {unit}"
    return f"{int(n)} B"


def gb(n: Optional[float]) -> str:
    """Format bytes as GiB with one decimal — the unit people think in."""
    return "unknown" if n is None else f"{n / GIB:.1f} GB"


def parse_size(text: str) -> Optional[int]:
    """Parse '8', '8G', '8 GiB', '512MB' into bytes. Bare numbers mean GiB."""
    if text is None:
        return None
    match = _SIZE_RE.match(str(text))
    if not match:
        return None
    value, suffix = match.group(1), (match.group(2) or "").lower()
    if not suffix:
        return int(float(value) * GIB)
    return int(float(value) * _MULTIPLIERS.get(suffix, 1))


def human_tokens(n: int) -> str:
    """32768 -> '32K'. Context lengths read better in K."""
    if n >= 1_000_000 and n % 1_048_576 == 0:
        return f"{n // 1_048_576}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
