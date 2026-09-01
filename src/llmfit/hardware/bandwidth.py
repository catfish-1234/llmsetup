"""Memory bandwidth lookup, and the throughput model built on it.

Token generation is memory-bandwidth bound, not compute bound: every token
requires streaming the active weights through the memory system once. So

    tokens/sec  ~=  memory_bandwidth / active_weight_bytes

is a first-order model that gets within a factor of ~1.3 of reality on most
setups, which is enough to rank candidates and to sanity-check the throughput
the verifier actually measures. The efficiency factor accounts for attention
overhead, kernel launch gaps and the fact that nobody hits peak bandwidth.

Table entries are peak theoretical bandwidth in GB/s. Unknown devices fall back
to a capacity-based heuristic rather than failing, and that fallback is reported.
"""

from __future__ import annotations

import re
from typing import Optional

from llmfit.types import GPU, Accelerator, HardwareProfile

GB = 1000 ** 3  # vendors quote bandwidth in decimal GB/s

# Fraction of peak bandwidth actually achieved by a well-tuned inference runtime.
DECODE_EFFICIENCY = 0.72

_NVIDIA = {
    "5090": 1792, "5080": 960, "5070 ti": 896, "5070": 672, "5060 ti": 448,
    "4090": 1008, "4080 super": 736, "4080": 717, "4070 ti super": 672,
    "4070 ti": 504, "4070 super": 504, "4070": 504, "4060 ti": 288, "4060": 272,
    "3090 ti": 1008, "3090": 936, "3080 ti": 912, "3080": 760, "3070": 448,
    "3060 ti": 448, "3060": 360, "3050": 224,
    "2080 ti": 616, "2080": 448, "2070": 448, "2060": 336,
    "a100": 1935, "h100": 3350, "h200": 4800, "b200": 8000,
    "l40s": 864, "l40": 864, "l4": 300, "a6000": 768, "a5000": 768,
    "a4000": 448, "a10": 600, "a40": 696, "v100": 900,
    "titan rtx": 672,
}

_AMD = {
    "7900 xtx": 960, "7900 xt": 800, "7900 gre": 576, "7800 xt": 624,
    "7700 xt": 432, "7600": 288, "9070 xt": 645, "9070": 645,
    "6900 xt": 512, "6800 xt": 512, "6800": 512, "6700 xt": 384,
    "mi300x": 5300, "mi250x": 3200, "mi210": 1600, "mi100": 1200,
    "w7900": 864, "w7800": 576,
}

# Apple Silicon: unified memory bandwidth is the whole story on these machines.
_APPLE = {
    "m1 ultra": 800, "m1 max": 400, "m1 pro": 200, "m1": 68,
    "m2 ultra": 800, "m2 max": 400, "m2 pro": 200, "m2": 100,
    "m3 ultra": 819, "m3 max": 400, "m3 pro": 150, "m3": 100,
    "m4 max": 546, "m4 pro": 273, "m4": 120,
    "m5 max": 600, "m5 pro": 300, "m5": 150,
}

_INTEL = {"a770": 560, "a750": 512, "a580": 512, "b580": 456, "b570": 380}


def gpu_bandwidth_bytes(gpu: GPU) -> Optional[int]:
    """Peak memory bandwidth for a GPU, or None if it isn't in the table."""
    name = gpu.name.lower()
    table = {
        Accelerator.NVIDIA: _NVIDIA,
        Accelerator.AMD: _AMD,
        Accelerator.APPLE: _APPLE,
        Accelerator.INTEL: _INTEL,
    }.get(gpu.vendor)
    if not table:
        return None

    # Longest key first so "4070 ti super" wins over "4070".
    for key in sorted(table, key=len, reverse=True):
        if key in name:
            return int(table[key] * GB)
    return None


def estimated_gpu_bandwidth_bytes(gpu: GPU) -> int:
    """Bandwidth with a capacity-based fallback for unrecognised cards."""
    known = gpu_bandwidth_bytes(gpu)
    if known:
        return known
    # Rough industry correlation: bigger cards ship wider buses.
    vram_gb = gpu.vram_gb
    if vram_gb >= 60:
        guess = 1500
    elif vram_gb >= 40:
        guess = 1000
    elif vram_gb >= 20:
        guess = 700
    elif vram_gb >= 12:
        guess = 400
    else:
        guess = 280
    return int(guess * GB)


def cpu_bandwidth_bytes(hw: HardwareProfile) -> int:
    """System RAM bandwidth, the ceiling for any layer that isn't offloaded."""
    if hw.unified_memory and hw.gpus:
        return estimated_gpu_bandwidth_bytes(hw.gpus[0])
    name = hw.cpu_name.lower()
    if re.search(r"\b(epyc|xeon|threadripper)\b", name):
        return int(150 * GB)     # many channels, server-class
    if re.search(r"ddr5|ryzen ai|core ultra", name):
        return int(80 * GB)
    return int(50 * GB)          # typical dual-channel DDR4 desktop/laptop


def estimate_tokens_per_second(
    hw: HardwareProfile,
    active_weight_bytes: int,
    gpu_fraction: float = 1.0,
    efficiency: float = DECODE_EFFICIENCY,
) -> float:
    """Predicted decode throughput.

    With partial offload the two tiers run in series, so the times add:
    the CPU-resident fraction dominates fast, which is why a model that "almost
    fits" is usually much worse than a smaller one that fits entirely.
    """
    if active_weight_bytes <= 0:
        return 0.0

    gpu_fraction = max(0.0, min(1.0, gpu_fraction))
    gpu_bw = estimated_gpu_bandwidth_bytes(hw.gpus[0]) if hw.gpus else 0
    cpu_bw = cpu_bandwidth_bytes(hw)

    seconds = 0.0
    if gpu_fraction > 0 and gpu_bw > 0:
        seconds += (active_weight_bytes * gpu_fraction) / (gpu_bw * efficiency)
    elif gpu_fraction > 0:
        seconds += (active_weight_bytes * gpu_fraction) / (cpu_bw * efficiency)
    if gpu_fraction < 1.0:
        seconds += (active_weight_bytes * (1.0 - gpu_fraction)) / (cpu_bw * efficiency)

    return 1.0 / seconds if seconds > 0 else 0.0


def describe_bandwidth(hw: HardwareProfile) -> str:
    if hw.gpus:
        gpu = hw.gpus[0]
        known = gpu_bandwidth_bytes(gpu)
        value = known or estimated_gpu_bandwidth_bytes(gpu)
        suffix = "" if known else " (estimated)"
        return f"{value / GB:.0f} GB/s{suffix}"
    return f"{cpu_bandwidth_bytes(hw) / GB:.0f} GB/s (system RAM, estimated)"
