"""GPU discovery across NVIDIA, AMD, Apple and Intel.

Each vendor gets its own probe because each exposes memory differently, and
memory is the number the whole planner depends on. A vendor probe that fails
returns an empty list and a warning rather than raising, because "no nvidia-smi"
is the normal case on most machines.
"""

from __future__ import annotations

import platform
import re
from typing import List, Tuple

from llmsetup.types import GPU, Accelerator
from llmsetup.util.proc import have, parse_csv_table, run

MIB = 1024 ** 2


def detect_gpus() -> Tuple[List[GPU], List[str]]:
    """Return (gpus, warnings). Probes every vendor; a machine may have several."""
    gpus: List[GPU] = []
    warnings: List[str] = []

    if _is_apple_silicon():
        found, warn = detect_apple()
        gpus.extend(found)
        warnings.extend(warn)
        # Apple's GPU is the whole machine; no point probing discrete vendors.
        return gpus, warnings

    for probe in (detect_nvidia, detect_amd, detect_intel):
        try:
            found, warn = probe()
        except Exception as exc:
            warnings.append(f"{probe.__name__} failed ({exc}); continuing other GPU probes")
            continue
        gpus.extend(found)
        warnings.extend(warn)

    for i, gpu in enumerate(gpus):
        gpu.index = i
    return gpus, warnings


# --------------------------------------------------------------------------- #
# NVIDIA
# --------------------------------------------------------------------------- #


def detect_nvidia() -> Tuple[List[GPU], List[str]]:
    if not have("nvidia-smi"):
        return [], []

    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=20,
    )
    if not result.ok:
        # A driver/library mismatch is worth surfacing: the card is there but unusable.
        return [], [f"nvidia-smi present but failed: {result.output.splitlines()[:1]}"]

    gpus: List[GPU] = []
    for row in parse_csv_table(result.stdout):
        if len(row) < 4:
            continue
        try:
            gpus.append(
                GPU(
                    index=int(row[0]),
                    name=row[1],
                    vendor=Accelerator.NVIDIA,
                    vram_bytes=int(float(row[2])) * MIB,
                    vram_free_bytes=int(float(row[3])) * MIB,
                    compute_capability=row[4] if len(row) > 4 and row[4] != "[N/A]" else None,
                    driver=row[5] if len(row) > 5 else None,
                )
            )
        except (ValueError, IndexError):
            continue
    if not gpus:
        return [], ["nvidia-smi returned no parseable GPUs"]
    return gpus, []


# --------------------------------------------------------------------------- #
# AMD
# --------------------------------------------------------------------------- #


def detect_amd() -> Tuple[List[GPU], List[str]]:
    if have("rocm-smi"):
        gpus, warn = _amd_via_rocm_smi()
        if gpus:
            return gpus, warn
    if have("amd-smi"):
        gpus, warn = _amd_via_amd_smi()
        if gpus:
            return gpus, warn
    return [], []


def _amd_via_rocm_smi() -> Tuple[List[GPU], List[str]]:
    result = run(["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--csv"], timeout=25)
    if not result.ok:
        return [], ["rocm-smi present but failed; treating AMD GPU as unavailable"]

    gpus: List[GPU] = []
    rows = [r for r in parse_csv_table(result.stdout) if r and r[0].lower().startswith("card")]
    header = next((r for r in parse_csv_table(result.stdout) if r and r[0].lower() == "device"), [])
    total_idx = _index_containing(header, "vram total memory")
    used_idx = _index_containing(header, "vram total used memory")
    name_idx = _index_containing(header, "card series") or _index_containing(header, "card model")

    for i, row in enumerate(rows):
        total = _safe_int(row, total_idx)
        if not total:
            continue
        used = _safe_int(row, used_idx)
        gpus.append(
            GPU(
                index=i,
                name=(row[name_idx] if name_idx is not None and name_idx < len(row) else "AMD GPU"),
                vendor=Accelerator.AMD,
                vram_bytes=total,
                vram_free_bytes=(total - used) if used is not None else None,
            )
        )
    return gpus, []


def _amd_via_amd_smi() -> Tuple[List[GPU], List[str]]:
    result = run(["amd-smi", "static", "--json"], timeout=25)
    if not result.ok:
        return [], []
    import json

    try:
        data = json.loads(result.stdout)
    except ValueError:
        return [], []
    entries = data if isinstance(data, list) else [data]

    gpus: List[GPU] = []
    for i, entry in enumerate(entries):
        vram = (entry.get("vram") or {}).get("size") or {}
        total_mb = vram.get("value")
        if not total_mb:
            continue
        name = (entry.get("asic") or {}).get("market_name") or "AMD GPU"
        gpus.append(
            GPU(index=i, name=name, vendor=Accelerator.AMD, vram_bytes=int(total_mb) * MIB)
        )
    return gpus, []


# --------------------------------------------------------------------------- #
# Apple Silicon
# --------------------------------------------------------------------------- #


def detect_apple() -> Tuple[List[GPU], List[str]]:
    """Apple Silicon has no separate VRAM: the GPU addresses system RAM.

    We still report a GPU so downstream code can reason about Metal, but the
    memory figure is the unified pool and :meth:`HardwareProfile.usable_accel_bytes`
    is what applies the safety fraction.
    """
    from llmsetup.hardware.cpu import cpu_name, memory_bytes

    total_ram, _ = memory_bytes()
    chip = cpu_name()
    cores = None
    result = run(["system_profiler", "SPDisplaysDataType"], timeout=25)
    if result.ok:
        match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
        if match:
            cores = int(match.group(1))
        chip_match = re.search(r"Chipset Model:\s*(.+)", result.stdout)
        if chip_match:
            chip = chip_match.group(1).strip()

    name = f"{chip}" + (f" ({cores}-core GPU)" if cores else "")
    return (
        [GPU(index=0, name=name, vendor=Accelerator.APPLE, vram_bytes=total_ram)],
        [],
    )


# --------------------------------------------------------------------------- #
# Intel
# --------------------------------------------------------------------------- #


def detect_intel() -> Tuple[List[GPU], List[str]]:
    """Intel Arc / Xe via xpu-smi. Integrated Xe shares system RAM, so we only
    report a discrete card when the tool gives us a real memory figure."""
    if not have("xpu-smi"):
        return [], []
    result = run(["xpu-smi", "discovery", "-j"], timeout=20)
    if not result.ok:
        return [], []
    import json

    try:
        data = json.loads(result.stdout)
    except ValueError:
        return [], []

    gpus: List[GPU] = []
    for i, device in enumerate(data.get("device_list", [])):
        mem_mb = device.get("memory_physical_size_byte")
        if mem_mb:
            vram = int(mem_mb)
        else:
            continue
        gpus.append(
            GPU(
                index=i,
                name=device.get("device_name", "Intel GPU"),
                vendor=Accelerator.INTEL,
                vram_bytes=vram,
            )
        )
    return gpus, []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _is_apple_silicon() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )


def _index_containing(header: List[str], needle: str):
    for i, cell in enumerate(header):
        if needle.lower() in cell.lower():
            return i
    return None


def _safe_int(row: List[str], idx):
    if idx is None or idx >= len(row):
        return None
    try:
        return int(re.sub(r"[^0-9]", "", row[idx]) or 0)
    except ValueError:
        return None
