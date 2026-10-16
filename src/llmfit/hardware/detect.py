"""Assemble a :class:`HardwareProfile` from the individual probes."""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Optional

from llmfit.hardware import cpu as cpu_probe
from llmfit.hardware.gpu import detect_gpus
from llmfit.types import GIB, GPU, Accelerator, HardwareProfile
from llmfit.util.fs import free_bytes, model_store
from llmfit.util.units import parse_size


def detect_hardware(probe_gpus: bool = True) -> HardwareProfile:
    """Describe the current machine. Never raises; gaps become warnings."""
    warnings = []

    physical, logical = cpu_probe.core_counts()
    ram_total, ram_available = cpu_probe.memory_bytes()
    if not ram_total:
        warnings.append("could not read system RAM; assuming 8 GB (override with --ram)")
        ram_total = 8 * GIB

    gpus: List[GPU] = []
    gpu_warnings: List[str] = []
    if probe_gpus:
        try:
            gpus, gpu_warnings = detect_gpus()
        except Exception as exc:
            gpu_warnings = [f"GPU detection failed ({exc}); planning for CPU only"]
    warnings.extend(gpu_warnings)

    unified = cpu_probe.is_apple_silicon()
    if not gpus and not unified:
        warnings.append("no GPU detected; planning for CPU inference")

    profile = HardwareProfile(
        os=platform.system().lower(),
        arch=cpu_probe.normalized_arch(),
        cpu_name=cpu_probe.cpu_name(),
        physical_cores=physical,
        logical_cores=logical,
        ram_bytes=ram_total,
        ram_available_bytes=ram_available,
        unified_memory=unified,
        gpus=gpus,
        disk_free_bytes=free_bytes(model_store()),
        detection_warnings=warnings,
    )
    return profile


def apply_overrides(profile: HardwareProfile, overrides: Dict[str, Any]) -> HardwareProfile:
    """Apply user-supplied corrections (``--vram 24 --ram 64 --gpu-vendor nvidia``).

    Detection is heuristic and people know their own machines. Overrides are
    applied verbatim, and recorded as warnings so the report shows the plan was
    made against asserted rather than measured numbers.
    """
    updated = profile

    ram = _as_bytes(overrides.get("ram"))
    if ram:
        updated.ram_bytes = ram
        updated.ram_available_bytes = None
        updated.detection_warnings.append(f"RAM overridden to {ram / GIB:.0f} GB by user")

    vram = _as_bytes(overrides.get("vram"))
    vendor_name = overrides.get("gpu_vendor")
    if vram is not None:
        vendor = _vendor(vendor_name) or (updated.gpus[0].vendor if updated.gpus else Accelerator.NVIDIA)
        if vram == 0:
            updated.gpus = []
            updated.detection_warnings.append("GPU disabled by user; planning for CPU only")
        elif updated.gpus:
            updated.gpus[0].vram_bytes = vram
            updated.gpus[0].vram_free_bytes = None
            updated.gpus[0].vendor = vendor
            updated.gpus = updated.gpus[:1]
            updated.detection_warnings.append(f"VRAM overridden to {vram / GIB:.0f} GB by user")
        else:
            updated.gpus = [
                GPU(index=0, name=overrides.get("gpu_name") or f"{vendor.value} GPU (declared)",
                    vendor=vendor, vram_bytes=vram)
            ]
            updated.detection_warnings.append(
                f"GPU declared by user: {vram / GIB:.0f} GB {vendor.value}"
            )
    elif vendor_name and updated.gpus:
        declared_vendor = _vendor(vendor_name)
        if declared_vendor:
            for gpu in updated.gpus:
                gpu.vendor = declared_vendor

    if overrides.get("cpu_only"):
        updated.gpus = []
        updated.unified_memory = False
        updated.detection_warnings.append("--cpu-only: ignoring any GPU")

    cores = overrides.get("threads")
    if cores:
        updated.physical_cores = int(cores)

    return updated


def _as_bytes(value) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Bare numbers are GB, the unit people speak in.
        return int(value * GIB) if value < 4096 else int(value)
    return parse_size(str(value))


def _vendor(name: Optional[str]) -> Optional[Accelerator]:
    if not name:
        return None
    try:
        return Accelerator(str(name).lower())
    except ValueError:
        return None


def summarize(profile: HardwareProfile) -> str:
    """One-line summary used in reports and log headers."""
    from llmfit.util.units import gb

    parts = [f"{profile.os}/{profile.arch}", profile.cpu_name]
    if profile.physical_cores:
        parts.append(f"{profile.physical_cores}C")
    parts.append(f"{gb(profile.ram_bytes)} RAM")
    if profile.unified_memory:
        parts.append("unified memory")
    elif profile.gpus:
        parts.append(", ".join(f"{g.name} {gb(g.vram_bytes)}" for g in profile.gpus))
    else:
        parts.append("no GPU")
    return " | ".join(parts)
