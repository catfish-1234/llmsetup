"""Assemble a :class:`HardwareProfile` from the individual probes."""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Optional

from llmsetup.hardware import cpu as cpu_probe
from llmsetup.hardware.gpu import detect_gpus
from llmsetup.types import GIB, GPU, Accelerator, HardwareProfile
from llmsetup.util.fs import free_bytes, model_store
from llmsetup.util.units import parse_size


def detect_hardware(probe_gpus: bool = True) -> HardwareProfile:
    """Describe the current machine. Never raises; gaps become warnings."""
    warnings = []

    def probe(label, function, default):
        try:
            return function()
        except Exception as exc:
            warnings.append(f"could not detect {label} ({exc}); using {default!r}")
            return default

    physical, logical = probe("CPU cores", cpu_probe.core_counts, (1, 1))
    ram_total, ram_available = probe("RAM", cpu_probe.memory_bytes, (0, None))
    if not ram_total:
        warnings.append("could not read system RAM; conservatively assuming 2 GB")
        ram_total = 2 * GIB
    if ram_available is None:
        warnings.append("available RAM unknown; reserving 25% of total RAM for other work")
    if not physical or not logical:
        warnings.append("CPU core count incomplete; using one core where unknown")

    gpus: List[GPU] = []
    gpu_warnings: List[str] = []
    if probe_gpus:
        try:
            gpus, gpu_warnings = detect_gpus()
        except Exception as exc:
            gpu_warnings = [f"GPU detection failed ({exc}); planning for CPU only"]
    warnings.extend(gpu_warnings)

    unified = probe("unified memory", cpu_probe.is_apple_silicon, False)
    usable = []
    for gpu in gpus:
        if gpu.vram_bytes <= 0:
            warnings.append(f"{gpu.name}: VRAM unknown; using CPU instead")
            continue
        if gpu.vram_free_bytes is None and not unified:
            gpu.vram_free_bytes = int(gpu.vram_bytes * 0.8)
            warnings.append(f"{gpu.name}: free VRAM unknown; reserving 20%")
        if gpu.vendor is Accelerator.NVIDIA and not gpu.compute_capability:
            warnings.append(f"{gpu.name}: CUDA capability unknown; disabling flash attention initially")
        usable.append(gpu)
    gpus = usable
    from llmsetup.runtimes.registry import scan_runtimes

    inventory, runtime_warnings = scan_runtimes()
    warnings.extend(runtime_warnings)
    if not gpus and not unified:
        warnings.append("no GPU detected; planning for CPU inference")

    profile = HardwareProfile(
        os=probe("OS", platform.system, "unknown").lower(),
        arch=probe("architecture", cpu_probe.normalized_arch, "unknown"),
        cpu_name=probe("CPU name", cpu_probe.cpu_name, "unknown"),
        physical_cores=physical or 1,
        logical_cores=logical or physical or 1,
        ram_bytes=ram_total,
        ram_available_bytes=ram_available,
        unified_memory=unified,
        gpus=gpus,
        disk_free_bytes=probe("free disk space", lambda: free_bytes(model_store()), 0),
        runtime_availability=inventory,
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
    from llmsetup.util.units import gb

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
