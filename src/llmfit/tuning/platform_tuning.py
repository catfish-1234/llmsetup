"""Platform-specific settings the runtime cannot infer.

These are the adjustments that live outside the inference engine, an OS limit,
an environment variable, a memory policy, and they are the ones people most
often miss, because nothing in the engine's output points at them. A Mac that
refuses to load a model it has the RAM for, or an AMD card the runtime does not
recognise, both look like a broken setup rather than a missing sysctl.

Each item says what to run, what it changes, and whether it needs privileges.
Nothing here is applied automatically: these touch the machine outside the
scope of what the user asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llmfit.types import (
    GIB,
    Accelerator,
    HardwareProfile,
    RuntimeConfig,
    Task,
    UseCaseProfile,
)


@dataclass
class PlatformAction:
    """One machine-level adjustment, with its cost stated."""

    title: str
    reason: str
    command: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    needs_root: bool = False
    persists: bool = False
    #: True when skipping it merely leaves performance on the table, False when
    #: the setup will not work without it.
    optional: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title, "reason": self.reason, "command": self.command,
            "env": self.env, "needs_root": self.needs_root,
            "persists": self.persists, "optional": self.optional,
        }


def plan_platform(hardware: HardwareProfile, config: RuntimeConfig,
                  estimate_bytes: int = 0,
                  profile: Optional[UseCaseProfile] = None) -> List[PlatformAction]:
    """Machine-level adjustments worth making for this setup."""
    actions: List[PlatformAction] = []
    actions += _apple(hardware, estimate_bytes)
    actions += _amd(hardware)
    actions += _numa(hardware, config)
    actions += _nvidia(hardware)
    actions += _batch_invariance(hardware, config, profile)
    return actions


# --------------------------------------------------------------------------- #
# Apple Silicon
# --------------------------------------------------------------------------- #

#: Apple's guidance is to leave at least ~30% of unified memory to the system,
#: which is where the planner's default budget comes from.
_APPLE_SAFE_FRACTION = 0.70

#: What the wired limit can be raised to before the desktop starts suffering.
#: Beyond this the window server and page cache lose the memory they need.
_APPLE_MAX_FRACTION = 0.85


def _apple(hardware: HardwareProfile, estimate_bytes: int) -> List[PlatformAction]:
    if not hardware.unified_memory:
        return []

    default_budget = int(hardware.ram_bytes * _APPLE_SAFE_FRACTION)
    if estimate_bytes <= default_budget:
        return []

    # The model wants more than the default Metal working set allows. The GPU
    # driver allocates from a wired (non-pageable) pool whose cap is
    # conservative, and raising it is the difference between loading and not.
    ceiling = int(hardware.ram_bytes * _APPLE_MAX_FRACTION)
    target_mb = int(min(estimate_bytes * 1.08, ceiling) / (1024 ** 2))
    return [PlatformAction(
        title="Raise the Metal working-set limit",
        reason=(
            f"This model needs about {estimate_bytes / GIB:.1f} GB, more than the "
            f"~{default_budget / GIB:.1f} GB the GPU driver will wire by default. "
            f"On Apple Silicon the GPU allocates from a non-pageable pool with a "
            f"conservative cap, so the load fails with an out-of-memory error even "
            f"though the machine has the RAM. Raising the cap fixes that; going "
            f"much past {_APPLE_MAX_FRACTION:.0%} of RAM starves the window server "
            f"and makes the desktop unusable."
        ),
        command=f"sudo sysctl iogpu.wired_limit_mb={target_mb}",
        needs_root=True,
        persists=False,
        optional=False,
    )]


# --------------------------------------------------------------------------- #
# AMD
# --------------------------------------------------------------------------- #

#: Architectures ROCm recognises natively. Cards outside this set need to be
#: presented as a supported one, which is what the override does.
_AMD_NATIVE = re.compile(
    r"\b(?:7900|7800|7700|9070|9060|mi\d{3}|w7\d00|radeon pro w7)\b", re.I
)

#: Older consumer cards and their nearest supported architecture.
_AMD_OVERRIDES = [
    (re.compile(r"\b6[89]\d0\b|\brx 6[0-9]{3}\b", re.I), "10.3.0", "RDNA2"),
    (re.compile(r"\b780m\b|\b680m\b|\b760m\b", re.I), "11.0.0", "RDNA3 integrated"),
    (re.compile(r"\b5[567]00\b", re.I), "10.1.0", "RDNA1"),
]


def _amd(hardware: HardwareProfile) -> List[PlatformAction]:
    gpus = [g for g in hardware.gpus if g.vendor is Accelerator.AMD]
    if not gpus:
        return []

    name = gpus[0].name
    if _AMD_NATIVE.search(name):
        # Recent ROCm recognises these directly; the override used to be needed
        # and is now actively unhelpful.
        return []

    for pattern, version, label in _AMD_OVERRIDES:
        if pattern.search(name):
            return [PlatformAction(
                title="Tell ROCm which architecture to target",
                reason=(
                    f"{name} is {label}, which ROCm does not enable by default. "
                    f"Without this the runtime falls back to CPU and looks simply "
                    f"slow, with nothing to indicate the GPU went unused. The "
                    f"override presents the card as a supported architecture."
                ),
                env={"HSA_OVERRIDE_GFX_VERSION": version},
                command=f"export HSA_OVERRIDE_GFX_VERSION={version}",
                optional=False,
            )]

    return [PlatformAction(
        title="Check whether ROCm recognises this card",
        reason=(
            f"{name} is not one we recognise as natively supported or as needing a "
            f"specific override. If throughput comes out near CPU speed, the GPU is "
            f"probably not being used: set HSA_OVERRIDE_GFX_VERSION to the nearest "
            f"supported architecture for your card."
        ),
        command="rocminfo | grep -i gfx",
    )]


# --------------------------------------------------------------------------- #
# NUMA
# --------------------------------------------------------------------------- #


def _numa(hardware: HardwareProfile, config: RuntimeConfig) -> List[PlatformAction]:
    # Only meaningful for CPU-resident work on a multi-socket machine. Apple
    # Silicon's unified memory is a single pool and is not NUMA in this sense.
    if hardware.unified_memory or hardware.os != "linux":
        return []
    if config.gpu_layers >= config.total_layers and hardware.gpus:
        return []
    if (hardware.physical_cores or 0) < 32:
        return []

    return [PlatformAction(
        title="Spread CPU work across memory nodes",
        reason=(
            f"This machine has {hardware.physical_cores} physical cores, which "
            f"usually means more than one memory node, and part of this model runs "
            f"on the CPU. Without a NUMA policy the threads on one socket read "
            f"weights held on the other, across a slower interconnect. Distributing "
            f"the allocation avoids that."
        ),
        command="llama-server --numa distribute ...",
    )]


# --------------------------------------------------------------------------- #
# NVIDIA
# --------------------------------------------------------------------------- #


def _nvidia(hardware: HardwareProfile) -> List[PlatformAction]:
    gpus = [g for g in hardware.gpus if g.vendor is Accelerator.NVIDIA]
    if len(gpus) < 2:
        return []
    return [PlatformAction(
        title="Check peer-to-peer access between the GPUs",
        reason=(
            f"{len(gpus)} NVIDIA GPUs are present and the model is split across "
            f"them. If the cards cannot talk directly, every cross-GPU transfer "
            f"goes through system memory and the split costs more than it saves."
        ),
        command="nvidia-smi topo -m",
    )]


#: Batch-invariant kernels need tensor cores of this generation or newer.
_BATCH_INVARIANT_MIN_COMPUTE = 8.0


def _batch_invariance(hardware: HardwareProfile, config: RuntimeConfig,
                      profile) -> List[PlatformAction]:
    """Make the server give the same answer regardless of what it batched with.

    A fixed seed and temperature zero are not enough. Floating-point addition
    is not associative, so the same prompt accumulated in a different order,
    because it happened to share a batch with three other requests, can come out
    different. Batch-invariant kernels remove that, and it is the only thing
    that does.

    Offered rather than applied, because it costs real throughput: measured at
    roughly 60% longer on the same work. That is a fine trade for a regression
    suite and a poor one for an interactive endpoint, and only the person
    running it knows which they have.
    """
    if config.runtime != "vllm" or profile is None:
        return []
    if profile.primary_task not in _REPRODUCIBLE_TASKS:
        return []
    capable = [g for g in hardware.gpus
               if g.vendor is Accelerator.NVIDIA
               and _compute_capability(g) >= _BATCH_INVARIANT_MIN_COMPUTE]
    if not capable:
        return []
    return [PlatformAction(
        title="Make results reproducible under concurrent load",
        reason=(
            "A seed fixes the sampler, not the arithmetic. Two identical "
            "prompts can still differ when they are batched with different "
            "neighbours, because the additions happen in a different order. "
            "Batch-invariant kernels remove that, at roughly 60% longer per "
            "request, so it is worth it for a regression suite and not for an "
            "interactive endpoint."
        ),
        env={"VLLM_BATCH_INVARIANT": "1"},
    )]


def _compute_capability(gpu) -> float:
    try:
        return float(gpu.compute_capability)
    except (TypeError, ValueError):
        return 0.0


#: Where two runs disagreeing is a defect rather than variety.
_REPRODUCIBLE_TASKS = {Task.EXTRACT, Task.CODING, Task.TRANSLATE}


def describe(actions: List[PlatformAction]) -> List[str]:
    """One line per action, for the report."""
    lines = []
    for action in actions:
        prefix = "" if action.optional else "required: "
        lines.append(f"{prefix}{action.title}: {action.reason}")
    return lines
