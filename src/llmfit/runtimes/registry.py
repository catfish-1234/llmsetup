"""Runtime selection.

The decision is made in this order, because each rule dominates the ones below:

1. The engine must exist for the platform (no CUDA-only vLLM on a Mac).
2. The use case may demand one (real concurrency wants a batching server).
3. Prefer what the user already has installed, a working setup they recognise
   beats a marginally better one they have to install.
4. Fall back to the easiest thing to install.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from llmfit.runtimes.base import Runtime
from llmfit.types import Accelerator, Client, HardwareProfile, UseCaseProfile

_REGISTRY: Dict[str, Runtime] = {}
#: Whether the adapter modules have been imported. A separate flag rather than
#: "is the registry non-empty", because importing any single adapter directly
#: registers that one and would otherwise convince the loader its work was
#: already done, leaving every other engine invisible for the rest of the
#: process.
_LOADED = False


def register(runtime: Runtime) -> Runtime:
    _REGISTRY[runtime.name] = runtime
    return runtime


def get_runtime(name: str) -> Runtime:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown runtime {name!r}; known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name]


def all_runtimes() -> List[Runtime]:
    _ensure_loaded()
    return list(_REGISTRY.values())


def available_runtimes() -> List[Runtime]:
    return [r for r in all_runtimes() if r.is_available()]


def _ensure_loaded() -> None:
    global _LOADED

    if _LOADED:
        return
    # Imported here to avoid a circular import at module load.
    from llmfit.runtimes import (  # noqa: F401
        llamacpp,
        lmstudio,
        mlx,
        ollama,
        tabby,
        vllm,
    )
    _LOADED = True


def choose_runtime(
    hardware: HardwareProfile, profile: UseCaseProfile
) -> Tuple[str, str]:
    """Return (runtime name, human-readable reason)."""
    _ensure_loaded()
    installed = {r.name for r in available_runtimes()}

    # 2. Concurrency demands a real batching server, and vLLM is the only one of
    #    these built for it, but only where it can actually run.
    if profile.concurrent_users >= 4 and hardware.os == "linux" and \
            hardware.accelerator is Accelerator.NVIDIA:
        if "vllm" in installed:
            return "vllm", (
                f"serving {profile.concurrent_users} concurrent users on NVIDIA/Linux; "
                "vLLM's continuous batching is built for this and is already installed"
            )
        return "vllm", (
            f"serving {profile.concurrent_users} concurrent users on NVIDIA/Linux; "
            "vLLM's continuous batching handles this far better than single-stream engines"
        )

    # 3. Prefer something already installed, in order of how well it fits.
    preference = _preference_order(hardware, profile)
    for name in preference:
        if name in installed:
            return name, f"{_label(name)} is already installed and suits this machine"

    # 4. Nothing installed: recommend the easiest good option.
    fallback = preference[0]
    return fallback, (
        f"no inference engine detected; {_label(fallback)} is the simplest to install "
        f"for {_describe_machine(hardware)}"
    )


def _preference_order(hardware: HardwareProfile, profile: UseCaseProfile) -> List[str]:
    """Ranked runtimes for this machine, best first."""
    wants_lmstudio = Client.LM_STUDIO in profile.clients
    wants_llamacpp = Client.LLAMA_CPP in profile.clients

    if hardware.unified_memory:
        # Apple Silicon: MLX is fastest for dense models, but Ollama/llama.cpp
        # have far broader model coverage and are what most Mac users have.
        order = ["ollama", "llamacpp", "mlx", "lmstudio"]
    elif hardware.accelerator is Accelerator.NVIDIA:
        order = ["ollama", "llamacpp", "vllm", "tabby", "lmstudio"]
    elif hardware.accelerator in (Accelerator.AMD, Accelerator.INTEL):
        # llama.cpp's Vulkan/ROCm backends are the most reliable path here.
        order = ["llamacpp", "ollama", "lmstudio"]
    else:
        order = ["ollama", "llamacpp", "lmstudio"]

    # Honour an explicit client preference by promoting it.
    if wants_llamacpp:
        order.insert(0, order.pop(order.index("llamacpp")))
    elif wants_lmstudio and "lmstudio" in order:
        order.insert(0, order.pop(order.index("lmstudio")))
    return order


def _label(name: str) -> str:
    try:
        return get_runtime(name).label
    except KeyError:
        return name


def _describe_machine(hardware: HardwareProfile) -> str:
    if hardware.unified_memory:
        return "Apple Silicon"
    if hardware.gpus:
        return f"{hardware.accelerator.value.upper()} GPUs"
    return "CPU-only inference"
