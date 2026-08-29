"""Shared fixtures.

Every test runs against a temporary LLMFIT_HOME so nothing touches the
developer's real model store, LM Studio folder, or saved profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmfit.catalog.catalog import load_catalog
from llmfit.types import (
    GIB,
    GPU,
    Accelerator,
    HardwareProfile,
    Priority,
    Task,
    UseCaseProfile,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMFIT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LLMFIT_MODEL_STORE", str(tmp_path / "models"))
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(tmp_path / "lmstudio"))
    monkeypatch.setenv("JAN_DATA_DIR", str(tmp_path / "jan"))
    monkeypatch.delenv("LLMFIT_CATALOG", raising=False)
    return tmp_path


@pytest.fixture
def catalog():
    return load_catalog(include_user_dir=False)


def make_hardware(vram_gb=24, ram_gb=64, vendor=Accelerator.NVIDIA,
                  name="NVIDIA GeForce RTX 4090", unified=False, cores=8,
                  os_name="linux", compute="8.9"):
    gpus = []
    if vram_gb:
        gpus = [GPU(0, name, vendor, int(vram_gb * GIB), compute_capability=compute)]
    return HardwareProfile(
        os=os_name, arch="x86_64", cpu_name="Test CPU", physical_cores=cores,
        logical_cores=cores * 2, ram_bytes=int(ram_gb * GIB),
        unified_memory=unified, gpus=gpus, disk_free_bytes=500 * GIB,
    )


@pytest.fixture
def gpu_24gb():
    return make_hardware()


@pytest.fixture
def gpu_12gb():
    return make_hardware(vram_gb=12, ram_gb=32, name="NVIDIA GeForce RTX 3060",
                         compute="8.6")


@pytest.fixture
def cpu_only():
    return make_hardware(vram_gb=0, ram_gb=16, vendor=None, cores=4)


@pytest.fixture
def apple_silicon():
    return make_hardware(vram_gb=64, ram_gb=64, vendor=Accelerator.APPLE,
                         name="Apple M3 Max (40-core GPU)", unified=True,
                         os_name="darwin")


@pytest.fixture
def chat_profile():
    return UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)


@pytest.fixture
def coding_profile():
    return UseCaseProfile(tasks=[Task.CODING, Task.AGENT], needs_tools=True,
                          context_tokens=32768, priority=Priority.BALANCED,
                          commercial_use=True)
