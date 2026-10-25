"""WSL driver discovery uses only Linux tools and a subprocess-local environment."""

import os

import pytest

from llmsetup.hardware import gpu
from llmsetup.types import Accelerator
from llmsetup.util.proc import CommandResult


@pytest.fixture
def wsl_layout(monkeypatch, tmp_path):
    driver_dir = tmp_path / "usr/lib/wsl/lib"
    driver_dir.mkdir(parents=True)
    smi = driver_dir / "nvidia-smi"
    smi.write_text("fixture executable; never executed")
    smi.chmod(0o755)
    (driver_dir / "libcuda.so.1").write_text("fixture library; never loaded")
    monkeypatch.setattr(gpu, "_WSL_LIB_DIR", driver_dir)
    monkeypatch.setattr(gpu.platform, "system", lambda: "Linux")
    monkeypatch.setattr(gpu.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    clean_path = tmp_path / "empty-bin"
    clean_path.mkdir()
    monkeypatch.setenv("PATH", str(clean_path))
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    return driver_dir


@pytest.mark.parametrize("library_path", [None, "", "/existing/cuda/lib"])
@pytest.mark.parametrize("release", ["6.6.87.2-microsoft-standard-WSL2", "4.4.0-19041-Microsoft"])
def test_wsl_nvidia_without_path_uses_driver_directory(monkeypatch, wsl_layout, library_path, release):
    monkeypatch.setattr(gpu.platform, "release", lambda: release)
    if library_path is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", library_path)
    environment = dict(os.environ)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[0] == str(wsl_layout / "nvidia-smi")
        expected = str(wsl_layout) + (f":{library_path}" if library_path else "")
        assert kwargs["env"] == {"LD_LIBRARY_PATH": expected}
        return CommandResult(0, "0, NVIDIA GeForce RTX 5060 Laptop GPU, 8151, 6000, 12.0, 581.42\n", "")

    monkeypatch.setattr(gpu, "run", run)
    found, warnings = gpu.detect_nvidia()
    assert len(found) == 1
    assert len(commands) == 1
    assert found[0].name == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert found[0].vendor is Accelerator.NVIDIA
    assert found[0].vram_bytes == 8151 * gpu.MIB
    assert found[0].vram_free_bytes == 6000 * gpu.MIB
    assert found[0].compute_capability == "12.0"
    assert warnings == []
    assert dict(os.environ) == environment


@pytest.mark.parametrize("system,release", [
    ("Linux", "6.8.0-generic"),
    ("Darwin", "24.0.0"),
    ("Windows", "microsoft-WSL2"),
])
def test_non_wsl_ignores_wsl_driver_directory(monkeypatch, wsl_layout, system, release):
    monkeypatch.setattr(gpu.platform, "system", lambda: system)
    monkeypatch.setattr(gpu.platform, "release", lambda: release)
    monkeypatch.setattr(gpu, "run", lambda *a, **kw: pytest.fail("probed WSL on a non-WSL OS"))
    assert gpu.detect_nvidia() == ([], [])


def test_non_wsl_path_probe_keeps_its_environment(monkeypatch, wsl_layout):
    monkeypatch.setattr(gpu.platform, "release", lambda: "6.8.0-generic")
    monkeypatch.setattr(gpu, "have", lambda name: name == "nvidia-smi")
    environment = dict(os.environ)

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs.get("env") is None
        return CommandResult(0, "0, NVIDIA GPU, 8192, 6000, 8.9, 550.0\n", "")

    monkeypatch.setattr(gpu, "run", run)
    found, warnings = gpu.detect_nvidia()
    assert len(found) == 1
    assert warnings == []
    assert dict(os.environ) == environment


def test_wsl_prefers_tool_on_path_and_still_provides_driver_libraries(monkeypatch, wsl_layout):
    monkeypatch.setattr(gpu, "have", lambda name: name == "nvidia-smi")

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[0] == str(wsl_layout)
        return CommandResult(0, "0, NVIDIA GPU, 8151, 6000, 12.0, 581.42\n", "")

    monkeypatch.setattr(gpu, "run", run)
    found, warnings = gpu.detect_nvidia()
    assert len(found) == 1
    assert warnings == []


def test_wsl_without_tool_skips_probe(monkeypatch, wsl_layout):
    (wsl_layout / "nvidia-smi").unlink()
    monkeypatch.setattr(gpu, "run", lambda *a, **kw: pytest.fail("ran a missing tool"))
    assert gpu.detect_nvidia() == ([], [])


def test_wsl_tool_failure_keeps_driver_warning(monkeypatch, wsl_layout):
    monkeypatch.setattr(gpu, "run", lambda *a, **kw: CommandResult(1, "", "driver unavailable"))
    found, warnings = gpu.detect_nvidia()
    assert found == []
    assert "driver unavailable" in warnings[0]
