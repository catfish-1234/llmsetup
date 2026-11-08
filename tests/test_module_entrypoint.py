"""The module entry point should be usable from the selected Python interpreter."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag,expected", [("--version", "llmsetup 0.1.1"),
                                           ("--help", "Pick, tune, verify")])
def test_module_entrypoint(flag, expected):
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run([sys.executable, "-m", "llmsetup", flag],
                            env={**os.environ, "PYTHONPATH": str(source)},
                            capture_output=True, text=True, check=True)
    assert expected in result.stdout
