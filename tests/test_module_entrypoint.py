"""The module entry point should be usable from the selected Python interpreter."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("flag,expected", [("--version", "llmsetup 0.6.1"),
                                           ("--help", "Pick, tune, verify")])
def test_module_entrypoint(flag, expected):
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run([sys.executable, "-m", "llmsetup", flag],
                            env={**os.environ, "PYTHONPATH": str(source)},
                            capture_output=True, text=True, check=True)
    assert expected in result.stdout


def test_one_version_everywhere():
    """pyproject, --version and every User-Agent come from llmsetup.version."""
    import re
    from pathlib import Path

    from llmsetup import download
    from llmsetup.version import __version__

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert download.USER_AGENT.startswith(f"llmsetup/{__version__} ")
    sources = Path(__file__).resolve().parents[1] / "src" / "llmsetup"
    stale = [str(path) for path in sources.rglob("*.py")
             if re.search(r'"llmsetup/\d', path.read_text(encoding="utf-8"))]
    assert stale == []
