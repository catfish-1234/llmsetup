"""Do not require Python-3.10-only tooling while advertising Python 3.9 support."""

from pathlib import Path

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@pytest.mark.parametrize("python_version,coverage,setuptools", [
    ("3.9", "7.10.7", "79.0.1"),
    ("3.10", "7.16.2", "84.0.0"),
    ("3.14", "7.16.2", "84.0.0"),
])
def test_supported_python_has_compatible_tool_requirements(python_version, coverage, setuptools):
    root = Path(__file__).resolve().parents[1]
    build = tomllib.loads((root / "pyproject.toml").read_text())["build-system"]["requires"]
    ci = (root / "requirements-ci.txt").read_text().splitlines()
    active = [Requirement(line) for line in [*build, *ci] if line and not line.startswith("#")]
    active = [req for req in active if not req.marker or
              req.marker.evaluate({"python_version": python_version})]
    for name, version in (("setuptools", setuptools), ("coverage", coverage)):
        matches = [req for req in active if req.name == name]
        assert len(matches) == 1
        assert version in matches[0].specifier
