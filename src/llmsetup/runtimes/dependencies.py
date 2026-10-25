"""Remediation hints for Linux loader failures; never install dependencies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def _distro_ids() -> List[str]:
    for filename in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            contents = Path(filename).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        fields = {}
        for line in contents.splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in {"ID", "ID_LIKE"}:
                fields[key.strip()] = value.strip().strip("\"'")
        return f"{fields.get('ID', '')} {fields.get('ID_LIKE', '')}".lower().split()
    return []


def shared_library_hint(output: str) -> str:
    """Identify loader errors and suggest a known package or a library lookup."""
    match = re.search(
        r"error while loading shared libraries:\s*([\w.+-]+\.so(?:\.[0-9]+)*)\s*:",
        output, re.IGNORECASE,
    )
    if not match:
        return ""
    library = match.group(1)
    if library == "libgomp.so.1":
        packages = {
            "ubuntu": "apt install libgomp1",
            "debian": "apt install libgomp1",
            "fedora": "dnf install libgomp",
            "rhel": "dnf install libgomp",
            "centos": "dnf install libgomp",
            "arch": "pacman -S gcc-libs",
            "opensuse": "zypper install libgomp1",
            "suse": "zypper install libgomp1",
        }
        for distro in _distro_ids():
            if distro in packages:
                return (f"Missing shared library {library}. Install its system package "
                        f"with `{packages[distro]}`, then rerun `llmsetup setup`.")
    return (f"Missing shared library {library}. Install the system package providing "
            "it for your distribution, then rerun `llmsetup setup`.")
