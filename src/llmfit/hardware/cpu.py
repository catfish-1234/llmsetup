"""CPU and system memory probing.

No psutil dependency: each platform has a cheap native source (``/proc`` on
Linux, ``sysctl`` on macOS, ``wmic``/``GlobalMemoryStatusEx`` on Windows), and
adding a binary dependency to a tool people install on a whim is a bad trade.
"""

from __future__ import annotations

import os
import platform
import re
from typing import Optional, Tuple

from llmfit.util.proc import run


def cpu_name() -> str:
    system = platform.system().lower()
    if system == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
                    if line.lower().startswith("hardware"):  # ARM SBCs
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "darwin":
        result = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        if result.ok and result.stdout.strip():
            return result.stdout.strip()
    elif system == "windows":
        result = run(["wmic", "cpu", "get", "name"], timeout=10)
        if result.ok:
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            if len(lines) > 1:
                return lines[1]
    return platform.processor() or platform.machine() or "unknown"


def core_counts() -> Tuple[Optional[int], Optional[int]]:
    """(physical, logical). Physical matters: llama.cpp thread counts should not
    count hyperthreads, which cost more in contention than they return."""
    logical = os.cpu_count()
    physical: Optional[int] = None
    system = platform.system().lower()

    if system == "linux":
        physical = _linux_physical_cores()
    elif system == "darwin":
        result = run(["sysctl", "-n", "hw.physicalcpu"], timeout=5)
        if result.ok and result.stdout.strip().isdigit():
            physical = int(result.stdout.strip())
    elif system == "windows":
        result = run(["wmic", "cpu", "get", "NumberOfCores"], timeout=10)
        if result.ok:
            nums = [int(x) for x in re.findall(r"\d+", result.stdout)]
            if nums:
                physical = sum(nums)

    if physical is None and logical:
        # Assume SMT on x86 (usually right), none on ARM (also usually right).
        physical = max(1, logical // 2) if platform.machine().lower() in ("x86_64", "amd64") else logical
    return physical, logical


def _linux_physical_cores() -> Optional[int]:
    """Count distinct (physical_id, core_id) pairs in /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None

    pairs, current = set(), {}
    for line in text.splitlines():
        if not line.strip():
            if "physical id" in current and "core id" in current:
                pairs.add((current["physical id"], current["core id"]))
            current = {}
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    if "physical id" in current and "core id" in current:
        pairs.add((current["physical id"], current["core id"]))
    return len(pairs) or None


def memory_bytes() -> Tuple[int, Optional[int]]:
    """(total, available) system RAM in bytes."""
    system = platform.system().lower()

    if system == "linux":
        total = available = None
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        available = int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        if total:
            return total, available

    elif system == "darwin":
        result = run(["sysctl", "-n", "hw.memsize"], timeout=5)
        if result.ok and result.stdout.strip().isdigit():
            total = int(result.stdout.strip())
            return total, _darwin_available(total)

    elif system == "windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except Exception:
            pass

    # POSIX fallback.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size, None
    except (ValueError, OSError, AttributeError):
        return 0, None


def _darwin_available(total: int) -> Optional[int]:
    """Free + inactive + speculative pages from vm_stat, macOS 'free' is misleading."""
    result = run(["vm_stat"], timeout=5)
    if not result.ok:
        return None
    page_size = 4096
    match = re.search(r"page size of (\d+) bytes", result.stdout)
    if match:
        page_size = int(match.group(1))
    counts = {}
    for line in result.stdout.splitlines():
        m = re.match(r'"?Pages ([^:"]+)"?:\s+(\d+)', line.strip())
        if m:
            counts[m.group(1).strip()] = int(m.group(2))
    reusable = sum(counts.get(k, 0) for k in ("free", "inactive", "speculative"))
    return min(total, reusable * page_size) if reusable else None


def is_apple_silicon() -> bool:
    return platform.system().lower() == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def normalized_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x86_64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine or "unknown"
