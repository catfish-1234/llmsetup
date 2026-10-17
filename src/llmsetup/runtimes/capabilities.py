"""What the installed binary actually accepts.

llama.cpp moves fast. Flags get renamed, deprecated and replaced between
releases, and a user's build is whatever their package manager last shipped. A
launch command written against the current documentation can therefore fail on a
perfectly good older binary, and one written against an old build silently gives
up features the new one has.

Rather than guess from a version string, which llama.cpp does not version
semantically anyway, ask the binary: `--help` lists every flag it accepts. That
turns a whole class of breakage into a question with an answer.

The result is cached per binary path and modification time, so the subprocess
runs once per install rather than once per launch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from llmsetup.util.proc import run, which

#: Long-form flags only. Short aliases are unstable and ambiguous, and llmsetup
#: emits long forms everywhere.
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9.-]*)")

_cache: Dict[Tuple[str, float], Set[str]] = {}


def _binary_key(path: str) -> Tuple[str, float]:
    try:
        return (path, Path(path).stat().st_mtime)
    except OSError:
        return (path, 0.0)


def supported_flags(binary: str, timeout: float = 20.0) -> Set[str]:
    """Every long flag the installed binary lists in its own help.

    An empty set means the question could not be answered: the binary is
    missing, refused to run, or printed something unrecognisable. Callers treat
    that as "assume everything is supported" rather than stripping the command
    down to nothing, because a wrong strip is worse than a flag the binary will
    reject with a clear message of its own.
    """
    resolved = which(binary) or binary
    key = _binary_key(resolved)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    result = run([resolved, "--help"], timeout=timeout)
    # llama.cpp prints help to stdout and exits 0, but some builds and some
    # wrappers use stderr and a non-zero status. Take whatever text there is.
    text = result.output or ""
    flags = set(_FLAG_RE.findall(text))
    # Anything under a handful of flags means the output was not a help screen.
    if len(flags) < 10:
        flags = set()
    _cache[key] = flags
    return flags


def filter_flags(command, supported: Set[str], drop_value: Optional[Set[str]] = None):
    """Remove flags the binary does not accept, along with their values.

    ``drop_value`` names the flags that take no value, so the token after them
    is kept. Everything else is assumed to take exactly one value, which holds
    for every flag llmsetup emits.

    With an empty ``supported`` set nothing is removed: not knowing is not the
    same as knowing it is unsupported.
    """
    if not supported:
        return list(command)
    valueless = drop_value or set()
    out, i = [], 0
    while i < len(command):
        token = command[i]
        if token.startswith("--") and token not in supported:
            i += 2 if (token not in valueless
                       and i + 1 < len(command)
                       and not command[i + 1].startswith("--")) else 1
            continue
        out.append(token)
        i += 1
    return out


def clear_cache() -> None:
    """Forget what was learned. Used by tests, and after an upgrade."""
    _cache.clear()
