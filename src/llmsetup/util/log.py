"""Terminal output.

One place decides colour and verbosity, so `--quiet`, `--json` and plain
non-TTY pipes all behave without every module re-deciding.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO


def write_console(stream: TextIO, text: str) -> None:
    """Keep legacy Windows consoles usable without changing their code page."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding))
    stream.flush()


class Console:
    def __init__(self, stream: Optional[TextIO] = None, quiet: bool = False,
                 verbose: bool = False, color: Optional[bool] = None):
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.verbose = verbose
        if color is None:
            color = (
                hasattr(self.stream, "isatty")
                and self.stream.isatty()
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )
        self.color = bool(color)

    def _c(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, text: str) -> str:
        return self._c(text, "2")

    def bold(self, text: str) -> str:
        return self._c(text, "1")

    def _emit(self, text: str) -> None:
        write_console(self.stream, text + "\n")

    def print(self, text: str = "") -> None:
        if not self.quiet:
            self._emit(text)

    def step(self, text: str) -> None:
        if not self.quiet:
            self._emit(f"{self._c('==>', '1;34')} {self.bold(text)}")

    def info(self, text: str) -> None:
        if not self.quiet:
            self._emit(f"  {text}")

    def ok(self, text: str) -> None:
        if not self.quiet:
            self._emit(f"  {self._c('✓', '32')} {text}")

    def warn(self, text: str) -> None:
        self._emit(f"  {self._c('!', '33')} {text}")

    def error(self, text: str) -> None:
        self._emit(f"  {self._c('✗', '31')} {text}")

    def debug(self, text: str) -> None:
        if self.verbose and not self.quiet:
            self._emit(self.dim(f"    {text}"))


CONSOLE = Console()
