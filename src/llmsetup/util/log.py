"""Terminal output.

One place decides colour and verbosity, so `--quiet`, `--json` and plain
non-TTY pipes all behave without every module re-deciding.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional, TextIO

_ANSI = re.compile(r"\033\[[0-9;]*m")
# Where a wrapped line's continuation rows start: under the first word after the
# indent and any leading markers ("!", "*", "1.", "==>").
_HANG = re.compile(r"\s*(?:(?:[!*✓✗]|==>|\d+\.)\s+)*")
_MIN_WRAP_COLUMNS = 40


def wrap_line(line: str, columns: int) -> str:
    """Wrap one line of prose to `columns`, continuing under its first word.

    Breaks only at spaces and never inside an escape code, so colours carry over
    and a word longer than the row is left whole rather than split.
    """
    limit = columns - 1
    if columns < _MIN_WRAP_COLUMNS or len(_ANSI.sub("", line)) <= limit:
        return line
    hang = _HANG.match(_ANSI.sub("", line))
    indent = min(hang.end() if hang else 0, limit // 2)
    rows: List[str] = []
    row, col, gap = "", 0, ""
    for piece in re.split(r"( +)", line):
        if not piece:
            continue
        if not piece.strip(" "):
            gap = piece
            continue
        width = len(_ANSI.sub("", piece))
        if width and col > indent and col + len(gap) + width > limit:
            rows.append(row)
            row, col = " " * indent + piece, indent + width
        else:
            row, col = row + gap + piece, col + len(gap) + width
        gap = ""
    rows.append(row)
    return "\n".join(rows)


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
                 verbose: bool = False, color: Optional[bool] = None,
                 columns: Optional[int] = None):
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.verbose = verbose
        tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        if color is None:
            color = (
                tty
                and os.environ.get("NO_COLOR") is None
                and os.environ.get("TERM") != "dumb"
            )
        self.color = bool(color)
        # Prose is wrapped only for a person at a terminal; pipes, files and
        # tests get each line whole, exactly as before. 0 means never wrap.
        self._columns = columns if columns is not None else (None if tty else 0)

    def columns(self) -> int:
        """The terminal's width now (it can be resized), or 0 for no wrapping."""
        if self._columns is not None:
            return self._columns
        override = os.environ.get("COLUMNS", "")
        if override.isdigit():
            return int(override)
        try:
            return os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            return 0

    def _prose(self, line: str, wrap: bool) -> str:
        return wrap_line(line, self.columns()) if wrap else line

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

    def info(self, text: str, wrap: bool = False) -> None:
        """`wrap=True` is for prose only: never for a command, path or snippet
        someone will copy, which must stay on one line."""
        if not self.quiet:
            self._emit(self._prose(f"  {text}", wrap))

    def ok(self, text: str) -> None:
        if not self.quiet:
            self._emit(f"  {self._c('✓', '32')} {text}")

    def warn(self, text: str, wrap: bool = False) -> None:
        self._emit(self._prose(f"  {self._c('!', '33')} {text}", wrap))

    def error(self, text: str) -> None:
        self._emit(f"  {self._c('✗', '31')} {text}")

    def debug(self, text: str) -> None:
        if self.verbose and not self.quiet:
            self._emit(self.dim(f"    {text}"))


CONSOLE = Console()
