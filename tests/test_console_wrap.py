"""Prose wraps to the terminal under its first word; everything else stays whole."""

from __future__ import annotations

import io
import re

from llmsetup.util.log import Console, wrap_line

ANSI = re.compile(r"\033\[[0-9;]*m")

WARNING = ("  !    uses 86% of usable VRAM; a long prompt or another application "
           "could push this into swap or an out-of-memory error")


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def _visible(text: str) -> str:
    return ANSI.sub("", text)


def test_a_line_that_fits_is_unchanged():
    assert wrap_line("  short and sweet", 80) == "  short and sweet"


def test_a_long_line_wraps_under_its_first_word_and_keeps_every_word():
    rows = wrap_line(WARNING, 60).split("\n")

    assert len(rows) > 1
    assert all(len(row) <= 59 for row in rows)
    assert rows[0].startswith("  !    uses 86%")
    assert all(row.startswith(" " * 7) and row[7] != " " for row in rows[1:])
    assert " ".join(" ".join(rows).split()) == " ".join(WARNING.split())


def test_colour_codes_are_not_counted_or_split():
    line = "  \033[2m   " + " ".join(["word"] * 30) + "\033[0m"
    rows = wrap_line(line, 50).split("\n")

    assert len(rows) > 1
    assert all(len(_visible(row)) <= 49 for row in rows)
    assert "".join(rows).count("\033[2m") == 1
    assert rows[-1].endswith("\033[0m")


def test_a_word_longer_than_the_row_stays_whole():
    path = "/a/very/long/path/" + "x" * 80
    rows = wrap_line(f"  see {path} for details", 50).split("\n")

    assert [row.strip() for row in rows] == ["see", path, "for details"]


def test_a_narrow_or_unknown_width_never_wraps():
    assert wrap_line(WARNING, 30) == WARNING
    assert wrap_line(WARNING, 0) == WARNING


def test_output_that_is_not_a_terminal_is_never_wrapped():
    stream = io.StringIO()
    Console(stream=stream).warn(WARNING.strip()[2:], wrap=True)

    assert stream.getvalue().count("\n") == 1


def test_only_lines_marked_as_prose_wrap():
    stream = io.StringIO()
    console = Console(stream=stream, columns=60)
    command = "llmsetup connect opencode --model " + "m" * 20 + " --base-url http://127.0.0.1:8080/v1"

    console.info(command)
    console.warn("a long prompt or another application could push this into swap or an error",
                 wrap=True)

    lines = stream.getvalue().splitlines()
    assert lines[0] == f"  {command}"
    assert len(lines) == 3
    assert lines[2].startswith("    ") and not lines[2].startswith("     ")


def test_a_terminal_wraps_at_its_columns(monkeypatch):
    monkeypatch.setenv("COLUMNS", "50")
    stream = _Terminal()
    console = Console(stream=stream, color=False)

    assert console.columns() == 50
    console.info("   " + " ".join(["word"] * 20), wrap=True)

    assert all(len(row) <= 49 for row in stream.getvalue().splitlines())
    assert stream.getvalue().count("\n") > 1
