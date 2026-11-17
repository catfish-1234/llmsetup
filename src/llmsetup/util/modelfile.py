"""Values written into an Ollama Modelfile.

Ollama's parser reads a directive per line. A ``"..."`` value is taken
literally, without escapes, and one left open at the end of a line continues
onto the next until a line ends with the closing quote. So a value is safe in
quotes unless it holds both a quote and a line break: that combination could
close the string early and turn the rest of the value into new directives
such as ``SYSTEM`` or ``ADAPTER``.
"""

from __future__ import annotations

from typing import Optional

from llmsetup.util.errors import ProvisionError

_BREAKS = ("\n", "\r", "\u2028", "\u2029")


def modelfile_text(value: str) -> str:
    """Quote ``value`` as a triple-quoted block, for SYSTEM or TEMPLATE."""
    text = str(value)
    if '"""' in text:
        raise ProvisionError("cannot write text containing three double quotes into a Modelfile")
    return f'"""{text}"""'


def modelfile_string(value: str) -> str:
    """Quote ``value`` as one Modelfile parameter value."""
    text = str(value)
    if '"' in text and any(mark in text for mark in _BREAKS):
        raise ProvisionError(
            f"cannot write {text!r} into a Modelfile: it contains both a quote and a line break")
    return f'"{text}"'


def modelfile_bare(value: Optional[str]) -> str:
    """A single-line, unquoted value such as a FROM source."""
    text = "" if value is None else str(value)
    if not text.strip() or any(mark in text for mark in _BREAKS):
        raise ProvisionError(f"cannot write {text!r} into a Modelfile: it must be one line")
    return text
