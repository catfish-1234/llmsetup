"""The interview itself.

Three ways in, one output:

* interactive, a TTY wizard,
* ``--answers profile.yaml``, for scripting and repeatability,
* CLI flags, for one-liners.

All three converge on the same answers dict, which :func:`profile_from_answers`
validates into a :class:`UseCaseProfile`. That single path is why a scripted run
and an interactive one cannot drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import yaml

from llmfit.interview.questions import QUESTIONS, Option, Question, infer_flags
from llmfit.types import Client, Priority, Task, UseCaseProfile
from llmfit.util.errors import LlmfitError
from llmfit.util.log import Console


def run_interview(console: Console, defaults: Optional[Dict[str, Any]] = None,
                  stream: Optional[TextIO] = None,
                  input_fn=None) -> UseCaseProfile:
    """Ask the questions. ``defaults`` pre-answers some and skips those."""
    answers: Dict[str, Any] = dict(defaults or {})
    reader = input_fn or input
    out = stream or sys.stderr

    def emit(text: str = "") -> None:
        out.write(text + "\n")
        out.flush()

    emit("")
    emit(console.bold("Let's work out what you need."))
    emit(console.dim("Press Enter to accept the default shown in [brackets]."))
    emit("")

    for question in QUESTIONS:
        if question.key in answers:
            continue                       # already supplied by a flag
        if question.when and not question.when(answers):
            continue
        answers[question.key] = _ask(question, console, emit, reader)

    emit("")
    return profile_from_answers(infer_flags(answers))


def _ask(question: Question, console: Console, emit, reader) -> Any:
    emit(console.bold(question.prompt))
    if question.why:
        emit(console.dim(f"  why: {question.why}"))

    if question.kind == "choice":
        return _ask_choice(question, console, emit, reader)
    if question.kind == "multi":
        return _ask_multi(question, console, emit, reader)
    if question.kind == "bool":
        return _ask_bool(question, emit, reader)
    if question.kind == "int":
        return _ask_int(question, emit, reader)
    return _ask_text(question, emit, reader)


def _render_options(options: List[Option], console: Console, emit) -> None:
    for index, option in enumerate(options, 1):
        hint = console.dim(f"  ({option.hint})") if option.hint else ""
        emit(f"  {index}) {option.label}{hint}")


def _ask_choice(question: Question, console: Console, emit, reader) -> Any:
    _render_options(question.options, console, emit)
    default_index = next(
        (i for i, o in enumerate(question.options, 1) if o.value == str(question.default)), 1
    )
    while True:
        raw = _read(reader, emit, f"  choice [{default_index}]: ")
        if not raw:
            value = question.options[default_index - 1].value
            return question.parse(value) if question.parse else value
        if raw.isdigit() and 1 <= int(raw) <= len(question.options):
            value = question.options[int(raw) - 1].value
            return question.parse(value) if question.parse else value
        match = next((o for o in question.options if o.value == raw.lower()), None)
        if match:
            return question.parse(match.value) if question.parse else match.value
        # A free-typed number is a legitimate answer for context length.
        if question.parse:
            try:
                return question.parse(raw)
            except (ValueError, TypeError):
                pass
        emit("  please pick one of the numbers above")


def _ask_multi(question: Question, console: Console, emit, reader) -> List[str]:
    _render_options(question.options, console, emit)
    default_labels = ", ".join(question.default or []) or "none"
    while True:
        raw = _read(reader, emit,
                    f"  choose one or more, comma-separated [{default_labels}]: ")
        if not raw:
            return list(question.default or [])
        picked, bad = [], []
        for part in raw.replace(" ", ",").split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part.isdigit() and 1 <= int(part) <= len(question.options):
                picked.append(question.options[int(part) - 1].value)
            elif any(o.value == part for o in question.options):
                picked.append(part)
            else:
                bad.append(part)
        if bad:
            emit(f"  not recognised: {', '.join(bad)}")
            continue
        if picked:
            # Preserve the order typed: the first task is treated as primary.
            return list(dict.fromkeys(picked))
        emit("  pick at least one")


def _ask_bool(question: Question, emit, reader) -> bool:
    default = question.default
    suffix = "y/n" if default is None else ("Y/n" if default else "y/N")
    while True:
        raw = _read(reader, emit, f"  [{suffix}]: ").lower()
        if not raw:
            return bool(default)
        if raw in ("y", "yes", "true", "1"):
            return True
        if raw in ("n", "no", "false", "0"):
            return False
        emit("  please answer y or n")


def _ask_int(question: Question, emit, reader) -> int:
    while True:
        raw = _read(reader, emit, f"  [{question.default}]: ")
        if not raw:
            return int(question.default)
        try:
            value = int(raw)
        except ValueError:
            emit("  please enter a whole number")
            continue
        if value < 1:
            emit("  must be at least 1")
            continue
        return value


def _ask_text(question: Question, emit, reader) -> Any:
    raw = _read(reader, emit, f"  [{question.default}]: ")
    value = raw if raw else str(question.default or "")
    return question.parse(value) if question.parse else value


def _read(reader, emit, prompt: str) -> str:
    try:
        return reader(prompt).strip()
    except EOFError:
        # Non-interactive stdin: take defaults rather than crashing.
        emit("")
        return ""
    except KeyboardInterrupt:
        emit("\naborted")
        raise SystemExit(130) from None


# --------------------------------------------------------------------------- #
# non-interactive paths
# --------------------------------------------------------------------------- #


def load_answers(path: Path) -> Dict[str, Any]:
    """Read an answers file (the same shape `llmfit setup --save-profile` writes)."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LlmfitError(f"could not read answers file {path}: {exc}",
                          hint="it must be valid YAML or JSON") from None
    if not isinstance(data, dict):
        raise LlmfitError(f"{path} must contain a mapping of answers")
    # Accept a full saved profile as well as a bare answers file.
    return data.get("profile", data)


def profile_from_answers(answers: Dict[str, Any]) -> UseCaseProfile:
    """Validate answers into a profile, with clear errors on bad values."""
    data = dict(answers)

    data["tasks"] = _enum_list(data.get("tasks") or ["chat"], Task, "task")
    data["clients"] = _enum_list(data.get("clients") or [], Client, "client")
    try:
        data["priority"] = Priority(str(data.get("priority", "balanced")).lower())
    except ValueError:
        raise LlmfitError(
            f"unknown priority {data.get('priority')!r}",
            hint="use one of: " + ", ".join(p.value for p in Priority),
        ) from None

    context = int(data.get("context_tokens") or 8192)
    if context < 512:
        raise LlmfitError(f"context_tokens={context} is too small to be usable",
                          hint="512 is the practical minimum; 8192 is a good default")
    data["context_tokens"] = context

    users = int(data.get("concurrent_users") or 1)
    data["concurrent_users"] = max(1, users)

    if data.get("max_disk_gb") is not None:
        data["max_disk_gb"] = float(data["max_disk_gb"])
    if isinstance(data.get("languages"), str):
        data["languages"] = [p.strip() for p in data["languages"].split(",") if p.strip()]
    data.setdefault("languages", ["en"])

    for flag in ("needs_tools", "needs_json", "needs_vision", "needs_embeddings",
                 "needs_reasoning", "commercial_use", "offline_only"):
        data[flag] = bool(data.get(flag, False))

    known = set(UseCaseProfile.__dataclass_fields__)
    return UseCaseProfile(**{k: v for k, v in data.items() if k in known})


def _enum_list(values, enum_cls, label: str):
    out = []
    for value in values:
        if isinstance(value, enum_cls):
            out.append(value)
            continue
        try:
            out.append(enum_cls(str(value).lower()))
        except ValueError:
            raise LlmfitError(
                f"unknown {label}: {value!r}",
                hint="valid values: " + ", ".join(m.value for m in enum_cls),
            ) from None
    return out


def save_profile(profile: UseCaseProfile, path: Path) -> Path:
    """Write answers back out so the whole run can be reproduced."""
    from llmfit.util.fs import write_text

    return write_text(path, yaml.safe_dump(profile.to_dict(), sort_keys=False))
