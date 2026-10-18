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

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import yaml

from llmsetup.catalog.catalog import Catalog
from llmsetup.interview.questions import QUESTIONS, Option, Question, infer_flags
from llmsetup.types import Candidate, Client, HardwareProfile, Priority, Task, UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console


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


def review_model_choice(profile: UseCaseProfile, hardware: HardwareProfile,
                        catalog: Catalog, console: Console, *,
                        runtime: Optional[str] = None, interactive: bool = True,
                        stream: Optional[TextIO] = None, input_fn=None) -> UseCaseProfile:
    """Explain a named model before the user keeps it or chooses an alternative.

    Non-interactive callers keep their explicit choice. The planner enforces
    required capabilities and memory limits even when the user keeps a poor fit.
    """
    from llmsetup.planner.model_fit import assess_requested_model

    reader = input_fn or input

    def emit(text: str = "") -> None:
        if stream is None:
            console.info(text)
        else:
            stream.write(text + "\n")
            stream.flush()

    while profile.requested_model:
        try:
            assessment = assess_requested_model(profile, hardware, catalog, runtime=runtime)
        except LlmsetupError as exc:
            if not interactive:
                raise
            emit(str(exc))
            emit("Enter a catalog model ID or name, or leave blank for automatic selection.")
            name = _ask_text(Question(
                "requested_model", "", "text", default="",
            ), emit, reader)
            profile = replace(profile, requested_model=name or None)
            continue

        profile = replace(profile, requested_model=assessment.selected_model.id)
        emit(f"Model review: {assessment.selected_model.name} ({profile.requested_model})")
        for message in assessment.messages:
            emit(message)
        alternatives: List[Candidate] = []
        if assessment.poor_fit:
            emit("This model is a poor fit for the requested setup.")
            for candidate in assessment.alternatives:
                if candidate.model.id in {item.model.id for item in alternatives}:
                    continue
                alternatives.append(candidate)
            if not alternatives:
                emit("No compatible alternative fits the detected hardware.")
        if not interactive:
            return profile

        options = [Option("keep", f"Keep {assessment.selected_model.name}"),
                   *[Option(candidate.model.id, f"Use {candidate.model.name}")
                     for candidate in alternatives],
                   Option("auto", "Choose a model automatically")]
        choice = _ask_choice(Question(
            "requested_model", "", "choice", options=options, default="keep",
        ), console, emit, reader)
        if choice == "keep":
            emit(f"Planning will keep {profile.requested_model} fixed and tune the other settings.")
            return profile
        profile = replace(profile, requested_model=None if choice == "auto" else choice)
    return profile


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
    """Read an answers file (the same shape `llmsetup setup --save-profile` writes)."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LlmsetupError(f"could not read answers file {path}: {exc}",
                          hint="it must be valid YAML or JSON") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise LlmsetupError(f"{path} must contain a mapping of answers")
    # Accept a full saved profile as well as a bare answers file.
    answers = data.get("profile", data)
    if not isinstance(answers, dict):
        raise LlmsetupError(f"the profile in {path} must contain a mapping of answers")
    return answers


def profile_from_answers(answers: Dict[str, Any]) -> UseCaseProfile:
    """Validate answers into a profile, with clear errors on bad values."""
    if not isinstance(answers, dict):
        raise LlmsetupError("answers must be a mapping")
    data = dict(answers)

    requested_model = data.get("requested_model")
    if requested_model is not None and not isinstance(requested_model, str):
        raise LlmsetupError("requested_model must be a model ID or name")
    data["requested_model"] = (requested_model.strip() or None) if requested_model else None

    data["tasks"] = _enum_list(data.get("tasks") or ["chat"], Task, "task")
    data["clients"] = _enum_list(data.get("clients") or [], Client, "client")
    try:
        data["priority"] = Priority(str(data.get("priority", "balanced")).lower())
    except ValueError:
        raise LlmsetupError(
            f"unknown priority {data.get('priority')!r}",
            hint="use one of: " + ", ".join(p.value for p in Priority),
        ) from None

    context = _number(data.get("context_tokens") or 8192, "context_tokens", int)
    if context < 512:
        raise LlmsetupError(f"context_tokens={context} is too small to be usable",
                          hint="512 is the practical minimum; 8192 is a good default")
    data["context_tokens"] = context

    users = _number(data.get("concurrent_users") or 1, "concurrent_users", int)
    data["concurrent_users"] = max(1, users)

    if data.get("max_disk_gb") is not None:
        data["max_disk_gb"] = _number(data["max_disk_gb"], "max_disk_gb", float)
        if data["max_disk_gb"] <= 0:
            raise LlmsetupError("max_disk_gb must be positive")
    if isinstance(data.get("languages"), str):
        data["languages"] = [p.strip() for p in data["languages"].split(",") if p.strip()]
    if data.get("languages") is None:
        data["languages"] = ["en"]
    if not isinstance(data["languages"], (list, tuple)) or \
            any(not isinstance(language, str) for language in data["languages"]):
        raise LlmsetupError("languages must be a list of language codes")

    for flag in ("needs_tools", "needs_json", "needs_vision", "needs_embeddings",
                 "needs_reasoning", "commercial_use", "offline_only"):
        value = data.get(flag, False)
        if isinstance(value, str):
            normal = value.strip().lower()
            if normal not in ("true", "false", "yes", "no", "on", "off", "1", "0"):
                raise LlmsetupError(f"{flag} must be true or false, got {value!r}")
            value = normal in ("true", "yes", "on", "1")
        data[flag] = bool(value)

    known = set(UseCaseProfile.__dataclass_fields__)
    return UseCaseProfile(**{k: v for k, v in data.items() if k in known})


def _number(value, field: str, caster):
    try:
        number = caster(value)
        if isinstance(value, bool) or not math.isfinite(number):
            raise ValueError
        return number
    except (TypeError, ValueError, OverflowError):
        raise LlmsetupError(f"{field} must be a finite number, got {value!r}") from None


def _enum_list(values, enum_cls, label: str):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise LlmsetupError(f"{label}s must be a list")
    out = []
    for value in values:
        if isinstance(value, enum_cls):
            out.append(value)
            continue
        try:
            out.append(enum_cls(str(value).lower()))
        except ValueError:
            raise LlmsetupError(
                f"unknown {label}: {value!r}",
                hint="valid values: " + ", ".join(m.value for m in enum_cls),
            ) from None
    return out


def save_profile(profile: UseCaseProfile, path: Path) -> Path:
    """Write answers back out so the whole run can be reproduced."""
    from llmsetup.util.fs import write_text

    return write_text(path, yaml.safe_dump(profile.to_dict(), sort_keys=False))
