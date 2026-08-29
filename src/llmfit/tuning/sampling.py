"""Sampling defaults.

Two things drive these, and both matter more than people expect:

* **the model family** — reasoning models are trained to explore during their
  thinking phase and go into repetition loops at the low temperatures that suit
  an instruct model. Publishers state recommended values; using 0.7/0.9 for
  everything discards that.
* **the task** — code and structured extraction want near-deterministic output;
  creative writing wants the opposite. The same model should not be sampled the
  same way for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llmfit.types import ModelSpec, Task, UseCaseProfile


@dataclass
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repeat_penalty: float
    stop: List[str]
    rationale: str = ""


# Per-task baselines. Deterministic at the top, exploratory at the bottom.
_TASK_DEFAULTS = {
    Task.EXTRACT:   (0.0,  1.00, 0,  0.00, 1.00, "structured extraction must be reproducible"),
    Task.CODING:    (0.2,  0.95, 20, 0.00, 1.05, "code wants near-deterministic output"),
    Task.AGENT:     (0.3,  0.95, 20, 0.00, 1.05, "agent tool calls need stable argument formatting"),
    Task.RAG:       (0.3,  0.90, 40, 0.05, 1.05, "answers should stay close to retrieved context"),
    Task.SUMMARIZE: (0.3,  0.90, 40, 0.05, 1.05, "summaries should not invent detail"),
    Task.TRANSLATE: (0.2,  0.90, 40, 0.05, 1.05, "translation favours fidelity over variety"),
    Task.CHAT:      (0.7,  0.90, 40, 0.05, 1.10, "general conversation"),
    Task.CREATIVE:  (0.95, 0.95, 60, 0.03, 1.05, "creative writing benefits from variety"),
    Task.VISION:    (0.4,  0.90, 40, 0.05, 1.05, "image description should stay grounded"),
}

_DEFAULT = (0.7, 0.9, 40, 0.05, 1.1, "general default")


def sampling_for(spec: ModelSpec, profile: UseCaseProfile) -> Sampling:
    temperature, top_p, top_k, min_p, repeat_penalty, why = _TASK_DEFAULTS.get(
        profile.primary_task, _DEFAULT
    )
    notes = [why]

    if spec.is_reasoning:
        # Reasoning models degenerate into loops when sampled too tightly; the
        # published guidance for this family is ~0.6/0.95 with no repeat penalty,
        # so we lift a low task temperature to that floor rather than override it.
        temperature = max(temperature, 0.6)
        top_p = max(top_p, 0.95)
        top_k = max(top_k, 20)
        repeat_penalty = 1.0
        notes.append(
            "raised to the reasoning-model floor: these degenerate into "
            "repetition loops when sampled too greedily"
        )

    stop: List[str] = []
    if profile.needs_json and profile.primary_task is Task.EXTRACT:
        # Not a hard constraint — just stops runaway prose after a JSON object.
        stop = ["```", "\n\n\n"]
        notes.append("stop sequences added to keep structured output tight")

    return Sampling(
        temperature=round(temperature, 3),
        top_p=round(top_p, 3),
        top_k=int(top_k),
        min_p=round(min_p, 3),
        repeat_penalty=round(repeat_penalty, 3),
        stop=stop,
        rationale="; ".join(notes),
    )
