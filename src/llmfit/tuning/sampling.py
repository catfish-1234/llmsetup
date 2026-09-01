"""Sampling defaults.

Two things drive these, and both matter more than people expect:

* **the model family**, reasoning models are trained to explore during their
  thinking phase and go into repetition loops at the low temperatures that suit
  an instruct model. Publishers state recommended values; using 0.7/0.9 for
  everything discards that.
* **the task**. Code and structured extraction want near-deterministic output;
  creative writing wants the opposite. The same model should not be sampled the
  same way for both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from llmfit.tuning import publisher
from llmfit.types import ModelSpec, Task, UseCaseProfile

#: Tasks where repeated tokens are legitimate output rather than a defect.
_CODE_LIKE = {Task.CODING, Task.AGENT, Task.EXTRACT}

#: Tasks that want reproducible output, where a publisher permits it.
_DETERMINISTIC = {Task.EXTRACT, Task.CODING, Task.TRANSLATE}


@dataclass
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repeat_penalty: float
    stop: List[str]
    presence_penalty: float = 0.0
    rationale: str = ""
    #: Where the base values came from, a publisher's model card, or our own
    #: task defaults. Reported so the user can tell the difference.
    source: str = ""


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


def sampling_for(spec: ModelSpec, profile: UseCaseProfile,
                 thinking: bool = False) -> Sampling:
    """Sampling settings for this model and task.

    Publisher recommendations win the base values where they exist. A model is
    tuned and benchmarked at specific settings, and substituting generic ones is
    a known cause of local output that is quietly worse than the model's
    reputation, with nothing in the setup to point at.

    The task then adjusts *within* what the publisher sanctions: nudging
    temperature down for code is reasonable, but replacing a stated presence
    penalty of 1.5 with 1.1 because "chat usually wants that" is not.
    """
    temperature, top_p, top_k, min_p, repeat_penalty, why = _TASK_DEFAULTS.get(
        profile.primary_task, _DEFAULT
    )
    notes = [why]
    presence_penalty = 0.0

    # `thinking` is the decision, not the capability: a reasoning model with
    # thinking disabled must get the publisher's instruct settings, because that
    # is the mode it will actually run in.
    published = publisher.lookup(spec.family, thinking)
    if published is not None:
        temperature = published.temperature
        top_p = published.top_p
        top_k = published.top_k
        min_p = published.min_p
        repeat_penalty = published.repeat_penalty
        presence_penalty = published.presence_penalty

        # The publisher's own carve-out for code-like output.
        if profile.primary_task in _CODE_LIKE and \
                published.code_presence_penalty is not None:
            presence_penalty = published.code_presence_penalty
            notes.append(
                f"presence penalty dropped to {presence_penalty} for code, which "
                f"the publisher specifies separately"
            )

        notes = [f"{published.source}: temperature {temperature}, top-p {top_p}"]
        if published.notes:
            notes.append(published.notes)

        # Determinism is a task preference, but never past what the publisher
        # warns against, several families loop at temperature 0.
        if profile.primary_task in _DETERMINISTIC and \
                "temperature 0" not in published.avoid and \
                "greedy decoding" not in published.avoid:
            lowered = min(temperature, 0.2)
            if lowered < temperature:
                temperature = lowered
                notes.append(
                    f"temperature lowered to {temperature} for reproducible output"
                )
        elif profile.primary_task in _DETERMINISTIC:
            notes.append(
                "temperature left at the published value: this family warns "
                "against greedy decoding, which sends it into repetition loops"
            )

    if spec.is_reasoning and published is None:
        # Only when the publisher said nothing. Reasoning models degenerate into
        # loops when sampled too tightly, so a generic low task temperature is
        # lifted to a safe floor, but a stated value is never overridden.
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
        # Not a hard constraint, just stops runaway prose after a JSON object.
        stop = ["```", "\n\n\n"]
        notes.append("stop sequences added to keep structured output tight")

    return Sampling(
        temperature=round(temperature, 3),
        top_p=round(top_p, 3),
        top_k=int(top_k),
        min_p=round(min_p, 3),
        repeat_penalty=round(repeat_penalty, 3),
        presence_penalty=round(presence_penalty, 3),
        stop=stop,
        rationale="; ".join(notes),
        source=published.source if published else "llmfit task defaults",
    )
