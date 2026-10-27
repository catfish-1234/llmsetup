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

from llmsetup.tuning import publisher
from llmsetup.types import ModelSpec, Task, UseCaseProfile

#: Tasks where repeated tokens are legitimate output rather than a defect.
_CODE_LIKE = {Task.CODING, Task.AGENT, Task.EXTRACT, Task.AUTOCOMPLETE}

#: Tasks that want reproducible output, where a publisher permits it.
_DETERMINISTIC = {Task.EXTRACT, Task.CODING, Task.TRANSLATE, Task.AUTOCOMPLETE}


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
    Task.CODING:    (0.2,  0.95, 20, 0.00, 1.00, "code wants near-deterministic output"),
    Task.AGENT:     (0.3,  0.95, 20, 0.00, 1.00, "agent tool calls need stable argument formatting"),
    Task.RAG:       (0.3,  0.90, 40, 0.05, 1.05, "answers should stay close to retrieved context"),
    Task.SUMMARIZE: (0.3,  0.90, 40, 0.05, 1.05, "summaries should not invent detail"),
    Task.TRANSLATE: (0.2,  0.90, 40, 0.05, 1.05, "translation favours fidelity over variety"),
    Task.CHAT:      (0.7,  0.90, 40, 0.05, 1.10, "general conversation"),
    Task.CREATIVE:  (0.95, 0.95, 60, 0.03, 1.05, "creative writing benefits from variety"),
    Task.VISION:    (0.4,  0.90, 40, 0.05, 1.05, "image description should stay grounded"),
    Task.AUTOCOMPLETE: (0.1, 0.95, 20, 0.00, 1.00,
                        "inline completion should finish the line the obvious "
                        "way rather than suggest something interesting"),
}

_DEFAULT = (0.7, 0.9, 40, 0.05, 1.1, "general default")

#: Tasks where variety is the point and a low published temperature is the
#: wrong instrument, however well it serves the use the publisher had in mind.
_OPEN_ENDED = {Task.CREATIVE}

#: Where creative work lands. Temperature near one with min-p to truncate is
#: what the writing communities have converged on, and the two only work as a
#: pair: the temperature supplies the variety, the min-p keeps it coherent.
_CREATIVE_TEMPERATURE = 1.0
_CREATIVE_MIN_P = 0.05

#: Where work that has to repeat itself lands. Low enough to be stable, not so
#: low that the model has no choice at all.
_DETERMINISTIC_TEMPERATURE = 0.2

#: Inline completion. Low but not zero: the model should finish the line the
#: obvious way, and every family that warns about greedy decoding warns about
#: zero specifically.
_COMPLETION_TEMPERATURE = 0.2


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
            lowered = min(temperature, _DETERMINISTIC_TEMPERATURE)
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

        # Infilling is a third mode, alongside instruct and thinking. It does
        # not go through the chat template at all: the model is handed a prefix
        # and a suffix on a separate endpoint and asked to join them. A card's
        # temperature describes the conversational mode and does not govern
        # this one, and a completion that varies between identical cursor
        # positions is noise rather than variety. No publisher states a figure
        # for it, so this is llmsetup's own and is labelled as such.
        if profile.primary_task is Task.AUTOCOMPLETE and \
                temperature > _COMPLETION_TEMPERATURE and not spec.is_reasoning:
            temperature = _COMPLETION_TEMPERATURE
            notes.append(
                f"temperature {temperature} for infilling, which is a separate "
                f"endpoint and a different mode from the conversational one the "
                f"card describes"
            )

        # The mirror of the case above. A published number describes the use the
        # publisher had in mind, which is ordinary instruct-style work, and some
        # cards state a very low one: Mistral says 0.15. Handing that to someone
        # writing fiction gives them flat, repetitive prose and no way to see
        # why. Creative writing is a deliberately different use, so it gets the
        # settings that make it work.
        if profile.primary_task in _OPEN_ENDED:
            if temperature < _CREATIVE_TEMPERATURE:
                notes.append(
                    f"temperature raised from the published {temperature} to "
                    f"{_CREATIVE_TEMPERATURE} for creative work, which the "
                    f"published figure is not aimed at"
                )
                temperature = _CREATIVE_TEMPERATURE
            if min_p < _CREATIVE_MIN_P:
                # This is what makes a high temperature usable rather than
                # random: it cuts the tail relative to the most likely token, so
                # the model stays coherent while still having room to choose.
                min_p = _CREATIVE_MIN_P
                notes.append(
                    f"min-p {min_p} added, which is what keeps a temperature "
                    f"this high coherent instead of merely random"
                )
            if repeat_penalty > 1.0:
                repeat_penalty = 1.0
                notes.append(
                    "repetition penalty off, since DRY handles repeated phrasing "
                    "without penalising words a passage legitimately reuses"
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

    # Match the advanced chain's disabled repetition window on every runtime,
    # including adapters that only consume this flat sampling configuration.
    if profile.primary_task in _CODE_LIKE:
        repeat_penalty = 1.0
        notes.append(
            "repetition penalty off: code and structured output may need "
            "repeated identifiers and tokens"
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
        source=published.source if published else "llmsetup task defaults",
    )
