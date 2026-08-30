"""Publisher-recommended sampling settings.

Model publishers state the sampling parameters their model was tuned and
evaluated with, and those values are not interchangeable between families:
Qwen3 asks for a presence penalty of 1.5 in non-thinking mode where most models
want none, and its thinking mode wants a different temperature again.

Ignoring that is a known cause of bad local output — the model behaves worse
than its benchmarks for reasons the user cannot see. So a publisher's numbers
are the **base**, and the task profile only adjusts within what the publisher
sanctions rather than overriding it wholesale.

Every entry records where it came from. When a family is not listed the tuner
falls back to task defaults and says so, rather than inventing a citation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Recommendation:
    """One publisher's stated sampling settings for one mode."""

    temperature: float
    top_p: float
    top_k: int = 40
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repeat_penalty: float = 1.0
    #: Values the publisher explicitly overrides for code-like output.
    code_presence_penalty: Optional[float] = None
    source: str = ""
    notes: str = ""
    #: Settings the publisher warns against.
    avoid: List[str] = field(default_factory=list)


#: Keyed by tokenizer family, then mode ("thinking" / "instruct").
#: Sources are the model cards and technical reports for each family.
RECOMMENDATIONS: Dict[str, Dict[str, Recommendation]] = {
    "qwen3": {
        "instruct": Recommendation(
            temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repeat_penalty=1.0,
            code_presence_penalty=0.0,
            source="Qwen3 model card / technical report",
            notes=("Qwen3 asks for a presence penalty of 1.5 in non-thinking mode, "
                   "which is unusual — it suppresses the repetition this family is "
                   "prone to. The publisher drops it to 0 for code, where repeated "
                   "tokens are legitimate."),
            avoid=["greedy decoding"],
        ),
        "thinking": Recommendation(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
            presence_penalty=0.0, repeat_penalty=1.0,
            source="Qwen3 model card (thinking mode)",
            notes=("Thinking mode wants a lower temperature and a wider top-p than "
                   "instruct mode. Greedy decoding is explicitly discouraged: it "
                   "sends these models into repetition loops mid-thought."),
            avoid=["greedy decoding", "temperature 0"],
        ),
    },
    "qwen2.5": {
        "instruct": Recommendation(
            temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05,
            source="Qwen2.5 model card",
        ),
        "thinking": Recommendation(
            temperature=0.6, top_p=0.95, top_k=20,
            source="DeepSeek-R1 distill guidance",
            notes=("R1 distills inherit Qwen2.5's tokenizer and DeepSeek's sampling "
                   "guidance: a narrow temperature band, and no system prompt."),
            avoid=["temperature 0", "system prompt"],
        ),
    },
    "llama3": {
        "instruct": Recommendation(
            temperature=0.6, top_p=0.9, top_k=40,
            source="Llama 3 model card",
        ),
    },
    "gemma3": {
        "instruct": Recommendation(
            temperature=1.0, top_p=0.95, top_k=64, min_p=0.0,
            source="Gemma 3 model card",
            notes=("Gemma 3 is evaluated at temperature 1.0 with a wide top-k — "
                   "much looser than most families. Turning it down to the usual "
                   "0.7 makes it noticeably flatter than its benchmarks suggest."),
        ),
    },
    "mistral": {
        "instruct": Recommendation(
            temperature=0.15, top_p=1.0, top_k=-1,
            source="Mistral Small 3.x model card",
            notes=("Mistral Small 3.x is tuned for a very low temperature — the "
                   "publisher recommends 0.15, far below the common default."),
        ),
    },
    "gptoss": {
        "thinking": Recommendation(
            temperature=1.0, top_p=1.0, top_k=0,
            source="gpt-oss model card",
            notes=("gpt-oss is served unsampled: temperature 1.0 with no "
                   "truncation. Reasoning depth is controlled by reasoning_effort "
                   "rather than by sampling."),
        ),
    },
    "phi4": {
        "instruct": Recommendation(
            temperature=0.8, top_p=0.95, top_k=50,
            source="Phi-4 model card",
        ),
    },
}


def lookup(family: str, thinking: bool) -> Optional[Recommendation]:
    """The publisher's settings for a family and mode, if stated.

    Falls back from the requested mode to the other one — a family that
    publishes only instruct values is better guidance for a thinking run than
    generic defaults are.
    """
    modes = RECOMMENDATIONS.get(family)
    if not modes:
        return None
    wanted = "thinking" if thinking else "instruct"
    return modes.get(wanted) or modes.get("instruct") or modes.get("thinking")


def known_families() -> List[str]:
    return sorted(RECOMMENDATIONS)
