"""The question bank.

Questions are data so the same definitions drive the interactive wizard, the
non-interactive answers file, and the CLI flag validation. Each carries the
*reason* it is asked, which the wizard shows, people give better answers when
they know why a question matters, and "how long is your context" is meaningless
without knowing it decides model size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class Option:
    value: str
    label: str
    hint: str = ""


@dataclass
class Question:
    key: str
    prompt: str
    kind: str                      # choice | multi | int | bool | text
    options: List[Option] = field(default_factory=list)
    default: Any = None
    why: str = ""
    #: Only asked when this returns True for the answers gathered so far.
    when: Optional[Callable[[dict], bool]] = None
    parse: Optional[Callable[[str], Any]] = None


TASK_OPTIONS = [
    Option("chat", "General chat and Q&A", "the everyday assistant case"),
    Option("coding", "Writing and reviewing code", "wants a code-tuned model"),
    Option("agent", "Agents and tool use", "requires reliable function calling"),
    Option("rag", "Questions over my own documents", "needs long context + embeddings"),
    Option("summarize", "Summarizing long documents", "long context matters most"),
    Option("creative", "Creative and long-form writing", "favours varied sampling"),
    Option("extract", "Pulling structured data out of text", "needs JSON output"),
    Option("translate", "Translation", "needs a multilingual model"),
    Option("vision", "Describing or reading images", "restricts to vision models"),
]

CLIENT_OPTIONS = [
    Option("lm_studio", "LM Studio"),
    Option("ollama", "Ollama"),
    Option("open_webui", "Open WebUI"),
    Option("jan", "Jan"),
    Option("continue", "Continue (VS Code / JetBrains)"),
    Option("aider", "Aider"),
    Option("zed", "Zed"),
    Option("cursor", "Cursor"),
    Option("llama_cpp", "llama.cpp directly"),
    Option("vllm", "vLLM"),
    Option("custom_openai", "My own code, via the OpenAI API"),
]

PRIORITY_OPTIONS = [
    Option("speed", "Speed", "smaller model, more tokens per second"),
    Option("balanced", "Balanced", "the usual right answer"),
    Option("quality", "Quality", "the best model that fits, even if slower"),
]

CONTEXT_OPTIONS = [
    Option("4096", "4K, short chats", "smallest memory footprint"),
    Option("8192", "8K, normal conversation", "a sensible default"),
    Option("32768", "32K, long documents and codebases", "costs real VRAM"),
    Option("65536", "64K, large document sets", "often decides the model size"),
    Option("131072", "128K, everything at once", "the KV cache may exceed the weights"),
]

QUESTIONS: List[Question] = [
    Question(
        key="tasks", prompt="What will you use the model for?", kind="multi",
        options=TASK_OPTIONS, default=["chat"],
        why="This decides more than your hardware does: it filters which models "
            "are eligible and which capabilities get tested afterwards.",
    ),
    Question(
        key="clients", prompt="Which apps do you want to use it from?", kind="multi",
        options=CLIENT_OPTIONS, default=[],
        why="Everything you pick gets the model registered or a ready-made config "
            "snippet, so the model is not trapped in one tool.",
    ),
    Question(
        key="context_tokens", prompt="How much context do you need?", kind="choice",
        options=CONTEXT_OPTIONS, default="8192",
        why="The KV cache grows linearly with this and can easily outgrow the "
            "model's own weights, so it directly decides how large a model fits.",
        parse=int,
    ),
    Question(
        key="priority", prompt="If speed and quality conflict, which wins?",
        kind="choice", options=PRIORITY_OPTIONS, default="balanced",
        why="Shifts the ranking weights; it genuinely changes the answer rather "
            "than reordering the same list.",
    ),
    Question(
        key="concurrent_users",
        prompt="How many people or agents will use it at the same time?",
        kind="int", default=1,
        why="Each concurrent request needs its own KV cache slot, so this "
            "multiplies memory and can change which engine is right.",
    ),
    Question(
        key="needs_tools", prompt="Do you need reliable tool / function calling?",
        kind="bool", default=None,
        why="Not every open model does this well, and runtimes need specific flags "
            "for it. Saying yes makes it a filter and a required test.",
        when=lambda a: "agent" not in a.get("tasks", []),
    ),
    Question(
        key="needs_json", prompt="Do you need guaranteed structured JSON output?",
        kind="bool", default=None,
        why="Enables constrained decoding where the runtime supports it, and adds "
            "a schema-validation test.",
        when=lambda a: "extract" not in a.get("tasks", []),
    ),
    Question(
        key="needs_vision", prompt="Do you need to send images to the model?",
        kind="bool", default=None,
        why="Restricts the catalog to vision models, which is a much smaller set.",
        when=lambda a: "vision" not in a.get("tasks", []),
    ),
    Question(
        key="needs_embeddings",
        prompt="Do you need embeddings (for search or RAG)?", kind="bool", default=None,
        why="Adds a second, small embedding model, a chat model makes a poor "
            "retriever, so these are separate jobs.",
        when=lambda a: "rag" not in a.get("tasks", []),
    ),
    Question(
        key="languages", prompt="Which languages, as comma-separated codes?",
        kind="text", default="en",
        why="Multilingual work needs a model trained for it, and changes which "
            "embedding model is appropriate.",
        parse=lambda s: [p.strip().lower() for p in s.split(",") if p.strip()],
    ),
    Question(
        key="commercial_use", prompt="Will you use this commercially?", kind="bool",
        default=False,
        why="Filters out models whose licence does not permit it, so you do not "
            "find out later.",
    ),
    Question(
        key="max_disk_gb", prompt="Disk budget for weights, in GB (blank for no limit)?",
        kind="text", default="",
        why="Caps download size; the largest models are tens of gigabytes.",
        parse=lambda s: float(s) if s.strip() else None,
    ),
]


def question_by_key(key: str) -> Optional[Question]:
    return next((q for q in QUESTIONS if q.key == key), None)


def infer_flags(answers: dict) -> dict:
    """Derive capability requirements implied by the chosen tasks.

    Someone who says "agents" has said they need tool calling, whether or not
    they know the term. Inferring it here means the planner filters correctly and
    the verifier tests it, without making the user answer a question they already
    effectively answered.
    """
    tasks = set(answers.get("tasks") or [])
    derived = dict(answers)

    if "agent" in tasks:
        derived.setdefault("needs_tools", True)
    if "extract" in tasks:
        derived.setdefault("needs_json", True)
    if "vision" in tasks:
        derived.setdefault("needs_vision", True)
    if "rag" in tasks:
        derived.setdefault("needs_embeddings", True)
        # Retrieval over documents is not useful in a short window.
        if int(derived.get("context_tokens") or 0) < 16384:
            derived["context_tokens"] = 32768
    return {k: v for k, v in derived.items() if v is not None}
