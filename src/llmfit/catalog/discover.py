"""Automatic model discovery.

The catalog is a snapshot of a landscape that turns over every few weeks, so the
only sustainable answer is for it to refresh itself. This module finds new
open-weight models and derives real catalog entries for them without a human
typing anything.

The reason this is possible at all: everything the memory model needs —
``num_hidden_layers``, ``num_key_value_heads``, ``hidden_size``, ``head_dim`` —
is published in a model's ``config.json``. So discovery reads architecture from
the source of truth rather than guessing, and an auto-added entry is as
trustworthy as a hand-written one.

What it will not do is add a model it cannot describe. An entry with a guessed
layer count produces a wrong memory plan, which is worse than no entry — so
every gate below is a reason to skip rather than to approximate.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llmfit.download import HF_ENDPOINT, _auth_headers, validate_repo
from llmfit.util.log import Console

# Quality gates. These exist to keep the catalog a curated set rather than a
# dump of everything on the Hub — a recommendation engine is only as good as
# what it is allowed to recommend.
MIN_DOWNLOADS = 5_000
MIN_LIKES = 20

#: Architectures whose config.json we know how to read correctly.
KNOWN_ARCHITECTURES = {
    "llama", "qwen2", "qwen3", "qwen3_moe", "gemma", "gemma2", "gemma3",
    "gemma3_text", "mistral", "mixtral", "phi3", "phi", "gpt_oss", "cohere",
    "deepseek_v2", "deepseek_v3", "starcoder2", "granite", "olmo2", "exaone",
    "glm4", "internlm2", "minicpm", "stablelm", "nemotron",
}

#: Substrings that mark a repo we never want in a recommendation catalog.
#: Names that mean "not a base model somebody should be handed by default".
#:
#: Three groups. Placeholders and experiments, which are not models. Repos that
#: are somebody else's quantization of a model already covered, which would
#: otherwise be catalogued as if they were distinct models. And decensored
#: derivatives, which are a legitimate thing to want and the wrong thing to
#: recommend to a user who did not ask; "heretic" is the current name for what
#: used to be called abliteration, so both are listed.
_EXCLUDE = re.compile(
    r"(?:^|[\b_-])(?:"
    r"untrained|random|test|debug|broken|dummy|toy|example|"
    r"abliterated|heretic|decensored|uncensored|nsfw|erp|roleplay-?nsfw|"
    r"awq|gptq|gguf|exl2|exl3|exllama|int4|int8|w4a16|w8a8|fp8|bnb|"
    r"lora|adapter|qlora"
    r")(?:$|[\b_-])",
    re.I,
)


@dataclass
class Candidate:
    """A discovered model, with everything needed for a catalog entry."""

    id: str
    repo: str
    publisher: str
    name: str
    params_b: float
    layers: int
    hidden_size: int
    kv_heads: int
    head_dim: int
    max_context: int
    license: str = "unknown"
    active_params_b: Optional[float] = None
    supports_vision: bool = False
    gguf_repo: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    tags: List[str] = field(default_factory=list)

    def to_entry(self) -> Dict[str, Any]:
        """Render as a catalog YAML entry."""
        entry: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "publisher": self.publisher,
            "params_b": round(self.params_b, 2),
            "layers": self.layers,
            "hidden_size": self.hidden_size,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "max_context": self.max_context,
            "license": self.license,
            "commercial_ok": _commercial_ok(self.license),
            "quality_tier": _quality_tier(self.params_b, self.downloads, self.likes),
            "tasks": _tasks_for(self.id, self.tags, self.supports_vision),
            "supports_tools": _likely_tools(self.id, self.tags),
            "supports_vision": self.supports_vision,
            "supports_json": True,
            "hf_repo": self.repo,
            "recommended_quants": ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"],
            # Read from config.json, which is the publisher's own statement.
            "arch_source": "measured",
            "discovered": True,
        }
        if self.active_params_b:
            entry["active_params_b"] = round(self.active_params_b, 2)
        if self.gguf_repo:
            entry["gguf_repo"] = self.gguf_repo
        return entry


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _get_json(url: str, timeout: float = 20.0) -> Optional[Any]:
    """GET and parse JSON. Returns None on any failure — discovery is best-effort."""
    try:
        request = urllib.request.Request(url, headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def list_trending(limit: int = 60, timeout: float = 20.0) -> List[Dict[str, Any]]:
    """Most-downloaded text-generation models on the Hub."""
    query = urllib.parse.urlencode({
        "pipeline_tag": "text-generation",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
        "full": "true",
    })
    data = _get_json(f"{HF_ENDPOINT}/api/models?{query}", timeout)
    return data if isinstance(data, list) else []


def fetch_config(repo: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    """A model's config.json — the authoritative architecture description."""
    try:
        safe = urllib.parse.quote(validate_repo(repo), safe="/")
    except Exception:
        # An invalid repo id is simply skipped, not fatal to discovery.
        return None
    data = _get_json(f"{HF_ENDPOINT}/{safe}/resolve/main/config.json", timeout)
    return data if isinstance(data, dict) else None


def find_gguf_repo(repo: str, timeout: float = 20.0) -> Optional[str]:
    """Look for a community GGUF conversion of a model.

    Most base repos ship safetensors only; llama.cpp and Ollama need GGUF, and
    the conversions live under well-known publishers.
    """
    name = repo.split("/")[-1]
    query = urllib.parse.urlencode({
        "search": f"{name} GGUF", "sort": "downloads", "direction": "-1", "limit": "5",
    })
    for item in _get_json(f"{HF_ENDPOINT}/api/models?{query}", timeout) or []:
        candidate = item.get("id") or item.get("modelId") or ""
        if candidate.lower().endswith("gguf") and name.lower() in candidate.lower():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# architecture extraction
# --------------------------------------------------------------------------- #


def architecture_from_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull the memory-model fields out of a config.json.

    Returns None when any field the memory model needs is missing. That is the
    central rule of this module: a partial description yields no entry, because
    a wrong layer count is a wrong memory plan and an out-of-memory error at
    inference time.
    """
    # Multimodal configs nest the language model one level down.
    text = config.get("text_config") or config.get("language_config") or config

    layers = _first_int(text, ("num_hidden_layers", "n_layer", "num_layers",
                               "n_layers", "block_count"))
    hidden = _first_int(text, ("hidden_size", "n_embd", "d_model", "hidden_dim"))
    heads = _first_int(text, ("num_attention_heads", "n_head", "num_heads"))
    kv_heads = _first_int(text, ("num_key_value_heads", "n_kv_heads",
                                 "num_kv_heads", "multi_query_group_num"))
    context = _first_int(text, ("max_position_embeddings", "n_positions",
                                "seq_length", "max_sequence_length"))
    head_dim = _first_int(text, ("head_dim", "head_size", "attention_head_size"))

    if not layers or not hidden:
        return None
    # Without GQA information, assume multi-head attention: kv_heads == heads.
    # That over-estimates the cache for a GQA model, which is the safe direction.
    if not kv_heads:
        kv_heads = heads
    if not kv_heads or not heads:
        return None
    if not head_dim:
        if hidden % heads:
            return None
        head_dim = hidden // heads

    return {
        "layers": layers,
        "hidden_size": hidden,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "max_context": context or 8192,
        "architecture": str(
            (config.get("model_type") or text.get("model_type") or "")
        ).lower(),
        "experts": _first_int(text, ("num_experts", "num_local_experts",
                                     "n_routed_experts")),
        "experts_per_token": _first_int(text, ("num_experts_per_tok",
                                               "num_experts_per_token")),
        "vision": bool(config.get("vision_config") or config.get("vision_tower")),
    }


def _first_int(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return None


def params_from_metadata(item: Dict[str, Any], repo: str) -> Optional[float]:
    """Parameter count in billions.

    Prefers the Hub's own safetensors parameter total, which is exact. Falls
    back to the size in the model name (``-8B``, ``-30B-A3B``), which is what
    publishers reliably encode.
    """
    total = ((item.get("safetensors") or {}).get("total"))
    if isinstance(total, (int, float)) and total > 1e8:
        return float(total) / 1e9

    match = re.search(r"[-_](\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", repo)
    if match:
        return float(match.group(1))
    return None


def active_params_from_name(repo: str) -> Optional[float]:
    """MoE active parameters, encoded by convention as ``-A3B``."""
    match = re.search(r"[-_][aA](\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", repo)
    return float(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def discover(
    console: Console,
    limit: int = 60,
    min_downloads: int = MIN_DOWNLOADS,
    min_likes: int = MIN_LIKES,
    known_ids: Optional[Sequence[str]] = None,
    find_gguf: bool = True,
    timeout: float = 20.0,
) -> List[Candidate]:
    """Find catalog-worthy models that are not already known."""
    known = {i.lower() for i in (known_ids or [])}
    found: List[Candidate] = []
    skipped: Dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    listing = list_trending(limit, timeout)
    if not listing:
        console.warn("could not reach the Hugging Face API; nothing discovered")
        return []

    console.info(f"examining {len(listing)} trending models …")
    for item in listing:
        repo = item.get("id") or item.get("modelId") or ""
        if not repo or "/" not in repo:
            skip("no repo id")
            continue
        if _EXCLUDE.search(repo):
            skip("excluded by name")
            continue

        downloads = int(item.get("downloads") or 0)
        likes = int(item.get("likes") or 0)
        if downloads < min_downloads and likes < min_likes:
            skip("below the popularity threshold")
            continue

        model_id = slug_for(repo)
        if model_id in known or repo.lower() in known:
            skip("already in the catalog")
            continue

        config = fetch_config(repo, timeout)
        if config is None:
            skip("no readable config.json")
            continue
        arch = architecture_from_config(config)
        if arch is None:
            skip("config.json lacks the architecture fields")
            continue
        if arch["architecture"] and arch["architecture"] not in KNOWN_ARCHITECTURES:
            skip(f"unrecognised architecture ({arch['architecture']})")
            continue

        params = params_from_metadata(item, repo)
        if not params:
            skip("could not determine parameter count")
            continue
        if params > 700:
            skip("too large to be runnable locally")
            continue

        gguf_repo = find_gguf_repo(repo, timeout) if find_gguf else None
        if find_gguf and not gguf_repo:
            # A catalog entry with nowhere to download from is worse than no
            # entry: it competes in the ranking, wins on paper, and then fails
            # at provisioning with nothing to fall back on.
            skip("no GGUF build published yet")
            continue

        found.append(Candidate(
            id=model_id,
            repo=repo,
            publisher=repo.split("/")[0],
            name=_display_name(repo),
            params_b=params,
            active_params_b=active_params_from_name(repo),
            layers=arch["layers"],
            hidden_size=arch["hidden_size"],
            kv_heads=arch["kv_heads"],
            head_dim=arch["head_dim"],
            max_context=arch["max_context"],
            license=_license_of(item),
            supports_vision=arch["vision"],
            gguf_repo=gguf_repo,
            downloads=downloads,
            likes=likes,
            tags=[str(t) for t in (item.get("tags") or [])],
        ))
        console.ok(f"discovered {model_id} ({params:g}B, {arch['layers']} layers)")

    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        console.debug(f"skipped {count}: {reason}")
    return found


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def slug_for(repo: str) -> str:
    """A stable, filesystem-safe catalog id derived from the repo name."""
    name = repo.split("/")[-1].lower()
    name = re.sub(r"-(?:instruct|chat|it|hf)$", "", name)
    name = re.sub(r"[^a-z0-9.\-]+", "-", name).strip("-")
    return name or repo.replace("/", "-").lower()


def _display_name(repo: str) -> str:
    return repo.split("/")[-1].replace("-", " ").replace("_", " ")


def _license_of(item: Dict[str, Any]) -> str:
    for tag in item.get("tags") or []:
        text = str(tag)
        if text.startswith("license:"):
            return text.split(":", 1)[1]
    return str((item.get("cardData") or {}).get("license") or "unknown")


_PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "cc0-1.0", "apache2"}


def _commercial_ok(license_name: str) -> bool:
    """Conservative: only licences known to permit commercial use say yes.

    An unknown licence is treated as restricted, because wrongly telling someone
    they may ship a model commercially is the expensive direction to be wrong in.
    """
    return license_name.lower() in _PERMISSIVE


def _quality_tier(params_b: float, downloads: int, likes: int) -> int:
    """A 1-5 tier from size and adoption.

    Adoption is a weak proxy for quality, but it is the only signal available
    without running benchmarks — so the scale is deliberately compressed toward
    the middle, and a discovered model never outranks a curated one on tier
    alone.
    """
    tier = 2
    if params_b >= 20:
        tier += 1
    if downloads >= 500_000 or likes >= 500:
        tier += 1
    return max(1, min(4, tier))


def _tasks_for(model_id: str, tags: Sequence[str], vision: bool) -> List[str]:
    tasks = ["chat"]
    blob = f"{model_id} {' '.join(str(t) for t in tags)}".lower()
    if any(word in blob for word in ("coder", "code", "starcoder", "codestral")):
        tasks += ["coding", "agent"]
    if vision:
        tasks.append("vision")
    if any(word in blob for word in ("instruct", "chat", "it")):
        tasks += ["summarize", "rag"]
    if "multilingual" in blob or "translation" in blob:
        tasks.append("translate")
    return sorted(set(tasks))


def _likely_tools(model_id: str, tags: Sequence[str]) -> bool:
    """Tool support is verified at setup time; this is only an initial guess.

    Being wrong here is cheap: the tool-calling check runs against the real
    endpoint and the repair loop swaps the model if it cannot call tools.
    """
    blob = f"{model_id} {' '.join(str(t) for t in tags)}".lower()
    return any(word in blob for word in
               ("instruct", "tool", "function", "agent", "coder", "chat"))
