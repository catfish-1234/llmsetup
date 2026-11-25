"""Any Hugging Face GGUF repo, or a named variant of a catalog model, as a catalog entry.

The bundled catalog is curated: it is what llmsetup recommends. Some people want a model it does
not list: a fine-tune, a quantization from a publisher they trust, or a decensored ("abliterated")
derivative, which discovery deliberately never recommends to someone who did not ask for one.
These are the ways to ask:

    llmsetup setup --model hf:bartowski/SomeModel-GGUF
    llmsetup setup --model qwen3-8b --variant abliterated

Either way the result is an ordinary entry in your overlay catalog ($LLMSETUP_HOME/catalog.d),
planned, fitted and verified like every other model. A variant keeps its base model's
architecture, which a fine-tune or abliteration does not change, so its memory plan is as exact
as the base's, and it drops every pointer to the base's own weights (its Ollama tag, its
safetensors and MLX repos), so nothing can quietly serve the original under the variant's name.
A repo of unknown lineage takes its architecture from its own config.json or its base model's;
without either the plan is marked estimated, and the GGUF header corrects it once downloaded.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from llmsetup.catalog.discover import (
    _QUANT_RE,
    _get_json,
    _license_of,
    _likely_tools,
    _tasks_for,
    active_params_from_name,
    architecture_from_config,
    fetch_config,
    params_from_metadata,
    slug_for,
)
from llmsetup.download import HF_ENDPOINT, is_projector, validate_repo
from llmsetup.types import ModelSpec
from llmsetup.util.errors import CatalogError

GetJson = Callable[..., Optional[Any]]

#: Names a technique goes by. Abliteration is now mostly published as "heretic".
VARIANT_ALIASES: Dict[str, tuple] = {
    "abliterated": ("abliterated", "heretic"),
    "heretic": ("heretic", "abliterated"),
    "uncensored": ("uncensored", "abliterated", "heretic"),
}

#: Fields that point at a model's weights; a variant replaces them all with its own GGUF repo.
SOURCE_FIELDS = ("ollama_name", "hf_repo", "mlx_repo", "gguf_revision", "hf_revision", "mlx_revision")


def parse_hub_ref(value: str) -> Optional[str]:
    """``owner/repo`` from ``hf:owner/repo`` or a huggingface.co URL; None for anything else."""
    text = value.strip()
    match = re.match(r"^(?:hf:|(?:https?://)?(?:www\.)?huggingface\.co/)([\w.-]+/[\w.-]+)(?:/.*)?$", text)
    return match.group(1) if match else None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _repo_files(repo: str, get_json: GetJson) -> Optional[Dict[str, Any]]:
    info = get_json(f"{HF_ENDPOINT}/api/models/{urllib.parse.quote(repo, safe='/')}")
    return info if isinstance(info, dict) else None


def _gguf_facts(info: Dict[str, Any]) -> Dict[str, Any]:
    """Quantizations published, and whether a vision projector ships beside them."""
    names = [str(s.get("rfilename") or "") for s in info.get("siblings") or []]
    ggufs = [n for n in names if n.lower().endswith(".gguf")]
    quants = sorted({m.group(1) for n in ggufs if not is_projector(n) for m in [_QUANT_RE.search(n)] if m})
    return {"quants": quants, "projector": any(is_projector(n) for n in ggufs), "has_gguf": bool(ggufs)}


def find_variants(spec: ModelSpec, keyword: str, *, get_json: GetJson = _get_json,
                  check: int = 5) -> List[Dict[str, Any]]:
    """GGUF repos of ``keyword`` derivatives of ``spec``, most downloaded first.

    A candidate must name the base model (so a different size of the same family is not taken
    for it), carry one of the keyword's names, and publish GGUF quantizations.
    """
    base = (spec.hf_repo or spec.gguf_repo or spec.id).split("/")[-1]
    base = re.sub(r"(?i)[-_.]gguf$", "", base)
    base_key = _norm(base)
    terms = VARIANT_ALIASES.get(keyword.lower(), (keyword.lower(),))

    seen: Dict[str, Dict[str, Any]] = {}
    for search in [base, *(f"{base}-{term}" for term in terms)]:
        query = urllib.parse.urlencode({"search": search, "filter": "gguf", "sort": "downloads",
                                        "direction": "-1", "limit": "100"})
        for item in get_json(f"{HF_ENDPOINT}/api/models?{query}") or []:
            repo = str(item.get("id") or item.get("modelId") or "")
            lowered = repo.lower()
            if not repo or repo in seen or base_key not in _norm(repo.split("/")[-1]):
                continue
            if not any(term in lowered for term in terms):
                continue
            seen[repo] = {"repo": repo, "downloads": int(item.get("downloads") or 0),
                          "likes": int(item.get("likes") or 0)}

    ranked = sorted(seen.values(), key=lambda c: (c["downloads"], c["likes"]), reverse=True)
    usable = []
    for candidate in ranked[:check]:
        info = _repo_files(candidate["repo"], get_json)
        facts = _gguf_facts(info) if info else None
        if facts and facts["quants"]:
            usable.append({**candidate, **facts})
    return usable


def variant_entry(spec: ModelSpec, keyword: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """The catalog entry for one variant: the base's architecture, the variant's weights."""
    entry = {k: v for k, v in asdict(spec).items() if v is not None and k not in SOURCE_FIELDS}
    published = list(candidate["quants"])
    preferred = [q for q in spec.recommended_quants if q in published]
    entry.update({
        "id": f"{spec.id}-{_norm(keyword)}",
        "name": f"{spec.name} ({keyword})",
        "publisher": candidate["repo"].split("/")[0],
        "gguf_repo": candidate["repo"],
        "recommended_quants": preferred or published[:4],
        # A repo without a projector serves the language model only.
        "supports_vision": bool(spec.supports_vision and candidate.get("projector")),
        "notes": (f"{keyword} derivative of {spec.name} from {candidate['repo']}, chosen explicitly "
                  f"with --variant. Same architecture as {spec.id}; its behaviour and licence terms "
                  f"are the publisher's, so read the model card."),
    })
    return entry


def repo_entry(repo: str, *, get_json: GetJson = _get_json,
               config_for: Callable[[str], Optional[Dict[str, Any]]] = fetch_config) -> Dict[str, Any]:
    """The catalog entry for an arbitrary Hugging Face GGUF repo."""
    repo = validate_repo(repo)
    info = _repo_files(repo, get_json)
    if info is None:
        raise CatalogError(f"could not read {repo} from Hugging Face",
                           hint="check the name, your network, and HF_TOKEN for a gated repo")
    facts = _gguf_facts(info)
    if not facts["has_gguf"]:
        raise CatalogError(f"{repo} publishes no GGUF files, which llmsetup needs",
                           hint="look for a GGUF conversion of it (repos usually ending in -GGUF)")
    card = info.get("cardData") or {}
    bases = card.get("base_model") or []
    first = bases[0] if isinstance(bases, list) and bases else bases
    base: Optional[str] = first if isinstance(first, str) and first else None

    config = config_for(repo) or (config_for(base) if base else None)
    arch = architecture_from_config(config) if config else None
    base_info = _repo_files(base, get_json) if base else None
    params = params_from_metadata(info, repo)
    if params is None and base_info is not None and base:
        params = params_from_metadata(base_info, str(base))
    if params is None:
        raise CatalogError(f"cannot tell how large {repo} is",
                           hint=f"add it with `llmsetup catalog add <id> --gguf-repo {repo} --params <billions>`")
    tags = [str(t) for t in info.get("tags") or []]
    entry: Dict[str, Any] = {
        "id": slug_for(repo),
        "name": repo.split("/")[-1],
        "publisher": repo.split("/")[0],
        "params_b": round(float(params), 3),
        "license": _license_of(info),
        "tasks": _tasks_for(repo, tags, bool(arch and arch.get("vision"))),
        "supports_tools": _likely_tools(repo, tags),
        "supports_vision": bool(arch and arch.get("vision") and facts["projector"]),
        "gguf_repo": repo,
        "recommended_quants": [q for q in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0") if q in facts["quants"]]
        or facts["quants"][:4],
        "arch_source": "measured" if arch else "estimated",
        "notes": (f"Added from {repo} with --model hf:{repo}."
                  + ("" if arch else " Architecture is estimated until the GGUF header is read on download.")),
    }
    active = active_params_from_name(repo)
    if active:
        entry["active_params_b"] = active
    if arch:
        entry.update({k: arch[k] for k in ("layers", "hidden_size", "kv_heads", "head_dim", "max_context")})
    return entry
