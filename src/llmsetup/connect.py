"""Point another tool at the verified model by writing its own config: ``llmsetup connect``.

`llmsetup export` writes a snippet per client and leaves pasting it in to you. `connect` does the
pasting: it merges an llmsetup entry into the tool's config file, keeps everything else in it, and
journals the edit with a copy of the file as it was, so `llmsetup undo` puts it back exactly.

Two promises. It only ever replaces what llmsetup itself added there: the ``llmsetup`` provider,
entries named ``... (llmsetup)``, the model it lists. And it never quietly drops your comments: a
JSON or YAML rewrite loses them, so a file that has any is left alone and the snippet printed,
unless ``--rewrite`` is given (the original is still backed up and restored by undo).

Tools configured only through their own UI (Open WebUI, Cursor) get the exact steps instead.
"""

from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from llmsetup import changes
from llmsetup.interop.openai_clients import DUMMY_KEY, snippets
from llmsetup.runtimes.base import Endpoint
from llmsetup.types import ModelSpec, RuntimeConfig
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import ensure_dir, write_text

#: The name every entry connect writes carries, which is also how it recognises its own entries.
MARK = "(llmsetup)"


class ConnectRefused(LlmsetupError):
    """The file could not be edited safely; the message says why and what to paste instead."""


@dataclass
class Harness:
    id: str
    name: str
    format: str  # json | yaml
    path: Callable[[bool], Path]
    merge: Callable[[Dict[str, Any], "Target"], Dict[str, Any]]


@dataclass
class Target:
    """What is being connected: the endpoint and what the verified setup knows about it."""

    endpoint: Endpoint
    spec: ModelSpec
    config: RuntimeConfig
    make_default: bool = False

    @property
    def base(self) -> str:
        return f"{self.endpoint.base_url.rstrip('/')}/v1"

    @property
    def model(self) -> str:
        return self.endpoint.model


# --------------------------------------------------------------------------- #
# where each tool keeps its config
# --------------------------------------------------------------------------- #


def _xdg_config() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def _opencode_path(project: bool) -> Path:
    return Path.cwd() / "opencode.json" if project else _xdg_config() / "opencode" / "opencode.json"


def _continue_path(project: bool) -> Path:
    base = os.environ.get("CONTINUE_GLOBAL_DIR")
    return Path(base) / "config.yaml" if base else Path.home() / ".continue" / "config.yaml"


def _aider_path(project: bool) -> Path:
    return Path.cwd() / ".aider.conf.yml" if project else Path.home() / ".aider.conf.yml"


def _zed_path(project: bool) -> Path:
    if project:
        return Path.cwd() / ".zed" / "settings.json"
    if platform.system().lower() == "windows":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Zed" / "settings.json"
    return _xdg_config() / "zed" / "settings.json"


# --------------------------------------------------------------------------- #
# merging the llmsetup entry into each tool's document
# --------------------------------------------------------------------------- #


def _merge_opencode(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    entry = json.loads(snippets(target.endpoint, target.spec, target.config)["opencode"]["content"])
    doc.setdefault("$schema", entry["$schema"])
    providers = doc.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise ConnectRefused("its `provider` field is not an object, so it is not a config connect understands")
    providers["llmsetup"] = entry["provider"]["llmsetup"]
    if target.make_default:
        doc["model"] = entry["model"]
    return doc


def _merge_continue(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    ours = yaml.safe_load(snippets(target.endpoint, target.spec, target.config)["continue"]["content"])["models"]
    models = doc.get("models") or []
    if not isinstance(models, list):
        raise ConnectRefused("its `models` field is not a list, so it is not a config connect understands")
    kept = [m for m in models if not (isinstance(m, dict) and str(m.get("name", "")).endswith(MARK))]
    doc["models"] = kept + ours
    return doc


def _merge_aider(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    doc["openai-api-base"] = target.base
    doc["openai-api-key"] = DUMMY_KEY
    doc["model"] = f"openai/{target.model}"
    return doc


def _merge_zed(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    models = doc.setdefault("language_models", {})
    if not isinstance(models, dict):
        raise ConnectRefused("its `language_models` field is not an object")
    openai = models.setdefault("openai", {})
    current = str(openai.get("api_url") or "")
    # Zed has one OpenAI provider. Pointing it at llmsetup when it already points at a real
    # OpenAI-compatible service would quietly break that, so only a local or empty URL is replaced.
    if current and not re.match(r"https?://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:/|$)", current):
        raise ConnectRefused(f"Zed's OpenAI provider already points at {current}, which connect will not replace")
    openai["api_url"] = target.base
    listed = openai.get("available_models") or []
    entry = {"name": target.model, "display_name": f"{target.model} {MARK}",
             "max_tokens": target.config.context_tokens}
    openai["available_models"] = [m for m in listed if not (isinstance(m, dict) and m.get("name") == target.model)] + [entry]
    return doc


HARNESSES: Dict[str, Harness] = {
    "opencode": Harness("opencode", "OpenCode", "json", _opencode_path, _merge_opencode),
    "continue": Harness("continue", "Continue (VS Code / JetBrains)", "yaml", _continue_path, _merge_continue),
    "aider": Harness("aider", "Aider", "yaml", _aider_path, _merge_aider),
    "zed": Harness("zed", "Zed", "json", _zed_path, _merge_zed),
}

#: Configured in the tool's own interface, so connect prints the steps.
INSTRUCTIONS_ONLY = {"open_webui", "cursor", "custom_openai"}


# --------------------------------------------------------------------------- #
# comments, which a rewrite would lose
# --------------------------------------------------------------------------- #


def _json_comments(text: str) -> bool:
    """True when JSON text carries // or /* */ comments outside its strings (JSONC)."""
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "/" and text[i + 1:i + 2] in ("/", "*"):
            return True
    return False


def strip_jsonc(text: str) -> str:
    """JSONC to JSON: comments outside strings removed, and trailing commas."""
    out: List[str] = []
    i, in_string, escaped = 0, False, False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
        elif ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
        else:
            out.append(ch)
            i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _yaml_comments(text: str) -> bool:
    for line in text.splitlines():
        bare = re.sub(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', "", line)
        if re.search(r"(?:^|\s)#", bare):
            return True
    return False


# --------------------------------------------------------------------------- #
# connecting
# --------------------------------------------------------------------------- #


def connect(harness_id: str, target: Target, *, path: Optional[Path] = None, project: bool = False,
            rewrite: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """Merge the llmsetup entry into one tool's config. Returns what was (or would be) done."""
    harness = HARNESSES[harness_id]
    path = path or harness.path(project)
    original = path.read_text(encoding="utf-8") if path.exists() else None
    snippet = snippets(target.endpoint, target.spec, target.config)[harness_id]

    def refuse(reason: str) -> ConnectRefused:
        return ConnectRefused(
            f"{path} was not changed: {reason}",
            hint=(f"paste this into it yourself, or rerun with --rewrite to let llmsetup rewrite the "
                  f"file (the original is backed up, and `llmsetup undo` restores it):\n\n{snippet['content']}"))

    doc: Dict[str, Any] = {}
    if original is not None and original.strip():
        commented = _json_comments(original) if harness.format == "json" else _yaml_comments(original)
        if commented and not rewrite:
            raise refuse("it has comments, which rewriting it would remove")
        try:
            loaded = json.loads(strip_jsonc(original)) if harness.format == "json" else yaml.safe_load(original)
        except (ValueError, yaml.YAMLError) as exc:
            raise refuse(f"it could not be parsed ({exc})") from None
        if loaded is not None and not isinstance(loaded, dict):
            raise refuse("its top level is not a mapping")
        doc = loaded or {}

    try:
        merged = harness.merge(doc, target)
    except ConnectRefused as exc:
        raise refuse(exc.message) from None
    if harness.format == "json":
        text = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    else:
        text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)

    result = {"harness": harness_id, "name": harness.name, "path": str(path), "model": target.model,
              "base_url": target.base, "changed": text != original, "content": text}
    if not result["changed"] or dry_run:
        return result
    new_folders = []
    folder = path.parent
    while not folder.exists():
        new_folders.append(folder)
        folder = folder.parent
    backup = changes.backup_before_edit(path)
    ensure_dir(path.parent)
    for created in reversed(new_folders):
        changes.created_dir(harness_id, created)
    write_text(path, text)
    changes.modified(harness_id, path, backup)
    result["backup"] = str(backup) if backup else None
    return result


def instructions(harness_id: str, target: Target) -> Dict[str, str]:
    """For a tool configured in its own UI: where to go and what to enter."""
    return snippets(target.endpoint, target.spec, target.config)[harness_id]


def supported() -> List[str]:
    return [*HARNESSES, *sorted(INSTRUCTIONS_ONLY)]
