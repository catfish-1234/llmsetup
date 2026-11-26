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
    #: What to run next, for a tool that does not pick the new entry up by itself.
    then: Optional[Callable[["Target"], str]] = None


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

    @property
    def api_key(self) -> Optional[str]:
        """The key the server was started with, if any."""
        return target_key(self.endpoint)


def target_key(endpoint: Endpoint) -> Optional[str]:
    return endpoint.headers.get("Authorization", "").removeprefix("Bearer ").strip() or None


#: The environment variable tools that read their key from one are pointed at.
KEY_ENV = "LLMSETUP_API_KEY"


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


def _appdata() -> Path:
    return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")


def _windows() -> bool:
    return platform.system().lower() == "windows"


def _zed_path(project: bool) -> Path:
    if project:
        return Path.cwd() / ".zed" / "settings.json"
    if _windows():
        return _appdata() / "Zed" / "settings.json"
    return _xdg_config() / "zed" / "settings.json"


def _qwen_code_path(project: bool) -> Path:
    return (Path.cwd() if project else Path.home()) / ".qwen" / "settings.json"


def _goose_path(project: bool) -> Path:
    """Goose reads each custom provider from its own file, so llmsetup's is one it owns outright."""
    if project:
        raise ConnectRefused("Goose has no per-project provider file",
                             hint="rerun without --project; the provider is added for every Goose session "
                                  "but only used when you pick it")
    root = os.environ.get("GOOSE_PATH_ROOT")
    if root and Path(root).is_absolute():
        config = Path(root) / "config"
    elif _windows():
        config = _appdata() / "Block" / "goose" / "config"
    else:
        config = _xdg_config() / "goose"
    return config / "custom_providers" / "llmsetup.json"


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


def _merge_qwen_code(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    """Qwen Code lists OpenAI-compatible models under ``modelProviders.openai``, each reading its key
    from the environment variable ``envKey`` names, which its own ``env`` block can set."""
    env = doc.setdefault("env", {})
    providers = doc.setdefault("modelProviders", {})
    if not isinstance(env, dict) or not isinstance(providers, dict):
        raise ConnectRefused("its `env` or `modelProviders` field is not an object")
    listed = providers.setdefault("openai", [])
    if not isinstance(listed, list):
        raise ConnectRefused("its `modelProviders.openai` field is not a list")
    ours = [m for m in listed if isinstance(m, dict) and str(m.get("name", "")).endswith(MARK)]
    if any(isinstance(m, dict) and m.get("id") == target.model and m not in ours for m in listed):
        raise ConnectRefused(f"it already lists a model `{target.model}` that llmsetup did not add")
    # Local servers without a key accept any placeholder; a key the server was started with goes in.
    env[KEY_ENV] = target.api_key or "llmsetup"
    providers["openai"] = [m for m in listed if m not in ours] + [{
        "id": target.model,
        "name": f"{target.model} {MARK}",
        "envKey": KEY_ENV,
        "baseUrl": target.base,
        "generationConfig": {"contextWindowSize": target.config.context_tokens},
    }]
    if target.make_default:
        security = doc.setdefault("security", {})
        model = doc.setdefault("model", {})
        if not isinstance(security, dict) or not isinstance(model, dict):
            raise ConnectRefused("its `security` or `model` field is not an object")
        auth = security.setdefault("auth", {})
        if not isinstance(auth, dict):
            raise ConnectRefused("its `security.auth` field is not an object")
        auth["selectedType"] = "openai"
        model["name"] = target.model
    return doc


def _merge_goose(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    if doc and doc.get("name") not in (None, "llmsetup"):
        raise ConnectRefused(f"it defines a provider named {doc.get('name')!r}, not llmsetup's")
    key = target.api_key
    return {
        "name": "llmsetup",
        "engine": "openai",
        "display_name": f"{target.model} {MARK}",
        "description": "The local model llmsetup verified",
        # Goose reads the key from this variable; with none, it sends no Authorization header.
        "api_key_env": KEY_ENV if key else "",
        "base_url": target.base,
        "models": [{"name": target.model, "context_limit": target.config.context_tokens}],
        "supports_streaming": True,
        "requires_auth": bool(key),
        # The listed model only: no call to the server's model list, which may name other files.
        "dynamic_models": False,
    }


def _goose_then(target: Target) -> str:
    key = target.api_key
    posix = f"GOOSE_PROVIDER=llmsetup GOOSE_MODEL={target.model}"
    powershell = f'$env:GOOSE_PROVIDER="llmsetup"; $env:GOOSE_MODEL="{target.model}"'
    if key:
        posix = f"{KEY_ENV}={key} {posix}"
        powershell = f'$env:{KEY_ENV}="{key}"; {powershell}'
    return (f"start Goose on it with:\n  bash / zsh:  {posix} goose session\n"
            f"  PowerShell:  {powershell}; goose session\n"
            "or pick it as the provider in `goose configure`")


def crush_steps(target: Target) -> Dict[str, str]:
    key = f' --api-key "{target.api_key}"' if target.api_key else ""
    context = target.config.context_tokens
    return {
        "name": "Crush",
        "path": "crushrc in ~/.config/crush/ (all projects) or a project's folder",
        "format": "text",
        "content": (
            f'provider add llmsetup --type completions --base-url "{target.base}"{key}\n'
            f'model add llmsetup/{target.model} --name "{target.model} {MARK}" --context-window {context} '
            f"--default-max-tokens {min(4096, max(context // 4, 256))}\n"
            f"model large llmsetup/{target.model}\n"
            f"model small llmsetup/{target.model}\n"),
    }


def claude_code_env(target: Target) -> Dict[str, str]:
    """What points Claude Code at the local server, which must speak the Anthropic Messages API.

    The base URL is the server root, without /v1. Every model alias Claude Code asks for (its
    main model, and the small one it uses for background work) is the local model, so it never
    asks the server for a model name it does not serve.
    """
    token = target.api_key or "llmsetup"
    model = target.model
    return {
        "ANTHROPIC_BASE_URL": target.endpoint.base_url.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
    }


def _claude_code_path(project: bool) -> Path:
    if not project:
        # The user-wide settings would send every Claude Code session to the local model.
        raise ConnectRefused("connect writes Claude Code's settings only for one project",
                             hint="rerun with --project in that project's folder, or launch it with the "
                                  "command `llmsetup connect claude-code` prints")
    return Path.cwd() / ".claude" / "settings.local.json"


def _merge_claude_code(doc: Dict[str, Any], target: Target) -> Dict[str, Any]:
    env = doc.setdefault("env", {})
    if not isinstance(env, dict):
        raise ConnectRefused("its `env` field is not an object")
    env.update(claude_code_env(target))
    return doc


#: Claude Code's system prompt and tool definitions alone run to tens of thousands of tokens.
CLAUDE_CODE_MIN_CONTEXT = 32768


def claude_code_steps(target: Target) -> Dict[str, str]:
    env = claude_code_env(target)
    posix = " ".join(f"{k}={v}" for k, v in env.items())
    powershell = "; ".join(f'$env:{k}="{v}"' for k, v in env.items())
    note = ""
    if target.config.context_tokens < CLAUDE_CODE_MIN_CONTEXT:
        note = (f"\nThe verified context is {target.config.context_tokens:,} tokens; Claude Code's prompts "
                f"need at least {CLAUDE_CODE_MIN_CONTEXT:,}. Rerun setup with "
                f"--set context_tokens={CLAUDE_CODE_MIN_CONTEXT * 2} if it fits.\n")
    return {
        "name": "Claude Code",
        "path": "per launch, or a project's .claude/settings.local.json with --project",
        "format": "text",
        "content": (
            "The server must speak the Anthropic Messages API: llama.cpp from January 2026, "
            "Ollama, and LM Studio 0.4.1 or later do.\n\n"
            f"bash / zsh:\n  {posix} claude\n\n"
            f"PowerShell:\n  {powershell}; claude\n\n"
            "Or for one project, which writes .claude/settings.local.json (undo restores it):\n"
            f"  llmsetup connect claude-code --project\n{note}"),
    }


def codex_steps(target: Target) -> Dict[str, str]:
    return {
        "name": "Codex CLI",
        "path": "~/.codex/config.toml",
        "format": "toml",
        "content": (
            "[model_providers.llmsetup]\n"
            'name = "llmsetup (local)"\n'
            f'base_url = "{target.base}"\n'
            'wire_api = "responses"\n\n'
            "[profiles.llmsetup]\n"
            'model_provider = "llmsetup"\n'
            f'model = "{target.model}"\n\n'
            "# then: codex --profile llmsetup\n"
            "# Codex speaks only the Responses API. Some local servers reject the non-function\n"
            "# tools recent Codex releases send; if it fails on tools, OpenCode or Claude Code\n"
            "# (`llmsetup connect opencode`, `llmsetup connect claude-code`) work with this model.\n"),
    }


HARNESSES: Dict[str, Harness] = {
    "opencode": Harness("opencode", "OpenCode", "json", _opencode_path, _merge_opencode),
    "continue": Harness("continue", "Continue (VS Code / JetBrains)", "yaml", _continue_path, _merge_continue),
    "aider": Harness("aider", "Aider", "yaml", _aider_path, _merge_aider),
    "zed": Harness("zed", "Zed", "json", _zed_path, _merge_zed),
    "claude_code": Harness("claude_code", "Claude Code", "json", _claude_code_path, _merge_claude_code),
    "qwen_code": Harness("qwen_code", "Qwen Code", "json", _qwen_code_path, _merge_qwen_code),
    "goose": Harness("goose", "Goose", "json", _goose_path, _merge_goose, then=_goose_then),
}

#: Steps for the tools connect prints instructions for rather than editing a file.
STEPS: Dict[str, Callable[[Target], Dict[str, str]]] = {
    "claude_code": claude_code_steps,
    "codex": codex_steps,
    "crush": crush_steps,
}

#: Configured in the tool's own interface, or through a file connect does not edit: it prints the steps.
INSTRUCTIONS_ONLY = {"open_webui", "cursor", "custom_openai", "codex", "crush"}


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
    snippet = _snippet(harness_id, target)

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


def _snippet(harness_id: str, target: Target) -> Dict[str, str]:
    if harness_id in STEPS:
        return STEPS[harness_id](target)
    clients = snippets(target.endpoint, target.spec, target.config)
    if harness_id in clients:
        return clients[harness_id]
    # A tool with no `llmsetup export` snippet: what connect would write is the snippet.
    harness = HARNESSES[harness_id]
    return {"name": harness.name, "path": str(harness.path(False)), "format": harness.format,
            "content": json.dumps(harness.merge({}, target), indent=2) + "\n"}


def instructions(harness_id: str, target: Target) -> Dict[str, str]:
    """For a tool connect does not write to: where to go and what to enter."""
    return _snippet(harness_id, target)


def describe(harness_id: str) -> str:
    """One line for ``connect --list``: what connect does for this tool."""
    if harness_id == "claude_code":
        return "prints the launch command; --project writes .claude/settings.local.json"
    if harness_id in HARNESSES:
        try:
            return f"writes {HARNESSES[harness_id].path(False)}"
        except ConnectRefused:
            return f"writes {HARNESSES[harness_id].path(True)} (with --project)"
    return {"codex": "prints the ~/.codex/config.toml provider and profile",
            "crush": "prints the crushrc lines that add it"}.get(harness_id, "prints the steps for its settings screen")


def supported() -> List[str]:
    return [*HARNESSES, *sorted(INSTRUCTIONS_ONLY)]
