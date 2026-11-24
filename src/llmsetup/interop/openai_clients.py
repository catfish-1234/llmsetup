"""Config snippets for everything that speaks the OpenAI API.

Open WebUI, Continue, Aider, Zed, Cursor and OpenCode all just need a base URL, a model
name and a dummy key. Rather than editing their config files, which are the
user's, and which we would have to merge correctly, llmsetup *emits* the exact
snippet for each and tells the user where it goes. That keeps the tool honest
about what it changed and leaves their editor config intact.
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List

from llmsetup.runtimes.base import Endpoint
from llmsetup.types import Client, ModelSpec, RuntimeConfig

# Local servers ignore the key, but most clients refuse to start without one.
DUMMY_KEY = "local-no-key-needed"


def snippets(endpoint: Endpoint, spec: ModelSpec,
             config: RuntimeConfig) -> Dict[str, Dict[str, str]]:
    """{client id: {"path": where it goes, "format": …, "content": snippet}}."""
    base = endpoint.base_url.rstrip("/")
    openai_base = f"{base}/v1"
    model = endpoint.model

    return {
        Client.CONTINUE.value: {
            "name": "Continue (VS Code / JetBrains)",
            "path": "~/.continue/config.yaml",
            "format": "yaml",
            "content": _continue_yaml(model, openai_base, spec, config),
        },
        Client.AIDER.value: {
            "name": "Aider",
            "path": "~/.aider.conf.yml  (plus the two environment variables)",
            "format": "yaml",
            "content": _aider(model, openai_base),
        },
        Client.OPEN_WEBUI.value: {
            "name": "Open WebUI",
            "path": "Settings → Connections → OpenAI API",
            "format": "text",
            "content": (f"Base URL: {openai_base}\n"
                        f"API Key:  {DUMMY_KEY}\n\n"
                        f"Or start the container with:\n"
                        f"  -e OPENAI_API_BASE_URL={openai_base} "
                        f"-e OPENAI_API_KEY={DUMMY_KEY}\n\n"
                        f"Note: inside Docker, 127.0.0.1 is the container. Use\n"
                        f"  host.docker.internal (macOS/Windows) or --network=host (Linux).\n"),
        },
        Client.ZED.value: {
            "name": "Zed",
            "path": "~/.config/zed/settings.json",
            "format": "json",
            "content": _zed(model, openai_base, config),
        },
        Client.CURSOR.value: {
            "name": "Cursor",
            "path": "Settings → Models → OpenAI API Key → Override base URL",
            "format": "text",
            "content": (f"Base URL:   {openai_base}\n"
                        f"API Key:    {DUMMY_KEY}\n"
                        f"Model name: {model}\n\n"
                        f"Cursor requires a publicly reachable URL for some features;\n"
                        f"expose the local port with a tunnel if chat does not connect.\n"),
        },
        Client.OPENCODE.value: {
            "name": "OpenCode",
            "path": "opencode.json in the project, or ~/.config/opencode/opencode.json",
            "format": "json",
            "content": _opencode(model, openai_base, config),
        },
        Client.CUSTOM_OPENAI.value: {
            "name": "Any OpenAI-compatible client / SDK",
            "path": "environment variables",
            "format": "shell",
            "content": (f"export OPENAI_BASE_URL={shlex.quote(openai_base)}\n"
                        f"export OPENAI_API_KEY={shlex.quote(DUMMY_KEY)}\n"
                        f"export OPENAI_MODEL={shlex.quote(model)}\n"),
        },
    }


def _q(value: str) -> str:
    """A YAML scalar that cannot change the document's structure.

    Model names come from the catalog and from Hugging Face, and Ollama tags
    contain colons. A JSON string is a valid YAML double-quoted scalar, so
    ``#``, ``: `` and line breaks stay inside the value.
    """
    return json.dumps(str(value), ensure_ascii=False)


def _continue_yaml(model: str, base: str, spec: ModelSpec,
                   config: RuntimeConfig) -> str:
    return f"""models:
  - name: {_q(f"{spec.name} (llmsetup)")}
    provider: openai
    model: {_q(model)}
    apiBase: {_q(base)}
    apiKey: {DUMMY_KEY}
    defaultCompletionOptions:
      contextLength: {config.context_tokens}
      temperature: {config.temperature}
      topP: {config.top_p}
    roles:
      - chat
      - edit
{"      - apply" if spec.supports_tools else ""}
{_continue_autocomplete(model, base, spec) if spec.supports_fim else ""}"""


def _continue_autocomplete(model: str, base: str, spec: ModelSpec) -> str:
    """The inline-completion role, which is a separate entry from chat.

    Continue keeps autocomplete apart from chat because the two want different
    models: one answers between keystrokes, the other reasons about a change.
    Where the same model can do both, it still has to be listed twice, because
    the roles carry different settings.
    """
    return f"""  - name: {_q(f"{spec.name} autocomplete (llmsetup)")}
    provider: openai
    model: {_q(model)}
    apiBase: {_q(base)}
    apiKey: {DUMMY_KEY}
    defaultCompletionOptions:
      # Completions are read at a glance and replaced by the next keystroke, so
      # they are kept short and cheap rather than thorough.
      maxTokens: 256
      temperature: 0.1
    roles:
      - autocomplete
"""


def _aider(model: str, base: str) -> str:
    one_line = " ".join(f"openai/{model}".split())
    return f"""# ~/.aider.conf.yml
openai-api-base: {_q(base)}
openai-api-key: {DUMMY_KEY}
model: {_q(f"openai/{model}")}

# Or on the command line:
#   aider --openai-api-base {shlex.quote(" ".join(base.split()))} \\
#         --openai-api-key {DUMMY_KEY} \\
#         --model {shlex.quote(one_line)}
"""


def _zed(model: str, base: str, config: RuntimeConfig) -> str:
    return json.dumps(
        {
            "language_models": {
                "openai": {
                    "api_url": base,
                    "available_models": [
                        {
                            "name": model,
                            "display_name": f"{model} (local)",
                            "max_tokens": config.context_tokens,
                        }
                    ],
                }
            }
        },
        indent=2,
    ) + "\n"


def _opencode(model: str, base: str, config: RuntimeConfig) -> str:
    """OpenCode's provider block: any OpenAI-compatible server through @ai-sdk/openai-compatible.

    The key under ``models`` has to be the name the server serves, which is the endpoint's model
    name; ``/models`` inside OpenCode then lists it under the llmsetup provider.
    """
    return json.dumps(
        {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "llmsetup": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "llmsetup (local)",
                    "options": {"baseURL": base, "apiKey": DUMMY_KEY},
                    "models": {
                        model: {
                            "name": f"{model} (local)",
                            "limit": {"context": config.context_tokens, "output": 4096},
                        }
                    },
                }
            },
            "model": f"llmsetup/{model}",
        },
        indent=2,
    ) + "\n"


def client_ids() -> List[str]:
    return [
        Client.CONTINUE.value, Client.AIDER.value, Client.OPEN_WEBUI.value,
        Client.ZED.value, Client.CURSOR.value, Client.OPENCODE.value, Client.CUSTOM_OPENAI.value,
    ]
