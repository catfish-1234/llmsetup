"""The OpenCode snippet is a complete, valid provider block for the endpoint that was set up."""

from __future__ import annotations

import json

from llmsetup.interop.openai_clients import snippets
from llmsetup.runtimes.base import Endpoint
from llmsetup.types import RuntimeConfig


def test_opencode_snippet_is_a_valid_provider_for_the_served_model(catalog):
    config = RuntimeConfig(runtime="ollama", model_id="qwen3-8b", quant="Q4_K_M", context_tokens=16384)
    endpoint = Endpoint(base_url="http://127.0.0.1:11435/v1", model="qwen3:8b", runtime="ollama")
    emitted = snippets(endpoint, catalog.require("qwen3-8b"), config)["opencode"]

    parsed = json.loads(emitted["content"])
    provider = parsed["provider"]["llmsetup"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    # A pasted /v1 base URL is not doubled, and the models key is the name the server serves.
    assert provider["options"]["baseURL"] == "http://127.0.0.1:11435/v1"
    assert list(provider["models"]) == ["qwen3:8b"]
    assert provider["models"]["qwen3:8b"]["limit"]["context"] == config.context_tokens
    assert parsed["model"] == "llmsetup/qwen3:8b"
    assert emitted["format"] == "json" and "opencode.json" in emitted["path"]
