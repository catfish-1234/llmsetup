"""Failure signatures — the shared vocabulary between verify and repair.

A check never explains *how* to fix something; it names *what went wrong* using
one of these tokens. The repair table maps tokens to remediation strategies.
Keeping the join key here, in one place, is what stops the two halves drifting
apart.
"""

from __future__ import annotations

import re
from typing import Optional

# --- server / lifecycle ---------------------------------------------------- #
SERVER_UNREACHABLE = "server_unreachable"
MODEL_NOT_LOADED = "model_not_loaded"
CRASH_DURING_LOAD = "crash_during_load"

# --- memory ---------------------------------------------------------------- #
OUT_OF_MEMORY = "out_of_memory"
CONTEXT_EXCEEDED = "context_exceeded"
KV_CACHE_FULL = "kv_cache_full"

# --- output quality -------------------------------------------------------- #
EMPTY_RESPONSE = "empty_response"
INCOHERENT_OUTPUT = "incoherent_output"
REPETITION_LOOP = "repetition_loop"
NEEDLE_NOT_FOUND = "needle_not_found"

# --- capabilities ---------------------------------------------------------- #
TOOLS_UNSUPPORTED = "tools_unsupported"
TOOL_CALL_MALFORMED = "tool_call_malformed"
JSON_INVALID = "json_invalid"
JSON_MODE_UNSUPPORTED = "json_mode_unsupported"
STREAMING_UNSUPPORTED = "streaming_unsupported"
EMBEDDINGS_UNSUPPORTED = "embeddings_unsupported"
VISION_UNSUPPORTED = "vision_unsupported"

# --- performance ----------------------------------------------------------- #
THROUGHPUT_TOO_LOW = "throughput_too_low"
LATENCY_TOO_HIGH = "latency_too_high"
REQUEST_TIMEOUT = "request_timeout"

# --- unknown --------------------------------------------------------------- #
UNKNOWN_ERROR = "unknown_error"


# Ordered most-specific first: an OOM message often also contains the word
# "error", so generic patterns must never win.
_PATTERNS = [
    (OUT_OF_MEMORY, re.compile(
        r"out of memory|cuda(?:_| )error(?:.*)?out of memory|oom|"
        r"failed to allocate|cudamalloc|hipmalloc|insufficient memory|"
        r"unable to allocate|ggml_backend_alloc|not enough memory|"
        r"vk::.*outofdevicememory|mps backend out of memory", re.I)),
    (CONTEXT_EXCEEDED, re.compile(
        r"context (?:length|window|size) exceeded|maximum context length|"
        r"too many tokens|exceeds? the (?:model|maximum|available) context|"
        r"exceeds? the available context size|"
        r"requested (?:\d+) tokens.*maximum|n_tokens.*n_ctx|prompt is too long|"
        r"input (?:is )?too long", re.I)),
    (KV_CACHE_FULL, re.compile(
        r"kv cache (?:is )?full|no (?:free )?kv (?:cache )?slot|"
        r"cache_prompt.*full|slot.*unavailable", re.I)),
    (TOOLS_UNSUPPORTED, re.compile(
        r"tools? (?:are )?not supported|does not support tools|"
        r"tool_choice.*not supported|no tool support|function calling.*not", re.I)),
    (JSON_MODE_UNSUPPORTED, re.compile(
        r"response_format.*(?:not supported|unsupported|invalid)|"
        r"json_schema.*not supported|grammar.*not supported", re.I)),
    (EMBEDDINGS_UNSUPPORTED, re.compile(
        r"embeddings? (?:are )?not supported|"
        r"model does not support embeddings|--embeddings?\b.*required|"
        r"pooling type", re.I)),
    (VISION_UNSUPPORTED, re.compile(
        r"image (?:input )?(?:is )?not supported|no (?:vision|mmproj|multimodal)|"
        r"image_url.*not supported|does not support images", re.I)),
    (MODEL_NOT_LOADED, re.compile(
        r"model\b[^.\n]{0,80}?\bnot (?:found|loaded)|model is not loaded|"
        r"no model (?:loaded|is loaded)|unknown model|404.*model|"
        r"failed to load model|try pulling it first", re.I)),
    (REQUEST_TIMEOUT, re.compile(r"timed out|timeout|read operation timed out", re.I)),
    (SERVER_UNREACHABLE, re.compile(
        r"connection refused|connection reset|failed to establish|"
        r"cannot connect|no route to host|urlopen error|actively refused", re.I)),
]


def classify(text: str) -> Optional[str]:
    """Map an error string to a signature. None when nothing matches.

    Returning None rather than UNKNOWN_ERROR is deliberate: the caller decides
    whether an unrecognised error is worth a generic signature, and a wrong
    classification is worse than none — it sends repair down the wrong path.
    """
    if not text:
        return None
    for signature, pattern in _PATTERNS:
        if pattern.search(text):
            return signature
    return None


def classify_or_unknown(text: str) -> str:
    return classify(text) or UNKNOWN_ERROR
