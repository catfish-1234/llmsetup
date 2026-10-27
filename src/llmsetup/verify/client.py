"""A tiny OpenAI-compatible client for the verifier.

Every runtime llmsetup supports exposes an OpenAI-shaped API, so the checks are
written once against this client rather than per engine. Errors are returned as
values, never raised: a check needs the failure *text* to classify it, and an
exception loses that.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llmsetup.runtimes.base import Endpoint
from llmsetup.util import http


@dataclass
class ChatResult:
    ok: bool
    text: str = ""
    status: int = 0
    error: str = ""
    raw: Optional[dict] = None
    latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: str = ""
    tool_calls: List[dict] = field(default_factory=list)
    #: The thinking block, when the server splits it out of the visible answer
    #: (llama.cpp ``reasoning_content``, Ollama ``thinking``). Empty when the
    #: server folds thinking into the content or the model does not reason.
    reasoning_text: str = ""

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.completion_tokens and self.latency_s > 0:
            return self.completion_tokens / self.latency_s
        return None


class OpenAIClient:
    def __init__(self, endpoint: Endpoint, timeout: float = 180.0):
        self.endpoint = endpoint
        self.timeout = timeout

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        tools: Optional[List[dict]] = None,
        response_format: Optional[dict] = None,
        timeout: Optional[float] = None,
        **extra: Any,
    ) -> ChatResult:
        body: Dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = extra.pop("tool_choice", "auto")
        if response_format:
            body["response_format"] = response_format
        body.update(extra)

        started = time.monotonic()
        response = http.request(
            self.endpoint.chat_url, method="POST", json_body=body,
            headers=self.endpoint.headers, timeout=timeout or self.timeout,
        )
        latency = time.monotonic() - started

        if not response.ok:
            return ChatResult(
                ok=False, status=response.status, latency_s=latency,
                error=(response.error or "") + " " + response.body[:2000],
            )

        data = response.json()
        if not isinstance(data, dict) or not data.get("choices"):
            return ChatResult(ok=False, status=response.status, latency_s=latency,
                              error=f"unexpected response shape: {response.body[:500]}")

        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            ok=True,
            text=message.get("content") or "",
            status=response.status,
            raw=data,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason") or "",
            tool_calls=message.get("tool_calls") or [],
            reasoning_text=(message.get("reasoning_content")
                            or message.get("reasoning")
                            or message.get("thinking") or ""),
        )

    # ------------------------------------------------------------- streaming

    def stream_chat(
        self, messages: List[Dict[str, Any]], max_tokens: int = 128,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Stream a completion, measuring time to first token.

        TTFT is the number people actually feel, and it cannot be recovered from
        a buffered request, which is the whole reason this exists separately.
        """
        body = {
            "model": self.endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        started = time.monotonic()
        first_token_at: Optional[float] = None
        chunks = 0
        events = 0
        delta_keys: set = set()
        text = []
        error = ""
        completed = False

        try:
            for line in http.stream_lines(
                self.endpoint.chat_url, body, headers=self.endpoint.headers,
                timeout=timeout or self.timeout,
            ):
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    completed = True
                    break
                try:
                    parsed = json.loads(payload)
                except ValueError:
                    continue
                events += 1
                if parsed.get("error"):
                    error = str(parsed["error"])
                    break
                choice = (parsed.get("choices") or [{}])[0]
                completed = completed or bool(choice.get("finish_reason"))
                delta = choice.get("delta") or {}
                delta_keys.update(k for k, v in delta.items() if v)
                content = delta.get("content")
                # A reasoning model streams its thinking block first; those
                # chunks are still incremental output the client sees arriving.
                # llama.cpp uses reasoning_content, Ollama's OpenAI endpoint uses
                # reasoning, and Ollama's native API uses thinking.
                thinking = (delta.get("reasoning_content") or delta.get("reasoning")
                            or delta.get("thinking"))
                if content or thinking:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    chunks += 1
                    if content:
                        text.append(content)
        except Exception as exc:
            error = str(exc)

        if chunks and not completed and not error:
            error = "server closed the stream before completion"

        total = time.monotonic() - started
        return {
            "ok": bool(chunks) and not error,
            "error": error,
            "chunks": chunks,
            "events": events,
            "delta_keys": sorted(delta_keys),
            "text": "".join(text),
            "ttft_s": (first_token_at - started) if first_token_at else None,
            "total_s": total,
            "tokens_per_second": (
                chunks / (total - (first_token_at - started))
                if first_token_at and total > (first_token_at - started) else None
            ),
        }

    # ------------------------------------------------------------ embeddings

    def embed(self, texts: List[str], timeout: Optional[float] = None) -> Dict[str, Any]:
        response = http.request(
            self.endpoint.embeddings_url, method="POST",
            json_body={"model": self.endpoint.model, "input": texts},
            headers=self.endpoint.headers, timeout=timeout or self.timeout,
        )
        if not response.ok:
            return {"ok": False, "error": (response.error or "") + response.body[:500],
                    "status": response.status}
        data = response.json() or {}
        vectors = [item.get("embedding") for item in data.get("data", [])]
        return {"ok": bool(vectors and vectors[0]), "vectors": vectors,
                "dim": len(vectors[0]) if vectors and vectors[0] else 0}

    # ----------------------------------------------------------------- meta

    def infill(self, prefix: str, suffix: str, max_tokens: int = 64,
               timeout: Optional[float] = None) -> Dict[str, Any]:
        """Ask the server to fill the gap between a prefix and a suffix.

        A different endpoint from chat, because it is a different operation:
        the model is not answering a question about the code, it is continuing
        it from the middle with the rest already in view.
        """
        body = {
            "input_prefix": prefix,
            "input_suffix": suffix,
            "n_predict": max_tokens,
            "temperature": 0.0,
        }
        started = time.monotonic()
        response = http.request(
            self.endpoint.infill_url, method="POST", json_body=body,
            headers=self.endpoint.headers, timeout=timeout or self.timeout,
        )
        latency = time.monotonic() - started
        if not response.ok:
            return {"ok": False,
                    "error": (response.error or "") + response.body[:500],
                    "status": response.status, "latency_s": latency}
        data = response.json() or {}
        return {
            "ok": True,
            "text": data.get("content") or data.get("completion") or "",
            "latency_s": latency,
            "raw": data,
        }

    def list_models(self) -> Dict[str, Any]:
        response = http.request(self.endpoint.models_url,
                                headers=self.endpoint.headers, timeout=20)
        if not response.ok:
            return {"ok": False, "error": response.error or response.body[:300],
                    "status": response.status}
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list) or \
                any(not isinstance(item, dict) for item in data["data"]):
            return {"ok": False, "error": "unexpected model listing response shape",
                    "status": response.status}
        return {"ok": True, "ids": [m.get("id") for m in data["data"]]}
