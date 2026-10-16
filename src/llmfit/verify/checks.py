"""The individual verification checks.

Each check takes a :class:`CheckContext` and returns a :class:`CheckResult`. A
check never raises. A check that blows up is itself a failure result, because
the harness has to keep going and the repair loop needs the signature.

Checks are selected by the user's :class:`UseCaseProfile`, so the suite proves
*their* requirements. Severity follows the same rule: a capability the user
asked for is REQUIRED and blocks; one they did not is ADVISORY and only informs.
"""

from __future__ import annotations

import base64
import json
import re
import statistics
import time
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from llmfit.hardware.bandwidth import estimate_tokens_per_second
from llmfit.types import (
    CheckResult,
    CheckStatus,
    HardwareProfile,
    ModelSpec,
    RuntimeConfig,
    Severity,
    Task,
    UseCaseProfile,
)
from llmfit.verify import signatures as sig
from llmfit.verify.client import OpenAIClient


@dataclass
class CheckContext:
    client: OpenAIClient
    spec: ModelSpec
    config: RuntimeConfig
    profile: UseCaseProfile
    hardware: HardwareProfile
    server_log: Optional[str] = None

    def log_tail(self, lines: int = 60) -> str:
        """Recent server output, errors that never reach the HTTP layer live here."""
        if not self.server_log:
            return ""
        try:
            with open(self.server_log, "r", encoding="utf-8", errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return ""


def _timed(fn: Callable[[], CheckResult]) -> CheckResult:
    started = time.monotonic()
    try:
        result = fn()
    except Exception as exc:
        result = CheckResult(
            name="unknown", status=CheckStatus.FAIL,
            message=f"the check itself raised: {exc}",
            failure_signature=sig.classify_or_unknown(str(exc)),
        )
    result.duration_s = round(time.monotonic() - started, 3)
    return result


def _fail(name: str, severity: Severity, message: str, signature: Optional[str] = None,
          detail: str = "", **metrics) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.FAIL, severity=severity,
                       message=message, failure_signature=signature, detail=detail,
                       metrics=metrics)


def _pass(name: str, severity: Severity, message: str, **metrics) -> CheckResult:
    return CheckResult(name=name, status=CheckStatus.PASS, severity=severity,
                       message=message, metrics=metrics)


# --------------------------------------------------------------------------- #
# 1. the server is up and serving the model we think it is
# --------------------------------------------------------------------------- #


def check_health(ctx: CheckContext) -> CheckResult:
    name = "health"
    listing = ctx.client.list_models()
    if not listing.get("ok"):
        error = str(listing.get("error", ""))
        return _fail(
            name, Severity.CRITICAL,
            "the inference server is not answering",
            sig.classify(error) or sig.SERVER_UNREACHABLE,
            detail=(error + "\n" + ctx.log_tail())[:2000],
        )

    ids = [i for i in listing.get("ids", []) if i]
    wanted = ctx.client.endpoint.model
    if not any(_names_match(wanted, str(i)) for i in ids):
        # Serving *something*, but not what we configured, a real failure mode
        # when a previous model is still resident.
        return _fail(
            name, Severity.CRITICAL,
            f"the server is up but does not list {wanted!r}",
            sig.MODEL_NOT_LOADED,
            detail=f"models present: {', '.join(str(i) for i in ids[:10])}",
        )
    return _pass(name, Severity.CRITICAL, f"serving {wanted}", models=ids[:10])


def _names_match(wanted: str, served: str) -> bool:
    """Compare a requested model name against one the server reports.

    Servers decorate names. LM Studio prefixes a publisher, Ollama appends a
    tag, llama.cpp may serve a filename. So an exact match is too strict. But a
    plain substring test is too loose: a short name matches almost anything,
    which turns "the wrong model is loaded" into a silent pass. The rule is
    equality or a match at a separator boundary.
    """
    def normalise(value: str) -> str:
        value = value.strip().lower()
        value = value.rsplit("/", 1)[-1]          # drop a publisher prefix
        value = re.sub(r"\.gguf$", "", value)
        return re.sub(r"[:@_.\s]+", "-", value).strip("-")

    a, b = normalise(wanted), normalise(served)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    # Require a real name, not an initial, and a boundary-aligned match.
    if len(shorter) < 3:
        return False
    return bool(re.search(rf"(^|-){re.escape(shorter)}(-|$)", longer))


# --------------------------------------------------------------------------- #
# 2. it can complete, and the completion is not nonsense
# --------------------------------------------------------------------------- #
# 2. it can complete, and the completion is not nonsense
# --------------------------------------------------------------------------- #

# A question with one short, unambiguous answer that any usable model gets
# right. This is a smoke test for a broken quantization or a mismatched chat
# template, not a benchmark.
_SANITY = [
    ("What is the capital of France? Answer with only the city name.",
     re.compile(r"\bparis\b", re.I)),
    ("What is 17 + 25? Reply with only the number.", re.compile(r"\b42\b")),
]


def check_completion(ctx: CheckContext) -> CheckResult:
    name = "completion"
    correct = 0
    transcript = []

    for prompt, expected in _SANITY:
        result = ctx.client.chat(
            [{"role": "user", "content": prompt}], max_tokens=512, temperature=0.0
        )
        if not result.ok:
            return _fail(
                name, Severity.CRITICAL, "the model could not produce a completion",
                sig.classify(result.error) or sig.UNKNOWN_ERROR,
                detail=(result.error + "\n" + ctx.log_tail())[:2000],
            )
        text = _strip_thinking(result.text)
        transcript.append(f"{prompt!r} -> {text[:120]!r}")
        if not text.strip():
            return _fail(name, Severity.CRITICAL,
                         "the model returned an empty response",
                         sig.EMPTY_RESPONSE, detail="\n".join(transcript))
        if expected.search(text):
            correct += 1

    if correct == 0:
        # Both wrong means something structural: a broken quant, or a chat
        # template that is not being applied.
        return _fail(
            name, Severity.CRITICAL,
            "the model answers basic factual questions incorrectly, which "
            "usually means a corrupt quantization or a mis-applied chat template",
            sig.INCOHERENT_OUTPUT, detail="\n".join(transcript),
        )
    if correct < len(_SANITY):
        return CheckResult(
            name=name, status=CheckStatus.WARN, severity=Severity.CRITICAL,
            message=f"{correct}/{len(_SANITY)} sanity answers correct",
            detail="\n".join(transcript),
        )
    return _pass(name, Severity.CRITICAL, "produces correct, coherent completions",
                 sanity_correct=correct)


# --------------------------------------------------------------------------- #
# 3. streaming works and time-to-first-token is measured
# --------------------------------------------------------------------------- #


def check_streaming(ctx: CheckContext) -> CheckResult:
    name = "streaming"
    result = ctx.client.stream_chat(
        [{"role": "user", "content": "Count from 1 to 20, separated by spaces."}],
        max_tokens=128,
    )
    if not result["ok"]:
        return _fail(
            name, Severity.ADVISORY,
            "streaming did not produce incremental chunks; clients will appear "
            "to hang until the whole reply is ready",
            sig.classify(result.get("error", "")) or sig.STREAMING_UNSUPPORTED,
            detail=str(result.get("error", ""))[:1000],
        )
    ttft = result.get("ttft_s")
    return _pass(
        name, Severity.ADVISORY,
        f"streams incrementally ({result['chunks']} chunks"
        + (f", first token in {ttft:.2f}s" if ttft else "") + ")",
        chunks=result["chunks"], ttft_s=round(ttft, 3) if ttft else None,
        stream_tokens_per_s=(round(result["tokens_per_second"], 1)
                             if result.get("tokens_per_second") else None),
    )


# --------------------------------------------------------------------------- #
# 4. the configured context length actually works
# --------------------------------------------------------------------------- #


def check_context(ctx: CheckContext) -> CheckResult:
    """Needle-in-a-haystack at the configured context length.

    This is the check that catches the most common silent misconfiguration: a
    server started with a smaller context than requested, which does not error
    it just truncates the prompt, and the model answers about text it never saw.
    """
    name = "context"
    target = ctx.config.context_tokens
    if target <= 4096:
        # Too small to be worth the tokens; the completion check covers it.
        return CheckResult(name=name, status=CheckStatus.SKIP, severity=Severity.ADVISORY,
                           message=f"context is only {target} tokens; covered by other checks")

    # Fill to ~75% of the window, leaving room for the reply and template overhead.
    fill_tokens = int(target * 0.75)
    needle_code = "ZX-4417-QQ"
    haystack, approx_tokens = _build_haystack(fill_tokens, needle_code)

    result = ctx.client.chat(
        [{"role": "user", "content": haystack}], max_tokens=256, temperature=0.0,
        timeout=600,
    )
    if not result.ok:
        signature = sig.classify(result.error) or sig.CONTEXT_EXCEEDED
        return _fail(
            name, Severity.REQUIRED,
            f"a ~{approx_tokens:,}-token prompt failed against a configured "
            f"{target:,}-token context",
            signature, detail=(result.error + "\n" + ctx.log_tail())[:2000],
            requested_context=target, prompt_tokens=approx_tokens,
        )

    text = _strip_thinking(result.text)
    actual_prompt_tokens = result.prompt_tokens or approx_tokens
    if needle_code.lower() not in text.lower():
        return _fail(
            name, Severity.REQUIRED,
            f"the model could not retrieve a fact placed mid-prompt at "
            f"~{actual_prompt_tokens:,} tokens, so the prompt is likely being "
            f"silently truncated, or KV-cache quantization has degraded recall",
            sig.NEEDLE_NOT_FOUND,
            detail=f"expected {needle_code!r} in the answer, got: {text[:300]!r}",
            requested_context=target, prompt_tokens=actual_prompt_tokens,
        )
    return _pass(
        name, Severity.REQUIRED,
        f"retrieved a fact from the middle of a ~{actual_prompt_tokens:,}-token prompt",
        requested_context=target, prompt_tokens=actual_prompt_tokens,
    )


def _build_haystack(target_tokens: int, needle_code: str) -> tuple:
    """Filler text with a distinctive fact buried in the middle.

    Placement matters: models are best at the start and end of a context and
    weakest in the middle, so a needle at the midpoint is the honest test.
    """
    filler = (
        "The quarterly logistics review noted routine variance in regional "
        "throughput. Warehouse utilisation remained within expected bounds and "
        "no exceptions were escalated to the operations committee. "
    )
    # ~4 characters per token is a good approximation across these tokenizers.
    needed_chars = target_tokens * 4
    repeats = max(1, needed_chars // len(filler))
    half = filler * (repeats // 2)
    needle = f"\n\nIMPORTANT: The verification code for this audit is {needle_code}.\n\n"
    body = half + needle + half
    question = (
        "\n\nBased only on the text above, what is the verification code for "
        "this audit? Reply with just the code."
    )
    full = body + question
    return full, len(full) // 4


# --------------------------------------------------------------------------- #
# 5. tool / function calling
# --------------------------------------------------------------------------- #

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def check_tools(ctx: CheckContext) -> CheckResult:
    """Verify the model emits a *structured* tool call, not prose describing one.

    This is where local setups most often disappoint: the model is capable, but
    the runtime was started without its chat template (llama.cpp needs
    ``--jinja``) or without a tool parser (vLLM), so calls come back as text and
    every agent framework breaks.
    """
    name = "tool_calling"
    severity = Severity.REQUIRED if ctx.profile.needs_tools else Severity.ADVISORY

    result = ctx.client.chat(
        [{"role": "user", "content": "What is the weather in Tokyo right now?"}],
        tools=[_WEATHER_TOOL], max_tokens=512, temperature=0.0,
    )
    if not result.ok:
        return _fail(
            name, severity, "the request with tools attached was rejected",
            sig.classify(result.error) or sig.TOOLS_UNSUPPORTED,
            detail=(result.error + "\n" + ctx.log_tail())[:2000],
        )

    if not result.tool_calls:
        text = _strip_thinking(result.text)
        mentions = "get_current_weather" in text or "tool_call" in text.lower()
        return _fail(
            name, severity,
            ("the model described the tool call in prose instead of returning a "
             "structured tool_calls field, the chat template is probably not "
             "being applied" if mentions else
             "the model did not call the tool at all"),
            sig.TOOLS_UNSUPPORTED if not mentions else sig.TOOL_CALL_MALFORMED,
            detail=f"response text: {text[:400]!r}",
        )

    call = result.tool_calls[0]
    function = call.get("function") or {}
    if function.get("name") != "get_current_weather":
        return _fail(name, severity,
                     f"called an unexpected function {function.get('name')!r}",
                     sig.TOOL_CALL_MALFORMED, detail=json.dumps(call)[:500])
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except ValueError:
        return _fail(name, severity,
                     "the tool call's arguments are not valid JSON",
                     sig.TOOL_CALL_MALFORMED,
                     detail=str(function.get("arguments"))[:500])
    if "city" not in arguments:
        return _fail(name, severity,
                     "the tool call omitted the required 'city' argument",
                     sig.TOOL_CALL_MALFORMED, detail=json.dumps(arguments)[:300])

    return _pass(name, severity,
                 f"emits well-formed tool calls (city={arguments.get('city')!r})",
                 tool_name=function.get("name"), arguments=arguments)


# --------------------------------------------------------------------------- #
# 6. structured JSON output
# --------------------------------------------------------------------------- #

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "languages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "languages"],
}


def check_json(ctx: CheckContext) -> CheckResult:
    name = "json_output"
    severity = Severity.REQUIRED if ctx.profile.needs_json else Severity.ADVISORY
    prompt = (
        "Extract the person as JSON with keys name (string), age (integer) and "
        "languages (array of strings). Text: 'Amara Okafor is 34 and speaks "
        "Igbo, English and French.' Reply with only the JSON object."
    )

    # Prefer constrained decoding, which makes valid JSON a guarantee rather
    # than a hope. Fall back to plain prompting if the server rejects it.
    formats: List[Optional[dict]] = [
        {"type": "json_schema",
         "json_schema": {"name": "person", "schema": _JSON_SCHEMA, "strict": True}},
        {"type": "json_object"},
        None,
    ]
    last_error = ""
    for response_format in formats:
        result = ctx.client.chat(
            [{"role": "user", "content": prompt}], max_tokens=512, temperature=0.0,
            response_format=response_format,
        )
        if not result.ok:
            last_error = result.error
            if sig.classify(result.error) in (sig.JSON_MODE_UNSUPPORTED, None):
                continue
            return _fail(name, severity, "the JSON request was rejected",
                         sig.classify(result.error) or sig.UNKNOWN_ERROR,
                         detail=result.error[:1000])

        parsed, raw = _extract_json(_strip_thinking(result.text))
        if parsed is None:
            last_error = f"unparseable output: {raw[:300]}"
            continue

        missing = [key for key in _JSON_SCHEMA["required"] if key not in parsed]
        if missing:
            last_error = f"missing keys {missing} in {raw[:200]}"
            continue
        if not isinstance(parsed.get("age"), int):
            last_error = f"'age' is {type(parsed.get('age')).__name__}, not an integer"
            continue

        mode = (response_format or {}).get("type", "prompt-only")
        return _pass(name, severity,
                     f"produces schema-valid JSON (via {mode})",
                     mode=mode, parsed=parsed)

    return _fail(name, severity,
                 "could not get schema-valid JSON out of the model",
                 sig.JSON_INVALID, detail=last_error[:1000])


def _extract_json(text: str):
    """Parse JSON from a reply, tolerating a code fence or surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text), text
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate), candidate
        except ValueError:
            return None, text
    return None, text


# --------------------------------------------------------------------------- #
# 7. embeddings
# --------------------------------------------------------------------------- #


def check_embeddings(ctx: CheckContext) -> CheckResult:
    """Embeddings must be *useful*, not merely present.

    A server can return vectors that are all near-identical (a broken pooling
    configuration), which passes a naive "did I get a vector" check and then
    silently ruins retrieval. So this compares a related and an unrelated pair.
    """
    name = "embeddings"
    severity = Severity.REQUIRED if ctx.profile.needs_embeddings else Severity.ADVISORY

    texts = [
        "How do I reset my password?",
        "What is the procedure for recovering a forgotten login?",
        "The migratory patterns of arctic terns span both poles.",
    ]
    result = ctx.client.embed(texts)
    if not result.get("ok"):
        return _fail(name, severity, "the endpoint did not return embeddings",
                     sig.classify(str(result.get("error", ""))) or sig.EMBEDDINGS_UNSUPPORTED,
                     detail=str(result.get("error", ""))[:1000])

    vectors = result["vectors"]
    if len(vectors) != 3:
        return _fail(name, severity,
                     f"asked for 3 embeddings, got {len(vectors)}",
                     sig.EMBEDDINGS_UNSUPPORTED)

    related = _cosine(vectors[0], vectors[1])
    unrelated = _cosine(vectors[0], vectors[2])
    if related <= unrelated:
        return _fail(
            name, severity,
            f"embeddings do not separate related from unrelated text "
            f"(related {related:.3f} <= unrelated {unrelated:.3f}); retrieval "
            f"built on these would return noise",
            sig.EMBEDDINGS_UNSUPPORTED,
            detail="this usually means the wrong pooling type or a non-embedding model",
        )
    return _pass(name, severity,
                 f"{result['dim']}-dimensional embeddings separate meaning correctly "
                 f"(related {related:.3f} vs unrelated {unrelated:.3f})",
                 dim=result["dim"], related=round(related, 4),
                 unrelated=round(unrelated, 4))


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# 8. vision
# --------------------------------------------------------------------------- #


def check_vision(ctx: CheckContext) -> CheckResult:
    name = "vision"
    severity = Severity.REQUIRED if ctx.profile.needs_vision else Severity.ADVISORY

    image = _solid_png(64, 64, (220, 30, 30))
    data_url = "data:image/png;base64," + base64.b64encode(image).decode()
    result = ctx.client.chat(
        [{"role": "user", "content": [
            {"type": "text",
             "text": "What single colour fills this image? Answer with one word."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        max_tokens=256, temperature=0.0,
    )
    if not result.ok:
        return _fail(name, severity, "the image request was rejected",
                     sig.classify(result.error) or sig.VISION_UNSUPPORTED,
                     detail=(result.error + "\n" + ctx.log_tail())[:1500])

    text = _strip_thinking(result.text).lower()
    if re.search(r"\bred\b|\bcrimson\b|\bscarlet\b", text):
        return _pass(name, severity, "correctly describes image content")
    return _fail(
        name, severity,
        "the endpoint accepted an image but described it incorrectly, which "
        "usually means the vision projector (mmproj) was not loaded",
        sig.VISION_UNSUPPORTED, detail=f"expected 'red', got: {text[:200]!r}",
    )


def _solid_png(width: int, height: int, rgb: tuple) -> bytes:
    """A minimal valid PNG, built without an image library."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        import struct

        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    import struct

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# 9. throughput
# --------------------------------------------------------------------------- #


def check_throughput(ctx: CheckContext) -> CheckResult:
    """Measure real decode throughput and compare it to what the plan predicted.

    A large shortfall against the prediction is diagnostic: it usually means
    layers silently fell back to CPU, so the number is checked against the model
    rather than against an arbitrary floor.
    """
    name = "throughput"
    result = ctx.client.chat(
        [{"role": "user",
          "content": "Write a single paragraph about the history of the printing press."}],
        max_tokens=200, temperature=0.0, timeout=300,
    )
    if not result.ok:
        return _fail(name, Severity.ADVISORY, "the benchmark request failed",
                     sig.classify(result.error) or sig.UNKNOWN_ERROR,
                     detail=result.error[:800])

    measured = result.tokens_per_second
    if measured is None:
        # No usage accounting: approximate from characters rather than give up.
        approx = len(_strip_thinking(result.text)) / 4
        measured = approx / result.latency_s if result.latency_s else 0.0

    bpw = 4.85
    active_bytes = int(ctx.spec.compute_params_b * 1e9 * bpw / 8)
    gpu_fraction = (ctx.config.gpu_layers / ctx.config.total_layers
                    if ctx.config.total_layers else 0.0)
    predicted = estimate_tokens_per_second(ctx.hardware, active_bytes, gpu_fraction)

    metrics = {
        "tokens_per_second": round(measured, 1),
        "predicted_tokens_per_second": round(predicted, 1),
        "latency_s": round(result.latency_s, 2),
    }

    # Below reading speed the setup is not interactively usable.
    if measured < 3.0:
        return _fail(
            name, Severity.ADVISORY,
            f"{measured:.1f} tokens/s is below usable interactive speed",
            sig.THROUGHPUT_TOO_LOW,
            detail=f"predicted {predicted:.0f} tok/s for this configuration",
            **metrics,
        )
    if predicted > 0 and measured < predicted * 0.35:
        return CheckResult(
            name=name, status=CheckStatus.WARN, severity=Severity.ADVISORY,
            message=(f"{measured:.1f} tokens/s is far below the predicted "
                     f"{predicted:.0f}, layers may have fallen back to CPU"),
            failure_signature=sig.THROUGHPUT_TOO_LOW, metrics=metrics,
        )
    return _pass(name, Severity.ADVISORY,
                 f"{measured:.1f} tokens/s (predicted {predicted:.0f})", **metrics)


#: What an inline completion has to come back within. A suggestion that lands
#: after the next keystroke is worse than no suggestion, because the reader has
#: already moved on and now has to read it anyway.
_INFILL_LATENCY_BUDGET_S = 1.0


def check_infill(ctx: CheckContext) -> CheckResult:
    """Fill a gap in a function and see whether the result is usable.

    Two things can go wrong and they look the same from the outside. The model
    may not be trained for this at all, in which case it ignores the suffix and
    writes something plausible that does not join up. Or the server may not have
    the endpoint, in which case nothing happens. Both leave an editor plugin
    silently useless, so both are worth separating out here.
    """
    name = "infill"
    severity = (Severity.REQUIRED if Task.AUTOCOMPLETE in ctx.profile.tasks
                else Severity.ADVISORY)

    prefix = "def add_numbers(a, b):\n    total = "
    suffix = "\n    return total\n"
    result = ctx.client.infill(prefix, suffix, max_tokens=32, timeout=60)

    if not result.get("ok"):
        return _fail(
            name, severity,
            "the infill endpoint is not available on this server",
            sig.classify(result.get("error", "")) or sig.INFILL_UNSUPPORTED,
            detail=str(result.get("error", ""))[:600],
        )

    text = (result.get("text") or "").strip()
    latency = float(result.get("latency_s") or 0.0)
    metrics: Dict[str, Any] = {"latency_s": round(latency, 3)}

    if not text:
        return _fail(name, severity,
                     "the endpoint answered but filled the gap with nothing",
                     sig.INFILL_UNSUPPORTED, **metrics)

    # The gap is the right-hand side of an assignment whose result is returned.
    # Anything that completes it mentions the two arguments; prose about the
    # function does not, and prose is what a model without infill training
    # produces here.
    joined = "a" in text and "b" in text
    if not joined:
        return _fail(
            name, severity,
            "the endpoint accepted the request but the model does not appear "
            "to be trained for infilling, since it ignored the code after the "
            "cursor",
            sig.INFILL_UNSUPPORTED, detail=f"filled the gap with: {text[:200]!r}",
            **metrics,
        )

    if latency > _INFILL_LATENCY_BUDGET_S:
        return CheckResult(
            name=name, status=CheckStatus.WARN, severity=Severity.ADVISORY,
            message=(f"infill works but takes {latency:.1f}s, which is longer "
                     f"than a typing pause, so suggestions will arrive after "
                     f"they are useful"),
            failure_signature=sig.THROUGHPUT_TOO_LOW, metrics=metrics,
        )
    return _pass(name, severity,
                 f"filled a gap in a function in {latency * 1000:.0f}ms",
                 **metrics)


def check_prefill(ctx: CheckContext) -> CheckResult:
    """Measure prompt processing, which is a different number from decode.

    Decode speed is what a chat feels like. Prefill is what everything else
    waits on: a RAG answer, an agent turn carrying its history, a file pasted
    into a coding assistant. They are bound by different things -- decode by
    memory bandwidth, prefill by compute -- so a setup can be fine at one and
    unusable at the other, and only measuring one hides that.

    Measured exactly rather than inferred: one long prompt with a single token
    of output, so the server's own prompt-token count divided by the latency
    *is* the prefill rate.
    """
    name = "prefill"
    # Long enough to be dominated by prefill rather than by request overhead,
    # and never longer than the context this setup was actually configured for.
    target = max(512, min(4096, ctx.config.context_tokens // 2))
    prompt, approx_tokens = _build_haystack(target, "IGNORED")

    result = ctx.client.chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1, temperature=0.0, timeout=300,
    )
    if not result.ok:
        return _fail(name, Severity.ADVISORY, "the prefill measurement failed",
                     sig.classify(result.error) or sig.UNKNOWN_ERROR,
                     detail=result.error[:800])

    tokens = result.prompt_tokens or approx_tokens
    if result.latency_s <= 0:
        return _pass(name, Severity.ADVISORY, "prefill was too fast to time")
    rate = tokens / result.latency_s

    # The number people actually experience: how long they stare at nothing
    # before the first word, with the context they asked for filled.
    full_context_wait = ctx.config.context_tokens / rate if rate else 0.0
    metrics: Dict[str, Any] = {
        "prefill_tokens_per_second": round(rate, 1),
        "prefill_prompt_tokens": tokens,
        "full_context_wait_s": round(full_context_wait, 1),
    }

    if rate < _PREFILL_FLOOR:
        return _fail(
            name, Severity.ADVISORY,
            f"prompt processing is {rate:.0f} tokens/s, so a full-context "
            f"prompt would take {full_context_wait / 60:.0f} minutes before "
            f"the first word",
            sig.PREFILL_TOO_SLOW, **metrics,
        )
    if _long_prompt_workload(ctx.profile) and full_context_wait > _PREFILL_PATIENCE:
        return CheckResult(
            name=name, status=CheckStatus.WARN, severity=Severity.ADVISORY,
            message=(f"{rate:.0f} tokens/s prompt processing, a full "
                     f"{ctx.config.context_tokens:,}-token prompt waits "
                     f"{full_context_wait:.0f}s before the first word, which "
                     f"this use case will hit routinely"),
            failure_signature=sig.PREFILL_TOO_SLOW, metrics=metrics,
        )
    return _pass(name, Severity.ADVISORY,
                 f"{rate:.0f} tokens/s prompt processing "
                 f"({full_context_wait:.0f}s at full context)", **metrics)


#: Below this, prompt processing is not a slow setup -- it is a broken one. A
#: CPU-only machine on a small model clears it comfortably; a setup that does
#: not has fallen off its accelerator entirely.
_PREFILL_FLOOR = 15.0

#: How long a first-word wait stays tolerable for a workload that fills its
#: context on every request.
_PREFILL_PATIENCE = 45.0


def _long_prompt_workload(profile) -> bool:
    """Whether this user sends long prompts as a matter of course."""
    return bool({Task.RAG, Task.AGENT, Task.SUMMARIZE, Task.CODING}
                & set(profile.tasks))


# --------------------------------------------------------------------------- #
# 10. stability under repeated load
# --------------------------------------------------------------------------- #


def check_stability(ctx: CheckContext) -> CheckResult:
    """Several requests back to back.

    Catches the failure that only appears on the second request: a KV cache that
    is not released, a slot leak, or memory that was only just sufficient for a
    single generation.
    """
    name = "stability"
    rounds = 4
    latencies = []
    for index in range(rounds):
        result = ctx.client.chat(
            [{"role": "user", "content": f"In one sentence, describe the number {index + 1}."}],
            max_tokens=96, temperature=0.0, timeout=180,
        )
        if not result.ok:
            return _fail(
                name, Severity.CRITICAL,
                f"request {index + 1} of {rounds} failed after earlier ones "
                f"succeeded, which points at a resource leak or marginal memory",
                sig.classify(result.error) or sig.UNKNOWN_ERROR,
                detail=(result.error + "\n" + ctx.log_tail())[:2000],
                failed_at_request=index + 1,
            )
        if not _strip_thinking(result.text).strip():
            return _fail(name, Severity.CRITICAL,
                         f"request {index + 1} returned empty output",
                         sig.EMPTY_RESPONSE, failed_at_request=index + 1)
        latencies.append(result.latency_s)

    drift = (max(latencies) / min(latencies)) if min(latencies) > 0 else 1.0
    metrics = {"requests": rounds, "median_latency_s": round(statistics.median(latencies), 2),
               "latency_drift": round(drift, 2)}
    if drift > 4.0:
        return CheckResult(
            name=name, status=CheckStatus.WARN, severity=Severity.CRITICAL,
            message=(f"latency varied {drift:.1f}x across {rounds} identical-size "
                     f"requests, which suggests memory pressure or thermal throttling"),
            metrics=metrics,
        )
    return _pass(name, Severity.CRITICAL,
                 f"{rounds} consecutive requests succeeded with stable latency", **metrics)


# --------------------------------------------------------------------------- #
# 11. degenerate output
# --------------------------------------------------------------------------- #


def check_repetition(ctx: CheckContext) -> CheckResult:
    """Detect the repetition loops that badly tuned sampling causes.

    Worth its own check because it is invisible to every other one: the model
    responds, quickly, with well-formed text that happens to be the same clause
    forty times.
    """
    name = "repetition"
    result = ctx.client.chat(
        [{"role": "user",
          "content": "Write three sentences about the sea. Do not repeat yourself."}],
        max_tokens=300, temperature=ctx.config.temperature,
        top_p=ctx.config.top_p,
    )
    if not result.ok:
        return _fail(name, Severity.ADVISORY, "the request failed",
                     sig.classify(result.error) or sig.UNKNOWN_ERROR)

    text = _strip_thinking(result.text)
    ratio = _repetition_ratio(text)
    if ratio > 0.5:
        return _fail(
            name, Severity.ADVISORY,
            f"{ratio:.0%} of the output is repeated phrasing, so the model is "
            f"looping, which usually means sampling is too greedy for this family",
            sig.REPETITION_LOOP, detail=text[:400], repetition_ratio=round(ratio, 3),
        )
    return _pass(name, Severity.ADVISORY, "output is varied, with no repetition loop",
                 repetition_ratio=round(ratio, 3))


def _repetition_ratio(text: str) -> float:
    """Fraction of 6-word windows that are duplicates. 0 is varied, 1 is a loop."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 24:
        return 0.0
    grams = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
    return 1.0 - (len(set(grams)) / len(grams))


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _strip_thinking(text: str) -> str:
    """Remove reasoning blocks before evaluating an answer.

    Reasoning models wrap their scratchpad in <think>…</think>. Grading the
    scratchpad instead of the answer produces both false passes (the needle is
    quoted while thinking, then dropped) and false failures.
    """
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    cleaned = re.sub(r"<\|channel\|>analysis.*?<\|(?:end|message)\|>", "", cleaned,
                     flags=re.S | re.I)
    # An unterminated block means the reply was cut off mid-thought; keep the
    # part before it rather than returning nothing.
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>")[0]
    return cleaned.strip()


ALL_CHECKS: Dict[str, Callable[[CheckContext], CheckResult]] = {
    "health": check_health,
    "completion": check_completion,
    "streaming": check_streaming,
    "context": check_context,
    "tool_calling": check_tools,
    "json_output": check_json,
    "embeddings": check_embeddings,
    "vision": check_vision,
    "throughput": check_throughput,
    "prefill": check_prefill,
    "infill": check_infill,
    "stability": check_stability,
    "repetition": check_repetition,
}
