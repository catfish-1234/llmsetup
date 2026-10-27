"""Check selection and execution.

Which checks run is derived from the user's profile, not from a fixed list. That
is the point: the tool verifies what the user said they needed, and does not
spend three minutes proving vision works for someone who only wants to chat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from llmsetup.types import (
    CheckResult,
    CheckStatus,
    HardwareProfile,
    ModelSpec,
    RuntimeConfig,
    Severity,
    Task,
    UseCaseProfile,
)
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console
from llmsetup.verify.checks import ALL_CHECKS, CheckContext
from llmsetup.verify.client import OpenAIClient

# Always run: these establish that anything at all works, so everything after
# them is meaningful.
BASELINE = ["health", "completion", "stability"]

# Cheap and broadly informative, so they run unless explicitly skipped.
STANDARD = ["streaming", "repetition", "throughput", "prefill"]


def select_checks(profile: UseCaseProfile, spec: ModelSpec,
                  quick: bool = False) -> List[str]:
    """The check list for this user, in execution order."""
    selected = list(BASELINE)
    if not quick:
        selected += STANDARD

    # A configured context above the trivial range is worth proving: silent
    # truncation is the most common and least visible misconfiguration there is.
    if profile.context_tokens > 8192:
        selected.append("context")

    if profile.needs_tools or Task.AGENT in profile.tasks:
        selected.append("tool_calling")
    elif spec.supports_tools and not quick:
        selected.append("tool_calling")     # advisory: report a capability they have

    if profile.needs_json or Task.EXTRACT in profile.tasks:
        selected.append("json_output")

    if profile.needs_vision or Task.VISION in profile.tasks:
        selected.append("vision")

    if Task.AUTOCOMPLETE in profile.tasks or spec.supports_fim:
        selected.append("infill")

    if profile.needs_embeddings or spec.is_embedding:
        selected.append("embeddings")

    # An embedding model cannot chat; running chat checks against it produces
    # confusing failures rather than useful ones.
    if spec.is_embedding:
        selected = ["health", "embeddings"]

    ordered, seen = [], set()
    for name in selected:
        if name in ALL_CHECKS and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


@dataclass
class Harness:
    """Runs a selected set of checks against a live endpoint."""

    client: OpenAIClient
    spec: ModelSpec
    config: RuntimeConfig
    profile: UseCaseProfile
    hardware: HardwareProfile
    console: Console
    server_log: Optional[str] = None

    def context(self) -> CheckContext:
        return CheckContext(
            client=self.client, spec=self.spec, config=self.config,
            profile=self.profile, hardware=self.hardware, server_log=self.server_log,
        )

    def run(self, names: Iterable[str], stop_on_critical: bool = True) -> List[CheckResult]:
        """Execute checks in order.

        Stops early on a failing CRITICAL check: once the server is unreachable
        or the model is incoherent, every later result is noise, and the repair
        loop wants the first real signature rather than a cascade.
        """
        names = list(names)
        unknown = [name for name in names if name not in ALL_CHECKS]
        if unknown:
            raise LlmsetupError(
                f"unknown verification check(s): {', '.join(unknown)}",
                hint="valid checks: " + ", ".join(sorted(ALL_CHECKS)),
            )
        ctx = self.context()
        results: List[CheckResult] = []

        for name in names:
            check = ALL_CHECKS.get(name)
            if check is None:
                continue
            self.console.debug(f"running check: {name}")
            result = _run_one(check, name, ctx)
            results.append(result)
            self._report(result)

            if stop_on_critical and result.status is CheckStatus.FAIL and \
                    result.severity is Severity.CRITICAL:
                remaining = [n for n in names if n not in {r.name for r in results}]
                for skipped in remaining:
                    results.append(CheckResult(
                        name=skipped, status=CheckStatus.SKIP, severity=Severity.ADVISORY,
                        message=f"skipped after {name} failed",
                    ))
                break
        return results

    def _report(self, result: CheckResult) -> None:
        line = f"{result.name}: {result.message}"
        if result.status is CheckStatus.PASS:
            self.console.ok(line)
        elif result.status is CheckStatus.WARN:
            self.console.warn(line)
        elif result.status is CheckStatus.FAIL:
            self.console.error(line)
        else:
            self.console.debug(line)


def _run_one(check, name: str, ctx: CheckContext) -> CheckResult:
    """Run one check, converting an unexpected exception into a failure result."""
    import time

    from llmsetup.verify import signatures as sig

    started = time.monotonic()
    try:
        result = check(ctx)
    except Exception as exc:
        result = CheckResult(
            # A broken check cannot establish that a required capability works.
            # Keep ordinary, intentionally advisory failures as returned by checks.
            name=name, status=CheckStatus.FAIL, severity=Severity.CRITICAL,
            message=f"the check raised an unexpected error: {exc}",
            failure_signature=sig.classify_or_unknown(str(exc)),
            detail=repr(exc),
        )
    result.name = name
    result.duration_s = round(time.monotonic() - started, 3)
    return result


def run_checks(harness: Harness, names: Optional[Iterable[str]] = None,
               quick: bool = False) -> List[CheckResult]:
    chosen = list(names) if names else select_checks(harness.profile, harness.spec, quick)
    return harness.run(chosen)


def summarize(results: List[CheckResult]) -> Dict[str, int]:
    counts = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
    for result in results:
        counts[result.status.value] += 1
    return counts


def blocking_failures(results: List[CheckResult]) -> List[CheckResult]:
    return [r for r in results if r.blocking]
