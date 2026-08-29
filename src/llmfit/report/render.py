"""Report rendering.

The invariant: every claim printed here traces back to a :class:`CheckResult` or
a computed value. Nothing is asserted because it was intended — if a capability
was not tested, the report says so rather than implying it works.
"""

from __future__ import annotations

from typing import List

from llmfit.hardware.bandwidth import describe_bandwidth
from llmfit.hardware.detect import summarize as summarize_hardware
from llmfit.runtimes.registry import get_runtime
from llmfit.types import (
    GIB,
    CheckResult,
    CheckStatus,
    SetupResult,
    Severity,
)
from llmfit.util.log import Console
from llmfit.util.units import gb, human_tokens

_MARK = {
    CheckStatus.PASS: "PASS",
    CheckStatus.FAIL: "FAIL",
    CheckStatus.WARN: "WARN",
    CheckStatus.SKIP: "SKIP",
}


def render_terminal(result: SetupResult, console: Console) -> None:
    """The summary printed at the end of a run."""
    plan, config, spec = result.plan, result.config, _spec(result)

    console.print()
    console.print(console.bold("=" * 68))
    if result.succeeded:
        headline = "Setup complete and verified"
    elif not result.checks_ran:
        headline = "Plan ready — nothing was tested"
    else:
        headline = "Setup finished with unresolved problems"
    console.print(console.bold(f"  {headline}"))
    console.print(console.bold("=" * 68))

    console.print()
    console.print(console.bold("Model"))
    console.info(f"{spec.name} at {config.quant}  ({spec.license})")
    if plan.notes:
        console.info(console.dim(plan.notes[0]))

    console.print()
    console.print(console.bold("Machine"))
    console.info(summarize_hardware(result.hardware))
    console.info(f"memory bandwidth: {describe_bandwidth(result.hardware)}")

    console.print()
    console.print(console.bold("Configuration"))
    estimate = plan.best.estimate if plan.best else None
    if estimate:
        console.info(
            f"memory: {estimate.total_gb:.1f} GB "
            f"(weights {gb(estimate.weights)} + KV cache {gb(estimate.kv_cache)} "
            f"+ overhead {gb(estimate.overhead)})"
        )
    console.info(
        f"context: {human_tokens(config.context_tokens)} tokens"
        + (f" x {config.parallel} slots" if config.parallel > 1 else "")
    )
    console.info(
        f"GPU layers: {config.gpu_layers}/{config.total_layers}"
        + ("  (fully offloaded)" if config.full_gpu_offload else "  (partial — CPU is the bottleneck)")
    )
    console.info(f"KV cache: {config.kv_cache_type}   flash attention: "
                 f"{'on' if config.flash_attention else 'off'}")
    console.info(f"sampling: temp {config.temperature}, top_p {config.top_p}, "
                 f"top_k {config.top_k}, repeat penalty {config.repeat_penalty}")
    console.info(f"runtime: {plan.runtime} — {plan.runtime_reason}")

    _render_checks(result.checks, console)
    _render_embedding(result, console)
    _render_repairs(result, console)
    _render_registrations(result, console)
    _render_next_steps(result, console)


def _render_checks(checks: List[CheckResult], console: Console) -> None:
    if not checks:
        return
    console.print()
    console.print(console.bold("Verification"))
    for check in checks:
        line = f"[{_MARK[check.status]}] {check.name}: {check.message}"
        if check.status is CheckStatus.PASS:
            console.ok(line)
        elif check.status is CheckStatus.WARN:
            console.warn(line)
        elif check.status is CheckStatus.FAIL:
            console.error(line)
            if check.severity is not Severity.ADVISORY and check.detail:
                console.info(console.dim(f"      {check.detail.splitlines()[0][:160]}"))
        else:
            console.info(console.dim(line))

    tested = [c.name for c in checks if c.status is CheckStatus.PASS]
    untested = [c.name for c in checks if c.status is CheckStatus.SKIP]
    if untested:
        console.info(console.dim(
            f"not tested: {', '.join(untested)} — no claim is made about these"
        ))
    console.info(console.dim(f"verified: {len(tested)} of {len(checks)} checks"))


def _render_embedding(result: SetupResult, console: Console) -> None:
    record = result.embedding
    if not record:
        return
    console.print()
    console.print(console.bold("Retrieval (embeddings)"))
    if record.get("ok"):
        dims = record.get("dimensions")
        console.ok(
            f"{record.get('name', record.get('model'))} "
            + (f"— {dims}-dimensional vectors, " if dims else "— ")
            + "verified to separate related from unrelated text"
        )
        if record.get("base_url"):
            console.info(console.dim(f"    endpoint {record['base_url']}"))
        if record.get("note"):
            console.info(console.dim(f"    {record['note']}"))
    else:
        console.warn(
            f"no working embedding model: {record.get('reason', 'unknown')}. "
            f"Retrieval quality will suffer until this is fixed."
        )


def _render_repairs(result: SetupResult, console: Console) -> None:
    if not result.repairs:
        return
    console.print()
    console.print(console.bold("Automatic fixes applied"))
    console.info(console.dim(
        "these settings differ from the initial plan because a test failed:"
    ))
    for action in result.repairs:
        status = {True: "resolved", False: "did not resolve", None: ""}[action.resolved]
        console.info(f"- [{action.cost} cost] {action.description}"
                     + (f"  ({status})" if status else ""))
        console.info(console.dim(f"    triggered by: {action.trigger}"))


def _render_registrations(result: SetupResult, console: Console) -> None:
    if not result.registrations:
        return
    console.print()
    console.print(console.bold("Where you can use this model"))
    for record in result.registrations:
        prefix = "  ✓" if record.get("ok") else "  -"
        name = record.get("name") or record.get("target")
        console.print(f"{prefix} {name}: {record.get('note', '')}")
        if record.get("target") == "openai_clients":
            console.info(console.dim(
                f"    endpoint {record.get('base_url')}  model {record.get('model')}"
            ))


def _render_next_steps(result: SetupResult, console: Console) -> None:
    spec, config = _spec(result), result.config
    console.print()
    console.print(console.bold("Run it yourself"))
    try:
        command = get_runtime(result.plan.runtime).launch_command(spec, config)
        console.info(" ".join(command))
    except Exception:
        pass

    failures = [c for c in result.checks if c.blocking]
    if failures:
        console.print()
        console.print(console.bold("Still broken"))
        for check in failures:
            console.error(f"{check.name}: {check.message}")
        console.info(console.dim(
            "run `llmfit doctor` after changing anything to re-test"
        ))


def render_markdown(result: SetupResult) -> str:
    """A shareable report — what was chosen, why, and what was proven."""
    spec, config, plan = _spec(result), result.config, result.plan
    estimate = plan.best.estimate if plan.best else None
    lines: List[str] = []

    lines.append(f"# Local LLM setup: {spec.name}")
    lines.append("")
    lines.append(f"*Generated by llmfit on {result.finished_at or ''}*")
    lines.append("")
    if result.succeeded:
        verdict = "**Verified: yes — every required check passed**"
    elif not result.checks_ran:
        verdict = ("**Verified: no — this is a plan only. Nothing was downloaded, "
                   "launched or tested, so no claim is made that it works.**")
    else:
        verdict = "**Verified: no — see Unresolved below**"
    lines.append(verdict)
    lines.append("")

    lines.append("## What was chosen")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Model | {spec.name} ({spec.params_b:g}B"
                 + (f", {spec.active_params_b:g}B active" if spec.is_moe else "")
                 + f", {spec.license}) |")
    lines.append(f"| Quantization | {config.quant} |")
    lines.append(f"| Runtime | {plan.runtime} |")
    lines.append(f"| Context | {config.context_tokens:,} tokens"
                 + (f" x {config.parallel} slots" if config.parallel > 1 else "") + " |")
    if estimate:
        lines.append(f"| Memory | {estimate.total_gb:.1f} GB "
                     f"(weights {estimate.weights / GIB:.1f} + "
                     f"KV {estimate.kv_cache / GIB:.1f} + "
                     f"overhead {estimate.overhead / GIB:.1f}) |")
    lines.append(f"| GPU layers | {config.gpu_layers} / {config.total_layers} |")
    lines.append(f"| KV cache | {config.kv_cache_type} |")
    lines.append("")
    lines.append(f"Runtime chosen because: {plan.runtime_reason}.")
    lines.append("")

    if plan.notes:
        lines.append("### Why this model")
        lines.append("")
        for note in plan.notes:
            lines.append(f"- {note}")
        lines.append("")

    if plan.best and plan.best.warnings:
        lines.append("### Trade-offs you should know about")
        lines.append("")
        for warning in plan.best.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Machine")
    lines.append("")
    lines.append(f"- {summarize_hardware(result.hardware)}")
    lines.append(f"- Memory bandwidth: {describe_bandwidth(result.hardware)}")
    for warning in result.hardware.detection_warnings:
        lines.append(f"- Note: {warning}")
    lines.append("")

    lines.append("## Verification")
    lines.append("")
    if result.checks:
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for check in result.checks:
            lines.append(f"| {check.name} | {_MARK[check.status]} | "
                         f"{check.message.replace('|', '/')} |")
    else:
        lines.append("_No checks were run._")
    lines.append("")

    if result.repairs:
        lines.append("## Automatic fixes")
        lines.append("")
        lines.append("The settings above differ from the initial plan because these "
                     "tests failed and were repaired:")
        lines.append("")
        lines.append("| Fix | Cost | Triggered by | Resolved |")
        lines.append("|---|---|---|---|")
        for action in result.repairs:
            resolved = {True: "yes", False: "no", None: "-"}[action.resolved]
            lines.append(f"| {action.description.replace('|', '/')} | {action.cost} | "
                         f"{action.trigger} | {resolved} |")
        lines.append("")

    unresolved = [c for c in result.checks if c.blocking]
    if unresolved:
        lines.append("## Unresolved")
        lines.append("")
        for check in unresolved:
            lines.append(f"- **{check.name}**: {check.message}")
            if check.detail:
                lines.append(f"  ```\n  {check.detail.strip()[:500]}\n  ```")
        lines.append("")

    if result.embedding:
        record = result.embedding
        lines.append("## Retrieval (embeddings)")
        lines.append("")
        if record.get("ok"):
            dims = record.get("dimensions")
            lines.append(
                f"- **{record.get('name', record.get('model'))}**"
                + (f" — {dims}-dimensional vectors," if dims else " —")
                + " verified to separate related from unrelated text."
            )
            if record.get("base_url"):
                lines.append(f"- Endpoint: `{record['base_url']}`")
            if record.get("note"):
                lines.append(f"- {record['note']}")
        else:
            lines.append(
                f"- **Not working**: {record.get('reason', 'unknown')}. "
                f"A chat model makes a poor retriever, so RAG results will be weak "
                f"until a real embedding model is in place."
            )
        lines.append("")

    if result.registrations:
        lines.append("## Where this model is available")
        lines.append("")
        for record in result.registrations:
            mark = "yes" if record.get("ok") else "no"
            lines.append(f"- **{record.get('name') or record.get('target')}** "
                         f"({mark}): {record.get('note', '')}")
        lines.append("")

    try:
        command = get_runtime(plan.runtime).launch_command(spec, config)
        lines.append("## Running it without llmfit")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(command))
        lines.append("```")
        lines.append("")
    except Exception:
        pass

    if plan.fallbacks:
        lines.append("## If this stops working")
        lines.append("")
        lines.append("Next-best fits for this machine and use case:")
        lines.append("")
        for candidate in plan.fallbacks[:4]:
            lines.append(f"- {candidate.model.name} at {candidate.quant} "
                         f"(~{candidate.estimate.total_gb:.1f} GB)")
        lines.append("")

    return "\n".join(lines)


def _spec(result: SetupResult):
    if result.plan.best:
        return result.plan.best.model
    raise ValueError("cannot render a report without a chosen model")
