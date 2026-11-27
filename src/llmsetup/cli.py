"""Command line interface.

Subcommands mirror the pipeline stages, so any one of them can be run and
inspected on its own:

    llmsetup setup       the whole thing
    llmsetup detect      what hardware do I have?
    llmsetup plan        what would you pick, and why?
    llmsetup verify      does my current endpoint actually work?
    llmsetup doctor      re-test and repair an existing setup
    llmsetup start       start the last verified setup again (--detach for background)
    llmsetup status      is the background server running?
    llmsetup stop        stop the background server
    llmsetup restart     stop it and start the verified setup again
    llmsetup logs        show the background server's log (-f to follow)
    llmsetup export      register an existing model with my other apps
    llmsetup catalog     inspect or extend the model list
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmsetup.catalog.catalog import Catalog, load_catalog
from llmsetup.hardware import apply_overrides, detect_hardware
from llmsetup.hardware.detect import summarize as summarize_hardware
from llmsetup.interview.questions import infer_flags
from llmsetup.interview.wizard import (
    load_answers,
    profile_from_answers,
    review_model_choice,
    run_interview,
    save_profile,
)
from llmsetup.pipeline import Pipeline, companion_embedding_plan
from llmsetup.report.render import render_markdown, render_terminal
from llmsetup.runtimes.base import RuntimeUnavailable
from llmsetup.runtimes.registry import all_runtimes
from llmsetup.types import UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import app_dir, ensure_dir, write_text
from llmsetup.util.log import Console, write_console
from llmsetup.util.proc import format_command
from llmsetup.util.units import gb
from llmsetup.version import __version__ as VERSION


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from llmsetup.util.http import install_credential_safe_redirects

    install_credential_safe_redirects()
    console = Console(quiet=getattr(args, "quiet", False),
                      verbose=getattr(args, "verbose", False))

    handler = {
        "setup": cmd_setup, "detect": cmd_detect, "plan": cmd_plan,
        "verify": cmd_verify, "doctor": cmd_doctor, "export": cmd_export,
        "catalog": cmd_catalog, "runtimes": cmd_runtimes, "ui": cmd_ui,
        "report": cmd_report, "start": cmd_start, "status": cmd_status, "stop": cmd_stop,
        "restart": cmd_restart,
        "logs": cmd_logs, "chat": cmd_chat, "changes": cmd_changes, "undo": cmd_undo, "connect": cmd_connect, "disconnect": cmd_disconnect, "settings": cmd_settings,
    }.get(args.command)

    if handler is None:
        parser.print_help()
        return 2

    rerun = "Rerun: " + format_command(["llmsetup", *(argv if argv is not None else sys.argv[1:])])
    try:
        from llmsetup import changes

        # One journal run per command, so `llmsetup undo` reverses a whole setup or export at once.
        with changes.run(args.command):
            code = handler(args, console)
        # `status` reports state through its exit code (3: not running), like
        # `systemctl status`; that is an answer, not a failure to rerun.
        if code and args.command != "status":
            console.error(rerun)
        return code
    except LlmsetupError as exc:
        console.error(str(exc))
        console.error(rerun)
        return 1
    except (RuntimeUnavailable, OSError) as exc:
        console.error(str(exc))
        if isinstance(exc, RuntimeUnavailable) and exc.install_hint:
            console.error(exc.install_hint)
        console.error(rerun)
        return 1
    except KeyboardInterrupt:
        console.error("interrupted")
        return 130


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmsetup",
        description="Pick, tune, verify and register a local open-weight LLM.",
    )
    parser.add_argument("--version", action="version", version=f"llmsetup {VERSION}")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-q", "--quiet", action="store_true", help="only errors")
        p.add_argument("-v", "--verbose", action="store_true", help="show debug detail")
        p.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    def hardware_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vram", help="override detected VRAM, e.g. 24 or 24GB")
        p.add_argument("--ram", help="override detected system RAM")
        p.add_argument("--gpu-vendor", choices=["nvidia", "amd", "apple", "intel"],
                       help="override the detected GPU vendor")
        p.add_argument("--cpu-only", action="store_true", help="ignore any GPU")
        p.add_argument("--threads", type=_positive_int, help="override the physical core count")

    def profile_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--answers", type=Path,
                       help="YAML file of interview answers (skips the questions)")
        p.add_argument("--task", action="append", dest="tasks",
                       help="chat|coding|agent|rag|summarize|creative|extract|"
                            "translate|vision (repeatable)")
        p.add_argument("--client", action="append", dest="clients",
                       help="apps to register with (repeatable)")
        p.add_argument("--context", type=_positive_int, dest="context_tokens",
                       help="context length in tokens")
        p.add_argument("--priority", choices=["speed", "balanced", "quality"])
        p.add_argument("--users", type=_positive_int, dest="concurrent_users",
                       help="concurrent users to size for")
        p.add_argument("--tools", dest="needs_tools", action="store_true",
                       default=None, help="require tool calling")
        p.add_argument("--json-output", dest="needs_json", action="store_true",
                       default=None, help="require structured JSON output")
        p.add_argument("--vision", dest="needs_vision", action="store_true",
                       default=None, help="require image input")
        p.add_argument("--embeddings", dest="needs_embeddings", action="store_true",
                       default=None, help="also set up an embedding model")
        p.add_argument("--commercial", dest="commercial_use", action="store_true",
                       default=None, help="only permissively licensed models")
        p.add_argument("--languages", help="comma-separated language codes")
        p.add_argument("--offline", dest="offline_only", action="store_true", default=None,
                       help="skip automatic catalog network updates")
        p.add_argument("--yes", action="store_true",
                       help="accept all defaults instead of asking")

    p_setup = sub.add_parser("setup", help="the full flow, end to end")
    common(p_setup)
    hardware_flags(p_setup)
    profile_flags(p_setup)
    p_setup.add_argument("--runtime", help="force a runtime (ollama, llamacpp, vllm, …)")
    p_setup.add_argument("--model", help="plan around a catalog model ID or name, or any Hugging Face GGUF repo as hf:owner/repo")
    p_setup.add_argument("--variant", help="use a derivative of --model instead, e.g. abliterated, found on Hugging Face")
    p_setup.add_argument("--quant", help="force a quantization, e.g. Q4_K_M")
    p_setup.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                         help="override a tuned setting, e.g. --set flash_attention=true (repeatable; "
                              "`llmsetup settings` lists them)")
    p_setup.add_argument("--engine-arg", dest="engine_args", action="append", metavar="FLAG",
                         help="pass a flag straight to llama.cpp, vLLM or MLX, e.g. --engine-arg=--no-warmup "
                              "(repeatable)")
    p_setup.add_argument("--quick", action="store_true",
                         help="run only the essential checks")
    p_setup.add_argument("--dry-run", action="store_true",
                         help="plan and tune, but download and launch nothing")
    p_setup.add_argument("--no-register", action="store_true",
                         help="skip registering with other apps")
    p_setup.add_argument("--sweep", action="store_true",
                         help="measure batch settings on this machine and keep "
                              "the fastest; adds a few minutes and needs "
                              "llama-bench")
    p_setup.add_argument("--no-repair", action="store_true",
                         help="report failures instead of fixing them")
    p_setup.add_argument("--cached-only", action="store_true",
                         help="require retained model weights; never download weights "
                              "(Ollama/llama.cpp shared GGUF store)")
    p_setup.add_argument("--report", type=Path, help="write a markdown report here")
    p_setup.add_argument("--save-profile", type=Path,
                         help="save the answers for a repeatable run")
    p_setup.add_argument("--catalog", type=Path, action="append",
                         help="additional catalog file (repeatable)")

    p_detect = sub.add_parser("detect", help="show what hardware was found")
    common(p_detect)
    hardware_flags(p_detect)

    p_plan = sub.add_parser("plan", help="show the ranked recommendations")
    common(p_plan)
    hardware_flags(p_plan)
    profile_flags(p_plan)
    p_plan.add_argument("--runtime")
    p_plan.add_argument("--model", help="plan around a catalog model ID or name, or any Hugging Face GGUF repo as hf:owner/repo")
    p_plan.add_argument("--variant", help="use a derivative of --model instead, e.g. abliterated, found on Hugging Face")
    p_plan.add_argument("--top", type=_positive_int, default=5, help="how many to show")
    p_plan.add_argument("--catalog", type=Path, action="append")

    p_verify = sub.add_parser("verify", help="test an endpoint you already have")
    common(p_verify)
    profile_flags(p_verify)
    p_verify.add_argument("--base-url", default="http://127.0.0.1:11434",
                          help="OpenAI-compatible endpoint")
    p_verify.add_argument("--model", required=True, help="model name to request")
    p_verify.add_argument("--checks", help="comma-separated subset of checks")

    p_doctor = sub.add_parser("doctor", help="re-test the last setup and fix problems")
    common(p_doctor)
    hardware_flags(p_doctor)
    p_doctor.add_argument("--profile", type=Path, help="saved setup JSON to re-test")
    p_doctor.add_argument("--cached-only", action="store_true",
                          help="require retained model weights, including every repair fallback")

    p_start = sub.add_parser(
        "start", help="start the last verified setup again, exactly as it was tested")
    common(p_start)
    p_start.add_argument("--setup", type=Path, help="saved setup JSON to start")
    p_start.add_argument("--print", dest="print_only", action="store_true",
                         help="print the launch command instead of running it")
    p_start.add_argument("--detach", action="store_true",
                         help="run in the background with output in a log file; stop it with `llmsetup stop`")
    p_start.add_argument("--wait", type=float, default=180.0, metavar="SECONDS",
                         help="with --detach, how long to wait for the endpoint to answer (default 180)")

    p_status = sub.add_parser("status", help="is the background server running and answering?")
    common(p_status)

    p_stop = sub.add_parser("stop", help="stop the server started by `llmsetup start --detach`")
    common(p_stop)

    p_restart = sub.add_parser(
        "restart", help="stop the background server and start the verified setup again in the background")
    common(p_restart)
    p_restart.add_argument("--setup", type=Path, help="saved setup JSON to start")
    p_restart.add_argument("--wait", type=float, default=180.0, metavar="SECONDS",
                           help="how long to wait for the endpoint to answer (default 180)")

    p_logs = sub.add_parser("logs", help="show the log of the server started by `llmsetup start --detach`")
    common(p_logs)
    p_logs.add_argument("-n", "--lines", type=_positive_int, default=50,
                        help="how many lines from the end to show (default 50)")
    p_logs.add_argument("-f", "--follow", action="store_true",
                        help="keep printing new lines until the server stops or you press Ctrl-C")

    p_connect = sub.add_parser(
        "connect", help="write the verified model into a tool's own config (opencode, continue, aider, zed, ...)")
    common(p_connect)
    p_connect.add_argument("harness", nargs="?", help="the tool to connect; omit to list them")
    p_connect.add_argument("--list", action="store_true", help="list the tools connect knows")
    p_connect.add_argument("--config", type=Path, help="edit this config file instead of the tool's usual one")
    p_connect.add_argument("--project", action="store_true",
                           help="use the project's config in this folder instead of your user config")
    p_connect.add_argument("--default", action="store_true",
                           help="also make it the tool's default model, where the tool has one")
    p_connect.add_argument("--rewrite", action="store_true",
                           help="rewrite a config that has comments (they are lost; undo restores the original)")
    p_connect.add_argument("--dry-run", action="store_true", help="print the resulting config, write nothing")
    p_connect.add_argument("--setup", type=Path, help="saved setup JSON whose model to connect")
    p_connect.add_argument("--base-url", help="connect this OpenAI-compatible server instead")
    p_connect.add_argument("--model", help="model name the server serves")
    p_connect.add_argument("--api-key", help="bearer token for a server that needs one")

    p_disconnect = sub.add_parser(
        "disconnect", help="take llmsetup's entries back out of a tool's config, keeping your own edits")
    common(p_disconnect)
    p_disconnect.add_argument("harness", help="the tool to disconnect (as for connect)")
    p_disconnect.add_argument("--config", type=Path, help="edit this config file instead of the tool's usual one")
    p_disconnect.add_argument("--project", action="store_true",
                              help="use the project's config in this folder instead of your user config")
    p_disconnect.add_argument("--rewrite", action="store_true",
                              help="rewrite a config that has comments (they are lost; undo restores the original)")
    p_disconnect.add_argument("--dry-run", action="store_true", help="print the resulting config, write nothing")

    p_settings = sub.add_parser("settings", help="list the settings `setup --set KEY=VALUE` can override")
    common(p_settings)

    p_changes = sub.add_parser(
        "changes", help="list what llmsetup changed in other apps and config files, by run")
    common(p_changes)

    p_undo = sub.add_parser(
        "undo", help="reverse what the last run changed outside llmsetup (or --all, or --run ID)")
    common(p_undo)
    p_undo.add_argument("--run", help="undo this run (an id from `llmsetup changes`) instead of the last one")
    p_undo.add_argument("--all", dest="undo_all", action="store_true", help="undo every recorded change")
    p_undo.add_argument("--dry-run", action="store_true", help="show what would be undone, change nothing")
    p_undo.add_argument("--force", action="store_true",
                        help="also undo changes to files edited since llmsetup wrote them")

    p_chat = sub.add_parser(
        "chat", help="talk to the verified model in the terminal (or ask one question with -m)")
    common(p_chat)
    p_chat.add_argument("-m", "--message",
                        help="ask one question, print the reply and exit; - reads it from stdin")
    p_chat.add_argument("--system", help="a system prompt for the conversation")
    p_chat.add_argument("--setup", type=Path, help="saved setup JSON whose model to talk to")
    p_chat.add_argument("--base-url", help="talk to this OpenAI-compatible server instead")
    p_chat.add_argument("--model", help="model name to request (default: the one the setup serves)")
    p_chat.add_argument("--api-key", help="bearer token for a server that needs one")
    p_chat.add_argument("--max-tokens", type=_positive_int, default=1024,
                        help="longest reply, in tokens (default 1024)")
    p_chat.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (default 0.7)")
    p_chat.add_argument("--thinking", action="store_true",
                        help="also show the thinking a reasoning model streams separately")

    p_report = sub.add_parser(
        "report", help="reprint the last setup's report (paste this into a bug report)")
    common(p_report)
    p_report.add_argument("path", nargs="?", type=Path,
                          help="a saved report to print instead of the last one")

    p_export = sub.add_parser("export", help="register a model with your other apps")
    common(p_export)
    p_export.add_argument("--model", required=True, help="catalog model id")
    p_export.add_argument("--quant", default="Q4_K_M")
    p_export.add_argument("--gguf", type=Path, help="path to an existing GGUF file")
    p_export.add_argument("--base-url", default="http://127.0.0.1:11434")

    p_catalog = sub.add_parser("catalog", help="inspect or extend the model catalog")
    common(p_catalog)
    p_catalog.add_argument(
        "action",
        choices=["list", "show", "path", "add", "refresh", "update", "auto-update"],
        default="list", nargs="?",
        help="list | show <id> | path | add <id> | refresh | update | auto-update",
    )
    p_catalog.add_argument("model_id", nargs="?")
    p_catalog.add_argument("--gguf", type=Path,
                           help="add: read the architecture from this local GGUF")
    p_catalog.add_argument("--params", type=float,
                           help="add: parameter count in billions")
    p_catalog.add_argument("--gguf-repo", help="add: Hugging Face repo carrying GGUFs")
    p_catalog.add_argument("--hf-repo", help="add: safetensors repo")
    p_catalog.add_argument("--ollama-name", help="add: pullable Ollama tag")
    p_catalog.add_argument("--publisher", help="add: publisher name")
    p_catalog.add_argument("--model-name", help="add: human-readable name")
    p_catalog.add_argument("--license", dest="license_name", default="unknown",
                           help="add: licence identifier")
    p_catalog.add_argument("--task", action="append", dest="model_tasks",
                           help="add: a task this model suits (repeatable)")
    p_catalog.add_argument("--max-context", type=int, help="add: context window")
    p_catalog.add_argument("--supports-tools", action="store_true",
                           help="add: the model does tool calling")
    p_catalog.add_argument("--supports-vision", action="store_true",
                           help="add: the model accepts images")
    p_catalog.add_argument("--quant", action="append", dest="model_quants",
                           help="add: a recommended quantization (repeatable)")
    p_catalog.add_argument("--offline", action="store_true",
                           help="refresh: only check local files, skip the network")
    p_catalog.add_argument("--url", help="update: fetch the catalog from this URL")
    p_catalog.add_argument("--enable", action="store_true",
                           help="auto-update: turn automatic updates back on")
    p_catalog.add_argument("--disable", action="store_true",
                           help="auto-update: turn automatic updates off for good")
    p_catalog.add_argument("--dry-run", action="store_true",
                           help="refresh: report drift without writing an overlay")

    p_runtimes = sub.add_parser("runtimes", help="show which engines are installed")
    common(p_runtimes)

    p_ui = sub.add_parser("ui", help="open the local web interface")
    common(p_ui)
    p_ui.add_argument("--port", type=int, default=8765)
    p_ui.add_argument("--no-browser", action="store_true",
                      help="print the URL instead of opening a browser")

    return parser


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_detect(args, console: Console) -> int:
    hardware = _hardware(args)
    if args.json:
        print(json.dumps(hardware.to_dict(), indent=2))
        return 0

    from llmsetup.hardware.bandwidth import describe_bandwidth

    console.print(console.bold("Hardware"))
    console.info(summarize_hardware(hardware))
    console.info(f"memory bandwidth: {describe_bandwidth(hardware)}")
    console.info(f"usable for a model: {gb(hardware.memory_budget)}"
                 + (" (unified memory, capped for OS headroom)"
                    if hardware.unified_memory else ""))
    console.info(f"free disk in the model store: {gb(hardware.disk_free_bytes)}")
    for gpu in hardware.gpus:
        free = f", {gb(gpu.vram_free_bytes)} free" if gpu.vram_free_bytes else ""
        console.info(f"GPU {gpu.index}: {gpu.name}, {gb(gpu.vram_bytes)}{free}"
                     f"  flash-attention: "
                     f"{'yes' if gpu.supports_flash_attention else 'no'}")
    for warning in hardware.detection_warnings:
        console.warn(warning)
    return 0


def _materialize_model(args, console: Console) -> None:
    """Turn `--model hf:owner/repo` or `--variant NAME` into an ordinary overlay catalog entry.

    Everything after this point (planning, fit, provisioning, verification) then treats it like
    any catalog model, which is what keeps an explicitly requested model as well checked as a
    recommended one.
    """
    from llmsetup.catalog import hub

    model = getattr(args, "model", None)
    variant = getattr(args, "variant", None)
    repo = hub.parse_hub_ref(model) if model else None
    if not repo and not variant:
        return
    if getattr(args, "offline", False):
        raise LlmsetupError("--variant and --model hf:… look the model up on Hugging Face, which --offline rules out",
                            hint="drop --offline, or add the model once with `llmsetup catalog add`")
    if repo:
        if variant:
            raise LlmsetupError("--variant finds a derivative of a catalog model, but hf: already names the exact repo",
                                hint="use one or the other")
        console.step(f"Reading {repo} from Hugging Face")
        entry = hub.repo_entry(repo)
    else:
        if not model:
            raise LlmsetupError("--variant needs --model to name the model it is a variant of",
                                hint=f"for example: --model qwen3-8b --variant {variant}")
        from llmsetup.planner.model_fit import resolve_requested_model

        variant = str(variant)
        catalog = load_catalog(extra_paths=getattr(args, "catalog", None))
        base = resolve_requested_model(model, catalog)
        console.step(f"Looking for {variant} GGUF variants of {base.name} on Hugging Face")
        found = hub.find_variants(base, variant)
        if not found:
            raise LlmsetupError(f"found no {variant} GGUF variant of {base.name} on Hugging Face",
                                hint="name an exact repo with --model hf:owner/repo, or try another spelling")
        choice = found[0]
        console.ok(f"using {choice['repo']} ({choice['downloads']:,} downloads, "
                   f"quants {', '.join(choice['quants'][:6])})")
        if len(found) > 1:
            console.info("others: " + ", ".join(c["repo"] for c in found[1:])
                         + "; pick one with --model hf:owner/repo")
        entry = hub.variant_entry(base, variant, choice)
        entry["arch_source"] = catalog.arch_source(base.id)
    path = _save_hub_entry(entry)
    console.info(f"saved as {entry['id']} in {path}")
    args.model = entry["id"]


def _save_hub_entry(entry) -> Path:
    """Upsert an entry into the overlay catalog's hub.yaml."""
    import yaml

    from llmsetup.catalog.refresh import user_catalog_path
    from llmsetup.util.fs import write_text

    path = user_catalog_path("hub.yaml")
    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            existing = {}
    models = [m for m in existing.get("models") or [] if isinstance(m, dict) and m.get("id") != entry["id"]]
    existing.update({"schema_version": 1, "models": [*models, entry]})
    header = ("# Models added with --model hf:owner/repo or --variant. llmsetup's own file;\n"
              "# delete an entry to forget it.\n\n")
    write_text(path, header + yaml.safe_dump(existing, sort_keys=False, allow_unicode=True))
    return path


def cmd_plan(args, console: Console) -> int:
    from llmsetup.planner.plan import build_adaptive_plan
    from llmsetup.planner.score import explain

    _materialize_model(args, console)
    hardware = _hardware(args)
    _show_detection(hardware, console)
    profile = _profile(args, console)
    _auto_update(console, profile)
    catalog = load_catalog(extra_paths=getattr(args, "catalog", None))
    profile = _review_model(args, profile, hardware, catalog, console)
    plan = build_adaptive_plan(profile, hardware, catalog, runtime=args.runtime)
    profile = plan.profile or profile

    if args.json:
        print(json.dumps({
            "runtime": plan.runtime,
            "runtime_reason": plan.runtime_reason,
            "requested_model": profile.requested_model,
            "context_tokens": profile.context_tokens,
            "candidates": [{
                "model": c.model.id, "name": c.model.name, "quant": c.quant,
                "score": c.score, "breakdown": c.score_breakdown,
                "memory_gb": round(c.estimate.total_gb, 2),
                "kv_quant": c.estimate.kv_quant,
                "fits_on_gpu": c.fits_on_gpu, "warnings": c.warnings,
            } for c in plan.candidates[:args.top]],
            "notes": plan.notes,
        }, indent=2))
        return 0

    console.print(console.bold(f"Machine: {summarize_hardware(hardware)}"))
    console.print(console.bold(f"Runtime: {plan.runtime} - {plan.runtime_reason}"))
    console.print()
    console.print(console.bold(f"Top {min(args.top, len(plan.candidates))} for "
                               f"{', '.join(t.value for t in profile.tasks)} "
                               f"at {profile.context_tokens:,} tokens"))
    console.print()
    for index, candidate in enumerate(plan.candidates[:args.top], 1):
        marker = "*" if index == 1 else " "
        tps = candidate.score_breakdown.get("est_tokens_per_s")
        console.print(
            f" {marker} {index}. {candidate.model.name} ({candidate.quant})  "
            f"{candidate.estimate.total_gb:.1f} GB  "
            + (f"~{tps:.0f} tok/s  " if tps else "")
            + f"score {candidate.score:.3f}"
        )
        console.info(console.dim(f"   {explain(candidate, profile, hardware)}"), wrap=True)
        console.info(console.dim(
            f"   weights {gb(candidate.estimate.weights)} + "
            f"KV {gb(candidate.estimate.kv_cache)} ({candidate.estimate.kv_quant}) + "
            f"overhead {gb(candidate.estimate.overhead)}"
        ))
        for warning in candidate.warnings:
            console.warn(f"   {warning}", wrap=True)
    console.print()
    for note in plan.notes:
        console.info(console.dim(note), wrap=True)

    embedder = companion_embedding_plan(profile, hardware, catalog)
    if embedder:
        console.print()
        console.print(console.bold("Companion embedding model"))
        console.info(f"{embedder.name} ({embedder.license}): "
                     f"{embedder.max_context:,} token window")
    return 0


def cmd_setup(args, console: Console) -> int:
    from llmsetup.tuning.overrides import parse_overrides

    # Checked before anything slow, so a mistyped setting fails in a second, not after detection.
    overrides = parse_overrides(getattr(args, "overrides", None))
    _materialize_model(args, console)
    hardware = _hardware(args)
    _show_detection(hardware, console)
    profile = _profile(args, console)
    _auto_update(console, profile)
    catalog = load_catalog(extra_paths=getattr(args, "catalog", None))

    if profile.requested_model and args.quant:
        from llmsetup.planner.model_fit import resolve_requested_model

        selected = resolve_requested_model(profile.requested_model, catalog)
        # Keep the full catalog for fit comparisons and exhausted-budget recovery.
        catalog = Catalog(
            [replace(spec, recommended_quants=[args.quant]) if spec.id == selected.id else spec
             for spec in catalog.models],
            catalog.quant_bpw, catalog.updated, catalog.sources, catalog.extras,
        )
    profile = _review_model(args, profile, hardware, catalog, console)

    if args.save_profile:
        save_profile(profile, args.save_profile)
        console.ok(f"answers saved to {args.save_profile}")

    pipeline = Pipeline(
        profile, hardware, catalog, console,
        runtime_name=args.runtime, quick=args.quick, dry_run=args.dry_run,
        max_repairs=0 if args.no_repair else 12,
        register=not args.no_register,
        sweep=getattr(args, "sweep", False),
        cached_only=getattr(args, "cached_only", False),
        overrides=overrides, engine_args=getattr(args, "engine_args", None),
    )
    try:
        result = pipeline.run()
    finally:
        # llama.cpp/vLLM servers we launched are ours to clean up. Ollama and
        # LM Studio are shared daemons and are deliberately left running.
        pipeline.shutdown()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        render_terminal(result, console)

    _save_last(result)
    if args.report:
        write_text(args.report, render_markdown(result))
        console.ok(f"report written to {args.report}")

    if args.dry_run:
        # Nothing was attempted, so there is nothing to have failed.
        return 0
    return 0 if result.succeeded else 1


def _check_summary_line(results: List) -> str:
    """The terminal tally line. Only actual passes count as passed: a check
    skipped after an early critical failure is not a pass, and neither is a
    warning."""
    from llmsetup.types import CheckStatus

    passed = sum(1 for r in results if r.status is CheckStatus.PASS)
    skipped = sum(1 for r in results if r.status is CheckStatus.SKIP)
    warned = sum(1 for r in results if r.status is CheckStatus.WARN)
    failed_count = len(results) - passed - skipped - warned
    parts = []
    if failed_count:
        parts.append(f"{failed_count} failed")
    if warned:
        parts.append(f"{warned} warning(s)")
    if skipped:
        parts.append(f"{skipped} skipped")
    detail = f" ({', '.join(parts)})" if parts else ""
    return f"{passed}/{len(results)} checks passed{detail}"


def cmd_verify(args, console: Console) -> int:
    from llmsetup.runtimes.base import Endpoint
    from llmsetup.verify.client import OpenAIClient
    from llmsetup.verify.harness import Harness

    hardware = detect_hardware()
    profile = _profile(args, console)
    catalog = load_catalog()

    spec = catalog.get(args.model) or _synthetic_spec(args.model, catalog)
    endpoint = Endpoint(base_url=args.base_url, model=args.model, runtime="external")
    config = _config_for_external(spec, profile, args)

    harness = Harness(
        client=OpenAIClient(endpoint), spec=spec, config=config, profile=profile,
        hardware=hardware, console=console,
    )
    names = [name.strip() for name in args.checks.split(",")] if args.checks else None
    console.step(f"Verifying {args.model} at {args.base_url}")
    results = harness.run(names or _default_checks(profile, spec))

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    failed = [r for r in results if r.blocking]
    console.print()
    console.print(console.bold(_check_summary_line(results)))
    return 1 if failed else 0


def cmd_doctor(args, console: Console) -> int:
    """Re-test the last setup and repair anything that broke."""
    path = args.profile or (app_dir() / "last-setup.json")
    if not Path(path).exists():
        raise LlmsetupError(
            f"no saved setup at {path}",
            hint="run `llmsetup setup` first, or pass --profile",
        )
    from llmsetup.util.fs import read_json

    saved = read_json(Path(path))
    if not saved:
        raise LlmsetupError(f"{path} is not readable")

    profile = profile_from_answers(saved.get("profile", {}))
    catalog = load_catalog()
    hardware = _hardware(args)
    model_id = (saved.get("config") or {}).get("model_id")
    if not model_id:
        raise LlmsetupError(f"{path} does not record which model was set up")

    console.step(f"Re-testing {model_id}")
    profile = replace(profile, requested_model=model_id)

    pipeline = Pipeline(profile, hardware, catalog, console,
                        runtime_name=saved.get("runtime"),
                        cached_only=getattr(args, "cached_only", False))
    try:
        result = pipeline.run()
    finally:
        pipeline.shutdown()

    render_terminal(result, console)
    _save_last(result)
    return 0 if result.succeeded else 1


def cmd_export(args, console: Console) -> int:
    from llmsetup.interop.export import register_everywhere
    from llmsetup.runtimes.base import Endpoint
    from llmsetup.runtimes.gguf_store import find_local

    catalog = load_catalog()
    spec = catalog.require(args.model)
    gguf = args.gguf or find_local(spec, args.quant)
    if gguf and not Path(gguf).exists():
        raise LlmsetupError(f"{gguf} does not exist")
    if not gguf:
        console.warn(
            f"no local GGUF found for {spec.id}:{args.quant}; only HTTP client "
            f"configs will be emitted"
        )

    config = _config_for_external(spec, UseCaseProfile(), args)
    config.quant = args.quant
    endpoint = Endpoint(base_url=args.base_url, model=spec.id, runtime="external")

    console.step("Registering")
    records = register_everywhere(spec, config, endpoint, console,
                                  gguf_path=Path(gguf) if gguf else None)
    if args.json:
        print(json.dumps(records, indent=2))
    return 0


def cmd_catalog(args, console: Console) -> int:
    from llmsetup.catalog.refresh import add_model, refresh, user_catalog_dir

    catalog = load_catalog()

    if args.action == "add":
        if not args.model_id:
            raise LlmsetupError(
                "`catalog add` needs a model id",
                hint="llmsetup catalog add my-model --gguf ./model.gguf",
            )
        path = add_model(
            console, args.model_id, gguf_path=args.gguf, gguf_repo=args.gguf_repo,
            ollama_name=args.ollama_name, hf_repo=args.hf_repo, params_b=args.params,
            publisher=args.publisher, name=args.model_name, tasks=args.model_tasks,
            license_name=args.license_name, context=args.max_context,
            supports_tools=args.supports_tools, supports_vision=args.supports_vision,
            quants=args.model_quants,
        )
        # Prove the entry actually loads rather than just claiming it was written.
        reloaded = load_catalog()
        spec = reloaded.get(args.model_id)
        if spec is None:
            raise LlmsetupError(f"wrote {path} but {args.model_id} did not load back")
        console.ok(f"{spec.name} is now in the catalog "
                   f"({spec.params_b:g}B, {spec.layers} layers, "
                   f"{reloaded.arch_source(spec.id)} architecture)")
        return 0

    if args.action == "update":
        from llmsetup.catalog.autoupdate import apply_update

        result = apply_update(console, force=True, url=args.url)
        if result.get("updated"):
            console.ok(f"catalog updated: {result['models']} models from "
                       f"{result['url']}")
            console.info(console.dim(f"cached at {result['path']}"))
        else:
            console.warn(f"not updated: {result.get('reason')}")
        if args.json:
            print(json.dumps(result, indent=2))
        return 0 if result.get("updated") else 1

    if args.action == "auto-update":
        from llmsetup.catalog.autoupdate import (
            disable_persistently,
            enable_persistently,
            status,
        )

        if args.disable:
            console.ok(f"automatic catalog updates disabled "
                       f"({disable_persistently()})")
        elif args.enable:
            enable_persistently()
            console.ok("automatic catalog updates enabled")

        state = status()
        if args.json:
            print(json.dumps(state, indent=2))
            return 0
        console.print(console.bold("Automatic catalog updates"))
        console.info(f"enabled: {state['enabled']}")
        console.info(f"source: {state['url']}")
        console.info(f"every: {state['interval_hours']:g} hours")
        console.info(f"last checked: {state['last_checked'] or 'never'}")
        console.info(f"last updated: {state['last_updated'] or 'never'}")
        if state.get("cached_models"):
            console.info(f"cached catalog: {state['cached_models']} models "
                         f"at {state['cached_catalog']}")
        if state.get("last_error"):
            console.warn(f"last error: {state['last_error']}")
        return 0

    if args.action == "refresh":
        report = refresh(console, catalog, check_remote=not args.offline,
                         write_overlay=not args.dry_run)
        if args.json:
            print(json.dumps(report, indent=2))
        return 0

    if args.action == "path":
        for source in catalog.sources:
            console.print(source)
        console.info(console.dim(
            f"add your own entries in {user_catalog_dir()}/*.yaml, "
            f"or run `llmsetup catalog add <id>`"))
        return 0

    if args.action == "show":
        if not args.model_id:
            raise LlmsetupError("`catalog show` needs a model id")
        spec = catalog.require(args.model_id)
        print(json.dumps(spec.to_dict(), indent=2))
        return 0

    if args.json:
        print(json.dumps([m.to_dict() for m in catalog.models], indent=2))
        return 0

    console.print(console.bold(f"{len(catalog)} models (catalog updated "
                               f"{catalog.updated})"))
    console.print()
    header = f"  {'id':30s} {'params':>10s} {'ctx':>7s} {'caps':12s} license"
    console.print(console.bold(header))
    for spec in sorted(catalog.models, key=lambda m: (-m.params_b, m.id)):
        caps = "".join([
            "T" if spec.supports_tools else "-",
            "V" if spec.supports_vision else "-",
            "J" if spec.supports_json else "-",
            "R" if spec.is_reasoning else "-",
            "E" if spec.is_embedding else "-",
        ])
        params = (f"{spec.params_b:g}B"
                  + (f"/{spec.active_params_b:g}B" if spec.is_moe else ""))
        console.print(f"  {spec.id:30s} {params:>10s} "
                      f"{spec.max_context // 1024:>6d}K {caps:12s} {spec.license}")
    console.print()
    console.info(console.dim("caps: T=tools V=vision J=json R=reasoning E=embedding"))
    return 0


def cmd_ui(args, console: Console) -> int:
    """Serve the local web interface.

    Bound to loopback only. The UI plans and explains; it hands the actual
    download over as a command rather than starting one behind a button, so the
    irreversible step stays explicit.
    """
    from llmsetup.ui.server import serve

    serve(console, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_runtimes(args, console: Console) -> int:
    rows = []
    for runtime in all_runtimes():
        available = runtime.is_available()
        rows.append({"name": runtime.name, "label": runtime.label,
                     "available": available,
                     "version": runtime.version() if available else None,
                     "install": "" if available else runtime.install_hint()})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    console.print(console.bold("Inference engines"))
    for row in rows:
        if row["available"]:
            console.ok(f"{row['label']}: installed"
                       + (f" ({row['version']})" if row["version"] else ""))
        else:
            console.info(f"  - {row['label']}: not installed. {row['install']}")
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _auto_update(args_console, profile) -> None:
    """Refresh the catalog if it is due. Silent, bounded, and never fatal.

    Runs before planning so a model released since install is available, but is
    skipped for an offline profile and whenever the user has opted out.
    """
    from llmsetup.catalog.autoupdate import maybe_update

    result = maybe_update(args_console, offline=getattr(profile, "offline_only", False))
    if result.get("updated"):
        args_console.info(args_console.dim(
            f"catalog auto-updated: {result['models']} models "
            f"(dated {result.get('catalog_date', 'unknown')})"
        ))


def _hardware(args):
    hardware = detect_hardware(probe_gpus=not getattr(args, "cpu_only", False))
    overrides = {k: getattr(args, k, None)
                 for k in ("vram", "ram", "gpu_vendor", "cpu_only", "threads")}
    if any(v for v in overrides.values()):
        hardware = apply_overrides(hardware, overrides)
    return hardware


def _show_detection(hardware, console: Console) -> None:
    console.info(f"Detected: {summarize_hardware(hardware)}")
    installed = [name for name, present in hardware.runtime_availability.items() if present]
    console.info("Installed runtimes: " + (", ".join(installed) or "none detected"))
    for warning in hardware.detection_warnings:
        console.warn(warning)


def _profile(args, console: Console) -> UseCaseProfile:
    """Build a profile from an answers file, flags, or the interactive wizard."""
    answers = {}
    if getattr(args, "answers", None):
        answers.update(load_answers(args.answers))

    for key in ("tasks", "clients", "context_tokens", "priority",
                "concurrent_users", "needs_tools", "needs_json", "needs_vision",
                "needs_embeddings", "commercial_use", "offline_only", "requested_model"):
        value = getattr(args, key, None)
        if value is not None:
            answers[key] = value
    if getattr(args, "languages", None):
        answers["languages"] = args.languages
    if args.command in ("setup", "plan") and getattr(args, "model", None) is not None:
        answers["requested_model"] = args.model

    interactive = (sys.stdin.isatty() and not getattr(args, "yes", False)
                   and not getattr(args, "json", False))
    # Flags that already say what the model is for make the interview redundant.
    have_enough = "tasks" in answers or getattr(args, "answers", None) is not None

    if interactive and not have_enough:
        return run_interview(console, defaults=answers)
    return profile_from_answers(infer_flags(answers or {"tasks": ["chat"]}))


def _review_model(args, profile, hardware, catalog, console):
    interactive = (sys.stdin.isatty() and not getattr(args, "yes", False)
                   and not getattr(args, "json", False)
                   and getattr(args, "answers", None) is None)
    return review_model_choice(profile, hardware, catalog, console,
                               runtime=args.runtime, interactive=interactive)


def _default_checks(profile, spec):
    from llmsetup.verify.harness import select_checks

    return select_checks(profile, spec)


def _synthetic_spec(model_id: str, catalog):
    """A placeholder spec for verifying a model that is not in the catalog.

    Conservative on purpose: unknown capabilities are assumed absent so checks
    are advisory rather than falsely required.
    """
    from llmsetup.types import ModelSpec

    return ModelSpec(
        id=model_id, name=model_id, publisher="unknown", params_b=8.0,
        max_context=8192, license="unknown",
        tasks=["chat", "coding", "agent", "rag", "summarize", "extract"],
        supports_tools=True, supports_json=True,
        notes="not in the catalog; capabilities are unverified assumptions",
    )


def _config_for_external(spec, profile, args):
    from llmsetup.types import RuntimeConfig

    return RuntimeConfig(
        runtime="external", model_id=spec.id, quant=getattr(args, "quant", "unknown"),
        context_tokens=profile.context_tokens, total_layers=spec.layers,
        gpu_layers=spec.layers, served_name=getattr(args, "model", spec.id),
    )


def cmd_report(args, console: Console) -> int:
    """Reprint the report from the last setup.

    This is what a bug report should be pasted from: it carries the machine, the
    plan, every tuning decision with the reason it was made, and what the checks
    actually found -- which is the difference between a fixable report and "it
    did not work". It contains no credentials.
    """
    path = args.path or (app_dir() / "last-report.md")
    if not Path(path).exists():
        raise LlmsetupError(
            f"no saved report at {path}",
            hint="run `llmsetup setup` first, every run saves one",
        )
    write_console(sys.stdout, Path(path).read_text(encoding="utf-8"))
    return 0


def cmd_start(args, console: Console) -> int:
    """Run the last verified setup again, with the exact command it was tested with."""
    from llmsetup.launch import load_launch, start
    from llmsetup.util.fs import read_json

    path = args.setup or (app_dir() / "last-setup.json")
    if not Path(path).exists():
        raise LlmsetupError(
            f"no saved setup at {path}",
            hint="run `llmsetup setup` first; a verified run records how to start it",
        )
    record = load_launch(read_json(Path(path)), Path(path))
    if args.json:
        print(json.dumps(record, indent=2))
        return 0
    if args.detach and not args.print_only and not record.get("shared_daemon"):
        from llmsetup.launch import preflight
        from llmsetup.server import ensure_not_running, start_detached

        ensure_not_running()
        preflight(record)
        return start_detached(record, console, wait_s=args.wait)
    return start(record, console, print_only=args.print_only)


def cmd_status(args, console: Console) -> int:
    """Report the background server started by `llmsetup start --detach`."""
    from llmsetup.server import status

    report = status()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["running"] else 3
    if not report["managed"]:
        console.info("no background server is recorded; start one with `llmsetup start --detach`")
        return 3
    if not report["running"]:
        console.warn(f"the recorded server (pid {report['pid']}) is not running")
        console.info(f"its log is {report['log']}; start it again with `llmsetup start --detach`")
        return 3
    (console.ok if report["healthy"] else console.warn)(
        f"{report['model']} is running (pid {report['pid']}) and "
        + ("answering" if report["healthy"] else "not answering yet"))
    console.info(f"OpenAI-compatible endpoint: {report['base_url']}/v1")
    from llmsetup.server import format_duration

    uptime = f" (up {format_duration(report['uptime_s'])})" if report.get("uptime_s") is not None else ""
    console.info(console.dim(f"started {report['started_at']}{uptime}  log {report['log']}"))
    return 0


def cmd_stop(args, console: Console) -> int:
    from llmsetup.server import stop

    return stop(console)


def cmd_restart(args, console: Console) -> int:
    """Stop the background server, if one is running, and start the verified setup detached."""
    from llmsetup.server import read_state, server_running, stop

    state = read_state()
    if state is not None and server_running(state):
        stop(console)
    else:
        console.info("no background server was running; starting one")
    args.detach, args.print_only, args.json = True, False, False
    return cmd_start(args, console)


def cmd_logs(args, console: Console) -> int:
    from llmsetup.server import current_log, logs, tail

    if args.json:
        path = current_log()
        print(json.dumps({"log": str(path), "lines": tail(path, args.lines)}, indent=2))
        return 0
    return logs(console, lines=args.lines, follow=args.follow)


def _connect_context(endpoint, setup_path: Path):
    """The spec and settings the verified setup has for ``endpoint``, or conservative stand-ins."""
    from dataclasses import fields

    from llmsetup.types import RuntimeConfig
    from llmsetup.util.fs import read_json

    loaded = read_json(setup_path) if setup_path.exists() else None
    saved: Dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    stored: Dict[str, Any] = saved["config"] if isinstance(saved.get("config"), dict) else {}
    launch: Dict[str, Any] = saved["launch"] if isinstance(saved.get("launch"), dict) else {}
    served = launch.get("served_model") or stored.get("served_name") or launch.get("model_id")
    catalog = load_catalog()
    if stored and served == endpoint.model:
        known = {f.name for f in fields(RuntimeConfig)}
        config = RuntimeConfig(**{k: v for k, v in stored.items() if k in known})
        spec = catalog.get(config.model_id) or _synthetic_spec(endpoint.model, catalog)
        return spec, config
    spec = catalog.get(endpoint.model) or _synthetic_spec(endpoint.model, catalog)
    return spec, RuntimeConfig(runtime="external", model_id=endpoint.model, quant="unknown",
                               context_tokens=min(spec.max_context, 8192))


def cmd_connect(args, console: Console) -> int:
    """Write the verified endpoint into another tool's config, journalled so undo restores it."""
    from llmsetup.chat import resolve_endpoint
    from llmsetup.connect import (
        HARNESSES,
        INSTRUCTIONS_ONLY,
        Target,
        connect,
        describe,
        instructions,
        supported,
    )

    if args.list or not args.harness:
        for harness_id in supported():
            print(f"  {harness_id:<14} {describe(harness_id)}")
        return 0
    harness_id = args.harness.lower().replace("-", "_")
    if harness_id not in supported():
        raise LlmsetupError(f"connect does not know {args.harness}",
                            hint="one of: " + ", ".join(supported()))
    setup_path = args.setup or (app_dir() / "last-setup.json")
    endpoint = resolve_endpoint(args.base_url, args.model, setup_path, api_key=args.api_key)
    spec, config = _connect_context(endpoint, setup_path)
    target = Target(endpoint=endpoint, spec=spec, config=config, make_default=args.default)

    # Claude Code's user-wide settings would redirect every session, so it is written only per project.
    prints_steps = harness_id in INSTRUCTIONS_ONLY or (
        harness_id == "claude_code" and not args.project and args.config is None)
    if prints_steps:
        steps = instructions(harness_id, target)
        if args.json:
            print(json.dumps(steps, indent=2))
        else:
            console.step(f"{steps['name']}: {steps['path']}")
            print(steps["content"])
        return 0

    result = connect(harness_id, target, path=args.config, project=args.project,
                     rewrite=args.rewrite, dry_run=args.dry_run)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if args.dry_run or k != "content"}, indent=2))
    elif args.dry_run:
        console.step(f"{result['name']}: {result['path']} would become")
        print(result["content"], end="")
    elif not result["changed"]:
        console.ok(f"{result['name']} already lists {result['model']} at {result['base_url']}")
    else:
        console.ok(f"{result['name']} now lists {result['model']} at {result['base_url']} ({result['path']})")
        console.info("`llmsetup undo` puts the file back as it was")
    then = HARNESSES[harness_id].then
    if then and not args.json and not args.dry_run:
        print(then(target))
    return 0


def cmd_disconnect(args, console: Console) -> int:
    """Remove what connect wrote from a tool's config as it is now; journalled, so undo reverses it."""
    from llmsetup.connect import HARNESSES, disconnect, supported

    harness_id = args.harness.lower().replace("-", "_")
    if harness_id not in HARNESSES:
        known = ", ".join(HARNESSES)
        if harness_id in supported():
            raise LlmsetupError(f"connect only printed steps for {args.harness}, so there is nothing to remove",
                                hint=f"disconnect works on the tools connect writes to: {known}")
        raise LlmsetupError(f"disconnect does not know {args.harness}", hint="one of: " + known)
    result = disconnect(harness_id, path=args.config, project=args.project, rewrite=args.rewrite,
                        dry_run=args.dry_run)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if args.dry_run or k != "content"}, indent=2))
    elif not result["changed"]:
        console.ok(f"{result['name']}: nothing of llmsetup's in {result['path']}")
    elif args.dry_run:
        if result["removed"]:
            console.step(f"{result['name']}: {result['path']} would be removed")
        else:
            console.step(f"{result['name']}: {result['path']} would become")
            print(result["content"], end="")
    else:
        what = "removed" if result["removed"] else "no longer lists llmsetup's model"
        console.ok(f"{result['name']}: {result['path']} {what}")
        console.info("`llmsetup undo` puts it back")
    return 0


def cmd_settings(args, console: Console) -> int:
    """The runtime settings --set can override, with their type and the tuner's starting default."""
    import typing
    from dataclasses import MISSING, fields

    from llmsetup.tuning.overrides import settable
    from llmsetup.types import RuntimeConfig

    allowed = settable()
    defaults = {f.name: f.default for f in fields(RuntimeConfig) if f.default is not MISSING}
    rows = []
    for name, kind in allowed.items():
        label = getattr(kind, "__name__", None) or str(kind).replace("typing.", "")
        args_ = typing.get_args(kind)
        if args_:
            label = "/".join(getattr(a, "__name__", str(a)) for a in args_ if a is not type(None)) + " or none"
        rows.append({"key": name, "type": label, "default": defaults.get(name, "(tuned)")})
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0
    console.info("override with `llmsetup setup --set KEY=VALUE` (repeatable); the tuner sets most of these "
                 "for your hardware, so the default is only where it starts")
    for row in rows:
        print(f"  {row['key']:<24} {row['type']:<16} {row['default']}")
    return 0


def cmd_changes(args, console: Console) -> int:
    """List the journal of changes llmsetup made outside its own directory."""
    from llmsetup import changes

    history = changes.runs()
    if args.json:
        print(json.dumps(history, indent=2))
        return 0
    if not history:
        console.info("llmsetup has not changed anything outside its own directory")
        return 0
    for group in history:
        state = f"{group['pending']} to undo" if group["pending"] else "undone"
        print(f"{group['at']}  llmsetup {group['command']}  (run {group['run']}, {state})")
        for change in group["changes"]:
            what = change.get("path") or change.get("name") or change.get("note", "")
            mark = " " if change["status"] == "pending" else "x"
            print(f"  [{mark}] {change['target']:<10} {change['kind']:<12} {what}")
    console.print()
    console.info("`llmsetup undo` reverses the newest run; `--run ID` one run, `--all` everything")
    return 0


def cmd_undo(args, console: Console) -> int:
    """Reverse recorded changes: the newest run by default."""
    from llmsetup import changes

    pending = changes.pending_changes()
    if args.run:
        targets = [c for c in pending if c["run"] == args.run]
        if not targets:
            raise LlmsetupError(f"run {args.run} has nothing left to undo",
                                hint="`llmsetup changes` lists the runs and what each changed")
    elif args.undo_all:
        targets = pending
    else:
        # The journal is append-only, so its order is the order things happened in; timestamps
        # have one-second resolution and would tie for runs a second apart.
        newest = pending[-1]["run"] if pending else None
        targets = [c for c in pending if c["run"] == newest]
    if not targets:
        console.info("nothing to undo")
        return 0
    results = changes.undo(targets, force=args.force, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        verb = "would undo" if args.dry_run else "undid"
        for outcome in results:
            report = {"undone": console.ok, "already": console.info, "manual": console.warn,
                      "kept": console.warn, "failed": console.error}[outcome["status"]]
            report(f"{outcome['target']}: {outcome['note']}")
        done = sum(o["status"] in ("undone", "already") for o in results)
        console.print()
        console.info(f"{verb} {done} of {len(results)} change(s)"
                     + ("; run with --force to also undo files edited since" if any(
                         o["status"] == "kept" for o in results) else ""))
    return 1 if any(o["status"] == "failed" for o in results) else 0


def cmd_chat(args, console: Console) -> int:
    """Talk to the model from the last verified setup, or to any OpenAI-compatible server."""
    from llmsetup.chat import ensure_answering, resolve_endpoint, run

    message = args.message
    if message == "-":
        message = sys.stdin.read().strip()
        if not message:
            raise LlmsetupError("-m - read nothing from stdin", hint="pipe the question in, or pass it after -m")
    if args.json and message is None:
        raise LlmsetupError("--json needs a question to answer", hint='pass one with -m "..."')
    if message is None and not sys.stdin.isatty():
        raise LlmsetupError("chat needs a terminal to read from",
                            hint='ask one question with -m "...", or pipe it in with -m -')
    endpoint = resolve_endpoint(args.base_url, args.model, args.setup or (app_dir() / "last-setup.json"),
                                api_key=args.api_key)
    ensure_answering(endpoint)
    return run(endpoint, console, message=message, system=args.system, max_tokens=args.max_tokens,
               temperature=args.temperature, show_thinking=args.thinking, as_json=args.json)


def _save_last(result) -> None:
    """Record the run so `llmsetup doctor` can re-test it and `llmsetup report` can
    reprint it.

    The markdown is written every time rather than on request, because the run
    people most want a report of is the one they did not think to ask for one
    from.
    """
    from llmsetup.util.fs import write_json, write_text

    try:
        directory = ensure_dir(app_dir())
        from llmsetup.launch import launch_record

        record = result.to_dict()
        launch = launch_record(result)
        if launch is not None:
            record["launch"] = launch
        write_json(directory / "last-setup.json", record)
        write_text(directory / "last-report.md", render_markdown(result))
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
