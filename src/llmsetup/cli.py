"""Command line interface.

Subcommands mirror the pipeline stages, so any one of them can be run and
inspected on its own:

    llmsetup setup       the whole thing
    llmsetup detect      what hardware do I have?
    llmsetup plan        what would you pick, and why?
    llmsetup verify      does my current endpoint actually work?
    llmsetup doctor      re-test and repair an existing setup
    llmsetup export      register an existing model with my other apps
    llmsetup catalog     inspect or extend the model list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from llmsetup.catalog.catalog import load_catalog
from llmsetup.hardware import apply_overrides, detect_hardware
from llmsetup.hardware.detect import summarize as summarize_hardware
from llmsetup.interview.questions import infer_flags
from llmsetup.interview.wizard import (
    load_answers,
    profile_from_answers,
    run_interview,
    save_profile,
)
from llmsetup.pipeline import Pipeline, companion_embedding_plan
from llmsetup.report.render import render_markdown, render_terminal
from llmsetup.runtimes.registry import all_runtimes
from llmsetup.types import UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import app_dir, ensure_dir, write_text
from llmsetup.util.log import Console
from llmsetup.util.units import gb

VERSION = "0.1.0"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(quiet=getattr(args, "quiet", False),
                      verbose=getattr(args, "verbose", False))

    handler = {
        "setup": cmd_setup, "detect": cmd_detect, "plan": cmd_plan,
        "verify": cmd_verify, "doctor": cmd_doctor, "export": cmd_export,
        "catalog": cmd_catalog, "runtimes": cmd_runtimes, "ui": cmd_ui,
        "report": cmd_report,
    }.get(args.command)

    if handler is None:
        parser.print_help()
        return 2

    try:
        return handler(args, console)
    except LlmsetupError as exc:
        console.error(str(exc))
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
    p_setup.add_argument("--model", help="force a catalog model id")
    p_setup.add_argument("--quant", help="force a quantization, e.g. Q4_K_M")
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


def cmd_plan(args, console: Console) -> int:
    from llmsetup.planner.plan import build_adaptive_plan
    from llmsetup.planner.score import explain

    hardware = _hardware(args)
    _show_detection(hardware, console)
    profile = _profile(args, console)
    _auto_update(console, profile)
    catalog = load_catalog(extra_paths=getattr(args, "catalog", None))
    plan = build_adaptive_plan(profile, hardware, catalog, runtime=args.runtime)
    profile = plan.profile or profile

    if args.json:
        print(json.dumps({
            "runtime": plan.runtime,
            "runtime_reason": plan.runtime_reason,
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
        console.info(console.dim(f"   {explain(candidate, profile)}"))
        console.info(console.dim(
            f"   weights {gb(candidate.estimate.weights)} + "
            f"KV {gb(candidate.estimate.kv_cache)} ({candidate.estimate.kv_quant}) + "
            f"overhead {gb(candidate.estimate.overhead)}"
        ))
        for warning in candidate.warnings:
            console.warn(f"   {warning}")
    console.print()
    for note in plan.notes:
        console.info(console.dim(note))

    embedder = companion_embedding_plan(profile, hardware, catalog)
    if embedder:
        console.print()
        console.print(console.bold("Companion embedding model"))
        console.info(f"{embedder.name} ({embedder.license}): "
                     f"{embedder.max_context:,} token window")
    return 0


def cmd_setup(args, console: Console) -> int:
    hardware = _hardware(args)
    _show_detection(hardware, console)
    profile = _profile(args, console)
    _auto_update(console, profile)
    catalog = load_catalog(extra_paths=getattr(args, "catalog", None))

    if args.save_profile:
        save_profile(profile, args.save_profile)
        console.ok(f"answers saved to {args.save_profile}")

    if args.model:
        # An explicit model choice still gets tuned and verified; it just skips
        # the ranking. Validate it here so the error is immediate.
        catalog = catalog.restrict_to(args.model)
        if args.quant:
            catalog.models[0].recommended_quants = [args.quant]

    pipeline = Pipeline(
        profile, hardware, catalog, console,
        runtime_name=args.runtime, quick=args.quick, dry_run=args.dry_run,
        max_repairs=0 if args.no_repair else 12,
        register=not args.no_register,
        sweep=getattr(args, "sweep", False),
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
    console.print(console.bold(
        f"{len(results) - len(failed)}/{len(results)} checks passed"
    ))
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
    catalog = catalog.restrict_to(model_id)

    pipeline = Pipeline(profile, hardware, catalog, console,
                        runtime_name=saved.get("runtime"))
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
                "needs_embeddings", "commercial_use", "offline_only"):
        value = getattr(args, key, None)
        if value is not None:
            answers[key] = value
    if getattr(args, "languages", None):
        answers["languages"] = args.languages

    interactive = (sys.stdin.isatty() and not getattr(args, "yes", False)
                   and not getattr(args, "json", False))
    # Flags that already say what the model is for make the interview redundant.
    have_enough = "tasks" in answers or getattr(args, "answers", None) is not None

    if interactive and not have_enough:
        return run_interview(console, defaults=answers)
    return profile_from_answers(infer_flags(answers or {"tasks": ["chat"]}))


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
    sys.stdout.write(Path(path).read_text(encoding="utf-8"))
    return 0


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
        write_json(directory / "last-setup.json", result.to_dict())
        write_text(directory / "last-report.md", render_markdown(result))
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
