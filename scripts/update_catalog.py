#!/usr/bin/env python3
"""Refresh the bundled catalog from Hugging Face.

Run on a schedule by .github/workflows/catalog-update.yml. Discovers newly
popular open-weight models, derives their architecture from config.json, and
merges them into the shipped catalog.

The property that makes unattended commits acceptable: this script **validates
before it writes**. A discovered entry has to survive the catalog loader, a
plausibility check, and a real planning run before it is allowed into the file.
If any of that fails, nothing is written and the job fails loudly rather than
shipping a catalog that produces wrong memory plans.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from llmfit.catalog.catalog import BUNDLED, load_catalog  # noqa: E402
from llmfit.catalog.discover import discover  # noqa: E402
from llmfit.util.log import Console  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80,
                        help="how many trending models to examine")
    parser.add_argument("--max-new", type=int, default=8,
                        help="cap on entries added in one run, so a bad day on "
                             "the Hub cannot flood the catalog")
    parser.add_argument("--min-downloads", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    parser.add_argument("--output", type=Path, default=BUNDLED)
    args = parser.parse_args()

    console = Console(stream=sys.stdout, color=False)
    catalog = load_catalog(include_user_dir=False)
    known = {m.id for m in catalog.models} | {
        (m.hf_repo or "").lower() for m in catalog.models
    }

    console.step(f"catalog has {len(catalog)} models; looking for new ones")
    candidates = discover(
        console, limit=args.limit, min_downloads=args.min_downloads,
        known_ids=list(known),
    )
    if not candidates:
        console.info("nothing new found; the catalog is already current")
        return 0

    candidates.sort(key=lambda c: (-c.downloads, c.id))
    selected = candidates[:args.max_new]
    console.step(f"adding {len(selected)} of {len(candidates)} discovered models")

    document = yaml.safe_load(BUNDLED.read_text(encoding="utf-8"))
    existing_ids = {m["id"] for m in document["models"]}
    added: List[Dict[str, Any]] = []
    for candidate in selected:
        if candidate.id in existing_ids:
            continue
        entry = candidate.to_entry()
        document["models"].append(entry)
        existing_ids.add(candidate.id)
        added.append(entry)
        console.ok(f"+ {entry['id']} ({entry['params_b']}B, "
                   f"{entry['layers']} layers, {entry['license']})")

    if not added:
        console.info("every discovered model was already present")
        return 0

    original = BUNDLED.read_text(encoding="utf-8")
    text = _append_entries(original, added)

    if not validate(text, console, [e["id"] for e in added]):
        console.error("the updated catalog failed validation; nothing was written")
        return 1

    if args.dry_run:
        console.info(f"--dry-run: would add {len(added)} models")
        return 0

    args.output.write_text(text, encoding="utf-8")
    console.ok(f"wrote {args.output} with {len(added)} new models")
    _emit_summary(added)
    return 0


def validate(text: str, console: Console, new_ids: List[str]) -> bool:
    """Prove the updated catalog is usable before it is written.

    Three gates, each catching a different way an auto-generated entry goes
    wrong: it must parse, its numbers must be plausible, and it must survive a
    real planning run — which is what would actually expose a bad layer count.
    """
    import tempfile

    from llmfit.planner.plan import build_plan
    from llmfit.types import GIB, GPU, Accelerator, HardwareProfile, Task, UseCaseProfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            catalog = load_catalog(extra_paths=[path], include_user_dir=False)
        except Exception as exc:
            console.error(f"catalog does not load: {exc}")
            return False

        for model_id in new_ids:
            spec = catalog.get(model_id)
            if spec is None:
                console.error(f"{model_id} did not load back")
                return False
            for field, low, high in (("params_b", 0.05, 2000), ("layers", 1, 200),
                                     ("kv_heads", 1, 128),
                                     ("hidden_size", 128, 32768),
                                     ("max_context", 512, 10_000_000)):
                value = getattr(spec, field)
                if not low <= value <= high:
                    console.error(f"{model_id}.{field} = {value} is implausible")
                    return False

        hardware = HardwareProfile(
            os="linux", arch="x86_64", cpu_name="validator", physical_cores=8,
            logical_cores=16, ram_bytes=64 * GIB, disk_free_bytes=500 * GIB,
            gpus=[GPU(0, "NVIDIA RTX 4090", Accelerator.NVIDIA, 24 * GIB,
                      compute_capability="8.9")],
        )
        for task in (Task.CHAT, Task.CODING):
            try:
                plan = build_plan(
                    UseCaseProfile(tasks=[task], context_tokens=8192), hardware, catalog
                )
            except Exception as exc:
                console.error(f"planning for {task.value} failed: {exc}")
                return False
            if plan.best is None or plan.best.estimate.total <= 0:
                console.error(f"planning for {task.value} produced no usable result")
                return False

    console.ok("validated: catalog loads, values are plausible, planning works")
    return True


def _append_entries(original: str, entries: List[Dict[str, Any]]) -> str:
    """Add entries to the end of the catalog without rewriting the rest of it.

    Round-tripping the whole document through a YAML dumper looks tidier and is
    much worse: it drops every comment, reflows the layout, and turns the next
    hand edit into a merge conflict across the entire file. `models:` is the
    last top-level key, so new entries can go at the end and every existing byte
    can stay exactly as its author left it.
    """
    body = yaml.safe_dump({"models": entries}, sort_keys=False, width=100,
                          allow_unicode=True, default_flow_style=False)
    # Drop the "models:" line the dumper adds and indent the rest into the
    # existing list, which is written two spaces in.
    lines = body.splitlines()[1:]
    block = "\n".join("  " + line if line.strip() else line for line in lines)

    updated = re.sub(r'^updated:.*$',
                     f'updated: "{datetime.date.today().isoformat()}"',
                     original, count=1, flags=re.M)
    separator = "" if updated.endswith("\n") else "\n"
    return (updated + separator
            + "\n  # ---------------------------------------- added "
            + f"{datetime.date.today().isoformat()} by `llmfit catalog refresh`\n"
            + block + "\n")


def _emit_summary(added: List[Dict[str, Any]]) -> None:
    """Write a GitHub Actions step summary so the run is readable at a glance."""
    import os

    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    rows = "\n".join(
        f"| `{e['id']}` | {e['params_b']}B | {e['layers']} | {e['max_context']:,} "
        f"| {e['license']} |"
        for e in added
    )
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(
                f"### Added {len(added)} model(s)\n\n"
                "| id | params | layers | context | licence |\n"
                "|---|---|---|---|---|\n" + rows + "\n"
            )
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
