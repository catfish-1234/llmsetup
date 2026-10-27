#!/usr/bin/env python3
"""Report llama-server options llmsetup has no recorded decision about.

llmsetup emits a fraction of what the engine accepts, and the difference between
a deliberate omission and an oversight is only visible if somebody wrote the
decision down. `src/llmsetup/runtimes/data/llamacpp_flags.py` is that record, and
this is how it stays current: llama.cpp adds options constantly, and each one
arrives as a gap nobody has looked at.

Run it against an installed binary:

    python scripts/flag_coverage.py

or against a saved help dump, which is how it works with no llama.cpp present:

    llama-server --help > help.txt && python scripts/flag_coverage.py help.txt
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmsetup.runtimes.data.llamacpp_flags import DECISIONS, TODO, USED

FLAG = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9.-]*)")


def help_text(argument: str | None) -> str:
    if argument:
        return Path(argument).read_text(encoding="utf-8")
    try:
        result = subprocess.run(["llama-server", "--help"], capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not run llama-server --help: {exc}") from exc
    return (result.stdout or "") + (result.stderr or "")


def main() -> int:
    text = help_text(sys.argv[1] if len(sys.argv) > 1 else None)
    found = set(FLAG.findall(text))
    if len(found) < 20:
        raise SystemExit("that does not look like a help screen")

    unreviewed = sorted(found - set(DECISIONS))
    retired = sorted(set(DECISIONS) - found)
    used = sorted(f for f, (d, _) in DECISIONS.items() if d == USED)
    todo = sorted(f for f, (d, _) in DECISIONS.items() if d == TODO)

    print(f"{len(found)} options in this build")
    print(f"{len(used)} used by llmsetup, {len(todo)} recorded as worth having")
    print()

    if unreviewed:
        print(f"NEW, no decision recorded ({len(unreviewed)}):")
        for flag in unreviewed:
            print(f"  {flag}")
        print()
        print("Add each to DECISIONS with a decision and a reason.")
    else:
        print("Every option in this build has a recorded decision.")

    if retired:
        print()
        print(f"recorded but absent from this build ({len(retired)}), which is "
              f"expected on an older binary:")
        for flag in retired[:20]:
            print(f"  {flag}")
        if len(retired) > 20:
            print(f"  ... and {len(retired) - 20} more")

    return 1 if unreviewed else 0


if __name__ == "__main__":
    raise SystemExit(main())
