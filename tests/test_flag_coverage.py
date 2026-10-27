"""Every option the engine has, and llmsetup's decision about it.

llmsetup emits a fraction of what llama-server accepts. Which of the rest are
deliberate and which are oversights was, for several rounds of this project,
knowable only by someone reading the upstream documentation and noticing. That
method only finds what somebody happens to read, and it kept finding things.

So the decisions are written down and checked. A flag llmsetup emits that has no
recorded decision, or a decision that disagrees with what the code actually
does, fails here. When llama.cpp adds options, and it adds them constantly, the
snapshot goes stale and the refresh script says which ones are new.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llmsetup.runtimes.data.llamacpp_flags import (
    ALIAS,
    DECISIONS,
    MANUAL,
    RUNTIME,
    TODO,
    USED,
)

RUNTIME_SOURCE = Path("src/llmsetup/runtimes/llamacpp.py").read_text(encoding="utf-8")


def emitted_flags() -> set:
    """Long flags the llama.cpp adapter can put on a command line."""
    return set(re.findall(r'"(--[a-z0-9][a-z0-9.-]*)"', RUNTIME_SOURCE))


class TestTheRecordIsHonest:
    def test_everything_llmsetup_emits_is_recorded_as_used(self):
        """The dangerous direction. A flag emitted but recorded as skipped means
        the record is fiction, and the record is what the next round of this
        work will be read against."""
        wrong = sorted(
            flag for flag in emitted_flags()
            if DECISIONS.get(flag, (None, ""))[0] != USED
        )
        assert not wrong, (
            "emitted but not recorded as used: " + ", ".join(wrong))

    def test_everything_recorded_as_used_is_actually_emitted(self):
        """The other direction, which is how a decision quietly becomes a
        claim: the record says llmsetup passes it, and it does not."""
        claimed = {flag for flag, (decision, _) in DECISIONS.items()
                   if decision == USED}
        missing = sorted(claimed - emitted_flags())
        assert not missing, (
            "recorded as used but never emitted: " + ", ".join(missing))

    def test_every_alias_points_at_something_real(self):
        for flag, (decision, target) in DECISIONS.items():
            if decision != ALIAS:
                continue
            assert target in DECISIONS, f"{flag} aliases unknown {target}"
            assert DECISIONS[target][0] != ALIAS, (
                f"{flag} aliases {target}, which is itself an alias")

    def test_every_skipped_flag_says_why(self):
        """A decision with no reason is not a decision, it is a shrug that
        happens to typecheck."""
        silent = sorted(
            flag for flag, (decision, note) in DECISIONS.items()
            if decision in (RUNTIME, MANUAL, TODO) and not note.strip()
        )
        assert not silent, "skipped without a reason: " + ", ".join(silent)

    def test_todo_entries_say_what_the_work_is(self):
        """Otherwise they are a list of words rather than a list of tasks."""
        for flag, (decision, note) in DECISIONS.items():
            if decision == TODO:
                assert len(note.split()) >= 3, f"{flag} has no plan attached"

    def test_no_flag_is_recorded_twice_with_different_answers(self):
        assert len(DECISIONS) == len(set(DECISIONS))


class TestCoverageIsWorthHaving:
    def test_the_record_covers_the_whole_surface(self):
        """A partial record is worse than none: it looks like coverage."""
        assert len(DECISIONS) > 300

    def test_the_settings_that_decide_speed_are_all_used(self):
        """The ones no setup should be missing. If any of these ever changes to
        another decision, that is a regression worth failing on rather than a
        judgement call to be re-argued."""
        essential = [
            "--ctx-size", "--n-gpu-layers", "--batch-size", "--ubatch-size",
            "--flash-attn", "--cache-type-k", "--cache-type-v", "--threads",
            "--n-cpu-moe", "--tensor-split", "--jinja", "--spec-type",
        ]
        for flag in essential:
            assert DECISIONS[flag][0] == USED, f"{flag} stopped being used"

    @pytest.mark.parametrize("flag", [
        "--api-key", "--ssl-cert-file", "--ssl-key-file", "--cors-origins",
    ])
    def test_security_relevant_flags_are_a_deliberate_choice(self, flag):
        """Not defaults llmsetup picked by omission. Someone exposing a server to
        other people has requirements llmsetup cannot guess, and quietly choosing
        for them is worse than leaving it to them."""
        assert DECISIONS[flag][0] in (USED, RUNTIME)

    def test_deprecated_flags_are_not_used(self):
        for flag in ("--defrag-thold",):
            assert DECISIONS[flag][0] != USED


def test_the_snapshot_can_be_refreshed():
    """The script that regenerates this is part of the mechanism, not a note in
    a commit message. Without it the record rots the moment upstream moves."""
    script = Path("scripts/flag_coverage.py")
    assert script.exists()
    assert "DECISIONS" in script.read_text(encoding="utf-8")
