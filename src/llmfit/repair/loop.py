"""The repair loop.

Verify → diagnose → remediate → re-verify, with a budget, until either
everything the user needs passes or every applicable strategy is exhausted.

Three rules keep it from thrashing:

1. **Monotonic de-escalation.** Each (signature, strategy) pair is tried at most
   once, and strategies are ordered by cost, so the loop always moves toward
   more drastic measures and never returns to a setting that already failed.
   This alone guarantees termination: the number of moves is bounded by the size
   of the strategy table.
2. **Bounded.** ``max_repairs`` caps total work on top of that, so a pathological
   machine produces a reported outcome rather than a long hang. There is
   deliberately no per-signature cap, a hard memory limit legitimately needs to
   escalate through every rung of the ladder, and cutting that short leaves the
   user with a setup that still does not work.
3. **Journalled.** Every action records its trigger, what changed, and the cost,
   so the final report explains why the settings look the way they do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

from llmfit.repair.strategies import Outcome, RepairState, strategies_for
from llmfit.types import CheckResult, CheckStatus, RepairAction, RuntimeConfig
from llmfit.util.log import Console
from llmfit.verify import signatures as sig

#: A relaunch function: given the current config, bring up an endpoint and run
#: the named checks (or all of them when ``names`` is None).
Relauncher = Callable[[RuntimeConfig, Optional[Sequence[str]]], List[CheckResult]]


@dataclass
class RepairLoop:
    state: RepairState
    relaunch: Relauncher
    console: Console
    max_repairs: int = 12

    journal: List[RepairAction] = field(default_factory=list)
    _tried: Set[Tuple[str, str]] = field(default_factory=set)

    def run(self, results: List[CheckResult]) -> Tuple[List[CheckResult], List[RepairAction]]:
        """Repair until the setup passes or we run out of moves."""
        for attempt in range(self.max_repairs):
            failures = self._blocking(results)
            if not failures:
                if attempt:
                    self.console.ok(f"all required checks pass after {attempt} repair(s)")
                return results, self.journal

            failure = failures[0]
            signature = failure.failure_signature or sig.UNKNOWN_ERROR
            self.console.step(
                f"repair {attempt + 1}/{self.max_repairs}: "
                f"{failure.name} failed ({signature})"
            )

            outcome = self._apply_next(signature, failure)
            if outcome is None:
                # Every rung of the ladder for this signature has been tried.
                self.console.warn(
                    f"every available fix for {signature} has been tried; "
                    f"leaving the setup at its last working state"
                )
                return results, self.journal

            results = self._reverify(outcome, results)

        self.console.warn(f"repair budget of {self.max_repairs} exhausted")
        return results, self.journal

    # ------------------------------------------------------------- internals

    def _blocking(self, results: List[CheckResult]) -> List[CheckResult]:
        """Failures worth repairing, most severe first."""
        blocking = [r for r in results if r.blocking and r.failure_signature]
        # A WARN carrying a signature is repairable too, but only after real
        # failures are gone.
        warnings = [r for r in results
                    if r.status is CheckStatus.WARN and r.failure_signature]
        return blocking + warnings

    def _apply_next(self, signature: str, failure: CheckResult) -> Optional[Outcome]:
        """Cheapest applicable strategy that has not been tried for this signature."""
        for strategy in strategies_for(signature):
            key = (signature, strategy.name)
            if key in self._tried and not strategy.exhausts_itself:
                continue
            self._tried.add(key)

            outcome = strategy.apply(self.state)
            if outcome is None:
                self.console.debug(f"strategy {strategy.name} does not apply here")
                continue

            self.console.info(f"{strategy.name} [{outcome.cost}]: {outcome.description}")
            if outcome.switch_to is not None:
                self._switch_candidate(outcome)

            self.journal.append(RepairAction(
                strategy=strategy.name,
                trigger=f"{failure.name}: {signature}",
                description=outcome.description,
                cost=outcome.cost,
                before=outcome.before,
                after=outcome.after,
            ))
            return outcome
        return None

    def _switch_candidate(self, outcome: Outcome) -> None:
        """Adopt a different model, re-deriving its settings from scratch.

        Carrying the old model's tuning across would be wrong, layer counts,
        memory footprint and sampling defaults are all model-specific.
        """
        from llmfit.tuning import tune

        new_candidate = outcome.switch_to
        assert new_candidate is not None
        tuned = tune(
            new_candidate, self.state.hardware, self.state.profile,
            self.state.config.runtime, port=self.state.config.port,
            catalog=self.state.catalog,
        )
        # Preserve the shrunken budget: whatever forced the switch still applies.
        tuned.config.extra.update({
            k: v for k, v in self.state.config.extra.items()
            if k in ("ollama_port", "lmstudio_port")
        })
        self.state.candidate = new_candidate
        self.state.config = tuned.config

    def _reverify(self, outcome: Outcome, previous: List[CheckResult]) -> List[CheckResult]:
        """Re-run checks after a change, merging into the previous results.

        Only the checks a strategy could plausibly have affected are re-run when
        the strategy says so; anything touching memory or the model re-runs
        everything, because those change the whole picture.
        """
        names = list(outcome.recheck) if outcome.recheck else None
        fresh = self.relaunch(self.state.config, names)

        if names is None:
            return fresh

        by_name = {r.name: r for r in previous}
        for result in fresh:
            by_name[result.name] = result
        # Preserve the original ordering.
        order = [r.name for r in previous] + [
            r.name for r in fresh if r.name not in {p.name for p in previous}
        ]
        seen, merged = set(), []
        for name in order:
            if name in by_name and name not in seen:
                seen.add(name)
                merged.append(by_name[name])
        return merged


def mark_resolved(journal: List[RepairAction], results: List[CheckResult]) -> None:
    """Annotate each journal entry with whether its trigger check now passes."""
    status = {r.name: r.status for r in results}
    for action in journal:
        check_name = action.trigger.split(":", 1)[0].strip()
        current = status.get(check_name)
        if current is None:
            action.resolved = None
        else:
            action.resolved = current in (CheckStatus.PASS, CheckStatus.SKIP,
                                          CheckStatus.WARN)
