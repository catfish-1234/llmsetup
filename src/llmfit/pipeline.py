"""End-to-end orchestration: interview → detect → plan → tune → provision →
verify → repair → register.

This module owns the wiring only. Every decision belongs to the stage that makes
it; if you find logic here that decides something, it is in the wrong place.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from llmfit.catalog.catalog import Catalog
from llmfit.gguf import apply_to_spec, read_gguf_info
from llmfit.interop.export import register_everywhere
from llmfit.planner.plan import build_plan, plan_embedding_model
from llmfit.repair.loop import RepairLoop, mark_resolved
from llmfit.repair.strategies import RepairState
from llmfit.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmfit.runtimes.registry import get_runtime
from llmfit.tuning import tune
from llmfit.types import (
    CheckResult,
    CheckStatus,
    HardwareProfile,
    ModelSpec,
    RuntimeConfig,
    SetupResult,
    Severity,
    UseCaseProfile,
)
from llmfit.util.errors import LlmfitError
from llmfit.util.log import Console
from llmfit.verify import signatures as sig
from llmfit.verify.client import OpenAIClient
from llmfit.verify.harness import Harness, select_checks


class Pipeline:
    def __init__(self, profile: UseCaseProfile, hardware: HardwareProfile,
                 catalog: Catalog, console: Console, *,
                 runtime_name: Optional[str] = None, quick: bool = False,
                 dry_run: bool = False, max_repairs: int = 12,
                 register: bool = True):
        self.profile = profile
        self.hardware = hardware
        self.catalog = catalog
        self.console = console
        self.runtime_name = runtime_name
        self.quick = quick
        self.dry_run = dry_run
        self.max_repairs = max_repairs
        self.register = register
        self._endpoint: Optional[Endpoint] = None
        self._runtime: Optional[Runtime] = None

    # ------------------------------------------------------------------ run

    def run(self) -> SetupResult:
        started = _now()

        self.console.step("Planning")
        plan = build_plan(self.profile, self.hardware, self.catalog,
                          runtime=self.runtime_name)
        candidate = plan.best
        assert candidate is not None
        self.console.ok(
            f"{candidate.model.name} at {candidate.quant} "
            f"(~{candidate.estimate.total_gb:.1f} GB) on {plan.runtime}"
        )
        for note in plan.notes:
            self.console.info(self.console.dim(note))
        for warning in candidate.warnings:
            self.console.warn(warning)

        self.console.step("Tuning")
        tuned = tune(candidate, self.hardware, self.profile, plan.runtime)
        config = tuned.config
        for line in tuned.explain():
            self.console.info(self.console.dim(line))

        result = SetupResult(
            profile=self.profile, hardware=self.hardware, plan=plan,
            config=config, started_at=started,
        )
        if self.dry_run:
            self.console.info("--dry-run: stopping before download and launch")
            result.finished_at = _now()
            return result

        self.console.step("Provisioning")
        runtime = get_runtime(plan.runtime)
        self._runtime = runtime
        if not runtime.is_available():
            raise LlmfitError(
                f"{runtime.label} is not installed",
                hint=runtime.install_hint(),
            )
        config = runtime.ensure_model(candidate.model, config, self.console)

        # Ground truth beats the catalog: correct the plan from the real file.
        self._correct_from_gguf(candidate, config, plan)

        self.console.step("Starting the server")
        checks = self._launch_and_check(candidate.model, config, None)

        if any(c.blocking for c in checks):
            self.console.step("Repairing")
            state = RepairState(
                config=config, candidate=candidate, hardware=self.hardware,
                profile=self.profile, catalog=self.catalog,
                fallbacks=list(plan.fallbacks),
            )
            loop = RepairLoop(state=state, relaunch=self._relaunch,
                              console=self.console, max_repairs=self.max_repairs)
            checks, journal = loop.run(checks)
            mark_resolved(journal, checks)
            result.repairs = journal
            # The loop may have swapped model or settings.
            config = state.config
            candidate = state.candidate
            plan.candidates = [candidate] + [c for c in plan.candidates
                                             if c.label != candidate.label]

        result.config = config
        result.checks = checks

        if self.register:
            self.console.step("Registering with your other apps")
            gguf_path = runtime.local_model_path(candidate.model, config)
            result.registrations = register_everywhere(
                candidate.model, config, self._endpoint, self.console,
                gguf_path=gguf_path, requested=self.profile.clients,
            )

        result.finished_at = _now()
        return result

    # ---------------------------------------------------------- provisioning

    def _correct_from_gguf(self, candidate, config: RuntimeConfig, plan) -> None:
        """Replace catalog estimates with the architecture stated in the file."""
        if not config.model_path:
            return
        path = Path(config.model_path)
        if not path.exists() or path.suffix != ".gguf":
            return

        info = read_gguf_info(path)
        corrected, corrections = apply_to_spec(candidate.model, info)
        if not corrections:
            return

        self.console.info("read the real architecture from the GGUF header:")
        for correction in corrections:
            self.console.info(self.console.dim(f"  {correction}"))
        candidate.model = corrected

        # Re-derive the memory estimate; the old one was based on wrong numbers.
        from llmfit.planner.memory import estimate_memory

        bpw = self.catalog.bpw(config.quant)
        candidate.estimate = estimate_memory(
            corrected, bpw, config.context_tokens, quant=config.quant,
            kv_quant=config.kv_cache_type, runtime=config.runtime,
            ubatch=config.ubatch_size, parallel=config.parallel,
        )
        config.total_layers = corrected.layers
        if config.gpu_layers > corrected.layers:
            config.gpu_layers = corrected.layers
        plan.notes.append(
            "memory figures were recomputed from the downloaded file's own "
            "metadata rather than the catalog"
        )

    def _launch_and_check(self, spec: ModelSpec, config: RuntimeConfig,
                          names: Optional[Sequence[str]]) -> List[CheckResult]:
        """Bring up a server and run checks. Launch failures become check results."""
        assert self._runtime is not None
        self._stop_endpoint()

        try:
            endpoint = self._runtime.start(spec, config, self.console)
        except RuntimeUnavailable as exc:
            # A failed launch is a check failure, so repair can act on it rather
            # than the whole run aborting.
            return [CheckResult(
                name="health", status=CheckStatus.FAIL, severity=Severity.CRITICAL,
                message=str(exc),
                failure_signature=sig.classify(str(exc)) or sig.CRASH_DURING_LOAD,
                detail=getattr(exc, "install_hint", "") or "",
            )]
        self._endpoint = endpoint

        harness = Harness(
            client=OpenAIClient(endpoint), spec=spec, config=config,
            profile=self.profile, hardware=self.hardware, console=self.console,
            server_log=endpoint.log_path,
        )
        chosen = list(names) if names else select_checks(self.profile, spec, self.quick)
        return harness.run(chosen)

    def _relaunch(self, config: RuntimeConfig,
                  names: Optional[Sequence[str]]) -> List[CheckResult]:
        """Callback handed to the repair loop."""
        assert self._runtime is not None
        spec = self._current_spec(config)
        if not config.model_path or config.quant != getattr(
                self, "_provisioned_quant", config.quant):
            config = self._runtime.ensure_model(spec, config, self.console)
        self._provisioned_quant = config.quant
        return self._launch_and_check(spec, config, names)

    def _current_spec(self, config: RuntimeConfig) -> ModelSpec:
        return self.catalog.require(config.model_id)

    def _stop_endpoint(self) -> None:
        if self._endpoint and self._runtime:
            self._runtime.stop(self._endpoint)
            self._endpoint = None

    def shutdown(self) -> None:
        self._stop_endpoint()

    @property
    def endpoint(self) -> Optional[Endpoint]:
        return self._endpoint


def companion_embedding_plan(profile: UseCaseProfile, hardware: HardwareProfile,
                             catalog: Catalog) -> Optional[ModelSpec]:
    """The separate embedding model a RAG setup needs, if any."""
    return plan_embedding_model(profile, hardware, catalog)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
