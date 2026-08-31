"""End-to-end orchestration: interview → detect → plan → tune → provision →
verify → repair → register.

This module owns the wiring only. Every decision belongs to the stage that makes
it; if you find logic here that decides something, it is in the wrong place.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
    Task,
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
        tuned = tune(candidate, self.hardware, self.profile, plan.runtime,
                     catalog=self.catalog)
        config = tuned.config
        for line in tuned.explain():
            self.console.info(self.console.dim(line))

        result = SetupResult(
            profile=self.profile, hardware=self.hardware, plan=plan,
            config=config, started_at=started,
        )

        from llmfit.tuning.platform_tuning import plan_platform

        result.platform_actions = [
            action.to_dict() for action in plan_platform(
                self.hardware, config,
                estimate_bytes=candidate.estimate.total,
            )
        ]
        for action in result.platform_actions:
            if not action["optional"]:
                self.console.warn(f"{action['title']}: {action['command']}")
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
        self._ensure_draft_model(runtime, config)

        # Ground truth beats the catalog: correct the plan from the real file.
        self._correct_from_gguf(candidate, config, plan)

        self.console.step("Starting the server")
        checks = self._launch_and_check(candidate.model, config, None)

        if any(c.blocking for c in checks):
            self.console.step("Repairing")
            state = RepairState(
                config=config, candidate=candidate, hardware=self.hardware,
                profile=self.profile, catalog=self.catalog,
                fallbacks=list(plan.fallbacks), console=self.console,
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

        # A RAG setup is only half-built without a retriever, so this runs after
        # the chat model is verified rather than being left as advice.
        if self.profile.needs_embeddings or Task.RAG in self.profile.tasks:
            self.console.step("Setting up the embedding model for retrieval")
            result.embedding = self._setup_embedding(runtime)

        result.finished_at = _now()
        return result

    # ------------------------------------------------------------- embedding

    def _setup_embedding(self, runtime: Runtime) -> Dict[str, Any]:
        """Provision and verify the companion retriever.

        Deliberately non-fatal: a working chat model with a broken retriever is
        still a useful outcome, so this reports its own status instead of
        failing the run. But it is *verified* — an embedding endpoint that
        returns vectors which do not separate meaning is worse than none, since
        it fails silently at retrieval time.
        """
        spec = plan_embedding_model(self.profile, self.hardware, self.catalog)
        if spec is None:
            return {"ok": False, "reason": "no embedding model in the catalog"}

        quant = spec.recommended_quants[0] if spec.recommended_quants else "F16"
        record: Dict[str, Any] = {"model": spec.id, "name": spec.name, "quant": quant}

        config = RuntimeConfig(
            runtime=runtime.name,
            model_id=spec.id,
            quant=quant,
            # Embedding models are small and the window is the document chunk
            # size; there is nothing to gain from a larger one.
            context_tokens=min(spec.max_context, 8192),
            total_layers=spec.layers,
            gpu_layers=spec.layers if self.hardware.gpus else 0,
            batch_size=512,
            ubatch_size=512,
            port=self._free_port(),
            served_name=spec.id,
            extra={"llamacpp_flags": ["--embeddings", "--pooling", "mean"]},
        )

        try:
            config = runtime.ensure_model(spec, config, self.console)
            endpoint = runtime.start(spec, config, self.console)
        except (RuntimeUnavailable, LlmfitError) as exc:
            self.console.warn(f"could not start the embedding model: {exc}")
            return {**record, "ok": False, "reason": str(exc)}

        try:
            harness = Harness(
                client=OpenAIClient(endpoint), spec=spec, config=config,
                profile=self.profile, hardware=self.hardware, console=self.console,
                server_log=endpoint.log_path,
            )
            checks = harness.run(["health", "embeddings"])
            ok = not any(c.blocking for c in checks)
            record.update({
                "ok": ok,
                "base_url": f"{endpoint.base_url.rstrip('/')}/v1",
                "checks": [c.to_dict() for c in checks],
                "dimensions": next(
                    (c.metrics.get("dim") for c in checks if c.name == "embeddings"), None
                ),
            })
            if ok and self.register:
                gguf = runtime.local_model_path(spec, config)
                record["registrations"] = register_everywhere(
                    spec, config, endpoint, self.console, gguf_path=gguf,
                    requested=self.profile.clients, write_snippets=False,
                )
        finally:
            # The embedding server is left running only if the user's runtime
            # owns it (Ollama, LM Studio); ones we spawned are ours to stop.
            if endpoint.started_by_us:
                runtime.stop(endpoint)
                record["note"] = (
                    "the embedding server was stopped after verification; start it "
                    "again with the command in the report"
                )
        return record

    def _free_port(self) -> int:
        from llmfit.util import http

        return http.find_free_port("127.0.0.1", 11500)

    # ---------------------------------------------------------- provisioning

    def _ensure_draft_model(self, runtime: Runtime, config: RuntimeConfig) -> None:
        """Fetch the speculative draft model, if one was planned.

        Failing to get the draft is not fatal: speculation is an optimisation,
        so it is disabled and reported rather than blocking a working setup.
        """
        plan = getattr(config.advanced, "speculative", None)
        if plan is None or not plan.enabled or not runtime.uses_gguf:
            return

        spec = self.catalog.get(plan.draft_model_id)
        if spec is None:
            plan.draft_model_id = None
            return

        self.console.info(
            f"fetching {spec.name} as a speculative draft model "
            f"(~{plan.expected_speedup:.2f}x on predictable output)"
        )
        try:
            from llmfit.runtimes.gguf_store import fetch_gguf

            plan.draft_path = str(fetch_gguf(spec, plan.draft_quant, self.console))
        except LlmfitError as exc:
            self.console.warn(
                f"could not fetch the draft model ({exc.message}); continuing "
                f"without speculative decoding"
            )
            plan.draft_model_id = None
            plan.draft_path = None
            plan.rationale = f"disabled: the draft model could not be fetched ({exc.message})"

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
