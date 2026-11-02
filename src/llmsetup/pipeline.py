"""Orchestrate detected hardware and goals through planning, setup, and recovery.

Pure model decisions stay in the planner and tuner. This module bounds retries,
tries runtime alternatives, and registers only verified configurations.
"""

from __future__ import annotations

import datetime
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from llmsetup.catalog.catalog import Catalog
from llmsetup.gguf import apply_to_spec, read_gguf_info
from llmsetup.interop.export import register_everywhere
from llmsetup.planner.plan import build_adaptive_plan, plan_embedding_model
from llmsetup.repair.loop import RepairLoop, mark_resolved
from llmsetup.repair.strategies import RepairState
from llmsetup.runtimes.base import Endpoint, Runtime, RuntimeUnavailable
from llmsetup.runtimes.gguf_store import CachedModelUnavailable, require_cached
from llmsetup.runtimes.registry import get_runtime
from llmsetup.tuning import tune
from llmsetup.tuning.profiles import SpecType
from llmsetup.types import (
    CheckResult,
    CheckStatus,
    HardwareProfile,
    ModelSpec,
    RepairAction,
    RuntimeConfig,
    SetupResult,
    Severity,
    Task,
    UseCaseProfile,
)
from llmsetup.util.errors import LlmsetupError, PlanningError
from llmsetup.util.log import Console
from llmsetup.verify import signatures as sig
from llmsetup.verify.client import OpenAIClient
from llmsetup.verify.harness import Harness, select_checks

#: Failures produced by the model or its settings, not by the engine serving
#: it. When one of these follows the same model onto a second engine, a third
#: engine will not fix it either.
_MODEL_LEVEL_SIGNATURES = frozenset({
    sig.EMPTY_RESPONSE, sig.INCOHERENT_OUTPUT, sig.REPETITION_LOOP,
})

_CACHE_MISS = "cached_model_missing"


class Pipeline:
    def __init__(self, profile: UseCaseProfile, hardware: HardwareProfile,
                 catalog: Catalog, console: Console, *,
                 runtime_name: Optional[str] = None, quick: bool = False,
                 dry_run: bool = False, max_repairs: int = 12,
                 register: bool = True, sweep: bool = False, cached_only: bool = False):
        self.profile = profile
        self.hardware = hardware
        self.catalog = catalog
        self.console = console
        self.runtime_name = runtime_name
        self.quick = quick
        self.dry_run = dry_run
        self.max_repairs = max_repairs
        self.register = register
        self.sweep = sweep
        self.cached_only = cached_only
        self._endpoint: Optional[Endpoint] = None
        self._runtime: Optional[Runtime] = None

    # ------------------------------------------------------------------ run

    _unprovisionable: Set[str]

    def run(self) -> SetupResult:
        """Try the selected engine, compatible installed alternatives, then a CPU fallback."""
        if self.dry_run or self.max_repairs == 0:
            return self._run_once()
        from llmsetup.runtimes.install import desired_flavor, managed_binary
        from llmsetup.runtimes.registry import choose_runtime, recovery_order

        self._unprovisionable = getattr(self, "_unprovisionable", set())
        original_hardware, original_profile = self.hardware, self.profile
        requested_runtime = self.runtime_name
        self._runtime_reason: Optional[str] = None
        # Model-level failures that follow the same model onto a second engine
        # are not engine problems; trying a third engine only costs time.
        followed: Dict[Tuple[str, str], Set[str]] = {}
        stop_switching = False
        self._remaining_repairs = self.max_repairs
        queue = [(requested_runtime, "auto")]
        tried: Set[Tuple[Optional[str], str]] = set()
        journal: List[RepairAction] = []
        errors: List[str] = []
        last_result = None
        while queue:
            name, mode = queue.pop(0)
            cpu_only = mode != "auto"
            self.shutdown()
            self._runtime = None
            self.hardware = (replace(original_hardware, gpus=[], unified_memory=False)
                             if cpu_only else original_hardware)
            self.profile = original_profile
            self.runtime_name = name
            self._portable_cpu = mode == "portable"
            if mode == "portable":
                self._runtime_reason = "chosen by automatic recovery: verified user-local llama.cpp on CPU"
            elif cpu_only:
                self._runtime_reason = "chosen by automatic recovery: CPU after the GPU attempts failed"
            elif name != requested_runtime:
                self._runtime_reason = "chosen by automatic recovery after the previous attempt failed"
            else:
                self._runtime_reason = None
            try:
                engine = name or choose_runtime(self.hardware, self.profile)[0]
                if engine == "llamacpp" and not cpu_only and original_hardware.gpus:
                    self._ensure_gpu_runtime(original_hardware)
                if cpu_only:
                    self.profile = self._cpu_profile(
                        self.profile, self.hardware, last_result)
                    # The portable Linux/Windows build is CPU-only. Replan for
                    # RAM rather than claiming to use the detected GPU.
                    if self._portable_cpu and managed_binary(self.hardware) is None:
                        self._install_cpu_runtime()
                    journal.append(RepairAction(
                        strategy="portable_cpu_runtime" if self._portable_cpu else "cpu_runtime",
                        trigger=self._last_trigger(last_result),
                        description=("using verified user-local llama.cpp on CPU" if self._portable_cpu
                                     else "replanning for CPU with installed llama.cpp"), cost="high",
                    ))
                result = self._run_once()
                last_result = result
                tried.add((result.plan.runtime, mode))
                if any(check.failure_signature == _CACHE_MISS for check in result.checks):
                    result.repairs = journal + result.repairs
                    return result
                if not any(check.blocking for check in result.checks):
                    result.hardware = original_hardware
                    result.repairs = journal + result.repairs
                    mark_resolved(result.repairs, result.checks)
                    if journal:
                        result.plan.notes.append("automatic recovery tried other runtime or CPU paths; see repairs")
                    return result
                journal.extend(result.repairs)
                errors.extend(f"{result.plan.runtime}: {check.message}\n{check.detail}"
                              for check in result.checks if check.blocking)
                for check in result.checks:
                    if not check.blocking:
                        continue
                    signature = check.failure_signature or ""
                    if signature not in _MODEL_LEVEL_SIGNATURES:
                        continue
                    model_id = result.config.model_id or (
                        result.plan.best.model.id if result.plan.best else "")
                    seen = followed.setdefault((signature, model_id), set())
                    seen.add(result.plan.runtime)
                    if len(seen) >= 2:
                        errors.append(
                            f"{signature} followed {model_id} across "
                            f"{', '.join(sorted(seen))}: a model or settings "
                            "problem, not an engine problem, so no further "
                            "engines were tried")
                        queue = []
                        stop_switching = True
                        break
            except CachedModelUnavailable:
                # Draft/retriever misses also stop recovery rather than choosing
                # another engine or a different uncached model.
                raise
            except PlanningError as exc:
                if last_result is None and not original_profile.requested_model:
                    raise
                tried.add((name, mode))
                errors.append(f"{name or 'planning'}: {exc.message}" +
                              (f". {exc.hint}" if exc.hint else ""))
            except (LlmsetupError, RuntimeUnavailable, OSError) as exc:
                actual = self._runtime.name if self._runtime else name
                tried.add((actual, mode))
                message = exc.message if isinstance(exc, LlmsetupError) else str(exc)
                hint = getattr(exc, "hint", "") or getattr(exc, "install_hint", "")
                errors.append(f"{actual or 'planning'}: {message}" + (f"\n{hint}" if hint else ""))
                self.console.info(f"automatic recovery: {message}")
            if not stop_switching and not queue and mode == "auto":
                installed = recovery_order(original_hardware, original_profile)
                alternatives = [(engine, "auto") for engine in installed]
                if original_hardware.gpus and desired_flavor(original_hardware) != "cpu":
                    alternatives.append(("llamacpp", "auto"))
                if "llamacpp" in installed and original_hardware.gpus:
                    alternatives.append(("llamacpp", "cpu"))
                # Try an existing CPU engine before fetching another executable.
                alternatives.append(("llamacpp", "portable"))
                queue = [item for item in dict.fromkeys(alternatives) if item not in tried]
            if queue:
                journal.append(RepairAction(
                    strategy="runtime_fallback", trigger=self._last_trigger(last_result),
                    description=f"trying {queue[0][0]}{' on CPU' if queue[0][1] != 'auto' else ''} after the previous attempt failed",
                    cost="high",
                ))
        detail = "; ".join(dict.fromkeys(errors))
        hint = "Automatic recovery exhausted. " + detail + ". " + _recovery_instructions(detail)
        if original_profile.requested_model:
            hint += (f" Requested model {original_profile.requested_model} was kept fixed; "
                     "choose a different model explicitly if these settings cannot work.")
        self.console.warn(hint)
        if last_result is None:
            raise LlmsetupError("setup could not provision a supported runtime and model", hint=hint)
        last_result.hardware = original_hardware
        last_result.repairs = journal
        last_result.plan.notes.append(hint)
        for check in last_result.checks:
            if check.blocking:
                check.detail = hint
        return last_result

    @staticmethod
    def _last_trigger(last_result: Optional[SetupResult]) -> str:
        """The check that actually sent recovery down this path."""
        if last_result is not None:
            failure = next((c for c in last_result.checks if c.blocking), None)
            if failure is not None:
                signature = failure.failure_signature or "unknown"
                return f"{failure.name}: {signature}"
        return "health: runtime_unavailable"

    def _cpu_profile(self, profile: UseCaseProfile, hardware: HardwareProfile,
                     last_result: Optional[SetupResult]) -> UseCaseProfile:
        """Keep the model already fetched for the CPU attempt when it fits RAM.

        The CPU fallback replans against system memory, and a larger budget
        picks a larger model: after a small model fails on the GPU path, the
        fallback would fetch a much bigger one for a slower engine. Keeping the
        model that was already downloaded and tried gets to a working setup
        faster and leaves disk untouched. Falls back to a free replan when the
        earlier model cannot fit in RAM.
        """
        pinned = None
        if (last_result is not None and last_result.checks
                and last_result.plan is not None and last_result.plan.best is not None):
            # Verification ran, so this model's weights are already local.
            pinned = last_result.plan.best.model.id
        if pinned is not None:
            try:
                build_adaptive_plan(replace(profile, requested_model=pinned),
                                    hardware, self.catalog, runtime="llamacpp")
            except PlanningError:
                pass
            else:
                self.console.info(
                    f"keeping {pinned} for the CPU attempt instead of fetching "
                    "a larger model for a slower engine")
                return replace(profile, requested_model=pinned)
        build_adaptive_plan(profile, hardware, self.catalog, runtime="llamacpp")
        return profile

    def _install_cpu_runtime(self) -> None:
        from llmsetup.runtimes.install import install_llamacpp

        install_llamacpp(self.hardware, self.console, flavor="cpu")

    def _ensure_gpu_runtime(self, hardware: HardwareProfile) -> None:
        """Give the llama.cpp attempt a GPU-capable managed build.

        Without a system llama-server, the managed build is what serves the
        attempt. A CPU build forces the planner to ignore the detected GPU and
        run the largest model that fits in RAM on the slowest path, so when the
        machine has a GPU the managed build must be able to use it. No-op when
        a system llama-server exists or the right build is already installed.
        """
        from llmsetup.runtimes.install import (
            desired_flavor,
            install_llamacpp,
            managed_flavor,
        )
        from llmsetup.util.proc import which

        desired = desired_flavor(hardware)
        if desired == "cpu" or which("llama-server"):
            return
        if managed_flavor(hardware) == desired:
            return
        self.console.info(
            "fetching the GPU-capable llama.cpp build so the GPU is actually used")
        install_llamacpp(hardware, self.console, flavor=desired)

    def _run_once(self) -> SetupResult:
        started = _now()

        self.console.step("Planning")
        plan = build_adaptive_plan(self.profile, self.hardware, self.catalog,
                          runtime=self.runtime_name,
                          runtime_reason=getattr(self, "_runtime_reason", None),
                          exclude_models=getattr(self, "_unprovisionable", None))
        assert plan.profile is not None
        self.profile = plan.profile
        self.hardware = plan.hardware or self.hardware
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
        if getattr(self, "_portable_cpu", False):
            config.extra["portable_runtime"] = True
        for line in tuned.explain():
            self.console.info(self.console.dim(line))

        result = SetupResult(
            profile=self.profile, hardware=self.hardware, plan=plan,
            config=config, started_at=started,
        )

        from llmsetup.tuning.platform_tuning import plan_platform

        result.platform_actions = [
            action.to_dict() for action in plan_platform(
                self.hardware, config,
                estimate_bytes=candidate.estimate.total,
                profile=self.profile,
            )
        ]
        for action in result.platform_actions:
            if not action["optional"]:
                self.console.info(f"{action['title']}: trying runtime settings with current OS limits")
        if self.dry_run:
            self.console.info("--dry-run: stopping before download and launch")
            result.finished_at = _now()
            return result

        self.console.step("Provisioning")
        runtime = get_runtime(plan.runtime)
        self._runtime = runtime
        if not runtime.is_available():
            raise LlmsetupError(
                f"{runtime.label} is not installed",
                hint=runtime.install_hint(),
            )
        checks = self._provision(candidate.model, config)
        if not checks:
            self._ensure_draft_model(runtime, config)
            self._correct_from_gguf(candidate, config, plan)

        if not checks and self.sweep and runtime.uses_gguf:
            self.console.step("Measuring settings on this machine")
            result.sweep = self._sweep_settings(candidate, config)

        if not checks:
            self.console.step("Starting the server")
            checks = self._launch_and_check(candidate.model, config, None)

        if (any(c.blocking for c in checks)
                and not any(c.failure_signature == _CACHE_MISS for c in checks)):
            self.console.step("Repairing")
            state = RepairState(
                config=config, candidate=candidate, hardware=self.hardware,
                profile=self.profile, catalog=self.catalog,
                fallbacks=list(plan.fallbacks), console=self.console,
            )
            loop = RepairLoop(state=state, relaunch=self._relaunch,
                              console=self.console, max_repairs=getattr(self, "_remaining_repairs", self.max_repairs))
            checks, journal = loop.run(checks)
            mark_resolved(journal, checks)
            result.repairs = journal
            self._remaining_repairs = max(0, getattr(self, "_remaining_repairs", self.max_repairs) - len(journal))
            # The loop may have swapped model or settings.
            config = state.config
            candidate = state.candidate
            plan.candidates = [candidate] + [c for c in plan.candidates
                                             if c.label != candidate.label]

        self._recompute_estimate(candidate, config)
        result.config = config
        result.checks = checks

        if self.register and result.succeeded:
            self.console.step("Registering with your other apps")
            gguf_path = runtime.local_model_path(candidate.model, config)
            result.registrations = register_everywhere(
                candidate.model, config, self._endpoint, self.console,
                gguf_path=gguf_path, requested=self.profile.clients,
            )

        # A RAG setup is only half-built without a retriever, so this runs after
        # the chat model is verified rather than being left as advice.
        if result.succeeded and (self.profile.needs_embeddings or Task.RAG in self.profile.tasks):
            self.console.step("Setting up the embedding model for retrieval")
            result.embedding = self._setup_embedding(runtime)

        result.finished_at = _now()
        return result

    # ----------------------------------------------------------------- sweep

    def _sweep_settings(self, candidate, config: RuntimeConfig) -> Dict[str, Any]:
        """Try candidate settings on the real hardware and keep what wins.

        Every other number in the plan is predicted. This one is measured, which
        is the only way to find that a particular card processes prompts fastest
        at a micro-batch four times the default. Nothing is applied unless the
        difference is larger than the run-to-run spread.
        """
        from llmsetup.bench import apply_sweep, run_sweep

        bpw = (candidate.estimate.weights * 8) / (candidate.model.params_b * 1e9)
        outcome = run_sweep(
            candidate.model, config, self.profile,
            budget_bytes=self.hardware.memory_budget, bpw=bpw,
            console=self.console,
        )
        if apply_sweep(config, outcome):
            self.console.ok(outcome.rationale)
            # The compute buffers scale with the micro-batch, so the memory
            # figures the plan reports were computed for a batch size that is no
            # longer the one being used.
            self._recompute_estimate(candidate, config)
        else:
            self.console.info(outcome.rationale)
        return outcome.to_dict()

    def _recompute_estimate(self, candidate, config: RuntimeConfig) -> None:
        """Bring the memory figures back in step with the settings in force."""
        from llmsetup.planner.memory import estimate_memory

        candidate.estimate = estimate_memory(
            candidate.model, self.catalog.bpw(config.quant), config.context_tokens,
            quant=config.quant, kv_quant=config.kv_cache_type,
            runtime=config.runtime, ubatch=config.ubatch_size,
            parallel=config.parallel, swa_full=config.swa_full,
        )

    # ------------------------------------------------------------- embedding

    def _setup_embedding(self, runtime: Runtime) -> Dict[str, Any]:
        attempts = 2 if self.max_repairs else 1
        for attempt in range(attempts):
            record = self._setup_embedding_once(runtime, conservative=bool(attempt))
            if record.get("ok"):
                return record
            if attempt + 1 < attempts:
                self.console.info("retriever setup failed; retrying with shorter context on CPU")
        record["reason"] = record.get("reason") or "embedding verification failed"
        record["hint"] = "See the retriever checks and server log in `llmsetup report`; chat verification is separate."
        return record

    def _setup_embedding_once(self, runtime: Runtime, *, conservative: bool = False) -> Dict[str, Any]:
        """Provision and verify the companion retriever.

        Deliberately non-fatal: a working chat model with a broken retriever is
        still a useful outcome, so this reports its own status instead of
        failing the run. But it is *verified*, an embedding endpoint that
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
            context_tokens=min(spec.max_context, 2048 if conservative else 8192),
            total_layers=spec.layers,
            gpu_layers=spec.layers if self.hardware.gpus and not conservative else 0,
            batch_size=512,
            ubatch_size=512,
            port=self._free_port(),
            served_name=spec.id,
            extra={"llamacpp_flags": ["--embeddings", "--pooling", "mean"],
                   "portable_runtime": getattr(self, "_portable_cpu", False)},
        )

        failures = self._check_cached(spec, config, runtime)
        if failures:
            raise CachedModelUnavailable(failures[0].message)
        try:
            config = runtime.ensure_model(spec, config, self.console)
            endpoint = runtime.start(spec, config, self.console)
        except CachedModelUnavailable:
            raise
        except (RuntimeUnavailable, LlmsetupError, OSError) as exc:
            message = exc.message if isinstance(exc, LlmsetupError) else str(exc)
            self.console.info(f"could not start the embedding model: {message}")
            return {**record, "ok": False, "reason": message}

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
        from llmsetup.util import http

        return http.find_free_port("127.0.0.1", 11500)

    # ---------------------------------------------------------- provisioning

    def _ensure_draft_model(self, runtime: Runtime, config: RuntimeConfig) -> None:
        """Fetch the speculative draft model, if one was planned.

        Failing to get the draft is not fatal: speculation is an optimisation,
        so it is disabled and reported rather than blocking a working setup.
        """
        plan = getattr(config.advanced, "speculative", None)
        if plan is None or not runtime.uses_gguf:
            return
        # Not every kind of drafting has a second model. Prediction heads live
        # in the target's own file and n-gram drafting reads the prompt, so
        # there is nothing to fetch and nothing to fail at.
        if not plan.needs_draft_model:
            return

        spec = self.catalog.get(plan.draft_model_id) if plan.draft_model_id else None
        if spec is None:
            plan.spec_type = SpecType.NONE
            plan.draft_model_id = None
            return

        if self.cached_only:
            plan.draft_path = str(require_cached(spec, plan.draft_quant))
            return

        self.console.info(
            f"fetching {spec.name} as a speculative draft model "
            f"(~{plan.expected_speedup:.2f}x on predictable output)"
        )
        try:
            from llmsetup.runtimes.gguf_store import fetch_gguf

            plan.draft_path = str(fetch_gguf(spec, plan.draft_quant, self.console))
        except LlmsetupError as exc:
            self.console.warn(
                f"could not fetch the draft model ({exc.message}); continuing "
                f"without speculative decoding"
            )
            plan.spec_type = SpecType.NONE
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
        from llmsetup.planner.memory import estimate_memory

        bpw = self.catalog.bpw(config.quant)
        candidate.estimate = estimate_memory(
            corrected, bpw, config.context_tokens, quant=config.quant,
            kv_quant=config.kv_cache_type, runtime=config.runtime,
            ubatch=config.ubatch_size, parallel=config.parallel,
            swa_full=config.swa_full,
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
        failures = self._check_cached(spec, config, self._runtime)
        if failures:
            return failures
        self._stop_endpoint()

        try:
            endpoint = self._runtime.start(spec, config, self.console)
        except CachedModelUnavailable as exc:
            return self._cached_failure(exc)
        except (LlmsetupError, RuntimeUnavailable, OSError) as exc:
            # A failed launch is a check failure, so repair can act on it rather
            # than the whole run aborting.
            return [CheckResult(
                name="health", status=CheckStatus.FAIL, severity=Severity.CRITICAL,
                message=str(exc),
                failure_signature=sig.classify(str(exc)) or sig.SERVER_UNREACHABLE,
                detail=getattr(exc, "hint", "") or getattr(exc, "install_hint", "") or "",
            )]
        self._endpoint = endpoint

        harness = Harness(
            client=OpenAIClient(endpoint), spec=spec, config=config,
            profile=self.profile, hardware=self.hardware, console=self.console,
            server_log=endpoint.log_path,
        )
        # RAG's embedding capability is proved on the separate retriever, not
        # on the chat model, which generally has no embedding endpoint.
        check_profile = replace(self.profile, needs_embeddings=False) if not spec.is_embedding else self.profile
        chosen = list(names) if names else select_checks(check_profile, spec, self.quick)
        return harness.run(chosen)

    def _relaunch(self, config: RuntimeConfig,
                  names: Optional[Sequence[str]]) -> List[CheckResult]:
        """Callback handed to the repair loop."""
        assert self._runtime is not None
        spec = self._current_spec(config)
        failures = self._check_cached(spec, config, self._runtime)
        if failures:
            return failures
        if (not config.model_path or
                (self._runtime.uses_gguf and not Path(config.model_path).exists()) or
                (config.model_id, config.quant) != getattr(self, "_provisioned", None)):
            failures = self._provision(spec, config)
            if failures:
                return failures
        return self._launch_and_check(spec, config, names)

    def _provision(self, spec: ModelSpec, config: RuntimeConfig) -> List[CheckResult]:
        assert self._runtime is not None
        attempts = 2 if self.max_repairs else 1
        for attempt in range(attempts):
            failures = self._check_cached(spec, config, self._runtime)
            if failures:
                return failures
            try:
                updated = self._runtime.ensure_model(spec, config, self.console)
                if updated is not config:
                    config.__dict__.update(updated.__dict__)
                self._provisioned = (config.model_id, config.quant)
                return []
            except CachedModelUnavailable as exc:
                return self._cached_failure(exc)
            except (LlmsetupError, RuntimeUnavailable, OSError) as exc:
                message = exc.message if isinstance(exc, LlmsetupError) else str(exc)
                if attempt + 1 < attempts:
                    self.console.info(f"provisioning failed ({message}); retrying once")
                    continue
                if "has no" in message and "build" in message:
                    # The catalog entry points at files that do not exist.
                    # That is permanent for this run and engine-independent,
                    # so keep later attempts from picking the same model again.
                    self._unprovisionable = getattr(self, "_unprovisionable", set())
                    self._unprovisionable.add(spec.id)
                return [CheckResult(
                    name="health", status=CheckStatus.FAIL, severity=Severity.CRITICAL,
                    message=message, failure_signature=(sig.DOWNLOAD_UNAVAILABLE
                        if sig.classify(message) == sig.DOWNLOAD_UNAVAILABLE else sig.PROVISION_FAILED),
                    detail=getattr(exc, "hint", "") or getattr(exc, "install_hint", "") or "",
                )]
        return []

    def _check_cached(self, spec: ModelSpec, config: RuntimeConfig,
                      runtime: Runtime) -> List[CheckResult]:
        if not self.cached_only:
            return []
        config.extra["cached_only"] = True
        try:
            if runtime.name not in ("ollama", "llamacpp"):
                # Other adapters can fetch on launch through their own caches.
                # Without proof of exact identity and a local-only launch, stop.
                raise CachedModelUnavailable(
                    f"--cached-only: cannot verify retained weights for {spec.id}:{config.quant} "
                    f"on {runtime.name}; use Ollama or llama.cpp with the shared GGUF store"
                )
            config.model_path = str(require_cached(spec, config.quant))
            if runtime.name == "llamacpp" and spec.supports_vision:
                config.mmproj_path = str(require_cached(spec, config.quant, projector=True))
        except CachedModelUnavailable as exc:
            return self._cached_failure(exc)
        return []

    @staticmethod
    def _cached_failure(exc: CachedModelUnavailable) -> List[CheckResult]:
        return [CheckResult(
            name="model_cache", status=CheckStatus.FAIL, severity=Severity.CRITICAL,
            message=exc.message, failure_signature=_CACHE_MISS, detail=exc.hint or "",
        )]

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


def _recovery_instructions(detail: str) -> str:
    """Give the relevant next step only after automatic alternatives are exhausted."""
    text = detail.lower()
    if any(word in text for word in ("network", "blocked", "could not reach", "urlopen", "name resolution", "http 403")):
        return ("Allow HTTPS to huggingface.co and github.com, including their download hosts, "
                "then rerun `llmsetup setup`. For a blocked Hub, set HF_ENDPOINT to a reachable trusted mirror.")
    if "gated" in text or "licence" in text or "http 401" in text:
        return "Accept the model licence on Hugging Face, set HF_TOKEN with read access, then rerun `llmsetup setup`."
    if "space" in text or "disk" in text:
        return "Free space on the model-store volume, or set LLMSETUP_MODEL_STORE to a volume with room, then rerun `llmsetup setup`."
    if "memory" in text or "allocate" in text:
        return "Close memory-heavy apps and rerun `llmsetup setup`. If no catalog model fits, use a machine with more available RAM."
    if "portable" in text or "install" in text or "not found" in text:
        return ("Install llama-server for this OS and CPU from https://github.com/ggml-org/llama.cpp, "
                "make it available on PATH, then rerun `llmsetup setup`.")
    return "Open `llmsetup report`, inspect the named failed check and its server log, then run `llmsetup doctor` after correcting that failure."
