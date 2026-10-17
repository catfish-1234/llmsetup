"""Repair loop: does it escalate correctly, converge, and terminate?"""

from __future__ import annotations

import pytest
from conftest import make_hardware

from llmsetup.planner.memory import estimate_memory
from llmsetup.planner.plan import build_plan
from llmsetup.repair.loop import RepairLoop, mark_resolved
from llmsetup.repair.strategies import COST_ORDER, STRATEGIES, RepairState, strategies_for
from llmsetup.tuning.tuner import tune
from llmsetup.types import (
    GIB,
    CheckResult,
    CheckStatus,
    Severity,
    Task,
    UseCaseProfile,
)
from llmsetup.util.log import Console
from llmsetup.verify import signatures as sig


def build_state(catalog, hardware, profile, runtime="llamacpp"):
    plan = build_plan(profile, hardware, catalog, runtime=runtime)
    tuned = tune(plan.best, hardware, profile, runtime)
    return RepairState(config=tuned.config, candidate=plan.best, hardware=hardware,
                       profile=profile, catalog=catalog,
                       fallbacks=list(plan.fallbacks))


def fail(signature, name="health", severity=Severity.CRITICAL):
    return [CheckResult(name=name, status=CheckStatus.FAIL, severity=severity,
                        message="simulated", failure_signature=signature)]


def ok():
    return [CheckResult(name="health", status=CheckStatus.PASS,
                        severity=Severity.CRITICAL, message="ok")]


class TestStrategyTable:
    def test_every_strategy_has_at_least_one_trigger(self):
        for strategy in STRATEGIES:
            assert strategy.triggers, strategy.name

    def test_every_strategy_declares_a_known_cost(self):
        for strategy in STRATEGIES:
            assert strategy.cost in COST_ORDER, strategy.name

    def test_strategy_names_are_unique(self):
        names = [s.name for s in STRATEGIES]
        assert len(names) == len(set(names))

    def test_lookup_returns_cheapest_first(self):
        for signature in (sig.OUT_OF_MEMORY, sig.CONTEXT_EXCEEDED,
                          sig.TOOLS_UNSUPPORTED):
            costs = [COST_ORDER[s.cost] for s in strategies_for(signature)]
            assert costs == sorted(costs), signature

    def test_every_memory_failure_can_escalate_to_a_model_swap(self):
        """The last resort must always exist, or a hard limit has no answer."""
        for signature in (sig.OUT_OF_MEMORY, sig.CRASH_DURING_LOAD, sig.KV_CACHE_FULL):
            assert any(s.name == "switch_model" for s in strategies_for(signature))

    def test_every_signature_the_checks_can_emit_has_a_strategy(self):
        """A signature with no strategy is a dead end the user cannot escape."""
        emitted = {
            sig.OUT_OF_MEMORY, sig.CONTEXT_EXCEEDED, sig.KV_CACHE_FULL,
            sig.SERVER_UNREACHABLE, sig.MODEL_NOT_LOADED, sig.CRASH_DURING_LOAD,
            sig.TOOLS_UNSUPPORTED, sig.TOOL_CALL_MALFORMED,
            sig.EMBEDDINGS_UNSUPPORTED, sig.REPETITION_LOOP,
            sig.NEEDLE_NOT_FOUND, sig.THROUGHPUT_TOO_LOW, sig.INCOHERENT_OUTPUT,
            sig.EMPTY_RESPONSE, sig.VISION_UNSUPPORTED,
        }
        missing = {s for s in emitted if not strategies_for(s)}
        assert not missing, f"no repair path for: {missing}"


class TestEscalation:
    def test_converges_on_a_constrained_machine(self, catalog):
        """The scenario this exists for: reality has less memory than we
        estimated, and the loop must land somewhere that works."""
        hardware = make_hardware(vram_gb=12, ram_gb=32)
        profile = UseCaseProfile(tasks=[Task.CODING], context_tokens=32768)
        state = build_state(catalog, hardware, profile)
        limit = 6.0 * GIB

        def relaunch(config, names):
            spec = state.candidate.model
            bpw = (state.candidate.estimate.weights * 8) / (spec.params_b * 1e9)
            estimate = estimate_memory(spec, bpw, config.context_tokens,
                                       kv_quant=config.kv_cache_type,
                                       runtime="llamacpp", ubatch=config.ubatch_size,
                                       parallel=config.parallel)
            resident = (estimate.weights * (config.gpu_layers / max(1, config.total_layers))
                        + estimate.kv_cache + estimate.overhead)
            if config.flash_attention:
                resident *= 0.93
            return ok() if resident <= limit else fail(sig.OUT_OF_MEMORY)

        loop = RepairLoop(state=state, relaunch=relaunch, console=Console(quiet=True))
        results, journal = loop.run(relaunch(state.config, None))

        assert not any(r.blocking for r in results), "never converged"
        assert journal, "converged without recording what it changed"

    def test_escalates_from_cheap_to_expensive(self, catalog):
        state = build_state(catalog, make_hardware(vram_gb=12, ram_gb=32),
                            UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768))
        loop = RepairLoop(state=state,
                          relaunch=lambda c, n: fail(sig.OUT_OF_MEMORY),
                          console=Console(quiet=True))
        _, journal = loop.run(fail(sig.OUT_OF_MEMORY))
        costs = [COST_ORDER[a.cost] for a in journal]
        assert costs == sorted(costs), [a.strategy for a in journal]

    def test_terminates_when_nothing_ever_works(self, catalog):
        """A machine that fails no matter what must stop, not spin."""
        state = build_state(catalog, make_hardware(vram_gb=12, ram_gb=32),
                            UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768))
        calls = {"n": 0}

        def relaunch(config, names):
            calls["n"] += 1
            return fail(sig.OUT_OF_MEMORY)

        loop = RepairLoop(state=state, relaunch=relaunch, console=Console(quiet=True),
                          max_repairs=20)
        results, _ = loop.run(fail(sig.OUT_OF_MEMORY))
        assert any(r.blocking for r in results)
        assert calls["n"] <= 20

    def test_never_repeats_a_strategy_for_the_same_signature(self, catalog):
        state = build_state(catalog, make_hardware(vram_gb=12, ram_gb=32),
                            UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768))
        loop = RepairLoop(state=state, relaunch=lambda c, n: fail(sig.OUT_OF_MEMORY),
                          console=Console(quiet=True), max_repairs=20)
        _, journal = loop.run(fail(sig.OUT_OF_MEMORY))
        names = [a.strategy for a in journal]
        assert len(names) == len(set(names))

    def test_a_passing_setup_is_left_alone(self, catalog):
        state = build_state(catalog, make_hardware(), UseCaseProfile(tasks=[Task.CHAT]))
        before = state.config.to_dict()
        loop = RepairLoop(state=state, relaunch=lambda c, n: ok(),
                          console=Console(quiet=True))
        _, journal = loop.run(ok())
        assert journal == []
        assert state.config.to_dict() == before


class TestSpecificRemediations:
    def test_needle_failure_restores_kv_precision(self, catalog):
        """Aggressive KV quantization is the usual cause of lost long-context
        recall, so the fix is to trade context back for precision."""
        hardware = make_hardware(vram_gb=12, ram_gb=32)
        profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=131072)
        state = build_state(catalog, hardware, profile)
        if state.config.kv_cache_type == "f16":
            pytest.skip("this configuration did not need KV quantization")

        before_kv = state.config.kv_cache_type
        loop = RepairLoop(state=state, relaunch=lambda c, n: ok(),
                          console=Console(quiet=True))
        loop.run(fail(sig.NEEDLE_NOT_FOUND, name="context", severity=Severity.REQUIRED))
        assert state.config.kv_cache_type != before_kv

    def test_repetition_is_fixed_by_loosening_sampling(self, catalog):
        state = build_state(catalog, make_hardware(),
                            UseCaseProfile(tasks=[Task.EXTRACT]))
        state.config.temperature = 0.0
        loop = RepairLoop(state=state, relaunch=lambda c, n: ok(),
                          console=Console(quiet=True))
        loop.run(fail(sig.REPETITION_LOOP, name="repetition", severity=Severity.CRITICAL))
        assert state.config.temperature > 0.0

    def test_low_throughput_pushes_layers_back_onto_the_gpu(self, catalog):
        state = build_state(catalog, make_hardware(),
                            UseCaseProfile(tasks=[Task.CHAT]))
        state.config.gpu_layers = 4
        loop = RepairLoop(state=state, relaunch=lambda c, n: ok(),
                          console=Console(quiet=True))
        loop.run(fail(sig.THROUGHPUT_TOO_LOW, name="throughput",
                      severity=Severity.CRITICAL))
        assert state.config.gpu_layers > 4

    def test_model_swap_retunes_from_scratch(self, catalog):
        """Carrying the old model's layer count across would be wrong."""
        state = build_state(catalog, make_hardware(vram_gb=24),
                            UseCaseProfile(tasks=[Task.CHAT]))
        original = state.candidate.model.id
        loop = RepairLoop(state=state, relaunch=lambda c, n: ok(),
                          console=Console(quiet=True))
        # Exhaust the cheaper strategies so the swap is reached.
        for strategy in strategies_for(sig.OUT_OF_MEMORY):
            if strategy.name != "switch_model":
                loop._tried.add((sig.OUT_OF_MEMORY, strategy.name))
        loop.run(fail(sig.OUT_OF_MEMORY))
        assert state.candidate.model.id != original
        assert state.config.model_id == state.candidate.model.id
        assert state.config.total_layers == state.candidate.model.layers


class TestJournal:
    def test_records_trigger_cost_and_change(self, catalog):
        state = build_state(catalog, make_hardware(vram_gb=12, ram_gb=32),
                            UseCaseProfile(tasks=[Task.CHAT], context_tokens=32768))
        loop = RepairLoop(state=state, relaunch=lambda c, n: fail(sig.OUT_OF_MEMORY),
                          console=Console(quiet=True), max_repairs=3)
        _, journal = loop.run(fail(sig.OUT_OF_MEMORY))
        for action in journal:
            assert action.strategy and action.trigger and action.description
            assert action.cost in COST_ORDER

    def test_resolution_is_marked_against_final_results(self, catalog):
        state = build_state(catalog, make_hardware(), UseCaseProfile(tasks=[Task.CHAT]))
        attempts = {"n": 0}

        def relaunch(config, names):
            attempts["n"] += 1
            return ok()

        loop = RepairLoop(state=state, relaunch=relaunch, console=Console(quiet=True))
        results, journal = loop.run(fail(sig.OUT_OF_MEMORY))
        mark_resolved(journal, results)
        assert all(a.resolved is True for a in journal)
