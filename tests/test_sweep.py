"""Measuring settings on the machine instead of predicting them.

The rest of llmsetup computes. This part runs the thing and looks, which is what
an experienced person does and what a formula cannot: no amount of arithmetic
recovers the fact that a particular card, on a particular driver, processes
prompts fastest at four times the default micro-batch.

The risk in measuring is claiming more than the measurement supports, so most
of what is pinned here is restraint: not benchmarking settings that would not
fit, not calling a difference an improvement when it is inside the run-to-run
spread, and not optimising a number the use case does not care about.
"""

from __future__ import annotations

import json

import pytest

from llmsetup.bench import (
    BenchRow,
    SweepCandidate,
    apply_sweep,
    candidate_grid,
    parse_bench_json,
    pick_best,
    prefill_weight,
    score_rows,
)
from llmsetup.bench.sweep import bench_command, estimate_duration_s, run_sweep
from llmsetup.types import GIB, ModelSpec, RuntimeConfig, Task, UseCaseProfile


def spec(**kwargs) -> ModelSpec:
    base = {"id": "m", "name": "M", "publisher": "p", "params_b": 8.0,
            "layers": 32, "hidden_size": 4096, "kv_heads": 8, "head_dim": 128,
            "max_context": 131072}
    base.update(kwargs)
    return ModelSpec(**base)


def config(**kwargs) -> RuntimeConfig:
    base = {"runtime": "llamacpp", "model_id": "m", "quant": "Q4_K_M",
            "context_tokens": 8192, "gpu_layers": 32, "total_layers": 32,
            "batch_size": 2048, "ubatch_size": 512}
    base.update(kwargs)
    return RuntimeConfig(**base)


def rows(*specs) -> list:
    """(batch, ubatch, prefill_ts, decode_ts[, stddev]) -> measurement rows."""
    out = []
    for entry in specs:
        batch, ubatch, prefill, decode = entry[:4]
        stddev = entry[4] if len(entry) > 4 else 0.0
        out.append(BenchRow(n_batch=batch, n_ubatch=ubatch, n_prompt=2048,
                            n_gen=0, avg_ts=prefill, stddev_ts=stddev * prefill))
        out.append(BenchRow(n_batch=batch, n_ubatch=ubatch, n_prompt=0,
                            n_gen=128, avg_ts=decode, stddev_ts=stddev * decode))
    return out


class TestWhatGetsMeasured:
    def test_the_planned_settings_are_always_included(self):
        """Without them there is nothing to compare against, and a sweep that
        cannot compare cannot say it found anything."""
        grid = candidate_grid(spec(), config(), int(24 * GIB), 4.85)
        assert SweepCandidate(2048, 512) in grid

    def test_it_reaches_the_large_micro_batches(self):
        """That is where the measured gains are. A grid that stops at the
        default has nothing to find."""
        grid = candidate_grid(spec(), config(), int(24 * GIB), 4.85)
        assert max(c.ubatch_size for c in grid) >= 2048

    def test_nothing_is_benchmarked_that_would_not_fit(self):
        """A larger micro-batch buys throughput with compute-buffer memory, and
        finding that out by running out of memory mid-sweep is a poor way to
        learn it."""
        from llmsetup.planner.memory import estimate_memory

        cfg = config(context_tokens=32768)
        budget = int(10 * GIB)
        for candidate in candidate_grid(spec(), cfg, budget, 4.85):
            if candidate == SweepCandidate(cfg.batch_size, cfg.ubatch_size):
                continue  # the plan's own choice is always offered
            estimate = estimate_memory(spec(), 4.85, cfg.context_tokens,
                                       ubatch=candidate.ubatch_size)
            assert estimate.total <= budget

    def test_the_batch_is_never_smaller_than_the_micro_batch(self):
        """That is not a configuration, it is a value the engine would clamp."""
        for candidate in candidate_grid(spec(), config(), int(24 * GIB), 4.85):
            assert candidate.batch_size >= candidate.ubatch_size

    def test_a_tight_machine_gets_a_short_sweep(self):
        """With only one configuration that fits there is nothing to measure,
        and saying so beats spending five minutes proving it."""
        grid = candidate_grid(spec(params_b=70.0), config(context_tokens=65536),
                              int(6 * GIB), 4.85)
        assert len(grid) == 1

    def test_the_sweep_stays_within_its_budget(self):
        grid = candidate_grid(spec(), config(), int(80 * GIB), 4.85,
                              max_candidates=3)
        assert len(grid) <= 3
        assert SweepCandidate(2048, 512) in grid, "the baseline survives trimming"

    def test_trimming_keeps_both_ends_of_the_range(self):
        """The extremes say whether the curve rises or falls. Trimming from the
        end would drop the candidate most likely to win."""
        grid = candidate_grid(spec(), config(), int(80 * GIB), 4.85,
                              max_candidates=3)
        assert max(c.ubatch_size for c in grid) >= 2048
        assert min(c.ubatch_size for c in grid) <= 512

    def test_the_estimate_is_not_wildly_optimistic(self):
        """Someone told one minute and made to wait five trusts the next
        estimate less."""
        grid = candidate_grid(spec(), config(), int(24 * GIB), 4.85)
        assert estimate_duration_s(grid) >= 30 * len(grid)


class TestWhatCountsAsBetter:
    def test_prefill_and_decode_are_weighed_by_the_use_case(self):
        """Which of the two matters is not a property of the hardware. Someone
        completing code as they type and someone holding a conversation want
        opposite ends of the same curve."""
        assert prefill_weight(UseCaseProfile(tasks=[Task.AUTOCOMPLETE])) > 0.7
        assert prefill_weight(UseCaseProfile(tasks=[Task.CREATIVE])) < 0.3

    def test_a_prefill_heavy_use_case_picks_the_prefill_winner(self):
        measured = rows((2048, 512, 1000.0, 50.0), (2048, 2048, 1400.0, 48.0))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.8)
        assert result.best == SweepCandidate(2048, 2048)
        assert result.worth_applying

    def test_a_decode_heavy_use_case_picks_the_decode_winner(self):
        """The same measurements, a different question, a different answer.
        This is the part a single 'tokens per second' number cannot express."""
        measured = rows((2048, 512, 1000.0, 50.0), (2048, 2048, 1400.0, 48.0))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.1)
        assert result.best == SweepCandidate(2048, 512)

    def test_the_two_phases_are_normalised_before_weighing(self):
        """They are quoted in the same unit and differ by an order of magnitude,
        so a raw sum would let prompt processing decide everything regardless of
        the weight."""
        scores = score_rows(rows((1, 1, 1000.0, 10.0), (2, 2, 500.0, 20.0)), 0.5)
        assert scores[(1, 1)] == pytest.approx(scores[(2, 2)])

    def test_scores_work_when_only_one_phase_was_measured(self):
        only_decode = [BenchRow(n_batch=1, n_ubatch=1, n_prompt=0, n_gen=128,
                                avg_ts=40.0)]
        assert score_rows(only_decode, 0.9)[(1, 1)] > 0


class TestRestraint:
    def test_a_win_inside_the_noise_is_not_a_win(self):
        """llama-bench reports a spread with every number. A difference smaller
        than it is measurement error wearing a result's clothes."""
        measured = rows((2048, 512, 1000.0, 50.0, 0.10),
                        (2048, 2048, 1020.0, 50.0, 0.10))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.8)
        assert result.within_noise
        assert not result.worth_applying
        assert result.best == SweepCandidate(2048, 512)

    def test_a_win_larger_than_the_noise_is_taken(self):
        measured = rows((2048, 512, 1000.0, 50.0, 0.02),
                        (2048, 2048, 1300.0, 50.0, 0.02))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.8)
        assert not result.within_noise
        assert result.worth_applying

    def test_the_planned_settings_winning_is_a_real_outcome(self):
        measured = rows((2048, 512, 1400.0, 50.0), (2048, 2048, 1000.0, 48.0))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.8)
        assert result.best == SweepCandidate(2048, 512)
        assert not result.worth_applying
        assert "measured fastest" in result.rationale

    def test_no_measurements_means_no_claim(self):
        result = pick_best([], SweepCandidate(2048, 512), weight=0.5)
        assert not result.worth_applying
        assert result.best is None

    def test_an_unmeasured_baseline_yields_no_comparison(self):
        """If the planned settings were never run there is nothing to be better
        than, and picking the fastest of the rest would be a claim about a
        comparison that never happened."""
        measured = rows((4096, 4096, 2000.0, 60.0))
        result = pick_best(measured, SweepCandidate(2048, 512), weight=0.5)
        assert not result.worth_applying
        assert "no comparison" in result.rationale

    def test_applying_only_happens_for_a_real_win(self):
        cfg = config()
        result = pick_best(rows((2048, 512, 1000.0, 50.0, 0.10),
                                (2048, 2048, 1020.0, 50.0, 0.10)),
                           SweepCandidate(2048, 512), weight=0.8)
        assert apply_sweep(cfg, result) is False
        assert cfg.ubatch_size == 512

    def test_applying_a_real_win_changes_the_config(self):
        cfg = config()
        result = pick_best(rows((2048, 512, 1000.0, 50.0, 0.02),
                                (2048, 2048, 1300.0, 50.0, 0.02)),
                           SweepCandidate(2048, 512), weight=0.8)
        assert apply_sweep(cfg, result) is True
        assert cfg.ubatch_size == 2048
        assert cfg.batch_size >= cfg.ubatch_size


class TestReadingTheTool:
    def test_parses_what_llama_bench_writes(self):
        payload = json.dumps([
            {"n_batch": 2048, "n_ubatch": 512, "n_prompt": 2048, "n_gen": 0,
             "avg_ts": 1000.5, "stddev_ts": 12.0},
            {"n_batch": 2048, "n_ubatch": 512, "n_prompt": 0, "n_gen": 128,
             "avg_ts": 48.2, "stddev_ts": 0.4},
        ])
        parsed = parse_bench_json(payload)
        assert len(parsed) == 2
        assert parsed[0].is_prefill and not parsed[1].is_prefill

    def test_a_row_missing_a_field_is_skipped_not_fatal(self):
        """A partial measurement is still worth something, and this tool's
        output has changed shape before."""
        payload = json.dumps([
            {"n_batch": 1, "n_ubatch": 1, "avg_ts": 10.0},
            {"n_batch": 2, "nonsense": True},
        ])
        assert len(parse_bench_json(payload)) == 1

    def test_output_that_is_not_json_yields_nothing(self):
        assert parse_bench_json("error: unknown argument") == []
        assert parse_bench_json("") == []

    def test_the_command_asks_for_every_candidate_at_once(self):
        """One process rather than one per candidate: the model loads once, and
        on a large model that is most of the wall clock."""
        candidates = [SweepCandidate(2048, 512), SweepCandidate(2048, 2048)]
        command = bench_command("llama-bench", "/m.gguf", candidates, config())
        assert command[command.index("-ub") + 1] == "512,2048"
        assert command[command.index("-o") + 1] == "json"

    def test_the_command_carries_the_settings_being_kept_fixed(self):
        """Measuring at a different cache type or layer count than the plan uses
        would answer a question nobody asked."""
        cfg = config(kv_cache_type="q8_0", threads=8, gpu_layers=20)
        command = bench_command("llama-bench", "/m.gguf",
                                [SweepCandidate(2048, 512)], cfg)
        assert command[command.index("-ctk") + 1] == "q8_0"
        assert command[command.index("-ngl") + 1] == "20"
        assert command[command.index("-t") + 1] == "8"


class TestRunning:
    class _Console:
        def info(self, *_a, **_k):
            pass

        ok = warn = debug = info

    def test_a_missing_benchmark_binary_is_reported_not_crashed(self, monkeypatch):
        from llmsetup.util import proc

        monkeypatch.setattr(proc, "which", lambda _b: None)
        result = run_sweep(spec(), config(model_path="/m.gguf"),
                           UseCaseProfile(tasks=[Task.CHAT]), int(24 * GIB),
                           4.85, self._Console())
        assert not result.worth_applying
        assert "not installed" in result.rationale

    def test_weights_not_on_disk_means_nothing_to_measure(self, monkeypatch):
        from llmsetup.util import proc

        monkeypatch.setattr(proc, "which", lambda _b: "/usr/bin/llama-bench")
        result = run_sweep(spec(), config(), UseCaseProfile(tasks=[Task.CHAT]),
                           int(24 * GIB), 4.85, self._Console())
        assert not result.worth_applying
        assert "not on disk" in result.rationale

    def test_a_failed_benchmark_leaves_the_plan_alone(self, monkeypatch):
        from llmsetup.util import proc

        class Failed:
            ok = False
            output = "error: failed to load model"

        monkeypatch.setattr(proc, "which", lambda _b: "/usr/bin/llama-bench")
        monkeypatch.setattr(proc, "run", lambda *_a, **_k: Failed())
        result = run_sweep(spec(), config(model_path="/m.gguf"),
                           UseCaseProfile(tasks=[Task.CHAT]), int(24 * GIB),
                           4.85, self._Console())
        assert not result.worth_applying
        assert result.best == SweepCandidate(2048, 512)
