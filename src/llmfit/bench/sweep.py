"""Trying settings on the real machine and keeping what wins.

Everything else in llmfit predicts. The memory model is exact arithmetic and the
throughput estimate is grounded in bandwidth, but neither can know that this
card, on this driver, with this build, processes prompts fastest at a micro-batch
of 2048 rather than the default 512. Measurements on real hardware put that gap
at 25-30% of prompt throughput, and it is not the kind of thing a formula
recovers: it comes from kernel shapes and cache sizes.

An experienced person handles this by running llama-bench a few times and
eyeballing the numbers. That works and it is also where llmfit can do better,
not because the method is clever but because a machine will sit through twelve
configurations in a fixed order and a person will try three and stop.

Three rules keep it honest:

* **Nothing is benchmarked that would not fit.** Candidates go through the same
  memory model as the plan, at the context the user actually asked for, so the
  sweep cannot recommend a setting that runs out of memory later.
* **The objective follows the use case.** Prefill and decode trade against each
  other, and which one matters is not a property of the hardware. Someone
  completing code as they type and someone holding a conversation want opposite
  ends of the same curve.
* **A win inside the noise is not a win.** llama-bench reports a standard
  deviation with every number. A candidate that beats the default by less than
  the run-to-run spread is reported as a tie, because saying otherwise would be
  inventing a result out of measurement error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llmfit.types import ModelSpec, RuntimeConfig, Task, UseCaseProfile

#: Micro-batch sizes worth trying. The default is 512, and the measured optimum
#: on some hardware is four times that, so the grid spans a range wide enough to
#: contain both without becoming a long job.
_UBATCH_CANDIDATES = (256, 512, 1024, 2048)

#: Logical batch size is a ceiling on the micro-batch, so only values at or
#: above it are meaningful pairings.
_BATCH_CANDIDATES = (1024, 2048, 4096)

#: How long one llama-bench invocation is allowed to take. A sweep that hangs is
#: worse than no sweep: the user is sitting in front of it.
BENCH_TIMEOUT_S = 900.0


@dataclass
class SweepCandidate:
    """One configuration to measure."""

    batch_size: int
    ubatch_size: int

    @property
    def label(self) -> str:
        return f"b={self.batch_size} ub={self.ubatch_size}"


@dataclass
class BenchRow:
    """One measurement out of llama-bench.

    Field names mirror the tool's own JSON so there is no translation layer to
    get subtly wrong.
    """

    n_batch: int
    n_ubatch: int
    n_prompt: int
    n_gen: int
    avg_ts: float
    stddev_ts: float = 0.0

    @property
    def is_prefill(self) -> bool:
        """Prompt-processing rows have a prompt and no generation."""
        return self.n_prompt > 0 and self.n_gen == 0


@dataclass
class SweepResult:
    """What the sweep found, and whether it is worth acting on."""

    best: Optional[SweepCandidate] = None
    baseline: Optional[SweepCandidate] = None
    best_score: float = 0.0
    baseline_score: float = 0.0
    improvement: float = 0.0
    within_noise: bool = False
    rows: List[BenchRow] = field(default_factory=list)
    rationale: str = ""

    @property
    def worth_applying(self) -> bool:
        """A real, measured improvement over what the plan already chose."""
        return (self.best is not None
                and self.baseline is not None
                and self.best != self.baseline
                and not self.within_noise)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best": vars(self.best) if self.best else None,
            "baseline": vars(self.baseline) if self.baseline else None,
            "best_score": round(self.best_score, 2),
            "baseline_score": round(self.baseline_score, 2),
            "improvement": round(self.improvement, 4),
            "within_noise": self.within_noise,
            "worth_applying": self.worth_applying,
            "rationale": self.rationale,
            "measurements": len(self.rows),
        }


# --------------------------------------------------------------------------- #
# what to measure
# --------------------------------------------------------------------------- #


def candidate_grid(
    spec: ModelSpec,
    config: RuntimeConfig,
    budget_bytes: int,
    bpw: float,
    max_candidates: int = 8,
) -> List[SweepCandidate]:
    """Configurations worth measuring on this machine, cheapest risk first.

    Only combinations that fit are included. A larger micro-batch buys prompt
    throughput with compute-buffer memory, and finding that out by running out
    of memory halfway through a sweep is a poor way to learn it, so the memory
    model rules them out first.

    The configuration the plan already chose is always present, because a sweep
    with nothing to compare against cannot tell whether it found anything.
    """
    from llmfit.planner.memory import estimate_memory

    def fits(ubatch: int) -> bool:
        estimate = estimate_memory(
            spec, bpw, config.context_tokens, quant=config.quant,
            kv_quant=config.kv_cache_type, runtime=config.runtime,
            ubatch=ubatch, parallel=config.parallel, swa_full=config.swa_full,
        )
        return estimate.total <= budget_bytes

    planned = SweepCandidate(config.batch_size, config.ubatch_size)
    batches = sorted({*_BATCH_CANDIDATES, config.batch_size})

    # One candidate per micro-batch size, because that is the axis the gains sit
    # on and pairing every batch with every micro-batch spends the whole budget
    # covering a dimension that barely moves. The logical batch is the smallest
    # one that can actually hold the micro-batch, since a larger one changes
    # nothing except how many tokens are queued ahead of it.
    candidates: List[SweepCandidate] = [planned]
    for ubatch in sorted({*_UBATCH_CANDIDATES, config.ubatch_size}):
        if ubatch == planned.ubatch_size or not fits(ubatch):
            continue
        batch = next((b for b in batches if b >= ubatch), ubatch)
        candidate = SweepCandidate(batch_size=batch, ubatch_size=ubatch)
        if candidate not in candidates:
            candidates.append(candidate)

    # Trim from the middle rather than the end: the extremes are what say
    # whether the curve rises or falls, and dropping the largest micro-batch
    # would discard the one most likely to win.
    while len(candidates) > max_candidates:
        candidates.pop(len(candidates) // 2)
    return candidates


# --------------------------------------------------------------------------- #
# what counts as better
# --------------------------------------------------------------------------- #

#: How much prefill matters, against decode, per use case. Both are real speeds
#: and neither is "the" speed: someone completing code as they type waits on
#: prompt processing every keystroke, while a conversation is dominated by how
#: fast words appear once the model starts.
_PREFILL_WEIGHT = {
    Task.AUTOCOMPLETE: 0.8,
    Task.RAG: 0.7,
    Task.SUMMARIZE: 0.7,
    Task.AGENT: 0.6,
    Task.CODING: 0.5,
    Task.EXTRACT: 0.5,
    Task.VISION: 0.4,
    Task.TRANSLATE: 0.3,
    Task.CHAT: 0.25,
    Task.CREATIVE: 0.2,
}
_DEFAULT_PREFILL_WEIGHT = 0.4


def prefill_weight(profile: UseCaseProfile) -> float:
    """How much this use case waits on reading rather than writing."""
    return _PREFILL_WEIGHT.get(profile.primary_task, _DEFAULT_PREFILL_WEIGHT)


def score_rows(rows: Sequence[BenchRow], weight: float) -> Dict[Tuple[int, int], float]:
    """Collapse each configuration's measurements into one comparable number.

    Prefill and decode are quoted in the same unit and differ by an order of
    magnitude, so a raw sum would let prompt throughput decide everything. Each
    is normalised against the best result for that phase, which turns both into
    a fraction of the best seen and makes the weight mean what it says.
    """
    prefill: Dict[Tuple[int, int], float] = {}
    decode: Dict[Tuple[int, int], float] = {}
    for row in rows:
        key = (row.n_batch, row.n_ubatch)
        target = prefill if row.is_prefill else decode
        # Several rows can share a configuration; keep the best of each phase.
        target[key] = max(target.get(key, 0.0), row.avg_ts)

    best_prefill = max(prefill.values(), default=0.0)
    best_decode = max(decode.values(), default=0.0)

    scores: Dict[Tuple[int, int], float] = {}
    for key in set(prefill) | set(decode):
        p = (prefill.get(key, 0.0) / best_prefill) if best_prefill else 0.0
        d = (decode.get(key, 0.0) / best_decode) if best_decode else 0.0
        if not prefill:
            scores[key] = d
        elif not decode:
            scores[key] = p
        else:
            scores[key] = weight * p + (1.0 - weight) * d
    return scores


def noise_floor(rows: Sequence[BenchRow], key: Tuple[int, int]) -> float:
    """Relative run-to-run spread for one configuration.

    llama-bench reports a standard deviation alongside every average. Expressed
    as a fraction of the average it is the smallest difference that means
    anything, and anything under it is measurement error wearing a result's
    clothes.
    """
    spreads = [row.stddev_ts / row.avg_ts
               for row in rows
               if (row.n_batch, row.n_ubatch) == key and row.avg_ts > 0]
    return max(spreads) if spreads else 0.0


def pick_best(rows: Sequence[BenchRow], baseline: SweepCandidate,
              weight: float) -> SweepResult:
    """Choose a configuration, or decline to.

    Declining is a real outcome and the common one. Most of the time the planned
    settings are already at or near the top of the curve, and reporting a
    quarter-percent improvement as a finding would be noise dressed up as
    tuning.
    """
    result = SweepResult(baseline=baseline, rows=list(rows))
    if not rows:
        result.rationale = "no measurements were taken"
        return result

    scores = score_rows(rows, weight)
    if not scores:
        result.rationale = "the benchmark produced no usable rows"
        return result

    baseline_key = (baseline.batch_size, baseline.ubatch_size)
    result.baseline_score = scores.get(baseline_key, 0.0)

    best_key = max(scores, key=lambda k: scores[k])
    result.best = SweepCandidate(batch_size=best_key[0], ubatch_size=best_key[1])
    result.best_score = scores[best_key]

    if result.baseline_score <= 0:
        # The planned configuration was never measured, so there is nothing to
        # compare against and no basis for claiming an improvement.
        result.best = baseline
        result.best_score = 0.0
        result.rationale = (
            "the planned settings were not among the measurements, so no "
            "comparison is possible"
        )
        return result

    result.improvement = (result.best_score / result.baseline_score) - 1.0
    floor = max(noise_floor(rows, best_key), noise_floor(rows, baseline_key))
    result.within_noise = result.improvement <= floor

    if best_key == baseline_key:
        result.rationale = (
            f"the planned settings ({baseline.label}) measured fastest of "
            f"{len(scores)} tried"
        )
    elif result.within_noise:
        result.rationale = (
            f"{result.best.label} came out {result.improvement:.1%} ahead of "
            f"{baseline.label}, which is inside the {floor:.1%} run-to-run "
            f"spread, so the planned settings stand"
        )
        result.best = baseline
    else:
        result.rationale = (
            f"{result.best.label} measured {result.improvement:.1%} faster than "
            f"the planned {baseline.label} on this machine, weighting prompt "
            f"processing at {weight:.0%} for this use case"
        )
    return result


# --------------------------------------------------------------------------- #
# running it
# --------------------------------------------------------------------------- #


def parse_bench_json(text: str) -> List[BenchRow]:
    """Read llama-bench's JSON output.

    Tolerant on purpose: a row missing a field it should have is skipped rather
    than taken down the whole sweep, because a partial measurement is still
    worth something and the tool's output has changed shape before.
    """
    try:
        payload = json.loads(text)
    except ValueError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("results") or []
    rows: List[BenchRow] = []
    for entry in payload if isinstance(payload, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            rows.append(BenchRow(
                n_batch=int(entry["n_batch"]),
                n_ubatch=int(entry["n_ubatch"]),
                n_prompt=int(entry.get("n_prompt", 0)),
                n_gen=int(entry.get("n_gen", 0)),
                avg_ts=float(entry["avg_ts"]),
                stddev_ts=float(entry.get("stddev_ts", 0.0)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def bench_command(binary: str, model_path: str, candidates: Sequence[SweepCandidate],
                  config: RuntimeConfig, prompt_tokens: int = 2048,
                  gen_tokens: int = 128, repetitions: int = 3) -> List[str]:
    """The llama-bench invocation that measures every candidate at once.

    One process rather than one per candidate: llama-bench takes lists and loads
    the model a single time, which is most of the wall clock on a large model.
    """
    batches = sorted({c.batch_size for c in candidates})
    ubatches = sorted({c.ubatch_size for c in candidates})
    cmd = [
        binary,
        "--model", model_path,
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-b", ",".join(str(b) for b in batches),
        "-ub", ",".join(str(u) for u in ubatches),
        "-ngl", str(config.gpu_layers),
        "-r", str(repetitions),
        "-o", "json",
    ]
    if config.kv_cache_type and config.kv_cache_type != "f16":
        cmd += ["-ctk", config.kv_cache_type, "-ctv", config.kv_cache_type]
    if config.threads:
        cmd += ["-t", str(config.threads)]
    cmd += ["-fa", "1" if config.flash_attention else "0"]
    return cmd


def run_sweep(
    spec: ModelSpec,
    config: RuntimeConfig,
    profile: UseCaseProfile,
    budget_bytes: int,
    bpw: float,
    console,
    binary: str = "llama-bench",
    repetitions: int = 3,
    max_candidates: int = 6,
) -> SweepResult:
    """Measure candidate settings on this machine and report what won.

    Returns a result rather than applying it. Whether to take a measured
    improvement is a decision with a cost attached, and the caller is the one
    holding the user's answer about that.
    """
    from llmfit.util.proc import run, which

    if which(binary) is None:
        return SweepResult(rationale=(
            f"{binary} is not installed, so nothing was measured. It ships with "
            f"llama.cpp and is what turns these settings from predicted into "
            f"known"
        ))
    if not config.model_path:
        return SweepResult(rationale=(
            "the weights are not on disk yet, and a sweep measures a real model"
        ))

    candidates = candidate_grid(spec, config, budget_bytes, bpw,
                                max_candidates=max_candidates)
    baseline = SweepCandidate(config.batch_size, config.ubatch_size)
    if len(candidates) < 2:
        return SweepResult(baseline=baseline, best=baseline, rationale=(
            "only the planned settings fit in the memory available at this "
            "context, so there was nothing to compare against"
        ))

    command = bench_command(binary, config.model_path, candidates, config,
                            repetitions=repetitions)
    console.info(
        f"measuring {len(candidates)} configurations "
        f"(about {estimate_duration_s(candidates, repetitions) / 60:.0f} minutes)"
    )
    console.debug(" ".join(command))

    result = run(command, timeout=BENCH_TIMEOUT_S)
    if not result.ok:
        return SweepResult(baseline=baseline, best=baseline, rationale=(
            f"the benchmark did not complete, so the planned settings stand: "
            f"{result.output.strip().splitlines()[-1][:200] if result.output else 'no output'}"
        ))

    rows = parse_bench_json(result.output)
    return pick_best(rows, baseline, prefill_weight(profile))


def apply_sweep(config: RuntimeConfig, sweep: SweepResult) -> bool:
    """Take a measured improvement. Returns whether anything changed."""
    if not sweep.worth_applying or sweep.best is None:
        return False
    config.batch_size = sweep.best.batch_size
    config.ubatch_size = sweep.best.ubatch_size
    return True


def estimate_duration_s(candidates: Sequence[SweepCandidate],
                        repetitions: int = 3) -> float:
    """Roughly how long the sweep will take, for asking before spending it.

    Deliberately generous. Someone told sixty seconds and made to wait three
    minutes trusts the next estimate less than someone told five.
    """
    # Each (batch, ubatch) pairing runs a prompt pass and a generation pass, and
    # llama-bench repeats each. Model load is paid once.
    per_measurement_s = 4.0
    pairs = len({(c.batch_size, c.ubatch_size) for c in candidates})
    return 30.0 + pairs * 2 * repetitions * per_measurement_s
