# llmsetup architecture

## The problem

Someone wants to run an open-weight LLM locally. Today that means personally
answering a chain of questions where every answer depends on the previous one:

1. Which model? (hundreds of them; the good answer changes every few weeks)
2. Which quantization? (depends on their VRAM *and* how long their context is)
3. Which runtime? (Ollama, llama.cpp, vLLM, LM Studio or MLX, depending on OS and GPU)
4. Which flags? (`-ngl`, `--ctx-size`, `--cache-type-k`, `--parallel`, …)
5. Does it actually work for what they wanted? (tool calling? JSON? 64K context?)
6. When it doesn't, *why* not, and what do you change?
7. And can they still open it in LM Studio afterwards, or is it trapped?

`llmsetup` answers all seven, in that order, and verifies each answer instead of
asserting it.

## Pipeline

```
  detect  ──►  interview  ──►  plan  ──►  tune  ──►  provision  ──►  verify ◄──┐
  (automatic)  (goals only)   (what?)   (how?)      (install)      (does it?)   │
                                                                       │       │
                                                                       ▼       │
                                                                    repair ────┘
                                                                       │
                                                                       ▼
                                                                    interop
                                                              (register everywhere)
```

Each stage is a pure-ish function from one dataclass to the next
(`llmsetup.types`). That is the central design constraint: the interesting logic
(fit math, tuning, failure diagnosis) is a pure transformation over plain data,
so it is unit-testable without a GPU, without a network, and without any
inference engine installed. Only `provision`, `verify` and `interop` touch the
outside world, and each of those goes through a narrow, fakeable seam.

| Stage | Input | Output | Touches the world? |
|---|---|---|---|
| interview | user answers | `UseCaseProfile` | TTY only |
| detect | (nothing) | `HardwareProfile` | reads `/proc`, `nvidia-smi`, `sysctl` |
| plan | profile + hardware + catalog | `Plan` (ranked `Candidate`s) | no |
| tune | candidate + hardware | `RuntimeConfig` | no |
| provision | `RuntimeConfig` | running endpoint | downloads, spawns servers |
| verify | endpoint + profile | `[CheckResult]` | HTTP to the local endpoint |
| repair | failures + config | patched `RuntimeConfig` | no (decides; loop applies) |
| interop | final config | registrations | writes files, symlinks |

## Why each stage looks the way it does

### interview
The use case decides more than the hardware does. A 24 GB card running a coding
agent wants a different model, a different context length, and different
sampling parameters than the same card doing bulk summarization. The interview
produces **capability requirements** (`needs_tools`, `needs_vision`,
`needs_json`, `min_context`) that later become *both* filter predicates in the
planner *and* the set of checks the verifier runs. One answer, two uses: a
requirement the user states is a requirement the tool proves.

### detect
Detection runs before the goal interview. It scans CPU, core counts, OS,
architecture, RAM, GPUs and VRAM, disk space, and all six runtime adapters.
Each runtime and GPU vendor is probed independently. A failed probe produces a
warning and conservative defaults, without asking the user for specs. Unknown
RAM means 2 GiB, unknown GPU capacity means CPU inference, and unknown free VRAM
reserves 20% of reported capacity. Known free memory limits the planning budget.
Advanced hardware flags remain available for diagnostics and simulated plans.
Detection does not install anything or require root.

### plan: the fit maths
Memory is modelled explicitly rather than with rules of thumb, because rules of
thumb are what get people OOM-killed at 32K context:

```
weights   = params * bits_per_weight / 8
kv_cache  = 2 * layers * kv_heads * head_dim * context * bytes_per_element
            (GQA-aware: kv_heads, not attention heads)
overhead  = compute buffers + activations + runtime fixed cost
total     = weights + kv_cache + overhead
```

The KV term is why the planner asks for context length up front: on a 24 GB card
the difference between 8K and 128K context is the difference between a 32B model
and a 14B one. MoE models are modelled by *total* parameters for memory (all
experts must be resident) and *active* parameters for speed. Conflating those
two is the single most common local-LLM sizing mistake.

Ranking is a weighted score over: capability match (hard filter first), quality
tier, measured fit headroom, expected throughput, and license permissiveness.
The planner always returns a ranked list with fallbacks, never a single answer,
because the repair loop needs somewhere to fall back *to*.

### tune
Given the chosen candidate and the real hardware, derive concrete flags: GPU
layer offload count from the per-layer byte cost, KV cache dtype from remaining
headroom, batch/ubatch from throughput-vs-memory, thread counts from physical
cores, `--parallel` from whether the use case is single-user chat or a served
endpoint. Sampling defaults are per-family (reasoning models want different
temperature than instruct models) and per-use-case.

### verify
The claim "your setup works" is only worth making if it was tested. Checks are
selected by the `UseCaseProfile`, so a user who said "I need tool calling" gets a
tool-calling check that must pass, and a user who didn't, doesn't wait for one.
Each check returns a structured `CheckResult` carrying a machine-readable
`failure_signature`, and that signature is the input to repair.

### repair
Diagnosis is a lookup from failure signature to *ranked* remediation strategies,
each with a cost (how much quality/speed it gives up) and a predicate (when it
is applicable). The loop applies the cheapest applicable untried strategy,
re-runs only the affected checks, and records every step in a journal so the
final report can explain *why* the settings ended up as they did. Bounded
retries, monotonic de-escalation, no infinite loops. Context reduction and model
fallback can repeat only while making finite progress. Model switches preserve
a reduced context window. Provisioning and server failures become blocking
checks, so recovery can act on them. An unclassified blocking failure still
blocks success.

The outer pipeline tries compatible installed engines, then a user-local
checksum-verified portable llama.cpp build on CPU. Runtime changes preserve the
remaining repair budget. Failed setups are not registered with other apps.
Only exhausted recovery emits manual instructions. See
[AUTOMATIC_SETUP.md](AUTOMATIC_SETUP.md) for bounds and installation details.

### interop
The requirement is "I can still see this model in LM Studio and everywhere else."
So weights live once, in a **shared store**, and each app is *registered against*
that store rather than handed its own copy:

- **LM Studio.** `~/.lmstudio/models/<publisher>/<model>/<file>.gguf`, the
  layout its UI parses for the publisher/model dropdowns.
- **Ollama.** A generated `Modelfile` with `FROM <abs path>` plus the tuned
  parameters, so Ollama's copy carries the tuning too.
- **Jan.** Model folder plus `model.json`.
- **Everything OpenAI-compatible** (Open WebUI, Continue, Aider, Zed, Cursor):
  emitted config snippets pointing at the local endpoint.

Registration is idempotent, uses hardlink → symlink → copy in that order (cheapest
that the filesystem supports), and never deletes user data.

## Failure philosophy

Three rules the code holds to:

1. **Never claim success without a check backing it.** Report text is generated
   from `CheckResult`s, not from intent.
2. **Degrade, don't crash.** Missing `nvidia-smi`, no network, an unknown GGUF:
   each narrows what can be concluded, and the narrowing is reported.
3. **Every automatic change is explainable.** The journal records the trigger,
   the strategy, and the cost of each repair, so the user can see the tool traded
   16K of context for stability rather than silently getting a worse setup.
