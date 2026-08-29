# llmfit

**Interview-driven setup for local open-weight LLMs.** It asks what you're going
to do, looks at your machine, picks the model, tunes the runtime, *proves the
setup works*, fixes it when it doesn't, and registers the result with LM Studio,
Ollama, Jan and everything else you already use.

```bash
pip install -e .
llmfit setup
```

---

## The problem

Running a local model means answering a chain of questions where every answer
depends on the last one:

1. Which of the hundreds of models? (and the right answer changed last month)
2. Which quantization? — depends on your VRAM **and** how long your context is
3. Which runtime? — Ollama, llama.cpp, vLLM, LM Studio, MLX
4. Which flags? — `-ngl`, `--ctx-size`, `--cache-type-k`, `--parallel`, …
5. Does it actually do what you needed? Tool calling? JSON? 64K context?
6. When it doesn't, *why*, and what do you change?
7. Can you still open it in LM Studio, or is it stuck in one tool?

`llmfit` answers all seven, in order, and **verifies each answer instead of
asserting it**.

## What a run looks like

```
$ llmfit setup

Let's work out what you need.
  What will you use the model for?          → coding, agents
  How much context do you need?             → 64K
  Which apps do you want to use it from?    → LM Studio, Continue

==> Planning
  ✓ Qwen3.8 27B at Q4_K_M (~19.6 GB) on ollama
    Qwen3.8 27B at Q4_K_M: strongest output quality in what fits.

==> Tuning
    gpu_layers = 62 (all) — the whole model and its KV cache fit in 22.1 GB
    flash_attention = on — reduces KV cache memory and speeds up prefill
    threads = 4 — fully on the GPU, so only a few CPU threads are needed
    sampling = temp=0.2 top_p=0.95 — code wants near-deterministic output

==> Provisioning        [downloads once, into a shared store]
==> Starting the server
  ✓ health: serving qwen3.8-27b
  ✓ completion: produces correct, coherent completions
  ✗ context: could not retrieve a fact placed mid-prompt at ~49,000 tokens

==> Repairing
  restore_kv_precision [medium]: restored KV cache precision to f16
    (context now 49,152 tokens) — aggressive KV quantization was costing recall
  ✓ context: retrieved a fact from the middle of a ~36,000-token prompt

==> Registering with your other apps
  ✓ lm_studio: placed under ~/.lmstudio/models as publisher/model/file.gguf
  ✓ openai_clients: config snippets written for Continue, Aider, Zed, …
```

## Design

Seven stages, each a transformation between plain dataclasses:

```
interview → detect → plan → tune → provision → verify ⇄ repair → interop
(what for?) (what on?) (what?) (how?)  (install)   (does it?)      (everywhere)
```

The interesting logic — fit maths, tuning, failure diagnosis — is a **pure
function over data**, so it is tested without a GPU, without a network and
without any inference engine installed. Only three stages touch the outside
world, each behind a narrow seam. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### The memory model

Sizing is computed, not guessed with rules of thumb — rules of thumb are what
get people OOM-killed at 32K context:

```
weights   = params × bits_per_weight / 8          # MoE: TOTAL params, all resident
kv_cache  = 2 × layers × kv_heads × head_dim × context × bytes_per_element
overhead  = compute buffers + activations + runtime fixed cost
```

Two details that are usually wrong elsewhere:

- **GQA-aware.** Using `kv_heads`, not attention heads. Every modern model uses
  grouped-query attention, and ignoring it overstates the cache several-fold.
- **MoE modelled twice.** *Total* parameters for memory (every expert is
  resident), *active* parameters for speed. Conflating those is the single most
  common local-LLM sizing mistake.

The KV term is why the interview asks about context up front: on a 24 GB card,
the difference between 8K and 128K is the difference between a 32B model and a
14B one — and at 128K a 14B model's cache costs more than its weights.

Throughput is predicted from physics rather than vibes — decode is
memory-bandwidth bound, so `tokens/sec ≈ bandwidth / active_weight_bytes` — and
the verifier measures the real number and compares. A large shortfall is
diagnostic: it usually means layers silently fell back to CPU.

### Verification

"It works" is only worth saying if it was tested. Checks are selected from *your*
answers, so a stated requirement is a tested one:

| Check | What it catches |
|---|---|
| `health` | server down; a *different* model still resident |
| `completion` | corrupt quantization, mis-applied chat template |
| `context` | **silent prompt truncation** — the server quietly ignoring your context setting |
| `tool_calling` | the model describing a tool call in prose instead of emitting one |
| `json_output` | schema-invalid structured output |
| `embeddings` | vectors that don't separate meaning (broken pooling ruins retrieval) |
| `vision` | an unloaded vision projector |
| `throughput` | layers that silently fell back to CPU |
| `stability` | leaks that only appear on the *second* request |
| `repetition` | degenerate loops from sampling that's too greedy |

Each failure returns a machine-readable signature, which is the join key into
repair.

### Repair

Diagnosis maps a signature to **ranked, costed** remediations. The loop applies
the cheapest applicable untried one and re-verifies. On a machine with less real
memory than estimated:

```
[low   ] reduce_batch          halved the prompt batch, freeing compute buffers
[low   ] quantize_kv_cache     KV cache → q8_0 (effectively lossless)
[medium] reduce_context        32,768 → 24,576 tokens — what actually fits
[high  ] reduce_gpu_layers     4 layers to CPU — frees VRAM, costs speed
✓ converged
```

It gives up a quantized KV cache before context, and context before the model.
Termination is structural: each (signature, strategy) pair is tried at most once,
so the loop cannot cycle. Every step is journalled, so the report explains *why*
your context ended up at 24,576 instead of silently handing you a worse setup.

### Ground truth beats the catalog

The catalog is dated YAML with per-entry provenance (`measured` vs `estimated`).
When weights land on disk, llmfit reads the **real** architecture out of the GGUF
header and recomputes the plan against it. A wrong catalog guess becomes a
corrected estimate instead of an OOM at inference time.

### Interop: the model is not trapped

Weights live **once**, in a shared store; every app is *linked* against it
(hardlink → symlink → copy, cheapest the filesystem allows):

- **LM Studio** — `~/.lmstudio/models/<publisher>/<model>/<file>.gguf`, the exact
  layout its UI parses (via `lms import` when the CLI is present)
- **Ollama** — a generated Modelfile carrying your tuning; derived from the
  official tag where one exists, so tuning costs no extra disk
- **Jan** — model folder plus `model.json`
- **Open WebUI, Continue, Aider, Zed, Cursor** — ready-to-paste config snippets

## Commands

```bash
llmfit setup            # the whole flow
llmfit detect           # what hardware do I have?
llmfit plan             # what would you pick, and why? (no downloads)
llmfit verify           # test an endpoint I already run
llmfit doctor           # re-test the last setup and repair it
llmfit export           # register an existing model with my other apps
llmfit catalog list     # what models are known
llmfit runtimes         # which engines are installed
```

Every stage runs standalone, so you can inspect the reasoning before committing
to a download:

```bash
# What would you pick for a 24 GB card doing agentic coding at 64K?
llmfit plan --vram 24 --gpu-vendor nvidia --task coding --task agent \
            --context 65536 --commercial

# Non-interactive and reproducible
llmfit setup --answers profile.yaml --report setup.md

# I already have something running — does it do what I need?
llmfit verify --base-url http://127.0.0.1:11434 --model qwen3:8b --tools
```

Detection is heuristic and you know your own machine, so everything is
overridable: `--vram`, `--ram`, `--gpu-vendor`, `--cpu-only`, `--threads`.
Overrides are recorded in the report, so a plan made against asserted rather
than measured numbers says so.

## Extending the catalog

The model landscape turns over every few weeks, so the catalog is **data, not
code**. Drop a file in `~/.local/share/llmfit/catalog.d/`:

```yaml
models:
  - id: my-finetune
    name: My Finetune 8B
    publisher: me
    params_b: 8
    layers: 32
    kv_heads: 8
    head_dim: 128
    max_context: 32768
    license: Apache-2.0
    tasks: [chat, coding]
    supports_tools: true
    gguf_repo: me/my-finetune-GGUF
```

Repeating an existing `id` patches just those fields, so overriding one setting
doesn't mean restating the whole entry.

## Requirements

Python 3.9+ and PyYAML — deliberately close to zero dependencies, because this
runs on whatever machine you already have. An inference engine is needed only for
`setup` and `verify`; `plan` and `detect` work with nothing installed.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 223 tests, no GPU or network required
```

Tests cover the fit maths against independently published figures (the KV-cache
formula is pinned exactly), catalog integrity, repair convergence and
termination, and every check against a scripted endpoint.

## License

Apache-2.0
