# llmfit

Setup assistant for local open-weight LLMs. It asks what you're going to do
with a model, looks at your machine, picks one, tunes the runtime for it, tests
that the result actually works, fixes it when it doesn't, and registers it with
LM Studio, Ollama, Jan and whatever else you already use.

```bash
pip install -e .
llmfit setup
```

---

## The problem

Running a local model means answering a chain of questions where every answer
depends on the last one:

1. Which of the hundreds of models, given that the right answer changed last month
2. Which quantization, which depends on your VRAM and on how long your context is
3. Which runtime: Ollama, llama.cpp, vLLM, LM Studio, MLX
4. Which flags: `-ngl`, `--ctx-size`, `--cache-type-k`, `--parallel`, and so on
5. Whether it does what you needed. Tool calling? JSON? 64K context?
6. When it doesn't, why, and what to change
7. Whether you can still open it in LM Studio, or it is stuck in one tool

llmfit works through all seven in order, and tests each answer rather than
claiming it.

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
    gpu_layers = 62 (all): the whole model and its KV cache fit in 22.1 GB
    flash_attention = on: reduces KV cache memory and speeds up prefill
    threads = 4: fully on the GPU, so only a few CPU threads are needed
    sampling = temp=0.2 top_p=0.95: code wants near-deterministic output

==> Provisioning        [downloads once, into a shared store]
==> Starting the server
  ✓ health: serving qwen3.8-27b
  ✓ completion: produces correct, coherent completions
  ✗ context: could not retrieve a fact placed mid-prompt at ~49,000 tokens

==> Repairing
  restore_kv_precision [medium]: restored KV cache precision to f16
    (context now 49,152 tokens): aggressive KV quantization was costing recall
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

The parts that make decisions (fit maths, tuning, failure diagnosis) are pure
functions over data, so they can be tested without a GPU, without a network and
without any inference engine installed. Only three stages touch the outside
world, each behind a narrow seam. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### The memory model

Sizing is computed rather than guessed at. Rules of thumb are what get people
OOM-killed at 32K context:

```
weights   = params × bits_per_weight / 8          # MoE: TOTAL params, all resident
kv_cache  = 2 × layers × kv_heads × head_dim × context × bytes_per_element
overhead  = compute buffers + activations + runtime fixed cost
```

Three details that calculators usually get wrong:

- **Grouped-query attention.** The cache is sized by `kv_heads`, not by
  attention heads. Every modern model uses GQA, and ignoring it overstates the
  cache several times over.
- **Mixture of experts, counted twice.** Total parameters for memory, because
  every expert stays resident, and active parameters for speed. Conflating the
  two is the most common local sizing mistake there is.
- **Sliding windows.** Some models interleave layers that attend globally with
  layers that only attend inside a fixed window, and the windowed ones stop
  caching once the window fills. Treating them as global put Gemma 3 27B at
  15.5 GB of predicted cache at 32K against a real 2.9 GB, which is the
  difference between fitting a 24 GB card and being ruled out of the plan.
- **Recurrent layers.** A hybrid replaces most of its attention layers with
  recurrent ones that hold a fixed-size state instead of a cache that grows.
  Three quarters of Qwen3.6 35B-A3B is built this way, so it was being charged
  24 GB of cache at 262K against a real 6 GB. On a 24 GB budget it reaches 172K
  tokens, where the old maths said 43K.

The KV term is why the interview asks about context up front: on a 24 GB card,
the difference between 8K and 128K is the difference between a 32B model and a
14B one, and at 128K a 14B model's cache costs more than its weights.

Throughput comes from the physics rather than from a feel for it. Decode is
memory-bandwidth bound, so `tokens/sec ≈ bandwidth / active_weight_bytes`, and
the verifier measures the real number and compares the two. A large shortfall is
diagnostic: it usually means layers silently fell back to CPU.

### Verification

"It works" is only worth saying if it was tested. Checks are selected from *your*
answers, so a stated requirement is a tested one:

| Check | What it catches |
|---|---|
| `health` | server down; a *different* model still resident |
| `completion` | corrupt quantization, mis-applied chat template |
| `context` | **silent prompt truncation**: the server quietly ignoring the context setting you gave it |
| `tool_calling` | the model describing a tool call in prose instead of emitting one |
| `json_output` | schema-invalid structured output |
| `embeddings` | vectors that don't separate meaning (broken pooling ruins retrieval) |
| `vision` | a missing projector: the model answering about an image it cannot see |
| `throughput` | layers that silently fell back to CPU |
| `prefill` | prompt processing too slow to use, which is the number RAG and agents wait on |
| `stability` | leaks that only appear on the *second* request |
| `repetition` | degenerate loops from sampling that's too greedy |

Each failure returns a machine-readable signature, which is the join key into
repair.

### Inline completion is a different setup

Autocomplete in an editor is not chat with a shorter prompt. It uses a separate
endpoint, `/infill`, where the model is handed the code before and after the
cursor and asked to join them. Instruction-tuned models generally cannot do that
at all, however good they are at writing code in a chat window, because they
were never taught the tokens that express it.

So asking for it changes the whole plan rather than a setting. llmfit picks a
model trained to infill, sized by *active* parameters rather than total, because
that is what decides how long the editor waits: a 30B mixture of experts that
activates 3B answers at 3B speed. It routes to an engine that has the endpoint,
since Ollama and LM Studio serve the same model perfectly well for chat and
simply have no infill route, which leaves an editor plugin looking broken. It
turns thinking off, because a reasoning block spends the entire latency budget
before the first character appears. It caches the prefix, since every keystroke
resends the file above the cursor. And it writes the `autocomplete` role into
the editor config alongside the chat one.

Infilling is treated as its own sampling mode, alongside instruct and thinking.
A model card's temperature describes the conversational mode, and completion
never goes through the chat template at all.

### RAG gets a real retriever

Asking for RAG provisions **two** models: the chat model, and a separate
embedding model chosen for your languages (a multilingual corpus gets a
multilingual retriever, because an English-only one returns plausible but wrong
neighbours). The retriever is verified like everything else, and a failure there
is reported rather than dressed up: a chat model makes a poor retriever, and
silently shipping one would degrade retrieval invisibly.

### Deep configuration

Past the obvious knobs, the tuner plans the settings that actually separate "it
runs" from "it runs well". Each is reported with the reason it was chosen:

| | |
|---|---|
| **MoE expert offload** | A mixture-of-experts model is ~96% routed expert weights, and only a few experts fire per token. Those stacks go to system RAM while attention and the KV cache stay on the GPU. This is what makes a 117B model runnable on a 12 GB card, and it costs far less speed than moving whole layers, because only *active* parameters cross. |
| **Speculative decoding** | Drafting ahead and verifying in one pass, in whichever form the model supports. Some models carry multi-token prediction heads inside their own weights, which costs no download and no second resident model, so nothing else competes when a build has them. Failing that, a small draft model from the catalog (same vocabulary, at least 5x smaller) when the task has predictable output and there is memory to spare. Failing that, drafting from the prompt itself, which costs nothing and pays when the answer quotes its source. Whether a build has the heads is read out of the file, because publishers ship an MTP build and a plain build of the same model. |
| **Reasoning budget** | `--reasoning on/off` and effort tiers. Suppressing the thinking block takes time-to-first-token from seconds to a fraction of one, so extraction and translation get it off while agents keep it. |
| **Publisher-recommended sampling** | Model authors publish the settings they evaluated their own model with, and those beat any generic default. Qwen3 in thinking mode wants `temp 0.6 / top-p 0.95`; the same model with thinking off wants `0.7 / 0.8` and a presence penalty of 1.5; Gemma 3 wants `temp 1.0 / top-k 64`; Mistral wants `0.15`. llmfit starts from the card for the family *and the mode it will actually run in*, then lets the task adjust only within what the publisher sanctioned. It refuses an adjustment the card explicitly warns against, such as greedy decoding on a Qwen3 reasoning model, which sends it into infinite repetition. Every entry cites its source. |
| **Reproducibility** | Extraction, coding and translation get a fixed seed, and the plan says what that does and does not buy. A seed removes the sampler as a source of variation; it does not make the engine deterministic, because floating-point addition is not associative and a request batched with three others accumulates in a different order. Where that matters and the hardware supports it, batch-invariant kernels are offered as an opt-in step with their real cost attached, which is roughly 60% longer per request. |
| **Tool-call parsers** | vLLM has to be told which shape a family emits tool calls in, and the shapes are not interchangeable. Only families with a documented parser get one; the rest get none, because a plausible-looking default causes the exact failure it is meant to prevent, where the model emits a good call and the server hands back prose. Which parser a build actually wants is settled by trying, so the tool check has a repair that walks the family's candidates. |
| **Full sampler chain** | DRY, XTC, top-n-sigma, typical-p, mirostat, penalty windows, and the *order* they run in, which changes output as much as the values. Code gets no repetition penalties at all, because repeated tokens in code are usually correct. |
| **RoPE / YaRN** | Context past the trained window, planned only when you ask for more than the model natively has, with the quality cost stated. |
| **Multi-GPU** | Tensor split proportional to VRAM, with the largest card holding the KV cache. |
| **Dynamic quantizations** | Not every layer deserves the same precision. Dynamic/imatrix builds (`UD-Q4_K_XL`) keep the layers that hurt most under quantization at higher precision and squeeze the rest, and measure better than the plain K-quant of the same size. llmfit prefers them when the repo publishes one and falls back cleanly when it doesn't, while still honouring an explicit quant you asked for. |
| **Vision projector** | A vision model is two files: the language model and a separate projector that turns image embeddings into tokens. Download only the first and the endpoint accepts images and answers, confidently, about what it cannot see. llmfit fetches the projector alongside the weights, prefers a full-precision one (vision towers degrade faster under quantization than the language model does, and the file is small), budgets memory for it, links it into LM Studio beside the model, and can repair a setup that is missing it. |
| **Serving** | Continuous batching, prefix caching (the biggest win for agents and RAG, where a long prefix repeats every turn), chunked prefill. |

A simple setup still produces a simple command line: everything here stays off
unless the tuner decided it was worth turning on, and the report lists only the
decisions that were actually made.

### The machine, not just the engine

Some of the biggest wins are not flags at all. They are settings on the machine
that the engine cannot reach, and llmfit reports them as explicit, opt-in steps
with the reason attached rather than running them behind your back:

- **Apple Silicon.** macOS caps how much unified memory the GPU may wire down
  (roughly 65–75% of RAM). A model that fits in principle will fall back to CPU
  without `sudo sysctl iogpu.wired_limit_mb=N`. llmfit raises it only when the
  plan actually exceeds the default budget, and never above 85% of RAM, so the
  OS keeps room to breathe.
- **AMD.** ROCm ships kernels for a subset of GFX targets. Cards that need
  `HSA_OVERRIDE_GFX_VERSION` to run at all (RDNA1/2, the 780M-class iGPUs) get
  the right value; cards that are natively supported are left alone, because
  setting it there makes things worse.
- **NUMA.** On a multi-socket Linux box doing CPU-resident work, memory
  interleaving is worth real throughput. Suggested only there, never on Apple
  unified memory, where the idea does not apply.

### Flags the installed binary actually has

llama.cpp renames and retires flags between releases, and people run whatever
their package manager last shipped. A command written against today's
documentation fails on a perfectly good older build, and one written against an
old build gives up features the new one has.

So llmfit asks the binary. `--help` lists every flag it accepts, which turns
that whole class of breakage into a question with an answer: anything the build
does not recognise is dropped, and the setup runs with one feature less instead
of the server refusing to start behind 400 lines of usage text. The answer is
cached per binary, so it costs one subprocess per install.

### Where a saving stops being free

Not every memory saving is worth taking, and the difference is measurable rather
than a matter of taste. A `q8_0` KV cache halves what context costs and measures
as effectively lossless against `f16`: a handful of changed answers in five
hundred, none at all across deep-context comparisons out to 41K tokens. Below
that rung the picture changes and stops being predictable: `q4_0` is strongly
model-dependent, ranging from a couple of changed answers to 375 in 500 on one
7B, and past long context it is *slower* than `f16` too, because dequantizing
inside attention costs more than the memory traffic it saves.

So `q8_0` is chosen freely, and anything below it is never a silent default: it
is a last-resort repair, priced as a real sacrifice, blocked at the context
lengths where it would be slower as well as worse, and reported in words that
say how to undo it. Given the choice between a lossy cache and a shorter
context, llmfit shortens the context.

### Measuring, not just predicting

Everything above computes. The memory model is exact arithmetic and the
throughput estimate is grounded in bandwidth, but neither can know that this
card, on this driver, with this build, processes prompts fastest at a
micro-batch of 2048 rather than the default 512. Measurements put that gap at
25-30% of prompt throughput on some hardware, and it is not the kind of thing a
formula recovers: it comes from kernel shapes and cache sizes.

`llmfit setup --sweep` runs the candidates through llama-bench and keeps what
wins. Three rules keep it honest:

- **Nothing is benchmarked that would not fit.** Candidates go through the same
  memory model as the plan, at the context you asked for, so the sweep cannot
  recommend a setting that runs out of memory later.
- **The objective follows the use case.** Prefill and decode trade against each
  other, and which one matters is not a property of the hardware. Inline
  completion is scored at 80% prefill; long-form writing at 20%.
- **A win inside the noise is not a win.** llama-bench reports a standard
  deviation with every number, and a candidate that beats the default by less
  than the run-to-run spread is reported as a tie. Most of the time the planned
  settings are already at the top of the curve, and that is a result worth
  printing too: settings nobody checked and settings that were checked and held
  up are different things.

This is the one place llmfit can beat careful manual work outright, and not
because the method is clever. A person tries three configurations and stops; a
machine sits through all of them in a fixed order and writes down the spread.

### Two speeds, not one

"Tokens per second" almost always means decode, or how fast words appear once the
model starts. Prefill, the rate the model *reads* at, is a different number
bound by a different thing (compute, not memory bandwidth), and it is what
actually keeps you waiting: a RAG answer, an agent turn carrying its history, a
file pasted into a coding assistant. A setup can be fine at one and unusable at
the other.

Both are measured, against the machine rather than an arbitrary floor. Prefill
is measured exactly. One long prompt, one token of output, so the server's own
prompt-token count over the latency *is* the rate. It is reported as the thing
you feel: *"512 tokens/s prompt processing (16s at full context)"*. Slow prefill
is a repairable signature of its own, and the first thing tried is the physical
micro-batch, the flag that moves prompt throughput most.

### Repair

Diagnosis maps a signature to **ranked, costed** remediations. The loop applies
the cheapest applicable untried one and re-verifies. On a machine with less real
memory than estimated:

```
[low   ] reduce_batch          halved the prompt batch, freeing compute buffers
[low   ] quantize_kv_cache     KV cache → q8_0 (effectively lossless)
[medium] reduce_context        32,768 → 24,576 tokens, what actually fits
[high  ] reduce_gpu_layers     4 layers to CPU, frees VRAM, costs speed
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

- **LM Studio.** `~/.lmstudio/models/<publisher>/<model>/<file>.gguf`, the exact
  layout its UI parses (via `lms import` when the CLI is present)
- **Ollama.** A generated Modelfile carrying your tuning, derived from the
  official tag where one exists, so tuning costs no extra disk
- **Jan.** Model folder plus `model.json`
- **Open WebUI, Continue, Aider, Zed, Cursor.** Ready-to-paste config snippets
- **llama-swap.** A generated config so one endpoint serves several models on
  demand, each with the command llmfit tuned for it

## Terminal or a window, whichever suits

The engine is a CLI. `llmfit ui` puts a local page in front of it:

```bash
llmfit ui        # opens http://127.0.0.1:8765
```

It shows what it detected about your machine, asks the same questions the
terminal wizard asks, and lays each recommendation out with its memory broken
into weights, KV cache and overhead against what your machine can actually
spare, which is the number that decides everything else.

It is a page, not an Electron app, and that is deliberate: no Node toolchain, no
build step, no per-platform packaging or code signing, no separate self-updater.
It ships in the same `pip install`, works everywhere the CLI does, and fetches
nothing: no CDN, no fonts, no analytics. A tool for running models
locally should work with the network off.

The UI plans and explains; it hands the download over as a command rather than
starting one behind a button, so the irreversible step stays explicit and the
terminal stays the source of truth. The two can't drift, because the page is a
thin client over the same functions.

## Commands

```bash
llmfit setup            # the whole flow
llmfit detect           # what hardware do I have?
llmfit plan             # what would you pick, and why? (no downloads)
llmfit verify           # test an endpoint I already run
llmfit setup --sweep    # measure batch settings here and keep the fastest
llmfit doctor           # re-test the last setup and repair it
llmfit report           # reprint the last run's report (paste this into a bug report)
llmfit export           # register an existing model with my other apps
llmfit catalog list     # what models are known
llmfit catalog add      # teach it about a model it doesn't know
llmfit catalog refresh  # check the catalog against reality
llmfit catalog update   # pull the newest catalog now
llmfit runtimes         # which engines are installed (llama.cpp, Ollama,
                        #   vLLM, LM Studio, MLX, TabbyAPI/ExLlamaV2)
llmfit ui               # the local web interface
```

Every stage runs standalone, so you can inspect the reasoning before committing
to a download:

```bash
# What would you pick for a 24 GB card doing agentic coding at 64K?
llmfit plan --vram 24 --gpu-vendor nvidia --task coding --task agent \
            --context 65536 --commercial

# Non-interactive and reproducible
llmfit setup --answers profile.yaml --report setup.md

# I already have something running. Does it do what I need?
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

Or let it read the architecture out of a file you already have, which is safer
than typing it, because layer and KV-head counts drive the whole memory model:

```bash
llmfit catalog add my-finetune --gguf ./my-finetune-Q4_K_M.gguf --publisher me
#  ✓ read architecture from the file: 48 layers, 8 KV heads, 128 head dim
#  ✓ my-finetune is now in the catalog (7.9B, 48 layers, measured architecture)
```

`llmfit catalog refresh` checks entries against local GGUFs and upstream repos
and reports drift: a renamed repo, a quantization that no longer exists, a
layer count that does not match the file. It writes a **proposed** overlay
rather than editing in place: a catalog that changes underneath you turns a
reproducible setup into a moving target.

## It updates itself

The catalog is a snapshot of a landscape that turns over every few weeks, so
nothing here depends on someone remembering to publish a release.

**Your install pulls new models on its own.** Once a day (configurable), llmfit
fetches a newer catalog, validates it, and caches it alongside your own entries.
A model released after you installed simply shows up.

```bash
llmfit catalog auto-update              # when it last checked, and from where
llmfit catalog auto-update --disable    # opt out for good
llmfit catalog update                   # check right now
```

It is built to be invisible when it fails: one short timeout, checked at most
once per interval, and **any** failure (offline, DNS, a 500, malformed YAML)
leaves the bundled catalog in place and the run continues. It never fetches for
an `--offline` profile.

**The published catalog refreshes itself too.** A scheduled workflow discovers
newly popular open-weight models on Hugging Face and commits them, so the file
your install pulls stays current with nobody pushing anything.

Discovery reads architecture from each model's `config.json`: `num_hidden_layers`,
`num_key_value_heads`, `hidden_size`, `head_dim`, which is exactly what the
memory model needs, so an auto-added entry is as trustworthy as a hand-written
one. **It will not add a model it cannot describe**: an incomplete config is a
reason to skip, because a guessed layer count is a wrong memory plan and an
out-of-memory error on someone's machine.

Before anything is committed, the updated catalog must load, have plausible
values, survive a real planning run, and pass the full test suite. If any of
that fails the job fails loudly and nothing is written. Point it elsewhere with
`LLMFIT_CATALOG_URL` (a `file://` or internal URL works, for air-gapped setups).

## Security posture

Catalog YAML is treated as **untrusted input**, because it is designed to be
extended, shared and refreshed:

- every catalog-derived path component is sanitised and the resulting path is
  re-checked against its root, so an entry cannot place a multi-gigabyte
  download outside the model store;
- repository ids are validated against an anchored pattern before they reach a
  URL;
- GGUF header lengths are bounded by the real file size, so a crafted 58-byte
  file cannot trigger a huge allocation. A hostile or corrupt file degrades to
  "use the catalog estimate" rather than crashing the run.

There is one runtime dependency (PyYAML), parsed with `safe_load`. No `shell=True`,
no `eval`, no `pickle`, and TLS verification is never disabled. CI runs
`pip-audit` against the declared dependency set on every push.

## Requirements

Python 3.9+ and PyYAML, deliberately close to zero dependencies, because this
runs on whatever machine you already have. An inference engine is needed only for
`setup` and `verify`; `plan` and `detect` work with nothing installed.

The 3.9 floor is tested, not assumed: CI runs the full suite on 3.9, 3.10, 3.11,
3.12 and 3.13.

## Development

```bash
pip install -e ".[dev]"
pytest                       # 714 tests, no GPU or network required
ruff check src/ tests/
```

Tests cover the fit maths against independently published figures (the KV-cache
formula is pinned exactly), catalog integrity, repair convergence and
termination, every check against a scripted endpoint, and a security suite that
reproduces each fixed weakness (path traversal, URL injection, hostile GGUF
headers). Nothing in the suite needs a GPU or the network, and anything reaching
for either is a bug in the test.

## Contributing

The most useful contribution is usually not code. It is a catalog entry, or a
report of a machine where the plan came out wrong. `llmfit report` prints
exactly what makes that fixable.

One rule governs the rest: **no entry is better than a guessed one.** Every
number here decides whether a model fits in a stranger's memory, and a
plausible guess that is 20% low becomes their out-of-memory error with no way
to trace it back. Facts read from a config, a model card or a GGUF header are
marked `measured`; inferences are marked `estimated` and re-derived from the
real file once the weights land; anything unknown is left out, and the catalog
refuses the entry loudly at load time. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
