# Design notes

These notes preserve the detailed design rationale. Hardware figures and speed
comparisons are estimates or cited observations, not guarantees for every engine,
model revision, driver, or machine. See the README for installation and limits.

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

llmsetup works through all seven in order, and tests each answer rather than
claiming it.

## What a run looks like

```
$ llmsetup setup

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
world, each behind a narrow seam. See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

### The memory model

Sizing is computed rather than guessed at. Rules of thumb are what get people
OOM-killed at 32K context:

```
weights   = params × bits_per_weight / 8          # MoE: TOTAL params, all resident
kv_cache  = 2 × layers × kv_heads × head_dim × context × bytes_per_element
overhead  = compute buffers + activations + runtime fixed cost
```

Details that sizing calculators can get wrong:

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

So asking for it changes the whole plan rather than a setting. llmsetup picks a
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
| **Publisher-recommended sampling** | Model authors publish the settings they evaluated their own model with, and those beat any generic default. Qwen3 in thinking mode wants `temp 0.6 / top-p 0.95`; the same model with thinking off wants `0.7 / 0.8` and a presence penalty of 1.5; Gemma 3 wants `temp 1.0 / top-k 64`; Mistral wants `0.15`. llmsetup starts from the card for the family *and the mode it will actually run in*, then lets the task adjust only within what the publisher sanctioned. It refuses an adjustment the card explicitly warns against, such as greedy decoding on a Qwen3 reasoning model, which sends it into infinite repetition. Every entry cites its source. |
| **Reproducibility** | Extraction, coding and translation get a fixed seed, and the plan says what that does and does not buy. A seed removes the sampler as a source of variation; it does not make the engine deterministic, because floating-point addition is not associative and a request batched with three others accumulates in a different order. Where that matters and the hardware supports it, batch-invariant kernels are offered as an opt-in step with their real cost attached, which is roughly 60% longer per request. |
| **Tool-call parsers** | vLLM has to be told which shape a family emits tool calls in, and the shapes are not interchangeable. Only families with a documented parser get one; the rest get none, because a plausible-looking default causes the exact failure it is meant to prevent, where the model emits a good call and the server hands back prose. Which parser a build actually wants is settled by trying, so the tool check has a repair that walks the family's candidates. |
| **Full sampler chain** | DRY, XTC, top-n-sigma, typical-p, mirostat, penalty windows, and the *order* they run in, which changes output as much as the values. Code gets no repetition penalties at all, because repeated tokens in code are usually correct. |
| **RoPE / YaRN** | Context past the trained window, planned only when you ask for more than the model natively has, with the quality cost stated. |
| **Multi-GPU** | Tensor split proportional to VRAM, with the largest card holding the KV cache. |
| **Dynamic quantizations** | Not every layer deserves the same precision. Dynamic/imatrix builds (`UD-Q4_K_XL`) keep the layers that hurt most under quantization at higher precision and squeeze the rest, and measure better than the plain K-quant of the same size. llmsetup prefers them when the repo publishes one and falls back cleanly when it doesn't, while still honouring an explicit quant you asked for. |
| **Vision projector** | A vision model is two files: the language model and a separate projector that turns image embeddings into tokens. Download only the first and the endpoint accepts images and answers, confidently, about what it cannot see. llmsetup fetches the projector alongside the weights, prefers a full-precision one (vision towers degrade faster under quantization than the language model does, and the file is small), budgets memory for it, links it into LM Studio beside the model, and can repair a setup that is missing it. |
| **Serving** | Continuous batching, prefix caching (the biggest win for agents and RAG, where a long prefix repeats every turn), chunked prefill. |

A simple setup still produces a simple command line: everything here stays off
unless the tuner decided it was worth turning on, and the report lists only the
decisions that were actually made.

### The machine, not just the engine

Some of the biggest wins are not flags at all. They are settings on the machine
that the engine cannot reach, and llmsetup reports them as explicit, opt-in steps
with the reason attached rather than running them behind your back:

- **Apple Silicon.** macOS caps how much unified memory the GPU may wire down
  (roughly 65–75% of RAM). A model that fits in principle will fall back to CPU
  without `sudo sysctl iogpu.wired_limit_mb=N`. llmsetup raises it only when the
  plan actually exceeds the default budget, and never above 85% of RAM, so the
  OS keeps room to breathe.
- **AMD.** ROCm ships kernels for a subset of GFX targets. Cards that need
  `HSA_OVERRIDE_GFX_VERSION` to run at all (RDNA1/2, the 780M-class iGPUs) get
  the right value; cards that are natively supported are left alone, because
  setting it there makes things worse.
- **NUMA.** On a multi-socket Linux box doing CPU-resident work, memory
  interleaving is worth real throughput. Suggested only there, never on Apple
  unified memory, where the idea does not apply.

### Every option, and a decision about it

llmsetup emits 72 of the roughly 350 options llama-server accepts. Which of the
rest are deliberate and which were oversights used to be knowable only by
someone reading the upstream documentation and noticing, which is how several
rounds of gaps in this project were found. That method only finds what somebody
happens to read, and it kept finding things.

So every option carries a recorded decision: used, an alias of one that is, the
engine's own default, something belonging to whoever operates the server, a
choice belonging to the person rather than the tool, or a considered gap with a
note on what the work would be. A test checks the record against the code in
both directions, so a flag emitted without a decision fails, and so does a
decision claiming llmsetup passes something it does not. `scripts/flag_coverage.py`
diffs the record against an installed binary and names anything new.

Writing the record found four options llmsetup should have been using, and the
test caught me recording all four as used before I had implemented any of them.

### Flags the installed binary actually has

llama.cpp renames and retires flags between releases, and people run whatever
their package manager last shipped. A command written against today's
documentation fails on a perfectly good older build, and one written against an
old build gives up features the new one has.

So llmsetup asks the binary. `--help` lists every flag it accepts, which turns
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
context, llmsetup shortens the context.

### Losing memory gently on a dense model

A dense model that did not fit lost whole layers from the accelerator, which
costs an order of magnitude, because a dropped layer takes its attention to the
CPU along with its weights and attention is the half that wants to be near the
cache. `--n-cpu-ffn` moves only the feed-forward weights, which are most of a
layer's bytes and none of its attention. llmsetup was making that argument for
mixture-of-experts models and not for dense ones, which is most models.

A 32B at Q4 on a 12 GB card now keeps all 64 layers resident with the
feed-forward of 48 of them in RAM, where before it dropped 29 layers outright.
Where the feed-forward weights are not enough on their own, it says so and falls
back rather than pretending.

### Two cards of different speeds

llama.cpp's automatic split divides the model in proportion to memory, and
llmsetup's did the same. That is right only when the cards match. Under a layer
split every token walks through each card in turn, so the time per token is the
sum of the per-card times, and a card given work in proportion to its capacity
rather than its speed holds up everything behind it. Pairing a 3090 with a 3060
is the common case: three times the bandwidth on one side, and the
memory-proportional split hands the slow card a third of every forward pass.

The work is divided by bandwidth now, then capped by what each card can
physically hold, with the overflow going back to the others. Capacity is a
constraint rather than the objective, which is the way round it should have
been. The shares round down, because a card sitting at its ceiling is there
because it cannot take more.

Two smaller corrections came out of reading the upstream multi-GPU
documentation. `--main-gpu` names the single card to use when the model is not
split at all, so passing it alongside a split says nothing; llmsetup was passing
it and describing that card as holding the KV cache, which is not how a layer
split works and would send someone debugging in the wrong direction. And `row`
split mode is deprecated upstream, so it is no longer offered.

### On a Mac, the faster engine comes first

MLX generates appreciably faster than llama.cpp on Apple Silicon. llmsetup ranked
it third, on the reasoning that the GGUF engines have broader model coverage,
which was measured against a catalog recording exactly one MLX build. That is
circular, and it cost Mac users real speed on every model that does have one.

The engine cannot be chosen before the model, because it depends on the model,
so that decision is revisited once one is picked: a model with an MLX build gets
MLX, and everything else falls back to the GGUF engines. The report says plainly
that this is a fallback rather than the better choice.

### Sizing from the model, not from the card

Batch size was chosen from a table indexed on VRAM, before anything about the
model was known. That is wrong in a case people hit constantly: a mixture of
experts moves most of its weight to system RAM, so the card ends up holding
attention and a cache and little else, and a 6 GB laptop card can then carry the
same compute buffer as a 24 GB one running a dense model. Someone published
exactly that setup running at `-ub 2048` with 350-400 prompt tokens per second;
llmsetup was giving that machine 256.

The batch is now settled after placement, and the order of what gets given up
is deliberate:

1. **Quantize the KV cache** to the rung where it is still free.
2. **Shrink the prompt batch**, because the context was the requirement and the
   batch size was not. Practitioners report a third more usable context from
   this trade with no throughput they could measure.
3. **Move experts to RAM**, on a mixture of experts, before cutting the batch
   further: only the experts that fire cross the bus, while a narrower batch is
   paid on every token of every prompt.
4. **Shorten the context**, and only then.
5. **Spill layers to CPU**, which costs an order of magnitude.

Then, once everything is placed, the batch grows back into whatever memory is
left over.

### Experts move by the tensor

`--n-cpu-moe` works in whole layers, so freeing one byte past a layer boundary
moves an entire layer's experts and leaves the card holding nothing in the space
that frees. People reclaim it by hand with patterns like
`-ot 'blk\.21\.ffn_down.*=CPU'` and arrive at the right layer by trial.

The boundary is arithmetic, so llmsetup computes it: whole layers where they fit,
then the individual expert tensors of the layer that straddles the line. Every
expert left on the card is one that does not cross the bus when it fires.

### Measuring, not just predicting

Everything above computes. The memory model is exact arithmetic and the
throughput estimate is grounded in bandwidth, but neither can know that this
card, on this driver, with this build, processes prompts fastest at a
micro-batch of 2048 rather than the default 512. Measurements put that gap at
25-30% of prompt throughput on some hardware, and it is not the kind of thing a
formula recovers: it comes from kernel shapes and cache sizes.

`llmsetup setup --sweep` runs the candidates through llama-bench and keeps what
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

This is the one place llmsetup can beat careful manual work outright, and not
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
When weights land on disk, llmsetup reads the **real** architecture out of the GGUF
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
  demand, each with the command llmsetup tuned for it
