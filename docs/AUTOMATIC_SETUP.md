# Automatic setup and recovery

Run `llmsetup setup`, then tell it what you want the model to do. The interview
asks about tasks, apps, context, and capabilities. It never asks for hardware
specs or disk capacity. `--task coding --yes` skips the goal interview too.

The scan runs first. It reads OS and architecture, CPU and core counts, system
RAM, GPUs and VRAM, free disk space, and runtime availability. It checks Ollama,
llama.cpp, LM Studio, vLLM, MLX, and TabbyAPI through their adapters. Services or
Python environments outside those adapters' search paths can remain invisible.
`llmsetup detect --json` includes the runtime inventory and detection warnings.

A failed probe does not stop the interview to ask for a number. Unknown RAM
uses a 2 GiB assumption. Unknown GPU capacity uses CPU inference. Unknown free
VRAM reserves 20% of detected capacity. Known available RAM and VRAM limit the
budget. The CLI and saved hardware report record these assumptions.

## What recovery tries

Planning first tries the requested context, then halves it down to 512 tokens
if needed. Tools, vision, licensing, and other required capabilities stay in
force. The plan records any shorter window.

During a real setup, provisioning gets one extra attempt after a failure.
Downloads also have their existing four-retry budget and resume valid partial
files. If downloads remain blocked, repair tries a compatible cached GGUF model
or quantization before giving up. Invalid range responses discard the bad partial before a retry. Missing
GGUF shards are fetched before the model is launched. If a server identifies
corrupt weights, repair quarantines GGUF files in that model's managed store
directory and fetches them again. It does not move an external user-supplied
model file. Quarantined files use a `.corrupt` suffix.

The verifier can restart an unreachable server once, lower memory use, shorten
context, reduce GPU offload, run on CPU when RAM permits, choose a smaller
quantization, or switch models. Context reductions and model switches make
finite progress. The 12-action repair budget is shared across runtime attempts.
A server that exits during startup fails promptly; its log helps classify the
failure, and owned processes are stopped before another attempt.

If an engine is missing or still fails, setup tries compatible installed
alternatives. Installed llama.cpp also gets a CPU plan before setup installs a
portable llama.cpp build as the last CPU fallback. Even an explicitly preferred `--runtime` can change during
recovery. Each installed engine is tried once with its normal plan. Installed
llama.cpp can get one additional CPU plan, followed by one portable CPU attempt. The final report includes the runtime changes and final configuration.

The companion RAG retriever gets one retry with shorter context on CPU. Its
status is reported separately from the chat model's checks. A working chat
model does not mean retrieval passed. Failed main-model verification prevents
app registration.

## The local fallback engine

Portable installation supports Linux, macOS, and Windows on x86-64 and ARM64.
It uses llama.cpp release `b11669`, with a separate pinned SHA256 digest for
each archive. Archives are verified before extraction or execution. The engine
lives under `$LLMSETUP_HOME/runtimes/`, or the normal app-data directory when
that variable is unset. Setup does not use sudo, change PATH, or run a shell
installer. Native libraries or CPU instructions required by a verified build
may still be unavailable on an older system.

This installation is a CPU fallback. It does not install GPU drivers or a CUDA
stack, alter privileged OS settings, or install all six engines. Existing GPU
engines are preferred. `plan`, `detect`, and `setup --dry-run` never install an
engine. `--no-repair` disables repairs, extra provisioning attempts, installation, and
runtime fallback. Download transport retries still apply.

## When setup needs help

After recovery is exhausted, the error names the failed attempts and gives the
relevant next step. Examples include allowing blocked download hosts, setting
`HF_ENDPOINT` to a trusted reachable mirror, accepting a gated model's licence
and setting `HF_TOKEN`, freeing storage, or installing a compatible llama-server
when the portable build cannot run. If even the smallest compatible model
cannot fit, setup explains the memory limit instead of asking you to type specs.

Reports retain failed checks and the repair journal. `llmsetup doctor` rescans
and retries the saved goal and model after the underlying problem changes.
Smoke tests establish only what they tested, not general model quality or
support for every driver and runtime version.
