# llmsetup

llmsetup sets up an open-weight LLM on your machine. It scans your hardware and
installed runtimes first, then asks what you want to do, such as coding, chat,
or running agents. You never need to enter hardware specs. It picks a model,
downloads the weights, starts a server, tests the capabilities you need, and
registers the working setup with your local apps.

When something fails, it tries repairs and fallback options automatically.
It only gives you a manual step after those options have failed.

This is the Python project at
[catfish-1234/llmsetup](https://github.com/catfish-1234/llmsetup),
licensed under Apache-2.0. It was renamed from llmfit to avoid confusion with the
unrelated [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) Rust project.
See [the name change and migration notes](docs/POSITIONING.md).

## Install

Use Python 3.9 or newer, preferably a version that still receives security
updates. PyYAML is the only runtime Python dependency.

```bash
python -m pip install llmsetup
llmsetup --version
```

To install from source instead:

```bash
git clone https://github.com/catfish-1234/llmsetup.git
cd llmsetup
python -m venv .venv
source .venv/bin/activate
python -m pip install .
llmsetup --version
llmsetup --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. For
source development, use `python -m pip install -e .`.

You can make plans without a GPU or inference engine. Adapters cover Ollama,
llama.cpp, LM Studio, vLLM, MLX, and TabbyAPI. Setup prefers an installed engine.
If that engine is missing or fails, it tries compatible installed alternatives,
including llama.cpp on CPU. It can install a pinned, checksum-verified llama.cpp
build in its own data directory. Windows x86-64 machines with a detected GPU
get the Vulkan build; CPU-only machines get the CPU build. CPU fallback is
available when GPU attempts fail. It does not need administrator access.

## Quickstart

```bash
# Inspect hardware, current memory use, and available runtimes.
llmsetup detect

# Scan the machine and ask about your goals.
llmsetup setup

# Skip the interview when you already know what you want.
llmsetup plan --task coding --task agent --context 32768 --commercial --yes

# Inspect tuning without installing an engine, downloading weights, or starting a server.
llmsetup setup --dry-run --task coding --context 32768 --yes

# Download, start, verify, and repair automatically.
llmsetup setup --task coding --context 8192 --yes

# Read the saved settings, checks, repair history, and launch command.
llmsetup report

# Start the last verified setup again, exactly as it was tested.
llmsetup start

# Or keep it running in the background, then check on it, read its log, and stop it.
llmsetup start --detach
llmsetup status
llmsetup logs --follow
llmsetup restart
llmsetup stop

# Talk to the verified model, or ask it one question.
llmsetup chat
llmsetup chat -m "Write a haiku about VRAM"
```

`--task` is repeatable, for example `--task coding --task agent`. `--context`
sets the desired context in tokens; setup can shorten it to fit memory.
Without an explicit context, coding and agent profiles aim for 32K with a
16K floor, and warn if memory forces them below that floor.
`--yes` skips goal questions. A dry run writes a local report but runs no
verification, so it cannot tell you whether a model works on your hardware.
Use `llmsetup detect` to inspect the scan, or `llmsetup runtimes` to inspect
runtime availability. Neither command installs anything.

Detection covers OS, CPU and core counts, RAM, GPUs and VRAM, and installed
runtimes where they can be detected. A failed probe does not ask you for specs.
It uses conservative assumptions, reports them, and proceeds. If the requested
context cannot fit, planning tries shorter windows while keeping the required
capabilities. Dedicated GPU plans use total VRAM with a 10% reserve. Current
usage by other apps produces a warning and does not downgrade the model.
If no GPU configuration fits, compatible runtimes try system RAM on CPU.

You can also ask for a specific model by name during setup. llmsetup explains
how that model fits your hardware, including the quantization and context it
can hold and the speed to expect, and points out better-fitting alternatives
when the fit is poor. If you keep the model, planning and recovery tune
everything else around it instead of swapping it out. See
[asking for a specific model](docs/AUTOMATIC_SETUP.md#asking-for-a-specific-model).
For a scripted choice, use `llmsetup setup --model qwen3-8b --task chat --yes`.
If that model cannot work after recovery, setup exits with a failure and keeps
the requested model in the report. Alternatives require an explicit choice.

Setup retries transient provisioning failures and server starts, replaces
corrupt managed weights, tries cached weights when downloads are blocked, and
can reduce context, change quantization, offload
to CPU, swap models, or switch runtimes. Recovery has a fixed budget. The report
shows the final settings and repair history. A blocked network, an unsupported
portable runtime, or a requirement no available model can satisfy may still
need your help. The final error gives the next step. See
[automatic setup and recovery](docs/AUTOMATIC_SETUP.md) for the limits.

Setup can download many gigabytes, install the local fallback engine, start a
server, write reports, and register verified models with detected apps.
`--no-register` skips app registration. `--no-repair` disables repairs, extra
provisioning attempts, installation, and runtime fallback. Download transport
retries still apply. A preferred `--runtime` may change during recovery.

After verification, llmsetup stops servers it started. Shared Ollama and
LM Studio daemons keep running. `llmsetup start` runs the verified model again
in the foreground with the exact command setup tested, nothing re-planned;
`llmsetup start --print` prints that command instead. `llmsetup start --detach`
runs it in the background with output in `server.log` under the data directory,
waits until the endpoint answers, and records the process so `llmsetup status`
can report whether it is alive and answering and `llmsetup stop` can end it.
`llmsetup logs` prints the end of that log (`-n` lines, default 50), and
`--follow` keeps printing until the server stops; escape sequences in the log
are shown inert. `llmsetup restart` stops the background server and starts the
verified setup again, detached; `status` shows how long it has been up.
`stop` only signals a process that is still the recorded server, so a reused
process ID is never touched. For Ollama and LM Studio
it shows the endpoint, since their daemons serve the model on demand. Only a
setup whose checks passed can be started this way; after changing anything,
run `llmsetup doctor`. Generated client snippets need a running endpoint.

`llmsetup chat` talks to the model in the terminal: the background server if
one is running, otherwise the last verified setup's endpoint, which has to be
answering (`llmsetup start --detach` first). Replies stream as they are
generated and are printed inert, like the log. `/reset` starts the conversation
over, `/exit` or Ctrl-D leaves, and Ctrl-C stops a reply without ending the
session. `-m "..."` asks one question and exits (`-m -` reads it from stdin,
`--json` prints `{"model", "base_url", "reply"}`), `--system` sets a system
prompt, and `--thinking` also shows the thinking a reasoning model streams
separately. `--base-url` and `--model` point it at any other
OpenAI-compatible server, with `--api-key` for one that needs a token.

## Any model, any setting, any tool, and undo

The catalog is what llmsetup recommends. It does not limit what you can run:

```bash
# Any GGUF repo on Hugging Face; architecture comes from its config or its base model.
llmsetup setup --model hf:bartowski/SomeModel-GGUF

# A derivative of a catalog model, found on Hugging Face: abliterated (or heretic,
# uncensored, or any other name), a fine-tune, a merge.
llmsetup setup --model qwen3-8b --variant abliterated

# Override anything the tuner chose, and pass flags straight to the engine.
llmsetup settings                       # what --set can change
llmsetup setup --set context_tokens=32768 --set flash_attention=true   --set kv_cache_type=q8_0 --engine-arg=--no-warmup
```

`--model hf:` and `--variant` add an ordinary entry to your overlay catalog
(`catalog.d/hub.yaml` in the data directory), so the model is planned, fitted
and verified like any other. A variant keeps its base model's architecture,
which a fine-tune or abliteration does not change, so its memory plan is as
exact as the base's, and it never falls back to the base's own Ollama tag or
weights. Discovery still never *recommends* a decensored model; `--variant` is
how you ask for one. Read the variant's model card: its behaviour and licence
terms are its publisher's. `--set` takes any runtime setting (`llmsetup
settings` lists them with their types) and is applied after tuning and before
verification, so what is verified, saved and restarted is what you asked for;
`--no-repair` keeps your values exactly instead of letting repair adjust one
that fails. `--engine-arg` reaches llama.cpp, vLLM and MLX as given. Setting
`host` beyond loopback is called out, since anyone who can reach the machine
can then use the model.

`llmsetup connect` writes the verified model into another tool's own config
instead of leaving a snippet to paste:

```bash
llmsetup connect --list          # the tools it knows, and the file it edits for each
llmsetup connect opencode --default
llmsetup connect continue
llmsetup connect aider --project
llmsetup connect zed --dry-run   # print the result, change nothing
llmsetup connect claude-code     # the launch command; --project writes .claude/settings.local.json
llmsetup connect codex           # the ~/.codex/config.toml provider and profile
```

It merges an llmsetup entry and keeps everything else in the file, replaces
only what it added before, and leaves a config that has comments alone (with
the snippet printed) unless you pass `--rewrite`, since rewriting JSON or YAML
loses them. Open WebUI and Cursor are set up in their own screens, so `connect`
prints the steps. Claude Code needs a server that speaks the Anthropic Messages
API (llama.cpp from January 2026, Ollama, LM Studio 0.4.1 or later) and its
prompts need at least 32K tokens of context; `connect claude-code` prints the
per-launch command, and writes settings only for one project with `--project`,
since the user-wide file would send every Claude Code session to the local
model. Codex speaks only the Responses API, and some local servers reject the
tools recent releases send, so `connect codex` prints its config with that
caveat rather than writing it.

Everything llmsetup changes outside its own directory is recorded: links in
LM Studio's and Jan's model folders, Jan's `model.json`, Ollama tags, and every
config `connect` writes (with a copy of the file as it was). Undo is one
command:

```bash
llmsetup changes                 # what each run changed
llmsetup undo                    # reverse the newest run
llmsetup undo --run <id>         # or one run, or --all; --dry-run previews
```

Undo removes or restores a file only while it is still what llmsetup left;
a file edited since is reported and kept, unless `--force`. Downloaded weights
stay in the shared store; they are llmsetup's own files, not a change to undo.

## Other commands

```bash
# Reuse a profile with automatic catalog updates disabled.
llmsetup plan --answers examples/coding.yaml --json
llmsetup setup --answers examples/coding.yaml --dry-run --report setup.md

# Plan for CPU inference.
llmsetup plan --cpu-only --task chat --context 8192 --yes

# Check an existing OpenAI-compatible server. Give its root URL, without /v1.
llmsetup verify --base-url http://127.0.0.1:11434 --model qwen3:8b \
  --tools --checks health,completion,tool_calling --yes

# Print the last report. Review it before sharing.
llmsetup report

# Inspect the catalog or add architecture details from a local GGUF.
llmsetup catalog list
llmsetup catalog show qwen3-8b
llmsetup catalog add my-finetune --gguf ./my-finetune-Q4_K_M.gguf --publisher me

# Open the local planning UI. Downloads are handed off to the CLI.
llmsetup ui --no-browser
```

Profiles contain `tasks`, `clients`, `context_tokens`, `priority`,
`concurrent_users`, capability flags such as `needs_tools`, and `offline_only`.
Save interview answers with `llmsetup setup --save-profile profile.yaml`.
Command-line options override matching profile values. Use
`llmsetup <command> --help` for the available flags.

JSON results go to stdout; progress and diagnostics go to stderr. Setup and
verification return a nonzero exit code when a blocking check fails. A dry run
can exit successfully without verifying a model.

## Troubleshooting

Start with `llmsetup detect`, `llmsetup runtimes`, and `llmsetup report`.
Failures print the rerun command with your original options. Runtime failures
also include the engine command or server log when available.

| Symptom | What to do |
| --- | --- |
| Windows NVIDIA setup uses CPU unexpectedly | Run `llmsetup detect` to confirm the GPU. With recovery enabled, setup upgrades a managed CPU build to Vulkan before replanning. A CPU-only system build on PATH takes precedence; install a GPU-capable llama-server or remove that PATH entry. Review GPU layers in the report. |
| VRAM is busy or setup runs out of memory | Close GPU-heavy apps and rerun the printed command. Recovery can reduce context, quantization, or GPU layers, then try CPU. Use `--context 8192` to request a smaller window explicitly. |
| A reasoning model returns an empty first answer | Verification retries with more output-token room for thinking. Streaming recognizes `reasoning_content`, `reasoning`, and `thinking`. If the retry stays empty, the check fails and the report records it. |
| Port is in use or a server exits | llama.cpp selects a free nearby port. A conflicting service on Ollama's port is reported. Stop the conflicting service, inspect the named log, then rerun the printed command. Failed owned servers are stopped before retrying. |
| Download is partial, blocked, or times out | Valid `.part` files remain available for resume. Rerun with the same model and options. Check disk space, network access, model licence, and `HF_TOKEN` for gated models. A trusted mirror can be selected with `HF_ENDPOINT`. |
| Runtime is missing or cannot load a native library | Recovery tries compatible installed engines and the managed fallback. If those fail, install a compatible runtime on PATH and rerun. `--no-repair` disables automatic installation and runtime fallback. |
| Windows console uses cp1252 | Status, interview, and saved-report output replace unsupported characters; subprocess output is decoded as UTF-8 with replacement. Changing the code page is optional. |

External operations are bounded. Ollama import has a 60-minute timeout and
daemon startup has 45 seconds. llama.cpp startup allows 90 seconds plus 12
seconds per GiB of weights. Model downloads use a 60-second socket timeout,
a two-hour total budget per file, and at most four transport retries. Runtime
archive downloads use a 30-second socket timeout and a five-minute total budget.
An in-flight read can take up to the socket timeout before the total budget is
checked again. A timed-out operation fails with a resume or rerun instruction.
These limits apply per attempt; setup recovery can make further bounded attempts.

## Catalogs and downloads

The bundled YAML catalog is extended by files in `$LLMSETUP_HOME/catalog.d/`,
then `LLMSETUP_CATALOG` and explicit `--catalog` files, in that order. Later
entries with the same ID patch individual fields. Architecture values are
marked `measured` or `estimated`; metadata in a downloaded GGUF can correct
estimates.

Automatic catalog refreshes are enabled by default and checked at most daily.
They use this repository's raw catalog URL. If the source is unreachable, the
bundled catalog stays available. A private repository needs an accessible
mirror or local catalog source. Do not put access tokens in URLs or catalog files.

```bash
llmsetup catalog auto-update             # show source, cadence, and status
llmsetup catalog auto-update --disable   # persistent opt-out
llmsetup catalog auto-update --enable    # allow automatic checks again
llmsetup catalog update                 # explicitly fetch now
llmsetup catalog refresh --offline      # compare only local GGUFs
```

`LLMSETUP_AUTO_UPDATE=0` disables automatic catalog refreshes for one process.
`LLMSETUP_CATALOG_URL` selects a trusted HTTP(S) or `file://` source. Disabling
catalog refreshes, including through `offline_only`, does not stop a real
`setup` from downloading weights. Use `plan` or `setup --dry-run` to inspect a
configuration without provisioning.

Model downloads resolve and save a Hugging Face commit revision. You can also
pin revisions in a catalog overlay. Ollama imports the pinned GGUF and keeps
its own copy. See [reproducibility notes](docs/REPRODUCIBILITY.md) for details.

## Local files

Default data lives in `$XDG_DATA_HOME/llmsetup` or `~/.local/share/llmsetup` on
Unix, and `%LOCALAPPDATA%\llmsetup` on Windows. `LLMSETUP_HOME` overrides the
location of profiles, catalog overlays, reports, and generated configs.
`LLMSETUP_MODEL_STORE` overrides the shared model store.

If you used the old name, you can point these variables at your existing
directories. The rename does not move or delete data. Follow the
[migration notes](docs/POSITIONING.md#migrating-an-existing-installation) before
changing an existing installation.

Weights can be large. App registration can replace colliding model links or
generated files, so check the destinations and back up app configuration you
maintain yourself.

## Limits

- Memory and speed estimates depend on the catalog, engine and driver versions,
  free memory, and other running work. A predicted fit can still run out of memory.
- Verification runs targeted smoke tests, not a general model-quality benchmark.
  Passing a tool or JSON check does not prove reliability on every workload.
- Runtime adapters depend on external command-line interfaces that can change.
  Unit tests use fakes; hardware compatibility needs tests with the actual engine.
- Repairs may shorten context, offload work to CPU, or change the model. Read the
  repair journal and final settings before using the result.
- `doctor` detects hardware again and replans for the saved model and profile.
  It does not replay every previous low-level setting.
- Byte counts and trusted transport do not prove model provenance. Review
  publishers' licenses and sources. Model and software licenses are separate.
- The UI is a local loopback development server without authentication. Keep it
  local. Generated shell examples use POSIX syntax unless stated otherwise.

## Development

```bash
python -m pip install -c requirements-ci.txt -e ".[dev]"
python -m pytest
python -m pytest --cov=llmsetup --cov-report=term-missing
ruff check src/ tests/ scripts/
```

Tests use temporary app directories and fake runtimes. UI tests need local
loopback sockets. No GPU or downloaded model is required.

- [Architecture](docs/ARCHITECTURE.md) and [design notes](docs/DESIGN.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)
- [Repository review and priorities](docs/REPOSITORY_REVIEW.md)
- [Project name and migration](docs/POSITIONING.md)

## License

Apache-2.0; see [LICENSE](LICENSE). Downloaded model weights and external
inference engines retain their own licenses.
