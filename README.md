# llmsetup

llmsetup sets up an open-weight LLM on your machine. It scans your hardware and
installed runtimes first, then asks what you want to do, such as coding, chat,
or running agents. You never need to enter hardware specs. It picks a model,
downloads the weights, starts a server, tests the capabilities you need, and
registers the working setup with your local apps.

When something fails, it tries repairs and fallback options automatically.
It only gives you a manual step after those options have failed.

This is the Python project at
[catfish-1234/Open-source-llm](https://github.com/catfish-1234/Open-source-llm),
licensed under Apache-2.0. It was renamed from llmfit to avoid confusion with the
unrelated [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) Rust project.
See [the name change and migration notes](docs/POSITIONING.md).

## Install

Use Python 3.9 or newer, preferably a version that still receives security
updates. PyYAML is the only runtime Python dependency. Install from this
repository; the renamed package has not been published to PyPI. You need
repository access to clone it.

```bash
git clone https://github.com/catfish-1234/Open-source-llm.git
cd Open-source-llm
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
including llama.cpp on CPU. As a last resort, it can install a pinned, checksum-verified llama.cpp build in
its own data directory and run on CPU. It does not need administrator access.

## Plan or set up a model

```bash
# Scan the machine and ask about your goals.
llmsetup setup

# Skip the interview when you already know what you want.
llmsetup plan --task coding --task agent --context 32768 --commercial --yes

# Inspect tuning without installing an engine, downloading weights, or starting a server.
llmsetup setup --dry-run --task coding --context 32768 --yes

# Download, start, verify, and repair automatically.
llmsetup setup --task coding --context 8192 --yes
```

`--yes` skips goal questions. A dry run writes a local report but runs no
verification, so it cannot tell you whether a model works on your hardware.
Use `llmsetup detect` to inspect the scan, or `llmsetup runtimes` to inspect
runtime availability. Neither command installs anything.

Detection covers OS, CPU and core counts, RAM, GPUs and VRAM, and installed
runtimes where they can be detected. A failed probe does not ask you for specs.
It uses conservative assumptions, reports them, and proceeds. If the requested
context cannot fit, planning tries shorter windows while keeping the required
capabilities.

You can also ask for a specific model by name during setup. llmsetup explains
how that model fits your hardware, including the quantization and context it
can hold and the speed to expect, and points out better-fitting alternatives
when the fit is poor. If you keep the model, planning and recovery tune
everything else around it instead of swapping it out. See
[asking for a specific model](docs/AUTOMATIC_SETUP.md#asking-for-a-specific-model).

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
LM Studio daemons keep running. The report includes a command to start the model
again; generated client snippets need a running endpoint.

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
