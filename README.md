# llmfit — local model setup assistant

Interview, plan, tune, verify, and repair local open-weight LLM setups. This
Python project is **catfish-1234/Open-source-llm**, licensed under Apache-2.0.
It is independent of the unrelated [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit)
Rust project. See the [positioning and naming proposal](docs/POSITIONING.md).

llmfit turns your tasks, context window, and hardware into explained model and
runtime recommendations. It can then provision a runtime, test the result, try
bounded repairs, and generate registrations or configuration snippets for local
apps. Predictions are estimates; verification establishes what worked on your
machine for the checks that actually ran.

## Install from this repository

Use Python 3.9 or newer; a currently maintained Python version is recommended.
PyYAML is the only runtime Python dependency. Repository access is required to
clone; a bare `pip install llmfit` is not an installation instruction for this
checkout and may select a different distribution.

```bash
git clone https://github.com/catfish-1234/Open-source-llm.git
cd Open-source-llm
python -m venv .venv
source .venv/bin/activate
python -m pip install .
llmfit --version
llmfit --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell. Use
`python -m pip install -e .` if you want source edits to take effect immediately.

Planning and detection need no inference engine or GPU. For a real setup,
install a supported engine separately and check `llmfit runtimes` for detection
and install hints. Adapters exist for Ollama, llama.cpp, LM Studio, vLLM, MLX,
and TabbyAPI; their hardware support and model formats differ.

## Quickstart: inspect a plan first

```bash
llmfit detect
llmfit runtimes

# Disable automatic catalog network refreshes; use the bundled snapshot.
llmfit catalog auto-update --disable

# No model downloads or server launches.
llmfit plan --vram 24 --ram 64 --gpu-vendor nvidia \
  --task coding --task agent --context 32768 --commercial --yes

# Include concrete tuning decisions, still without provisioning.
llmfit setup --dry-run --vram 24 --ram 64 --gpu-vendor nvidia \
  --task coding --context 32768 --yes
```

Memory overrides describe the machine you want to plan for. They do not reserve
memory or change your hardware. `--yes` skips interview questions; it does not
turn a real setup into a dry run. A dry run writes a local report but runs no
verification and makes no claim that the setup works.

Once the plan looks suitable and your engine is installed:

```bash
llmfit setup --runtime ollama --task coding --context 8192 --yes
```

This step can download many gigabytes, start a server, create a tuned model,
write reports, and register models with detected apps. Use `--no-register` to
skip registrations and `--no-repair` to report failures without attempting
repairs. llmfit stops servers it owns after verification; shared Ollama and LM
Studio daemons remain running. The report includes a command to run the model
again. Generated client snippets require a running endpoint.

## Examples

```bash
# A reproducible profile, with catalog network updates disabled in the profile.
llmfit plan --answers examples/coding.yaml --json
llmfit setup --answers examples/coding.yaml --dry-run --report setup.md

# CPU planning, without an accelerator.
llmfit plan --cpu-only --ram 16 --task chat --context 8192 --yes

# Test an existing OpenAI-compatible server. Use its root URL, without /v1.
llmfit verify --base-url http://127.0.0.1:11434 --model qwen3:8b \
  --tools --checks health,completion,tool_calling --yes

# Reprint the last report; review it before sharing.
llmfit report

# Inspect and extend the catalog with architecture from a local GGUF.
llmfit catalog list
llmfit catalog show qwen3-8b
llmfit catalog add my-finetune --gguf ./my-finetune-Q4_K_M.gguf --publisher me

# A local browser interface for planning; downloads are handed off to the CLI.
llmfit ui --no-browser
```

Profiles use `tasks`, `clients`, `context_tokens`, `priority`, `concurrent_users`,
capability flags such as `needs_tools`, and `offline_only`. Save interview
answers with `llmfit setup --save-profile profile.yaml`. CLI options override
corresponding answers. Run `llmfit <command> --help` for the authoritative flags.
JSON results go to stdout; progress and diagnostics go to stderr. Setup and
verification return a nonzero exit code for blocking failures; a dry run can
exit successfully without verifying a model.

## Catalogs and local files

The bundled YAML catalog is layered with files in `$LLMFIT_HOME/catalog.d/`,
then `LLMFIT_CATALOG` and explicit `--catalog` files. Entries with the same ID
patch individual fields. Architecture provenance is recorded as `measured` or
`estimated`; local GGUF metadata can correct estimates after download.

Automatic catalog refreshes are enabled by default, checked at most daily,
and use this repository's raw catalog URL. An unreachable source leaves the
bundled catalog available. Private repositories require an accessible mirror
or local catalog source; do not embed access tokens in URLs or catalog files.

```bash
llmfit catalog auto-update             # show source, cadence, and status
llmfit catalog auto-update --disable   # persistent opt-out
llmfit catalog auto-update --enable    # allow automatic checks again
llmfit catalog update                 # explicitly fetch now
llmfit catalog refresh --offline      # compare only local GGUFs
```

For one process, `LLMFIT_AUTO_UPDATE=0` also disables automatic refreshes.
`LLMFIT_CATALOG_URL` selects a trusted HTTP(S) or `file://` source. Neither
disabling catalog refresh nor setting `offline_only` prevents a real `setup`
from provisioning weights through an engine; use `plan` or `setup --dry-run`
to inspect without provisioning.

Default data lives in `$XDG_DATA_HOME/llmfit` or `~/.local/share/llmfit` on Unix,
and `%LOCALAPPDATA%\llmfit` on Windows. `LLMFIT_HOME` overrides profiles,
catalog overlays, reports, and generated configs; `LLMFIT_MODEL_STORE` overrides
the shared model store. Weights can be large; Ollama may also keep its own copy.
Registration can replace colliding model links or generated files, so inspect
destinations and keep backups of app configuration you maintain yourself.

## Limitations

- Memory and speed estimates depend on catalog accuracy, engine versions,
  driver behavior, free memory, and concurrent work. They are not OOM guarantees.
- Verification uses targeted smoke tests, not a general model-quality benchmark.
  A passing tool or JSON check does not establish reliability on every workload.
- Runtime adapters depend on external command-line interfaces that can change.
  Unit tests use fakes; real hardware compatibility needs runtime-specific runs.
- Automatic repairs may shorten context, offload work to CPU, or change a model.
  Read the repair journal and final settings before relying on the result.
- `doctor` re-detects and replans for the saved model and profile; it does not
  replay every previous low-level setting verbatim.
- Download byte counts and trusted transport do not establish model provenance.
  Review publisher licenses and sources separately; software and model licenses
  are independent.
- The UI is a loopback development server, without authentication. Keep it local.
  Generated shell examples target POSIX shells unless stated otherwise.

## Development and project docs

```bash
python -m pip install -e ".[dev]" "ruff==0.17.1"
python -m pytest
python -m pytest --cov=llmfit --cov-report=term-missing
ruff check src/ tests/ scripts/
```

Tests use temporary app directories and faked runtimes. UI tests need local
loopback sockets; no GPU or downloaded model is required.

- [Architecture](docs/ARCHITECTURE.md) and [detailed design notes](docs/DESIGN.md)
- [Contributing](CONTRIBUTING.md) and [security reporting](SECURITY.md)
- [Repository review and priorities](docs/REPOSITORY_REVIEW.md)
- [Positioning and proposed naming direction](docs/POSITIONING.md)

## License

Apache-2.0; see [LICENSE](LICENSE). Downloaded model weights and external
inference engines retain their own licenses.
