# Contributing

The most valuable contribution to llmsetup is usually not code. It is a catalog
entry, or a report of a machine where the plan was wrong.

## The one rule

**No entry is better than a guessed one.** Every number llmsetup prints is used to
decide whether a model fits in someone's memory. A plausible-looking guess that
is 20% low turns into an out-of-memory error on a stranger's machine, and they
have no way to know it came from a guess. So:

- Numbers you read out of a config, a model card, or a GGUF header are facts.
  Mark the entry `arch_source: measured`.
- Numbers you inferred from a family resemblance are estimates. Mark them
  `arch_source: estimated`, and llmsetup will re-derive them from the real file
  once the weights are on disk.
- Numbers you cannot find, leave out. An incomplete entry is refused loudly at
  load time, which is the correct outcome.

The same applies to sampling defaults: `src/llmsetup/tuning/publisher.py` requires
a `source` on every entry, because "this feels right" is not a model card.

## Adding a model

Catalog entries live in `src/llmsetup/catalog/data/models.yaml`. The fastest way
to get the architecture right is to let the tool read it:

```bash
llmsetup catalog add my-model --gguf ./my-model-Q4_K_M.gguf --publisher me
```

That reads the local GGUF architecture and writes a user catalog overlay. Review
the overlay from `llmsetup catalog path` before adding an entry to the bundled
catalog, particularly `params_b`, `layers`,
`kv_heads` and `head_dim`, which drive the memory model.

The catalog also refreshes itself on a schedule, so a model that is merely
*new* usually needs no pull request at all. Open one when a model needs
something the automation cannot infer: a task list, a quality tier, a draft-model
pairing, or a publisher's sampling settings.

## Running the checks

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
python -m pip install -c requirements-ci.txt -e ".[dev]"
python -m pytest             # no GPU or downloaded model required
python -m pytest --cov=llmsetup --cov-report=term-missing
ruff check src/ tests/ scripts/
# Type checks run on Python 3.10+ (CI uses 3.12):
python -m pip install -c requirements-ci.txt mypy types-PyYAML
mypy src/llmsetup
```

Every test runs against a temporary `LLMSETUP_HOME`, so nothing touches your real
model store, LM Studio folder, or saved profiles.
UI integration tests bind local loopback sockets; an environment that forbids
all socket creation cannot run those tests. Fake downloads and endpoints at the
I/O boundary; do not make unit tests depend on the internet, installed engines,
real credentials, a user's configuration, or a particular model being popular.

Build and inspect distributable artifacts without publishing:

```bash
python -m pip install "build==1.3.0" "twine==6.2.0"
python -m build
python -m twine check dist/*
```

Test a wheel in a fresh virtual environment from outside the checkout; editable
installs can hide missing catalog YAML or UI assets. Check that the console
entry point, bundled resources, metadata, and license notices are included.

The decision logic is deliberately pure: interview, planning, tuning and repair
are dataclass transformations with no I/O, which is why the whole suite runs in
seconds on a laptop with no accelerator. Keep new logic on that side of the line
where you can. If a function needs a GPU or the network to be tested, it is
usually doing two jobs.

## Reporting a bad plan

This is the most useful bug report there is, and it needs three things:

1. `llmsetup report` from the run (it contains the machine, the plan,
   and every decision with its reason).
2. What actually happened: the error, or the speed you measured.
3. What you expected.

Review reports and logs before sharing. Redact tokens, private URLs, paths, and
server output you do not want to disclose. Follow [SECURITY.md](SECURITY.md) for
suspected vulnerabilities.

## Pull requests

- Explain *why* in the commit message. The what is in the diff.
- A behaviour change needs a test that fails without it.
- Comments should say why the code is the way it is, not restate what it does.
- Start a focused branch from current `main`; keep unrelated changes in separate
  PRs. Preserve command names, profile keys, and ordinary output contracts.
- Include the exact lint and full-suite results, Python version, and any real
  hardware or engine validation. Distinguish predictions from measurements.
- Use full commit SHAs for third-party Actions, read-only CI permissions unless
  a specific job needs writes, and environment variables for untrusted workflow
  inputs. Never interpolate such inputs directly into shell code.
- Do not add credentials or model weights. Retain Apache-2.0 notices and review
  the licenses of any copied or vendored material. Downloaded model licenses are
  independent of this project's software license.
- Publication, naming migrations, and changes to support policy need their own
  explicit maintainer decision. See [the naming and migration notes](docs/POSITIONING.md).
