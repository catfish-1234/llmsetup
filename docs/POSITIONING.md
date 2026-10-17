# Project name and migration

The project is now **llmsetup**, a Python setup assistant for local open-weight
LLMs under Apache-2.0. The distribution name, Python import, and CLI command are
all `llmsetup`. The version remains `0.1.0`.

The source repository stays at
[catfish-1234/Open-source-llm](https://github.com/catfish-1234/Open-source-llm).
The GitHub repository is not being renamed. Install from that repository;
this rename does not publish a package, create a tag, or create a release.

## Why the name changed

The former name, `llmfit`, collided with the unrelated
[AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) Rust project under MIT.
That project also covers hardware detection, model recommendations, runtime
providers, and benchmarking. The shared name caused confusion in search,
installation instructions, executable lookup, and support. There is no affiliation.

A registry check on 2026-10-16 found [PyPI `llmfit`](https://pypi.org/project/llmfit/)
occupied by version 1.1.17, whose metadata links to AlexsJones/llmfit.
`pip install llmfit` installs that distribution, not this repository.

The maintainer confirmed that `llmsetup` was available on PyPI and npm on
2026-10-16 and selected it for this project. This records availability at the
time of the decision, not publication or a reservation. This repository remains
a Python project; no npm package is added.

## Migrating an existing installation

1. Install this checkout in a fresh virtual environment using the README's
   instructions. Run `llmsetup` or `python -m llmsetup`. The old command and
   import name are not installed as compatibility aliases.
2. Update scripts and imports from `llmfit` to `llmsetup`. Environment variables
   now use the `LLMSETUP_` prefix instead of `LLMFIT_`, including `HOME`,
   `MODEL_STORE`, `CATALOG`, `CATALOG_URL`, and `AUTO_UPDATE` settings.
3. Default app data now uses a directory named `llmsetup`: `$XDG_DATA_HOME/llmsetup`
   or `~/.local/share/llmsetup` on Unix, and `%LOCALAPPDATA%\llmsetup` on Windows.
   To reuse an existing data directory without moving it, set `LLMSETUP_HOME` to
   its path. If you used a separate model store, set `LLMSETUP_MODEL_STORE` to
   that path too. Profiles, catalog overlays, reports, and saved revision records
   keep the same formats. Nothing is moved or deleted automatically.
4. Regenerate launch scripts and app registrations when you need the new names.
   Generated model aliases now use the `llmsetup-` prefix. Existing registrations
   and model files are left in place; check destinations before replacing them.

The workflow remains interview, planning, runtime tuning, capability checks,
bounded repair, and app registration. Memory and speed figures are estimates
until tested with an actual engine. Report measured results and the checks that
ran, without implying that a smoke test establishes general model quality.
