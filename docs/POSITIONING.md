# Positioning and naming proposal

Status: proposal only. No package, executable, repository, or version is renamed.

This repository is `catfish-1234/Open-source-llm`, a Python setup assistant under
Apache-2.0. The unrelated [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit)
is a Rust project under MIT with hardware detection, model recommendations,
CLI/TUI and dashboard interfaces, runtime providers, and benchmarking. There
is substantial feature overlap. Calling this project merely “llmfit” creates
confusion in search, installation instructions, binary resolution, and support.
There is no affiliation implied by the shared name.

Use **“llmfit — local model setup assistant (catfish-1234)”** in the README,
issue reports, and development documentation now. Lead with its workflow:
interview → explained plan → runtime tuning → capability checks → bounded repair
→ application registrations. Describe measured results and limitations rather
than claiming unique features or better performance without comparative data.
Install from this exact repository in a virtual environment so an unrelated
`llmfit` executable on PATH does not silently select the wrong tool.

Before a first public distribution, the recommendation is to choose a distinct
name. **`local-llm-setup` is a candidate, not an approved or availability-checked
name.** Check PyPI normalized names, existing executable names, GitHub search,
and applicable trademark concerns before deciding. The present private repo
and package metadata do not establish registry availability or naming rights.

A later migration proposal should list the chosen distribution name, import
path, CLI name, and data-directory policy separately. Preserve the current
`llmfit` command as a documented compatibility wrapper for a stated period,
keep existing profiles and stores readable, test both entry points, and provide
an explicit migration guide. A rename or publication requires a separate
maintainer decision; this note authorizes neither.
