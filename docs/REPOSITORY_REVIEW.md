# Repository review — 2026-10-15

Baseline: `main` at `72efd27707d8b2aeefb9152d981e9e022e5b5d7a`, package
`llmfit` 0.1.0. Scope: all tracked areas, execution and trust boundaries,
planning/tuning/repair design, CLI and UI, catalog data and automation, tests,
packaging, workflows, documentation, license, and project identity.

## Baseline evidence

- Ruff passed for `src/`, `tests/`, and `scripts/`.
- Full suite: **796 passed** on Python 3.12.14, with **74.91% line coverage**.
  Loopback sockets are required. A first restricted run failed 23 socket-based
  tests; all passed when socket access was enabled, without code changes.
- A wheel built and passed `twine check`. This alone did not prove all assets
  work from a non-editable installation.
- PyYAML 6.0.3, the installed runtime dependency, had no known vulnerabilities
  in the pip-audit database at review time. Library metadata allows `PyYAML>=6.0`.
- Targeted scans for common GitHub/AWS token and private-key patterns found no
  matches in tracked baseline files. This is not a comprehensive secret audit.

## Findings and response

| Priority | Area | Finding | Response |
|---|---|---|---|
| P1 | Verification | An unexpected check exception becomes advisory; an empty served-model list passes health; unknown check names are silently ignored. Model listing omits configured auth headers. | Focused verification PR with negative-path and client tests. |
| P1 | Downloads | A short body can become a completed GGUF. Resume accepts mismatched ranges and miscounts totals when Range is ignored. | Validate transport sizes and ranges; preserve partials and test retry behavior. |
| P1 | Filesystem and commands | A missing destination can escape through a symlinked parent. Several generated POSIX commands interpolate catalog values without reliable quoting. | Resolve missing paths and quote shell arguments; test actual script execution. |
| P1 | Repair | The first quantization repair can keep the old weights because the initially provisioned quant was not recorded. | Record the initial quant and test the first repair's provisioning and unchanged-quant reuse. |
| P2 | CLI UX | Malformed answer values raise low-level exceptions; string `false` becomes true; JSON output and partial answers can still prompt. Counts accept negative values. | Validate profiles and positive CLI counts; make automation predictable; add catalog-refresh opt-out flag. |
| P2 | CI and security | Actions use mutable tags; default token permissions are implicit; dispatch strings enter shell source; lint omits scripts; tools float; coverage and built-wheel contents are not gated. | SHA pins, scoped permissions, safe input transport, timeouts, pinned direct tools, coverage, source-security and artifact checks, dependency-update PRs. |
| P2 | Packaging | Dev extra omits documented Ruff. License metadata uses the deprecated table/classifier form. No NOTICE or module entry point is shipped. | Add Ruff, SPDX metadata and explicit license files, NOTICE, source/issue URLs, and `python -m llmfit`. Keep name and version. |
| P2 | Documentation | README starts with an editable install without cloning and buries practical limits. Contributor examples contain unsupported `--add` and `report --markdown` flags. No security policy or naming guidance exists. | Shorten onboarding, preserve design notes, add runnable examples, security policy, contributor checks, and a naming proposal. |

The core architecture is useful: data classes separate much of the sizing,
ranking, tuning, and repair policy from I/O, and the repair budget bounds work.
The strongest existing tests cover these pure decisions. Coverage is uneven:
planning is roughly 92–96% and most tuning 92–100%, while GPU probes are 26%,
the verifier client 30%, downloads 52%, and several runtime adapters 37–68%.
Source-wide percentage alone therefore overstates confidence in real setup.

## License and identity

The repository contains Apache-2.0 terms and a project copyright statement. No
NOTICE existed at baseline. An original Apache-2.0 project need not invent
third-party notices, but any required upstream notices must be retained when
material is copied or vendored. The proposed NOTICE repeats the existing
project attribution and separates model-weight licensing. PyYAML and external
engines retain their own licenses; this review is not a complete provenance or
legal clearance of every historical contribution.

The [unrelated Rust llmfit project](https://github.com/AlexsJones/llmfit) already
occupies the same user-facing name and overlaps in functionality. See the
[positioning proposal](POSITIONING.md): distinguish this repository immediately
and evaluate a distinct name before public distribution. No rename or registry
publication is part of this review.

## Recommended next work, in order

1. Require PR review and passing checks on `main`. Branch protection was disabled
   at review time. Move scheduled catalog updates from direct main commits to
   reviewable PRs; confirm repository settings permit bot PR creation first.
2. Add a small, opt-in real-engine matrix covering one pinned llama.cpp build and
   Ollama, plus periodic Apple/AMD testing. Cover provisioning failures, process
   cleanup, first repair, model identity, split GGUF completeness, and registration
   collisions before expanding the catalog or tuning surface.
3. Track immutable model revisions, ETags, and trusted digests. Byte-count
   verification cannot detect same-size content changes. A cached first shard
   does not prove every split shard is present. Avoid treating catalog arithmetic
   as measured runtime allocation, especially after dynamic-quant selection.
4. Strengthen catalog schema validation for non-finite numbers, quoted booleans,
   invalid IDs, and unknown schema versions. Review the updater's source trust,
   failure behavior, and precedence relative to user overlays. Automated
   architecture discovery does not establish task quality or license rights.
5. Review remaining generated YAML/Modelfile serialization, URL scheme and
   credential handling, report redaction, and file ownership before replacement.
   A path check is not protection against a concurrent local symlink race.
   Bandit excludes B310 because user-selected local catalog URLs are supported;
   that exclusion is not evidence every URL boundary is secure.
6. Add a fully resolved, hash-locked development toolchain or equivalent trusted
   build environment. Exact direct CI pins improve repeatability but leave
   transitive resolution and build isolation floating. Keep library runtime
   ranges compatible and test lower supported bounds separately. Plan Python
   3.9 retirement explicitly; interpreter support does not mean upstream
   security maintenance.
7. Enable an appropriate private security-reporting channel and repository
   secret scanning if available. Run a dedicated history/entropy scan and audit
   development/build dependencies in addition to the runtime dependency set.
8. Decide positioning and a migration plan before any public distribution.
   Evaluate model and runtime claims using reproducible workloads and publish
   measurements only in a separately authorized release effort.

Each implementation branch starts from the recorded main commit and targets
`main` independently. PR descriptions record per-branch lint and full-suite
results. No merge, package publication, deployment, or announcement is included.
