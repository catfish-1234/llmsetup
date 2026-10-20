# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Context now follows how the model will be used instead of a fixed 8K. Coding
  and agent setups aim for 32K and keep a 16K floor, retrieval keeps a 16K
  floor, chat keeps a small default. An explicit `--context` is still
  respected. The planner only goes below the floor when nothing larger fits,
  and then says so plainly, along with the context, KV cache type, model and
  offload it chose. Ollama's docs recommend at least 64K for agent tools, so
  the floor is a minimum, not the goal.
- New asymmetric KV cache option for llama.cpp (keys q8_0, values q4_0). Public
  perplexity measurements show keys are the sensitive half and values are not,
  so this saves about 40% of the cache where symmetric q4_0 breaks the model.
  For coding, agent and retrieval profiles the planner skips symmetric caches
  below q8_0 and shortens context instead.

### Fixed

- The streaming check read only `reasoning_content` and `thinking`, but
  Ollama's OpenAI endpoint streams thinking under `reasoning`, so a working
  reasoning model counted zero chunks and failed. It now counts all three, the
  probe budget includes the model's thinking budget, and a real failure reports
  which stream fields were seen.

- The hardware budget is sized from the card's total VRAM with a 10% safety
  margin instead of the momentarily free VRAM. Free VRAM is usage by other
  apps, not capacity, and sizing from it downgraded the model choice whenever
  something else happened to be running. A busy GPU now produces a warning
  suggesting GPU-heavy apps be closed for best speed.
- MiniCPM5-2B is now cataloged as a reasoning model, which it is: its
  thinking block consumed the small probe budget and the old empty-response
  verdict sent recovery across both engines for nothing. The stability and
  completion probes no longer trust the catalog card over the observed
  response shape, and the stability failure detail includes the truncated raw
  server response so the next misbehaving model can be diagnosed without a
  rerun.
- The MiniCPM5-2B-DSpark entry was removed: its GGUF repo publishes a single
  unquantized file, so the planner picked it, provisioning failed, and the
  pipeline silently switched models mid-run. Auto-discovery now skips any
  model whose GGUF repo publishes none of the recommended quantizations and
  trims entries to the quants actually published, and the pipeline does not
  retry a model whose provisioning already failed in the same run.
- Subprocess reads are decoded as UTF-8 with replacement instead of the
  platform default, ending the cp1252 UnicodeDecodeError crashes during model
  creation on Windows.
- The tuner no longer prints two contradictory batch/ubatch lines when it
  grows the batch into leftover headroom.

- Verification no longer fails a working reasoning model: a small probe budget
  that the thinking block consumes is reported as such instead of an empty
  response, and the completion probe retries with room for the thinking budget
  before failing. This had sent recovery through every installed runtime and a
  CPU fallback on real hardware.
- The stability probe now handles servers that report a thinking-truncated
  reply with no reasoning field, no usage, and a plain stop finish reason: it
  retries once with room for the thinking budget, and failures report the
  response shape instead of a bare empty-response verdict.
- Streaming verification counts thinking deltas, so reasoning models that
  stream their thinking before visible text no longer fail the streaming
  check.
- Windows machines with an NVIDIA GPU now get the Vulkan llama.cpp build
  instead of the CPU build, and the planner keeps the GPU when the managed
  runtime can use it instead of replanning against RAM with CPU-only
  estimates. Recovery installs the GPU build before a llama.cpp attempt on
  machines that previously fell back to slow CPU inference.
- Recovery stops switching engines when the same model-level failure follows
  the same model onto a second engine, instead of trying every remaining
  engine and CPU path.
- The CPU fallback keeps the model already fetched when it fits in RAM rather
  than downloading a larger model for a slower engine.
- The report names the real trigger for runtime and CPU fallbacks, and runtime
  choices made by automatic recovery are labelled as recovery, not as an
  explicit user request.
- A plan with zero GPU layers reads as CPU-only instead of partial offload.

### Added

- Catalog validation on load, with errors identifying the file, model, and bad
  field. ([#17])
- Offline integration tests covering CLI setup, verification, saved reports,
  cleanup, failure handling, and dry runs. ([#17])
- A mypy gate for the library, with fixes for existing type errors. ([#17])
- A `python -m llmsetup` entry point and complete developer installation tooling.
  ([#5])

### Changed

- Setup scans hardware and installed runtimes before asking about goals, with no
  hardware or disk-size questions. Uncertain probes use conservative defaults
  and record their assumptions.
- Planning shortens an oversized context automatically. Setup retries provisioning
  and server failures, tries compatible installed engines and cached weights,
  and can install a
  pinned, checksum-verified user-local llama.cpp CPU fallback. Manual instructions
  appear after automatic recovery is exhausted.

- Renamed the project from `llmfit` to `llmsetup`, including the distribution,
  Python module, CLI, environment variables, and generated names. The version
  remains 0.1.0 and the GitHub repository URL is unchanged. See the
  [migration notes](docs/POSITIONING.md#migrating-an-existing-installation).

- Model downloads use pinned Hugging Face commit revisions, persisted first-use
  resolutions, and optional explicit catalog pins. Shards and vision projectors
  share a revision; vLLM pins its tokenizer, MLX serves a local snapshot, and
  Ollama imports the pinned GGUF. ([#17])
- Python version markers retain compatible coverage and setuptools requirements
  on Python 3.9 while using newer versions on Python 3.10 and later. ([#17])
- Onboarding now includes practical installation, planning, dry-run, and profile
  examples, alongside contributor, security, reproducibility, and naming guidance.
  Package artifacts include modern license metadata and notices. ([#5], [#9], [#17])
- CI uses full Action SHA pins, restricted permissions, safer workflow inputs,
  Python 3.9–3.14 tests, a coverage floor, dependency audits, and installed-wheel
  checks. Actions and build/security tools have also been updated.
  ([#7], [#10], [#11], [#12], [#14], [#15])

### Fixed

- Corrupt managed GGUF weights are quarantined and fetched again; invalid resume
  responses restart cleanly, and incomplete split models fetch missing shards.
- Dead server processes fail promptly with log details and cleanup. Model swaps
  keep reduced context, unclassified blocking failures remain failures, and app
  registration waits for successful verification.

- Quantization repairs fetch matching weights on the first relaunch; repairs
  that only change settings reuse existing weights. ([#8])
- CLI profiles reject malformed values with actionable errors. JSON output and
  answers-file planning avoid interactive prompts, and `--offline` skips automatic
  catalog refreshes. ([#6])
- Truncated downloads and mismatched resumed ranges cannot enter the completed
  model store; retries preserve valid partial data. ([#3])
- Verification errors block success, check names are validated before requests,
  and model listing requires a matching served model and forwards authentication.
  ([#2])

### Security

- Resolve symlink boundaries for missing destination files and quote generated
  shell commands so catalog values remain literal arguments. ([#4])

## [0.1.0] - 2026-08-29

### Added

- An interview-driven setup assistant for local open-weight LLMs, recommending
  models and runtime settings from hardware, tasks, and capability requirements.
- Runtime adapters for Ollama, llama.cpp, LM Studio, vLLM, MLX, and TabbyAPI.
- Provisioning that downloads model weights and starts inference servers.
- Capability smoke-test verification with bounded, journaled automatic repair.
- Model registration and configuration exports for local LLM applications.
- A YAML model catalog with user extensions and field-level overrides.
- A `doctor` command to re-test and repair an existing setup.
- A local browser UI served over loopback.

[Unreleased]: https://github.com/catfish-1234/Open-source-llm/compare/9937189e39c46810d35a4b531ddf7964ce5dff35...HEAD
[0.1.0]: https://github.com/catfish-1234/Open-source-llm/pull/1
[#2]: https://github.com/catfish-1234/Open-source-llm/pull/2
[#3]: https://github.com/catfish-1234/Open-source-llm/pull/3
[#4]: https://github.com/catfish-1234/Open-source-llm/pull/4
[#5]: https://github.com/catfish-1234/Open-source-llm/pull/5
[#6]: https://github.com/catfish-1234/Open-source-llm/pull/6
[#7]: https://github.com/catfish-1234/Open-source-llm/pull/7
[#8]: https://github.com/catfish-1234/Open-source-llm/pull/8
[#9]: https://github.com/catfish-1234/Open-source-llm/pull/9
[#10]: https://github.com/catfish-1234/Open-source-llm/pull/10
[#11]: https://github.com/catfish-1234/Open-source-llm/pull/11
[#12]: https://github.com/catfish-1234/Open-source-llm/pull/12
[#14]: https://github.com/catfish-1234/Open-source-llm/pull/14
[#15]: https://github.com/catfish-1234/Open-source-llm/pull/15
[#17]: https://github.com/catfish-1234/Open-source-llm/pull/17
