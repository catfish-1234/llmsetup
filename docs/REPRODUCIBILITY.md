# Repeating a setup

Hugging Face model downloads resolve a full commit SHA before fetching files.
The first resolution is saved under `$LLMSETUP_HOME/revisions/` (keyed by endpoint
and repository); subsequent runs reuse it. GGUF shards and vision projectors use
that same commit, and different revisions occupy separate store directories.
Existing local files without revision metadata remain usable, but their origin
cannot be established retroactively.

For reproducibility on another machine, copy the revision records or set full
40-character `gguf_revision`, `hf_revision`, and/or `mlx_revision` values in a
catalog overlay alongside the corresponding repository. Each repository needs
its own SHA; a conversion's commit is different from the base model's commit.
Explicit pins override saved resolutions. To update deliberately, change the
explicit pin or remove the repository's revision record before provisioning.

vLLM receives both model and tokenizer revision arguments; MLX serves a local
snapshot downloaded at that revision. Their saved setup JSON includes the
resolved revision. Ollama imports the shared pinned GGUF rather than pulling a
mutable native tag; this requires a GGUF source and uses additional disk space
for Ollama's copy. Discovery also reads architecture configs at a pinned commit.
There are no dataset downloads in this repository. Catalog discovery itself is
live; use the same catalog snapshot and explicit model/quant for repeat runs.
These pins identify model content, not identical output across hardware or
runtime versions, and do not add cryptographic verification of cached bytes.

# Dependabot check findings (2026-10-16)

- [#13](https://github.com/catfish-1234/llmsetup/pull/13), job
  [116804628253](https://github.com/catfish-1234/llmsetup/actions/runs/38848611472/job/116804628253):
  Python 3.9 could not install coverage 7.16.2 (`Requires-Python >=3.10`).
  This was dependency resolution, not the coverage floor. Keep 7.10.7 on 3.9
  and use 7.16.2 on newer Python versions.
- [#16](https://github.com/catfish-1234/llmsetup/pull/16), job
  [116717399449](https://github.com/catfish-1234/llmsetup/actions/runs/38821036719/job/116717399449):
  Python 3.9 could not install the isolated build requirement setuptools >=84
  (84 requires Python >=3.10). Retain the existing security floor on 3.9;
  require >=84 on newer interpreters. Both fixes belong on main; the Dependabot
  branches are untouched and will need their changes reconciled with these markers.
- [#10](https://github.com/catfish-1234/llmsetup/pull/10) keeps checkout
  pinned to `3d3c42e5aac5ba805825da76410c181273ba90b1` in both workflows.
- [#11](https://github.com/catfish-1234/llmsetup/pull/11) keeps setup-python
  pinned to `5fda3b95a4ea91299a34e894583c3862153e4b97` in both workflows.
  Both retain the full-SHA policy; neither was merged.
