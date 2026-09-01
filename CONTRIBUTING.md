# Contributing

The most valuable contribution to llmfit is usually not code. It is a catalog
entry, or a report of a machine where the plan was wrong.

## The one rule

**No entry is better than a guessed one.** Every number llmfit prints is used to
decide whether a model fits in someone's memory. A plausible-looking guess that
is 20% low turns into an out-of-memory error on a stranger's machine, and they
have no way to know it came from a guess. So:

- Numbers you read out of a config, a model card, or a GGUF header are facts.
  Mark the entry `arch_source: measured`.
- Numbers you inferred from a family resemblance are estimates. Mark them
  `arch_source: estimated`, and llmfit will re-derive them from the real file
  once the weights are on disk.
- Numbers you cannot find, leave out. An incomplete entry is refused loudly at
  load time, which is the correct outcome.

The same applies to sampling defaults: `src/llmfit/tuning/publisher.py` requires
a `source` on every entry, because "this feels right" is not a model card.

## Adding a model

Catalog entries live in `src/llmfit/catalog/data/models.yaml`. The fastest way
to get the architecture right is to let the tool read it:

```bash
python scripts/update_catalog.py --add <hf-repo-id>
```

That fetches the config, derives the fields llmfit needs, and marks provenance
honestly. Check the diff before committing, particularly `params_b`, `layers`,
`kv_heads` and `head_dim`, which drive the memory model.

The catalog also refreshes itself on a schedule, so a model that is merely
*new* usually needs no pull request at all. Open one when a model needs
something the automation cannot infer: a task list, a quality tier, a draft-model
pairing, or a publisher's sampling settings.

## Running the checks

```bash
pip install -e ".[dev]"
pytest                       # no GPU and no network required
ruff check src/ tests/ scripts/
```

Every test runs against a temporary `LLMFIT_HOME`, so nothing touches your real
model store, LM Studio folder, or saved profiles.

The decision logic is deliberately pure: interview, planning, tuning and repair
are dataclass transformations with no I/O, which is why the whole suite runs in
seconds on a laptop with no accelerator. Keep new logic on that side of the line
where you can. If a function needs a GPU or the network to be tested, it is
usually doing two jobs.

## Reporting a bad plan

This is the most useful bug report there is, and it needs three things:

1. `llmfit report --markdown` from the run (it contains the machine, the plan,
   and every decision with its reason).
2. What actually happened: the error, or the speed you measured.
3. What you expected.

Do not paste tokens or API keys; the report does not contain any, and llmfit
never asks for one.

## Pull requests

- Explain *why* in the commit message. The what is in the diff.
- A behaviour change needs a test that fails without it.
- Comments should say why the code is the way it is, not restate what it does.
