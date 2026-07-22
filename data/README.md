# Benchmark dataset

22 real commits from [psf/requests](https://github.com/psf/requests), used to
compare the baseline (no-KG) and KG-augmented test-generation arms (see
`pipeline.run_baseline` / `pipeline.run_kg_augmented`).

## How this was curated

Sampled every 40th non-merge commit touching `requests/*.py` across the
repo's full history (2011-2023), then filtered to commits touching exactly
one file (to keep `code_file` well-defined), keeping the natural spread of
diff sizes and shapes that produced -- **not** filtered down to only
commits already known to produce a clean single-function match. See the
earlier n=4 benchmark (issue #16) for why this matters: that batch was
cherry-picked for cleanliness, made prior KG-serialization bugs (#6, #14)
easy to miss, and wasn't representative of how real patches actually look.

Each instance was verified against `PatchParser.extract_changed_functions`
and labeled by what it produces (see each `data/instances/*.json`'s `note`
field):

- **11 clean single-function** commits (one unambiguous function changed)
- **5 multi-function/multi-method** commits (e.g. a class plus multiple
  methods changed in one commit, or several free functions edited together)
- **1 ambiguous class-vs-method** case (`resolve_redirects_2013`) --
  deliberately included because it has the same shape as the real
  regression behind issue #14 (a method whose enclosing class is also
  swept into the diff hunk)
- **2 NONE-resolution** commits (the patch doesn't map to any function --
  e.g. a module-level constant change, a version string bump) --
  included because real-world patches sometimes look like this, and the
  pipeline's behavior on them (a clean `ValueError` from
  `resolve_target_function`, not a crash) is itself worth having data on

## Layout

- `instances/<name>.json` -- one file per instance: `name`, `repo`,
  `base_commit` (the commit *before* the change), `patch_file` (relative
  path to the diff), `code_file`, `test_file`, `note` (why this instance
  was picked / what it represents).
- `patches/<name>.diff` -- the actual unified diff, generated via
  `git diff -U<N> <parent> <commit> -- <code_file>` with enough context
  (`-U`) for `PatchParser` to see the enclosing function/class's `def`/
  `class` line. Context width varies per instance (10-20 lines) --
  `status_codes_init_2018` specifically needed `-U20` to resolve at all,
  which is itself a small data point about how sensitive diff-based
  function resolution is to context width on real commits.

## Loading

```python
from kg_test_generation.dataset import load_instance, load_all_instances

instance = load_instance("send_2012")
instances = load_all_instances()  # all 22, sorted by name
```

Each loaded instance is the `Dict` shape `run_baseline`/`run_kg_augmented`
expect directly (`repo`, `base_commit`, `patch`, `code_file`, `test_file`).

## Caveats

- Single-repo (psf/requests only) -- not representative of cross-project
  variation in coding style, dependency era, or codebase size. See #16 and
  #49 for related open questions this dataset is meant to help answer with
  a larger, still-single-repo sample first.
- LLM generation is non-deterministic (temperature 0.7) -- rerunning the
  same instance can produce meaningfully different pass/fail results
  across runs. A single run over this dataset is one data point, not a
  final verdict; see issue #16 for the caveats that applied to the
  earlier, smaller batch and still apply here at larger scale.
