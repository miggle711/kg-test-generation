# kg-test-generation

Generates unit tests for buggy/target functions using an LLM (Groq), and compares
a **baseline** (no structural context) against a **KG-augmented** approach that uses
the Knowledge Graph produced by [`repo-kg-construction`](https://github.com/miggle711/repo-kg-construction)
as extra context.

Owners: Mun, Joe (baseline + eval pipeline).

## Why

`repo-kg-construction` builds a structural Knowledge Graph (nodes = files/classes/functions,
edges = calls/inherits/overrides/uses/accesses/etc.) from a repo at a given commit, and can
extract a bounded subgraph around a specific changed function (`TestContextExtractor`) plus
serialize it into LLM-friendly JSON (`LLMSerializer`).

This repo is the consumer. It depends on `kg-construction` as a real package dependency,
drives the KG build and extraction, then owns everything downstream: calling an LLM to
generate tests, parsing the output, running it, and scoring it. That's true for both the
baseline (no KG) and the KG-augmented arm, evaluated against TestGenEval / SWE-bench data.

## Dependency on repo-kg-construction

This repo imports `kg_construction` directly rather than reading loose JSON files off
disk. Here's the public surface it relies on (`src/kg_construction/__init__.py` in that repo):

```python
from kg_construction import (
    RepoKGBuilder, KGQueryEngine, KGValidator,
    TestContextExtractor, TestContext, TestContextValidator,
    LLMSerializer, LLMInput,
)
from kg_construction.pipeline import extract_and_validate, serialize_context
```

`extract_and_validate(instance, depth=2)` returns `(context, report)`. `serialize_context(context)`
turns that `TestContext` into the hierarchical JSON payload that's ready for an LLM prompt.
That's where `repo-kg-construction`'s job ends. Everything from "call an LLM" onward
(previously `GroqTestGenerator`/`AnthropicTestGenerator`, now removed from that repo) is
owned here instead, since it doesn't touch the KG schema at all.

## Pipeline

1. **Load target instance**: pull a bug-fix instance from TestGenEval / SWE-bench (repo,
   commit, target function/file, gold patch, gold tests).
2. **Build context**: depends on the arm.
   - *Baseline*: raw code context only, no KG.
   - *KG-augmented*: `extract_and_validate()` + `serialize_context()` from `kg_construction`.
3. **Generate**: send context to an LLM via Groq (or another provider), get back generated
   test code. We own the generator implementation directly now (it used to live in
   `repo-kg-construction`, but moved here since it's just "call an LLM API" with no
   KG-schema coupling).
4. **Transform output**: turn the LLM's raw output into an actual runnable test file.
5. **Execute**: run the generated test file(s) against the target repo checkout.
6. **Collect metrics**: TBD, see below.

## Open questions

These are unresolved design decisions. Flagging them here so they don't get silently
decided by whoever writes code first.

- **Baseline context strategy**: one-pass (dump the target function/file, maybe the whole
  file, into a single prompt) vs. agentic (give the LLM tool access to search/read the
  checked-out repo iteratively). Affects cost, complexity, and what "baseline" is even
  measuring.
- **KG-augmented context strategy**: does the KG subgraph *replace* raw code context, or
  *augment* it (both raw target code and structural JSON in the same prompt)? Affects how
  directly comparable the two arms are.
- **Execution sandbox**: Docker isolation is assumed, but not confirmed. Do we reuse
  SWE-bench's existing per-instance Docker images (they already pin Python version, deps,
  and repo state at the target commit), or build custom images? Needs deciding before the
  "run generated tests" step can be implemented.
- **Existing benchmark coverage**: need to check whether TestGenEval / SWE-bench datasets
  already ship reference/baseline test-generation implementations we should reuse or
  compare against, rather than re-deriving from scratch.
- **Metrics**: not yet defined. TBD once the team decides what "good" means here (test
  validity, correctness against the gold patch, coverage, or something else).

## Status

Early scaffold. No implementation yet.
