"""
context.py

Build context: produces the prompt context for either
the baseline test (no KG) or the KG-augmented test, for a given instance.

Not implemented yet: still blocked on open design questions in the README:
  - Baseline context strategy: one-pass vs. agentic.
  - KG-augmented context strategy: replace raw code, or augment it.
See https://github.com/miggle711/kg-test-generation/issues/1.
"""


def build_baseline_context(instance: any) -> dict:
    """Build prompt context for the baseline (no-KG) test.

    Args:
        instance: Any object that can be used to build a baseline context. The exact shape is TBD (see module docstring).

    Returns:
        Provider-agnostic context payload, shape TBD (see module docstring).
    """
    raise NotImplementedError("blocked on baseline context strategy decision, see issue #1")


def build_kg_augmented_context(instance: any, depth: int = 2) -> dict:
    """Build prompt context for the KG-augmented test.

    Args:
        instance: Same shape as build_baseline_context.
        depth: BFS depth passed through to kg_construction's subgraph extraction.

    Returns:
        Provider-agnostic context payload, shape TBD (see module docstring).
    """
    raise NotImplementedError("blocked on KG-augmented context strategy decision, see issue #1")
