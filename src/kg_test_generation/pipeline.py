"""
pipeline.py

Top-level orchestration: instance -> context -> generate -> parse -> execute
-> score, for both the baseline and KG-augmented arms.

Not implemented yet -- this ties together context.py, generate.py,
parse_output.py, execute.py, and metrics.py, none of which have real
implementations until the open design questions in the README are
resolved. See https://github.com/miggle711/kg-test-generation/issues/1.
"""


def run_baseline(instance: dict) -> dict:
    """Run the full baseline (no-KG) pipeline for one instance.

    Args:
        instance: Dict with repo, base_commit, patch, code_file, test_file.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    raise NotImplementedError("depends on all pipeline stages, see issue #1")


def run_kg_augmented(instance: dict, depth: int = 2) -> dict:
    """Run the full KG-augmented pipeline for one instance.

    Args:
        instance: Same shape as run_baseline.
        depth: BFS depth passed through to kg_construction's subgraph extraction.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    raise NotImplementedError("depends on all pipeline stages, see issue #1")


def main():
    """CLI entry point (kg-testgen console script)."""
    raise NotImplementedError("no CLI yet, see issue #1")


if __name__ == "__main__":
    main()
