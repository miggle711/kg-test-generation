"""
metrics.py

Pipeline stage 6 (Collect metrics): scores execution results.

Not implemented yet -- blocked on the metrics decision. See
https://github.com/miggle711/kg-test-generation/issues/1.
"""

from kg_test_generation.execute import ExecutionResult


def score(result: ExecutionResult) -> dict:
    """Score an execution result.

    Args:
        result: Output of execute.run_test_file.

    Returns:
        Metrics dict, shape TBD (see module docstring).
    """
    raise NotImplementedError("blocked on metrics decision, see issue #1")
