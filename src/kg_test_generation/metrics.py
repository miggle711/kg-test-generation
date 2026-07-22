"""
metrics.py

Scores execution results and aggregates scores across many instances.

v1 scope: Any Pass@1 and All Pass@1 only, matching what execute.py
currently produces. Coverage@pass and Mutation Score are tracked as
separate follow-ups (issues #9 and #10) -- both need changes to
execute.py itself (coverage instrumentation, a mutant re-run loop) that
don't fit into scoring an already-completed ExecutionResult, so they're
deliberately out of scope here rather than half-implemented.
"""

from typing import Dict, List

from kg_test_generation.execute import ExecutionResult


def score(result: ExecutionResult) -> Dict:
    """Score a single execution result.

    Args:
        result: Output of execute.run_test_file.

    Returns:
        {
            "collected": bool,
            "num_tests": int,        # real test cases (excludes a
                                      # collection failure's synthetic entry)
            "num_passed": int,
            "any_passed": bool,      # Any Pass@1
            "all_passed": bool,      # All Pass@1
        }
    """
    return {
        "collected": result.collected,
        "num_tests": len(result.test_cases),
        "num_passed": sum(1 for tc in result.test_cases if tc.passed),
        "any_passed": result.any_passed,
        "all_passed": result.all_passed,
    }


def aggregate(scores: List[Dict]) -> Dict:
    """Aggregate per-instance scores into summary statistics.

    Takes a flat list of score() outputs -- the caller is responsible for
    filtering/splitting by arm (baseline vs. KG-augmented) or by any other
    grouping before calling this, so this function stays a simple,
    reusable summary over whatever list it's given rather than being
    opinionated about experiment structure.

    Args:
        scores: List of score() outputs.

    Returns:
        {
            "num_instances": int,
            "num_collected": int,
            "num_any_passed": int,
            "num_all_passed": int,
            "any_pass_rate": float,   # num_any_passed / num_instances, 0.0 if empty
            "all_pass_rate": float,   # num_all_passed / num_instances, 0.0 if empty
        }
    """
    num_instances = len(scores)
    num_collected = sum(1 for s in scores if s["collected"])
    num_any_passed = sum(1 for s in scores if s["any_passed"])
    num_all_passed = sum(1 for s in scores if s["all_passed"])

    return {
        "num_instances": num_instances,
        "num_collected": num_collected,
        "num_any_passed": num_any_passed,
        "num_all_passed": num_all_passed,
        "any_pass_rate": num_any_passed / num_instances if num_instances else 0.0,
        "all_pass_rate": num_all_passed / num_instances if num_instances else 0.0,
    }
