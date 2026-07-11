"""
execute.py

Pipeline stage 5 (Execute): runs generated test file(s) against the target
repo checkout and reports pass/fail results.

Not implemented yet -- blocked on the execution sandbox decision (Docker
vs. plain subprocess against a local checkout). See
https://github.com/miggle711/kg-test-generation/issues/1.
"""

from pathlib import Path
from typing import NamedTuple


class ExecutionResult(NamedTuple):
    passed: bool
    stdout: str
    stderr: str


def run_test_file(test_file: Path, repo_checkout: Path) -> ExecutionResult:
    """Run a generated test file against a repo checkout.

    Args:
        test_file: Path to the generated test file (output of
                   parse_output.write_test_file).
        repo_checkout: Path to the target repo checked out at the
                       instance's base_commit.

    Returns:
        ExecutionResult with pass/fail and captured output.
    """
    raise NotImplementedError("blocked on execution sandbox decision, see issue #1")
