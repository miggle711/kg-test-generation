"""
execute.py

Runs a generated test file against a target repo checkout and reports
per-test pass/fail results.

v1 decision (issue #1): plain subprocess against a local checkout, not
Docker. repo-kg-construction's RepoManager already does commit-level
source extraction this way (git archive into a temp dir, no checkout of
the actual working tree) -- reused here for isolation between runs at
different commits, though this does NOT sandbox untrusted code execution
the way Docker would. Acceptable for a research prototype running known
benchmark repos (TestGenEval/SWE-bench); would need revisiting before
running genuinely untrusted input.

Uses pytest's built-in --junit-xml reporting (no extra plugin needed) to
get structured per-test results, since Any Pass@1 needs to know whether
AT LEAST ONE test passed, not just the overall subprocess exit code --
one failing test in a file must not be indistinguishable from every test
failing, or from the file failing to even collect.
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, NamedTuple, Optional


class TestCaseResult(NamedTuple):
    name: str
    passed: bool
    # Failure/error message if not passed, else None.
    message: Optional[str]


class ExecutionResult(NamedTuple):
    # True if the file collected successfully (no import/syntax error at
    # collection time) AND ran to completion -- distinct from whether any
    # individual test passed. A collection failure means test_cases is
    # always empty, since pytest never got to run anything.
    collected: bool
    test_cases: List[TestCaseResult]
    stdout: str
    stderr: str
    returncode: int

    @property
    def any_passed(self) -> bool:
        """Any Pass@1: does at least one test in this file execute
        without error. False for a collection failure (nothing ran) or a
        file with zero test cases.
        """
        return any(tc.passed for tc in self.test_cases)

    @property
    def all_passed(self) -> bool:
        """All Pass@1: every test in this file passes. False for a
        collection failure or a file with zero test cases (there is
        nothing to vacuously call "all passing").
        """
        return self.collected and bool(self.test_cases) and all(
            tc.passed for tc in self.test_cases
        )


def run_test_file(
    test_file: Path,
    repo_checkout: Path,
    timeout: int = 60,
    python_executable: str = "python3",
) -> ExecutionResult:
    """Run a generated test file against a repo checkout via subprocess.

    Args:
        test_file: Path to the generated test file (output of
                   parse_output.write_test_file). Must already be inside
                   repo_checkout (or importable from it) for the target
                   module's imports to resolve.
        repo_checkout: Path to the target repo checked out at the
                       instance's base_commit (e.g. via
                       kg_construction.kg.repo_manager.RepoManager).
        timeout: Wall-clock seconds to allow the subprocess before killing
                 it (guards against a generated test that hangs, e.g. an
                 accidental infinite loop or a real network call with no
                 timeout of its own).
        python_executable: Path to the Python interpreter to run pytest
                            with. Defaults to whatever "python3" resolves
                            to on PATH; pass a venv's interpreter (e.g.
                            "<venv>/bin/python") to run against a repo's
                            own installed dependencies rather than
                            whatever's importable in the calling process.

    Returns:
        ExecutionResult with per-test pass/fail and captured output.
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        junit_path = Path(f.name)

    try:
        proc = subprocess.run(
            [python_executable, "-m", "pytest", str(test_file), f"--junit-xml={junit_path}"],
            cwd=repo_checkout,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        test_cases = _parse_junit_xml(junit_path)
        # pytest exits 0 (all passed), 1 (some failed but ran), or other
        # codes for collection errors / internal errors / no tests found.
        # "Collected" means pytest actually attempted to run tests, i.e.
        # we got real per-test results rather than a blanket failure.
        collected = proc.returncode in (0, 1) and bool(test_cases)
        return ExecutionResult(
            collected=collected,
            test_cases=test_cases,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as e:
        return ExecutionResult(
            collected=False,
            test_cases=[],
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + f"\n[execute.py] Timed out after {timeout}s",
            returncode=-1,
        )
    finally:
        junit_path.unlink(missing_ok=True)


def _parse_junit_xml(junit_path: Path) -> List[TestCaseResult]:
    """Parse a pytest --junit-xml report into per-test results.

    Returns an empty list if the file is missing or malformed (e.g. a
    collection error can prevent pytest from writing a report at all).
    """
    if not junit_path.exists():
        return []

    try:
        tree = ET.parse(junit_path)
    except ET.ParseError:
        return []

    results = []
    for testcase in tree.getroot().iter("testcase"):
        name = testcase.get("name", "")
        failure = testcase.find("failure")
        error = testcase.find("error")
        skipped = testcase.find("skipped")

        if skipped is not None:
            continue  # a skipped test is neither a pass nor a failure

        if error is not None and error.get("message") == "collection failure":
            # A synthetic <testcase> pytest emits for the whole file when
            # it can't even be imported (e.g. a bad import statement) --
            # not a real test function, so it must not appear in
            # test_cases at all (see ExecutionResult's collected field
            # for how this case is actually surfaced).
            continue

        if failure is not None or error is not None:
            node = failure if failure is not None else error
            results.append(TestCaseResult(name=name, passed=False, message=node.get("message")))
        else:
            results.append(TestCaseResult(name=name, passed=True, message=None))

    return results
