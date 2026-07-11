"""
parse_output.py

Pipeline stage 4 (Transform output): turns the LLM's raw generated text
into an actual runnable test file on disk.

Not implemented yet -- straightforward once stages 1-3 exist (extract code
block(s) from the LLM's response, write to a test_*.py file), but there's
no real generated output to parse against until generate.py works.
"""

from pathlib import Path


def extract_test_code(raw_output: str) -> str:
    """Extract runnable Python test code from an LLM's raw text response.

    Args:
        raw_output: Raw string returned by generate.GroqTestGenerator.generate().

    Returns:
        Clean, runnable Python source (e.g. with markdown code fences stripped).
    """
    raise NotImplementedError


def write_test_file(test_code: str, dest: Path) -> Path:
    """Write generated test code to a test file at dest.

    Args:
        test_code: Output of extract_test_code.
        dest: Path to write the test file to.

    Returns:
        The path written to (same as dest).
    """
    raise NotImplementedError
