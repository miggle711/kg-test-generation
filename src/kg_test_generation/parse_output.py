"""
parse_output.py

Turns the LLM's raw generated text into an actual runnable test file on
disk.

The system prompt (generate.system_prompt) instructs the model to "Output
ONLY the test code, no explanations," but real output doesn't always obey
that: it may come back as clean code, wrapped in a single fenced code
block, wrapped with surrounding prose, or split across multiple fenced
blocks. extract_test_code handles all of these.
"""

import ast
import re
from pathlib import Path


_FENCE_PATTERN = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_test_code(raw_output: str) -> str:
    """Extract runnable Python test code from an LLM's raw text response.

    Strategy:
      1. If the response has one or more fenced code blocks, use those --
         concatenated in order, separated by a blank line -- keeping only
         blocks that individually parse as valid Python. A block that
         doesn't parse (explanatory prose, a partial snippet) is dropped
         rather than corrupting the rest of the file; the model's own
         "no explanations" instruction failing for one block shouldn't
         break collection of an otherwise-valid file.
      2. If there are no fenced blocks at all, treat the whole response as
         code (the "no explanations" instruction was followed exactly).

    Args:
        raw_output: Raw string returned by generate.GroqTestGenerator.generate().

    Returns:
        Clean, runnable Python source.

    Raises:
        ValueError: If no valid Python code could be extracted at all
                    (every fenced block failed to parse, and the whole
                    response also isn't valid Python).
    """
    fenced_blocks = _FENCE_PATTERN.findall(raw_output)

    if fenced_blocks:
        valid_blocks = [block for block in fenced_blocks if _is_valid_python(block)]
        if valid_blocks:
            return "\n\n".join(block.strip() for block in valid_blocks)
        # Every fenced block failed to parse -- fall through and try the
        # whole raw response as a last resort, in case the fences were
        # incidental (e.g. a code comment containing ``` text).

    if _is_valid_python(raw_output):
        return raw_output.strip()

    raise ValueError(
        "Could not extract valid Python test code from the model's response "
        "(no fenced code block parsed as valid Python, and the full response "
        "isn't valid Python either)"
    )


def write_test_file(test_code: str, dest: Path) -> Path:
    """Write generated test code to a test file at dest.

    Args:
        test_code: Output of extract_test_code (already validated as
                   parseable Python).
        dest: Path to write the test file to. Parent directories are
              created if needed.

    Returns:
        The path written to (same as dest).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(test_code, encoding="utf-8")
    return dest


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
