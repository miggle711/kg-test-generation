"""Tests for parse_output.py: extracting runnable test code from an LLM's
raw text response and writing it to disk.

The system prompt (generate.system_prompt) asks the model to output only
code, but real responses don't always comply -- these cover the realistic
shapes a response can take: clean code, a single fenced block (with or
without a language tag), prose wrapped around a fence, multiple fenced
blocks, and a mix of valid and invalid blocks.
"""

import ast

import pytest

from kg_test_generation.parse_output import extract_test_code, write_test_file


class TestExtractTestCode:
    def test_clean_code_with_no_fences(self):
        raw = "def test_foo():\n    assert foo() == 1\n"
        assert extract_test_code(raw) == "def test_foo():\n    assert foo() == 1"

    def test_single_python_tagged_fence(self):
        raw = "Here are the tests:\n```python\ndef test_foo():\n    assert True\n```\n"
        code = extract_test_code(raw)
        assert code == "def test_foo():\n    assert True"

    def test_bare_fence_no_language_tag(self):
        raw = "```\ndef test_foo():\n    assert True\n```"
        code = extract_test_code(raw)
        assert code == "def test_foo():\n    assert True"

    def test_prose_before_and_after_fence(self):
        raw = (
            "Sure! Here's the test:\n\n"
            "```python\ndef test_foo():\n    assert True\n```\n\n"
            "This covers the happy path."
        )
        code = extract_test_code(raw)
        assert code == "def test_foo():\n    assert True"
        assert "Sure!" not in code
        assert "covers the happy path" not in code

    def test_multiple_valid_fenced_blocks_are_concatenated(self):
        raw = (
            "```python\ndef test_a():\n    assert True\n```\n\n"
            "And another:\n\n"
            "```python\ndef test_b():\n    assert True\n```"
        )
        code = extract_test_code(raw)
        assert "def test_a():" in code
        assert "def test_b():" in code
        assert ast.parse(code)  # combined result must itself be valid

    def test_invalid_block_is_dropped_valid_block_is_kept(self):
        """A block that isn't valid Python (e.g. explanatory prose that
        got accidentally fenced) must not corrupt the whole extraction --
        only the valid block should survive.
        """
        raw = (
            "```python\ndef test_a():\n    assert True\n```\n\n"
            "```\nThis is not code, just explanation text.\n```"
        )
        code = extract_test_code(raw)
        assert code == "def test_a():\n    assert True"

    def test_realistic_multi_function_response_with_imports_and_docstrings(self):
        raw = (
            "Here are comprehensive tests:\n\n"
            "```python\n"
            "import pytest\n"
            "from unittest.mock import MagicMock\n\n"
            "def test_happy_path():\n"
            '    """Docstring."""\n'
            "    assert True\n\n\n"
            "def test_error_case():\n"
            "    with pytest.raises(ValueError):\n"
            "        raise ValueError\n"
            "```\n\n"
            "These cover the main scenarios."
        )
        code = extract_test_code(raw)
        assert "import pytest" in code
        assert "def test_happy_path" in code
        assert "def test_error_case" in code
        ast.parse(code)  # must be valid as a whole file

    def test_raises_when_nothing_extractable(self):
        with pytest.raises(ValueError, match="Could not extract valid Python"):
            extract_test_code("I cannot help with that request.")

    def test_raises_when_only_invalid_fenced_blocks_and_invalid_whole_response(self):
        raw = "Sorry, I can't do that.\n```\nneither can this parse as code\n```"
        with pytest.raises(ValueError, match="Could not extract valid Python"):
            extract_test_code(raw)


class TestWriteTestFile:
    def test_writes_content_to_the_given_path(self, tmp_path):
        dest = tmp_path / "test_foo.py"
        result = write_test_file("def test_foo():\n    assert True\n", dest)

        assert result == dest
        assert dest.read_text() == "def test_foo():\n    assert True\n"

    def test_creates_missing_parent_directories(self, tmp_path):
        dest = tmp_path / "nested" / "dir" / "test_foo.py"
        write_test_file("def test_foo():\n    pass\n", dest)

        assert dest.exists()

    def test_overwrites_existing_file(self, tmp_path):
        dest = tmp_path / "test_foo.py"
        dest.write_text("old content")

        write_test_file("def test_foo():\n    pass\n", dest)

        assert "old content" not in dest.read_text()
