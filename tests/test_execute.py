"""Tests for execute.py: running a generated test file via subprocess and
parsing pytest's --junit-xml report into structured per-test results.

Uses real pytest subprocess runs against synthetic files (no mocking of
the subprocess itself, since correctly invoking and parsing real pytest
output is exactly what's under test) -- matches repo-kg-construction's
own testing style of exercising real behavior over mocked internals.
"""

import time

from kg_test_generation.execute import run_test_file


class TestRunTestFile:
    def test_known_mix_of_pass_fail_error_skip(self, tmp_path):
        (tmp_path / "test_mix.py").write_text(
            "import pytest\n\n"
            "def test_passes():\n"
            "    assert 1 + 1 == 2\n\n"
            "def test_fails():\n"
            "    assert 1 + 1 == 3\n\n"
            "def test_errors():\n"
            '    raise RuntimeError("boom")\n\n'
            '@pytest.mark.skip(reason="skip")\n'
            "def test_skipped():\n"
            "    assert False\n"
        )

        result = run_test_file(tmp_path / "test_mix.py", tmp_path)

        assert result.collected is True
        names = {tc.name: tc.passed for tc in result.test_cases}
        assert names == {
            "test_passes": True,
            "test_fails": False,
            "test_errors": False,
        }
        # A skipped test is neither counted as passing nor failing.
        assert "test_skipped" not in names
        assert result.any_passed is True
        assert result.all_passed is False

    def test_all_tests_pass(self, tmp_path):
        (tmp_path / "test_all_pass.py").write_text(
            "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n"
        )
        result = run_test_file(tmp_path / "test_all_pass.py", tmp_path)

        assert result.collected is True
        assert result.any_passed is True
        assert result.all_passed is True

    def test_collection_failure_from_bad_import(self, tmp_path):
        """The core scenario execute.py was built to distinguish
        correctly: a file that can't even be imported (e.g. the LLM left
        a placeholder import) must not be reported as having any real
        test cases, and must not be counted as any_passed/all_passed.
        """
        (tmp_path / "test_broken.py").write_text(
            "from a_module_that_does_not_exist import Something\n\n"
            "def test_something():\n    assert True\n"
        )
        result = run_test_file(tmp_path / "test_broken.py", tmp_path)

        assert result.collected is False
        assert result.test_cases == []
        assert result.any_passed is False
        assert result.all_passed is False

    def test_syntax_error(self, tmp_path):
        (tmp_path / "test_syntax_error.py").write_text(
            "def test_broken(:\n    pass\n"  # invalid syntax
        )
        result = run_test_file(tmp_path / "test_syntax_error.py", tmp_path)

        assert result.collected is False
        assert result.test_cases == []
        assert result.any_passed is False

    def test_empty_file_with_zero_tests(self, tmp_path):
        (tmp_path / "test_empty.py").write_text("x = 1\n")
        result = run_test_file(tmp_path / "test_empty.py", tmp_path)

        assert result.collected is False
        assert result.test_cases == []
        assert result.any_passed is False
        assert result.all_passed is False

    def test_timeout_kills_a_hanging_test(self, tmp_path):
        (tmp_path / "test_hangs.py").write_text(
            "import time\n\ndef test_hangs_forever():\n    time.sleep(300)\n"
        )

        start = time.time()
        result = run_test_file(tmp_path / "test_hangs.py", tmp_path, timeout=2)
        elapsed = time.time() - start

        assert elapsed < 30  # killed well before the 300s sleep would finish
        assert result.collected is False
        assert result.returncode == -1
        assert "Timed out" in result.stderr

    def test_failure_message_is_captured(self, tmp_path):
        (tmp_path / "test_fail_msg.py").write_text(
            "def test_specific_failure():\n    assert 2 + 2 == 5, 'math is broken'\n"
        )
        result = run_test_file(tmp_path / "test_fail_msg.py", tmp_path)

        assert len(result.test_cases) == 1
        tc = result.test_cases[0]
        assert tc.passed is False
        assert tc.message is not None
