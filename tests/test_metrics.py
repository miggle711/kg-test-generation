"""Tests for metrics.py: per-instance scoring (Any/All Pass@1) and
aggregation across many instances.

score() is tested against real execute.run_test_file results (real
subprocess runs), matching test_execute.py's style, since the whole
point is confirming score() correctly summarizes what execute.py
actually produces -- not a synthetic ExecutionResult shape that might
drift from reality.
"""

from kg_test_generation.execute import run_test_file
from kg_test_generation.metrics import aggregate, score


class TestScore:
    def test_mixed_pass_fail(self, tmp_path):
        (tmp_path / "test_mix.py").write_text(
            "def test_a():\n    assert True\n\ndef test_b():\n    assert False\n"
        )
        result = run_test_file(tmp_path / "test_mix.py", tmp_path)
        s = score(result)

        assert s == {
            "collected": True,
            "num_tests": 2,
            "num_passed": 1,
            "any_passed": True,
            "all_passed": False,
        }

    def test_all_pass(self, tmp_path):
        (tmp_path / "test_all.py").write_text("def test_a():\n    assert True\n")
        result = run_test_file(tmp_path / "test_all.py", tmp_path)
        s = score(result)

        assert s["any_passed"] is True
        assert s["all_passed"] is True
        assert s["num_passed"] == 1

    def test_collection_failure(self, tmp_path):
        (tmp_path / "test_broken.py").write_text(
            "from nowhere import Thing\n\ndef test_a():\n    assert True\n"
        )
        result = run_test_file(tmp_path / "test_broken.py", tmp_path)
        s = score(result)

        assert s["collected"] is False
        assert s["num_tests"] == 0
        assert s["any_passed"] is False
        assert s["all_passed"] is False


class TestAggregate:
    def test_mixed_scores(self):
        scores = [
            {"collected": True, "num_tests": 2, "num_passed": 1, "any_passed": True, "all_passed": False},
            {"collected": True, "num_tests": 1, "num_passed": 1, "any_passed": True, "all_passed": True},
            {"collected": False, "num_tests": 0, "num_passed": 0, "any_passed": False, "all_passed": False},
        ]
        result = aggregate(scores)

        assert result["num_instances"] == 3
        assert result["num_collected"] == 2
        assert result["num_any_passed"] == 2
        assert result["num_all_passed"] == 1
        assert result["any_pass_rate"] == 2 / 3
        assert result["all_pass_rate"] == 1 / 3

    def test_empty_list_does_not_raise(self):
        result = aggregate([])
        assert result["num_instances"] == 0
        assert result["any_pass_rate"] == 0.0
        assert result["all_pass_rate"] == 0.0

    def test_all_instances_pass(self):
        scores = [
            {"collected": True, "num_tests": 1, "num_passed": 1, "any_passed": True, "all_passed": True},
            {"collected": True, "num_tests": 2, "num_passed": 2, "any_passed": True, "all_passed": True},
        ]
        result = aggregate(scores)
        assert result["any_pass_rate"] == 1.0
        assert result["all_pass_rate"] == 1.0

    def test_no_instances_pass(self):
        scores = [
            {"collected": False, "num_tests": 0, "num_passed": 0, "any_passed": False, "all_passed": False},
        ]
        result = aggregate(scores)
        assert result["any_pass_rate"] == 0.0
        assert result["all_pass_rate"] == 0.0
