"""Tests for context.resolve_target_function: the guardrail (issue #43)
ensuring build_baseline_context and build_kg_augmented_context agree on
which function they're testing for a given instance.

build_baseline_context/build_kg_augmented_context themselves need a real
repo checkout (via kg_construction.kg.repo_manager.RepoManager) and are
not covered here -- see the manual validation notes in the PR description
for end-to-end verification against real psf/requests data.
"""

import pytest

from kg_test_generation.context import resolve_target_function


def _instance(patch: str, code_file: str = "mod.py") -> dict:
    return {
        "repo": "test/repo",
        "base_commit": "deadbeef",
        "patch": patch,
        "code_file": code_file,
        "test_file": "test_mod.py",
    }


class TestResolveTargetFunction:
    def test_resolves_single_changed_function(self):
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def target():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        assert resolve_target_function(_instance(patch)) == "target"

    def test_deterministic_when_patch_is_ambiguous(self):
        """A patch whose diff context sweeps in more than one def line
        must still resolve to the SAME name every time it's called --
        this is the actual guarantee the guardrail provides. It does not
        claim to pick the "correct" name when a patch is genuinely
        ambiguous, only that repeated calls (i.e. both context builders)
        agree with each other.
        """
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,10 +1,10 @@\n"
            " def alpha():\n"
            "     pass\n"
            "\n"
            " def beta():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        first = resolve_target_function(_instance(patch))
        second = resolve_target_function(_instance(patch))
        assert first == second

    def test_raises_when_no_changed_function_found(self):
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        with pytest.raises(ValueError, match="No changed function found"):
            resolve_target_function(_instance(patch))

    def test_ignores_changes_in_other_files(self):
        patch = (
            "--- a/other.py\n"
            "+++ b/other.py\n"
            "@@ -1,3 +1,3 @@\n"
            " def unrelated():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        with pytest.raises(ValueError, match="No changed function found"):
            resolve_target_function(_instance(patch, code_file="mod.py"))
