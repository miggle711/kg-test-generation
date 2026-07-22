"""Tests for context.resolve_target_function: the guardrail (issue #43)
ensuring build_baseline_context and build_kg_augmented_context agree on
which function they're testing for a given instance.

build_baseline_context/build_kg_augmented_context themselves need a real
repo checkout (via kg_construction.kg.repo_manager.RepoManager) and are
not covered here -- see the manual validation notes in the PR description
for end-to-end verification against real psf/requests data.
"""

import pytest

from kg_test_generation.context import _find_function_source, resolve_target_function


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


class TestFindFunctionSource:
    """_find_function_source needs to report the enclosing class name (or
    "" for a module-level function) -- without it, the baseline arm can't
    tell the model whether a target is importable directly or needs to be
    called on a class instance, which caused real, reproducible failures
    on real instances (see issue #25: a method the model assumed was a
    free function -> ImportError; a method attributed to the wrong of two
    similarly-named classes -> AttributeError).
    """

    def test_module_level_function_has_empty_class_name(self):
        source = "def target():\n    return 1\n"
        func_source, class_name = _find_function_source(source, "target")

        assert "def target():" in func_source
        assert class_name == ""

    def test_method_reports_its_class_name(self):
        source = (
            "class Foo:\n"
            "    def method(self):\n"
            "        return 1\n"
        )
        func_source, class_name = _find_function_source(source, "method")

        assert "def method(self):" in func_source
        assert class_name == "Foo"

    def test_disambiguates_same_named_methods_on_different_classes(self):
        """Two classes in the same file can each define a method with the
        same name -- the exact shape of issue #25's second failure
        (prepare_content_length existed as a real method of PreparedRequest,
        but the model attributed it to a similarly-named Request class
        instead). This test doesn't assert which class "wins" (matches
        PatchParser's own name-only granularity per the docstring), only
        that a class name is returned rather than silently blank.
        """
        source = (
            "class Alpha:\n"
            "    def shared_name(self):\n"
            "        return 'alpha'\n"
            "\n"
            "class Beta:\n"
            "    def shared_name(self):\n"
            "        return 'beta'\n"
        )
        func_source, class_name = _find_function_source(source, "shared_name")

        assert class_name in ("Alpha", "Beta")

    def test_returns_none_when_function_not_found(self):
        source = "def other():\n    return 1\n"
        assert _find_function_source(source, "missing") is None

    def test_nested_function_inside_a_method_is_not_mistaken_for_a_class_method(self):
        """A function defined inside a method's body (a closure) is not a
        class method -- it must not incorrectly report the outer class as
        its enclosing class once the search descends past the method
        that isn't the target.
        """
        source = (
            "class Foo:\n"
            "    def outer(self):\n"
            "        def inner():\n"
            "            return 1\n"
            "        return inner()\n"
        )
        func_source, class_name = _find_function_source(source, "inner")

        assert "def inner():" in func_source
        assert class_name == ""
