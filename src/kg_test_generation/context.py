"""
context.py

Build context: produces the prompt context for either the baseline arm
(no KG) or the KG-augmented arm, for a given instance.

v1 prototype, per the decisions in issue #1 (revisit as we learn more):
  - Baseline: one-pass -- the target function's raw source dumped into a
    single prompt, no agentic search of the repo.
  - KG-augmented: the serialized KG subgraph REPLACES raw code context
    (a clean ablation), rather than augmenting it.

Guardrail (issue #43): PatchParser.extract_changed_functions can return
more than one candidate name for a single patch (a wide diff context
window can sweep an unrelated function's def line into a hunk -- fixed
at the source in repo-kg-construction#44, but the underlying ambiguity
is inherent to diff parsing and can't be eliminated entirely). Both arms
independently call into that same parser for the same instance --
build_baseline_context directly, build_kg_augmented_context indirectly
via repo-kg-construction's TestContextExtractor -- with no coordination
between them. If they picked different names, the two experiment arms
would silently evaluate different functions for what's nominally one
instance, invalidating the baseline-vs-KG-augmented comparison entirely.

resolve_target_function() resolves the name once, and both arms are
verified against it: build_baseline_context uses it directly;
build_kg_augmented_context checks the KG arm's own seed selection
against it and raises if they disagree, rather than trusting two
independent picks to silently agree.
"""

import ast
import tempfile
from pathlib import Path
from typing import Dict, Optional

from kg_construction.extraction.patch import PatchParser
from kg_construction.kg.repo_manager import RepoManager
from kg_construction.pipeline import extract_and_validate, serialize_context


def resolve_target_function(instance: Dict) -> str:
    """Resolve a single, canonical target function name for an instance.

    Both build_baseline_context and build_kg_augmented_context must agree
    on this name -- see module docstring for why. Picks the first name in
    sorted order for determinism (Set iteration order is not guaranteed);
    this does not resolve which candidate is the "correct" one when a
    patch is genuinely ambiguous, only that all callers make the same
    (deterministic) choice.

    Args:
        instance: Dict with repo, base_commit, patch, code_file, test_file.

    Returns:
        The resolved target function/method name.

    Raises:
        ValueError: If no changed function is found in code_file for this patch.
    """
    changed_names = PatchParser.extract_changed_functions(
        instance["patch"], instance["code_file"]
    )
    if not changed_names:
        raise ValueError(
            f"No changed function found in {instance['code_file']} for this patch"
        )
    return sorted(changed_names)[0]


def build_baseline_context(instance: Dict) -> Dict:
    """Build prompt context for the baseline (no-KG) arm: one-pass, just
    the target function's own source, no callers, callees, or related
    classes.

    Args:
        instance: Dict with repo, base_commit, patch, code_file, test_file
                  (same shape kg_construction.pipeline.extract_and_validate
                  expects).

    Returns:
        {
            "function_name": str,
            "class_name": str,       # owning class name, "" if a module-level function
            "source_code": str,      # the target function's own source
            "file_path": str,        # instance['code_file']
        }

    Raises:
        ValueError: If the target function can't be found in code_file.
    """
    target_name = resolve_target_function(instance)
    source_code, class_name = _extract_function_source(instance, target_name)
    return {
        "function_name": target_name,
        "class_name": class_name,
        "source_code": source_code,
        "file_path": instance["code_file"],
    }


def build_kg_augmented_context(instance: Dict, depth: int = 2) -> Dict:
    """Build prompt context for the KG-augmented arm: the serialized KG
    subgraph, which replaces raw code context rather than augmenting it.

    Args:
        instance: Same shape as build_baseline_context.
        depth: BFS depth passed through to kg_construction's subgraph
               extraction.

    Returns:
        Hierarchical {seed, context, instructions} dict from
        kg_construction.pipeline.serialize_context.

    Raises:
        ValueError: If the KG arm's own seed selection disagrees with the
                    canonically resolved target function (see module
                    docstring) -- surfaces the ambiguity instead of
                    silently generating tests for a different function
                    than the baseline arm.
    """
    target_name = resolve_target_function(instance)
    context, _report = extract_and_validate(instance, depth=depth, verbose=False)

    seed_names = {seed.get("label") for seed in context.seeds}
    if target_name not in seed_names:
        raise ValueError(
            f"KG-augmented arm's seed selection {seed_names!r} does not include "
            f"the canonically resolved target function {target_name!r} -- the "
            f"patch is ambiguous enough that the two arms disagree (see issue #43 "
            f"in repo-kg-construction). Refusing to silently generate tests for a "
            f"different function than the baseline arm would."
        )

    return serialize_context(context)


def _extract_function_source(instance: Dict, function_name: str) -> tuple:
    """Extract a named function/method's source from instance['code_file']
    at instance['base_commit'], independent of any KG machinery -- the
    baseline arm must not depend on KG construction at all.

    Returns:
        (source_code, class_name) -- class_name is "" for a module-level
        function, else the name of its enclosing class. Without this, the
        baseline prompt has no way to tell the model whether the target is
        importable directly or needs to be called on a class instance, and
        the model has to guess -- sometimes wrong (see issue #25: this
        caused a hard ImportError for a method the model assumed was a
        free function, and an AttributeError for a method attributed to
        the wrong of two similarly-named classes).
    """
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "src"
        RepoManager().extract_at_commit(instance["repo"], instance["base_commit"], dest)
        file_path = dest / instance["code_file"]
        source = file_path.read_text(encoding="utf-8", errors="replace")

    result = _find_function_source(source, function_name)
    if result is None:
        raise ValueError(
            f"Changed function '{function_name}' not found in "
            f"{instance['code_file']} at {instance['base_commit'][:8]}"
        )
    return result


def _find_function_source(source: str, function_name: str) -> Optional[tuple]:
    """Find a top-level function or method's source (and its enclosing
    class name, if any) by name via ast.

    Returns the first match (module-level function, or method of any
    class) -- matches PatchParser's own name-only granularity, which
    doesn't disambiguate between same-named methods on different classes.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    def _search(node, enclosing_class):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == function_name:
                func_source = "".join(lines[child.lineno - 1 : child.end_lineno])
                return func_source, (enclosing_class or "")
            if isinstance(child, ast.ClassDef):
                found = _search(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A def nested inside this function's body is a closure,
                # not a class method, even if this function itself is one
                # -- reset enclosing_class rather than propagate it further.
                found = _search(child, None)
            else:
                found = _search(child, enclosing_class)
            if found is not None:
                return found
        return None

    return _search(tree, None)
