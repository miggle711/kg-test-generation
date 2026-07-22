"""
pipeline.py

Top-level orchestration: instance -> context -> generate -> parse ->
execute -> score, for both the baseline and KG-augmented arms.

Each run gets a fresh repo checkout, and dependency installation +
pytest execution happen inside a disposable Docker container (see
execute.py's container_image path) rather than on the host. This isn't
just sandboxing against malicious generated code (issue #8) -- it's also
the fix for issue #13: old target commits pin dependencies (e.g.
urllib3's vendored six.moves shim, collections.Mapping) that are
incompatible with whatever Python happens to be installed on the host.
DEFAULT_CONTAINER_IMAGE pins a Python version contemporaneous with the
benchmark repos this pipeline currently targets (TestGenEval/SWE-bench-era
commits), so the container -- not the host -- decides which Python
actually runs the generated test.
"""

import tempfile
from pathlib import Path
from typing import Dict

from kg_construction.kg.repo_manager import RepoManager

from kg_test_generation.context import build_baseline_context, build_kg_augmented_context
from kg_test_generation.execute import run_test_file
from kg_test_generation.generate import GroqTestGenerator
from kg_test_generation.metrics import score
from kg_test_generation.parse_output import extract_test_code, write_test_file

DEFAULT_CONTAINER_IMAGE = "python:3.8-slim"


def run_baseline(instance: Dict, container_image: str = DEFAULT_CONTAINER_IMAGE) -> Dict:
    """Run the full baseline (no-KG) pipeline for one instance.

    Args:
        instance: Dict with repo, base_commit, patch, code_file, test_file.
        container_image: Docker image to run the generated test in (see
                         execute.run_test_file). Defaults to a Python
                         version compatible with older benchmark commits.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    context = build_baseline_context(instance)
    return _generate_and_score(instance, context, container_image)


def run_kg_augmented(
    instance: Dict, depth: int = 2, container_image: str = DEFAULT_CONTAINER_IMAGE
) -> Dict:
    """Run the full KG-augmented pipeline for one instance.

    Args:
        instance: Same shape as run_baseline.
        depth: BFS depth passed through to kg_construction's subgraph extraction.
        container_image: Same as run_baseline.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    context = build_kg_augmented_context(instance, depth=depth)
    return _generate_and_score(instance, context, container_image)


def _generate_and_score(instance: Dict, context: Dict, container_image: str) -> Dict:
    """Shared tail of both arms: generate -> parse -> execute -> score,
    inside a fresh repo checkout that's torn down when this function
    returns, regardless of outcome. Dependency install + pytest execution
    happen inside container_image, not on the host.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_checkout = Path(tmp) / "repo"
        RepoManager().extract_at_commit(instance["repo"], instance["base_commit"], repo_checkout)

        generator = GroqTestGenerator()
        raw_output = generator.generate(context)
        test_code = extract_test_code(raw_output)

        test_file = repo_checkout / "test_generated.py"
        write_test_file(test_code, test_file)

        result = run_test_file(test_file, repo_checkout, container_image=container_image)
        return score(result)
        # tmp (repo_checkout) is deleted here on context exit, whether or
        # not the above raised.


def main():
    """CLI entry point (kg-testgen console script)."""
    raise NotImplementedError("no CLI yet -- import run_baseline/run_kg_augmented directly for now")


if __name__ == "__main__":
    main()
