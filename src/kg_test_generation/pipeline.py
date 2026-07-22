"""
pipeline.py

Top-level orchestration: instance -> context -> generate -> parse ->
execute -> score, for both the baseline and KG-augmented arms.

Each run gets a fresh, disposable virtual environment: the target repo's
own dependencies (needed for the generated test file to even import) are
installed there, never into the environment this pipeline itself runs in,
and the whole checkout+venv is deleted afterward regardless of outcome.
This is NOT sandboxing against malicious code (see issue #8 for the
Docker follow-up) -- it only isolates *dependency installation* between
runs and keeps the calling environment clean.
"""

import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Dict

from kg_construction.kg.repo_manager import RepoManager

from kg_test_generation.context import build_baseline_context, build_kg_augmented_context
from kg_test_generation.execute import run_test_file
from kg_test_generation.generate import GroqTestGenerator
from kg_test_generation.metrics import score
from kg_test_generation.parse_output import extract_test_code, write_test_file


def run_baseline(instance: Dict) -> Dict:
    """Run the full baseline (no-KG) pipeline for one instance.

    Args:
        instance: Dict with repo, base_commit, patch, code_file, test_file.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    context = build_baseline_context(instance)
    return _generate_and_score(instance, context)


def run_kg_augmented(instance: Dict, depth: int = 2) -> Dict:
    """Run the full KG-augmented pipeline for one instance.

    Args:
        instance: Same shape as run_baseline.
        depth: BFS depth passed through to kg_construction's subgraph extraction.

    Returns:
        Metrics dict for this instance (see metrics.score).
    """
    context = build_kg_augmented_context(instance, depth=depth)
    return _generate_and_score(instance, context)


def _generate_and_score(instance: Dict, context: Dict) -> Dict:
    """Shared tail of both arms: generate -> parse -> execute -> score,
    inside a fresh repo checkout + disposable venv that's torn down when
    this function returns, regardless of outcome.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo_checkout = Path(tmp) / "repo"
        RepoManager().extract_at_commit(instance["repo"], instance["base_commit"], repo_checkout)

        venv_dir = Path(tmp) / "venv"
        python_bin = _create_disposable_venv(venv_dir)
        _install_target_repo_deps(python_bin, repo_checkout)

        generator = GroqTestGenerator()
        raw_output = generator.generate(context)
        test_code = extract_test_code(raw_output)

        test_file = repo_checkout / "test_generated.py"
        write_test_file(test_code, test_file)

        result = run_test_file(test_file, repo_checkout, python_executable=str(python_bin))
        return score(result)
        # tmp (repo_checkout + venv_dir) is deleted here on context exit,
        # whether or not the above raised.


def _create_disposable_venv(venv_dir: Path) -> Path:
    """Create a fresh virtual environment and return its Python executable path."""
    venv.create(venv_dir, with_pip=True)
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _install_target_repo_deps(python_bin: Path, repo_checkout: Path) -> None:
    """Install the target repo's own dependencies (needed for the
    generated test file to import it) plus pytest, into the disposable
    venv -- never into the environment this pipeline itself runs in.

    Best-effort: if the target repo has no setup.py/pyproject.toml at all,
    installing it is skipped rather than failing the whole run (the
    generated test will simply fail to import, which run_test_file
    already handles as a collection failure).
    """
    subprocess.run(
        [str(python_bin), "-m", "pip", "install", "--quiet", "pytest"],
        check=True, timeout=120,
    )

    has_setup = (repo_checkout / "setup.py").exists() or (repo_checkout / "pyproject.toml").exists()
    if has_setup:
        subprocess.run(
            [str(python_bin), "-m", "pip", "install", "--quiet", "-e", str(repo_checkout)],
            check=False, timeout=300,  # best-effort: a broken setup.py must not abort the run
        )


def main():
    """CLI entry point (kg-testgen console script)."""
    raise NotImplementedError("no CLI yet -- import run_baseline/run_kg_augmented directly for now")


if __name__ == "__main__":
    main()
