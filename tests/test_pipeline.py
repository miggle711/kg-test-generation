"""Tests for pipeline.py.

run_baseline/run_kg_augmented need a real GROQ_API_KEY and network access
(they call the real Groq API and clone real repos), so they're not
exercised end-to-end here -- manually verified during development against
real psf/requests data (see PR description). What IS covered here without
network access: the disposable-venv helpers, which are real, local
subprocess/filesystem operations.
"""

from pathlib import Path

from kg_test_generation.pipeline import _create_disposable_venv, _install_target_repo_deps


class TestCreateDisposableVenv:
    def test_creates_a_working_python_executable(self, tmp_path):
        venv_dir = tmp_path / "venv"
        python_bin = _create_disposable_venv(venv_dir)

        assert python_bin.exists()

        import subprocess
        result = subprocess.run(
            [str(python_bin), "-c", "print('hello from venv')"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "hello from venv" in result.stdout


class TestInstallTargetRepoDeps:
    def test_installs_pytest_even_without_setup_py(self, tmp_path):
        """A bare directory (no setup.py/pyproject.toml) must not crash --
        pytest still gets installed so the file can at least attempt to
        run (and fail with a normal ImportError, handled by
        execute.run_test_file as a collection failure, not a crash here).
        """
        venv_dir = tmp_path / "venv"
        python_bin = _create_disposable_venv(venv_dir)
        repo_checkout = tmp_path / "repo"
        repo_checkout.mkdir()

        _install_target_repo_deps(python_bin, repo_checkout)

        import subprocess
        result = subprocess.run(
            [str(python_bin), "-c", "import pytest; print('pytest OK')"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "pytest OK" in result.stdout

    def test_installs_target_repo_when_setup_py_present(self, tmp_path):
        venv_dir = tmp_path / "venv"
        python_bin = _create_disposable_venv(venv_dir)
        repo_checkout = tmp_path / "repo"
        (repo_checkout / "mypkg").mkdir(parents=True)
        (repo_checkout / "mypkg" / "__init__.py").write_text(
            "VALUE = 42\n"
        )
        (repo_checkout / "setup.py").write_text(
            "from setuptools import setup, find_packages\n"
            "setup(name='mypkg', version='0.1', packages=find_packages())\n"
        )

        _install_target_repo_deps(python_bin, repo_checkout)

        import subprocess
        result = subprocess.run(
            [str(python_bin), "-c", "from mypkg import VALUE; print(VALUE)"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "42" in result.stdout

    def test_broken_setup_py_does_not_raise(self, tmp_path):
        """A target repo with a setup.py that fails to install (bad deps,
        syntax error, etc.) must not abort the whole run -- the caller
        should still get a chance to see the resulting import failure as
        a normal collection failure downstream, not an unhandled
        exception from this helper.
        """
        venv_dir = tmp_path / "venv"
        python_bin = _create_disposable_venv(venv_dir)
        repo_checkout = tmp_path / "repo"
        repo_checkout.mkdir()
        (repo_checkout / "setup.py").write_text("this is not valid python syntax(((\n")

        _install_target_repo_deps(python_bin, repo_checkout)  # must not raise
