"""Tests for pipeline.py.

run_baseline/run_kg_augmented need a real GROQ_API_KEY, network access,
and Docker (they call the real Groq API, clone real repos, and run
generated tests in a container), so they're not exercised end-to-end here
-- manually verified during development against real psf/requests data
(see PR description). The container-based install-and-run logic itself
(execute._run_test_file_in_container) is covered directly in
test_execute.py, since that's where it now lives.
"""

from kg_test_generation.pipeline import DEFAULT_CONTAINER_IMAGE, run_baseline, run_kg_augmented


def test_default_container_image_is_set():
    assert DEFAULT_CONTAINER_IMAGE


def test_run_functions_accept_container_image_override():
    import inspect

    assert "container_image" in inspect.signature(run_baseline).parameters
    assert "container_image" in inspect.signature(run_kg_augmented).parameters
