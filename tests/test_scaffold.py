"""Smoke test: confirms the package scaffold imports cleanly and the
kg_construction dependency is wired up correctly. Stage modules are not
implemented yet (see issue #1), so this only checks import-time wiring,
not behavior.
"""


def test_package_imports():
    import kg_test_generation  # noqa: F401
    from kg_test_generation import context, generate, parse_output, execute, metrics, pipeline  # noqa: F401


def test_kg_construction_dependency_is_importable():
    """The whole point of depending on kg-construction as a real package:
    confirm its public API is reachable from here.
    """
    from kg_construction import (  # noqa: F401
        RepoKGBuilder, KGQueryEngine, KGValidator,
        TestContextExtractor, TestContext, TestContextValidator,
        LLMSerializer, LLMInput,
    )
    from kg_construction.pipeline import extract_and_validate, serialize_context  # noqa: F401
