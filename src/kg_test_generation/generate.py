"""
generate.py

Calls an LLM (Groq) with a built context and returns raw generated test
code as a string.

This responsibility used to live in repo-kg-construction as
GroqTestGenerator/AnthropicTestGenerator (removed in that repo's PR #24,
since it's pure "call an LLM API" with no KG-schema coupling). Not carried
over verbatim: context.py's two arms produce genuinely different payload
shapes (baseline: flat {function_name, source_code, file_path};
KG-augmented: hierarchical {seed, context, instructions}), so prompt
building is split into two functions rather than assuming one shape.
"""

import os
from typing import Dict, Optional


def system_prompt() -> str:
    """Shared system prompt for both arms -- the task and output contract
    are identical; only the context payload shape differs.

    Consolidates the fixes for issues #17 and #22 into one mocking-
    guidance section (see issue #29) rather than two separate one-off
    instructions -- both are confirmed-live problems as of a recheck
    against real generation (see #27's latest comment): #17 causes tests
    to hit the live network (e.g. http://example.com, httpbin.org)
    instead of mocking, and #22 causes every test in a file to fail when
    the function under test uses a context manager (`with X() as x:`),
    because `mock_cls.return_value` is NOT what such a block binds to
    `x` -- that's `mock_cls.return_value.__enter__.return_value`, and the
    old guideline #6 ("Mock external dependencies") was too abstract for
    the model to apply correctly to either case.
    """
    return (
        "You are an expert Python test engineer. Your task is to generate comprehensive, "
        "well-structured unit tests for the given function.\n\n"
        "Guidelines:\n"
        "1. Generate tests that cover boundary conditions, happy paths, error cases, and edge cases.\n"
        "2. Use pytest framework conventions.\n"
        "3. Each test should have a clear docstring explaining its purpose.\n"
        "4. Use meaningful assertion messages.\n"
        "5. Follow the naming convention: test_<function>_<scenario>.\n"
        "6. Mock external dependencies but test real business logic. See the Mocking Guidance "
        "section below for the two most common ways to get this wrong.\n"
        "7. Ensure tests are independent and can run in any order.\n"
        "8. Include parametrized tests where appropriate.\n\n"
        "# Mocking Guidance\n\n"
        "## Never make a real network call\n"
        "Do not let any generated test actually reach the network, even to a URL that looks "
        "safe or well-known (e.g. http://example.com, https://httpbin.org). A real request "
        "makes the test flaky (depends on live server behavior, which can change or be "
        "unreachable) and untestable offline/in CI. Mock the HTTP layer instead, e.g.:\n\n"
        "```python\n"
        "with patch('requests.adapters.HTTPAdapter.send') as mock_send:\n"
        "    mock_send.return_value = Mock(status_code=200)\n"
        "    response = session.get('http://example.com')\n"
        "```\n\n"
        "## Mocking a context manager\n"
        "If the function under test uses `with SomeClass(...) as x:`, patching `SomeClass` "
        "alone is not enough -- `mock_cls.return_value` is what `SomeClass(...)` returns, "
        "but `x` inside the `with` block is bound to `mock_cls.return_value.__enter__.return_value` "
        "(what `__enter__()` returns), which is a DIFFERENT mock object by default. Wire up "
        "the one the code actually uses:\n\n"
        "```python\n"
        "with patch('module.SomeClass') as mock_cls:\n"
        "    mock_instance = mock_cls.return_value.__enter__.return_value\n"
        "    mock_instance.some_method.return_value = 'expected result'\n"
        "    # now `with SomeClass() as x: x.some_method()` inside the code under test\n"
        "    # will see mock_instance.some_method's mocked return value/side_effect\n"
        "```\n\n"
        "Output ONLY the test code, no explanations."
    )


def build_prompt(context: Dict) -> str:
    """Build a user prompt from either arm's context payload.

    Dispatches on shape: a KG-augmented payload has a top-level "seed" key
    (from LLMSerializer.serialize()); a baseline payload has a top-level
    "function_name" key directly (from context.build_baseline_context()).
    """
    if "seed" in context:
        return _build_kg_augmented_prompt(context)
    return _build_baseline_prompt(context)


def _build_baseline_prompt(context: Dict) -> str:
    """Build a prompt from context.build_baseline_context()'s flat payload:
    {function_name, class_name, source_code, file_path}. No callers/callees/
    related classes -- that's the whole point of the baseline arm.
    """
    class_name = context.get("class_name", "")
    function_name = context.get("function_name", "")
    file_path = context.get("file_path", "")

    parts = [
        "# FUNCTION TO TEST",
        f"Function: {function_name}",
        f"File: {file_path}",
    ]
    if class_name:
        parts.append(f"Class: {class_name}")
    parts.extend([
        "",
        "Source Code:",
        "```python",
        context.get("source_code", ""),
        "```",
        "",
    ])

    if class_name:
        parts.extend([
            f"IMPORTANT: `{function_name}` is a method of `{class_name}`, not a "
            f"standalone function. Import the class and call the method on an "
            f"instance -- do not attempt to import `{function_name}` directly, "
            "it is not a module-level name.",
            "",
        ])
    elif function_name:
        parts.extend([
            f"IMPORTANT: `{function_name}` is a module-level function in "
            f"`{file_path}`. Do not invent a placeholder module name.",
            "",
        ])

    parts.extend([
        "# GENERATE COMPREHENSIVE TESTS",
        "Create pytest-compatible test cases below:",
        "",
    ])
    return "\n".join(parts)


def _qualified_name(node: Dict) -> str:
    """Render a context node (caller/callee/related) as "module.Class.name"
    for a method, "module.name" for a module-level function, or bare "name"
    when no module is known -- so the model can tell a class method apart
    from an importable function (see issue #14: without this, methods were
    indistinguishable from free functions and got fabricated, nonexistent
    imports).
    """
    name = node.get("name", "")
    module = node.get("module", "")
    class_name = node.get("class_name", "")
    if module and class_name:
        return f"{module}.{class_name}.{name}"
    if module:
        return f"{module}.{name}"
    return name


def _render_source_snippets(label: str, nodes: list, limit: int = 3) -> list:
    """Render real source bodies for up to `limit` nodes that have one.

    Callers/callees are often numerous (a seed can have dozens -- e.g.
    issue #18 found 33 callees on one real instance) and their names are
    already listed in full above this; capping the bodies shown keeps
    token cost bounded (see #18/#49's verbosity concerns) while giving the
    model at least a few real implementations to read instead of only
    knowing that something is called (see issue #30: LLMSerializer
    already includes source_code for every caller/callee via
    _node_to_snippet, but it was never rendered here -- the model had no
    way to know what a callee actually does, e.g. which exception it
    raises, without seeing its body).
    """
    parts = []
    shown = 0
    for node in nodes:
        if shown >= limit:
            break
        source_code = node.get("source_code", "")
        if not source_code:
            continue
        parts.append(f"### {label}: {_qualified_name(node)}")
        parts.extend(["```python", source_code, "```", ""])
        shown += 1
    return parts


def _build_kg_augmented_prompt(hierarchical_json: Dict) -> str:
    """Build a prompt from context.build_kg_augmented_context()'s
    hierarchical {seed, context, instructions} payload.
    """
    seed = hierarchical_json.get("seed", {})
    context = hierarchical_json.get("context", {})
    instructions = hierarchical_json.get("instructions", {})

    class_name = seed.get("class_name", "")
    function_name = seed.get("function_name", "")

    parts = [
        "# SEED FUNCTION (Modified Function to Test)",
        f"Function: {function_name}",
        f"Module: {seed.get('module', '')}",
    ]
    if class_name:
        parts.append(f"Class: {class_name}")
    parts.extend([
        "",
        "Signature:",
        "```python",
        seed.get("signature", ""),
        "```",
        "",
    ])

    if seed.get("module") and class_name:
        parts.extend([
            f"IMPORTANT: `{function_name}` is a method of `{class_name}` "
            f"(module `{seed['module']}`), not a standalone function. Import "
            f"the class and call the method on an instance, e.g. "
            f"`from {seed['module']} import {class_name}` then "
            f"`{class_name}().{function_name}(...)`. Do not attempt to "
            f"import `{function_name}` directly -- it is not a module-level "
            "name.",
            "",
        ])
    elif seed.get("module"):
        parts.extend([
            f"IMPORTANT: Import the function/class under test from `{seed['module']}` "
            f"(e.g. `from {seed['module']} import {function_name}`). "
            "Do not invent a placeholder module name.",
            "",
        ])

    if seed.get("docstring"):
        parts.extend(['Docstring:', f'"""{seed["docstring"]}"""', ""])

    if seed.get("exceptions"):
        parts.extend(["Declared Exceptions:", ", ".join(seed["exceptions"]), ""])

    if seed.get("source_code"):
        parts.extend(["Source Code:", "```python", seed["source_code"], "```", ""])

    parts.append("# EXECUTION CONTEXT")
    parts.append("")

    if context.get("callers"):
        parts.append("## Callers (Functions that call this function):")
        for caller in context["callers"]:
            parts.append(f"- {_qualified_name(caller)}")
        parts.append("")
        parts.extend(_render_source_snippets("Caller", context["callers"]))

    if context.get("callees"):
        parts.append("## Callees (Functions called by this function):")
        for callee in context["callees"]:
            parts.append(f"- {_qualified_name(callee)}")
        parts.append("")
        parts.extend(_render_source_snippets("Callee", context["callees"]))

    if context.get("related"):
        parts.append("## Related Classes:")
        for rel in context["related"]:
            parts.append(f"- {rel.get('type', '')}: {_qualified_name(rel)}")
        parts.append("")

    if context.get("sibling_methods"):
        # Context a flat single-function extraction (the baseline arm)
        # structurally cannot provide -- other methods on the seed's own
        # class (e.g. __init__, or a setup method like prepare()) whose
        # side effects the seed's own body depends on but doesn't itself
        # establish (see issue #50: this caused generated tests to
        # instantiate an object and call the seed method directly without
        # ever calling the real setup method that initializes state the
        # seed method reads unconditionally).
        parts.append("## Other Methods on the Same Class:")
        for sm in context["sibling_methods"]:
            parts.append(f"- {_qualified_name(sm)}")
        parts.append("")
        parts.extend(_render_source_snippets("Method", context["sibling_methods"]))

    if context.get("existing_tests"):
        parts.append("## Existing Tests (for reference):")
        # Capped to 2 (not the 3 previously used for names-only) now that
        # full bodies are rendered, not just names -- bounds token cost
        # while still anchoring the model on the codebase's real mocking
        # conventions and assertion style (see issue #19: previously
        # source_code was computed and serialized but silently dropped
        # here, so the model never saw a single real test from the
        # codebase, only names).
        for test in context["existing_tests"][:2]:
            parts.append(f"### {test.get('name', '')}")
            if test.get("source_code"):
                parts.extend(["```python", test["source_code"], "```"])
        parts.append("")

    if context.get("patterns"):
        parts.append("## Patterns Observed:")
        patterns = context["patterns"]
        if patterns.get("control_flow"):
            parts.append(f"Control Flow: {', '.join(patterns['control_flow'])}")
        if patterns.get("error_handling"):
            parts.append(f"Error Handling: {', '.join(patterns['error_handling'])}")
        parts.append("")

    parts.append("# TEST GENERATION INSTRUCTIONS")
    parts.append("")

    if instructions.get("coverage_targets"):
        parts.append("## Coverage Targets:")
        for target in instructions["coverage_targets"]:
            parts.append(f"- {target}")
        parts.append("")

    if instructions.get("conventions"):
        parts.append("## Code Conventions:")
        for key, value in instructions["conventions"].items():
            parts.append(f"- {key}: {value}")
        parts.append("")

    parts.extend([
        "# GENERATE COMPREHENSIVE TESTS",
        "Create pytest-compatible test cases below:",
        "",
    ])
    return "\n".join(parts)


class GroqTestGenerator:
    """Calls the Groq API to generate test code from a context payload."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Groq API key. If None, reads from the GROQ_API_KEY
                     environment variable.

        Raises:
            ValueError: If no API key is found.
            ImportError: If the `groq` package is not installed
                         (install via `pip install kg-test-generation[groq]`).
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set via parameter or GROQ_API_KEY env var."
            )

        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Install with: pip install kg-test-generation[groq]"
            )
        self.client = Groq(api_key=self.api_key)

    def generate(self, context: Dict, model: str = "llama-3.3-70b-versatile") -> str:
        """Generate test code from a context payload.

        Args:
            context: Output of context.build_baseline_context or
                     context.build_kg_augmented_context.
            model: Groq model to use.

        Returns:
            Raw generated test code as a string.

        Raises:
            ValueError: If the API call fails.
        """
        prompt = build_prompt(context)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Groq API call failed: {e}")
