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
        "6. Mock external dependencies but test real business logic.\n"
        "7. Ensure tests are independent and can run in any order.\n"
        "8. Include parametrized tests where appropriate.\n\n"
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
    {function_name, source_code, file_path}. No callers/callees/related
    classes -- that's the whole point of the baseline arm.
    """
    parts = [
        "# FUNCTION TO TEST",
        f"Function: {context.get('function_name', '')}",
        f"File: {context.get('file_path', '')}",
        "",
        "Source Code:",
        "```python",
        context.get("source_code", ""),
        "```",
        "",
        "# GENERATE COMPREHENSIVE TESTS",
        "Create pytest-compatible test cases below:",
        "",
    ]
    return "\n".join(parts)


def _qualified_name(node: Dict) -> str:
    """Render a context node (caller/callee/related) as "module.name" when
    a module is known, else bare "name" -- gives the model a real import
    path for context nodes instead of just a bare, ambiguous name.
    """
    name = node.get("name", "")
    module = node.get("module", "")
    return f"{module}.{name}" if module else name


def _build_kg_augmented_prompt(hierarchical_json: Dict) -> str:
    """Build a prompt from context.build_kg_augmented_context()'s
    hierarchical {seed, context, instructions} payload.
    """
    seed = hierarchical_json.get("seed", {})
    context = hierarchical_json.get("context", {})
    instructions = hierarchical_json.get("instructions", {})

    parts = [
        "# SEED FUNCTION (Modified Function to Test)",
        f"Function: {seed.get('function_name', '')}",
        f"Module: {seed.get('module', '')}",
        "",
        "Signature:",
        "```python",
        seed.get("signature", ""),
        "```",
        "",
    ]

    if seed.get("module"):
        parts.extend([
            f"IMPORTANT: Import the function/class under test from `{seed['module']}` "
            f"(e.g. `from {seed['module']} import {seed.get('function_name', '')}`). "
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

    if context.get("callees"):
        parts.append("## Callees (Functions called by this function):")
        for callee in context["callees"]:
            parts.append(f"- {_qualified_name(callee)}")
        parts.append("")

    if context.get("related"):
        parts.append("## Related Classes:")
        for rel in context["related"]:
            parts.append(f"- {rel.get('type', '')}: {_qualified_name(rel)}")
        parts.append("")

    if context.get("existing_tests"):
        parts.append("## Existing Tests (for reference):")
        for test in context["existing_tests"][:3]:
            parts.append(f"- {test.get('name', '')}")
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
