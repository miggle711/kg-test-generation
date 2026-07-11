"""
generate.py

Generate: calls an LLM (Groq) with a built context and
returns raw generated test code as a string.

This is the responsibility that used to live in repo-kg-construction as
GroqTestGenerator/AnthropicTestGenerator (removed in that repo's PR #24,
since it's pure "call an LLM API" with no KG-schema coupling). Not
reimplemented yet -- deliberately not carried over verbatim, since the
context payload shape isn't settled yet (see context.py).
"""

from typing import Optional


class GroqTestGenerator:
    """Calls the Groq API to generate test code from a context payload."""

    def __init__(self, api_key: Optional[str] = None):
        raise NotImplementedError("depends on context.py's payload shape, see issue #1")

    def generate(self, context: dict) -> str:
        """Generate test code from a context payload.

        Args:
            context: Output of context.build_baseline_context or
                     context.build_kg_augmented_context.

        Returns:
            Raw generated test code as a string.
        """
        raise NotImplementedError
