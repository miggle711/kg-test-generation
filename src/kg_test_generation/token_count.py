"""
token_count.py

Measures the token cost of the actual prompt text sent to the LLM (the
rendered string from generate.build_prompt(), NOT the raw
LLMSerializer.serialize() JSON dict from repo-kg-construction the two
sizes aren't guaranteed to track each other, since build_prompt() reformats
the JSON into prose and can expand or compress it in the process).

Uses tiktoken as a cross-model proxy tokenizer (Groq's Llama models don't
expose a public tokenizer the way OpenAI's API does). This is NOT the
exact token count Groq bills for -- it's a consistent, offline stand-in
good enough for *relative* comparisons (this context variant vs. that one,
baseline vs. KG-augmented), which is the only thing #23's ablation and
#49's verbosity work actually need. Don't read these numbers as real
Groq-billed cost.
"""

from typing import Dict

from kg_test_generation.generate import build_prompt

_ENCODING_NAME = "cl100k_base"
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install kg-test-generation[tokens]"
            )
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    """Count tokens in a string via tiktoken's cl100k_base encoding.

    A cross-model proxy, not Groq/Llama's exact tokenizer -- see module
    docstring.
    """
    return len(_get_encoding().encode(text))


def count_prompt_tokens(context: Dict) -> int:
    """Count tokens in the rendered prompt for a given context payload
    (either arm's shape -- see generate.build_prompt).

    Args:
        context: Output of context.build_baseline_context or
                 context.build_kg_augmented_context.

    Returns:
        Token count of the actual string that would be sent to the LLM.
    """
    return count_tokens(build_prompt(context))
