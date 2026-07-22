"""Tests for token_count.py: measuring the rendered prompt's token cost via
tiktoken as a cross-model proxy (not Groq's exact billed count).
"""

from kg_test_generation.token_count import count_prompt_tokens, count_tokens


class TestCountTokens:
    def test_counts_a_simple_string(self):
        assert count_tokens("hello world") > 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long = count_tokens("hello " * 100)
        assert long > short

    def test_empty_string_has_zero_tokens(self):
        assert count_tokens("") == 0


class TestCountPromptTokens:
    def test_baseline_shape(self):
        context = {
            "function_name": "send",
            "source_code": "def send(self, request, **kwargs):\n    return None\n",
            "file_path": "requests/sessions.py",
        }
        assert count_prompt_tokens(context) > 0

    def test_kg_augmented_shape(self):
        context = {
            "seed": {
                "function_name": "send",
                "module": "requests.sessions",
                "signature": "def send(self, request, **kwargs)",
                "source_code": "def send(...): ...",
            },
            "context": {
                "callers": [{"name": "request", "module": "requests.sessions"}],
                "callees": [{"name": "rebuild_proxies", "module": "requests.sessions"}],
            },
            "instructions": {"coverage_targets": ["happy path"]},
        }
        assert count_prompt_tokens(context) > 0

    def test_more_context_produces_more_tokens(self):
        """The whole point of this utility: comparing token cost across
        context variants (e.g. full vs. trimmed callers/callees, see
        issue #23) -- a payload with more context content must count more
        tokens than a minimal one built from the same seed.
        """
        minimal = {
            "seed": {"function_name": "send", "module": "requests.sessions"},
            "context": {},
            "instructions": {},
        }
        verbose = {
            "seed": {"function_name": "send", "module": "requests.sessions"},
            "context": {
                "callers": [{"name": f"caller_{i}", "module": "requests.sessions"} for i in range(20)],
                "callees": [{"name": f"callee_{i}", "module": "requests.sessions"} for i in range(20)],
            },
            "instructions": {},
        }
        assert count_prompt_tokens(verbose) > count_prompt_tokens(minimal)
