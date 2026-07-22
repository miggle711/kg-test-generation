"""Tests for generate.py: prompt building for both context shapes, and
GroqTestGenerator's construction/error paths.

The actual Groq API call (GroqTestGenerator.generate with a valid key) is
not covered here -- no network access / API key in CI. Manually verified
during development: a fake key reaches the real API and correctly surfaces
a 401 as a wrapped ValueError, confirming the request path is wired
correctly end to end; only the "valid key" case (real generated text
instead of an auth error) is untested.
"""

import pytest

from kg_test_generation.generate import (
    GroqTestGenerator,
    build_prompt,
    system_prompt,
)


class TestBuildPrompt:
    def test_baseline_shape_produces_function_source_prompt(self):
        context = {
            "function_name": "send",
            "source_code": "def send(self, request, **kwargs):\n    return None\n",
            "file_path": "requests/sessions.py",
        }
        prompt = build_prompt(context)

        assert "FUNCTION TO TEST" in prompt
        assert "send" in prompt
        assert "requests/sessions.py" in prompt
        assert "def send(self, request, **kwargs):" in prompt
        # Baseline arm must not have KG-only sections
        assert "EXECUTION CONTEXT" not in prompt
        assert "Callers" not in prompt

    def test_kg_augmented_shape_produces_hierarchical_prompt(self):
        context = {
            "seed": {
                "function_name": "send",
                "module": "requests.sessions",
                "signature": "def send(self, request, **kwargs)",
                "docstring": "Send a request.",
                "source_code": "def send(...): ...",
                "exceptions": ["ValueError"],
            },
            "context": {
                "callers": [{"name": "request", "module": "requests.sessions"}],
                "callees": [{"name": "rebuild_proxies", "module": "requests.sessions"}],
                "related": [
                    {"type": "parent_class", "name": "SessionRedirectMixin", "module": "requests.sessions"}
                ],
                "existing_tests": [{"name": "test_send_basic"}],
                "patterns": {"control_flow": ["Branches: 3"], "error_handling": ["ValueError"]},
            },
            "instructions": {
                "coverage_targets": ["happy path", "error cases"],
                "conventions": {"naming": "test_<function>_<scenario>"},
            },
        }
        prompt = build_prompt(context)

        assert "SEED FUNCTION" in prompt
        assert "send" in prompt
        assert "Send a request." in prompt
        assert "ValueError" in prompt
        assert "EXECUTION CONTEXT" in prompt
        assert "requests.sessions.request" in prompt  # caller, qualified
        assert "requests.sessions.rebuild_proxies" in prompt  # callee, qualified
        assert "requests.sessions.SessionRedirectMixin" in prompt  # related, qualified
        assert "test_send_basic" in prompt  # existing test
        assert "happy path" in prompt  # coverage target

    def test_seed_module_produces_explicit_import_instruction(self):
        """The model must be told the real import path explicitly, not just
        shown it once in a "Module:" line -- this is the fix for issue #6,
        where the model fabricated "from your_module import X" placeholder
        imports when it had no real module path to work from.
        """
        context = {
            "seed": {"function_name": "send", "module": "requests.sessions"},
            "context": {},
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "from requests.sessions import send" in prompt
        assert "Do not invent a placeholder module" in prompt

    def test_missing_seed_module_omits_import_instruction(self):
        """No module info available (e.g. an older KG snapshot without the
        field) must not produce a broken/empty import instruction -- it's
        simply omitted, same as any other optional section.
        """
        context = {"seed": {"function_name": "send"}, "context": {}, "instructions": {}}
        prompt = build_prompt(context)

        assert "Do not invent a placeholder module" not in prompt

    def test_context_nodes_without_module_fall_back_to_bare_name(self):
        """A caller/callee/related entry with no module info (e.g. an
        external stdlib symbol the KG didn't resolve a filepath for) must
        still render its bare name rather than crashing or showing "None.name".
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "callers": [{"name": "caller_without_module"}],
                "callees": [{"name": "callee_without_module"}],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "caller_without_module" in prompt
        assert "None.caller_without_module" not in prompt

    def test_dispatches_on_seed_key_presence(self):
        """The dispatch in build_prompt keys off whether "seed" is a
        top-level field -- confirms baseline and kg-augmented payloads are
        routed to their respective builders, not accidentally conflated.
        """
        baseline = {"function_name": "f", "source_code": "", "file_path": "x.py"}
        kg_augmented = {"seed": {}, "context": {}, "instructions": {}}

        assert "FUNCTION TO TEST" in build_prompt(baseline)
        assert "SEED FUNCTION" in build_prompt(kg_augmented)

    def test_kg_augmented_handles_missing_optional_sections_gracefully(self):
        """A minimal hierarchical payload (empty context/instructions)
        must not crash -- optional sections should just be omitted.
        """
        context = {"seed": {"function_name": "f"}, "context": {}, "instructions": {}}
        prompt = build_prompt(context)
        assert "SEED FUNCTION" in prompt
        assert "f" in prompt


class TestSystemPrompt:
    def test_mentions_pytest_and_output_only_code(self):
        prompt = system_prompt()
        assert "pytest" in prompt.lower()
        assert "Output ONLY the test code" in prompt


class TestGroqTestGenerator:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GROQ_API_KEY not found"):
            GroqTestGenerator()

    def test_constructs_with_explicit_api_key(self):
        gen = GroqTestGenerator(api_key="fake-key-for-testing")
        assert gen.api_key == "fake-key-for-testing"

    def test_constructs_from_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "fake-key-from-env")
        gen = GroqTestGenerator()
        assert gen.api_key == "fake-key-from-env"

    def test_generate_wraps_api_errors_as_value_error(self):
        """generate() must wrap any exception the Groq SDK raises in a
        ValueError rather than letting it propagate raw. Mocked -- no
        network access needed. Manually verified once against the real
        API during development: a fake key reaches Groq and returns a
        real 401, confirming the request path itself is wired correctly;
        this test only covers the wrapping behavior on top of that.
        """
        gen = GroqTestGenerator(api_key="fake-key-for-testing")
        gen.client.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("401 Invalid API Key")
        )
        context = {"function_name": "f", "source_code": "def f(): pass", "file_path": "x.py"}
        with pytest.raises(ValueError, match="Groq API call failed"):
            gen.generate(context)

    def test_generate_returns_response_content_on_success(self):
        gen = GroqTestGenerator(api_key="fake-key-for-testing")

        class _FakeMessage:
            content = "def test_f_returns_none():\n    assert f() is None\n"

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        gen.client.chat.completions.create = lambda **kwargs: _FakeResponse()
        context = {"function_name": "f", "source_code": "def f(): pass", "file_path": "x.py"}
        result = gen.generate(context)
        assert "def test_f_returns_none" in result
