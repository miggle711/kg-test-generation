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

    def test_baseline_method_produces_class_instantiation_instruction(self):
        """A method (class_name present) must not get a plain
        "import function_name directly" instruction -- that import doesn't
        exist for a method (see issue #25: the model fabricated exactly
        that, "from requests.auth import handle_401", for a method of
        HTTPDigestAuth, and separately attributed another method to the
        wrong of two similarly-named classes).
        """
        context = {
            "function_name": "handle_401",
            "class_name": "HTTPDigestAuth",
            "source_code": "def handle_401(self, r, **kwargs):\n    ...\n",
            "file_path": "requests/auth.py",
        }
        prompt = build_prompt(context)

        assert "Class: HTTPDigestAuth" in prompt
        assert "method of `HTTPDigestAuth`" in prompt
        assert "do not attempt to import `handle_401` directly" in prompt

    def test_baseline_function_without_class_name_keeps_plain_instruction(self):
        context = {
            "function_name": "get",
            "class_name": "",
            "source_code": "def get(url, **kwargs):\n    ...\n",
            "file_path": "requests/api.py",
        }
        prompt = build_prompt(context)

        assert "module-level function in `requests/api.py`" in prompt
        assert "Class:" not in prompt

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

    def test_existing_test_source_code_is_rendered_not_just_name(self):
        """Previously (issue #19) source_code was computed and serialized
        by LLMSerializer but silently dropped here -- the model never saw
        a single real test from the codebase, only names. This is the
        actual anchor for the codebase's mocking/assertion conventions.
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "existing_tests": [
                    {"name": "test_a", "source_code": "def test_a():\n    assert f() == 1\n"},
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "def test_a():" in prompt
        assert "assert f() == 1" in prompt

    def test_existing_tests_capped_at_two(self):
        """Full bodies are real content now, not just names -- capped
        tighter than before (2, not 3) to bound token cost.
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "existing_tests": [
                    {"name": f"test_{i}", "source_code": f"def test_{i}():\n    pass\n"}
                    for i in range(5)
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "test_0" in prompt
        assert "test_1" in prompt
        assert "test_2" not in prompt

    def test_existing_test_without_source_code_falls_back_to_name_only(self):
        """A test node with no source_code (e.g. an older/incomplete KG
        snapshot) must not render an empty code block -- just the name,
        same as before this fix.
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {"existing_tests": [{"name": "test_no_source"}]},
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "test_no_source" in prompt
        assert "```python\n```" not in prompt

    def test_caller_source_code_is_rendered_not_just_qualified_name(self):
        """Previously (issue #30) source_code was already computed and
        serialized for every caller/callee (LLMSerializer._node_to_snippet),
        but this repo's prompt-rendering only ever showed the qualified
        name. The model had no way to know what a caller/callee actually
        does -- e.g. which exception a callee raises -- without seeing its
        body (see issue #27's exception-guessing failures).
        """
        context = {
            "seed": {"function_name": "send"},
            "context": {
                "callers": [
                    {
                        "name": "request", "module": "requests.sessions",
                        "source_code": "def request(self, method, url):\n    return self.send(...)\n",
                    },
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "def request(self, method, url):" in prompt
        assert "return self.send(...)" in prompt

    def test_callee_source_code_is_rendered(self):
        context = {
            "seed": {"function_name": "send"},
            "context": {
                "callees": [
                    {
                        "name": "get_adapter", "module": "requests.sessions",
                        "source_code": "def get_adapter(self, url):\n    raise InvalidSchema(url)\n",
                    },
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "def get_adapter(self, url):" in prompt
        assert "raise InvalidSchema(url)" in prompt

    def test_caller_callee_bodies_capped_at_three(self):
        """Callers/callees can be numerous (issue #18 found 33 callees on
        one real instance) -- bodies must be capped independently of the
        (uncapped) qualified-name list above them, to bound token cost.
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "callees": [
                    {"name": f"callee_{i}", "source_code": f"def callee_{i}(): pass\n"}
                    for i in range(10)
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        # All 10 names still listed (unchanged behavior).
        for i in range(10):
            assert f"callee_{i}" in prompt
        # But only 3 bodies rendered (the seed's own empty Signature
        # block also renders a ```python fence, hence 3 + 1 = 4 total).
        assert prompt.count("```python") == 4

    def test_caller_without_source_code_renders_name_only(self):
        """A caller/callee with no source_code (e.g. an external stdlib
        symbol the KG didn't resolve a body for) must not render an empty
        code block -- just its qualified name, same as before this fix.
        """
        context = {
            "seed": {"function_name": "f"},
            "context": {"callers": [{"name": "caller_no_source"}]},
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "caller_no_source" in prompt
        assert "```python\n```" not in prompt

    def test_sibling_methods_are_rendered(self):
        """Issue #50: other methods on the seed's own class (e.g.
        __init__, or a setup method like prepare()) are context a flat
        single-function extraction (baseline) structurally cannot
        provide -- must actually reach the prompt, not just be computed.
        """
        context = {
            "seed": {"function_name": "prepare_content_length", "class_name": "PreparedRequest"},
            "context": {
                "sibling_methods": [
                    {
                        "name": "prepare", "module": "requests.models", "class_name": "PreparedRequest",
                        "source_code": "def prepare(self, ...):\n    self.headers = {}\n",
                    },
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "Other Methods on the Same Class" in prompt
        assert "def prepare(self, ...):" in prompt
        assert "self.headers = {}" in prompt

    def test_sibling_methods_bodies_capped_at_three(self):
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "sibling_methods": [
                    {"name": f"method_{i}", "source_code": f"def method_{i}(self): pass\n"}
                    for i in range(9)
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        for i in range(9):
            assert f"method_{i}" in prompt
        assert prompt.count("```python") == 4  # seed signature (empty) + 3 sibling bodies

    def test_no_sibling_methods_omits_the_section(self):
        context = {"seed": {"function_name": "f"}, "context": {}, "instructions": {}}
        prompt = build_prompt(context)

        assert "Other Methods on the Same Class" not in prompt

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

    def test_method_seed_produces_class_instantiation_instruction(self):
        """A method (class_name present) must not get the plain "from
        module import function_name" instruction -- that import doesn't
        exist for a method (see issue #14, where the model fabricated
        exactly that: "from requests.sessions import resolve_redirects").
        Instead it must be told to import and instantiate the class.
        """
        context = {
            "seed": {
                "function_name": "resolve_redirects",
                "module": "requests.sessions",
                "class_name": "Session",
            },
            "context": {},
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "Class: Session" in prompt
        assert "method of `Session`" in prompt
        assert "from requests.sessions import Session" in prompt
        assert "Session().resolve_redirects(...)" in prompt
        # Must not suggest importing the method itself as a module-level name.
        assert "from requests.sessions import resolve_redirects" not in prompt

    def test_function_seed_without_class_name_keeps_plain_import_instruction(self):
        """A module-level function (no class_name) must still get the
        original "from module import function_name" instruction -- the
        method-specific branch must not affect the function case.
        """
        context = {
            "seed": {"function_name": "get", "module": "requests.api", "class_name": ""},
            "context": {},
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "from requests.api import get" in prompt
        assert "Class:" not in prompt

    def test_caller_qualified_name_includes_class_for_a_method(self):
        context = {
            "seed": {"function_name": "f"},
            "context": {
                "callers": [
                    {"name": "request", "module": "requests.sessions", "class_name": "Session"}
                ],
            },
            "instructions": {},
        }
        prompt = build_prompt(context)

        assert "requests.sessions.Session.request" in prompt

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
