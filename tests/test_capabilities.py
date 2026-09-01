"""Asking the binary what it accepts, instead of guessing from a version.

llama.cpp renames and retires flags between releases and people run whatever
their package manager last shipped. A command written against today's
documentation can fail on a perfectly good older build, and one written against
an old build gives up features the new one has. Reading the binary's own help
turns that whole class of breakage into a question with an answer.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llmfit.runtimes import capabilities
from llmfit.runtimes.llamacpp import _VALUELESS_FLAGS


@pytest.fixture(autouse=True)
def clear_cache():
    capabilities.clear_cache()
    yield
    capabilities.clear_cache()


class TestFiltering:
    def test_removes_an_unsupported_flag_and_its_value(self):
        command = ["llama-server", "--model", "a.gguf", "--defrag-thold", "0.1"]
        assert capabilities.filter_flags(command, {"--model"}) == [
            "llama-server", "--model", "a.gguf"]

    def test_keeps_the_token_after_a_flag_that_takes_no_value(self):
        command = ["llama-server", "--mlock", "--model", "a.gguf"]
        filtered = capabilities.filter_flags(command, {"--model"},
                                             drop_value={"--mlock"})
        assert filtered == ["llama-server", "--model", "a.gguf"]

    def test_a_negative_number_is_still_read_as_a_value(self):
        """Several llama.cpp defaults are -1, and a value is not a flag just
        because it starts with a minus."""
        command = ["llama-server", "--yarn-ext-factor", "-1.0", "--model", "a.gguf"]
        assert capabilities.filter_flags(command, {"--model"}) == [
            "llama-server", "--model", "a.gguf"]

    def test_an_unsupported_flag_at_the_end_is_dropped_cleanly(self):
        command = ["llama-server", "--model", "a.gguf", "--swa-full"]
        assert capabilities.filter_flags(command, {"--model"}) == [
            "llama-server", "--model", "a.gguf"]

    def test_not_knowing_is_not_the_same_as_knowing_it_is_unsupported(self):
        """An empty answer means the binary could not be asked. Stripping the
        command down on that basis would break working setups on machines where
        the help output is merely unusual."""
        command = ["llama-server", "--model", "a.gguf", "--anything"]
        assert capabilities.filter_flags(command, set()) == command

    def test_supported_flags_survive_untouched(self):
        command = ["llama-server", "--model", "a.gguf", "--ctx-size", "8192"]
        assert capabilities.filter_flags(
            command, {"--model", "--ctx-size"}) == command


class TestAskingTheBinary:
    def _fake(self, monkeypatch, output, ok=True):
        class Result:
            def __init__(self):
                self.output = output
                self.ok = ok

        monkeypatch.setattr(capabilities, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(capabilities, "run", lambda *a, **k: Result())

    def test_reads_the_long_flags_out_of_help(self, monkeypatch):
        self._fake(monkeypatch, "\n".join(
            f"  -x, --flag-{i} N   description" for i in range(15)))
        flags = capabilities.supported_flags("llama-server")
        assert "--flag-0" in flags and "--flag-14" in flags

    def test_output_that_is_not_a_help_screen_answers_nothing(self, monkeypatch):
        """Better to say "unknown" than to conclude that a working binary
        supports two flags and strip everything else off its command line."""
        self._fake(monkeypatch, "error: unrecognised option --help")
        assert capabilities.supported_flags("llama-server") == set()

    def test_a_missing_binary_answers_nothing(self, monkeypatch):
        self._fake(monkeypatch, "", ok=False)
        assert capabilities.supported_flags("llama-server") == set()

    def test_the_binary_is_only_asked_once(self, monkeypatch):
        calls = []

        class Result:
            output = "\n".join(f"  --flag-{i} N" for i in range(15))
            ok = True

        monkeypatch.setattr(capabilities, "which", lambda b: "/usr/bin/" + b)
        monkeypatch.setattr(capabilities, "run",
                            lambda *a, **k: (calls.append(1), Result())[1])
        capabilities.supported_flags("llama-server")
        capabilities.supported_flags("llama-server")
        assert len(calls) == 1

    def test_short_flags_are_ignored(self, monkeypatch):
        """Short aliases are ambiguous and unstable, and llmfit emits long forms
        everywhere, so matching on them would only create false positives."""
        self._fake(monkeypatch, "\n".join(
            [f"  -{chr(97 + i)}, --long-{i} N" for i in range(15)]))
        flags = capabilities.supported_flags("llama-server")
        assert not any(f.startswith("-") and not f.startswith("--") for f in flags)


def test_every_standalone_flag_is_declared_as_taking_no_value():
    """The filter assumes a flag takes one value unless told otherwise, so a
    standalone flag missing from the table would eat the token after it. That
    fails silently and produces a command that is subtly wrong rather than
    obviously broken."""
    source = Path("src/llmfit/runtimes/llamacpp.py").read_text(encoding="utf-8")
    appended = set(re.findall(r'(?:cmd|flags)\.append\("(--[a-z-]+)"\)', source))
    missing = appended - _VALUELESS_FLAGS
    assert not missing, f"add these to _VALUELESS_FLAGS: {sorted(missing)}"


def test_importing_one_adapter_does_not_hide_the_others():
    """The registry fills itself by importing the adapter modules, and used to
    decide it had already done so whenever anything was registered. Importing a
    single adapter first therefore left every other engine invisible for the
    rest of the process, which is exactly what a caller reaching for one
    runtime by name does."""
    import llmfit.runtimes.llamacpp  # noqa: F401
    from llmfit.runtimes.registry import all_runtimes

    names = {r.name for r in all_runtimes()}
    assert {"llamacpp", "ollama", "vllm", "lmstudio", "mlx"} <= names
