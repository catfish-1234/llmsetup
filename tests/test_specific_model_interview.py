"""Model choices are explained before selection and survive scripted profiles."""

import io
import json
from dataclasses import replace

import pytest
from conftest import make_hardware

from llmsetup.catalog.catalog import Catalog
from llmsetup.cli import main
from llmsetup.interview.questions import QUESTIONS
from llmsetup.interview.wizard import (
    load_answers,
    profile_from_answers,
    review_model_choice,
    run_interview,
    save_profile,
)
from llmsetup.types import ModelSpec, Task, UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console


@pytest.fixture
def choices():
    small = ModelSpec(
        id="small-chat", name="Small Chat", publisher="test", params_b=1,
        layers=16, hidden_size=1024, kv_heads=4, head_dim=64,
        max_context=32768, tasks=["chat"], supports_tools=True,
        recommended_quants=["Q8_0", "Q4_K_M"], gguf_repo="test/small",
    )
    large = replace(small, id="large-chat", name="Large Chat", params_b=100)
    return Catalog([small, large], {"Q8_0": 8.5, "Q4_K_M": 4.85})


def review(profile, choices, script=(), interactive=True):
    stream = io.StringIO()
    answers = iter(script)
    seen = []

    def answer(prompt):
        # The model explanation must precede the keep/alternative decision.
        seen.append(stream.getvalue())
        return next(answers, "")

    selected = review_model_choice(
        profile, make_hardware(vram_gb=0, ram_gb=8), choices,
        Console(color=False, stream=stream), runtime="llamacpp",
        interactive=interactive, stream=stream, input_fn=answer,
    )
    return selected, stream.getvalue(), seen


def test_interview_accepts_a_model_name_without_asking_for_specs():
    defaults = {q.key: q.default for q in QUESTIONS if q.key != "requested_model"}
    stream = io.StringIO()
    profile = run_interview(Console(color=False, stream=stream), defaults=defaults,
                            stream=stream, input_fn=lambda _: "Small Chat")
    assert profile.requested_model == "Small Chat"
    assert "specific model" in stream.getvalue()
    assert "How much RAM" not in stream.getvalue()


def test_fit_is_explained_before_user_keeps_the_model(choices):
    profile, transcript, seen = review(UseCaseProfile(requested_model="Small Chat"), choices)
    assert profile.requested_model == "small-chat"
    assert seen and "Model review: Small Chat" in seen[0]
    assert "tokens" in seen[0] and "Q" in seen[0]
    assert "fixed" in transcript


def test_poor_fit_names_an_alternative_without_changing_the_choice(choices):
    profile, transcript, _ = review(UseCaseProfile(requested_model="large-chat"), choices)
    assert profile.requested_model == "large-chat"
    assert "poor fit" in transcript
    assert "Alternative: Small Chat (small-chat)" in transcript
    assert "GiB" in transcript
    assert transcript.count("Alternative: Small Chat") == 1


def test_user_can_choose_and_review_a_better_alternative(choices):
    profile, transcript, seen = review(UseCaseProfile(requested_model="large-chat"),
                                       choices, ["small-chat", "keep"])
    assert profile.requested_model == "small-chat"
    assert "Model review: Small Chat" in transcript
    assert len(seen) == 2


def test_user_can_return_to_automatic_selection(choices):
    profile, _, _ = review(UseCaseProfile(requested_model="large-chat"), choices, ["auto"])
    assert profile.requested_model is None


def test_unknown_model_can_be_corrected_in_interactive_review(choices):
    profile, transcript, _ = review(UseCaseProfile(requested_model="missing"), choices,
                                   ["Small Chat", "keep"])
    assert profile.requested_model == "small-chat"
    assert "missing" in transcript


def test_noninteractive_review_keeps_explicit_model_and_does_not_read_input(choices):
    profile, transcript, seen = review(UseCaseProfile(requested_model="large-chat"),
                                       choices, interactive=False)
    assert profile.requested_model == "large-chat"
    assert not seen
    assert "Alternative:" in transcript


def test_unknown_noninteractive_model_reports_a_clean_error(choices):
    with pytest.raises(LlmsetupError):
        review(UseCaseProfile(requested_model="missing"), choices, interactive=False)


@pytest.mark.parametrize("value", [42, False, [], {}])
def test_invalid_model_preferences_are_rejected(value):
    with pytest.raises(LlmsetupError, match="requested_model"):
        profile_from_answers({"requested_model": value})


def test_saved_model_preference_round_trips(tmp_path):
    original = UseCaseProfile(tasks=[Task.CODING], requested_model="small-chat")
    path = tmp_path / "profile.yaml"
    save_profile(original, path)
    assert profile_from_answers(load_answers(path)) == original
    assert profile_from_answers({"requested_model": "  "}).requested_model is None


@pytest.fixture
def offline_cli(monkeypatch):
    monkeypatch.setenv("LLMSETUP_AUTO_UPDATE", "0")
    monkeypatch.setattr("llmsetup.cli.detect_hardware", lambda **kw: make_hardware())
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def unexpected_input(*args):
        raise AssertionError("scripted setup must not prompt")

    monkeypatch.setattr("builtins.input", unexpected_input)


def test_plan_flag_explains_and_keeps_a_model_name(offline_cli, capsys):
    assert main(["plan", "--model", "Qwen3 8B", "--task", "chat", "--yes",
                 "--runtime", "llamacpp", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["candidates"][0]["model"] == "qwen3-8b"
    assert "Model review: Qwen3 8B" in captured.err


def test_setup_saves_model_choice_and_retains_full_catalog(offline_cli, capsys, tmp_path,
                                                         monkeypatch):
    from llmsetup.pipeline import Pipeline

    original = Pipeline.run
    received = []

    def run(self):
        received.append((self.profile.requested_model, len(self.catalog)))
        return original(self)

    monkeypatch.setattr(Pipeline, "run", run)
    saved = tmp_path / "answers.yaml"
    assert main(["setup", "--model", "Qwen3 8B", "--quant", "Q4_K_M", "--task", "chat",
                 "--yes", "--dry-run", "--json", "--save-profile", str(saved)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert received[0][0] == "qwen3-8b" and received[0][1] > 1
    assert payload["config"]["model_id"] == "qwen3-8b"
    assert payload["config"]["quant"] == "Q4_K_M"
    assert load_answers(saved)["requested_model"] == "qwen3-8b"


def test_answers_file_keeps_model_and_cli_flag_overrides_it(offline_cli, capsys, tmp_path):
    saved = tmp_path / "answers.yaml"
    saved.write_text("tasks: [chat]\nrequested_model: qwen3-8b\n")
    assert main(["plan", "--answers", str(saved), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"][0]["model"] == "qwen3-8b"
    assert main(["plan", "--answers", str(saved), "--model", "qwen3-4b", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"][0]["model"] == "qwen3-4b"


def test_unknown_model_flag_returns_error_without_traceback(offline_cli, capsys):
    assert main(["setup", "--model", "does-not-exist", "--yes", "--dry-run"]) == 1
    captured = capsys.readouterr()
    assert "does-not-exist" in captured.err
    assert "Traceback" not in captured.err


def test_doctor_retries_the_saved_model_with_the_full_catalog(offline_cli, capsys, monkeypatch):
    from llmsetup.pipeline import Pipeline
    from llmsetup.util.fs import app_dir

    assert main(["setup", "--model", "qwen3-8b", "--yes", "--dry-run", "--json"]) == 0
    capsys.readouterr()
    path = app_dir() / "last-setup.json"
    saved = json.loads(path.read_text())
    # The last working model can differ after an exhausted-budget fallback.
    saved["profile"]["requested_model"] = "qwen3-32b"
    path.write_text(json.dumps(saved))
    original = Pipeline.run
    received = []

    def run(self):
        received.append((self.profile.requested_model, len(self.catalog)))
        self.dry_run = True
        return original(self)

    monkeypatch.setattr(Pipeline, "run", run)
    assert main(["doctor"]) == 1  # A dry run has no passing verification checks.
    assert received[0][0] == "qwen3-8b" and received[0][1] > 1
