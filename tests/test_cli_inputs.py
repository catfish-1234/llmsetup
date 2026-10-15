"""Invalid profiles should produce useful CLI errors instead of tracebacks."""

import json

import pytest

from llmfit.cli import build_parser, main
from llmfit.interview.wizard import profile_from_answers
from llmfit.types import Task
from llmfit.util.errors import LlmfitError


@pytest.mark.parametrize("flag", ["--top", "--threads", "--users", "--context"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_nonpositive_cli_counts_are_usage_errors(flag, value, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["plan", flag, value])
    assert exc.value.code == 2
    assert "positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("answers", ["[]", "false", "profile: []", "profile: null",
                                    "context_tokens: nope", "context_tokens: .inf",
                                    "concurrent_users: many", "max_disk_gb: .nan",
                                    "needs_tools: maybe", "tasks: 42", "languages: 42",
                                    "languages: null", "languages: [42]"])
def test_bad_answers_file_reports_a_clean_error(tmp_path, capsys, answers):
    path = tmp_path / "answers.yaml"
    path.write_text(answers)
    assert main(["plan", "--answers", str(path), "--yes", "--offline"]) == 1
    assert capsys.readouterr().err


@pytest.mark.parametrize("value,expected", [("false", False), ("no", False),
                                           ("true", True), ("yes", True)])
def test_string_booleans_have_their_stated_meaning(value, expected):
    assert profile_from_answers({"needs_tools": value}).needs_tools is expected


def test_scalar_task_keeps_task_inference():
    from llmfit.interview.questions import infer_flags

    profile = profile_from_answers(infer_flags({"tasks": "agent"}))
    assert profile.tasks == [Task.AGENT]
    assert profile.needs_tools


def test_offline_flag_prevents_catalog_fetch(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("offline planning must not fetch a catalog")

    monkeypatch.setattr("llmfit.catalog.autoupdate.fetch_catalog", unexpected)
    assert main(["plan", "--offline", "--json", "--cpu-only", "--ram", "64",
                 "--task", "chat", "--yes"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"]


def test_json_mode_never_starts_the_interview(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("JSON output must not start an interactive wizard")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("llmfit.cli.run_interview", unexpected)
    assert main(["plan", "--json", "--offline", "--cpu-only", "--ram", "64"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"]


def test_answers_file_skips_interview_even_without_tasks(tmp_path, monkeypatch, capsys):
    path = tmp_path / "profile.yaml"
    path.write_text("context_tokens: 8192\n")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def unexpected(*args, **kwargs):
        pytest.fail("--answers is documented to skip questions")

    monkeypatch.setattr("llmfit.cli.run_interview", unexpected)
    assert main(["plan", "--answers", str(path), "--offline", "--cpu-only", "--ram", "64"]) == 0


def test_profile_numeric_errors_name_the_field():
    with pytest.raises(LlmfitError, match="concurrent_users"):
        profile_from_answers({"concurrent_users": "many"})
