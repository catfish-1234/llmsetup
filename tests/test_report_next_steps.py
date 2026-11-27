"""A verified setup ends by naming the commands that use it; an unverified one does not."""

from __future__ import annotations

import sys

import pytest

from llmsetup.types import CheckResult, CheckStatus, Severity, Task, UseCaseProfile


def _result(catalog, runtime: str, status: CheckStatus):
    from conftest import make_hardware

    from llmsetup.pipeline import Pipeline
    from llmsetup.util.log import Console

    result = Pipeline(UseCaseProfile(tasks=[Task.CODING]), make_hardware(),
                      catalog, Console(quiet=True), dry_run=True).run()
    result.plan.runtime = runtime
    result.checks = [CheckResult("chat", status, Severity.CRITICAL, "replied")]
    return result


def _report(result, capsys) -> str:
    from llmsetup.report.render import render_terminal
    from llmsetup.util.log import Console

    render_terminal(result, Console(stream=sys.stdout, color=False))
    return capsys.readouterr().out


def test_a_verified_setup_names_connect_chat_and_undo(catalog, capsys):
    out = _report(_result(catalog, "llamacpp", CheckStatus.PASS), capsys)

    use_it = out.split("Use it", 1)[1]
    for command in ("llmsetup start --detach", "llmsetup connect claude-code",
                    "llmsetup chat", "llmsetup undo"):
        assert command in use_it


@pytest.mark.parametrize("runtime", ["ollama", "lmstudio"])
def test_a_shared_daemon_is_not_told_to_start_a_server(catalog, capsys, runtime):
    out = _report(_result(catalog, runtime, CheckStatus.PASS), capsys)

    assert "llmsetup connect claude-code" in out
    assert "start --detach" not in out


def test_a_failed_setup_points_at_fixing_it_not_using_it(catalog, capsys):
    out = _report(_result(catalog, "llamacpp", CheckStatus.FAIL), capsys)

    assert "Use it" not in out
    assert "llmsetup connect" not in out


def test_the_launch_command_is_quoted_for_this_platforms_shell(catalog, capsys):
    """POSIX single quotes are not quotes to cmd.exe, so Windows gets its own quoting."""
    from llmsetup.report.render import _spec
    from llmsetup.runtimes.registry import get_runtime
    from llmsetup.util.proc import format_command

    result = _result(catalog, "llamacpp", CheckStatus.PASS)
    command = get_runtime("llamacpp").launch_command(_spec(result), result.config)
    out = _report(result, capsys)

    assert f"  {format_command(command)}\n" in out


def test_an_untested_plan_does_not_offer_to_connect(catalog, capsys):
    from conftest import make_hardware

    from llmsetup.pipeline import Pipeline
    from llmsetup.util.log import Console

    plan_only = Pipeline(UseCaseProfile(tasks=[Task.CODING]), make_hardware(),
                         catalog, Console(quiet=True), dry_run=True).run()
    out = _report(plan_only, capsys)

    assert "Use it" not in out
