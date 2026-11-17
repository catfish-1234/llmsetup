"""`llmsetup start`: rerun exactly the command a verified setup tested."""

import json
import socket

import pytest
from conftest import make_hardware

from llmsetup import cli
from llmsetup.launch import launch_record, load_launch, start
from llmsetup.planner.plan import build_plan
from llmsetup.tuning.tuner import tune
from llmsetup.types import CheckResult, CheckStatus, SetupResult, Severity, Task, UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.fs import app_dir, read_json
from llmsetup.util.log import Console


def result_for(catalog, runtime, passed=True):
    hardware = make_hardware(vram_gb=24, ram_gb=64)
    hardware.runtime_availability = {runtime: True}
    profile = UseCaseProfile(tasks=[Task.CHAT], context_tokens=8192)
    plan = build_plan(profile, hardware, catalog.restrict_to("qwen3-8b"), runtime=runtime)
    config = tune(plan.best, hardware, profile, plan.runtime, catalog=catalog).config
    config.model_path = "/models/qwen3-8b-Q4_K_M.gguf"
    status = CheckStatus.PASS if passed else CheckStatus.FAIL
    checks = [CheckResult("completion", status, Severity.CRITICAL)]
    return SetupResult(profile, hardware, plan, config, checks=checks)


def test_a_verified_run_records_the_command_it_tested(catalog):
    record = launch_record(result_for(catalog, "llamacpp"))
    assert record is not None
    assert record["runtime"] == "llamacpp"
    assert record["shared_daemon"] is False
    assert "/models/qwen3-8b-Q4_K_M.gguf" in record["command"]
    assert record["base_url"].startswith("http://127.0.0.1:")


def test_an_unverified_or_failed_run_records_nothing(catalog):
    assert launch_record(result_for(catalog, "llamacpp", passed=False)) is None
    untested = result_for(catalog, "llamacpp")
    untested.checks = []
    assert launch_record(untested) is None


def test_save_last_writes_the_record_and_start_reads_it(catalog, capsys):
    cli._save_last(result_for(catalog, "llamacpp"))
    saved = read_json(app_dir() / "last-setup.json")
    assert saved["launch"]["command"]
    assert cli.main(["start", "--print"]) == 0
    printed = capsys.readouterr().out.strip()
    assert "qwen3-8b-Q4_K_M.gguf" in printed
    assert cli.main(["start", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["runtime"] == "llamacpp"


def test_start_explains_a_failed_or_old_setup(tmp_path):
    with pytest.raises(LlmsetupError, match="did not pass verification"):
        load_launch({"succeeded": False}, tmp_path / "s.json")
    with pytest.raises(LlmsetupError, match="older llmsetup") as error:
        load_launch({"succeeded": True}, tmp_path / "s.json")
    assert "llmsetup doctor" in error.value.hint
    for bad in ({"command": []}, {"command": "llama-server -m x"}, {"command": ["", "x"]}):
        with pytest.raises(LlmsetupError, match="malformed"):
            load_launch({"launch": bad}, tmp_path / "s.json")


def test_start_without_a_saved_setup_says_how_to_make_one():
    assert cli.main(["start"]) == 1


def test_a_shared_daemon_has_nothing_to_start(catalog):
    record = {"runtime": "ollama", "served_model": "llmsetup-qwen3-8b", "base_url": "http://127.0.0.1:11434",
              "command": ["ollama", "run", "llmsetup-qwen3-8b"], "shared_daemon": True}
    ran = []
    assert start(record, Console(quiet=True), runner=lambda cmd: ran.append(cmd) or 0) == 0
    assert ran == []


def test_start_runs_the_recorded_command_in_the_foreground(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    record = {"runtime": "llamacpp", "served_model": "qwen3-8b", "base_url": f"http://127.0.0.1:{port}",
              "command": [str(binary), "-m", "/models/x.gguf", "--port", str(port)], "shared_daemon": False}
    ran = []
    assert start(record, Console(quiet=True), runner=lambda cmd: ran.append(cmd) or 7) == 7
    assert ran == [record["command"]]


def test_start_refuses_a_busy_port_and_a_missing_binary(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        record = {"runtime": "llamacpp", "served_model": "m", "base_url": f"http://127.0.0.1:{port}",
                  "command": [str(binary)], "shared_daemon": False}
        with pytest.raises(LlmsetupError, match="already in use") as error:
            start(record, Console(quiet=True), runner=lambda cmd: 0)
        assert "llmsetup verify" in error.value.hint
    record["command"] = [str(tmp_path / "gone")]
    with pytest.raises(LlmsetupError, match="no longer installed"):
        start(record, Console(quiet=True), runner=lambda cmd: 0)
