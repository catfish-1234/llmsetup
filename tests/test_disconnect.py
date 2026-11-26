"""`llmsetup disconnect`: take llmsetup's entries back out of a tool's config, keeping your edits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llmsetup import changes
from llmsetup.cli import main
from llmsetup.connect import ConnectRefused, Target, connect, disconnect
from llmsetup.runtimes.base import Endpoint
from llmsetup.types import RuntimeConfig


@pytest.fixture
def target(catalog):
    endpoint = Endpoint(base_url="http://127.0.0.1:11435", model="qwen3:8b", runtime="ollama")
    config = RuntimeConfig(runtime="ollama", model_id="qwen3-8b", quant="Q4_K_M", context_tokens=16384)
    return Target(endpoint=endpoint, spec=catalog.require("qwen3-8b"), config=config, make_default=True)


@pytest.fixture(autouse=True)
def _fresh_run():
    changes._current = None
    yield
    changes._current = None


def edit(path, change):
    """You, editing the file after llmsetup wrote it."""
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc, indent=2))


def test_removes_only_its_provider_and_keeps_what_you_changed_since(tmp_path, target):
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"theme": "dark", "provider": {"anthropic": {"options": {}}}}))
    connect("opencode", target, path=path)
    edit(path, lambda doc: doc.update(theme="light"))

    # undo keeps a file edited since llmsetup wrote it; disconnect works on it as it is now.
    assert changes.undo(changes.pending_changes())[0]["status"] == "kept"
    with changes.run("disconnect"):
        assert disconnect("opencode", path=path)["changed"]
    assert json.loads(path.read_text()) == {"$schema": "https://opencode.ai/config.json", "theme": "light",
                                            "provider": {"anthropic": {"options": {}}}}


def test_undo_reverses_a_disconnect(tmp_path, target):
    path = tmp_path / "settings.json"
    connect("zed", target, path=path)
    connected = path.read_text()
    with changes.run("disconnect"):
        disconnect("zed", path=path)
    assert json.loads(path.read_text()) == {}
    newest = changes.pending_changes()[-1]["run"]
    changes.undo([c for c in changes.pending_changes() if c["run"] == newest])
    assert path.read_text() == connected


def test_zed_keeps_a_model_you_added_beside_llmsetups(tmp_path, target):
    path = tmp_path / "settings.json"
    connect("zed", target, path=path)
    edit(path, lambda doc: doc["language_models"]["openai"]["available_models"].append(
        {"name": "mine", "display_name": "Mine", "max_tokens": 4096}))
    disconnect("zed", path=path)
    openai = json.loads(path.read_text())["language_models"]["openai"]
    assert openai["available_models"] == [{"name": "mine", "display_name": "Mine", "max_tokens": 4096}]
    assert openai["api_url"] == "http://127.0.0.1:11435/v1"  # still serving your model


def test_continue_and_aider(tmp_path, target):
    cont = tmp_path / "config.yaml"
    cont.write_text(yaml.safe_dump({"name": "me", "models": [{"name": "GPT", "provider": "openai"}]}))
    connect("continue", target, path=cont)
    disconnect("continue", path=cont)
    assert yaml.safe_load(cont.read_text())["models"] == [{"name": "GPT", "provider": "openai"}]

    aider = tmp_path / ".aider.conf.yml"
    aider.write_text(yaml.safe_dump({"dark-mode": True}))
    connect("aider", target, path=aider)
    disconnect("aider", path=aider)
    assert yaml.safe_load(aider.read_text()) == {"dark-mode": True}


def test_aider_keys_set_by_you_are_not_llmsetups(tmp_path):
    aider = tmp_path / ".aider.conf.yml"
    mine = {"openai-api-base": "http://127.0.0.1:9000/v1", "openai-api-key": "real", "model": "openai/x"}
    aider.write_text(yaml.safe_dump(mine))
    assert disconnect("aider", path=aider)["changed"] is False
    assert yaml.safe_load(aider.read_text()) == mine


def test_claude_code_leaves_a_gateway_you_configured(tmp_path, target):
    path = tmp_path / ".claude" / "settings.local.json"
    connect("claude_code", target, path=path)
    disconnect("claude_code", path=path)
    assert json.loads(path.read_text()) == {}

    gateway = {"env": {"ANTHROPIC_BASE_URL": "https://gateway.example.com", "ANTHROPIC_MODEL": "x"}}
    path.write_text(json.dumps(gateway))
    assert disconnect("claude_code", path=path)["changed"] is False


def test_qwen_code_drops_its_entry_key_and_selection(tmp_path, target):
    path = tmp_path / "settings.json"
    mine = {"id": "gpt-4o", "name": "GPT-4o", "envKey": "OPENAI_API_KEY", "baseUrl": "https://api.openai.com/v1"}
    path.write_text(json.dumps({"theme": "Dracula", "modelProviders": {"openai": [mine]}}))
    connect("qwen_code", target, path=path)
    disconnect("qwen_code", path=path)
    doc = json.loads(path.read_text())
    assert doc["modelProviders"] == {"openai": [mine]} and "env" not in doc
    assert "name" not in doc.get("model", {})
    assert doc["theme"] == "Dracula"


def test_goose_provider_file_is_removed_and_undo_brings_it_back(tmp_path, monkeypatch, target):
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    path = Path(connect("goose", target)["path"])
    written = path.read_text()
    with changes.run("disconnect"):
        assert disconnect("goose")["removed"]
    assert not (tmp_path / "config" / "custom_providers" / "llmsetup.json").exists()
    newest = changes.pending_changes()[-1]["run"]
    changes.undo([c for c in changes.pending_changes() if c["run"] == newest])
    assert path.read_text() == written


def test_goose_file_it_did_not_write_is_left(tmp_path):
    path = tmp_path / "llmsetup.json"
    path.write_text(json.dumps({"name": "corp"}))
    with pytest.raises(ConnectRefused, match="not llmsetup's"):
        disconnect("goose", path=path)
    assert path.exists()


def test_nothing_to_do_and_dry_run(tmp_path, target):
    path = tmp_path / "opencode.json"
    assert disconnect("opencode", path=path)["changed"] is False  # no file
    connect("opencode", target, path=path)
    before = path.read_text()
    result = disconnect("opencode", path=path, dry_run=True)
    assert result["changed"] and '"llmsetup"' not in result["content"]
    assert path.read_text() == before


def test_cli(tmp_path, capsys):
    path = tmp_path / "opencode.json"
    base = ["--base-url", "http://127.0.0.1:8080", "--model", "local", "--config", str(path)]
    assert main(["connect", "opencode", *base]) == 0
    assert main(["disconnect", "opencode", "--config", str(path)]) == 0
    assert "llmsetup" not in json.loads(path.read_text()).get("provider", {})
    assert main(["disconnect", "crush"]) == 1
    assert "only printed steps" in capsys.readouterr().err
