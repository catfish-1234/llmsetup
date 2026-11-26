"""`llmsetup connect`: writing the verified endpoint into a tool's own config, reversibly."""

from __future__ import annotations

import json

import pytest
import yaml

from llmsetup import changes
from llmsetup.cli import main
from llmsetup.connect import ConnectRefused, Target, connect, strip_jsonc
from llmsetup.runtimes.base import Endpoint
from llmsetup.types import RuntimeConfig


@pytest.fixture
def target(catalog):
    endpoint = Endpoint(base_url="http://127.0.0.1:11435", model="qwen3:8b", runtime="ollama")
    config = RuntimeConfig(runtime="ollama", model_id="qwen3-8b", quant="Q4_K_M", context_tokens=16384)
    return Target(endpoint=endpoint, spec=catalog.require("qwen3-8b"), config=config)


@pytest.fixture(autouse=True)
def _fresh_run():
    changes._current = None
    yield
    changes._current = None


def test_opencode_is_added_beside_existing_providers_and_undo_restores_it(tmp_path, target):
    path = tmp_path / "opencode.json"
    original = json.dumps({"theme": "dark", "provider": {"anthropic": {"options": {}}}}, indent=2)
    path.write_text(original)

    with changes.run("connect"):
        result = connect("opencode", target, path=path)
    doc = json.loads(path.read_text())
    assert result["changed"] and doc["theme"] == "dark" and "anthropic" in doc["provider"]
    assert doc["provider"]["llmsetup"]["options"]["baseURL"] == "http://127.0.0.1:11435/v1"
    assert "model" not in doc  # only with --default

    changes.undo(changes.pending_changes())
    assert path.read_text() == original


def test_connecting_twice_changes_nothing_the_second_time(tmp_path, target):
    path = tmp_path / "opencode.json"
    connect("opencode", target, path=path)
    assert connect("opencode", target, path=path)["changed"] is False


def test_default_makes_it_the_tools_default_model(tmp_path, target):
    path = tmp_path / "opencode.json"
    target.make_default = True
    connect("opencode", target, path=path)
    assert json.loads(path.read_text())["model"] == "llmsetup/qwen3:8b"


def test_a_file_with_comments_is_left_alone_unless_rewrite(tmp_path, target):
    path = tmp_path / "settings.json"
    original = '{\n  // my theme\n  "theme": "One Dark",\n}\n'
    path.write_text(original)
    with pytest.raises(ConnectRefused) as refused:
        connect("zed", target, path=path)
    assert "comments" in str(refused.value) and '"available_models"' in str(refused.value)
    assert path.read_text() == original

    connect("zed", target, path=path, rewrite=True)
    doc = json.loads(path.read_text())
    assert doc["theme"] == "One Dark"
    assert doc["language_models"]["openai"]["available_models"][0]["max_tokens"] == 16384


def test_zed_will_not_repoint_a_real_openai_provider(tmp_path, target):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"language_models": {"openai": {"api_url": "https://api.example.com/v1"}}}))
    with pytest.raises(ConnectRefused, match=r"api.example.com"):
        connect("zed", target, path=path)


def test_continue_keeps_your_models_and_replaces_only_its_own(tmp_path, target):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"name": "mine", "models": [{"name": "GPT", "provider": "openai"}]}))
    connect("continue", target, path=path)
    connect("continue", target, path=path)
    names = [m["name"] for m in yaml.safe_load(path.read_text())["models"]]
    assert names[0] == "GPT"
    assert sum(name.endswith("(llmsetup)") for name in names) == len(set(names) - {"GPT"})


def test_aider_points_at_the_local_endpoint(tmp_path, target):
    path = tmp_path / ".aider.conf.yml"
    connect("aider", target, path=path)
    doc = yaml.safe_load(path.read_text())
    assert doc["model"] == "openai/qwen3:8b" and doc["openai-api-base"].endswith("/v1")


def test_a_new_config_and_its_folders_are_removed_by_undo(tmp_path, target):
    path = tmp_path / "xdg" / "opencode" / "opencode.json"
    with changes.run("connect"):
        connect("opencode", target, path=path)
    assert path.exists()
    changes.undo(changes.pending_changes())
    assert not (tmp_path / "xdg").exists()


def test_dry_run_writes_nothing(tmp_path, target):
    path = tmp_path / "opencode.json"
    result = connect("opencode", target, path=path, dry_run=True)
    assert '"llmsetup"' in result["content"] and not path.exists()


def test_strip_jsonc_keeps_slashes_inside_strings():
    text = '{"url": "http://x//y", /* note */ "a": [1, 2,], // tail\n "b": "/*not a comment*/"}'
    assert json.loads(strip_jsonc(text)) == {"url": "http://x//y", "a": [1, 2], "b": "/*not a comment*/"}


class TestCommand:
    def test_connect_through_the_cli_and_undo(self, tmp_path, capsys):
        path = tmp_path / "opencode.json"
        args = ["connect", "opencode", "--base-url", "http://127.0.0.1:8080", "--model", "local",
                "--config", str(path)]
        assert main(args) == 0
        assert json.loads(path.read_text())["provider"]["llmsetup"]["models"]["local"]
        assert main(["undo"]) == 0
        assert not path.exists()

    def test_a_ui_only_tool_gets_the_steps(self, capsys):
        assert main(["connect", "open-webui", "--base-url", "http://127.0.0.1:8080", "--model", "local"]) == 0
        assert "host.docker.internal" in capsys.readouterr().out

    def test_an_unknown_tool_lists_the_known_ones(self, capsys):
        assert main(["connect", "notepad", "--base-url", "http://127.0.0.1:8080", "--model", "x"]) == 1
        assert "opencode" in capsys.readouterr().err


class TestClaudeCodeAndCodex:
    def test_claude_code_without_project_prints_the_launch_command_and_writes_nothing(self, tmp_path, capsys,
                                                                                    monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main(["connect", "claude-code", "--base-url", "http://127.0.0.1:8080", "--model", "local"]) == 0
        out = capsys.readouterr().out
        assert "ANTHROPIC_BASE_URL=http://127.0.0.1:8080 " in out and "ANTHROPIC_MODEL=local" in out
        assert "/v1" not in out.split("ANTHROPIC_BASE_URL=")[1].split()[0]
        assert not (tmp_path / ".claude").exists()

    def test_claude_code_for_one_project_merges_its_env_and_undo_restores_it(self, tmp_path):
        path = tmp_path / ".claude" / "settings.local.json"
        path.parent.mkdir()
        original = json.dumps({"permissions": {"allow": ["Bash(ls)"]}, "env": {"MY_VAR": "1"}})
        path.write_text(original)
        args = ["connect", "claude-code", "--base-url", "http://127.0.0.1:8080", "--model", "local",
                "--config", str(path), "--api-key", "s3cret"]
        assert main(args) == 0
        doc = json.loads(path.read_text())
        assert doc["permissions"] == {"allow": ["Bash(ls)"]} and doc["env"]["MY_VAR"] == "1"
        assert doc["env"]["ANTHROPIC_AUTH_TOKEN"] == "s3cret"
        assert doc["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "local"
        assert main(["undo"]) == 0
        assert path.read_text() == original

    def test_claude_code_warns_when_the_context_is_too_small_for_its_prompts(self, target):
        from llmsetup.connect import claude_code_steps

        target.config.context_tokens = 8192
        assert "--set context_tokens=65536" in claude_code_steps(target)["content"]
        target.config.context_tokens = 65536
        assert "--set context_tokens" not in claude_code_steps(target)["content"]

    def test_codex_gets_a_responses_provider_and_a_caveat(self, capsys):
        assert main(["connect", "codex", "--base-url", "http://127.0.0.1:8080", "--model", "local"]) == 0
        out = capsys.readouterr().out
        assert 'wire_api = "responses"' in out and 'base_url = "http://127.0.0.1:8080/v1"' in out
        assert "codex --profile llmsetup" in out


class TestQwenCodeGooseAndCrush:
    def test_qwen_code_lists_the_model_beside_yours_and_undo_restores_it(self, tmp_path, target):
        path = tmp_path / ".qwen" / "settings.json"
        path.parent.mkdir()
        mine = {"id": "gpt-4o", "name": "GPT-4o", "envKey": "OPENAI_API_KEY", "baseUrl": "https://api.openai.com/v1"}
        original = json.dumps({"theme": "Dracula", "modelProviders": {"openai": [mine]}})
        path.write_text(original)

        with changes.run("connect"):
            connect("qwen_code", target, path=path)
        doc = json.loads(path.read_text())
        assert doc["theme"] == "Dracula" and doc["modelProviders"]["openai"][0] == mine
        ours = doc["modelProviders"]["openai"][1]
        assert ours["id"] == "qwen3:8b" and ours["baseUrl"] == "http://127.0.0.1:11435/v1"
        assert ours["generationConfig"]["contextWindowSize"] == 16384
        # A server without a key accepts any placeholder, which Qwen Code reads from its env block.
        assert doc["env"][ours["envKey"]] == "llmsetup"
        assert "model" not in doc and "security" not in doc  # only with --default

        changes.undo(changes.pending_changes())
        assert path.read_text() == original

    def test_qwen_code_default_selects_it_and_a_second_connect_replaces_its_own_entry(self, tmp_path, target):
        path = tmp_path / "settings.json"
        target.make_default = True
        connect("qwen_code", target, path=path)
        target.endpoint.base_url = "http://127.0.0.1:9000"
        connect("qwen_code", target, path=path)
        doc = json.loads(path.read_text())
        assert [m["baseUrl"] for m in doc["modelProviders"]["openai"]] == ["http://127.0.0.1:9000/v1"]
        assert doc["security"]["auth"]["selectedType"] == "openai" and doc["model"]["name"] == "qwen3:8b"

    def test_qwen_code_will_not_shadow_a_model_you_listed_yourself(self, tmp_path, target):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"modelProviders": {"openai": [{"id": "qwen3:8b", "name": "Mine"}]}}))
        with pytest.raises(ConnectRefused, match="did not add"):
            connect("qwen_code", target, path=path)

    def test_goose_gets_its_own_provider_file_which_undo_removes(self, tmp_path, monkeypatch, target):
        monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path / "goose"))
        with changes.run("connect"):
            result = connect("goose", target)
        path = tmp_path / "goose" / "config" / "custom_providers" / "llmsetup.json"
        assert result["path"] == str(path)
        doc = json.loads(path.read_text())
        assert doc["name"] == "llmsetup" and doc["engine"] == "openai"
        assert doc["base_url"] == "http://127.0.0.1:11435/v1"
        assert doc["models"] == [{"name": "qwen3:8b", "context_limit": 16384}]
        assert doc["requires_auth"] is False and doc["api_key_env"] == ""

        changes.undo(changes.pending_changes())
        assert not (tmp_path / "goose").exists()

    def test_goose_reads_a_servers_key_from_the_environment_not_the_file(self, tmp_path, target):
        target.endpoint.headers["Authorization"] = "Bearer s3cret"
        path = tmp_path / "llmsetup.json"
        connect("goose", target, path=path)
        doc = json.loads(path.read_text())
        assert doc["requires_auth"] is True and doc["api_key_env"] == "LLMSETUP_API_KEY"
        assert "s3cret" not in path.read_text()

    def test_goose_leaves_a_provider_file_it_did_not_write(self, tmp_path, target):
        path = tmp_path / "llmsetup.json"
        path.write_text(json.dumps({"name": "corp", "engine": "openai"}))
        with pytest.raises(ConnectRefused, match="not llmsetup's"):
            connect("goose", target, path=path)

    def test_goose_has_no_project_file(self, target):
        with pytest.raises(ConnectRefused, match="per-project"):
            connect("goose", target, project=True)

    def test_goose_through_the_cli_says_how_to_start_it(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
        assert main(["connect", "goose", "--base-url", "http://127.0.0.1:8080", "--model", "local"]) == 0
        assert "GOOSE_PROVIDER=llmsetup GOOSE_MODEL=local goose session" in capsys.readouterr().out

    def test_crush_gets_the_crushrc_lines(self, capsys):
        assert main(["connect", "crush", "--base-url", "http://127.0.0.1:8080", "--model", "local"]) == 0
        out = capsys.readouterr().out
        assert 'provider add llmsetup --type completions --base-url "http://127.0.0.1:8080/v1"' in out
        assert "model large llmsetup/local" in out and "--api-key" not in out

    def test_every_known_tool_is_listed_with_what_connect_does(self, capsys):
        assert main(["connect", "--list"]) == 0
        out = capsys.readouterr().out
        for tool in ("opencode", "qwen_code", "goose", "crush", "claude_code", "codex"):
            assert f"  {tool} " in out
