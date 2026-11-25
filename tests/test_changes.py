"""The change journal: what llmsetup changed outside its own directory, and undoing it."""

from __future__ import annotations

import json

import pytest

from llmsetup import changes
from llmsetup.cli import main
from llmsetup.util.fs import link_or_copy


def _pending():
    return changes.pending_changes()


def test_a_created_file_is_removed_by_undo(tmp_path):
    target = tmp_path / "link.gguf"
    target.write_bytes(b"weights")
    with changes.run("setup"):
        changes.created("lm_studio", target)
    results = changes.undo(_pending())
    assert [r["status"] for r in results] == ["undone"]
    assert not target.exists()
    assert _pending() == []


def test_a_file_edited_since_is_kept_unless_forced(tmp_path):
    target = tmp_path / "model.json"
    target.write_text("{}")
    with changes.run("setup"):
        changes.created("jan", target)
    target.write_text('{"edited": true}')
    assert changes.undo(_pending())[0]["status"] == "kept"
    assert target.exists()
    # Kept changes stay pending, so they can be forced later.
    assert changes.undo(_pending(), force=True)[0]["status"] == "undone"
    assert not target.exists()


def test_a_created_folder_goes_only_once_empty(tmp_path):
    folder = tmp_path / "models" / "qwen3-8b"
    folder.mkdir(parents=True)
    stray = folder / "user-notes.txt"
    stray.write_text("mine")
    with changes.run("setup"):
        changes.created_dir("jan", folder)
    assert changes.undo(_pending())[0]["status"] == "kept"
    stray.unlink()
    assert changes.undo(_pending())[0]["status"] == "undone"
    assert not folder.exists()


def test_an_edited_config_is_restored_from_its_backup(tmp_path):
    config = tmp_path / "opencode.json"
    config.write_text('{"theme": "dark"}')
    with changes.run("connect"):
        backup = changes.backup_before_edit(config)
        config.write_text('{"theme": "dark", "provider": {}}')
        changes.modified("opencode", config, backup)
    assert changes.undo(_pending())[0]["status"] == "undone"
    assert config.read_text() == '{"theme": "dark"}'


def test_a_config_llmsetup_created_is_removed_and_an_edited_one_kept(tmp_path):
    fresh = tmp_path / "new.yml"
    with changes.run("connect"):
        fresh.write_text("model: x\n")
        changes.modified("aider", fresh, None)
    assert changes.undo(_pending())[0]["status"] == "undone"
    assert not fresh.exists()

    edited = tmp_path / "edited.json"
    edited.write_text("{}")
    with changes.run("connect"):
        backup = changes.backup_before_edit(edited)
        edited.write_text('{"a": 1}')
        changes.modified("zed", edited, backup)
    edited.write_text('{"a": 1, "user": true}')
    outcome = changes.undo(_pending())[0]
    assert outcome["status"] == "kept" and str(backup) in outcome["note"]
    assert json.loads(edited.read_text()) == {"a": 1, "user": True}


def test_link_or_copy_moves_an_unrelated_file_aside_and_undo_puts_it_back(tmp_path):
    source = tmp_path / "store.gguf"
    source.write_bytes(b"llmsetup weights")
    destination = tmp_path / "lmstudio" / "model.gguf"
    destination.parent.mkdir()
    destination.write_bytes(b"the user's own download")

    with changes.run("export"):
        _method, path = link_or_copy(source, destination)
        changes.created("lm_studio", path)
    assert destination.read_bytes() == b"llmsetup weights"
    aside = destination.with_name("model.gguf.llmsetup-backup")
    assert aside.read_bytes() == b"the user's own download"

    changes.undo(_pending())
    assert destination.read_bytes() == b"the user's own download"
    assert not aside.exists()
    assert source.read_bytes() == b"llmsetup weights"


def test_an_ollama_model_is_removed_through_ollama():
    removed = []
    with changes.run("export"):
        changes.ollama_model("llmsetup-qwen3-8b")
    results = changes.undo(_pending(), ollama_rm=lambda name: removed.append(name) or True)
    assert removed == ["llmsetup-qwen3-8b"] and results[0]["status"] == "undone"


def test_a_dry_run_changes_nothing(tmp_path):
    target = tmp_path / "link.gguf"
    target.write_bytes(b"x")
    with changes.run("setup"):
        changes.created("jan", target)
    assert changes.undo(_pending(), dry_run=True)[0]["status"] == "undone"
    assert target.exists() and len(_pending()) == 1


def test_jan_registration_is_fully_undone(tmp_path, monkeypatch, catalog):
    from llmsetup.interop import jan
    from llmsetup.types import RuntimeConfig

    monkeypatch.setenv("JAN_DATA_DIR", str(tmp_path / "jan"))
    (tmp_path / "jan" / "models").mkdir(parents=True)
    gguf = tmp_path / "store" / "qwen3-8b-Q4_K_M.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF" + b"\0" * 64)
    config = RuntimeConfig(runtime="llamacpp", model_id="qwen3-8b", quant="Q4_K_M", context_tokens=8192)

    with changes.run("export"):
        record = jan.register_in_jan(catalog.require("qwen3-8b"), gguf, config)
    assert record["ok"]
    assert {c["kind"] for c in _pending()} == {"created_dir", "created", "modified"}

    changes.undo(_pending())
    assert list((tmp_path / "jan" / "models").iterdir()) == []
    assert gguf.exists()


class TestCommands:
    def _two_runs(self, tmp_path):
        first, second = tmp_path / "first.txt", tmp_path / "second.txt"
        for path, command in ((first, "setup"), (second, "export")):
            path.write_text(command)
            with changes.run(command):
                changes.created("jan", path)
        return first, second

    def test_undo_reverses_only_the_newest_run(self, tmp_path):
        first, second = self._two_runs(tmp_path)
        assert main(["undo"]) == 0
        assert first.exists() and not second.exists()
        assert main(["undo"]) == 0
        assert not first.exists()

    def test_undo_all_and_dry_run(self, tmp_path):
        first, second = self._two_runs(tmp_path)
        assert main(["undo", "--all", "--dry-run"]) == 0
        assert first.exists() and second.exists()
        assert main(["undo", "--all"]) == 0
        assert not first.exists() and not second.exists()

    def test_changes_lists_runs_as_json(self, tmp_path, capsys):
        self._two_runs(tmp_path)
        assert main(["changes", "--json"]) == 0
        history = json.loads(capsys.readouterr().out)
        assert [group["command"] for group in history] == ["export", "setup"]
        assert all(group["pending"] == 1 for group in history)

    def test_undo_of_an_unknown_run_says_where_to_look(self, capsys):
        assert main(["undo", "--run", "nope"]) == 1
        assert "llmsetup changes" in capsys.readouterr().err


@pytest.fixture(autouse=True)
def _no_active_run():
    """Each test starts outside any run, as a fresh process would."""
    changes._current = None
    yield
    changes._current = None
