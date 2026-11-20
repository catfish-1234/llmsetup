"""scripts/release.py: the built-in release command, against a real repository."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
spec = importlib.util.spec_from_file_location("release_script", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def changelog(unreleased, versions):
    body = "# Changelog\n\n## [Unreleased]\n" + (f"\n{unreleased}\n" if unreleased else "") + "\n"
    return body + "\n".join(
        f"## [{v}]{f' - {d}' if d else ''}\n\n### Added\n\n- Things in {v}.\n" for v, d in versions)


COMMITS = {"0.4.0": "c040", "0.3.0": "c030", "0.2.0": "c020"}


def test_reads_versions_and_the_unreleased_section():
    text = changelog("### Fixed\n\n- A bug.", [("0.4.0", "2026-11-19"), ("0.3.0", None)])
    assert release.changelog_versions(text) == [("0.4.0", "2026-11-19"), ("0.3.0", None)]
    assert release.unreleased_entries(text) == "### Fixed\n\n- A bug."
    assert release.unreleased_entries(changelog("", [("0.4.0", "2026-11-19")])) == ""


def test_orders_versions_numerically_with_prereleases_first():
    versions = ["0.10.0", "0.9.0", "1.0.0", "1.0.0-rc.1"]
    assert sorted(versions, key=release.version_key) == ["0.9.0", "0.10.0", "1.0.0-rc.1", "1.0.0"]


def test_tags_head_and_lists_older_gaps_without_backfill():
    plan = release.plan_release("0.4.0", changelog("", [("0.4.0", "2026-11-19"), ("0.3.0", "2026-11-18")]),
                                set(), "HEAD", COMMITS.get)
    assert plan["current"] == {"version": "0.4.0", "tag": "v0.4.0", "commit": "HEAD"}
    assert plan["backfill"] == []
    assert "Older versions without a tag: 0.3.0" in " ".join(plan["notes"])


def test_backfills_oldest_first_at_each_release_commit():
    plan = release.plan_release(
        "0.4.0", changelog("", [("0.4.0", "2026-11-19"), ("0.3.0", "2026-11-18"), ("0.2.0", "2026-11-17")]),
        {"v0.2.0"}, "HEAD", COMMITS.get, backfill=True)
    assert [(b["tag"], b["commit"]) for b in plan["backfill"]] == [("v0.3.0", "c030")]


def test_unreleased_entries_move_the_tag_to_the_release_commit():
    plan = release.plan_release("0.4.0", changelog("- Not yet.", [("0.4.0", "2026-11-19")]), set(), "HEAD", COMMITS.get)
    assert plan["current"]["commit"] == "c040"


def test_refuses_an_undescribed_or_undated_version():
    assert "no \"## [0.5.0]\"" in release.plan_release("0.5.0", changelog("", [("0.4.0", "2026-11-19")]),
                                                       set(), "H", COMMITS.get)["errors"][0]
    assert "no release date" in release.plan_release("0.4.0", changelog("", [("0.4.0", None)]),
                                                    set(), "H", COMMITS.get)["errors"][0]


def test_parses_remotes_and_project_versions():
    assert release.github_repo("git@github.com:catfish-1234/llmsetup.git") == "catfish-1234/llmsetup"
    assert release.github_repo("https://example.com/a/b") is None
    assert release.project_version('[project]\nname = "x"\nversion = "0.4.0"\n') == "0.4.0"


# --------------------------------------------------------------------------- #
# end to end, against a bare "origin"
# --------------------------------------------------------------------------- #


@pytest.fixture
def repo(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)

    def git(*args, cwd=work):
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    git("config", "user.email", "release@test")
    git("config", "user.name", "Release Test")
    git("checkout", "-q", "-b", "main")

    def commit(version, unreleased, versions):
        (work / "pyproject.toml").write_text(f'[project]\nname = "demo"\nversion = "{version}"\n')
        (work / "CHANGELOG.md").write_text(changelog(unreleased, versions))
        git("add", "-A")
        git("commit", "-qm", f"Release {version}")
        return git("rev-parse", "HEAD")

    def run(*args):
        env = {**os.environ, "RELEASE_NO_GH": "1", "GIT_TERMINAL_PROMPT": "0"}
        return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=work, env=env,
                              capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def remote_tags():
        return [t for t in git("tag", "-l", cwd=origin).splitlines() if t]

    return {"git": git, "commit": commit, "run": run, "tags": remote_tags, "origin": origin}


def test_pushes_the_current_version_and_backfills_in_order(repo):
    first = repo["commit"]("0.3.0", "", [("0.3.0", "2026-11-18")])
    head = repo["commit"]("0.4.0", "", [("0.4.0", "2026-11-19"), ("0.3.0", "2026-11-18")])
    repo["git"]("push", "-q", "origin", "main")

    dry = repo["run"]("--dry-run", "--backfill")
    assert dry.returncode == 0, dry.stderr
    assert "v0.3.0" in dry.stdout and "(current, pushed last)" in dry.stdout
    assert repo["tags"]() == []

    result = repo["run"]("--backfill", "--yes")
    assert result.returncode == 0, result.stderr
    assert result.stdout.index("Pushed v0.3.0") < result.stdout.index("Pushed v0.4.0")
    assert repo["tags"]() == ["v0.3.0", "v0.4.0"]
    assert repo["git"]("rev-parse", "v0.3.0^{commit}", cwd=repo["origin"]) == first
    assert repo["git"]("rev-parse", "v0.4.0^{commit}", cwd=repo["origin"]) == head
    assert "Nothing to release" in repo["run"]("--yes").stdout


def test_refuses_unsafe_states(repo):
    repo["commit"]("0.4.0", "", [("0.4.0", "2026-11-19")])
    repo["git"]("push", "-q", "origin", "main")

    unconfirmed = repo["run"]()
    assert unconfirmed.returncode == 1 and "rerun with --yes" in unconfirmed.stderr

    repo["commit"]("0.5.0", "", [("0.5.0", "2026-11-20"), ("0.4.0", "2026-11-19")])
    assert "HEAD is not origin/main" in repo["run"]("--yes").stderr

    repo["git"]("push", "-q", "origin", "main")
    pyproject = repo["git"]("rev-parse", "--show-toplevel")
    Path(pyproject, "pyproject.toml").write_text('[project]\nversion = "9.9.9"\n')
    assert "uncommitted changes" in repo["run"]("--yes").stderr
    assert repo["tags"]() == []
