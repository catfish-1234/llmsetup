"""Tag and publish a release from a clean, up-to-date main: ``python scripts/release.py``.

Pushing a v* tag runs .github/workflows/release.yml, which builds the distributions, checks the
tag against the package version, publishes to PyPI, and creates the GitHub release. This command
is the safe way to push that tag::

    python scripts/release.py              tag the version in pyproject.toml and push it
    python scripts/release.py --dry-run    show what would be tagged and pushed, change nothing
    python scripts/release.py --backfill   also tag older CHANGELOG versions never tagged
    python scripts/release.py --yes        do not ask for confirmation (for scripts)

The current version is tagged at HEAD only when CHANGELOG.md has no unreleased entries; otherwise
HEAD holds work that version does not describe, and the tag goes on the commit that introduced the
version. Each tag is pushed on its own, because GitHub creates no workflow events when more than
three tags arrive in one push, and older versions go first so the current one is published last.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

WORKFLOW = "release.yml"
VERSION_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\](?:\s*-\s*(\d{4}-\d{2}-\d{2}))?", re.M)


# --------------------------------------------------------------------------- #
# pure helpers (tested directly)
# --------------------------------------------------------------------------- #


def changelog_versions(text: str) -> List[Tuple[str, Optional[str]]]:
    """(version, date) for every ``## [x.y.z] - date`` heading, in file order."""
    return [(m.group(1), m.group(2)) for m in VERSION_HEADING.finditer(text)]


def unreleased_entries(text: str) -> str:
    """The body of ``## [Unreleased]``, without link references; empty when nothing is unreleased."""
    match = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
    if not match:
        return ""
    return re.sub(r"^\[[^\]]+\]:.*$", "", match.group(1), flags=re.M).strip()


def version_key(version: str) -> Tuple[int, int, int, int, str]:
    core, _, pre = version.partition("-")
    major, minor, patch = (int(part) for part in core.split("."))
    # A release sorts above its prereleases.
    return (major, minor, patch, 0 if pre else 1, pre)


def plan_release(version: str, changelog: str, tagged: Set[str], head: str,
                 release_commit: Callable[[str], Optional[str]], backfill: bool = False) -> Dict:
    """What to tag: ``{"current", "backfill", "notes", "errors"}``; backfill is oldest first."""
    errors: List[str] = []
    notes: List[str] = []
    versions = changelog_versions(changelog)
    dates = dict(versions)
    current = None
    if version not in dates:
        errors.append(f'CHANGELOG.md has no "## [{version}]" section for the version in pyproject.toml.')
    elif not dates[version]:
        errors.append(f'CHANGELOG.md\'s "## [{version}]" section has no release date.')
    elif f"v{version}" in tagged:
        notes.append(f"v{version} is already tagged; nothing new to release for the current version.")
    elif unreleased_entries(changelog):
        commit = release_commit(version)
        if commit is None:
            errors.append(f"CHANGELOG.md has unreleased entries and no commit introduces version {version}.")
        else:
            current = {"version": version, "tag": f"v{version}", "commit": commit}
            notes.append(f"CHANGELOG.md has unreleased entries, so v{version} goes on the commit "
                         "that introduced it, not HEAD.")
    else:
        current = {"version": version, "tag": f"v{version}", "commit": head}

    missing = sorted((v for v, _ in versions
                      if version_key(v) < version_key(version) and f"v{v}" not in tagged), key=version_key)
    older = []
    if backfill:
        for v in missing:
            commit = release_commit(v)
            if commit:
                older.append({"version": v, "tag": f"v{v}", "commit": commit})
            else:
                notes.append(f"v{v} is in CHANGELOG.md but no commit introduces it; skipped.")
    elif missing:
        notes.append(f"Older versions without a tag: {', '.join(missing)}. "
                     "Add --backfill to tag and publish them first.")
    return {"current": current, "backfill": older, "notes": notes, "errors": errors}


def github_repo(url: str) -> Optional[str]:
    match = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    return f"{match.group(1)}/{match.group(2)}" if match else None


def project_version(pyproject: str) -> Optional[str]:
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    return match.group(1) if match else None


# --------------------------------------------------------------------------- #
# git and GitHub
# --------------------------------------------------------------------------- #


class ReleaseError(Exception):
    pass


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ReleaseError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def try_git(*args: str) -> Optional[str]:
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def first_commit_with_version(version: str) -> Optional[str]:
    """The first commit on HEAD whose pyproject.toml carries ``version``."""
    pattern = f'^version = "{re.escape(version)}"'
    out = try_git("log", "--format=%H", "--reverse", "-G", pattern, "--", "pyproject.toml")
    for sha in (out or "").splitlines():
        if project_version(try_git("show", f"{sha}:pyproject.toml") or "") == version:
            return sha
    return None


def gh_available() -> bool:
    if os.environ.get("RELEASE_NO_GH") == "1" or not shutil.which("gh"):
        return False
    return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0


def failing_checks(repo: str, sha: str) -> Optional[List[str]]:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/commits/{sha}/check-runs?per_page=100", "--jq",
         '.check_runs[] | [.name, .status, (.conclusion // "")] | @tsv'],
        capture_output=True, text=True)
    if result.returncode != 0:
        return None
    failing = []
    for line in result.stdout.splitlines():
        name, status, conclusion = [*line.split("\t"), "", "", ""][:3]
        if status != "completed" or conclusion not in ("success", "skipped", "neutral"):
            failing.append(f"{name} ({conclusion or status})")
    return failing


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/release.py",
        description="Tag the version in pyproject.toml and push it, which runs the release "
                    "workflow (build, publish to PyPI, GitHub release).")
    parser.add_argument("-n", "--dry-run", action="store_true", help="show the plan and change nothing")
    parser.add_argument("--backfill", action="store_true",
                        help="also tag older CHANGELOG.md versions that were never tagged, oldest first")
    parser.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--branch", default="main", help="release branch (default: main)")
    args = parser.parse_args(argv)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != args.branch:
        raise ReleaseError(f"Releases are cut from {args.branch}; this checkout is on {branch}.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ReleaseError("The working tree has uncommitted changes. Commit or stash them first.")
    git("fetch", "--quiet", "--tags", "origin", args.branch)
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", f"origin/{args.branch}"):
        raise ReleaseError(f"HEAD is not origin/{args.branch}. Push or pull first, so the tag points at what is on GitHub.")

    version = project_version(Path("pyproject.toml").read_text(encoding="utf-8"))
    if not version:
        raise ReleaseError("pyproject.toml has no version.")
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    remote = try_git("ls-remote", "--tags", "origin") or ""
    tagged = {line.split("refs/tags/", 1)[1].removesuffix("^{}") for line in remote.splitlines() if "refs/tags/" in line}
    plan = plan_release(version, changelog, tagged, head, first_commit_with_version, backfill=args.backfill)

    for note in plan["notes"]:
        print(note)
    if plan["errors"]:
        raise ReleaseError("\n".join(plan["errors"]))
    items = plan["backfill"] + ([plan["current"]] if plan["current"] else [])
    if not items:
        print("Nothing to release.")
        return 0
    for item in items:
        local = try_git("rev-parse", "--verify", "--quiet", f"refs/tags/{item['tag']}^{{commit}}")
        if local and local != item["commit"]:
            raise ReleaseError(f"A local tag {item['tag']} already points at {local[:7]}, not {item['commit'][:7]}. "
                               f"Delete it with `git tag -d {item['tag']}` if it is stale.")

    repo = github_repo(git("remote", "get-url", "origin"))
    if plan["current"]:
        if repo and gh_available():
            failing = failing_checks(repo, plan["current"]["commit"])
            short = plan["current"]["commit"][:7]
            if failing is None:
                print("Could not read CI results for that commit; the release workflow still builds and checks it.")
            elif failing:
                message = f"CI is not green on {short}: {', '.join(failing)}."
                if not args.dry_run:
                    raise ReleaseError(f"{message} Wait for it, or fix it, before releasing.")
                print(f"Warning: {message} A real run would stop here.")
            else:
                print(f"CI is green on {short}.")
        else:
            print("GitHub CLI not available or not signed in: CI status not checked.")

    print("\nPlan (each tag publishes its version when pushed):")
    for item in items:
        subject = git("log", "-1", "--format=%s", item["commit"])
        suffix = "  (current, pushed last)" if item is plan["current"] else ""
        print(f"  {item['tag']:<10} {item['commit'][:7]}  {subject}{suffix}")
    if args.dry_run:
        print("\nDry run: nothing was tagged or pushed.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            raise ReleaseError("Not a terminal: rerun with --yes to confirm.")
        try:
            answer = input("\nCreate and push these tags? [y/N] ")
        except EOFError:
            # Windows reports the NUL device as a terminal, so stdin redirected from it passes the
            # check above and then has nothing to read. That is still "not a terminal".
            raise ReleaseError("Not a terminal: rerun with --yes to confirm.") from None
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 1

    for item in items:
        if not try_git("rev-parse", "--verify", "--quiet", f"refs/tags/{item['tag']}"):
            git("tag", "-a", item["tag"], "-m", f"Release {item['tag']}", item["commit"])
        # One tag per push: GitHub creates no events when more than three tags arrive at once.
        git("push", "--quiet", "origin", f"refs/tags/{item['tag']}")
        print(f"Pushed {item['tag']}.")
    if repo:
        print(f"\nRelease workflow: https://github.com/{repo}/actions/workflows/{WORKFLOW}")
    return 0


def main() -> int:
    try:
        return run()
    except ReleaseError as error:
        print(f"release: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
