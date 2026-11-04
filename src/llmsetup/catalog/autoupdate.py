"""Keeping an installed copy's catalog current, without anyone pushing anything.

The shipped catalog is a snapshot. This module lets an installation pull a
newer one on a cadence, so a model released after install still shows up.

Four rules, because an auto-updater that misbehaves is worse than none:

1. **Fail open.** Any failure at all (offline, DNS, a 500, malformed YAML) leaves the
   bundled catalog in place and the run continues. A model picker must not stop
   working because a refresh server is down.
2. **Never block the user.** One short timeout, checked at most once per
   interval against a local stamp file. No retries, no spinner, no waiting.
3. **Validated before adoption.** A downloaded catalog is parsed and
   schema-checked in a temporary location first; it only replaces the cached
   copy if it loads cleanly. A broken update can never brick the install.
4. **Opt-out, and off by default in the paths that must be deterministic.**
   ``LLMSETUP_AUTO_UPDATE=0`` disables it, and it never runs when the user asked
   for an offline setup.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from llmsetup.util.fs import app_dir, ensure_dir, write_text
from llmsetup.util.log import Console

#: Where updated catalogs come from. Points at this project's own catalog on
#: main, which a scheduled workflow keeps current, so the repo is the
#: distribution channel and there is no extra infrastructure to run.
DEFAULT_CATALOG_URL = (
    "https://raw.githubusercontent.com/catfish-1234/llmsetup/"
    "main/src/llmsetup/catalog/data/models.yaml"
)

#: How often to check. Model releases happen weekly at most; checking more often
#: spends the user's network for nothing.
DEFAULT_INTERVAL_HOURS = 24

#: One short attempt. This runs on the path to a user's actual task.
TIMEOUT_SECONDS = 6.0

CACHE_NAME = "auto-update.yaml"
STAMP_NAME = "auto-update.json"


def catalog_url() -> str:
    return os.environ.get("LLMSETUP_CATALOG_URL", DEFAULT_CATALOG_URL)


def disabled_marker() -> Path:
    """A persistent opt-out that survives new shells, unlike an env var."""
    return app_dir() / "auto-update-disabled"


def enabled() -> bool:
    if os.environ.get("LLMSETUP_AUTO_UPDATE", "1").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return False
    return not disabled_marker().exists()


def interval_hours() -> float:
    try:
        return max(1.0, float(os.environ.get("LLMSETUP_AUTO_UPDATE_HOURS",
                                             DEFAULT_INTERVAL_HOURS)))
    except ValueError:
        return DEFAULT_INTERVAL_HOURS


def cache_dir() -> Path:
    return ensure_dir(app_dir() / "catalog.d")


def cache_path() -> Path:
    return cache_dir() / CACHE_NAME


def stamp_path() -> Path:
    return app_dir() / STAMP_NAME


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #


def _read_stamp() -> Dict[str, Any]:
    try:
        return json.loads(stamp_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_stamp(**fields: Any) -> None:
    stamp = _read_stamp()
    stamp.update(fields)
    stamp["last_checked"] = _now()
    try:
        write_text(stamp_path(), json.dumps(stamp, indent=2) + "\n")
    except OSError:
        pass          # a read-only home must not break the run


def is_due(now: Optional[datetime.datetime] = None) -> bool:
    """Whether enough time has passed since the last check."""
    stamp = _read_stamp()
    last = stamp.get("last_checked")
    if not last:
        return True
    try:
        checked = datetime.datetime.fromisoformat(last)
    except (TypeError, ValueError):
        return True
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if checked.tzinfo is None:
        checked = checked.replace(tzinfo=datetime.timezone.utc)
    return (now - checked).total_seconds() >= interval_hours() * 3600


def status() -> Dict[str, Any]:
    """What the user sees from `llmsetup catalog status`."""
    stamp = _read_stamp()
    cached = cache_path()
    return {
        "enabled": enabled(),
        "url": catalog_url(),
        "interval_hours": interval_hours(),
        "last_checked": stamp.get("last_checked"),
        "last_updated": stamp.get("last_updated"),
        "last_error": stamp.get("last_error"),
        "cached_catalog": str(cached) if cached.exists() else None,
        "cached_models": stamp.get("cached_models"),
        "due": is_due(),
    }


# --------------------------------------------------------------------------- #
# the update itself
# --------------------------------------------------------------------------- #


def fetch_catalog(url: str, timeout: float = TIMEOUT_SECONDS) -> Optional[str]:
    """Download a catalog document. None on any failure."""
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "llmsetup/0.1 auto-update"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Only HTTP responses carry a status; a file:// or other local
            # source has none, and that is a legitimate way to point an
            # air-gapped install at an internal catalog.
            status_code = getattr(response, "status", None)
            if status_code is not None and status_code != 200:
                return None
            # Cap the read: a catalog is tens of kilobytes, and an unbounded
            # read from a URL the user can point anywhere is not acceptable.
            return response.read(8 * 1024 * 1024).decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def validate_catalog_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse and sanity-check a downloaded catalog before adopting it.

    Rejects anything that is not a well-formed catalog with real entries, so a
    truncated download or an HTML error page never replaces a working catalog.
    """
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict):
        return None

    models = document.get("models")
    if not isinstance(models, list) or not models:
        return None
    for entry in models:
        if not isinstance(entry, dict) or not entry.get("id"):
            return None
    return document


def apply_update(console: Console, force: bool = False,
                 url: Optional[str] = None) -> Dict[str, Any]:
    """Fetch, validate and cache a newer catalog. Never raises."""
    target = url or catalog_url()

    if not force and not enabled():
        return {"updated": False, "reason": "auto-update is disabled"}
    if not force and not is_due():
        return {"updated": False, "reason": "checked recently"}

    text = fetch_catalog(target)
    if text is None:
        _write_stamp(last_error=f"could not fetch {target}")
        return {"updated": False, "reason": f"could not reach {target}"}

    document = validate_catalog_text(text)
    if document is None:
        _write_stamp(last_error="downloaded catalog failed validation")
        return {"updated": False, "reason": "the downloaded catalog is not valid"}

    # Prove it loads through the real loader before it can affect anything.
    if not _loads_cleanly(text):
        _write_stamp(last_error="downloaded catalog did not load")
        return {"updated": False, "reason": "the downloaded catalog did not load"}

    count = len(document.get("models") or [])
    header = (
        "# Auto-updated catalog, fetched by llmsetup. Do not edit: it is replaced\n"
        f"# on each update. Source: {target}\n"
        "# Disable with LLMSETUP_AUTO_UPDATE=0. Your own entries in other files in\n"
        "# this directory are layered on top of this one and are never touched.\n\n"
    )
    try:
        write_text(cache_path(), header + text)
    except OSError as exc:
        _write_stamp(last_error=str(exc))
        return {"updated": False, "reason": f"could not write the cache: {exc}"}

    _write_stamp(last_updated=_now(), cached_models=count, last_error=None)
    console.debug(f"catalog auto-updated from {target} ({count} models)")
    return {"updated": True, "models": count, "url": target,
            "path": str(cache_path()), "catalog_date": document.get("updated")}


def _loads_cleanly(text: str) -> bool:
    """Run a candidate catalog through the real loader in isolation."""
    from llmsetup.catalog.catalog import load_catalog

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.yaml"
        path.write_text(text, encoding="utf-8")
        try:
            catalog = load_catalog(extra_paths=[path], include_user_dir=False)
        except Exception:
            # Any failure at all is a reason to reject the download; the
            # bundled catalog stays in place.
            return False
        return len(catalog) > 0


def maybe_update(console: Console, offline: bool = False) -> Dict[str, Any]:
    """The hook called on ordinary runs. Silent, bounded, and never fatal."""
    if offline or not enabled():
        return {"updated": False, "reason": "skipped"}
    if not is_due():
        return {"updated": False, "reason": "checked recently"}
    try:
        return apply_update(console)
    except Exception as exc:
        # An updater must never break the run it is trying to improve.
        console.debug(f"catalog auto-update skipped: {exc}")
        return {"updated": False, "reason": str(exc)}


def disable_persistently() -> Path:
    """Write an opt-out the user does not have to re-export in every shell."""
    marker = disabled_marker()
    write_text(marker, "Delete this file to re-enable catalog auto-update.\n")
    return marker


def enable_persistently() -> None:
    """Remove the opt-out marker. The env var, if set, still wins."""
    try:
        disabled_marker().unlink()
    except OSError:
        pass


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
