"""Weight acquisition from Hugging Face.

Deliberately implemented over stdlib rather than `huggingface_hub`: this tool
should install and run on a fresh machine with nothing but Python. Downloads are
resumable (HTTP Range) because these files are tens of gigabytes and a dropped
connection at 90% should not start over.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from llmfit.util.errors import ProvisionError
from llmfit.util.fs import ensure_dir
from llmfit.util.units import human_bytes

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = "llmfit/0.1 (+https://github.com/catfish-1234/open-source-llm)"
CHUNK = 1024 * 1024


@dataclass
class RemoteFile:
    repo: str
    filename: str
    size: Optional[int] = None

    @property
    def url(self) -> str:
        repo = urllib.parse.quote(validate_repo(self.repo), safe="/")
        return f"{HF_ENDPOINT}/{repo}/resolve/main/{urllib.parse.quote(self.filename)}"

    @property
    def is_split(self) -> bool:
        """Files like `model-00001-of-00003.gguf` must all be fetched together."""
        return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", self.filename))

    @property
    def split_group(self) -> str:
        return re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", self.filename)


#: A Hugging Face repo id: `owner/name`, or a bare `name` for a canonical repo.
#: Anchored and dot-restricted so a catalog value cannot smuggle in a traversal.
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$")


def validate_repo(repo: str) -> str:
    """Reject anything that is not a plausible Hugging Face repo id.

    Catalog YAML is user-extensible and, in the case of a refreshed catalog,
    may be fetched. Validating the shape here means a hostile value fails loudly
    at the boundary instead of being quoted into a URL and hoping for the best.
    """
    text = str(repo or "").strip()
    if not _REPO_RE.match(text) or ".." in text:
        raise ProvisionError(
            f"invalid model repository id: {repo!r}",
            hint="expected the form `owner/name`; check the catalog entry",
        )
    return text


def _auth_headers() -> dict:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        # Needed for gated repos (Llama, Gemma) the user has already accepted.
        headers["Authorization"] = f"Bearer {token}"
    return headers


def list_repo_files(repo: str, timeout: float = 30.0) -> List[RemoteFile]:
    """List files in a HF repo. Raises ProvisionError with an actionable hint."""
    # Validated then quoted: the value comes from catalog YAML, and an unchecked
    # one with ".." or a query character can redirect the request off /api/models/.
    url = f"{HF_ENDPOINT}/api/models/{urllib.parse.quote(validate_repo(repo), safe='/')}"
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ProvisionError(
                f"access to {repo} is gated",
                hint=("accept the licence on the model page, then set HF_TOKEN "
                      "to a token with read access"),
            ) from None
        if exc.code == 404:
            raise ProvisionError(
                f"model repo {repo} not found",
                hint="the catalog entry may be stale; run `llmfit catalog refresh`",
            ) from None
        raise ProvisionError(f"listing {repo} failed: HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise ProvisionError(
            f"could not reach {HF_ENDPOINT}: {exc}",
            hint="check connectivity, or point HF_ENDPOINT at a mirror",
        ) from None

    files = []
    for sibling in data.get("siblings", []):
        name = sibling.get("rfilename")
        if name:
            files.append(RemoteFile(repo, name, sibling.get("size")))
    return files


def pick_gguf(files: List[RemoteFile], quant: str) -> Optional[RemoteFile]:
    """Choose the GGUF matching ``quant``.

    Repos are inconsistent: ``Q4_K_M``, ``q4_k_m``, and ``Q4_K_M-00001-of-00002``
    all appear. Exact-token matching avoids the classic bug where a search for
    ``Q4_0`` matches ``Q4_0_4_8`` (an ARM-specific repack that will not run on
    x86).
    """
    ggufs = [f for f in files if f.filename.lower().endswith(".gguf")]
    if not ggufs:
        return None

    wanted = quant.lower()
    exact = []
    for f in ggufs:
        stem = f.filename.lower().rsplit("/", 1)[-1]
        stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", stem)
        stem = stem[:-5] if stem.endswith(".gguf") else stem
        tokens = re.split(r"[-._]", stem)
        # Rejoin so multi-token quant names (q4_k_m) can be found as a substring
        # bounded by separators.
        joined = "-".join(tokens)
        if re.search(rf"(^|-){re.escape(wanted.replace('_', '-'))}(-|$)", joined):
            exact.append(f)

    pool = exact or [f for f in ggufs if wanted in f.filename.lower()]
    if not pool:
        return None

    # Prefer the first shard of a split set; the rest are fetched alongside it.
    splits = [f for f in pool if f.is_split]
    if splits:
        return sorted(splits, key=lambda f: f.filename)[0]
    return sorted(pool, key=lambda f: len(f.filename))[0]


def sibling_shards(files: List[RemoteFile], first: RemoteFile) -> List[RemoteFile]:
    """All shards belonging to the same split GGUF as ``first``."""
    if not first.is_split:
        return [first]
    group = first.split_group
    return sorted(
        (f for f in files if f.is_split and f.split_group == group),
        key=lambda f: f.filename,
    )


def download_file(
    remote: RemoteFile,
    dest: Path,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    timeout: float = 60.0,
    retries: int = 4,
) -> Path:
    """Download with resume. Returns the final path; skips work if already complete."""
    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and (remote.size is None or dest.stat().st_size == remote.size):
        if progress:
            progress(dest.stat().st_size, remote.size)
        return dest

    attempt = 0
    while True:
        existing = part.stat().st_size if part.exists() else 0
        headers = _auth_headers()
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            req = urllib.request.Request(remote.url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = remote.size
                length = resp.headers.get("Content-Length")
                if total is None and length:
                    total = int(length) + existing
                # A server that ignores Range restarts the file; don't append to it.
                mode = "ab" if (existing and resp.status == 206) else "wb"
                if mode == "wb":
                    existing = 0

                downloaded = existing
                with open(part, mode) as handle:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress(downloaded, total)
            os.replace(part, dest)
            return dest

        except (urllib.error.URLError, OSError, urllib.error.HTTPError) as exc:
            attempt += 1
            if attempt > retries:
                raise ProvisionError(
                    f"download of {remote.filename} failed after {retries} retries: {exc}",
                    hint=(f"the partial file is kept at {part}; rerun to resume, "
                          "or download it manually and pass --model-path"),
                ) from None
            time.sleep(min(30, 2 ** attempt))


def make_progress_printer(console, label: str) -> Callable[[int, Optional[int]], None]:
    """Throttled progress callback — one line, rewritten, at most 5x/second."""
    state = {"last": 0.0, "started": time.monotonic()}

    def report(done: int, total: Optional[int]) -> None:
        now = time.monotonic()
        finished = total is not None and done >= total
        if now - state["last"] < 0.2 and not finished:
            return
        state["last"] = now
        elapsed = max(0.001, now - state["started"])
        rate = done / elapsed
        if total:
            pct = 100.0 * done / total
            eta = (total - done) / rate if rate > 0 else 0
            console.info(
                f"{label}: {pct:5.1f}%  {human_bytes(done)}/{human_bytes(total)}  "
                f"{human_bytes(rate)}/s  ETA {eta / 60:.0f}m"
            )
        else:
            console.info(f"{label}: {human_bytes(done)}  {human_bytes(rate)}/s")

    return report
