"""Weight acquisition from Hugging Face.

Deliberately implemented over stdlib rather than `huggingface_hub`: this tool
should install and run on a fresh machine with nothing but Python. Downloads are
resumable (HTTP Range) because these files are tens of gigabytes and a dropped
connection at 90% should not start over.
"""

from __future__ import annotations

import hashlib
import http.client
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

from llmsetup.util.errors import ProvisionError
from llmsetup.util.fs import app_dir, ensure_dir, write_text
from llmsetup.util.units import human_bytes

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
USER_AGENT = "llmsetup/0.1 (+https://github.com/catfish-1234/llmsetup)"
CHUNK = 1024 * 1024


@dataclass
class RemoteFile:
    repo: str
    filename: str
    size: Optional[int] = None
    revision: Optional[str] = None

    @property
    def url(self) -> str:
        repo = urllib.parse.quote(validate_repo(self.repo), safe="/")
        revision = validate_revision(self.revision)
        return f"{HF_ENDPOINT}/{repo}/resolve/{revision}/{urllib.parse.quote(self.filename)}"

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


def validate_revision(revision: Optional[str]) -> str:
    """Only immutable Hub commit IDs are suitable for downloads."""
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ProvisionError(f"expected a full model commit SHA, got {revision!r}")
    return revision.lower()


def _revision_path(repo: str) -> Path:
    key = hashlib.sha256(f"{HF_ENDPOINT}/{validate_repo(repo)}".encode()).hexdigest()
    return app_dir() / "revisions" / f"{key}.txt"


def saved_revision(repo: str) -> Optional[str]:
    path = _revision_path(repo)
    return validate_revision(path.read_text().strip()) if path.exists() else None


def pin_revision(repo: str, revision: Optional[str] = None, timeout: float = 30.0) -> str:
    """Resolve once and persist the commit; explicit catalog pins take precedence."""
    if revision is not None:
        return validate_revision(revision)
    saved = saved_revision(repo)
    if saved:
        return saved
    sha = validate_revision(_repo_info(repo, timeout=timeout).get("sha"))
    write_text(_revision_path(repo), sha + "\n")
    return sha


def _repo_info(repo: str, revision: Optional[str] = None, timeout: float = 30.0) -> dict:
    # Validated then quoted: the value comes from catalog YAML, and an unchecked
    # one with ".." or a query character can redirect the request off /api/models/.
    url = f"{HF_ENDPOINT}/api/models/{urllib.parse.quote(validate_repo(repo), safe='/')}"
    if revision:
        url += f"/revision/{validate_revision(revision)}"
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
                hint="the catalog entry may be stale; run `llmsetup catalog refresh`",
            ) from None
        raise ProvisionError(f"listing {repo} failed: HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        raise ProvisionError(
            f"could not reach {HF_ENDPOINT}: {exc}",
            hint="check connectivity, or point HF_ENDPOINT at a mirror",
        ) from None

    if not isinstance(data, dict):
        raise ProvisionError(f"invalid model metadata for {repo}")
    return data


def list_repo_files(repo: str, timeout: float = 30.0,
                    revision: Optional[str] = None) -> List[RemoteFile]:
    """List and download every shard/projector from the same pinned commit."""
    sha = pin_revision(repo, revision, timeout)
    data = _repo_info(repo, sha, timeout)
    files = []
    for sibling in data.get("siblings", []):
        name = sibling.get("rfilename")
        if name:
            files.append(RemoteFile(repo, name, sibling.get("size"), sha))
    return files


#: Prefixes marking a per-layer optimised ("dynamic") quantization. These pick
#: a different precision for each tensor rather than one setting for the whole
#: model, and measure lower KL divergence than a plain K-quant of the same size.
#: Preferring them is free quality: same bytes, closer to the original weights.
_DYNAMIC_PREFIXES = ("ud-", "iq")

#: Suffixes a dynamic build uses in place of the plain K-quant name.
_DYNAMIC_SUFFIXES = ("_xl", "_l", "_m", "_s")


def dynamic_variants(quant: str) -> List[str]:
    """Dynamic-quant names to try before falling back to ``quant`` itself.

    ``Q4_K_M`` becomes ``UD-Q4_K_XL`` then ``UD-Q4_K_M``: the XL build spends its
    extra bits on the tensors that matter most, so it is the better choice at
    effectively the same size.
    """
    upper = quant.upper()
    if upper.startswith(("UD-", "IQ")):
        return [quant]
    base = upper.rsplit("_", 1)[0] if "_" in upper else upper
    candidates = [f"UD-{base}_XL", f"UD-{upper}"]
    # Never return a name identical to the input; the caller adds that itself.
    return [c for c in candidates if c.upper() != upper]


#: Vision models ship two files: the language model and a separate multimodal
#: projector (the CLIP/SigLIP tower plus the layer that maps image embeddings
#: into the model's token space). Repos name it inconsistently -- ``mmproj``,
#: ``mm-proj``, ``mmproj-model-f16``, occasionally ``projector`` -- but the
#: prefix is always there.
_PROJECTOR_RE = re.compile(r"(?:^|[-_])(?:mmproj|mm-proj|projector)", re.I)

#: Projector precision, best first. Vision towers are markedly more sensitive to
#: quantization than the language model, and the file is small enough that the
#: precision costs little, so a heavily quantized language model still deserves
#: a full-precision projector. Q4 projectors exist and are visibly worse.
_PROJECTOR_PRECISION = ("bf16", "f32", "f16", "q8_0")


def is_projector(filename: str) -> bool:
    """Whether a repo file is a multimodal projector rather than a model."""
    stem = filename.rsplit("/", 1)[-1]
    return bool(_PROJECTOR_RE.search(stem))


def pick_mmproj(files: List[RemoteFile]) -> Optional[RemoteFile]:
    """Choose the multimodal projector to pair with the language model.

    Returns ``None`` when the repo publishes none, which is the normal case for
    a text-only model and a reportable problem for a vision one.
    """
    projectors = [f for f in files
                  if f.filename.lower().endswith(".gguf") and is_projector(f.filename)]
    if not projectors:
        return None
    def rank(f: RemoteFile) -> tuple:
        name = f.filename.lower()
        for i, precision in enumerate(_PROJECTOR_PRECISION):
            if re.search(rf"(^|[-_.]){re.escape(precision)}([-_.]|$)", name):
                return (i, len(name))
        # Unlabelled precision: no reason to prefer it over a known-good one,
        # but better than nothing.
        return (len(_PROJECTOR_PRECISION), len(name))
    return sorted(projectors, key=rank)[0]


def pick_gguf(files: List[RemoteFile], quant: str,
              prefer_dynamic: bool = True) -> Optional[RemoteFile]:
    """Choose the GGUF matching ``quant``.

    Repos are inconsistent: ``Q4_K_M``, ``q4_k_m``, and ``Q4_K_M-00001-of-00002``
    all appear. Exact-token matching avoids the classic bug where a search for
    ``Q4_0`` matches ``Q4_0_4_8`` (an ARM-specific repack that will not run on
    x86).
    """
    ggufs = [f for f in files
             if f.filename.lower().endswith(".gguf") and not is_projector(f.filename)]
    if not ggufs:
        return None

    # A per-layer optimised build of the same nominal size is strictly better,
    # so try those names first and fall back to the plain quant.
    if prefer_dynamic:
        for variant in dynamic_variants(quant):
            match = pick_gguf(files, variant, prefer_dynamic=False)
            if match is not None:
                return match

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
    total_timeout: float = 7200.0,
) -> Path:
    """Download with resume. Returns the final path; skips work if already complete."""
    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and (remote.size is None or dest.stat().st_size == remote.size):
        if progress:
            progress(dest.stat().st_size, remote.size)
        return dest

    deadline = time.monotonic() + total_timeout
    attempt = 0
    while True:
        existing = part.stat().st_size if part.exists() else 0
        if remote.size is not None and existing > remote.size:
            part.unlink()
            existing = 0
        headers = _auth_headers()
        if existing:
            headers["Range"] = f"bytes={existing}-"

        try:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"download timed out after {total_timeout:g}s")
            req = urllib.request.Request(remote.url, headers=headers)
            with urllib.request.urlopen(req, timeout=min(timeout, deadline - time.monotonic())) as resp:
                # A server that ignores Range restarts the file; don't append to it.
                mode = "ab" if (existing and resp.status == 206) else "wb"
                if mode == "wb":
                    existing = 0

                total = remote.size
                length = resp.headers.get("Content-Length")
                response_size = int(length) if length is not None else None
                if resp.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                         resp.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != existing or int(match[2]) < existing:
                        raise ProvisionError("server returned an invalid download range",
                                             hint=f"partial file kept at {part}; retry later")
                    range_size = int(match[2]) - existing + 1
                    range_total = None if match[3] == "*" else int(match[3])
                    if (response_size is not None and response_size != range_size) or \
                            (range_total is not None and int(match[2]) >= range_total) or \
                            (total is not None and range_total is not None and
                             total != range_total):
                        raise ProvisionError("server returned inconsistent download sizes")
                    total = total if total is not None else range_total
                    response_size = range_size
                if total is None and response_size is not None:
                    total = response_size + existing

                downloaded = existing
                with open(part, mode) as handle:
                    # HTTPResponse.read(size) can wait for the entire chunk
                    # while a slow peer keeps resetting the socket timeout.
                    read_chunk = getattr(resp, "read1", resp.read)
                    while True:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"download timed out after {total_timeout:g}s")
                        chunk = read_chunk(CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress:
                            progress(downloaded, total)
                if (total is not None and downloaded != total) or \
                        (response_size is not None and downloaded - existing != response_size):
                    raise OSError(f"incomplete download: received {downloaded} bytes; "
                                  f"expected {total}")
            os.replace(part, dest)
            return dest

        except (urllib.error.URLError, OSError, http.client.HTTPException, ProvisionError, ValueError) as exc:
            # The final byte may have arrived before a previous process stopped.
            # A 416 confirms the remote length; only promote an exact match.
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 416:
                match = re.fullmatch(r"bytes \*/(\d+)",
                                     exc.headers.get("Content-Range", ""))
                if match and existing > 0 and existing == int(match[1]) and \
                        (remote.size is None or existing == remote.size):
                    os.replace(part, dest)
                    if progress:
                        progress(existing, existing)
                    return dest
            attempt += 1
            if attempt > retries or time.monotonic() >= deadline:
                raise ProvisionError(
                    f"download of {remote.filename} failed after {retries} retries: {exc}",
                    hint=(f"the partial file is kept at {part}; "
                          "rerun `llmsetup setup` with the same model and options to resume"),
                ) from None
            invalid_partial = (isinstance(exc, ProvisionError) or
                               (isinstance(exc, urllib.error.HTTPError) and exc.code == 416))
            if invalid_partial and part.exists():
                part.unlink()
            time.sleep(max(0, min(30, 2 ** attempt, deadline - time.monotonic())))


def make_progress_printer(console, label: str) -> Callable[[int, Optional[int]], None]:
    """Throttled progress callback, one line, rewritten, at most 5x/second."""
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
