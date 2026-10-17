"""Downloads are completed only after their advertised byte counts agree."""

import io
import urllib.error

import pytest

from llmsetup.download import RemoteFile, download_file
from llmsetup.util.errors import ProvisionError


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


def serve(monkeypatch, *responses):
    queued = iter(responses)
    requests = []

    def urlopen(request, **kwargs):
        requests.append(request)
        response = next(queued)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("llmsetup.download.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("llmsetup.download.time.sleep", lambda seconds: None)
    return requests


def test_short_response_stays_partial(tmp_path, monkeypatch):
    serve(monkeypatch, Response(b"abc", headers={"Content-Length": "6"}))
    dest = tmp_path / "model.gguf"
    with pytest.raises(ProvisionError, match="incomplete download"):
        download_file(RemoteFile("a/b", "model.gguf", revision="a" * 40), dest, retries=0)
    assert not dest.exists()
    assert dest.with_suffix(".gguf.part").read_bytes() == b"abc"


def test_catalog_size_detects_short_response_without_length(tmp_path, monkeypatch):
    serve(monkeypatch, Response(b"abc"))
    dest = tmp_path / "model.gguf"
    with pytest.raises(ProvisionError):
        download_file(RemoteFile("a/b", "model.gguf", 6, revision="a" * 40), dest, retries=0)
    assert not dest.exists()


def test_short_response_is_resumed_on_retry(tmp_path, monkeypatch):
    requests = serve(monkeypatch,
                     Response(b"abc", headers={"Content-Length": "6"}),
                     Response(b"def", 206, {"Content-Length": "3",
                                            "Content-Range": "bytes 3-5/6"}))
    dest = tmp_path / "model.gguf"
    assert download_file(RemoteFile("a/b", "model.gguf", revision="a" * 40), dest) == dest
    assert dest.read_bytes() == b"abcdef"
    assert requests[1].get_header("Range") == "bytes=3-"
    assert not dest.with_suffix(".gguf.part").exists()


def test_ignored_range_restarts_with_correct_progress_total(tmp_path, monkeypatch):
    serve(monkeypatch, Response(b"abcdef", headers={"Content-Length": "6"}))
    dest = tmp_path / "model.gguf"
    dest.with_suffix(".gguf.part").write_bytes(b"abc")
    progress = []
    download_file(RemoteFile("a/b", "model.gguf", revision="a" * 40), dest,
                  progress=lambda done, total: progress.append((done, total)))
    assert dest.read_bytes() == b"abcdef"
    assert progress[-1] == (6, 6)


@pytest.mark.parametrize("content_range", ["", "bytes 0-2/6", "bytes 3-5/7",
                                           "bytes 3-8/6", "bytes 3-4/6"])
def test_invalid_range_never_appends(tmp_path, monkeypatch, content_range):
    serve(monkeypatch, Response(b"def", 206, {"Content-Length": "3",
                                             "Content-Range": content_range}))
    dest = tmp_path / "model.gguf"
    part = dest.with_suffix(".gguf.part")
    part.write_bytes(b"abc")
    with pytest.raises(ProvisionError):
        download_file(RemoteFile("a/b", "model.gguf", 6, revision="a" * 40), dest, retries=0)
    assert part.read_bytes() == b"abc"
    assert not dest.exists()


def test_completed_partial_is_confirmed_by_range_response(tmp_path, monkeypatch):
    serve(monkeypatch, urllib.error.HTTPError(
        "https://example.invalid", 416, "range", {"Content-Range": "bytes */6"}, None))
    dest = tmp_path / "model.gguf"
    dest.with_suffix(".gguf.part").write_bytes(b"abcdef")
    download_file(RemoteFile("a/b", "model.gguf", revision="a" * 40), dest, retries=0)
    assert dest.read_bytes() == b"abcdef"


def test_existing_complete_file_needs_no_request(tmp_path, monkeypatch):
    requests = serve(monkeypatch)
    dest = tmp_path / "model.gguf"
    dest.write_bytes(b"abcdef")
    download_file(RemoteFile("a/b", "model.gguf", 6, revision="a" * 40), dest)
    assert not requests


def test_corrupt_oversized_partial_restarts(tmp_path, monkeypatch):
    requests = serve(monkeypatch, Response(b"abc", headers={"Content-Length": "3"}))
    dest = tmp_path / "model.gguf"
    dest.with_suffix(".gguf.part").write_bytes(b"too many bytes")
    download_file(RemoteFile("a/b", "model.gguf", 3, revision="a" * 40), dest)
    assert dest.read_bytes() == b"abc"
    assert requests[0].get_header("Range") is None


def test_invalid_resume_range_discards_partial_and_retries_cleanly(tmp_path, monkeypatch):
    requests = serve(monkeypatch,
                     Response(b"def", 206, {"Content-Range": "bytes 0-2/6"}),
                     Response(b"abcdef", headers={"Content-Length": "6"}))
    dest = tmp_path / "model.gguf"
    dest.with_suffix(".gguf.part").write_bytes(b"abc")
    download_file(RemoteFile("a/b", "model.gguf", 6, revision="a" * 40), dest, retries=1)
    assert dest.read_bytes() == b"abcdef"
    assert requests[0].get_header("Range") == "bytes=3-"
    assert requests[1].get_header("Range") is None


def test_cached_first_shard_fetches_missing_sibling(catalog, monkeypatch):
    from llmsetup.runtimes.gguf_store import fetch_gguf, find_local, store_path
    from llmsetup.util.log import Console

    spec = catalog.require("qwen3-8b")
    spec.gguf_revision = "a" * 40
    directory = store_path(spec, "Q4_K_M")
    directory.mkdir(parents=True)
    first = directory / "model-Q4_K_M-00001-of-00002.gguf"
    first.write_bytes(b"abc")
    assert find_local(spec, "Q4_K_M") is None
    files = [RemoteFile(spec.gguf_repo, name, 3, revision=spec.gguf_revision) for name in
             [first.name, "model-Q4_K_M-00002-of-00002.gguf"]]
    monkeypatch.setattr("llmsetup.runtimes.gguf_store.list_repo_files", lambda *a, **kw: files)
    requests = serve(monkeypatch, Response(b"def", headers={"Content-Length": "3"}))
    assert fetch_gguf(spec, "Q4_K_M", Console(quiet=True)) == first
    assert len(requests) == 1
    assert first.with_name(files[1].filename).read_bytes() == b"def"
    assert find_local(spec, "Q4_K_M") == first


def test_rejected_stale_partial_is_discarded_before_retry(tmp_path, monkeypatch):
    requests = serve(monkeypatch, urllib.error.HTTPError(
        "https://example.invalid", 416, "range", {"Content-Range": "bytes */6"}, None),
        Response(b"abcdef", headers={"Content-Length": "6"}))
    dest = tmp_path / "model.gguf"
    dest.with_suffix(".gguf.part").write_bytes(b"stale-too-long-partial")
    download_file(RemoteFile("a/b", "model.gguf", revision="a" * 40), dest, retries=1)
    assert dest.read_bytes() == b"abcdef"
    assert requests[1].get_header("Range") is None
