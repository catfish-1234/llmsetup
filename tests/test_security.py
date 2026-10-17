"""Security regression tests.

Each test here corresponds to a confirmed, reproduced weakness. The catalog is
designed to be extended, shared and (via `catalog refresh`) fetched, so catalog
YAML is treated as untrusted input throughout, as is any GGUF file on disk.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from llmsetup.download import RemoteFile, validate_repo
from llmsetup.gguf import read_gguf_info
from llmsetup.interop import jan, lmstudio
from llmsetup.runtimes.gguf_store import store_path
from llmsetup.types import ModelSpec, RuntimeConfig
from llmsetup.util.errors import ProvisionError
from llmsetup.util.fs import app_dir, link_or_copy, model_store, safe_component, safe_subpath

TRAVERSALS = [
    "../../../../tmp/pwned",
    "..",
    ".",
    "",
    "....//....//etc",
    "/absolute/path",
    "a/../../b",
    "\\..\\..\\windows",
]


class TestPathSanitisation:
    @pytest.mark.parametrize("value", TRAVERSALS)
    def test_no_component_escapes_or_stays_traversal_significant(self, value):
        cleaned = safe_component(value)
        assert "/" not in cleaned and "\\" not in cleaned
        assert cleaned not in ("", ".", "..")

    def test_legitimate_names_survive_intact(self):
        for value in ("Qwen", "qwen3.8-27b", "Q4_K_M", "meta-llama", "gpt-oss-20b"):
            assert safe_component(value) == value

    def test_components_are_length_capped(self):
        assert len(safe_component("a" * 5000)) <= 128

    @pytest.mark.parametrize("value", TRAVERSALS)
    def test_safe_subpath_stays_under_its_root(self, tmp_path, value):
        result = safe_subpath(tmp_path, value, "Q4_K_M")
        assert str(result.absolute()).startswith(str(tmp_path.absolute()))


class TestCatalogControlledPaths:
    """A hostile catalog entry must not place a multi-gigabyte download anywhere
    it likes. Every path built from catalog fields is checked here."""

    @pytest.fixture
    def evil(self):
        return ModelSpec(
            id="../../../../tmp/pwned", name="Evil", publisher="../../..",
            params_b=7, gguf_repo="x/y",
        )

    def test_weights_store_contains_the_path(self, evil):
        path = store_path(evil, "Q4_K_M")
        assert str(path.absolute()).startswith(str(model_store().absolute()))

    def test_lmstudio_target_contains_the_path(self, evil):
        path = lmstudio.target_path(evil, "Q4_K_M", "../../evil.gguf")
        assert str(path.absolute()).startswith(str(lmstudio.models_dir().absolute()))
        assert path.name == "evil.gguf"

    def test_jan_folder_contains_the_path(self, evil, tmp_path, monkeypatch, gguf_file):
        data_dir = tmp_path / "jan"
        (data_dir / "models").mkdir(parents=True)
        monkeypatch.setenv("JAN_DATA_DIR", str(data_dir))

        record = jan.register_in_jan(
            evil, gguf_file,
            RuntimeConfig(runtime="llamacpp", model_id=evil.id, quant="Q4_K_M",
                          context_tokens=8192),
        )
        assert record["ok"]
        assert str(Path(record["path"]).absolute()).startswith(str(data_dir.absolute()))

    def test_client_config_dir_contains_the_path(self, evil):
        assert str(safe_subpath(app_dir() / "clients", evil.id).absolute()).startswith(
            str((app_dir() / "clients").absolute())
        )


@pytest.fixture
def gguf_file(tmp_path):
    path = tmp_path / "w.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


class TestRepoValidation:
    @pytest.mark.parametrize("repo", [
        "Qwen/Qwen3-8B-GGUF", "unsloth/gemma-3-27b-it-GGUF", "bert-base-uncased",
        "openai/gpt-oss-20b",
    ])
    def test_real_repo_ids_are_accepted(self, repo):
        assert validate_repo(repo) == repo

    @pytest.mark.parametrize("repo", [
        "../../evil", "a/../../b", "x?y=1", "http://evil.com/x", "", "..",
        "a/b/c", "a/b?c=d", "../etc/passwd", "a b", "%2e%2e/x",
    ])
    def test_hostile_repo_ids_are_rejected(self, repo):
        with pytest.raises(ProvisionError):
            validate_repo(repo)

    def test_url_construction_rejects_a_hostile_repo(self):
        with pytest.raises(ProvisionError):
            _ = RemoteFile("../../evil", "m.gguf").url

    def test_url_construction_works_for_a_real_repo(self):
        url = RemoteFile("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", revision="a" * 40).url
        assert url.endswith(f"/Qwen/Qwen3-8B-GGUF/resolve/{'a' * 40}/Qwen3-8B-Q4_K_M.gguf")
        assert ".." not in url


class TestHostileGguf:
    """Header lengths are attacker-controlled uint64s. A crafted file must
    degrade to 'use the catalog estimate', never crash the run."""

    def _write(self, path, kv_payload, tensor_count=1, kv_count=1):
        path.write_bytes(
            b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count)
            + struct.pack("<Q", kv_count) + kv_payload
        )
        return path

    def _key(self, name):
        raw = name.encode()
        return struct.pack("<Q", len(raw)) + raw

    def test_absurd_fixed_array_count_is_rejected(self, tmp_path):
        kv = self._key("a") + struct.pack("<I", 9) + struct.pack("<I", 5) \
            + struct.pack("<Q", 2 ** 61)
        assert read_gguf_info(self._write(tmp_path / "a.gguf", kv)) is None

    def test_absurd_string_array_count_is_rejected(self, tmp_path):
        kv = self._key("a") + struct.pack("<I", 9) + struct.pack("<I", 8) \
            + struct.pack("<Q", 2 ** 63)
        assert read_gguf_info(self._write(tmp_path / "b.gguf", kv)) is None

    def test_absurd_string_length_is_rejected(self, tmp_path):
        kv = self._key("a") + struct.pack("<I", 8) + struct.pack("<Q", 2 ** 60)
        assert read_gguf_info(self._write(tmp_path / "c.gguf", kv)) is None

    def test_absurd_kv_count_is_rejected(self, tmp_path):
        assert read_gguf_info(
            self._write(tmp_path / "d.gguf", b"", kv_count=10 ** 9)
        ) is None

    def test_truncated_file_is_rejected(self, tmp_path):
        path = tmp_path / "e.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1))
        assert read_gguf_info(path) is None

    def test_random_bytes_are_rejected(self, tmp_path):
        path = tmp_path / "f.gguf"
        path.write_bytes(bytes(range(256)) * 8)
        assert read_gguf_info(path) is None


class TestLinkSafety:
    def test_refuses_to_delete_a_directory_at_the_destination(self, tmp_path):
        """Removing a directory to make room for a link could destroy unrelated
        user data, so it is refused rather than forced."""
        source = tmp_path / "a.bin"
        source.write_bytes(b"x")
        occupied = tmp_path / "dst"
        occupied.mkdir()
        (occupied / "important.txt").write_text("do not delete me")

        with pytest.raises(IsADirectoryError):
            link_or_copy(source, occupied)
        assert (occupied / "important.txt").read_text() == "do not delete me"

    def test_replaces_a_stale_symlink_without_touching_its_target(self, tmp_path):
        target = tmp_path / "target.bin"
        target.write_bytes(b"original")
        link = tmp_path / "link.bin"
        link.symlink_to(target)

        source = tmp_path / "new.bin"
        source.write_bytes(b"replacement")
        link_or_copy(source, link)

        assert target.read_bytes() == b"original"
