"""GGUF header parsing. The mechanism that replaces catalog guesses with truth."""

from __future__ import annotations

import struct

import pytest

from llmfit.gguf import apply_to_spec, read_gguf_info
from llmfit.types import ModelSpec


def _string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def build_gguf(path, pairs, tensor_count=291):
    """Write a synthetic GGUF header. `pairs` is [(key, type, payload)]."""
    body = b"".join(_string(k) + struct.pack("<I", t) + p for k, t, p in pairs)
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count)
        + struct.pack("<Q", len(pairs)) + body
    )
    return path


def qwen_header(tmp_path, layers=36, kv_heads=8, heads=32, hidden=4096,
                context=131072, with_vocab=True):
    pairs = [
        ("general.architecture", 8, _string("qwen3")),
        ("qwen3.block_count", 4, struct.pack("<I", layers)),
        ("qwen3.attention.head_count_kv", 4, struct.pack("<I", kv_heads)),
        ("qwen3.attention.head_count", 4, struct.pack("<I", heads)),
        ("qwen3.embedding_length", 4, struct.pack("<I", hidden)),
        ("qwen3.context_length", 4, struct.pack("<I", context)),
        ("tokenizer.chat_template", 8, _string("{% if tools %}x{% endif %}")),
    ]
    if with_vocab:
        vocab = b"".join(_string(f"tok{i}") for i in range(50_000))
        pairs.append(("tokenizer.ggml.tokens", 9,
                      struct.pack("<I", 8) + struct.pack("<Q", 50_000) + vocab))
    return build_gguf(tmp_path / "model.gguf", pairs)


class TestParsing:
    def test_reads_the_architecture_fields(self, tmp_path):
        info = read_gguf_info(qwen_header(tmp_path))
        assert info.architecture == "qwen3"
        assert info.layers == 36
        assert info.kv_heads == 8
        assert info.embedding_length == 4096
        assert info.context_length == 131072

    def test_derives_head_dim_when_not_stated(self, tmp_path):
        info = read_gguf_info(qwen_header(tmp_path, hidden=4096, heads=32))
        assert info.head_dim == 128

    def test_skips_a_large_vocabulary_without_materialising_it(self, tmp_path):
        """The token list is ~150k strings; decoding it would be slow and
        pointless, but the parser must still walk past it correctly."""
        info = read_gguf_info(qwen_header(tmp_path, with_vocab=True))
        assert info.layers == 36           # parsed correctly *after* the array
        assert info.chat_template is not None

    def test_infers_tool_support_from_the_chat_template(self, tmp_path):
        info = read_gguf_info(qwen_header(tmp_path))
        assert info.supports_tools is True

    def test_returns_none_for_a_non_gguf_file(self, tmp_path):
        path = tmp_path / "not.gguf"
        path.write_bytes(b"this is not a gguf file at all")
        assert read_gguf_info(path) is None

    def test_returns_none_for_a_truncated_file(self, tmp_path):
        """A half-finished download must degrade to the catalog estimate, not
        crash the setup."""
        full = qwen_header(tmp_path).read_bytes()
        truncated = tmp_path / "cut.gguf"
        truncated.write_bytes(full[:40])
        assert read_gguf_info(truncated) is None

    def test_returns_none_for_a_missing_file(self, tmp_path):
        assert read_gguf_info(tmp_path / "absent.gguf") is None

    def test_rejects_an_implausible_kv_count(self, tmp_path):
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1)
                         + struct.pack("<Q", 10 ** 9))
        assert read_gguf_info(path) is None


class TestCorrection:
    def base_spec(self, **kwargs):
        fields = {"id": "x", "name": "X", "publisher": "p", "params_b": 8.0,
                  "layers": 32, "hidden_size": 4096, "kv_heads": 8,
                  "head_dim": 128, "max_context": 131072}
        fields.update(kwargs)
        return ModelSpec(**fields)

    def test_corrects_a_wrong_layer_count(self, tmp_path):
        spec = self.base_spec(layers=32)
        info = read_gguf_info(qwen_header(tmp_path, layers=36))
        updated, corrections = apply_to_spec(spec, info)
        assert updated.layers == 36
        assert any("layer count" in c for c in corrections)

    def test_corrects_a_wrong_kv_head_count(self, tmp_path):
        """The field that most affects the KV cache estimate."""
        spec = self.base_spec(kv_heads=8)
        info = read_gguf_info(qwen_header(tmp_path, kv_heads=2))
        updated, corrections = apply_to_spec(spec, info)
        assert updated.kv_heads == 2
        assert any("KV head" in c for c in corrections)

    def test_reports_nothing_when_the_catalog_was_right(self, tmp_path):
        spec = self.base_spec(layers=36, kv_heads=8, hidden_size=4096, head_dim=128)
        info = read_gguf_info(qwen_header(tmp_path))
        _, corrections = apply_to_spec(spec, info)
        assert corrections == []

    def test_lowers_but_never_raises_the_context_ceiling(self, tmp_path):
        """Repos often omit RoPE extension, so a file claiming *more* context
        than the catalog is not evidence; one claiming less is."""
        spec = self.base_spec(max_context=131072)
        lower = read_gguf_info(qwen_header(tmp_path, context=32768))
        updated, _ = apply_to_spec(spec, lower)
        assert updated.max_context == 32768

        spec2 = self.base_spec(max_context=32768)
        higher = read_gguf_info(qwen_header(tmp_path, context=262144))
        updated2, _ = apply_to_spec(spec2, higher)
        assert updated2.max_context == 32768

    def test_a_missing_header_leaves_the_spec_untouched(self):
        spec = self.base_spec()
        updated, corrections = apply_to_spec(spec, None)
        assert updated is spec and corrections == []

    def test_correction_changes_the_memory_estimate(self, tmp_path):
        """The point of the whole mechanism: a wrong KV head count is a wrong
        memory plan, and this is what catches it before an OOM does."""
        from llmfit.planner.memory import estimate_memory

        # Match every other field so only the KV head count differs.
        spec = self.base_spec(layers=36, kv_heads=8, hidden_size=4096, head_dim=128)
        before = estimate_memory(spec, 4.85, 65536).kv_cache
        info = read_gguf_info(qwen_header(tmp_path, layers=36, kv_heads=2))
        corrected, _ = apply_to_spec(spec, info)
        after = estimate_memory(corrected, 4.85, 65536).kv_cache
        assert corrected.layers == 36 and corrected.kv_heads == 2
        assert after == pytest.approx(before / 4)
