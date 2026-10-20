"""Context follows usage: coding and agent work aim high and keep a floor."""
from llmsetup.interview.questions import infer_flags
from llmsetup.interview.wizard import profile_from_answers
from llmsetup.planner.memory import (
    KV_ASYMMETRIC, kv_element_bytes, kv_options, kv_split,
)
from llmsetup.runtimes.llamacpp import LlamaCppRuntime


def test_coding_defaults_to_32k_with_a_16k_floor():
    p = profile_from_answers(infer_flags({"tasks": ["coding"]}))
    assert p.context_tokens == 32768
    assert p.context_floor == 16384


def test_agent_gets_the_same_floor():
    p = profile_from_answers(infer_flags({"tasks": ["agent"]}))
    assert p.context_tokens == 32768 and p.context_floor == 16384


def test_chat_keeps_a_small_default_and_no_floor():
    p = profile_from_answers(infer_flags({"tasks": ["chat"]}))
    assert p.context_tokens == 8192 and p.context_floor == 0


def test_an_explicit_context_is_respected():
    p = profile_from_answers(infer_flags({"tasks": ["coding"], "context_tokens": 65536}))
    assert p.context_tokens == 65536


def test_asymmetric_cache_only_on_llamacpp_and_never_for_sensitive_symmetric_q4():
    assert KV_ASYMMETRIC in kv_options("llamacpp")
    assert KV_ASYMMETRIC not in kv_options("ollama")
    sensitive = kv_options("llamacpp", precision_sensitive=True)
    assert "q4_0" not in sensitive and "q5_1" not in sensitive
    assert "q4_0" in kv_options("ollama")


def test_asymmetric_cache_is_smaller_than_q8_and_larger_than_q4():
    assert kv_element_bytes("q4_0") < kv_element_bytes(KV_ASYMMETRIC) < kv_element_bytes("q8_0")
    assert kv_split(KV_ASYMMETRIC) == ("q8_0", "q4_0")
    assert kv_split("q8_0") == ("q8_0", "q8_0")


def test_llamacpp_gets_separate_key_and_value_flags():
    from test_reproducibility import config, spec

    cmd = LlamaCppRuntime().launch_command(spec(), config(kv_cache_type=KV_ASYMMETRIC))
    joined = " ".join(str(c) for c in cmd)
    assert "--cache-type-k q8_0" in joined and "--cache-type-v q4_0" in joined
