"""Interview: does it collect and validate answers correctly?"""

from __future__ import annotations

import io

import pytest

from llmsetup.interview.questions import QUESTIONS, infer_flags
from llmsetup.interview.wizard import (
    load_answers,
    profile_from_answers,
    run_interview,
    save_profile,
)
from llmsetup.types import Client, Priority, Task, UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.log import Console


def task_numbers(*names):
    """Positions of task options, looked up rather than hardcoded.

    The wizard numbers its options, and a test that spells the numbers out
    breaks the moment an option is inserted, which says nothing about the
    behaviour it was meant to pin.
    """
    from llmsetup.interview.questions import TASK_OPTIONS

    order = [option.value for option in TASK_OPTIONS]
    return ",".join(str(order.index(name) + 1) for name in names)


def drive(script):
    """Run the wizard against a scripted set of answers; unanswered = Enter."""
    answers = iter(script)
    stream = io.StringIO()
    profile = run_interview(
        Console(color=False, stream=stream),
        input_fn=lambda _: next(answers, ""),
        stream=stream,
    )
    return profile, stream.getvalue()


class TestQuestionBank:
    def test_every_question_explains_why_it_is_asked(self):
        for question in QUESTIONS:
            assert question.why, question.key

    def test_choice_questions_offer_options(self):
        for question in QUESTIONS:
            if question.kind in ("choice", "multi"):
                assert len(question.options) >= 2, question.key

    def test_question_keys_are_unique(self):
        keys = [q.key for q in QUESTIONS]
        assert len(keys) == len(set(keys))

    def test_every_key_maps_onto_the_profile(self):
        fields = set(UseCaseProfile.__dataclass_fields__)
        for question in QUESTIONS:
            assert question.key in fields, question.key


class TestInference:
    def test_agents_imply_tool_calling(self):
        """Someone who says 'agents' has said they need tools, whether or not
        they know the term."""
        assert infer_flags({"tasks": ["agent"]})["needs_tools"] is True

    def test_extraction_implies_json(self):
        assert infer_flags({"tasks": ["extract"]})["needs_json"] is True

    def test_rag_implies_embeddings_and_a_real_context(self):
        derived = infer_flags({"tasks": ["rag"], "context_tokens": 4096})
        assert derived["needs_embeddings"] is True
        assert derived["context_tokens"] >= 16384

    def test_an_explicit_answer_is_never_overridden(self):
        derived = infer_flags({"tasks": ["agent"], "needs_tools": False})
        assert derived["needs_tools"] is False

    def test_a_generous_context_is_not_reduced(self):
        derived = infer_flags({"tasks": ["rag"], "context_tokens": 131072})
        assert derived["context_tokens"] == 131072


class TestWizard:
    def test_collects_answers_by_number(self):
        profile, _ = drive([task_numbers("coding", "agent"),
                            "1", "3", "3", "1", "", "", "", "", "en", "n", ""])
        assert Task.CODING in profile.tasks and Task.AGENT in profile.tasks
        assert Client.LM_STUDIO in profile.clients
        assert profile.context_tokens == 32768
        assert profile.priority is Priority.QUALITY

    def test_accepts_names_as_well_as_numbers(self):
        profile, _ = drive(["coding", "ollama", "8192", "balanced"])
        assert profile.tasks == [Task.CODING]
        assert profile.clients == [Client.OLLAMA]

    def test_pressing_enter_takes_every_default(self):
        profile, _ = drive([])
        assert profile.tasks == [Task.CHAT]
        assert profile.context_tokens == 8192
        assert profile.priority is Priority.BALANCED

    def test_shows_the_reason_for_each_question(self):
        _, transcript = drive([])
        assert "why:" in transcript

    def test_preserves_the_order_tasks_were_given(self):
        """The first task named is treated as primary and drives sampling."""
        profile, _ = drive([task_numbers("agent", "coding")])
        assert profile.primary_task is Task.AGENT

    def test_reprompts_on_an_unrecognised_answer(self):
        profile, transcript = drive(["nonsense", "coding"])
        assert "not recognised" in transcript
        assert profile.tasks == [Task.CODING]

    def test_defaults_are_taken_for_questions_answered_by_flags(self):
        stream = io.StringIO()
        profile = run_interview(Console(color=False, stream=stream),
                                defaults={"tasks": ["vision"]},
                                input_fn=lambda _: "", stream=stream)
        assert profile.tasks == [Task.VISION]
        assert "What will you use the model for?" not in stream.getvalue()


class TestValidation:
    def test_rejects_an_unknown_task_with_the_valid_list(self):
        with pytest.raises(LlmsetupError) as excinfo:
            profile_from_answers({"tasks": ["nonsense"]})
        assert "chat" in excinfo.value.hint

    def test_rejects_an_unknown_priority(self):
        with pytest.raises(LlmsetupError):
            profile_from_answers({"priority": "fastest"})

    def test_rejects_an_unusably_small_context(self):
        with pytest.raises(LlmsetupError):
            profile_from_answers({"context_tokens": 10})

    def test_clamps_concurrent_users_to_at_least_one(self):
        assert profile_from_answers({"concurrent_users": 0}).concurrent_users == 1

    def test_accepts_languages_as_a_string_or_a_list(self):
        assert profile_from_answers({"languages": "en, zh"}).languages == ["en", "zh"]
        assert profile_from_answers({"languages": ["en"]}).languages == ["en"]

    def test_ignores_unknown_keys_rather_than_crashing(self):
        profile = profile_from_answers({"tasks": ["chat"], "not_a_field": 1})
        assert profile.tasks == [Task.CHAT]


class TestRoundTrip:
    def test_a_saved_profile_reloads_identically(self, tmp_path):
        """The guarantee that makes a run reproducible."""
        original = UseCaseProfile(
            tasks=[Task.CODING, Task.AGENT], clients=[Client.LM_STUDIO],
            priority=Priority.QUALITY, context_tokens=65536, needs_tools=True,
            concurrent_users=4, languages=["en", "de"], commercial_use=True,
        )
        path = tmp_path / "profile.yaml"
        save_profile(original, path)
        reloaded = profile_from_answers(load_answers(path))
        assert reloaded.to_dict() == original.to_dict()

    def test_a_full_setup_json_can_be_reloaded_as_answers(self, tmp_path):
        import json

        path = tmp_path / "last-setup.json"
        path.write_text(json.dumps({"profile": {"tasks": ["rag"],
                                                "context_tokens": 32768}}))
        assert profile_from_answers(load_answers(path)).tasks == [Task.RAG]

    def test_a_malformed_answers_file_reports_clearly(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("- this is a list, not a mapping\n")
        with pytest.raises(LlmsetupError):
            profile_from_answers(load_answers(path))
