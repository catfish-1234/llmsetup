"""Verification must not claim success when a check could not establish it."""

import json

import pytest
from conftest import make_hardware

from llmsetup.cli import main
from llmsetup.runtimes.base import Endpoint
from llmsetup.types import CheckResult, CheckStatus, RuntimeConfig, Severity, UseCaseProfile
from llmsetup.util.errors import LlmsetupError
from llmsetup.util.http import HttpResponse
from llmsetup.util.log import Console
from llmsetup.verify.checks import check_health
from llmsetup.verify.client import OpenAIClient
from llmsetup.verify.harness import ALL_CHECKS, Harness


def harness(catalog):
    return Harness(
        OpenAIClient(Endpoint("http://localhost:1234", "m", "external")),
        catalog.require("qwen3-8b"), RuntimeConfig("external", "m", "Q4_K_M", 8192),
        UseCaseProfile(), make_hardware(), Console(quiet=True),
    )


def test_unexpected_check_error_blocks_success(catalog, monkeypatch):
    def broken(ctx):
        raise ValueError("malformed server response")

    monkeypatch.setitem(ALL_CHECKS, "health", broken)
    results = harness(catalog).run(["health", "completion"])
    assert results[0].blocking
    assert "malformed server response" in results[0].detail
    assert results[1].status is CheckStatus.SKIP


def test_intentional_advisory_failure_keeps_its_severity(catalog, monkeypatch):
    monkeypatch.setitem(ALL_CHECKS, "streaming", lambda ctx: CheckResult(
        "streaming", CheckStatus.FAIL, Severity.ADVISORY, "optional feature"))
    assert not harness(catalog).run(["streaming"])[0].blocking


def test_unknown_checks_are_rejected_before_any_request(catalog, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("must validate names before contacting the endpoint")

    monkeypatch.setattr("llmsetup.util.http.request", unexpected)
    with pytest.raises(LlmsetupError, match="unknown verification check"):
        harness(catalog).run(["helath"])


def test_cli_unknown_check_exits_nonzero_with_hint(capsys):
    assert main(["verify", "--model", "m", "--checks", "helath", "--yes"]) == 1
    assert "valid checks:" in capsys.readouterr().err


def test_iterator_reports_remaining_checks_after_failure(catalog, monkeypatch):
    monkeypatch.setitem(ALL_CHECKS, "health", lambda ctx: CheckResult(
        "health", CheckStatus.FAIL, Severity.CRITICAL, "down"))
    results = harness(catalog).run(iter(["health", "completion", "stability"]))
    assert [r.name for r in results] == ["health", "completion", "stability"]
    assert all(r.status is CheckStatus.SKIP for r in results[1:])


def test_empty_model_listing_does_not_prove_health(catalog, monkeypatch):
    instance = harness(catalog)
    monkeypatch.setattr(instance.client, "list_models", lambda: {"ok": True, "ids": []})
    assert check_health(instance.context()).blocking


@pytest.mark.parametrize("payload", [None, [], {}, {"data": "wrong"}, {"data": [None]}])
def test_malformed_model_listing_is_a_failure(payload, monkeypatch):
    monkeypatch.setattr("llmsetup.util.http.request", lambda *a, **kw:
                        HttpResponse(200, json.dumps(payload)))
    assert not OpenAIClient(Endpoint("http://localhost", "m", "external")).list_models()["ok"]


def test_model_listing_sends_endpoint_auth_headers(monkeypatch):
    def request(url, **kwargs):
        assert kwargs["headers"] == {"Authorization": "Bearer test-token"}
        return HttpResponse(200, '{"data": [{"id": "m"}]}')

    monkeypatch.setattr("llmsetup.util.http.request", request)
    client = OpenAIClient(Endpoint("http://localhost", "m", "external",
                                  headers={"Authorization": "Bearer test-token"}))
    assert client.list_models() == {"ok": True, "ids": ["m"]}
