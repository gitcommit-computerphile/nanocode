"""Asking a provider what a key can actually reach.

The list is fetched rather than hardcoded, so what these pin is the failure
behaviour: a wrong list is worse than no list, because it looks authoritative.
No test touches the network.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from nanocode import models
from nanocode.models import ModelListError, is_chat_model, list_models


class FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured(monkeypatch):
    """Serve a canned payload and record the request that asked for it."""
    seen: dict = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.headers)
        return FakeResponse(seen.get("payload", {"data": []}))

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    return seen


# -- filtering ------------------------------------------------------------


@pytest.mark.parametrize("name", ["gpt-5.4-mini", "gpt-5.6-sol", "claude-opus-5", "o3"])
def test_chat_models_are_kept(name):
    assert is_chat_model(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "text-embedding-3-large",
        "whisper-1",
        "tts-1-hd",
        "dall-e-3",
        "omni-moderation-latest",
        "gpt-4o-realtime-preview",
        "davinci-002",
    ],
)
def test_non_chat_models_are_dropped(name):
    """A coding agent can't use these, and they'd bury the ones it can."""
    assert is_chat_model(name) is False


# -- the request ----------------------------------------------------------


def test_openai_is_asked_with_a_bearer_token(captured, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    captured["payload"] = {"data": [{"id": "gpt-5.4-mini"}]}

    assert list_models("openai") == ["openai:gpt-5.4-mini"]
    assert captured["url"] == "https://api.openai.com/v1/models"
    assert captured["headers"]["Authorization"] == "Bearer sk-openai"


def test_anthropic_is_asked_with_its_own_headers(captured, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    captured["payload"] = {"data": [{"id": "claude-opus-5"}]}

    assert list_models("anthropic") == ["anthropic:claude-opus-5"]
    assert captured["headers"]["X-api-key"] == "sk-ant"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"


def test_results_are_prefixed_deduped_and_newest_first(captured, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    captured["payload"] = {
        "data": [
            {"id": "gpt-5.4-mini"},
            {"id": "gpt-5.6-sol"},
            {"id": "gpt-5.4-mini"},
            {"id": "text-embedding-3-large"},
            {"not": "an id"},
        ]
    }

    assert list_models("openai") == ["openai:gpt-5.6-sol", "openai:gpt-5.4-mini"]


def test_an_explicit_key_beats_the_environment(captured, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured["payload"] = {"data": [{"id": "gpt-5.4-mini"}]}

    assert list_models("openai", api_key="sk-passed-in") == ["openai:gpt-5.4-mini"]


# -- failure --------------------------------------------------------------


def test_a_missing_key_is_reported_before_any_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelListError, match="OPENAI_API_KEY"):
        list_models("openai")


def test_a_rejected_key_says_so(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong")

    def unauthorized(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(models.urllib.request, "urlopen", unauthorized)
    with pytest.raises(ModelListError, match="401 — check the key"):
        list_models("openai")


def test_an_unreachable_provider_raises_rather_than_guessing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    def offline(request, timeout=None):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(models.urllib.request, "urlopen", offline)
    with pytest.raises(ModelListError, match="could not reach openai"):
        list_models("openai")


def test_an_unreadable_response_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    class Garbage:
        def read(self):
            return b"<html>not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(models.urllib.request, "urlopen", lambda r, timeout=None: Garbage())
    with pytest.raises(ModelListError, match="unreadable"):
        list_models("openai")


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ModelListError, match="unknown provider"):
        list_models("gemini")
