"""Retrying transient provider failures.

The classification is what actually matters here: retrying a bad API key three
times just makes the user wait longer for the same error. So these tests use
the *real* OpenAI and Anthropic exception classes rather than stand-ins — a
hand-rolled fake would happily agree with whatever the code assumed.

No test sleeps: `make_retrier` takes an injectable clock.
"""

from __future__ import annotations

import itertools

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from nanocode.cli import _drive
from nanocode.orchestrator import build_orchestrator
from nanocode.retry import (
    MAX_ATTEMPTS,
    delay_for,
    is_transient,
    make_retrier,
    retry_after,
)
from nanocode.ui import NanocodeUI

pytestmark = pytest.mark.anyio
_ids = itertools.count()


# -- classification, against the real SDK exception types -----------------


def _openai_error(cls_name: str, status: int):
    """Build a genuine openai exception, however that SDK wants it built."""
    import httpx
    import openai

    cls = getattr(openai, cls_name)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    "cls_name,status",
    [("RateLimitError", 429), ("InternalServerError", 500), ("InternalServerError", 529)],
)
def test_transient_provider_errors_are_retried(cls_name, status):
    assert is_transient(_openai_error(cls_name, status)) is True


@pytest.mark.parametrize(
    "cls_name,status",
    [("AuthenticationError", 401), ("BadRequestError", 400), ("PermissionDeniedError", 403)],
)
def test_permanent_provider_errors_are_not_retried(cls_name, status):
    """A bad key fails identically three times — surface it immediately."""
    assert is_transient(_openai_error(cls_name, status)) is False


def test_connection_and_timeout_errors_are_retried():
    """These carry no status code because the request never landed."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    assert is_transient(openai.APIConnectionError(request=request)) is True
    assert is_transient(openai.APITimeoutError(request=request)) is True
    assert is_transient(httpx.ConnectTimeout("slow")) is True
    assert is_transient(TimeoutError()) is True


def test_an_unrecognised_error_is_treated_as_permanent():
    """Retrying a bug just delays the traceback."""
    assert is_transient(ValueError("something in our own code broke")) is False
    assert is_transient(KeyError("nope")) is False


def test_a_status_code_beats_a_familiar_class_name():
    """A 400 must not be retried merely for inheriting a retryable-looking name."""

    class APITimeoutError(Exception):  # same name, wrong meaning
        status_code = 400

    assert is_transient(APITimeoutError()) is False


# -- the provider's own advice wins ---------------------------------------


class _Advising(Exception):
    def __init__(self, seconds, status=429):
        super().__init__("slow down")
        self.status_code = status

        class _Response:
            headers = {"retry-after": str(seconds)}

        self.response = _Response()


def test_retry_after_is_honoured_over_backoff():
    exc = _Advising(7)
    assert retry_after(exc) == 7.0
    assert delay_for(1, exc) == 7.0, "the provider knows its own limits"


def test_a_junk_retry_after_falls_back_to_backoff():
    exc = _Advising("next tuesday")
    assert retry_after(exc) is None
    assert 0 < delay_for(1, exc) <= 1.0


def test_backoff_grows_and_stays_jittered():
    plain = RuntimeError("no headers here")
    plain.status_code = 500
    first = [delay_for(1, plain) for _ in range(20)]
    third = [delay_for(3, plain) for _ in range(20)]

    assert max(first) <= 1.0 and min(first) > 0
    assert max(third) <= 4.0 and min(third) > 1.0, "attempt 3 waits longer than attempt 1"
    assert len(set(first)) > 1, "jitter keeps concurrent runs from retrying in lockstep"


# -- the middleware -------------------------------------------------------


def _request():
    return object()


def test_a_transient_failure_is_retried_then_succeeds():
    slept: list[float] = []
    attempts = itertools.count(1)

    def handler(_request):
        if next(attempts) < 3:
            raise _Advising(2)
        return "ok"

    retrier = make_retrier(on_retry=None, sleep=slept.append)
    assert retrier.wrap_model_call(_request(), handler) == "ok"
    assert slept == [2.0, 2.0], "waited between each attempt"


def test_a_permanent_failure_is_not_retried():
    calls: list[int] = []

    def handler(_request):
        calls.append(1)
        raise _openai_error("AuthenticationError", 401)

    with pytest.raises(Exception, match="boom"):
        make_retrier(sleep=lambda _: None).wrap_model_call(_request(), handler)
    assert len(calls) == 1, "a bad key should fail on the first try"


def test_it_gives_up_and_surfaces_the_original_error():
    def handler(_request):
        raise _Advising(1)

    with pytest.raises(_Advising):
        make_retrier(sleep=lambda _: None).wrap_model_call(_request(), handler)


def test_attempts_are_capped():
    calls: list[int] = []

    def handler(_request):
        calls.append(1)
        raise _Advising(1)

    with pytest.raises(_Advising):
        make_retrier(sleep=lambda _: None).wrap_model_call(_request(), handler)
    assert len(calls) == MAX_ATTEMPTS


def test_the_user_is_told_rather_than_left_watching_silence():
    notices: list[str] = []
    attempts = itertools.count(1)

    def handler(_request):
        if next(attempts) < 3:
            raise _Advising(2)
        return "ok"

    make_retrier(on_retry=notices.append, sleep=lambda _: None).wrap_model_call(_request(), handler)

    assert len(notices) == 2
    assert "rate limited" in notices[0]
    assert "retrying in 2s (2/3)" in notices[0]


async def test_the_async_path_retries_too():
    """nanocode-web drives the graph with astream — same behaviour required."""
    attempts = itertools.count(1)

    async def handler(_request):
        if next(attempts) < 2:
            raise _Advising(0)
        return "ok"

    retrier = make_retrier(sleep=lambda _: None)
    assert await retrier.awrap_model_call(_request(), handler) == "ok"


# -- through the real graph ----------------------------------------------


class Flaky(BaseChatModel):
    """Fails with a 529 a set number of times, then answers."""

    failures: int = 0
    calls: list = []

    def __init__(self, failures: int) -> None:
        super().__init__(failures=failures, calls=[])

    @property
    def _llm_type(self) -> str:
        return "flaky"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(1)
        if len(self.calls) <= self.failures:
            raise _Advising(0, status=529)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("recovered"))])


def test_a_real_run_survives_a_transient_blip(tmp_path, monkeypatch):
    """The whole point: a 529 twenty tool calls deep shouldn't end the task."""
    monkeypatch.setattr("nanocode.retry.time.sleep", lambda _: None)
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")

    notices: list[str] = []
    model = Flaky(failures=2)
    orch = build_orchestrator(model=model, root=tmp_path, on_retry=notices.append)
    state = _drive(orch, "do the thing", NanocodeUI(live=False))

    assert "recovered" in str(state["messages"][-1].content)
    assert len(model.calls) == 3, "two failures, then success"
    assert len(notices) == 2


def test_a_real_run_still_reports_a_permanent_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("nanocode.retry.time.sleep", lambda _: None)

    class Broken(Flaky):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls.append(1)
            raise _openai_error("AuthenticationError", 401)

    model = Broken(failures=0)
    orch = build_orchestrator(model=model, root=tmp_path)
    with pytest.raises(Exception, match="boom"):
        _drive(orch, "do the thing", NanocodeUI(live=False))
    assert len(model.calls) == 1, "no retries on a bad key"