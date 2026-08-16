"""Retrying transient provider failures.

A `429` or `529` is not a failure, it is a "not right now" — and without this,
one of them ends a task that was twenty tool calls deep. The middleware wraps
every model call, so nothing downstream (the agent loop, the tools, the UI)
needs to know a retry happened.

The classification is the load-bearing part. Retrying a bad API key three times
just makes the user wait longer for the same error, so the rule is narrow:
retry what is known to be transient, and let everything else through
immediately. An unrecognised error is treated as permanent on purpose — a bug
should surface fast, not after three sleeps.

Provider SDKs are deliberately not imported. OpenAI's and Anthropic's
exceptions expose the same `status_code` attribute and the same class names, so
duck-typing covers both, and any future provider that follows the convention.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

MAX_ATTEMPTS = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0

# Worth waiting out: rate limits, overloads, gateway hiccups, request timeouts.
# 529 is Anthropic's "overloaded", which is common enough to matter.
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Errors that never carry a status code because the request never landed.
TRANSIENT_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "APIResponseValidationError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "RemoteProtocolError",
        "ServiceUnavailableError",
        "OverloadedError",
    }
)


def is_transient(exc: BaseException) -> bool:
    """True when trying the identical request again might actually work."""
    status = _status_of(exc)
    if status is not None:
        # A status code is a definite answer — trust it over the class name, so
        # a 400 is never retried just because it inherits a familiar name.
        return status in TRANSIENT_STATUS
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in TRANSIENT_NAMES


def retry_after(exc: BaseException) -> float | None:
    """The provider's own instruction on how long to wait, if it sent one.

    A 429 usually carries `Retry-After`. The service knows its own limits far
    better than a fixed backoff curve does.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, min(float(value), MAX_DELAY))
    except (TypeError, ValueError):
        # The header may be an HTTP date rather than seconds; fall back.
        return None


def delay_for(attempt: int, exc: BaseException) -> float:
    """Seconds to wait before attempt N+1. Provider's advice wins."""
    if (advised := retry_after(exc)) is not None:
        return advised
    # Exponential, with jitter so concurrent runs don't retry in lockstep.
    backoff = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
    return backoff * (0.5 + random.random() / 2)


def make_retrier(
    max_attempts: int = MAX_ATTEMPTS,
    *,
    on_retry: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AgentMiddleware:
    """Middleware that retries transient model failures.

    `sleep` is injectable so tests don't actually wait. `on_retry` reports to
    the UI — several seconds of silence reads as a hang, so the wait is stated
    rather than hidden.
    """

    def _handle(exc: Exception, attempt: int) -> float:
        """Decide whether to retry, and how long to wait. Re-raises if not."""
        if attempt >= max_attempts or not is_transient(exc):
            raise exc
        wait = delay_for(attempt, exc)
        if on_retry:
            reason = _describe(exc)
            on_retry(f"{reason} — retrying in {wait:.0f}s ({attempt + 1}/{max_attempts})")
        return wait

    class _Retrier(AgentMiddleware):
        name = "retry_transient"

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            for attempt in range(1, max_attempts + 1):
                try:
                    return handler(request)
                except Exception as exc:  # noqa: BLE001 — re-raised unless transient
                    sleep(_handle(exc, attempt))
            raise AssertionError("unreachable")

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Any],
        ) -> ModelResponse:
            for attempt in range(1, max_attempts + 1):
                try:
                    return await handler(request)
                except Exception as exc:  # noqa: BLE001 — re-raised unless transient
                    await _async_sleep(_handle(exc, attempt))
            raise AssertionError("unreachable")

    return _Retrier()


async def _async_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _status_of(exc: BaseException) -> int | None:
    for attribute in ("status_code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _describe(exc: BaseException) -> str:
    status = _status_of(exc)
    if status == 429:
        return "rate limited"
    if status in (500, 502, 503, 504, 529):
        return "provider overloaded"
    if status is not None:
        return f"provider error {status}"
    return type(exc).__name__