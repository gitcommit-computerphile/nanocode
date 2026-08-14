"""Listing the models a key can actually reach.

Asked straight of the provider rather than kept as a hardcoded list, because a
baked-in list is wrong within weeks and wrong in the worst way — it hides the
model you wanted. The provider's own answer is the only one that stays true.

Both providers are queried over plain HTTP rather than through their SDKs. One
code path, no client construction, and nothing to keep in step when an SDK
changes shape.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .orchestrator import PROVIDER_KEYS

ENDPOINTS = {
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
}

TIMEOUT = 20

# The model list is everything the key can reach, most of which cannot hold a
# conversation. Filtering by what a coding agent can use keeps the picker short.
NOT_CHAT = (
    "embedding",
    "whisper",
    "tts",
    "dall-e",
    "moderation",
    "audio",
    "transcribe",
    "image",
    "realtime",
    "davinci",
    "babbage",
    "guard",
)


class ModelListError(Exception):
    """The provider could not be asked, or refused."""


def api_key_for(provider: str) -> str | None:
    """The key currently in the environment for a provider, if any."""
    var = PROVIDER_KEYS.get(provider)
    return os.environ.get(var) if var else None


def list_models(provider: str, api_key: str | None = None) -> list[str]:
    """Every chat model `provider` will serve this key, as `provider:name`.

    Raises ModelListError rather than returning a stale guess — a wrong list is
    worse than no list, because it looks authoritative.
    """
    provider = provider.strip().lower()
    if provider not in ENDPOINTS:
        raise ModelListError(f"unknown provider {provider!r}")

    key = api_key or api_key_for(provider)
    if not key:
        raise ModelListError(f"{PROVIDER_KEYS[provider]} is not set")

    request = urllib.request.Request(ENDPOINTS[provider], headers=_headers(provider, key))
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = "check the key" if exc.code in (401, 403) else exc.reason
        raise ModelListError(f"{provider} returned {exc.code} — {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ModelListError(f"could not reach {provider}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelListError(f"{provider} sent something unreadable: {exc}") from exc

    ids = [
        entry["id"]
        for entry in payload.get("data") or []
        if isinstance(entry, dict) and entry.get("id") and is_chat_model(entry["id"])
    ]
    return [f"{provider}:{name}" for name in sorted(set(ids), reverse=True)]


def is_chat_model(name: str) -> bool:
    lowered = name.lower()
    return not any(marker in lowered for marker in NOT_CHAT)


def _headers(provider: str, key: str) -> dict[str, str]:
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}
