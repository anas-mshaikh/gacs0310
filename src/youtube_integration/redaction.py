from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "client_secret",
    "refresh_token",
    "token",
    "youtube_credentials",
    "oauth_secret",
}


def redact_value(value: object) -> str:
    text = "" if value is None else str(value)

    if not text:
        return "<empty>"

    if len(text) <= 8:
        return "<redacted>"

    return f"{text[:4]}...{text[-4:]}<redacted>"


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}

    for key, value in payload.items():
        normalized_key = key.lower()

        if any(sensitive in normalized_key for sensitive in SENSITIVE_KEYS):
            redacted[key] = redact_value(value)
        elif isinstance(value, Mapping):
            redacted[key] = redact_mapping(value)
        else:
            redacted[key] = value

    return redacted
