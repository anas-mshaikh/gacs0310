from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping, Protocol

from .config import DEFAULT_READONLY_SCOPES
from .errors import YouTubeSecretError


DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"


@dataclass(frozen=True)
class YouTubeOAuthSecret:
    client_id: str
    client_secret: str
    refresh_token: str
    token_uri: str
    scopes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "YouTubeOAuthSecret":
        missing = [key for key in ("client_id", "client_secret", "refresh_token") if not payload.get(key)]

        if missing:
            raise YouTubeSecretError("Missing required YouTube OAuth secret fields: " + ", ".join(sorted(missing)))

        raw_scopes = payload.get("scopes", DEFAULT_READONLY_SCOPES)

        if isinstance(raw_scopes, str):
            scopes = tuple(scope.strip() for scope in raw_scopes.split(",") if scope.strip())
        elif isinstance(raw_scopes, list):
            scopes = tuple(str(scope).strip() for scope in raw_scopes if str(scope).strip())
        elif isinstance(raw_scopes, tuple):
            scopes = tuple(str(scope).strip() for scope in raw_scopes if str(scope).strip())
        else:
            raise YouTubeSecretError("YouTube OAuth secret field 'scopes' is invalid.")

        required = set(DEFAULT_READONLY_SCOPES)
        missing_scopes = required - set(scopes)

        if missing_scopes:
            raise YouTubeSecretError(
                "YouTube OAuth secret is missing required scopes: " + ", ".join(sorted(missing_scopes))
            )

        return cls(
            client_id=str(payload["client_id"]),
            client_secret=str(payload["client_secret"]),
            refresh_token=str(payload["refresh_token"]),
            token_uri=str(payload.get("token_uri") or DEFAULT_TOKEN_URI),
            scopes=scopes,
        )


class SecretBackend(Protocol):
    def load_youtube_oauth_secret(self, secret_name: str | None = None) -> YouTubeOAuthSecret: ...


class EnvSecretBackend:
    """Loads YouTube OAuth credentials from environment variables.

    This is intended for local development and tests. Staging/production should use
    a managed secret backend.
    """

    def load_youtube_oauth_secret(self, secret_name: str | None = None) -> YouTubeOAuthSecret:
        del secret_name

        json_payload = os.getenv("GACS_YOUTUBE_OAUTH_SECRET_JSON")

        if json_payload:
            try:
                payload = json.loads(json_payload)
            except json.JSONDecodeError as exc:
                raise YouTubeSecretError("GACS_YOUTUBE_OAUTH_SECRET_JSON is not valid JSON.") from exc

            return YouTubeOAuthSecret.from_mapping(payload)

        payload = {
            "client_id": os.getenv("GACS_YOUTUBE_CLIENT_ID"),
            "client_secret": os.getenv("GACS_YOUTUBE_CLIENT_SECRET"),
            "refresh_token": os.getenv("GACS_YOUTUBE_REFRESH_TOKEN"),
            "token_uri": os.getenv("GACS_YOUTUBE_TOKEN_URI", DEFAULT_TOKEN_URI),
            "scopes": os.getenv("GACS_YOUTUBE_READONLY_SCOPES", ",".join(DEFAULT_READONLY_SCOPES)),
        }

        return YouTubeOAuthSecret.from_mapping(payload)


class ManagedSecretBackendNotConfigured:
    """Placeholder for PR2.

    GCP/AWS implementation should be added when the deployment secret manager is
    confirmed by the GenTA team.
    """

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name

    def load_youtube_oauth_secret(self, secret_name: str | None = None) -> YouTubeOAuthSecret:
        raise YouTubeSecretError(
            f"{self.backend_name} backend is not implemented yet. "
            "Use env backend for local tests or configure the deployment secret manager."
        )


def build_secret_backend(name: str) -> SecretBackend:
    if name == "env":
        return EnvSecretBackend()

    if name in {"gcp_secret_manager", "aws_secrets_manager"}:
        return ManagedSecretBackendNotConfigured(name)

    raise YouTubeSecretError(f"Unsupported YouTube secret backend: {name}")
