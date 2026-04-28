from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from .errors import YouTubeOAuthError
from .secrets import YouTubeOAuthSecret


@dataclass(frozen=True)
class RefreshedCredentials:
    credentials: Credentials
    expiry: datetime | None


def build_google_credentials(secret: YouTubeOAuthSecret) -> Credentials:
    """Build refreshable Google OAuth credentials from a stored refresh token."""

    return Credentials(
        token=None,
        refresh_token=secret.refresh_token,
        token_uri=secret.token_uri,
        client_id=secret.client_id,
        client_secret=secret.client_secret,
        scopes=list(secret.scopes),
    )


def refresh_google_credentials(credentials: Credentials) -> RefreshedCredentials:
    """Refresh access token using the stored refresh token."""

    try:
        credentials.refresh(Request())
    except Exception as exc:  # Google auth raises several transport/token exceptions.
        raise YouTubeOAuthError("Failed to refresh YouTube OAuth access token.") from exc

    expiry = credentials.expiry

    if expiry is not None and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    return RefreshedCredentials(credentials=credentials, expiry=expiry)


def build_and_refresh_credentials(secret: YouTubeOAuthSecret) -> RefreshedCredentials:
    credentials = build_google_credentials(secret)
    return refresh_google_credentials(credentials)
