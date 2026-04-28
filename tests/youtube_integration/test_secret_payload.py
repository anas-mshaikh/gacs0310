import pytest

from src.youtube_integration.config import DEFAULT_READONLY_SCOPES
from src.youtube_integration.errors import YouTubeSecretError
from src.youtube_integration.secrets import YouTubeOAuthSecret


def test_secret_payload_accepts_valid_mapping():
    secret = YouTubeOAuthSecret.from_mapping(
        {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token",
            "scopes": list(DEFAULT_READONLY_SCOPES),
        }
    )

    assert secret.client_id == "client-id"
    assert secret.client_secret == "client-secret"
    assert secret.refresh_token == "refresh-token"
    assert secret.token_uri == "https://oauth2.googleapis.com/token"
    assert secret.scopes == DEFAULT_READONLY_SCOPES


def test_secret_payload_rejects_missing_required_fields():
    with pytest.raises(YouTubeSecretError):
        YouTubeOAuthSecret.from_mapping(
            {
                "client_id": "client-id",
                "client_secret": "client-secret",
            }
        )


def test_secret_payload_rejects_missing_scope():
    with pytest.raises(YouTubeSecretError):
        YouTubeOAuthSecret.from_mapping(
            {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token",
                "scopes": ["https://www.googleapis.com/auth/yt-analytics.readonly"],
            }
        )
