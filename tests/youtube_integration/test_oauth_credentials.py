from src.youtube_integration.config import DEFAULT_READONLY_SCOPES
from src.youtube_integration.oauth import build_google_credentials
from src.youtube_integration.secrets import YouTubeOAuthSecret


def test_build_google_credentials_from_refresh_secret():
    secret = YouTubeOAuthSecret(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        token_uri="https://oauth2.googleapis.com/token",
        scopes=DEFAULT_READONLY_SCOPES,
    )

    credentials = build_google_credentials(secret)

    assert credentials.refresh_token == "refresh-token"
    assert credentials.client_id == "client-id"
    assert credentials.client_secret == "client-secret"
    assert credentials.token_uri == "https://oauth2.googleapis.com/token"
    assert set(credentials.scopes or []) == set(DEFAULT_READONLY_SCOPES)
