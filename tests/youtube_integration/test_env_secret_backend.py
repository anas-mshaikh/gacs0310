import json

from src.youtube_integration.config import DEFAULT_READONLY_SCOPES
from src.youtube_integration.secrets import EnvSecretBackend


def test_env_secret_backend_loads_json_payload(monkeypatch):
    monkeypatch.setenv(
        "GACS_YOUTUBE_OAUTH_SECRET_JSON",
        json.dumps(
            {
                "client_id": "client-id",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token",
                "scopes": list(DEFAULT_READONLY_SCOPES),
            }
        ),
    )

    secret = EnvSecretBackend().load_youtube_oauth_secret()

    assert secret.client_id == "client-id"
    assert secret.refresh_token == "refresh-token"


def test_env_secret_backend_loads_split_env_vars(monkeypatch):
    monkeypatch.delenv("GACS_YOUTUBE_OAUTH_SECRET_JSON", raising=False)
    monkeypatch.setenv("GACS_YOUTUBE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GACS_YOUTUBE_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("GACS_YOUTUBE_REFRESH_TOKEN", "refresh-token")
    monkeypatch.setenv(
        "GACS_YOUTUBE_READONLY_SCOPES",
        ",".join(DEFAULT_READONLY_SCOPES),
    )

    secret = EnvSecretBackend().load_youtube_oauth_secret()

    assert secret.client_id == "client-id"
    assert secret.client_secret == "client-secret"
    assert secret.refresh_token == "refresh-token"
