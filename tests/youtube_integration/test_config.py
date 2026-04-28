import pytest

from src.youtube_integration.config import (
    DEFAULT_READONLY_SCOPES,
    YouTubeIntegrationConfig,
)
from src.youtube_integration.errors import YouTubeConfigError


def test_config_defaults_to_disabled_env_backend(monkeypatch):
    monkeypatch.delenv("GACS_YOUTUBE_ENABLED", raising=False)
    monkeypatch.delenv("GACS_YOUTUBE_SECRET_BACKEND", raising=False)

    config = YouTubeIntegrationConfig.from_env()

    assert config.enabled is False
    assert config.secret_backend == "env"
    assert config.analytics_ids == "channel==MINE"
    assert config.dry_run is True
    assert config.readonly_scopes == DEFAULT_READONLY_SCOPES
    assert config.max_videos_per_batch == 50


def test_config_rejects_invalid_batch_size(monkeypatch):
    monkeypatch.setenv("GACS_YOUTUBE_MAX_VIDEOS_PER_BATCH", "100")

    config = YouTubeIntegrationConfig.from_env()

    with pytest.raises(YouTubeConfigError):
        config.validate()


def test_config_requires_secret_name_for_managed_backend_when_enabled(monkeypatch):
    monkeypatch.setenv("GACS_YOUTUBE_ENABLED", "true")
    monkeypatch.setenv("GACS_YOUTUBE_SECRET_BACKEND", "gcp_secret_manager")
    monkeypatch.delenv("GACS_YOUTUBE_OAUTH_SECRET_NAME", raising=False)

    config = YouTubeIntegrationConfig.from_env()

    with pytest.raises(YouTubeConfigError):
        config.validate()


def test_config_rejects_missing_required_scope(monkeypatch):
    monkeypatch.setenv(
        "GACS_YOUTUBE_READONLY_SCOPES",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    )

    config = YouTubeIntegrationConfig.from_env()

    with pytest.raises(YouTubeConfigError):
        config.validate()
