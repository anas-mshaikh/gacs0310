from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import YouTubeConfigError


DEFAULT_ANALYTICS_IDS = "channel==MINE"

DEFAULT_READONLY_SCOPES = (
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
)


@dataclass(frozen=True)
class YouTubeIntegrationConfig:
    enabled: bool
    secret_backend: str
    oauth_secret_name: str | None
    analytics_ids: str
    dry_run: bool
    readonly_scopes: tuple[str, ...]
    max_videos_per_batch: int

    @classmethod
    def from_env(cls) -> "YouTubeIntegrationConfig":
        return cls(
            enabled=_get_bool("GACS_YOUTUBE_ENABLED", default=False),
            secret_backend=os.getenv("GACS_YOUTUBE_SECRET_BACKEND", "env"),
            oauth_secret_name=os.getenv("GACS_YOUTUBE_OAUTH_SECRET_NAME"),
            analytics_ids=os.getenv("GACS_YOUTUBE_ANALYTICS_IDS", DEFAULT_ANALYTICS_IDS),
            dry_run=_get_bool("GACS_YOUTUBE_SYNC_DRY_RUN", default=True),
            readonly_scopes=_get_scopes(),
            max_videos_per_batch=_get_int("GACS_YOUTUBE_MAX_VIDEOS_PER_BATCH", default=50),
        )

    def validate(self) -> None:
        allowed_backends = {"env", "gcp_secret_manager", "aws_secrets_manager"}

        if self.secret_backend not in allowed_backends:
            raise YouTubeConfigError(f"Unsupported YouTube secret backend: {self.secret_backend}")

        if self.enabled and self.secret_backend != "env" and not self.oauth_secret_name:
            raise YouTubeConfigError(
                "GACS_YOUTUBE_OAUTH_SECRET_NAME is required when YouTube sync is enabled with a managed secret backend."
            )

        if not self.analytics_ids.startswith(("channel==", "contentOwner==")):
            raise YouTubeConfigError("GACS_YOUTUBE_ANALYTICS_IDS must start with 'channel==' or 'contentOwner=='.")

        if self.max_videos_per_batch < 1 or self.max_videos_per_batch > 50:
            raise YouTubeConfigError("GACS_YOUTUBE_MAX_VIDEOS_PER_BATCH must be between 1 and 50.")

        required_scopes = set(DEFAULT_READONLY_SCOPES)
        configured_scopes = set(self.readonly_scopes)
        missing = required_scopes - configured_scopes

        if missing:
            raise YouTubeConfigError("Missing required YouTube OAuth scopes: " + ", ".join(sorted(missing)))


def _get_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    normalized = raw.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise YouTubeConfigError(f"{name} must be a boolean value.")


def _get_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)

    if raw is None:
        return default

    try:
        return int(raw)
    except ValueError as exc:
        raise YouTubeConfigError(f"{name} must be an integer.") from exc


def _get_scopes() -> tuple[str, ...]:
    raw = os.getenv("GACS_YOUTUBE_READONLY_SCOPES")

    if not raw:
        return DEFAULT_READONLY_SCOPES

    return tuple(scope.strip() for scope in raw.split(",") if scope.strip())
