from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime

from .sync_models import VideoLifecycleTier, YouTubeSyncTaskType


@dataclass(frozen=True)
class YouTubeSyncLogEvent:
    event: str
    sync_type: YouTubeSyncTaskType
    youtube_video_id: str
    lifecycle_tier: VideoLifecycleTier
    generated_at: datetime
    start_date: date | None = None
    end_date: date | None = None
    dry_run: bool = True
    rows_upserted: int | None = None
    quota_units_estimated: int | None = None
    error_reason: str | None = None
    duration_ms: int | None = None
    correlation_id: str | None = None

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)
