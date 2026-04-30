"""
Later API clients only need to consume YouTubeSyncTask.
Scheduler can be tested without Google APIs.
We avoid mixing scheduling decisions with API implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class VideoLifecycleTier(str, Enum):
    DAYS_0_TO_7 = "0_7_days"
    DAYS_8_TO_30 = "8_30_days"
    DAYS_31_PLUS = "31_plus_days"


class YouTubeSyncTaskType(str, Enum):
    CURRENT_STATS = "current_stats"
    DAILY_ANALYTICS = "daily_analytics"
    TRAFFIC_SOURCES = "traffic_sources"


@dataclass(frozen=True)
class YouTubeVideoSyncCandidate:
    youtube_video_id: str
    job_id: str
    book_id: str | None
    published_at: datetime
    last_data_api_sync_at: datetime | None
    last_analytics_sync_at: datetime | None


@dataclass(frozen=True)
class AnalyticsDateWindow:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class YouTubeSyncTask:
    youtube_video_id: str
    task_type: YouTubeSyncTaskType
    lifecycle_tier: VideoLifecycleTier
    analytics_window: AnalyticsDateWindow | None = None


@dataclass(frozen=True)
class YouTubeSyncPlan:
    generated_at: datetime
    tasks: list[YouTubeSyncTask]

    @property
    def task_count(self) -> int:
        return len(self.tasks)
