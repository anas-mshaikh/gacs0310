from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from .sync_models import (
    AnalyticsDateWindow,
    VideoLifecycleTier,
    YouTubeVideoSyncCandidate,
    YouTubeSyncPlan,
    YouTubeSyncTask,
    YouTubeSyncTaskType,
)
from .sync_ports import YouTubeSyncExecutor, YouTubeSyncRepository


def classify_lifecycle_tier(
    *,
    published_at: datetime,
    now: datetime,
) -> VideoLifecycleTier:
    published_date = published_at.astimezone(timezone.utc).date()
    now_date = now.astimezone(timezone.utc).date()

    age_days = max((now_date - published_date).days, 0)

    if age_days <= 7:
        return VideoLifecycleTier.DAYS_0_TO_7

    if age_days <= 30:
        return VideoLifecycleTier.DAYS_8_TO_30

    return VideoLifecycleTier.DAYS_31_PLUS


def current_stats_interval_for_tier(tier: VideoLifecycleTier) -> timedelta:
    if tier == VideoLifecycleTier.DAYS_0_TO_7:
        return timedelta(hours=6)

    if tier == VideoLifecycleTier.DAYS_8_TO_30:
        return timedelta(days=1)

    return timedelta(days=3)


def analytics_interval_for_tier(tier: VideoLifecycleTier) -> timedelta:
    if tier == VideoLifecycleTier.DAYS_0_TO_7:
        return timedelta(days=1)

    if tier == VideoLifecycleTier.DAYS_8_TO_30:
        return timedelta(days=1)

    return timedelta(days=3)


def analytics_window_for_tier(
    *,
    tier: VideoLifecycleTier,
    now: datetime,
) -> AnalyticsDateWindow:
    today = now.astimezone(timezone.utc).date()

    # Do not use same-day analytics as the only source of truth.
    # YouTube analytics can be delayed or revised.
    end_date = today - timedelta(days=1)

    if tier == VideoLifecycleTier.DAYS_0_TO_7:
        start_date = end_date - timedelta(days=6)
    elif tier == VideoLifecycleTier.DAYS_8_TO_30:
        start_date = end_date - timedelta(days=13)
    else:
        start_date = end_date - timedelta(days=6)

    return AnalyticsDateWindow(start_date=start_date, end_date=end_date)


def is_due(
    *,
    last_synced_at: datetime | None,
    now: datetime,
    interval: timedelta,
) -> bool:
    if last_synced_at is None:
        return True

    last_synced_utc = last_synced_at.astimezone(timezone.utc)
    now_utc = now.astimezone(timezone.utc)

    return now_utc - last_synced_utc >= interval


def build_sync_tasks_for_candidate(
    *,
    candidate: YouTubeVideoSyncCandidate,
    now: datetime,
) -> list[YouTubeSyncTask]:
    tier = classify_lifecycle_tier(
        published_at=candidate.published_at,
        now=now,
    )

    tasks: list[YouTubeSyncTask] = []

    if is_due(
        last_synced_at=candidate.last_data_api_sync_at,
        now=now,
        interval=current_stats_interval_for_tier(tier),
    ):
        tasks.append(
            YouTubeSyncTask(
                youtube_video_id=candidate.youtube_video_id,
                task_type=YouTubeSyncTaskType.CURRENT_STATS,
                lifecycle_tier=tier,
            )
        )

    if is_due(
        last_synced_at=candidate.last_analytics_sync_at,
        now=now,
        interval=analytics_interval_for_tier(tier),
    ):
        analytics_window = analytics_window_for_tier(tier=tier, now=now)

        tasks.append(
            YouTubeSyncTask(
                youtube_video_id=candidate.youtube_video_id,
                task_type=YouTubeSyncTaskType.DAILY_ANALYTICS,
                lifecycle_tier=tier,
                analytics_window=analytics_window,
            )
        )

        tasks.append(
            YouTubeSyncTask(
                youtube_video_id=candidate.youtube_video_id,
                task_type=YouTubeSyncTaskType.TRAFFIC_SOURCES,
                lifecycle_tier=tier,
                analytics_window=analytics_window,
            )
        )

    return tasks


class YouTubeLifecycleScheduler:
    def __init__(
        self,
        *,
        repository: YouTubeSyncRepository,
        executor: YouTubeSyncExecutor | None = None,
        dry_run: bool = True,
    ) -> None:
        self.repository = repository
        self.executor = executor
        self.dry_run = dry_run

    def build_plan(
        self,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> YouTubeSyncPlan:
        generated_at = now or datetime.now(timezone.utc)
        candidates = self.repository.list_sync_candidates(limit=limit)

        tasks: list[YouTubeSyncTask] = []

        for candidate in candidates:
            tasks.extend(
                build_sync_tasks_for_candidate(
                    candidate=candidate,
                    now=generated_at,
                )
            )

        return YouTubeSyncPlan(generated_at=generated_at, tasks=tasks)

    def run(
        self,
        *,
        now: datetime | None = None,
        limit: int | None = None,
    ) -> YouTubeSyncPlan:
        plan = self.build_plan(now=now, limit=limit)

        if self.dry_run:
            return plan

        if self.executor is None:
            raise RuntimeError("YouTube sync executor is required when dry_run=False.")

        for task in plan.tasks:
            self.executor.execute_task(task)

        return plan
