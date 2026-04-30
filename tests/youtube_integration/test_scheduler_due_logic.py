from datetime import datetime, timedelta, timezone

from src.youtube_integration.scheduler import build_sync_tasks_for_candidate
from src.youtube_integration.sync_models import (
    YouTubeVideoSyncCandidate,
    YouTubeSyncTaskType,
)


def test_new_video_without_sync_history_gets_all_tasks():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    candidate = YouTubeVideoSyncCandidate(
        youtube_video_id="abc123",
        job_id="job-1",
        book_id="book-1",
        published_at=now - timedelta(days=1),
        last_data_api_sync_at=None,
        last_analytics_sync_at=None,
    )

    tasks = build_sync_tasks_for_candidate(candidate=candidate, now=now)
    task_types = {task.task_type for task in tasks}

    assert task_types == {
        YouTubeSyncTaskType.CURRENT_STATS,
        YouTubeSyncTaskType.DAILY_ANALYTICS,
        YouTubeSyncTaskType.TRAFFIC_SOURCES,
    }


def test_new_video_current_stats_not_due_before_six_hours():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    candidate = YouTubeVideoSyncCandidate(
        youtube_video_id="abc123",
        job_id="job-1",
        book_id=None,
        published_at=now - timedelta(days=2),
        last_data_api_sync_at=now - timedelta(hours=5),
        last_analytics_sync_at=now,
    )

    tasks = build_sync_tasks_for_candidate(candidate=candidate, now=now)

    assert tasks == []


def test_old_video_current_stats_due_after_three_days():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    candidate = YouTubeVideoSyncCandidate(
        youtube_video_id="old-video",
        job_id="job-1",
        book_id=None,
        published_at=now - timedelta(days=40),
        last_data_api_sync_at=now - timedelta(days=3, minutes=1),
        last_analytics_sync_at=now,
    )

    tasks = build_sync_tasks_for_candidate(candidate=candidate, now=now)

    assert [task.task_type for task in tasks] == [YouTubeSyncTaskType.CURRENT_STATS]
