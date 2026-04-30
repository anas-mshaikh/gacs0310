from datetime import datetime, timedelta, timezone

from src.youtube_integration.scheduler import YouTubeLifecycleScheduler
from src.youtube_integration.sync_models import YouTubeVideoSyncCandidate


class FakeRepository:
    def list_sync_candidates(self, *, limit=None):
        del limit

        now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

        return [
            YouTubeVideoSyncCandidate(
                youtube_video_id="abc123",
                job_id="job-1",
                book_id=None,
                published_at=now - timedelta(days=1),
                last_data_api_sync_at=None,
                last_analytics_sync_at=None,
            )
        ]


class FailingExecutor:
    def execute_task(self, task):
        raise AssertionError("Dry-run scheduler should not execute tasks")


def test_scheduler_dry_run_builds_plan_without_execution():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    scheduler = YouTubeLifecycleScheduler(
        repository=FakeRepository(),
        executor=FailingExecutor(),
        dry_run=True,
    )

    plan = scheduler.run(now=now)

    assert plan.task_count == 3
