from datetime import datetime, timedelta, timezone

from src.youtube_integration.scheduler import classify_lifecycle_tier
from src.youtube_integration.sync_models import VideoLifecycleTier


def test_classifies_zero_to_seven_day_tier():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)
    published_at = now - timedelta(days=7)

    assert classify_lifecycle_tier(published_at=published_at, now=now) == (VideoLifecycleTier.DAYS_0_TO_7)


def test_classifies_eight_to_thirty_day_tier():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)
    published_at = now - timedelta(days=8)

    assert classify_lifecycle_tier(published_at=published_at, now=now) == (VideoLifecycleTier.DAYS_8_TO_30)


def test_classifies_thirty_one_plus_day_tier():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)
    published_at = now - timedelta(days=31)

    assert classify_lifecycle_tier(published_at=published_at, now=now) == (VideoLifecycleTier.DAYS_31_PLUS)


def test_future_publish_date_is_treated_as_new_video():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)
    published_at = now + timedelta(days=1)

    assert classify_lifecycle_tier(published_at=published_at, now=now) == (VideoLifecycleTier.DAYS_0_TO_7)
