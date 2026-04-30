from datetime import datetime, timezone

from src.youtube_integration.scheduler import analytics_window_for_tier
from src.youtube_integration.sync_models import VideoLifecycleTier


def test_zero_to_seven_day_window_uses_last_complete_seven_days():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    window = analytics_window_for_tier(
        tier=VideoLifecycleTier.DAYS_0_TO_7,
        now=now,
    )

    assert str(window.start_date) == "2026-04-23"
    assert str(window.end_date) == "2026-04-29"


def test_eight_to_thirty_day_window_uses_last_complete_fourteen_days():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    window = analytics_window_for_tier(
        tier=VideoLifecycleTier.DAYS_8_TO_30,
        now=now,
    )

    assert str(window.start_date) == "2026-04-16"
    assert str(window.end_date) == "2026-04-29"


def test_thirty_one_plus_window_uses_last_complete_seven_days():
    now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

    window = analytics_window_for_tier(
        tier=VideoLifecycleTier.DAYS_31_PLUS,
        now=now,
    )

    assert str(window.start_date) == "2026-04-23"
    assert str(window.end_date) == "2026-04-29"
