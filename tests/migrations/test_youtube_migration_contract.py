from pathlib import Path


def test_youtube_migration_revision_exists() -> None:
    versions_dir = Path("migrations/versions")
    migration_files = list(versions_dir.glob("*add_youtube_analytics_tables.py"))

    assert migration_files, "Expected Alembic migration for YouTube analytics tables"


def test_youtube_migration_contains_required_tables() -> None:
    migration_file = next(Path("migrations/versions").glob("*add_youtube_analytics_tables.py"))
    content = migration_file.read_text()

    required_tables = [
        "youtube_videos",
        "youtube_video_stats_current",
        "youtube_video_daily_metrics",
        "youtube_traffic_source_daily",
    ]

    for table_name in required_tables:
        assert table_name in content


def test_youtube_migration_links_to_gacs_core_tables() -> None:
    migration_file = next(Path("migrations/versions").glob("*add_youtube_analytics_tables.py"))
    content = migration_file.read_text()

    assert "video_jobs.job_id" in content
    assert "books.book_id" in content


def test_youtube_migration_has_idempotency_constraints() -> None:
    migration_file = next(Path("migrations/versions").glob("*add_youtube_analytics_tables.py"))
    content = migration_file.read_text()

    assert "uq_youtube_videos_youtube_video_id" in content
    assert "uq_youtube_video_daily_metrics_video_date" in content
    assert "uq_youtube_traffic_source_daily_identity" in content


def test_youtube_migration_has_reversible_downgrade() -> None:
    migration_file = next(Path("migrations/versions").glob("*add_youtube_analytics_tables.py"))
    content = migration_file.read_text()

    assert "def downgrade()" in content
    assert 'op.drop_table("youtube_traffic_source_daily")' in content
    assert 'op.drop_table("youtube_video_daily_metrics")' in content
    assert 'op.drop_table("youtube_video_stats_current")' in content
    assert 'op.drop_table("youtube_videos")' in content
