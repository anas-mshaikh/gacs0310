"""add youtube analytics tables

Revision ID: 8a94738aa003
Revises:
Create Date: 2026-04-28 16:42:06.401808

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8a94738aa003"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "youtube_videos",
        sa.Column(
            "youtube_record_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("video_jobs.job_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "book_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("books.book_id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("youtube_video_id", sa.String(length=32), nullable=False),
        sa.Column("youtube_channel_id", sa.String(length=128), nullable=True),
        sa.Column(
            "upload_status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'uploaded'"),
        ),
        sa.Column("privacy_status", sa.String(length=32), nullable=True),
        sa.Column("title_snapshot", sa.Text(), nullable=True),
        sa.Column("description_snapshot", sa.Text(), nullable=True),
        sa.Column("thumbnail_url", sa.Text(), nullable=True),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("uploaded_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_data_api_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_analytics_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "youtube_video_id",
            name="uq_youtube_videos_youtube_video_id",
        ),
    )

    op.create_index(
        "idx_youtube_videos_job_id",
        "youtube_videos",
        ["job_id"],
    )
    op.create_index(
        "idx_youtube_videos_book_id",
        "youtube_videos",
        ["book_id"],
    )
    op.create_index(
        "idx_youtube_videos_published_at",
        "youtube_videos",
        ["published_at"],
    )
    op.create_index(
        "idx_youtube_videos_last_data_api_sync_at",
        "youtube_videos",
        ["last_data_api_sync_at"],
    )
    op.create_index(
        "idx_youtube_videos_last_analytics_sync_at",
        "youtube_videos",
        ["last_analytics_sync_at"],
    )

    op.create_table(
        "youtube_video_stats_current",
        sa.Column(
            "youtube_video_id",
            sa.String(length=32),
            sa.ForeignKey("youtube_videos.youtube_video_id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "view_count",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("like_count", sa.BigInteger(), nullable=True),
        sa.Column("comment_count", sa.BigInteger(), nullable=True),
        sa.Column("favorite_count", sa.BigInteger(), nullable=True),
        sa.Column("duration_iso8601", sa.String(length=64), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("privacy_status", sa.String(length=32), nullable=True),
        sa.Column("upload_status", sa.String(length=32), nullable=True),
        sa.Column("live_broadcast_content", sa.String(length=32), nullable=True),
        sa.Column("etag", sa.Text(), nullable=True),
        sa.Column("raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index(
        "idx_youtube_video_stats_current_fetched_at",
        "youtube_video_stats_current",
        ["fetched_at"],
    )

    op.create_table(
        "youtube_video_daily_metrics",
        sa.Column(
            "metric_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "youtube_video_id",
            sa.String(length=32),
            sa.ForeignKey("youtube_videos.youtube_video_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric_date", sa.Date(), nullable=False),
        sa.Column(
            "views",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "estimated_minutes_watched",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("average_view_duration_seconds", sa.Numeric(12, 4), nullable=True),
        sa.Column("average_view_percentage", sa.Numeric(8, 4), nullable=True),
        sa.Column("likes", sa.BigInteger(), nullable=True),
        sa.Column("comments", sa.BigInteger(), nullable=True),
        sa.Column("shares", sa.BigInteger(), nullable=True),
        sa.Column("subscribers_gained", sa.BigInteger(), nullable=True),
        sa.Column("subscribers_lost", sa.BigInteger(), nullable=True),
        sa.Column("video_thumbnail_impressions", sa.BigInteger(), nullable=True),
        sa.Column("video_thumbnail_impressions_click_rate", sa.Numeric(10, 6), nullable=True),
        sa.Column("raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "youtube_video_id",
            "metric_date",
            name="uq_youtube_video_daily_metrics_video_date",
        ),
    )

    op.create_index(
        "idx_youtube_video_daily_metrics_video_date",
        "youtube_video_daily_metrics",
        ["youtube_video_id", sa.text("metric_date DESC")],
    )
    op.create_index(
        "idx_youtube_video_daily_metrics_date",
        "youtube_video_daily_metrics",
        ["metric_date"],
    )

    op.create_table(
        "youtube_traffic_source_daily",
        sa.Column(
            "traffic_source_metric_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "youtube_video_id",
            sa.String(length=32),
            sa.ForeignKey("youtube_videos.youtube_video_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("metric_date", sa.Date(), nullable=False),
        sa.Column("traffic_source_type", sa.String(length=64), nullable=False),
        sa.Column("traffic_source_detail", sa.Text(), nullable=True),
        sa.Column(
            "views",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "estimated_minutes_watched",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "youtube_video_id",
            "metric_date",
            "traffic_source_type",
            "traffic_source_detail",
            name="uq_youtube_traffic_source_daily_identity",
        ),
    )

    op.create_index(
        "idx_youtube_traffic_source_daily_video_date",
        "youtube_traffic_source_daily",
        ["youtube_video_id", sa.text("metric_date DESC")],
    )
    op.create_index(
        "idx_youtube_traffic_source_daily_type",
        "youtube_traffic_source_daily",
        ["traffic_source_type"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_youtube_traffic_source_daily_type",
        table_name="youtube_traffic_source_daily",
    )
    op.drop_index(
        "idx_youtube_traffic_source_daily_video_date",
        table_name="youtube_traffic_source_daily",
    )
    op.drop_table("youtube_traffic_source_daily")

    op.drop_index(
        "idx_youtube_video_daily_metrics_date",
        table_name="youtube_video_daily_metrics",
    )
    op.drop_index(
        "idx_youtube_video_daily_metrics_video_date",
        table_name="youtube_video_daily_metrics",
    )
    op.drop_table("youtube_video_daily_metrics")

    op.drop_index(
        "idx_youtube_video_stats_current_fetched_at",
        table_name="youtube_video_stats_current",
    )
    op.drop_table("youtube_video_stats_current")

    op.drop_index(
        "idx_youtube_videos_last_analytics_sync_at",
        table_name="youtube_videos",
    )
    op.drop_index(
        "idx_youtube_videos_last_data_api_sync_at",
        table_name="youtube_videos",
    )
    op.drop_index(
        "idx_youtube_videos_published_at",
        table_name="youtube_videos",
    )
    op.drop_index(
        "idx_youtube_videos_book_id",
        table_name="youtube_videos",
    )
    op.drop_index(
        "idx_youtube_videos_job_id",
        table_name="youtube_videos",
    )
    op.drop_table("youtube_videos")
