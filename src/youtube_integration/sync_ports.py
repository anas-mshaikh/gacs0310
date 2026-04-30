from __future__ import annotations

from typing import Protocol

from .sync_models import YouTubeVideoSyncCandidate, YouTubeSyncTask


class YouTubeSyncRepository(Protocol):
    """Database access required by the scheduler."""

    def list_sync_candidates(self, *, limit: int | None = None) -> list[YouTubeVideoSyncCandidate]: ...


class YouTubeSyncExecutor(Protocol):
    """Execution boundary for future API-backed sync clients.

    PR3 only defines the contract. Actual Data API / Analytics API clients plug in later.
    """

    def execute_task(self, task: YouTubeSyncTask) -> None: ...
