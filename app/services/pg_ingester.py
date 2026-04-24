from __future__ import annotations

import uuid
from typing import Any

from app.services.base import InfrastructureServiceClient


class PgIngesterClient(InfrastructureServiceClient):
    """HTTP client for pg-vector-ingester service.

    Contract:
      POST /ingest  { message_ids: [UUID, ...], options: {...} }
      POST /sync    { user_id: <str | null> }
    """

    async def trigger_ingestion(
        self,
        *,
        message_ids: list[uuid.UUID],
        force_reembed: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Embed the given messages and persist their vectors to PostgreSQL."""
        options: dict[str, Any] = {}
        if force_reembed:
            options["force_reembed"] = True
        if batch_size is not None:
            options["batch_size"] = batch_size

        return await self.post(
            "/ingest",
            {
                "message_ids": [str(mid) for mid in message_ids],
                "options": options or None,
            },
        )

    async def sync(
        self,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Incremental re-embed: find all messages with embedding IS NULL and embed them.

        If user_id is provided, scopes the sync to that user's messages only.
        If None, performs a global sync across all messages.
        """
        return await self.post("/sync", {"user_id": user_id})
