from __future__ import annotations

import uuid
from typing import Any

from app.services.base import InfrastructureServiceClient


class PgIngesterClient(InfrastructureServiceClient):
    """HTTP client for pg-vector-ingester service.

    Matches the /ingest contract of pg-vector-ingester:
      POST /ingest  { source_id: <file_id UUID>, options: {...} }
      POST /sync    { file_id: <UUID | null> }
    """

    async def trigger_ingestion(
        self,
        *,
        source_id: uuid.UUID,
        force_reembed: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Embed all messages for the given file and persist vectors to PostgreSQL."""
        options: dict[str, Any] = {}
        if force_reembed:
            options["force_reembed"] = True
        if batch_size is not None:
            options["batch_size"] = batch_size

        return await self.post("/ingest", {"source_id": str(source_id), "options": options})

    async def sync(
        self,
        *,
        file_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Incremental re-embed: find all messages with embedding IS NULL and embed them.

        If file_id is None, performs a global sync across all messages.
        """
        return await self.post("/sync", {"file_id": str(file_id) if file_id else None})
