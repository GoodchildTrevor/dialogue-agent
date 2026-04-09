from __future__ import annotations

from typing import Any

from app.services.base import InfrastructureServiceClient


class PgIngesterClient(InfrastructureServiceClient):
    # TODO: replace placeholder path and payload schema with the real pg_ingester API contract.
    async def trigger_ingestion(self, *, source_id: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.post("/ingest", {"source_id": source_id, "options": options or {}})
