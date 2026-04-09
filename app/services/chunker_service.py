from __future__ import annotations

from typing import Any

from app.services.base import InfrastructureServiceClient


class ChunkerServiceClient(InfrastructureServiceClient):
    # TODO: replace placeholder path and payload schema with the real chunker_service API contract.
    async def parse_and_chunk(self, *, file_url: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.post("/chunk", {"file_url": file_url, "metadata": metadata or {}})
