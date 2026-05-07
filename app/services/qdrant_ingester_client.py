from __future__ import annotations

import httpx
from typing import Optional


class QdrantIngesterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def ingest(
        self,
        collection: str,
        file_path: str,
        chunk_size: int = 512,
        overlap: int = 1,
        extra_payload: Optional[dict] = None
    ) -> dict:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.base_url}/ingest",
                headers={
                    "X-API-Key": self.api_key,
                },
                json={
                    "collection": collection,
                    "file_path": file_path,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "extra_payload": extra_payload
                },
            )

            response.raise_for_status()
            return response.json()
        