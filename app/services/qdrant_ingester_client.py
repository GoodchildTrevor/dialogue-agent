from __future__ import annotations

import logging

import httpx
from typing import Optional

logger = logging.getLogger(__name__)


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
        extra_payload: Optional[dict] = None,
        inline_threshold: Optional[int] = None,
    ) -> dict:
        payload = {
            "collection": collection,
            "file_path": file_path,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "extra_payload": extra_payload,
        }
        if inline_threshold is not None:
            payload["inline_threshold"] = inline_threshold

        logger.debug("QdrantIngesterClient.ingest payload: %s", payload)

        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.base_url}/ingest",
                headers={"X-API-Key": self.api_key},
                json=payload,
            )
            if not response.is_success:
                logger.error(
                    "QdrantIngesterClient.ingest error %s: %s",
                    response.status_code,
                    response.text,
                )
            response.raise_for_status()
            return response.json()
