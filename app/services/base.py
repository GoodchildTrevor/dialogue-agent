from __future__ import annotations

from typing import Any

import httpx


class InfrastructureServiceClient:
    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    async def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(path, json=payload)
        response.raise_for_status()
        return response.json()
