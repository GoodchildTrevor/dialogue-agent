from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.OLLAMA_BASE_URL.rstrip("/"),
            timeout=settings.TOOL_REQUEST_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=settings.HTTP_MAX_CONNECTIONS),
        )

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
        format: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options:
            payload["options"] = options
        if format:
            payload["format"] = format
        response = await self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def embeddings(self, *, model: str, prompt: str) -> list[float]:
        response = await self._client.post(
            "/api/embeddings",
            json={"model": model, "prompt": prompt},
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("embedding", [])

    async def aclose(self) -> None:
        await self._client.aclose()
