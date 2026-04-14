from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class LLMClient:
    """OpenAI-compatible HTTP client targeting LiteLLM proxy."""

    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.LLM_BASE_URL.rstrip("/"),
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
        # Ollama used "format": "json"; OpenAI-compatible uses "response_format"
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        if options:
            payload["extra_body"] = {"options": options}

        response = await self._client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        raw = response.json()

        # Normalise to ollama-like shape so nodes.py needs no changes
        return {
            "message": {"content": raw["choices"][0]["message"]["content"]},
            "prompt_eval_count": raw.get("usage", {}).get("prompt_tokens"),
            "eval_count": raw.get("usage", {}).get("completion_tokens"),
        }

    async def embeddings(self, *, model: str, prompt: str) -> list[float]:
        response = await self._client.post(
            "/v1/embeddings",
            json={"model": model, "input": prompt},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def aclose(self) -> None:
        await self._client.aclose()
