from __future__ import annotations

from typing import Any

import httpx

from app.core.config import Settings


class LLMClient:
    """OpenAI-compatible HTTP client targeting LiteLLM proxy.
    :param settings: Application settings containing LiteLLM configuration.
    """
    
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        headers = {}
        master_key = getattr(settings, "LITELLM_MASTER_KEY", None)
        if master_key:
            headers["Authorization"] = f"Bearer {master_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.LLM_BASE_URL.rstrip("/"),
            headers=headers,
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
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request to the LLM.

        :param model: The model identifier to use for chat completion.
        :param messages: A list of message dicts with 'role' and 'content' keys.
        :param stream: Whether to stream the response. Defaults to False.
        :param options: Optional extra options dict passed to the request.
        :param format: Response format specification (e.g., 'json' for JSON mode).
        :param timeout: Request timeout in seconds. Uses default if None.
        :return: A dict with keys 'message', 'prompt_eval_count', and 'eval_count'.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        if options:
            payload["extra_body"] = {"options": options}

        request_timeout = timeout or self._settings.TOOL_REQUEST_TIMEOUT_SECONDS

        response = await self._client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=request_timeout,
        )
        response.raise_for_status()
        raw = response.json()

        return {
            "message": {"content": raw["choices"][0]["message"]["content"]},
            "prompt_eval_count": raw.get("usage", {}).get("prompt_tokens"),
            "eval_count": raw.get("usage", {}).get("completion_tokens"),
        }

    async def embeddings(self, *, model: str, prompt: str) -> list[float]:
        """Generate embeddings for the given prompt.

        :param model: The model identifier to use for embeddings.
        :param prompt: The text input to generate embeddings for.
        :return: A list of float embedding values.
        """
        response = await self._client.post(
            "/v1/embeddings",
            json={"model": model, "input": prompt},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def aclose(self) -> None:
        """Close the underlying HTTP client connection.

        :return: None
        """
        await self._client.aclose()