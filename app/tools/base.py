from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import httpx

EmitStatus = Callable[[str], Awaitable[None]]


class ToolExecutionError(Exception):
    def __init__(self, message: str, *, code: str = "tool_error", retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(slots=True)
class ToolContext:
    user_id: str
    state: dict[str, Any]
    emit_status: EmitStatus | None = None


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]
    layer: str


class BaseTool(Protocol):
    spec: ToolSpec

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        ...


class ExternalToolAdapter:
    spec: ToolSpec
    status_message: str
    invoke_path: str = "/invoke"

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if context.emit_status is not None:
            await context.emit_status(self.status_message)
        try:
            response = await self._client.post(
                self.invoke_path,
                json={"arguments": arguments, "user_id": context.user_id},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ToolExecutionError("External tool returned a non-object payload", code="invalid_tool_response")
            return payload
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(str(exc), code="timeout", retryable=True) from exc
        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(str(exc), code="http_error", retryable=True) from exc
        except ValueError as exc:
            raise ToolExecutionError(str(exc), code="invalid_json", retryable=False) from exc
