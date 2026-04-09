from __future__ import annotations

import asyncio
from typing import Any, NotRequired, TypedDict


class ToolCall(TypedDict):
    tool: str
    arguments: dict[str, Any]


class ToolExecutionResult(TypedDict):
    tool: str
    ok: bool
    result: NotRequired[dict[str, Any]]
    error: NotRequired[dict[str, Any]]


class AssistantState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_id: str
    context: dict[str, Any]
    intermediate_steps: list[dict[str, Any]]
    is_complex_task: bool
    next_action: str
    final_answer: str
    pending_tool_calls: list[ToolCall]
    tool_results: list[ToolExecutionResult]
    last_tool_error: dict[str, Any] | None
    tool_retry_count: int
    status_queue: asyncio.Queue[str] | None
