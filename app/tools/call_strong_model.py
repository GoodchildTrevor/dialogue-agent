from __future__ import annotations

from typing import Any, Awaitable, Callable

from app.tools.base import ToolContext, ToolSpec


class CallStrongModelTool:
    spec = ToolSpec(
        name="call_strong_model",
        description="Delegate the task to the stronger reasoning model for coding or complex analytical work.",
        args_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string"},
            },
            "required": ["task"],
        },
        layer="internal",
    )

    def __init__(self, callback: Callable[[str, dict[str, Any]], Awaitable[str]]) -> None:
        self._callback = callback

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if context.emit_status is not None:
            await context.emit_status("Calling reasoning model...")
        task = str(arguments.get("task", "")).strip()
        answer = await self._callback(task, context.state)
        return {"answer": answer}
