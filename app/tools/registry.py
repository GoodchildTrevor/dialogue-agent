from __future__ import annotations

from typing import Any, Iterable

from app.tools.base import BaseTool, ToolContext, ToolExecutionError


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools}

    def describe_for_model(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "args_schema": tool.spec.args_schema,
                "layer": tool.spec.layer,
            }
            for tool in self._tools.values()
        ]

    async def invoke(self, name: str, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "code": "unknown_tool",
                    "message": f"Tool '{name}' is not registered.",
                    "retryable": False,
                },
            }
        try:
            result = await tool.invoke(arguments, context)
            return {"tool": name, "ok": True, "result": result}
        except ToolExecutionError as exc:
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": exc.retryable,
                },
            }
        except Exception as exc:
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "code": "unexpected_tool_error",
                    "message": str(exc),
                    "retryable": True,
                },
            }
