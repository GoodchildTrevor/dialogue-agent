from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from fastmcp import Client as MCPClient

from app.core.config import Settings
from app.graph.state import ToolExecutionResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolContext:
    """Context passed to tool invocations containing user/session info and status callback."""
    user_id: str
    state: dict[str, Any]
    emit_status: Callable[[str], Awaitable[None]] | None = None


class ToolRegistry:
    """Registry for MCP tools with invocation and description capabilities."""

    def __init__(self, settings: Settings, mcp_client: MCPClient) -> None:
        self._settings = settings
        self._mcp_client = mcp_client
        self._tools: dict[str, dict[str, Any]] = {}
        self._descriptions_cache: list[dict[str, Any]] | None = None

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def refresh_tools(self) -> None:
        """Fetch available tools from MCP server and cache them."""
        try:
            tools_list = await self._mcp_client.list_tools()
            self._tools.clear()
            for tool in tools_list:
                tool_name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                if tool_name:
                    self._tools[tool_name] = tool
            self._descriptions_cache = None
            logger.info(f"Refreshed {len(self._tools)} tools from MCP server")
        except Exception as e:
            logger.error(f"Failed to refresh tools from MCP server: {e}")
            self._tools = {}

    def describe_for_model(self) -> list[dict[str, Any]]:
        """Return tool descriptions in a format suitable for LLM consumption."""
        if self._descriptions_cache is not None:
            return self._descriptions_cache

        descriptions = []
        for name, tool_info in self._tools.items():
            if isinstance(tool_info, dict):
                description = {
                    "name": name,
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("inputSchema", tool_info.get("parameters", {})),
                }
            else:
                description = {
                    "name": name,
                    "description": getattr(tool_info, "description", ""),
                    "parameters": getattr(tool_info, "inputSchema", getattr(tool_info, "parameters", {})),
                }
            descriptions.append(description)

        self._descriptions_cache = descriptions
        return descriptions

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolExecutionResult:
        """Invoke a tool by name with the given arguments and context."""
        if tool_name not in self._tools:
            logger.warning(f"Unknown tool requested: {tool_name}")
            return {
                "tool": tool_name,
                "ok": False,
                "error": {"message": f"Unknown tool: {tool_name}"},
            }

        try:
            # Emit status update if provided
            if context.emit_status:
                await context.emit_status(f"Calling tool: {tool_name}")

            # Call the MCP tool
            result = await self._mcp_client.call_tool(tool_name, arguments=arguments)

            # Extract content from MCP result
            if isinstance(result, dict):
                content = result.get("content", result)
            else:
                # Handle object-style results
                content = getattr(result, "content", result)

            return {
                "tool": tool_name,
                "ok": True,
                "result": {"content": content} if not isinstance(content, dict) else content,
            }

        except Exception as e:
            logger.exception(f"Tool invocation failed: {tool_name}")
            return {
                "tool": tool_name,
                "ok": False,
                "error": {"message": str(e)},
            }
