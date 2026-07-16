from __future__ import annotations

import asyncio
import logging
import re
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


def _normalize_content(content: Any) -> tuple[str, list[dict[str, str]]]:
    """Normalise MCP tool output to a plain string and separate images."""
    text_parts: list[str] = []
    images: list[dict[str, str]] = []

    def _process_item(item: Any) -> None:
        if isinstance(item, str):
            text_parts.append(item)
        elif isinstance(item, dict):
            if item.get("type") == "image":
                images.append({
                    "mime_type": item.get("mimeType", "image/png"),
                    "data": item.get("data", "")
                })
                text_parts.append(f"[IMAGE:{len(images)}]")
            else:
                text_parts.append(item.get("text", str(item)))
        else:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(str(text))
            else:
                text_parts.append(str(item))

    if isinstance(content, str):
        if re.search(r"type\s*=?\s*['\"]image['\"]", content, re.IGNORECASE):
            data_matches = list(re.finditer(r"data\s*=\s*['\"]([A-Za-z0-9+/=]+)['\"]", content))
            if data_matches:
                images = [{"mime_type": "image/png", "data": m.group(1)} for m in data_matches]
                text_content = re.sub(r"data\s*=\s*['\"][A-Za-z0-9+/=]+['\"]", "[IMAGE DATA]", content)
                return text_content, images
        return content, []

    if isinstance(content, list):
        for item in content:
            _process_item(item)
        return "\n".join(text_parts), images

    if isinstance(content, dict):
        _process_item(content)
        return "\n".join(text_parts), images

    _process_item(content)
    return "\n".join(text_parts), images


class ToolRegistry:
    """Registry for MCP tools with invocation and description capabilities.

    Uses fastmcp.Client as a per-call context manager because
    StreamableHttpTransport is stateless — every list_tools / call_tool
    opens its own HTTP session and closes it when done.

    :param settings: Application configuration containing MCP server URL and timeouts.
    :param mcp_client: An MCPClient instance used to list and invoke remote tools.
    :param name: Human-readable label for this registry, used in log messages.
    """

    def __init__(self, settings: Settings, mcp_client: MCPClient, name: str = "default") -> None:
        self._settings = settings
        self._mcp_client = mcp_client
        self._name = name
        self._tools: dict[str, dict[str, Any]] = {}
        self._descriptions_cache: list[dict[str, Any]] | None = None
        self._refresh_lock = asyncio.Lock()

    async def startup(self) -> None:
        """Warm up the tool registry by fetching the tool list from MCP server."""
        await self.refresh_tools()

    async def shutdown(self) -> None:
        """No-op: per-call transport requires no persistent teardown."""

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def is_empty(self) -> bool:
        return len(self._tools) == 0

    async def refresh_tools(self) -> None:
        """Fetch available tools from MCP server and cache them."""
        async with self._refresh_lock:
            if not self.is_empty:
                return
            try:
                async with self._mcp_client as client:
                    tools_list = await client.list_tools()
                new_tools = {}
                for tool in tools_list:
                    tool_name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
                    if tool_name:
                        new_tools[tool_name] = tool
                self._tools = new_tools
                self._descriptions_cache = None
                logger.info("Refreshed %d tools from MCP server '%s'", len(self._tools), self._name)
            except Exception as e:
                logger.error("Failed to refresh tools from MCP server '%s': %s", self._name, e)

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
            logger.warning("Unknown tool requested: %s", tool_name)
            return {
                "tool": tool_name,
                "ok": False,
                "error": {"message": f"Unknown tool: {tool_name}"},
            }

        try:
            if context.emit_status:
                await context.emit_status(f"Calling tool: {tool_name}")

            logger.debug("Invoking tool %s with arguments: %r", tool_name, arguments)

            async with self._mcp_client as client:
                raw = await client.call_tool(tool_name, arguments=arguments)

            if isinstance(raw, dict):
                raw_content = raw.get("content", raw)
            else:
                raw_content = getattr(raw, "content", raw)

            text_content, image_data = _normalize_content(raw_content)

            logger.debug("Tool %s result (preview): %.300s", tool_name, text_content)

            return {
                "tool": tool_name,
                "ok": True,
                "result": {
                    "content": text_content,
                    "images": image_data,
                },
            }

        except Exception as e:
            logger.exception("Tool invocation failed: %s", tool_name)

            raw_err = str(e)
            lower = raw_err.lower()
            if "invalid utf-8" in lower or "invalid utf8" in lower or "error serializ" in lower or "cannot encode" in lower:
                user_message = (
                    "Tool returned non-UTF8/binary data which could not be JSON-serialized. "
                    "Binary outputs (images) must be encoded as text (for example base64) by the MCP tool. "
                    "Example: {\"type\": \"image\", \"mimeType\": \"image/png\", \"data\": \"<BASE64_STRING>\"}. "
                    f"Original error: {raw_err}"
                )
            else:
                user_message = raw_err

            return {
                "tool": tool_name,
                "ok": False,
                "error": {"message": user_message, "detail": raw_err},
            }
