import asyncio
import logging
import re
from typing import Any

from app.core.tracing import trace
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_ARG_PREVIEW_LEN = 200   # max chars for argument preview in logs
_RES_PREVIEW_LEN = 300   # max chars for result preview in logs


def _truncate(value: Any, max_len: int) -> str:
    """Return a compact string representation of *value*, capped at *max_len* chars.

    :param value: The value to truncate.
    :param max_len: Maximum allowed string length.
    :returns: The truncated string representation.
    """
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "\u2026"
    return text


class ToolExecutorNode():
    """Node responsible for executing tool calls in the dialogue graph.

    Invokes tools from registered registries and normalizes results,
    handling errors and collecting image payloads from tool responses.
    :param emit_status: Callback function to emit status updates.
    :param settings: Application settings containing configuration such as MAX_TOOL_RETRIES.
    :param tool_registries: List of ToolRegistry instances to look up and invoke tools.
    """

    def __init__(self, emit_status, settings, tool_registries: list[ToolRegistry]):
        self.emit_status = emit_status
        self.tool_registries = tool_registries
        self.tool_registry = tool_registries[0] if tool_registries else None
        self.settings = settings

    def _find_registry(self, tool_name: str) -> ToolRegistry | None:
        """Return the first registry that knows about this tool.

        :param tool_name: Name of the tool to look up.
        :returns: The first ToolRegistry that has the tool, or None if not found.
        """
        for registry in self.tool_registries:
            if registry.has_tool(tool_name):
                return registry
        return None

    async def action(self, state: AssistantState) -> dict[str, Any]:
        """Execute all pending tool calls and return the results.

        :param state: The current graph state containing pending tool calls.
        :returns: A dictionary with tool results, updated retry counts, images, and next action.
        """
        tool_calls = state.get("pending_tool_calls", [])
        payload = {"tool_calls": tool_calls}
        async with trace(step_name="tool_executor", user_id=state["user_id"], request_id=state["request_id"], input=payload) as t:
            tool_context = ToolContext(
                user_id=state["user_id"],
                state=state,
                emit_status=lambda s: self.emit_status(state, s),
            )

            async def _invoke_one(call: dict[str, Any]):
                tool_name = call["tool"]
                registry = self._find_registry(tool_name)
                if registry is None:
                    # Fall back to primary registry so it can return a proper "unknown tool" error
                    registry = self.tool_registries[0]
                return await registry.invoke(tool_name, call.get("arguments", {}), tool_context)

            results = await asyncio.gather(
                *(_invoke_one(call) for call in tool_calls),
                return_exceptions=True,
            )
            # Normalize exceptions into tool-like error dicts so the rest of the flow can handle them
            normalized_results: list[dict[str, Any]] = []
            for call, r in zip(tool_calls, results):
                tool_name = call.get("tool", "<unknown>")
                args_preview = _truncate(call.get("arguments", {}), _ARG_PREVIEW_LEN)
                if isinstance(r, Exception):
                    logger.error(
                        "[%s] tool_executor: tool=%s args=%s raised %s: %s",
                        state["request_id"],
                        tool_name,
                        args_preview,
                        type(r).__name__,
                        r,
                    )
                    normalized_results.append({"ok": False, "error": str(r)})
                else:
                    ok = r.get("ok")
                    result_preview = _truncate(
                        r.get("result") or r.get("error") or r, _RES_PREVIEW_LEN
                    )
                    logger.info(
                        "[%s] tool_executor: tool=%s args=%s ok=%s result_preview=%s",
                        state["request_id"],
                        tool_name,
                        args_preview,
                        ok,
                        result_preview,
                    )
                    normalized_results.append(r)
            results = normalized_results

            # Set rejection reason for metrics tracking
            errors = [result for result in results if not result.get("ok")]
            t.output = {"results": results}
            t.tool_names = [c.get("tool") for c in tool_calls]
            if errors:
                t.rejection_reason = "tool_error"

        new_step = {"tool_calls": tool_calls, "results": results}

        new_images: list[dict[str, str]] = []
        img_pattern = re.compile(r"data=['\"]([A-Za-z0-9+/=]+)['\"]")
        for r in results:
            if not r.get("ok"):
                continue
            res = r.get("result") or {}

            # If tool returned an explicit images list, use it.
            if isinstance(res, dict) and res.get("images"):
                try:
                    new_images.extend(res.get("images", []))
                    continue
                except Exception:
                    # fall through to content parsing
                    pass

            # Otherwise try to extract base64 payload from textual content.
            content = ""
            if isinstance(res, dict):
                content = res.get("content", "") or ""
            elif isinstance(res, str):
                content = res

            if isinstance(content, str) and (
                "type='image'" in content
                or 'type="image"' in content
                or "data='iVBOR" in content
                or 'data="iVBOR' in content
                or content.startswith("iVBOR")
            ):
                m = img_pattern.search(content)
                if m:
                    new_images.append({"data": m.group(1), "mime_type": "image/png"})

        logger.debug(
            "[%s] tool_executor: appending step with %d call(s), %d error(s)",
            state["request_id"],
            len(tool_calls),
            len(errors),
        )

        update: dict[str, Any] = {
            "intermediate_steps": [new_step],
            "tool_results": results,
            "pending_tool_calls": [],
            "context": state.get("context", {}),
            "images": new_images,
        }
        if errors:
            retries = state.get("tool_retry_count", 0) + 1
            update["tool_retry_count"] = retries
            update["last_tool_error"] = errors[0]["error"]
            update["next_action"] = "reasoning" if retries >= self.settings.MAX_TOOL_RETRIES else "orchestrator"
        else:
            update["tool_retry_count"] = 0
            update["last_tool_error"] = None
            update["next_action"] = "orchestrator"

        return update
