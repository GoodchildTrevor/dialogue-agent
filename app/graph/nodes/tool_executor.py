import asyncio
import logging
from typing import Any

from app.core.tracing import trace
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_ARG_PREVIEW_LEN = 200   # max chars for argument preview in logs
_RES_PREVIEW_LEN = 300   # max chars for result preview in logs


def _truncate(value: Any, max_len: int) -> str:
    """Return a compact string representation of *value*, capped at *max_len* chars."""
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


class ToolExecutorNode():
    def __init__(self, emit_status, settings, tool_registries: list[ToolRegistry]):
        self.emit_status = emit_status
        self.tool_registries = tool_registries
        # Keep backward-compat reference to the primary registry
        self.tool_registry = tool_registries[0] if tool_registries else None
        self.settings = settings

    def _find_registry(self, tool_name: str) -> ToolRegistry | None:
        """Return the first registry that knows about this tool."""
        for registry in self.tool_registries:
            if registry.has_tool(tool_name):
                return registry
        return None

    async def action(self, state: AssistantState) -> dict[str, Any]:
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
                    # Log a short preview of the result so we can tell whether the
                    # tool returned useful data without printing the entire payload.
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
            t.output = {"results": results}

        errors = [result for result in results if not result.get("ok")]
        new_steps = state.get("intermediate_steps", []) + [{"tool_calls": tool_calls, "results": results}]
        update: dict[str, Any] = {
            "intermediate_steps": new_steps,
            "tool_results": results,
            "pending_tool_calls": [],
            "context": {**state.get("context", {}), "tool_results": results},
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
