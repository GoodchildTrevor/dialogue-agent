import asyncio
import logging
from typing import Any

from app.core.tracing import trace
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolContext

logger = logging.getLogger(__name__)


class ToolExecutorNode():    
    async def tool_executor_node(self, state: AssistantState) -> dict[str, Any]:
        tool_calls = state.get("pending_tool_calls", [])
        payload = {"tool_calls": tool_calls}
        async with trace(step_name="tool_executor", user_id=state["user_id"], request_id=state["request_id"], input=payload) as t:
            tool_context = ToolContext(
                user_id=state["user_id"],
                state=state,
                emit_status=lambda s: self.emit_status(state, s),
            )
            results = await asyncio.gather(
                *(
                    self.tool_registry.invoke(call["tool"], call.get("arguments", {}), tool_context)
                    for call in tool_calls
                ),
                return_exceptions=True,
            )
            # Normalize exceptions into tool-like error dicts so the rest of the flow can handle them
            normalized_results: list[dict[str, Any]] = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Tool invocation raised: {r}")
                    normalized_results.append({"ok": False, "error": str(r)})
                else:
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
            return update
        update["last_tool_error"] = None
        update["next_action"] = "orchestrator"
        return update
    