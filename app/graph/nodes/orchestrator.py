import json
import logging
from typing import Any
from mcp.types import TextContent
from app.core.tracing import _make_json_safe, trace
from app.graph.prompt_fragments import ORCHESTRATOR_SYSTEM_PROMPT
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolRegistry
from app.graph.utils import (
    _extract_token_estimate,
    _normalize_tool_calls,
    _parse_json_object,
    inject_history_into_prompt,
)

log = logging.getLogger(__name__)

# How many recent messages (besides the current one) to include directly in the payload
_RECENT_MESSAGES_WINDOW = 10


def _make_serializable(obj: Any) -> Any:
    """Backward-compat shim — delegates to the shared helper in tracing."""
    return _make_json_safe(obj)


class OrchestratorNode:

    def __init__(self, llm_client, settings, tool_registries: list[ToolRegistry], history_service):
        self.llm_client = llm_client
        self.tool_registries = tool_registries
        self.tool_registry = tool_registries[0] if tool_registries else None
        self.settings = settings
        self.history_service = history_service

    def _all_tool_descriptions(self) -> list[dict[str, Any]]:
        result = []
        for registry in self.tool_registries:
            result.extend(registry.describe_for_model())
        return result

    async def action(self, state: AssistantState) -> dict[str, Any]:
        messages: list[dict] = state["messages"]
        user_message = messages[-1]["content"]
        user_id = state["user_id"]

        # --- RAG-based history for system prompt ---
        matches: list[dict[str, Any]] = []
        try:
            matches = await self.history_service.search(
                query=user_message,
                user_id=user_id,
                limit=self.settings.HISTORY_SEARCH_LIMIT,
            )
        except Exception as exc:
            log.warning("history search failed, continuing without context: %s", exc)

        system_prompt = inject_history_into_prompt(
            ORCHESTRATOR_SYSTEM_PROMPT, matches, self.settings
        )

        # --- Recent conversation window passed directly to the model ---
        # Include up to _RECENT_MESSAGES_WINDOW messages BEFORE the current one so the
        # model can resolve references like "save that" or "use the result above".
        recent_messages = [
            {"role": m["role"], "content": str(m["content"])}
            for m in messages[-(  _RECENT_MESSAGES_WINDOW + 1):-1]
        ]

        payload = {
            "message": user_message,
            "recent_messages": recent_messages,
            "context": state.get("context", {}),
            "intermediate_steps": state.get("intermediate_steps", []),
            "tool_retry_count": state.get("tool_retry_count", 0),
        }

        async with trace(
            step_name="orchestrator",
            user_id=user_id,
            request_id=state["request_id"],
            input=payload,
        ) as t:
            tool_descriptions = json.dumps(
                self._all_tool_descriptions(),
                ensure_ascii=False,
            )
            system_message = (
                f"{system_prompt}\n\n"
                f"Available tools (JSON): {tool_descriptions}\n"
            )
            response = await self.llm_client.chat(
                model=self.settings.ROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": json.dumps(_make_serializable(payload), ensure_ascii=False)},
                ],
                format="json",
            )
            t.estimated_tokens = _extract_token_estimate(response)
            content = response.get("message", {}).get("content", "")
            parsed = (
                _parse_json_object(content)
                or {"action": "escalate", "task": user_message}
            )
            t.output = parsed

        action = parsed.get("action")
        tool_names = [c.get("tool") for c in parsed.get("tool_calls", [])] if action == "tools" else []
        log.info(
            "[%s] orchestrator: action=%s tools=%s retry=%d",
            state["request_id"],
            action,
            tool_names,
            state.get("tool_retry_count", 0),
        )

        if action == "respond":
            return {"final_answer": str(parsed.get("answer", "")), "next_action": "end"}
        if action == "tools":
            tool_calls = _normalize_tool_calls(parsed.get("tool_calls", []))
            if not tool_calls:
                return {"next_action": "reasoning"}
            return {"pending_tool_calls": tool_calls, "next_action": "tools"}
        return {
            "next_action": "reasoning",
            "context": {
                **state.get("context", {}),
                "escalation_task": parsed.get("task", user_message),
            },
        }
