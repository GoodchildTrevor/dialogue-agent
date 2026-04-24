import json
import logging
from typing import Any

from app.core.tracing import trace
from app.graph.prompt_fragments import ORCHESTRATOR_SYSTEM_PROMPT
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolContext
from app.graph.utils import (
    _extract_token_estimate,
    _normalize_tool_calls,
    _parse_json_object,
)
from app.graph.utils import inject_history_into_prompt

log = logging.getLogger(__name__)


class OrchestratorNode:

    def __init__(self, llm_client, settings, tool_registry):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.settings = settings

    async def action(self, state: AssistantState) -> dict[str, Any]:
        user_message = state["messages"][-1]["content"]
        user_id = state["user_id"]

        matches: list[dict[str, Any]] = []
        if self.tool_registry and self.tool_registry.has_tool("search_history"):
            try:
                ctx = ToolContext(user_id=user_id, state=dict(state))
                result = await self.tool_registry.invoke(
                    "search_history",
                    arguments={
                        "query": user_message,
                        "user_id": user_id,
                        "limit": self.settings.HISTORY_SEARCH_LIMIT,
                    },
                    context=ctx,
                )
                if result.get("ok"):
                    content = result.get("result", {})
                    matches = content.get("matches", [])
            except Exception as exc:
                log.warning(
                    "search_history unavailable, skipping RAG loop: %s", exc
                )

        system_prompt = inject_history_into_prompt(ORCHESTRATOR_SYSTEM_PROMPT, matches, self.settings)

        payload = {
            "message": user_message,
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
                self.tool_registry.describe_for_model() if self.tool_registry else [],
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
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                format="json",
            )
            t.estimated_tokens = _extract_token_estimate(response)
            content = response.get("message", {}).get("content", "")
            parsed = _parse_json_object(content) or {"action": "escalate", "task": user_message}
            t.output = parsed

        action = parsed.get("action")
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
    