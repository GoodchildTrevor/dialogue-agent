import json
import logging
from typing import Any

from app.core.tracing import _make_json_safe, trace
from app.graph.prompt_fragments import ORCHESTRATOR_SYSTEM_PROMPT
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolRegistry
from app.graph.utils import (
    _extract_token_estimate,
    _parse_json_object,
    inject_history_into_prompt,
)

log = logging.getLogger(__name__)

_RECENT_MESSAGES_WINDOW = 10


def _safe_json_loads(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def _normalize_tool_calls(raw: Any) -> list[dict]:
    raw = _safe_json_loads(raw)

    if not isinstance(raw, list):
        return []

    result = []
    for item in raw:
        if isinstance(item, dict):
            tool = item.get("tool") or item.get("name")
            args = item.get("args") or item.get("arguments") or {}

            if isinstance(args, str):
                args = _safe_json_loads(args)
                if not isinstance(args, dict):
                    args = {}

            if tool:
                result.append({
                    "tool": str(tool),
                    "arguments": args if isinstance(args, dict) else {}
                })

        elif isinstance(item, str):
            result.append({
                "tool": item,
                "arguments": {}
            })

    return result


def _sanitize_llm_output(parsed: Any, fallback_task: str) -> dict:
    if not isinstance(parsed, dict):
        return {"action": "escalate", "task": fallback_task, "tool_calls": []}

    action = parsed.get("action")

    # Guard against LLM returning action as a dict or other non-string type
    if not isinstance(action, str):
        action = str(action) if action is not None else ""

    if action not in {"respond", "tools", "escalate"}:
        action = "escalate"

    tool_calls = _normalize_tool_calls(parsed.get("tool_calls"))

    return {
        "action": action,
        "answer": str(parsed.get("answer", "")),
        "task": parsed.get("task", fallback_task),
        "tool_calls": tool_calls,
    }


def _format_intermediate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reformat intermediate_steps so LLM receives plain-string tool results."""
    formatted = []
    for step in steps:
        calls = step.get("tool_calls", [])
        results = step.get("results", [])
        formatted_results = []
        for r in results:
            if not isinstance(r, dict):
                formatted_results.append({"ok": False, "content": str(r)})
                continue
            if not r.get("ok", False):
                error = r.get("error") or {}
                msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                formatted_results.append({"ok": False, "error": msg})
            else:
                raw_result = r.get("result", {})
                if isinstance(raw_result, dict):
                    content = raw_result.get("content", "")
                else:
                    content = str(raw_result)
                if not isinstance(content, str):
                    content = json.dumps(_make_json_safe(content), ensure_ascii=False)
                formatted_results.append({"ok": True, "tool": r.get("tool", ""), "content": content})
        formatted.append({"tool_calls": calls, "results": formatted_results})
    return formatted


class OrchestratorNode:

    def __init__(self, llm_client, settings, tool_registries: list[ToolRegistry], history_service):
        self.llm_client = llm_client
        self.tool_registries = tool_registries
        self.settings = settings
        self.history_service = history_service

    def _all_tool_descriptions(self) -> list[dict[str, Any]]:
        result = []
        for registry in self.tool_registries:
            result.extend(registry.describe_for_model())
        return result

    async def action(self, state: AssistantState) -> dict[str, Any]:
        messages = state["messages"]
        user_message = messages[-1]["content"]
        user_id = state["user_id"]

        try:
            matches = await self.history_service.search(
                query=user_message,
                user_id=user_id,
                limit=self.settings.HISTORY_SEARCH_LIMIT,
            )
        except Exception as exc:
            log.warning("history search failed: %s", exc)
            matches = []

        system_prompt = inject_history_into_prompt(
            ORCHESTRATOR_SYSTEM_PROMPT,
            matches,
            self.settings
        )

        recent_messages = [
            {"role": m["role"], "content": str(m["content"])}
            for m in messages[-(_RECENT_MESSAGES_WINDOW + 1):-1]
        ]

        raw_steps = state.get("intermediate_steps", [])
        formatted_steps = _format_intermediate_steps(raw_steps)

        last_tool_results: list[dict] = []
        if formatted_steps:
            last_tool_results = formatted_steps[-1].get("results", [])

        payload = {
            "message": user_message,
            "recent_messages": recent_messages,
            "context": state.get("context", {}),
            "intermediate_steps": formatted_steps,
            "last_tool_results": last_tool_results,
            "tool_retry_count": state.get("tool_retry_count", 0),
        }

        log.debug(
            "[%s] orchestrator payload intermediate_steps=%d last_tool_results=%d",
            state["request_id"],
            len(formatted_steps),
            len(last_tool_results),
        )

        async with trace(
            step_name="orchestrator",
            user_id=user_id,
            request_id=state["request_id"],
            input=payload,
        ) as t:

            system_message = (
                f"{system_prompt}\n\n"
                f"Available tools (JSON): "
                f"{json.dumps(self._all_tool_descriptions(), ensure_ascii=False)}"
            )

            try:
                response = await self.llm_client.chat(
                    model=self.settings.ROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": json.dumps(_make_json_safe(payload), ensure_ascii=False)},
                    ],
                    format="json",
                )
            except Exception as e:
                log.exception("LLM call failed")
                return self._fallback(state, user_message)

            t.estimated_tokens = _extract_token_estimate(response)

            content = response.get("message", {}).get("content", "")

            log.debug("[%s] orchestrator raw LLM response: %.500s", state["request_id"], content)

            parsed_raw = _parse_json_object(content)
            parsed = _sanitize_llm_output(parsed_raw, user_message)

            t.output = parsed

        log.info(
            "[%s] action=%s tools=%s",
            state["request_id"],
            parsed["action"],
            [tc["tool"] for tc in parsed["tool_calls"]],
        )

        if parsed["action"] == "respond":
            return {
                "final_answer": parsed["answer"],
                "next_action": "end"
            }

        if parsed["action"] == "tools":
            if not parsed["tool_calls"]:
                log.warning("[%s] action=tools but tool_calls is empty, escalating", state["request_id"])
                return {"next_action": "reasoning"}

            return {
                "pending_tool_calls": parsed["tool_calls"],
                "next_action": "tools"
            }

        return self._fallback(state, parsed["task"])

    def _fallback(self, state: AssistantState, task: str) -> dict[str, Any]:
        return {
            "next_action": "reasoning",
            "context": {
                **state.get("context", {}),
                "escalation_task": task,
            },
        }
