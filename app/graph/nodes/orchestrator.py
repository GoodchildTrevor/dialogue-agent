import json
import logging
from typing import Any

from app.core.tracing import _make_json_safe, trace
from app.graph.prompt_fragments import build_orchestrator_prompt
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolRegistry
from app.graph.utils import (
    _extract_token_estimate,
    _parse_json_object,
    inject_history_into_prompt,
)

log = logging.getLogger(__name__)

_RECENT_MESSAGES_WINDOW = 10

_JSON_REMINDER = (
    "\n\nREMINDER: Your entire response MUST be valid JSON only. "
    "No prose, no markdown, no explanations outside the JSON. "
    "Do NOT use OpenAI function-calling format. "
    'Use exactly one of these formats:\n'
    '  {"action": "respond", "answer": "..."}\n'
    '  {"action": "tools", "tool_calls": [{"tool": "name", "arguments": {...}}]}\n'
    '  {"action": "escalate", "task": "..."}\n'
    'CRITICAL: Only call tools that are listed in "Available tools" above. '
    'Never invent or guess tool names. If the tool you need is not listed, '
    'use action="respond" to answer directly or action="escalate" to delegate.'
)


def _uploaded_files_reminder(uploaded_files: list[dict]) -> str:
    """Build an inline instruction block injected directly into user_content.

    - If a file has inline_text: the full document text is embedded here;
      the model MUST answer directly without calling any tool.
    - If inline_text is absent/None: the file is chunked in Qdrant;
      the model MUST call document_searcher with the file_id.
    """
    if not uploaded_files:
        return ""

    inline_files = [f for f in uploaded_files if f.get("inline_text")]
    qdrant_files = [f for f in uploaded_files if not f.get("inline_text")]

    lines = ["\n\n[UPLOADED FILES — ACTION REQUIRED]"]

    if inline_files:
        lines.append(
            "The following file(s) are SMALL enough to be provided in full. "
            "Their complete text is included below. "
            "Answer the user's question directly using this text. "
            "Do NOT call any tool."
        )
        for f in inline_files:
            lines.append(f'\n--- FILE: "{f["filename"]}" ---')
            lines.append(f["inline_text"].strip())
            lines.append(f'--- END OF FILE: "{f["filename"]}" ---')

    if qdrant_files:
        lines.append(
            "\nThe following file(s) are indexed in the vector database. "
            "You MUST call `document_searcher` for each one to answer the user's question."
        )
        for f in qdrant_files:
            lines.append(f'  - filename: "{f["filename"]}"  file_id: "{f["file_id"]}"')
        lines += [
            "Call `document_searcher` with:",
            '  "query": <the user\'s question>',
            '  "file_id": <the file_id above>',
            "Do NOT ask the user to clarify. Do NOT respond without calling document_searcher first.",
        ]

    lines.append("[END UPLOADED FILES]")
    return "\n".join(lines)


def _safe_json_loads(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return val
    return val


def _normalize_tool_calls(raw: Any) -> list[dict]:
    """Normalise tool_calls regardless of which JSON schema the LLM used."""
    raw = _safe_json_loads(raw)

    if isinstance(raw, dict):
        name = raw.get("name") or raw.get("tool")
        args = raw.get("arguments") or raw.get("args") or {}
        if isinstance(args, str):
            args = _safe_json_loads(args)
        if name:
            return [{"tool": str(name), "arguments": args if isinstance(args, dict) else {}}]
        return []

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
            result.append({"tool": item, "arguments": {}})

    return result


def _sanitize_llm_output(parsed: Any, fallback_task: str) -> dict:
    if not isinstance(parsed, dict):
        return {"action": "escalate", "task": fallback_task, "tool_calls": [], "answer": ""}

    action = parsed.get("action")

    if not isinstance(action, str):
        action = str(action) if action is not None else ""

    if action == "function":
        fn = parsed.get("function", {})
        if isinstance(fn, dict):
            tool_calls = _normalize_tool_calls(fn)
            if tool_calls:
                log.warning("LLM used OpenAI function-calling format — normalizing to tools")
                return {
                    "action": "tools",
                    "answer": "",
                    "task": fallback_task,
                    "tool_calls": tool_calls,
                }
        action = "escalate"

    if action not in {"respond", "tools", "escalate"}:
        raw_tool_calls = parsed.get("tool_calls")
        if raw_tool_calls:
            action = "tools"
        else:
            action = "escalate"

    tool_calls = _normalize_tool_calls(parsed.get("tool_calls"))

    return {
        "action": action,
        "answer": str(parsed.get("answer", "")),
        "task": parsed.get("task", fallback_task),
        "tool_calls": tool_calls,
    }


def _sanitize_tool_content(content: str, tool_name: str) -> str:
    """Strip large binary payloads (e.g. base64 images) before sending to LLM."""
    if isinstance(content, str) and (
        "type='image'" in content
        or 'type="image"' in content
        or "data='iVBOR" in content
        or 'data="iVBOR' in content
        or content.startswith("iVBOR")
    ):
        return f"[Image generated successfully by '{tool_name}'. The result has been delivered to the user.]"
    return content


def _format_intermediate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                tool_name = r.get("tool", "")
                formatted_results.append({
                    "ok": True,
                    "tool": tool_name,
                    "content": _sanitize_tool_content(content, tool_name),
                })
        formatted.append({"tool_calls": calls, "results": formatted_results})
    return formatted


def _build_unknown_tool_error_step(
    unknown_names: list[str],
    known_tools: list[str],
) -> dict[str, Any]:
    return {
        "tool_calls": [{"tool": name, "arguments": {}} for name in unknown_names],
        "results": [
            {
                "ok": False,
                "tool": name,
                "error": (
                    f"Unknown tool '{name}'. "
                    f"Available tools: {json.dumps(known_tools)}. "
                    "Retry using only a tool from this list, or use action='respond'."
                ),
            }
            for name in unknown_names
        ],
    }


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
        log.debug(
            "[%s] orchestrator uploaded_files raw from state: %s",
            state["request_id"],
            state.get("uploaded_files"),
        )
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

        orchestrator_prompt = build_orchestrator_prompt()

        system_prompt = inject_history_into_prompt(
            orchestrator_prompt,
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

        tool_descriptions = self._all_tool_descriptions()
        tool_names = [t["name"] for t in tool_descriptions if isinstance(t, dict) and "name" in t]

        uploaded_files = state.get("uploaded_files") or []

        payload = {
            "message": user_message,
            "recent_messages": recent_messages,
            "context": state.get("context", {}),
            "intermediate_steps": formatted_steps,
            "last_tool_results": last_tool_results,
            "tool_retry_count": state.get("tool_retry_count", 0),
            "uploaded_files": [
                # strip inline_text from payload JSON to avoid bloating it twice
                # (the text is already injected via files_reminder above the payload)
                {k: v for k, v in f.items() if k != "inline_text"}
                for f in uploaded_files
            ],
        }

        log.debug(
            "[%s] orchestrator payload intermediate_steps=%d last_tool_results=%d "
            "uploaded_files=%d (inline=%d qdrant=%d) file_ids=%s",
            state["request_id"],
            len(formatted_steps),
            len(last_tool_results),
            len(uploaded_files),
            sum(1 for f in uploaded_files if f.get("inline_text")),
            sum(1 for f in uploaded_files if not f.get("inline_text")),
            [f.get("file_id") for f in uploaded_files],
        )

        async with trace(
            step_name="orchestrator",
            user_id=user_id,
            request_id=state["request_id"],
            input=payload,
        ) as t:

            available_tools_line = (
                "CRITICAL: AVAILABLE TOOLS — use ONLY these exact names, nothing else:\n"
                + "\n".join(f"  - {n}" for n in tool_names)
                + "\n\n"
            )
            system_message = (
                available_tools_line
                + f"{system_prompt}\n\n"
                + "Available tools (full schema): "
                + json.dumps(tool_descriptions, ensure_ascii=False)
            )

            files_reminder = _uploaded_files_reminder(uploaded_files)
            user_content = (
                files_reminder
                + "\n\n"
                + json.dumps(_make_json_safe(payload), ensure_ascii=False)
                + _JSON_REMINDER
            )

            try:
                response = await self.llm_client.chat(
                    model=self.settings.ROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content},
                    ],
                    format="json",
                    timeout=self.settings.ORCHESTRATOR_TIMEOUT_SECONDS,
                )
            except Exception as e:
                log.exception("LLM call failed")
                return self._fallback(state, user_message)

            t.estimated_tokens = _extract_token_estimate(response)

            content = response.get("message", {}).get("content", "")
            log.debug("[%s] orchestrator raw LLM response: %.500s", state["request_id"], content)

            parsed_raw = _parse_json_object(content)
            parsed = _sanitize_llm_output(parsed_raw, user_message)

            known_tools_set = set(tool_names)
            valid_calls = [tc for tc in parsed["tool_calls"] if tc["tool"] in known_tools_set]
            unknown_calls = [tc["tool"] for tc in parsed["tool_calls"] if tc["tool"] not in known_tools_set]

            if unknown_calls:
                log.warning("[%s] Dropping unknown tool calls: %s", state["request_id"], unknown_calls)

            parsed["tool_calls"] = valid_calls

            if parsed["action"] == "tools" and not valid_calls and unknown_calls:
                log.warning(
                    "[%s] All tool calls unknown — injecting error feedback into steps (unknown=%s)",
                    state["request_id"],
                    unknown_calls,
                )
                error_step = _build_unknown_tool_error_step(unknown_calls, tool_names)
                t.output = {"action": "tool_name_error", "unknown": unknown_calls}
                return {
                    "intermediate_steps": [error_step],
                    "next_action": "orchestrator",
                }

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
                "next_action": "end",
            }

        if parsed["action"] == "tools":
            return {
                "pending_tool_calls": parsed["tool_calls"],
                "next_action": "tools",
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
