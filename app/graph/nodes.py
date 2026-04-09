from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.ollama import OllamaClient
from app.core.tracing import trace
from app.graph.edges import after_orchestrator, after_router, after_tools
from app.graph.state import AssistantState, ToolCall
from app.services.chunker_service import ChunkerServiceClient
from app.services.pg_ingester import PgIngesterClient
from app.tools.base import ToolContext
from app.tools.call_strong_model import CallStrongModelTool
from app.tools.document_searcher import DocumentSearcherTool
from app.tools.file_converter import FileConverterTool
from app.tools.file_viewer import FileViewerTool
from app.tools.image_generator import ImageGeneratorTool
from app.tools.registry import ToolRegistry
from app.tools.search_history import SearchHistoryTool
from app.tools.web_searcher import WebSearcherTool

ORCHESTRATOR_SYSTEM_PROMPT = """You are a high-level Orchestrator for a corporate assistant.

Your responsibilities:
1. Analyze the user request carefully.
2. If the request is simple (greeting, small talk, trivial question), respond immediately without calling any tools.
3. If the request requires information or action, select the appropriate tool(s).
   - If multiple independent subtasks can be resolved in parallel (e.g., search documents AND search the web), call those tools concurrently.
4. If a tool call returned an error, read the error message, correct the arguments or choose a different tool, and retry. Do not surface raw errors to the user.
5. If the task requires deep expertise (code generation, mathematics, legal analysis) or the user expresses dissatisfaction, delegate to call_strong_model.
6. For any action that takes more than a moment, emit a status update so the user knows what is happening.
7. Always respond in the same language the user is writing in."""


@dataclass(slots=True)
class GraphRuntime:
    settings: Settings
    ollama: OllamaClient

    def __post_init__(self) -> None:
        self.chunker_service = ChunkerServiceClient(
            base_url=self.settings.CHUNKER_SERVICE_URL,
            timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
        )
        self.pg_ingester = PgIngesterClient(
            base_url=self.settings.PG_INGESTER_URL,
            timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
        )
        self.tool_registry = ToolRegistry(
            [
                SearchHistoryTool(settings=self.settings, ollama=self.ollama),
                CallStrongModelTool(self.invoke_reasoning_model),
                DocumentSearcherTool(
                    base_url=self.settings.DOCUMENT_SEARCHER_URL,
                    timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
                ),
                FileViewerTool(
                    base_url=self.settings.FILE_VIEWER_URL,
                    timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
                ),
                WebSearcherTool(
                    base_url=self.settings.WEB_SEARCHER_URL,
                    timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
                ),
                ImageGeneratorTool(
                    base_url=self.settings.IMAGE_GENERATOR_URL,
                    timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
                ),
                FileConverterTool(
                    base_url=self.settings.FILE_CONVERTER_URL,
                    timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
                ),
            ]
        )
        self.graph = self._build_graph()

    def build_initial_state(self, *, user_id: str, message: str, status_queue: asyncio.Queue[str] | None = None) -> AssistantState:
        return {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "context": {},
            "intermediate_steps": [],
            "is_complex_task": False,
            "tool_retry_count": 0,
            "status_queue": status_queue,
        }

    async def run(self, state: AssistantState) -> AssistantState:
        return await self.graph.ainvoke(state)

    async def router_node(self, state: AssistantState) -> dict[str, Any]:
        user_message = state["messages"][-1]["content"]
        payload = {"message": user_message}
        async with trace(step_name="router", user_id=state["user_id"], input=payload) as t:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON with keys: is_simple, needs_tools, is_complex_task, "
                        "needs_reasoning_model, answer."
                    ),
                },
                {"role": "user", "content": user_message},
            ]
            response = await self.ollama.chat(
                model=self.settings.ROUTER_MODEL,
                messages=prompt,
                format="json",
            )
            t.estimated_tokens = _extract_token_estimate(response)
            content = response.get("message", {}).get("content", "")
            parsed = _parse_json_object(content)
            update = _router_fallback(user_message) if parsed is None else parsed
            t.output = update

        if update.get("is_simple") and update.get("answer"):
            return {"final_answer": str(update["answer"]), "next_action": "end"}
        if update.get("needs_reasoning_model") or update.get("is_complex_task"):
            return {"is_complex_task": True, "next_action": "reasoning"}
        return {"is_complex_task": bool(update.get("is_complex_task", False)), "next_action": "orchestrator"}

    async def orchestrator_node(self, state: AssistantState) -> dict[str, Any]:
        user_message = state["messages"][-1]["content"]
        payload = {
            "message": user_message,
            "context": state.get("context", {}),
            "intermediate_steps": state.get("intermediate_steps", []),
            "tool_retry_count": state.get("tool_retry_count", 0),
        }
        async with trace(step_name="orchestrator", user_id=state["user_id"], input=payload) as t:
            tool_descriptions = json.dumps(self.tool_registry.describe_for_model(), ensure_ascii=False)
            system_message = (
                f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
                f"Available tools (JSON): {tool_descriptions}\n\n"
                "Return strict JSON using exactly one of these actions:\n"
                "1) {\"action\":\"respond\",\"answer\":\"...\"}\n"
                "2) {\"action\":\"tools\",\"tool_calls\":[{\"tool\":\"name\",\"arguments\":{}}]}\n"
                "3) {\"action\":\"escalate\",\"task\":\"...\"}\n"
                "If a previous tool call returned an error, analyze the cause, correct the arguments, and retry — or choose an alternative tool or path."
            )
            response = await self.ollama.chat(
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
        return {"next_action": "reasoning", "context": {**state.get("context", {}), "escalation_task": parsed.get("task", user_message)}}

    async def tool_executor_node(self, state: AssistantState) -> dict[str, Any]:
        tool_calls = state.get("pending_tool_calls", [])
        tool_context = ToolContext(user_id=state["user_id"], state=state, emit_status=lambda s: self.emit_status(state, s))
        payload = {"tool_calls": tool_calls}
        async with trace(step_name="tool_executor", user_id=state["user_id"], input=payload) as t:
            results = await asyncio.gather(
                *(self.tool_registry.invoke(call["tool"], call.get("arguments", {}), tool_context) for call in tool_calls)
            )
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
            if retries >= self.settings.MAX_TOOL_RETRIES:
                update["next_action"] = "reasoning"
            else:
                update["next_action"] = "orchestrator"
            return update
        update["last_tool_error"] = None
        update["next_action"] = "orchestrator"
        return update

    async def strong_model_node(self, state: AssistantState) -> dict[str, Any]:
        task = state.get("context", {}).get("escalation_task") or state["messages"][-1]["content"]
        payload = {
            "task": task,
            "context": state.get("context", {}),
            "intermediate_steps": state.get("intermediate_steps", []),
        }
        async with trace(step_name="reasoning_model", user_id=state["user_id"], input=payload) as t:
            answer = await self.invoke_reasoning_model(task, state)
            t.output = {"answer": answer}
        return {"final_answer": answer, "next_action": "end"}

    async def invoke_reasoning_model(self, task: str, state: dict[str, Any]) -> str:
        await self.emit_status(state, "Calling reasoning model...")
        response = await self.ollama.chat(
            model=self.settings.REASONING_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise corporate assistant. Use tool results when present."},
                {"role": "user", "content": json.dumps({"task": task, "state": state}, ensure_ascii=False, default=str)},
            ],
        )
        return response.get("message", {}).get("content", "")

    async def emit_status(self, state: dict[str, Any], message: str) -> None:
        queue = state.get("status_queue")
        if queue is not None:
            await queue.put(message)

    def _build_graph(self):
        graph = StateGraph(AssistantState)
        graph.add_node("router", self.router_node)
        graph.add_node("orchestrator", self.orchestrator_node)
        graph.add_node("tools", self.tool_executor_node)
        graph.add_node("reasoning", self.strong_model_node)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            after_router,
            {"end": END, "orchestrator": "orchestrator", "reasoning": "reasoning"},
        )
        graph.add_conditional_edges(
            "orchestrator",
            after_orchestrator,
            {"end": END, "tools": "tools", "reasoning": "reasoning"},
        )
        graph.add_conditional_edges(
            "tools",
            after_tools,
            {"orchestrator": "orchestrator"},
        )
        graph.add_edge("reasoning", END)
        return graph.compile()


def _extract_token_estimate(response: dict[str, Any]) -> int | None:
    prompt_tokens = response.get("prompt_eval_count")
    completion_tokens = response.get("eval_count")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return int(prompt_tokens or 0) + int(completion_tokens or 0)


def _parse_json_object(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _router_fallback(user_message: str) -> dict[str, Any]:
    message = user_message.lower().strip()
    if message in {"hi", "hello", "hey", "thanks", "thank you"}:
        return {
            "is_simple": True,
            "needs_tools": False,
            "is_complex_task": False,
            "needs_reasoning_model": False,
            "answer": "Hello! How can I help you today?" if message != "thanks" and message != "thank you" else "You're welcome!",
        }
    code_words = ["code", "implement", "architecture", "debug", "analyze", "compare", "design"]
    is_complex = any(word in message for word in code_words)
    return {
        "is_simple": False,
        "needs_tools": not is_complex,
        "is_complex_task": is_complex,
        "needs_reasoning_model": is_complex,
        "answer": "",
    }


def _normalize_tool_calls(raw_calls: Any) -> list[ToolCall]:
    normalized: list[ToolCall] = []
    if not isinstance(raw_calls, list):
        return normalized
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool")
        arguments = item.get("arguments", {})
        if isinstance(tool, str) and isinstance(arguments, dict):
            normalized.append({"tool": tool, "arguments": arguments})
    return normalized
