from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client as MCPClient
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.llm_client import LLMClient
from app.core.tracing import trace
from app.graph.edges import after_orchestrator, after_router, after_tools
from app.graph.prompt_fragments import ORCHESTRATOR_SYSTEM_PROMPT, REASONING_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT
from app.graph.state import AssistantState, ToolCall
from app.graph.tool_registry import ToolContext, ToolRegistry
from app.services.chunker_service import ChunkerServiceClient
from app.services.pg_ingester import PgIngesterClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphRuntime:
    settings: Settings
    ollama: LLMClient  # named 'ollama' for backward compat with main.py
    _mcp_client: MCPClient | None = field(default=None, init=False)
    tool_registry: ToolRegistry | None = field(default=None, init=False)
    _tool_descriptions: list[dict[str, Any]] = field(default_factory=list)
    _mcp_connected: bool = field(default=False, init=False)
    chunker_service: ChunkerServiceClient = field(init=False)
    pg_ingester: PgIngesterClient = field(init=False)
    graph: Any = field(init=False)

    def __post_init__(self) -> None:
        self._mcp_client = MCPClient(self.settings.MCP_SERVER_URL)
        self.tool_registry = ToolRegistry(self.settings, self._mcp_client)
        self.chunker_service = ChunkerServiceClient(
            base_url=self.settings.CHUNKER_SERVICE_URL,
            timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
        )
        self.pg_ingester = PgIngesterClient(
            base_url=self.settings.PG_INGESTER_URL,
            timeout_seconds=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
        )
        self.graph = self._build_graph()
        self._mcp_lock = asyncio.Lock()

    async def connect_mcp(self) -> None:
        """Connect to the MCP server."""
        if self._mcp_client and not self._mcp_connected:
            async with self._mcp_lock:
                if self._mcp_client and not self._mcp_connected:
                    try:
                        await self._mcp_client.__aenter__()
                        self._mcp_connected = True
                        logger.info("Connected to MCP server")
                    except Exception as e:
                        logger.error(f"Failed to connect to MCP server: {e}")
                        raise

    async def disconnect_mcp(self) -> None:
        """Disconnect from the MCP server."""
        if self._mcp_client and self._mcp_connected:
            try:
                await self._mcp_client.__aexit__(None, None, None)
                self._mcp_connected = False
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error disconnecting from MCP server: {e}")

    async def refresh_tool_descriptions(self) -> None:
        """Refresh tool descriptions from MCP server."""
        if not self._mcp_connected:
            await self.connect_mcp()
        if self.tool_registry:
            await self.tool_registry.refresh_tools()

    def build_initial_state(
        self,
        *,
        user_id: str,
        message: str,
        request_id: str,
        status_queue: asyncio.Queue[str] | None = None,
    ) -> AssistantState:
        return {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "request_id": request_id,
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
        async with trace(
            step_name="router",
            user_id=state["user_id"],
            request_id=state["request_id"],
            input=payload,
        ) as t:
            t.input_hash = hashlib.sha256(user_message.encode()).hexdigest()
            prompt = [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
            response = await self.ollama.chat(
                model=self.settings.ROUTER_MODEL,
                messages=prompt,
                format="json",
            )
            t.estimated_tokens = _extract_token_estimate(response)
            t.model_used = self.settings.ROUTER_MODEL
            content = response.get("message", {}).get("content", "")
            parsed = _parse_json_object(content)
            update = _router_fallback(user_message) if parsed is None else parsed
            t.output = update
            if update.get("is_simple") and update.get("answer"):
                t.route_decision = "simple"
            elif update.get("needs_reasoning_model") or update.get("is_complex_task"):
                t.route_decision = "reasoning"
            elif parsed is None:
                t.route_decision = "fallback"
            else:
                t.route_decision = "orchestrator"

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
        async with trace(step_name="orchestrator", user_id=state["user_id"], request_id=state["request_id"], input=payload) as t:
            tool_descriptions = json.dumps(
                self.tool_registry.describe_for_model() if self.tool_registry else [],
                ensure_ascii=False,
            )
            system_message = (
                f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
                f"Available tools (JSON): {tool_descriptions}\n"
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
        return {
            "next_action": "reasoning",
            "context": {
                **state.get("context", {}),
                "escalation_task": parsed.get("task", user_message),
            },
        }

    async def tool_executor_node(self, state: AssistantState) -> dict[str, Any]:
        tool_calls = state.get("pending_tool_calls", [])
        payload = {"tool_calls": tool_calls}
        async with trace(step_name="tool_executor", user_id=state["user_id"], request_id=state["request_id"], input=payload) as t:
            tool_context = ToolContext(
                user_id=state["user_id"],
                state=state,
                emit_status=lambda s: self.emit_status(state, s),
            )
            try:
                results = await asyncio.gather(
                    *(
                        self.tool_registry.invoke(call["tool"], call.get("arguments", {}), tool_context)
                        for call in tool_calls
                    )
                )
            except Exception as e:
                logger.error(f"Error in tool execution: {e}")
                return {
                    "intermediate_steps": state.get("intermediate_steps", []),
                    "tool_results": [{"ok": False, "error": str(e)}],
                    "pending_tool_calls": [],
                    "context": state.get("context", {}),
                    "tool_retry_count": state.get("tool_retry_count", 0) + 1,
                    "last_tool_error": str(e),
                    "next_action": "reasoning",
                }
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

    async def strong_model_node(self, state: AssistantState) -> dict[str, Any]:
        task = (
            state.get("context", {}).get("escalation_task")
            or state["messages"][-1]["content"]
        )
        payload = {
            "task": task,
            "context": state.get("context", {}),
            "intermediate_steps": state.get("intermediate_steps", []),
        }
        async with trace(
            step_name="reasoning_model",
            user_id=state["user_id"],
            request_id=state["request_id"],
            input=payload,
        ) as t:
            t.model_used = self.settings.REASONING_MODEL
            answer = await self._invoke_reasoning_model(task, state)
            t.output = {"answer": answer}
        return {"final_answer": answer, "next_action": "end"}

    async def emit_status(self, state: dict[str, Any], message: str) -> None:
        queue = state.get("status_queue")
        if queue is not None:
            await queue.put(message)

    async def _invoke_reasoning_model(self, task: str, state: AssistantState) -> str:
        """Invoke the reasoning model for complex tasks."""
        prompt = [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": task,
                        "context": state.get("context", {}),
                        "intermediate_steps": state.get("intermediate_steps", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = await self.ollama.chat(
            model=self.settings.REASONING_MODEL,
            messages=prompt,
        )
        return response.get("message", {}).get("content", "")

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
            {"orchestrator": "orchestrator", "reasoning": "reasoning"},
        )
        graph.add_edge("reasoning", END)
        return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            "answer": (
                "You're welcome!"
                if message in {"thanks", "thank you"}
                else "Hello! How can I help you today?"
            ),
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