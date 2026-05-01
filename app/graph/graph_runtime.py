from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client as MCPClient
from fastmcp.client.transports import StreamableHttpTransport
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings
from app.core.llm_client import LLMClient
from app.graph.edges import after_orchestrator, after_router, after_tools
from app.graph.state import AssistantState
from app.graph.tool_registry import ToolRegistry

from app.graph.nodes.router import RouterNode
from app.graph.nodes.orchestrator import OrchestratorNode
from app.graph.nodes.tool_executor import ToolExecutorNode
from app.graph.nodes.strong_model import StrongModelNode

from app.services.chunker_service import ChunkerServiceClient
from app.services.pg_ingester import IngesterService
from app.services.history_service import HistoryService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphRuntime:
    settings: Settings
    llm_client: LLMClient
    _mcp_client: MCPClient | None = field(default=None, init=False)
    _file_converter_client: MCPClient | None = field(default=None, init=False)
    tool_registry: ToolRegistry | None = field(default=None, init=False)
    file_converter_registry: ToolRegistry | None = field(default=None, init=False)
    chunker_service: ChunkerServiceClient = field(init=False)
    pg_ingester: IngesterService = field(init=False)
    history_service: HistoryService = field(init=False)
    graph: Any = field(init=False)

    _router_node: RouterNode = field(init=False)
    _orchestrator_node: OrchestratorNode = field(init=False)
    _tool_executor_node: ToolExecutorNode = field(init=False)
    _strong_model_node: StrongModelNode = field(init=False)

    def __post_init__(self) -> None:
        transport = StreamableHttpTransport(
            self.settings.MCP_SERVER_URL,
            headers={"Authorization": f"Bearer {self.settings.MCP_AUTH_TOKEN}"},
        )
        self._mcp_client = MCPClient(transport)
        self.tool_registry = ToolRegistry(self.settings, self._mcp_client)

        registries: list[ToolRegistry] = [self.tool_registry]

        if self.settings.FILE_CONVERTER_MCP_URL:
            transport2 = StreamableHttpTransport(
                self.settings.FILE_CONVERTER_MCP_URL,
                headers={"Authorization": f"Bearer {self.settings.FILE_CONVERTER_AUTH_TOKEN}"},
            )
            self._file_converter_client = MCPClient(transport2)
            self.file_converter_registry = ToolRegistry(self.settings, self._file_converter_client)
            registries.append(self.file_converter_registry)
        
        self.chunker_service = ChunkerServiceClient(
            base_url=self.settings.CHUNKER_SERVICE_URL,
            timeout=self.settings.TOOL_REQUEST_TIMEOUT_SECONDS,
        )
        self.pg_ingester = IngesterService()
        self.history_service = HistoryService(self.pg_ingester)

        self._router_node = RouterNode(self.llm_client, self.settings)
        self._orchestrator_node = OrchestratorNode(
            self.llm_client,
            self.settings,
            registries,
            self.history_service,
        )
        self._tool_executor_node = ToolExecutorNode(
            self.emit_status, self.settings, registries
        )
        self._strong_model_node = StrongModelNode(self.llm_client, self.settings)
        self.graph = self._build_graph()

    async def refresh_tool_descriptions(self) -> None:
        """Fetch and cache the tool list from all MCP servers."""
        if self.tool_registry:
            await self.tool_registry.refresh_tools()
        if self.file_converter_registry:
            await self.file_converter_registry.refresh_tools()

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
        return await self._router_node.action(state)

    async def orchestrator_node(self, state: AssistantState) -> dict[str, Any]:
        if self.tool_registry and self.tool_registry.is_empty:
            logger.warning("Tool cache is empty, refreshing before orchestration")
            await self.refresh_tool_descriptions()
        return await self._orchestrator_node.action(state)

    async def tool_executor_node(self, state: AssistantState) -> dict[str, Any]:
        return await self._tool_executor_node.action(state)

    async def strong_model_node(self, state: AssistantState) -> dict[str, Any]:
        return await self._strong_model_node.action(state)

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
            {"orchestrator": "orchestrator", "reasoning": "reasoning"},
        )
        graph.add_edge("reasoning", END)
        return graph.compile()
