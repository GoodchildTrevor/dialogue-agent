"""Graph runtime module for the assistant agent orchestration.

This module provides the :class:`GraphRuntime` class which orchestrates
the agent workflow using LangGraph. It manages MCP client connections,
tool registries, and the state machine that routes requests through
router, orchestrator, tool executor, and reasoning nodes.

Example:
    Basic usage::

        settings = Settings()
        llm_client = LLMClient(settings)
        runtime = GraphRuntime(settings, llm_client)
        await runtime.startup()
        try:
            state = runtime.build_initial_state(
                user_id="user_1",
                message="Hello",
                request_id="req_123",
            )
            result = await runtime.run(state)
        finally:
            await runtime.shutdown()
"""
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
from app.graph.edges import after_orchestrator, after_reasoning, after_router, after_tools
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

    """Core runtime orchestrator for the assistant agent.

    Manages the full lifecycle of the agent workflow, including MCP client
    initialization, tool registry management, node instantiation, and
    LangGraph state machine construction.

    The runtime coordinates four primary nodes:

    - **router** -- determines the initial flow based on the user input
    - **orchestrator** -- plans tool calls and delegates to tools
    - **tools** -- executes tool calls and collects results
    - **reasoning** -- performs deep reasoning via a strong LLM model

    MCP servers are configured via ``settings.mcp_servers_list`` which
    always includes the primary server first, followed by any additional
    servers declared in the ``MCP_SERVERS`` env var (JSON array).
    The legacy ``FILE_CONVERTER_MCP_URL`` is also supported and appended
    automatically when set.

    :param settings: Application configuration from :class:`Settings`
    :param llm_client: Client for interacting with the LLM backend

    :ivar settings: The application settings instance
    :ivar llm_client: The LLM client instance
    :ivar tool_registry: Registry of available tools from the primary MCP server.
    :ivar _all_registries: All ToolRegistry instances across all configured MCP servers.
    :ivar chunker_service: Client for the chunking service API.
    :ivar pg_ingester: Client for the PostgreSQL ingester service.
    :ivar history_service: Service for managing conversation history.
    :ivar graph: The compiled LangGraph state machine.
    :ivar _router_node: Internal router node instance.
    :ivar _orchestrator_node: Internal orchestrator node instance.
    :ivar _tool_executor_node: Internal tool executor node instance.
    :ivar _strong_model_node: Internal strong model (reasoning) node instance.
    """
    settings: Settings
    llm_client: LLMClient
    tool_registry: ToolRegistry | None = field(default=None, init=False)
    _all_registries: list[ToolRegistry] = field(default_factory=list, init=False)
    chunker_service: ChunkerServiceClient = field(init=False)
    pg_ingester: IngesterService = field(init=False)
    history_service: HistoryService = field(init=False)
    graph: Any = field(init=False)

    _router_node: RouterNode = field(init=False)
    _orchestrator_node: OrchestratorNode = field(init=False)
    _tool_executor_node: ToolExecutorNode = field(init=False)
    _strong_model_node: StrongModelNode = field(init=False)

    def __post_init__(self) -> None:
        """Initialize clients, registries, nodes, and compile the graph.

        Iterates over all MCP server configs returned by
        ``settings.mcp_servers_list`` and creates a :class:`ToolRegistry`
        for each one. The first registry is stored as ``tool_registry``
        for backward compatibility with code that references it directly.
        All registries are passed to the orchestrator, tool executor, and
        strong model nodes.
        """
        for server_cfg in self.settings.mcp_servers_list:
            transport = StreamableHttpTransport(
                server_cfg["url"],
                headers={"Authorization": f"Bearer {server_cfg['token']}"},
            )
            client = MCPClient(transport)
            registry = ToolRegistry(
                self.settings,
                client,
                name=server_cfg["name"],
            )
            self._all_registries.append(registry)

        # Keep tool_registry pointing to the primary server for backward compat
        self.tool_registry = self._all_registries[0] if self._all_registries else None

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
            self._all_registries,
            self.history_service,
        )
        self._tool_executor_node = ToolExecutorNode(
            self.emit_status, self.settings, self._all_registries
        )
        self._strong_model_node = StrongModelNode(self.llm_client, self.settings, self._all_registries)
        self.graph = self._build_graph()

    async def startup(self) -> None:
        """Startup the runtime by initializing MCP clients and refreshing tool registries.

        This method should be called before using the graph to ensure all
        MCP connections are established and tool descriptions are cached.

        :returns: None
        """
        for registry in self._all_registries:
            await registry.startup()

    async def shutdown(self) -> None:
        """Shutdown the runtime by closing MCP clients and releasing resources.

        This method should be called when the runtime is no longer needed
        to properly clean up MCP connections and tool registry state.

        :returns: None
        """
        for registry in self._all_registries:
            await registry.shutdown()

    async def refresh_tool_descriptions(self) -> None:
        """Refresh and cache the tool list from all connected MCP servers.

        Fetches the latest tool definitions from all configured MCP servers
        and updates the respective registries.

        :returns: None
        """
        for registry in self._all_registries:
            await registry.refresh_tools()

    def build_initial_state(
        self,
        *,
        user_id: str,
        message: str,
        request_id: str,
        status_queue: asyncio.Queue[str] | None = None,
        uploaded_files: list[dict[str, str]] | None = None,
    ) -> AssistantState:
        """Build the initial state for a new graph execution.

        Creates and returns an :class:`AssistantState` dictionary containing
        the starting configuration for the agent workflow, including user
        identification, message content, and optional metadata.

        :param user_id: Identifier for the user initiating the request
        :param message: The user's input message to process
        :param request_id: Unique identifier for this request
        :param status_queue: Optional async queue for streaming status updates
        :param uploaded_files: Optional list of dicts containing file metadata
            (e.g., from user file uploads)
        :returns: Initial :class:`AssistantState` dictionary
        """
        return {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
            "request_id": request_id,
            "context": {},
            "intermediate_steps": [],
            "is_complex_task": False,
            "tool_retry_count": 0,
            "status_queue": status_queue,
            "uploaded_files": uploaded_files or [],
            "images": [],
        }

    async def run(self, state: AssistantState) -> AssistantState:
        """Execute the graph workflow with the given initial state.

        Runs the compiled LangGraph state machine from the start node
        through the router, orchestrator, tools, and reasoning nodes
        as needed, returning the final state after completion.

        :param state: Initial :class:`AssistantState` to begin execution
        :returns: Final :class:`AssistantState` containing results, tool
            outputs, and intermediate processing data
        """
        return await self.graph.ainvoke(state)

    async def router_node(self, state: AssistantState) -> dict[str, Any]:
        """Route the user input through the router node.

        Determines the initial flow based on the user input by invoking
        the internal router node. Routes execution to either the
        orchestrator, reasoning, or directly to end.

        :param state: Current :class:`AssistantState` containing user message
            and context
        :returns: Dictionary of state updates from the router node
        """
        return await self._router_node.action(state)

    async def orchestrator_node(self, state: AssistantState) -> dict[str, Any]:
        """Execute the orchestrator node to plan and delegate tool calls.

        Checks whether the primary tool registry has cached descriptions
        before orchestration and refreshes all registries if empty.
        Then invokes the internal orchestrator node to plan tool calls
        based on the current state.

        :param state: Current :class:`AssistantState` containing message
            history, context, and intermediate steps
        :returns: Dictionary of state updates from the orchestrator node
            including planned tool calls
        """
        if self.tool_registry and self.tool_registry.is_empty:
            logger.warning("Tool cache is empty, refreshing before orchestration")
            await self.refresh_tool_descriptions()
        return await self._orchestrator_node.action(state)

    async def tool_executor_node(self, state: AssistantState) -> dict[str, Any]:
        """Execute tool calls planned by the orchestrator node.

        Invokes the internal tool executor node to perform all pending
        tool calls collected in the intermediate steps of the current
        state, then collects and stores the results.

        :param state: Current :class:`AssistantState` containing pending
            tool calls in intermediate steps
        :returns: Dictionary of state updates from the tool executor node
            with tool results
        """
        return await self._tool_executor_node.action(state)

    async def strong_model_node(self, state: AssistantState) -> dict[str, Any]:
        """Execute the strong model (reasoning) node.

        Invokes the internal strong model node to perform deep reasoning
        on the current state using a powerful LLM. This node is used for
        complex tasks requiring thorough analysis and reasoning.

        :param state: Current :class:`AssistantState` containing context
            and information needed for reasoning
        :returns: Dictionary of state updates from the strong model node
            with reasoning results
        """
        return await self._strong_model_node.action(state)

    async def emit_status(self, state: dict[str, Any], message: str) -> None:
        """Emit a status message to the status queue if available.

        Puts a status update message into the async queue provided in
        the state, enabling real-time status streaming to clients.

        :param state: State dictionary containing an optional ``status_queue``
            key with an :class:`asyncio.Queue` for status messages
        :param message: The status message to emit to the queue
        :returns: None
        """
        queue = state.get("status_queue")
        if queue is not None:
            await queue.put(message)

    def _build_graph(self):
        """Build and compile the LangGraph state machine.

        Constructs the :class:`StateGraph` with all four nodes (router,
        orchestrator, tools, reasoning), defines the edges connecting them,
        and configures conditional routing between nodes based on the
        output of each step. The compiled graph is stored in ``self.graph``.

        :returns: Compiled LangGraph state machine
        """
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
            {"end": END, "tools": "tools", "reasoning": "reasoning", "orchestrator": "orchestrator"},
        )
        graph.add_conditional_edges(
            "tools",
            after_tools,
            {"orchestrator": "orchestrator", "reasoning": "reasoning"},
        )
        graph.add_conditional_edges(
            "reasoning",
            after_reasoning,
            {"end": END, "tools": "tools"},
        )
        return graph.compile()
