# dialogue-bot

`dialogue-bot` is the assistant runtime application for a universal corporate assistant built with FastAPI, LangGraph, Ollama, and PostgreSQL/PGvector. It orchestrates internal tools via MCP (Model Context Protocol), tracing, history search, and SSE streaming, but it does **not** implement `chunker_service`, `pg_ingester`, document parsing, ingestion internals, or external tool backends.

## Prerequisites

- Docker
- Docker Compose
- Ollama installed and running
- At least `ROUTER_MODEL`, `REASONING_MODEL`, and `EMBEDDING_MODEL` pulled into Ollama
- MCP tool server (dialogue-agent-mcp) running and accessible at `MCP_SERVER_URL`

## Quick start

1. Copy the environment template and fill in real values:
   ```bash
   cp .env.example .env
   ```
2. Pull all three required models (use the names you set in `.env`):
   ```bash
   ollama pull $ROUTER_MODEL
   ollama pull $REASONING_MODEL
   ollama pull $EMBEDDING_MODEL
   ```
   The default values from `.env.example` are:
   | Variable | Default |
   |---|---|
   | `ROUTER_MODEL` | `llama3.1:8b` |
   | `REASONING_MODEL` | `qwen2.5-coder:14b` |
   | `EMBEDDING_MODEL` | `nomic-embed-text` |
3. Start the stack:
   ```bash
   docker compose up --build
   ```
4. Verify the healthcheck:
   ```bash
   curl http://localhost:8000/healthz
   ```
5. Test the SSE endpoint:
   ```bash
   curl -N -X POST http://localhost:8000/api/v1/stream \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test-user", "message": "Find documents about Q3 budget"}'
   ```

## Ollama setup

Ollama must be reachable at `OLLAMA_BASE_URL` from inside the Docker network. The default `.env.example` uses `http://host.docker.internal:11434`; on Linux, the compose file adds `host-gateway` so the container can reach the host.

All three model names are configured exclusively via environment variables — no model name is hardcoded anywhere in the application. Set `ROUTER_MODEL`, `REASONING_MODEL`, and `EMBEDDING_MODEL` in your `.env` file before starting the stack.

## Architecture

- `app/core/config.py` centralizes all environment-based settings with `pydantic-settings`.
- `app/core/tracing.py` measures latency and persists structured traces asynchronously.
- `app/db/` contains SQLAlchemy 2.0 models and database initialization with the `vector` extension.
- `app/graph/` contains the LangGraph state, nodes, and transitions.
- `app/graph/tool_registry.py` manages MCP client connection and tool invocation.
- `app/services/` contains async clients for external infrastructure services.
- `app/api/` exposes health, JSON chat, and SSE streaming endpoints.

## MCP Integration

Tools are provided via MCP (Model Context Protocol) from an external MCP server. The application:

1. Connects to the MCP server at startup via `fastmcp.Client`
2. Discovers available tools via `list_tools()`
3. Invokes tools via `call_tool()` when the orchestrator requests them
4. Properly disconnects from the MCP server on shutdown

The MCP server URL is configured via `MCP_SERVER_URL` in the environment.

## Notes

- Retrieval is designed to happen through external tools and PostgreSQL history search only.
- External adapter paths are placeholders and should be aligned with real downstream contracts.
- Tool execution happens through MCP; the application does not implement tools directly.
