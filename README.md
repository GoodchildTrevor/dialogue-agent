# dialogue-agent

`dialogue-agent` is the assistant runtime for a universal corporate assistant built with FastAPI, LangGraph, LiteLLM, and PostgreSQL/pgvector. It orchestrates internal tools via MCP (Model Context Protocol), semantic history search, file ingestion, and SSE streaming.

## Prerequisites

- Docker and Docker Compose
- A running **LiteLLM proxy** reachable at `LLM_BASE_URL` with model aliases matching `ROUTER_MODEL` and `REASONING_MODEL` from your `.env`
- MCP tool server (`dialogue-agent-mcp`) running and accessible at `MCP_SERVER_URL`
- _(Optional)_ File-converter MCP server at `FILE_CONVERTER_MCP_URL`

## Quick start

1. Copy the environment template and fill in real values:
   ```bash
   cp .env.example .env
   ```
2. Review the key variables in `.env`:

   | Variable | Default | Description |
   |---|---|---|
   | `LLM_BASE_URL` | `http://litellm:4000` | LiteLLM proxy base URL |
   | `ROUTER_MODEL` | `router-model` | Model alias for routing/classification |
   | `REASONING_MODEL` | `reasoning-model` | Model alias for main reasoning |
   | `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | fastembed model for in-process embedding |
   | `MCP_SERVER_URL` | `http://dialogue-agent-mcp:8000/mcp` | MCP tool server URL |
   | `MCP_AUTH_TOKEN` | — | Shared Bearer token (must match MCP server) |
   | `API_KEY` | — | `X-API-Key` header value required by all endpoints |

3. Start the stack:
   ```bash
   docker compose up --build
   ```
4. Verify the health endpoint (no auth required):
   ```bash
   curl http://localhost:8000/healthz
   ```
5. Test the SSE streaming endpoint:
   ```bash
   curl -N -X POST http://localhost:8000/api/v1/stream \
     -H "Content-Type: application/json" \
     -H "X-API-Key: <your-api-key>" \
     -d '{"user_id": "test-user", "message": "Find documents about Q3 budget"}'
   ```

## LiteLLM setup

All LLM calls are routed through a LiteLLM proxy. The model names in `.env` must match `model_name` entries in your `litellm_config.yaml`. No model name is hardcoded in the application — all are resolved via environment variables.

## API

All routes (except `/healthz`) require the `X-API-Key` header.

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Health check — returns `{"status": "ok"}` |
| `POST` | `/api/v1/chat` | Blocking JSON chat — returns `{"answer": "...", "images": [...]}` |
| `POST` | `/api/v1/stream` | SSE streaming chat — emits `status`, `answer`, and `[DONE]` frames |
| `POST` | `/api/v1/upload` | Upload one or more files for ingestion (multipart/form-data) |
| `GET` | `/api/v1/upload/{file_id}/status` | Poll ingestion status of an uploaded file |

### Chat request body (`/chat` and `/stream`)

```json
{
  "user_id": "alice",
  "message": "Summarise last quarter's report",
  "uploaded_files": [
    {"file_id": "uuid", "filename": "report.pdf"}
  ]
}
```

`uploaded_files` is optional. When omitted, the service auto-attaches recently indexed files for the user (controlled by `FILE_AUTO_ATTACH_MINUTES`).

### File upload (`/upload`)

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-API-Key: <your-api-key>" \
  -F "user_id=alice" \
  -F "files=@report.pdf"
# → {"files": [{"file_id": "...", "filename": "report.pdf", "status": "pending"}]}
```

Poll status until `"indexed"`:
```bash
curl http://localhost:8000/api/v1/upload/<file_id>/status \
  -H "X-API-Key: <your-api-key>"
```

## Architecture

- `app/core/config.py` — centralizes all environment-based settings with `pydantic-settings`.
- `app/metrics.py` — Prometheus metrics (counters, histograms, in-flight gauges).
- `app/db/` — SQLAlchemy 2.0 async models and database initialization with the `pgvector` extension.
- `app/graph/` — LangGraph state machine: nodes, transitions, and `GraphRuntime`.
- `app/graph/tool_registry.py` — MCP client connection lifecycle and tool invocation.
- `app/services/history_service.py` — saves messages to PostgreSQL; semantic search via pgvector.
- `app/services/pg_ingester.py` — async client that triggers embedding of saved messages.
- `app/services/chunker_service.py` — client for the external chunker service.
- `app/services/file_ingestion_service.py` — orchestrates the full file ingestion pipeline.
- `app/services/qdrant_ingester_client.py` — sends chunked file data to the Qdrant ingester.
- `app/api/routes.py` — health, JSON chat, SSE streaming, file upload, and status endpoints.

## Service ecosystem

`dialogue-agent` is the central orchestrator. It depends on several external services:

| Service | Repo | Default port | Purpose |
|---|---|---|---|
| `dialogue-agent` (this) | — | `8000` | LangGraph orchestrator, SSE streaming, history search |
| `dialogue-agent-mcp` | [dialogue-agent-mcp](https://github.com/GoodchildTrevor/dialogue-agent-mcp) | configured via `MCP_SERVER_URL` | MCP tool server — exposes internal tools |
| `qdrant-ingester` | [qdrant-ingester](https://github.com/GoodchildTrevor/qdrant-ingester) | `8000` (internal) | Chunks uploaded files, embeds and upserts into Qdrant |
| `postgres` | pgvector/pgvector:pg16 | `5432` (internal) | Stores messages, embeddings, file metadata |

## MCP integration

Tools are provided via MCP from an external server. On startup the application:

1. Connects to the MCP server at `MCP_SERVER_URL` via `fastmcp.Client` using a Bearer token.
2. Discovers available tools via `list_tools()`.
3. Invokes tools via `call_tool()` when the orchestrator requests them.
4. Disconnects cleanly on shutdown.

An optional second MCP server (`FILE_CONVERTER_MCP_URL`) can be configured for file conversion. Leave the variable blank to disable it.

## Embedding

Embeddings are computed **in-process** using [fastembed](https://github.com/qdrant/fastembed). The model is set via `EMBEDDING_MODEL_NAME` (default: `Qwen/Qwen3-Embedding-0.6B`). Batch sizes are tunable via `EMBEDDING_BATCH_SIZE` and `EMBEDDING_INSERT_BATCH_SIZE`.

Stored embeddings enable semantic search over past conversations:
```sql
SELECT content FROM messages
ORDER BY embedding <=> $query_vector
LIMIT 5;
```

## Monitoring

The application exposes a Prometheus-compatible `/metrics` endpoint. Prometheus is included in the `docker-compose.yml` and scrapes the app automatically — it is available at `http://localhost:9090`.

Dashboard configs for Grafana are provided in the `grafana/` directory. To use them, connect your own Grafana instance to the Prometheus data source at `http://localhost:9090` and import the JSON files from `grafana/`.

Retention is set to 30 days (`--storage.tsdb.retention.time=30d`).

## Running tests

```bash
docker compose --profile test run --rm test
```

## Notes

- Retrieval happens through MCP tools and PostgreSQL semantic history search only.
- Tool execution is fully delegated to the MCP server; no tools are implemented directly in this service.
