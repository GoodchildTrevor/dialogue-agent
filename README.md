# dialogue-bot

`dialogue-bot` is the assistant runtime application for a universal corporate assistant built with FastAPI, LangGraph, Ollama, and PostgreSQL/PGvector. It orchestrates internal tools, external tool adapters, tracing, history search, and SSE streaming, but it does **not** implement `chunker_service`, `pg_ingester`, document parsing, ingestion internals, or external tool backends.

## Prerequisites

- Docker
- Docker Compose
- Ollama installed and running
- At least `ROUTER_MODEL`, `REASONING_MODEL`, and `EMBEDDING_MODEL` pulled into Ollama

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
- `app/tools/` contains internal tools and external HTTP tool adapters.
- `app/services/` contains async clients for external infrastructure services.
- `app/api/` exposes health, JSON chat, and SSE streaming endpoints.

## Metabase

Run Metabase from the compose stack and connect it to the same PostgreSQL instance to build dashboards on the `traces` table. Recommended starter charts are latency by `step_name`, tool error rate, average estimated tokens, and per-user orchestration volume.

## Notes

- No Qdrant integration is included.
- Retrieval is designed to happen through external tools and PostgreSQL history search only.
- External adapter paths are placeholders and should be aligned with real downstream contracts.
