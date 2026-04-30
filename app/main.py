from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.graph.graph_runtime import GraphRuntime

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.title = settings.APP_NAME

    llm = LLMClient(settings)
    runtime = GraphRuntime(settings=settings, llm_client=llm)

    # Pre-warm the tool cache. If MCP is unavailable the warning is logged
    # and the orchestrator will retry on the first real request.
    try:
        await runtime.refresh_tool_descriptions()
    except Exception as e:
        logger.warning("Failed to refresh tool descriptions on startup: %s", e)

    app.state.runtime = runtime
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    yield

    # No persistent MCP connection to close — StreamableHttpTransport is
    # stateless and each call manages its own session.
    await llm.aclose()


app = FastAPI(title="dialogue-bot", lifespan=lifespan)
