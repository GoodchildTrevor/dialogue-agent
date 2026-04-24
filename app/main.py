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

    try:
        await runtime.refresh_tool_descriptions()
    except Exception as e:
        logger.warning(f"Failed to refresh tool descriptions on startup: {e}")
    app.state.runtime = runtime

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    yield

    await runtime.disconnect_mcp()
    await llm.aclose()


app = FastAPI(title="dialogue-bot", lifespan=lifespan)
