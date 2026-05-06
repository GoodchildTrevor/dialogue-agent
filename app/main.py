from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.graph.graph_runtime import GraphRuntime

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
    # Keep third-party loggers quieter so our app logs stand out
    for noisy in ("httpx", "httpcore", "uvicorn.access", "fastmcp", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.LOG_LEVEL)

    app.state.settings = settings
    llm = LLMClient(settings)
    runtime = GraphRuntime(settings=settings, llm_client=llm)

    await runtime.startup()

    try:
        await runtime.refresh_tool_descriptions()
    except Exception as e:
        logger.warning("Failed to refresh tool descriptions on startup: %s", e)

    app.state.runtime = runtime
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    yield

    await runtime.shutdown()
    await llm.aclose()


app = FastAPI(title="dialogue-bot", lifespan=lifespan)
