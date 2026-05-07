from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.metrics_router import router as metrics_router
from app.api.routes import router as api_router
from app.core.config import get_settings
from app.metrics import init_watermark
from app.core.llm_client import LLMClient
from app.graph.graph_runtime import GraphRuntime
from app.services.qdrant_ingester_client import QdrantIngesterClient
from app.services.file_ingestion_service import FileIngestionService
from app.services.pg_ingester import IngesterService

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )
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

    qdrant_ingester = QdrantIngesterClient(
        base_url=settings.QDRANT_INGESTER_URL,
        api_key=settings.API_KEY,
    )
    pg_ingester = IngesterService()
    file_ingestion = FileIngestionService(
        pg_ingester=pg_ingester,
        qdrant_ingester=qdrant_ingester,
        settings=settings,
    )

    app.state.runtime = runtime
    app.state.file_ingestion = file_ingestion
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    app.include_router(metrics_router)   # /metrics — no version prefix

    # Initialize metrics watermark to prevent double-counting on restart
    await init_watermark()

    yield

    await runtime.shutdown()
    await llm.aclose()


app = FastAPI(title="dialogue-bot", lifespan=lifespan)
