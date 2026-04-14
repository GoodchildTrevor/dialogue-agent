from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.ollama import OllamaClient
from app.db.session import init_db
from app.graph.nodes import GraphRuntime

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ollama = OllamaClient(settings)
    app.state.runtime = GraphRuntime(settings=settings, ollama=ollama)
    await init_db()
    yield
    await ollama.aclose()


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request.state.request_id = uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "request",
        extra={
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


# Single registration — endpoints available only at /api/v1/*
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
