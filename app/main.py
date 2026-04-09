from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.ollama import OllamaClient
from app.db.session import init_db
from app.graph.nodes import GraphRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ollama = OllamaClient(settings)
    app.state.runtime = GraphRuntime(settings=settings, ollama=ollama)
    await init_db()
    yield
    await ollama.aclose()


settings = get_settings()
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
app.include_router(api_router)
