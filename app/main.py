from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.llm_client import LLMClient
from app.db.session import init_db
from app.graph.nodes import GraphRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    llm = LLMClient(settings)
    runtime = GraphRuntime(settings=settings, ollama=llm)
    await init_db()
    await runtime.refresh_tool_descriptions()
    app.state.runtime = runtime
    yield
    await llm.aclose()


settings = get_settings()
app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.include_router(api_router, prefix=settings.API_V1_PREFIX)
