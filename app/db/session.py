from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine, AsyncEngine

from app.core.config import get_settings
from app.db.models import Base

_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.POSTGRES_URL, pool_pre_ping=True)
    return _engine


def get_session_maker() -> async_sessionmaker:
    global _session_maker
    if _session_maker is None:
        _session_maker = async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _session_maker


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.run_sync(Base.metadata.create_all)
