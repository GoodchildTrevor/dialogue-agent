#!/bin/sh
set -e

echo "Waiting for PostgreSQL..."
until python -c "
import os, asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import get_settings
async def check():
    engine = create_async_engine(get_settings().POSTGRES_URL, pool_pre_ping=True)
    async with engine.connect():
        pass
    await engine.dispose()
asyncio.run(check())
" 2>/dev/null; do
  echo "  PostgreSQL not ready yet, retrying in 2s..."
  sleep 2
done

echo "Running Alembic migrations..."
alembic upgrade heads

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
