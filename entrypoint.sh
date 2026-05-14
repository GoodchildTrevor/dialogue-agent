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

# If the 'files' table already exists but alembic_version is empty,
# it means the DB was created before Alembic was introduced.
# Stamp it at 0001 so that only incremental migrations (0002+) are applied.
CURRENT=$(alembic current 2>/dev/null || true)
if [ -z "$CURRENT" ]; then
  echo "  No Alembic version found — checking if schema already exists..."
  TABLE_EXISTS=$(python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.core.config import get_settings
async def check():
    engine = create_async_engine(get_settings().POSTGRES_URL)
    async with engine.connect() as conn:
        result = await conn.execute(text(\"SELECT to_regclass('public.files')\"))
        row = result.scalar()
        print('yes' if row else 'no')
    await engine.dispose()
asyncio.run(check())
" 2>/dev/null || echo "no")
  if [ "$TABLE_EXISTS" = "yes" ]; then
    echo "  Schema exists without Alembic tracking — stamping at 0001..."
    alembic stamp 0001
  fi
fi

alembic upgrade heads

echo "Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
