from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.core.config import Settings
from app.core.ollama import OllamaClient
from app.db.models import Message
from app.db.session import async_session_maker
from app.tools.base import ToolContext, ToolSpec


class SearchHistoryTool:
    spec = ToolSpec(
        name="search_history",
        description="Semantic search over the current user's previous messages stored in PostgreSQL/PGvector.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        layer="internal",
    )

    def __init__(self, *, settings: Settings, ollama: OllamaClient) -> None:
        self._settings = settings
        self._ollama = ollama

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        if context.emit_status is not None:
            await context.emit_status("Searching history...")

        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", self._settings.HISTORY_SEARCH_LIMIT))
        embedding = await self._ollama.embeddings(model=self._settings.EMBEDDING_MODEL, prompt=query)

        async with async_session_maker() as session:
            distance = Message.embedding.cosine_distance(embedding)
            stmt = (
                select(Message.id, Message.role, Message.content, Message.created_at, distance.label("distance"))
                .where(Message.user_id == context.user_id, Message.embedding.is_not(None))
                .order_by(distance)
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()

        return {
            "query": query,
            "matches": [
                {
                    "id": str(row.id),
                    "role": row.role,
                    "content": row.content,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "distance": float(row.distance),
                }
                for row in rows
            ],
        }
