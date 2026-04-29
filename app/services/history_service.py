from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select

from app.db.models import Message
from app.db.session import get_session_maker

logger = logging.getLogger(__name__)


class HistoryService:
    """Save conversation messages to PostgreSQL and search them semantically.

    Reuses IngesterService's embedding model so the model is loaded only once
    for the lifetime of the application.
    """

    def __init__(self, ingester_service) -> None:
        self._ingester = ingester_service

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save_message(
        self,
        *,
        user_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        """Insert a single message row (embedding IS NULL) and return its UUID."""
        msg = Message(
            user_id=user_id,
            role=role,
            content=content,
            metadata_json=metadata,
        )
        async with get_session_maker()() as session:
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
        logger.debug("Saved message %s (role=%s, user=%s)", msg.id, role, user_id)
        return msg.id

    # ------------------------------------------------------------------
    # Read / Search
    # ------------------------------------------------------------------

    async def search(
        self,
        *,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Cosine-distance search over already-embedded messages for a user.

        Returns an empty list when no embedded messages exist yet for the user
        or when the embedding model is not yet warm.
        """
        try:
            vectors = await self._ingester._embed_texts([query])
        except Exception as exc:
            logger.warning("Embedding query failed, skipping history search: %s", exc)
            return []

        if not vectors:
            return []
        query_vector = vectors[0]

        async with get_session_maker()() as session:
            distance = Message.embedding.cosine_distance(query_vector)
            stmt = (
                select(
                    Message.id,
                    Message.role,
                    Message.content,
                    Message.created_at,
                    distance.label("distance"),
                )
                .where(
                    Message.user_id == user_id,
                    Message.embedding.is_not(None),
                )
                .order_by(distance)
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()

        return [
            {
                "id": str(row.id),
                "role": row.role,
                "content": row.content,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "distance": float(row.distance),
            }
            for row in rows
        ]
