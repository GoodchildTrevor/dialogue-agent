from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Sequence

from more_itertools import chunked
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Message
from app.db.session import get_session_maker

logger = logging.getLogger(__name__)

# Maximum rows loaded into RAM per sync page
SYNC_PAGE_SIZE = 500


class IngesterService:
    """
    In-process embedding service: reads Message rows from the shared
    PostgreSQL session, embeds via fastembed, and bulk-updates
    messages.embedding — no HTTP hop required.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        self._insert_batch_size = settings.EMBEDDING_INSERT_BATCH_SIZE
        self._model_name = settings.EMBEDDING_MODEL_NAME
        self._model = None  # lazy-loaded on first embed call

    def _get_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        message_ids: list[uuid.UUID],
        *,
        force_reembed: bool = False,
    ) -> dict[str, int]:
        """Embed a specific list of messages by ID."""
        async with get_session_maker()() as session:
            messages = await self._fetch_by_ids(session, message_ids, force_reembed)
            skipped = len(message_ids) - len(messages)
            if not messages:
                return {"messages_embedded": 0, "messages_skipped": skipped}
            embedded = await self._embed_and_save(session, messages)
        return {"messages_embedded": embedded, "messages_skipped": skipped}

    async def sync(
        self,
        *,
        user_id: str | None = None,
    ) -> dict[str, int]:
        """Embed all messages WHERE embedding IS NULL, paginated to avoid OOM."""
        total_embedded = 0
        offset = 0
        while True:
            async with get_session_maker()() as session:
                messages = await self._fetch_without_embedding(
                    session, user_id=user_id, limit=SYNC_PAGE_SIZE, offset=offset
                )
                if not messages:
                    break
                embedded = await self._embed_and_save(session, messages)
            total_embedded += embedded
            offset += SYNC_PAGE_SIZE
        return {"re_embedded": total_embedded}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Run fastembed synchronously in a thread pool."""
        model = self._get_model()
        batch_size = self._batch_size

        def _sync() -> list[list[float]]:
            results: list[list[float]] = []
            for batch in chunked(texts, batch_size):
                embeddings = list(model.embed(batch))
                results.extend(emb.tolist() for emb in embeddings)
            return results

        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(None, _sync)
        logger.info("Embedded %d texts", len(vectors))
        return vectors

    async def _embed_and_save(
        self, session: AsyncSession, messages: list[Message]
    ) -> int:
        texts = [m.content for m in messages]
        vectors = await self._embed_texts(texts)
        return await self._save_embeddings(
            session,
            message_ids=[m.id for m in messages],
            vectors=vectors,
        )

    async def _save_embeddings(
        self,
        session: AsyncSession,
        message_ids: Sequence[uuid.UUID],
        vectors: Sequence[list[float]],
    ) -> int:
        """Bulk UPDATE via UPDATE ... FROM (VALUES ...) — one statement per batch."""
        updated = 0
        for batch in chunked(zip(message_ids, vectors), self._insert_batch_size):
            batch = list(batch)
            placeholders = ", ".join(
                f"(:id_{i}::uuid, :vec_{i}::vector)"
                for i in range(len(batch))
            )
            params: dict = {}
            for i, (msg_id, vec) in enumerate(batch):
                params[f"id_{i}"] = str(msg_id)
                params[f"vec_{i}"] = "[" + ",".join(str(v) for v in vec) + "]"

            stmt = text(
                f"""
                UPDATE messages
                SET embedding = v.embedding
                FROM (VALUES {placeholders}) AS v(id, embedding)
                WHERE messages.id = v.id
                """
            )
            await session.execute(stmt, params)
            await session.commit()
            updated += len(batch)
            logger.debug("Saved embedding batch of %d", len(batch))

        logger.info("Updated embeddings for %d messages", updated)
        return updated

    @staticmethod
    async def _fetch_by_ids(
        session: AsyncSession,
        message_ids: Sequence[uuid.UUID],
        force_reembed: bool,
    ) -> list[Message]:
        stmt = select(Message).where(Message.id.in_(message_ids))
        if not force_reembed:
            stmt = stmt.where(Message.embedding.is_(None))
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def _fetch_without_embedding(
        session: AsyncSession,
        user_id: str | None,
        limit: int,
        offset: int,
    ) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.embedding.is_(None))
            .order_by(Message.id)
            .limit(limit)
            .offset(offset)
        )
        if user_id is not None:
            stmt = stmt.where(Message.user_id == user_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())
