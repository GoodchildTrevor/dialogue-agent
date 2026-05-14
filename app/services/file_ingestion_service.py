from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import File
from app.db.session import get_session_maker
from app.services.pg_ingester import IngesterService
from app.services.qdrant_ingester_client import QdrantIngesterClient

logger = logging.getLogger(__name__)


class FileIngestionService:
    """Main pipeline for processing uploaded files.

    Handles the complete workflow from file upload to indexing in Qdrant:
    1. Load file record from PG
    2. Update status to 'upserting'
    3. Send to qdrant-ingester (chunking + embedding happens there)
       - If INLINE_THRESHOLD is set and document is small enough,
         qdrant-ingester returns raw text instead of ingesting into Qdrant.
         In that case inline_text is saved to the DB for direct LLM injection.
    4. Update status to 'indexed'
    On exception: Update status to 'error' with error message
    """

    def __init__(
        self,
        pg_ingester: IngesterService,
        qdrant_ingester: QdrantIngesterClient,
        settings: Settings,
    ) -> None:
        self.pg_ingester = pg_ingester
        self.qdrant_ingester = qdrant_ingester
        self.settings = settings

    async def process(self, file_id: uuid.UUID) -> None:
        """Process a file through the complete ingestion pipeline."""
        logger.info("Starting file processing for file_id=%s", file_id)

        db_file: File | None = None

        try:
            async with get_session_maker()() as session:
                db_file = await self._get_file_record(session, file_id)
                if not db_file:
                    logger.error("File not found: %s", file_id)
                    return
                await self._update_file_status(session, file_id, "upserting")

            storage_root = Path(self.settings.UPLOAD_STORAGE_DIR).resolve()
            abs_path = Path(db_file.storage_path).resolve()
            relative_path = abs_path.relative_to(storage_root)
            logger.debug(
                "Resolved file path: abs=%s, relative=%s, storage_root=%s",
                abs_path,
                relative_path,
                storage_root,
            )

            ingest_response = await self.qdrant_ingester.ingest(
                collection=self.settings.QDRANT_COLLECTION_DOCS,
                file_path=str(relative_path),
                chunk_size=self.settings.CHUNK_SIZE,
                overlap=self.settings.OVERLAP,
                extra_payload={"user_id": db_file.user_id},
                inline_threshold=self.settings.INLINE_THRESHOLD,
            )

            if ingest_response.get("status") == "failed":
                raise RuntimeError(
                    ingest_response.get("message", "Ingestion failed")
                )

            inline_text: str | None = ingest_response.get("inline_text") or None

            if inline_text:
                logger.info(
                    "File %s is inline (%d tokens), saving text to DB, skipping Qdrant upsert.",
                    file_id,
                    ingest_response.get("token_count", 0),
                )

            async with get_session_maker()() as session:
                await self._update_file_status(
                    session, file_id, "indexed", inline_text=inline_text
                )

        except Exception as exc:
            logger.exception("Failed to process file_id=%s: %s", file_id, exc)
            async with get_session_maker()() as session:
                await self._update_file_status(
                    session, file_id, "error", error_message=str(exc)
                )
            raise

    async def _get_file_record(self, session: AsyncSession, file_id: uuid.UUID) -> File | None:
        """Load file record from database."""
        stmt = select(File).where(File.id == file_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _update_file_status(
        self,
        session: AsyncSession,
        file_id: uuid.UUID,
        status: str,
        error_message: str | None = None,
        inline_text: str | None = None,
    ) -> None:
        """Update file status (and optionally inline_text) in database."""
        values: dict = {"status": status, "error_message": error_message}
        if inline_text is not None:
            values["inline_text"] = inline_text
        stmt = (
            update(File)
            .where(File.id == file_id)
            .values(**values)
        )
        await session.execute(stmt)
        await session.commit()
        logger.info("Updated file %s status to %s", file_id, status)
