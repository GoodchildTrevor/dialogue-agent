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

            ingest_response = await self.qdrant_ingester.ingest(
                collection=self.settings.QDRANT_COLLECTION_DOCS,
                file_path=str(Path(db_file.storage_path).resolve()),
                chunk_size=self.settings.CHUNK_SIZE,
                overlap=self.settings.OVERLAP,
                extra_payload={"user_id": db_file.user_id},
            )

            if ingest_response.get("status") == "failed":
                raise RuntimeError(
                    ingest_response.get("message", "Ingestion failed")
                )

            async with get_session_maker()() as session:
                await self._update_file_status(session, file_id, "indexed")

        except Exception as exc:
            logger.exception("Failed to process file_id=%s: %s", file_id, exc)
            async with get_session_maker()() as session:
                await self._update_file_status(
                    session, file_id, "error", str(exc)
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
    ) -> None:
        """Update file status in database."""
        stmt = (
            update(File)
            .where(File.id == file_id)
            .values(status=status, error_message=error_message)
        )
        await session.execute(stmt)
        await session.commit()
        logger.info("Updated file %s status to %s", file_id, status)
