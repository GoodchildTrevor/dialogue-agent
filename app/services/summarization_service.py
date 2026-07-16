from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import update

from app.core.config import Settings
from app.core.llm_client import LLMClient
from app.db.session import get_session_maker
from app.db.models import File

logger = logging.getLogger(__name__)


class SummarizationService:
    """Background service that summarizes large file content via LiteLLM.

    Should be called as an asyncio background task after successful
    Qdrant ingestion. Failures are logged and stored as summary_status='failed'
    without affecting the ingestion pipeline.

    :param llm_client: Shared LLMClient instance.
    :param settings: Application settings.
    """

    def __init__(self, llm_client: LLMClient, settings: Settings) -> None:
        self._llm = llm_client
        self._settings = settings

    async def summarize_file(
        self,
        file_id: uuid.UUID,
        full_text: str,
    ) -> None:
        """Summarize full_text and persist result into the File record.

        :param file_id: UUID of the File row to update.
        :param full_text: Extracted text content of the file (may be very long).
        """
        model = self._settings.SUMMARIZATION_MODEL
        if not model:
            return

        max_chars = self._settings.SUMMARIZATION_MAX_INPUT_CHARS
        text = full_text[:max_chars] if len(full_text) > max_chars else full_text

        logger.info(
            "summarize_file: starting for file_id=%s model=%s input_chars=%d",
            file_id,
            model,
            len(text),
        )

        try:
            result = await self._llm.summarize(text=text, model=model)
            await self._save_summary(
                file_id=file_id,
                summary_text=result["summary_text"],
                summary_keywords=result["summary_keywords"],
                status="ready",
                model=model,
            )
            logger.info("summarize_file: done for file_id=%s", file_id)
        except Exception as exc:
            logger.exception(
                "summarize_file: failed for file_id=%s: %s", file_id, exc
            )
            await self._save_summary(
                file_id=file_id,
                summary_text=None,
                summary_keywords=None,
                status="failed",
                model=model,
            )

    async def _save_summary(
        self,
        file_id: uuid.UUID,
        summary_text: str | None,
        summary_keywords: list[str] | None,
        status: str,
        model: str,
    ) -> None:
        values: dict = {
            "summary_status": status,
            "summary_model": model,
            "summary_updated_at": datetime.now(timezone.utc),
        }
        if summary_text is not None:
            values["summary_text"] = summary_text
        if summary_keywords is not None:
            values["summary_keywords"] = summary_keywords

        async with get_session_maker()() as session:
            stmt = (
                update(File)
                .where(File.id == file_id)
                .values(**values)
            )
            await session.execute(stmt)
            await session.commit()
        logger.debug("summarize_file: saved summary_status=%s for file_id=%s", status, file_id)
