from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

_EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class File(Base):
    """Registry of uploaded files. Binary content lives on the local volume at storage_path.

    Lifecycle (status field):
      pending   → file saved to storage, not yet sent to chunker
      upserting → request sent to qdrant-ingester
      indexed   → qdrant-ingester finished successfully
      error     → processing failed; see error_message for details

    inline_text:
      Set only when the file is small enough to fit within INLINE_THRESHOLD tokens.
      In that case the full document text is stored here and Qdrant upsert is skipped.
      NULL means the file was chunked and indexed in Qdrant normally.

    Summary fields (summary_status lifecycle):
      pending → summary task has been enqueued but not yet completed
      ready   → summary_text and summary_keywords are populated
      failed  → summarization failed; original ingestion is unaffected
      NULL    → summarization is disabled (SUMMARIZATION_MODEL not set)
               or the file was small enough to be inlined
    """
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    original_name: Mapped[str] = mapped_column(String(512))
    mime_type: Mapped[str] = mapped_column(String(128))
    storage_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    inline_text: Mapped[str | None] = mapped_column(Text, nullable=True)  # set for small files only
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # --- Document summary (populated asynchronously after Qdrant ingestion) ---
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_keywords: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    summary_status: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    summary_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    summary_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    messages: Mapped[list["Message"]] = relationship(back_populates="file", lazy="select")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    role: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="SET NULL"), nullable=True, index=True
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    file: Mapped["File | None"] = relationship(back_populates="messages", lazy="select")


class TraceRecord(Base):
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    step_name: Mapped[str] = mapped_column(String(128), index=True)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    output_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    route_decision: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    input_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    tool_names: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
