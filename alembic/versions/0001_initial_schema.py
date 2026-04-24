"""initial schema

Revision ID: 0001
Revises: 
Create Date: 2026-04-24

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("original_name", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("storage_path", sa.Text, nullable=False),
        sa.Column("size_bytes", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending", index=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("role", sa.String(32), nullable=False, index=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("files.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("metadata_json", postgresql.JSONB, nullable=True),
        sa.Column("embedding", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
    )

    op.create_table(
        "traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False, index=True),
        sa.Column("request_id", sa.String(64), nullable=False, index=True),
        sa.Column("step_name", sa.String(128), nullable=False, index=True),
        sa.Column("input_text", sa.Text, nullable=True),
        sa.Column("output_text", sa.Text, nullable=True),
        sa.Column("input_payload", postgresql.JSONB, nullable=True),
        sa.Column("output_payload", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=False),
        sa.Column("estimated_tokens", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
        sa.Column("route_decision", sa.String(32), nullable=True, index=True),
        sa.Column("model_used", sa.String(128), nullable=True, index=True),
        sa.Column("rejection_reason", sa.String(64), nullable=True, index=True),
        sa.Column("input_hash", sa.String(64), nullable=True, index=True),
        sa.Column("tool_names", postgresql.ARRAY(sa.String), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("traces")
    op.drop_table("messages")
    op.drop_table("files")
