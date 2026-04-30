"""fix embedding column type: text -> vector(1024)

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-30

The initial migration created messages.embedding as sa.Text by mistake.
This revision converts the column to pgvector's vector(1024) type,
matching the Qwen3-Embedding-0.6B output dimension.

Existing NULL rows are unaffected. Any text rows that are valid vector
literals (e.g. '[0.1,0.2,...]') will be cast automatically.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DIM = 1024


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.alter_column(
        "messages",
        "embedding",
        type_=Vector(_DIM),
        postgresql_using=f"embedding::vector({_DIM})",
    )


def downgrade() -> None:
    import sqlalchemy as sa

    op.alter_column(
        "messages",
        "embedding",
        type_=sa.Text,
        postgresql_using="embedding::text",
    )
