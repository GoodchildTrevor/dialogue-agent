"""add inline_text column to files table.

Stores the full document text for small files (size <= INLINE_THRESHOLD tokens).
When set, the LLM receives the text directly in the prompt and document_searcher
is not called. NULL means the file was chunked and indexed in Qdrant normally.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "files",
        sa.Column("inline_text", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("files", "inline_text")
