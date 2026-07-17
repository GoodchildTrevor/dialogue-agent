"""add file summary columns

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("files", sa.Column("summary_text", sa.Text(), nullable=True))
    op.add_column(
        "files",
        sa.Column(
            "summary_keywords",
            postgresql.ARRAY(sa.String()),
            nullable=True,
        ),
    )
    op.add_column(
        "files",
        sa.Column("summary_status", sa.String(16), nullable=True),
    )
    op.create_index(
        "ix_files_summary_status",
        "files",
        ["summary_status"],
        if_not_exists=True,
    )
    op.add_column(
        "files",
        sa.Column("summary_model", sa.String(128), nullable=True),
    )
    op.add_column(
        "files",
        sa.Column(
            "summary_updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_index("ix_files_summary_status", table_name="files")
    op.drop_column("files", "summary_updated_at")
    op.drop_column("files", "summary_model")
    op.drop_column("files", "summary_status")
    op.drop_column("files", "summary_keywords")
    op.drop_column("files", "summary_text")
