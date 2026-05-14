"""baseline: stamp existing schema so Alembic has a known starting point.

All tables were previously created by postgres/init/01-init.sql and direct
create_all calls. This migration is a no-op on a freshly-init'd DB (tables
already exist) but gives Alembic a revision anchor so that 0002+ can apply
incremental DDL changes.

Revision ID: 0001
Revises: (none)
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create extensions required by the schema (idempotent).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------ files
    op.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     VARCHAR(255) NOT NULL,
            original_name VARCHAR(512) NOT NULL,
            mime_type   VARCHAR(128) NOT NULL,
            storage_path TEXT NOT NULL,
            size_bytes  INTEGER,
            status      VARCHAR(32) NOT NULL DEFAULT 'pending',
            error_message TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_files_user_id  ON files (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_files_status   ON files (status)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_files_created_at ON files (created_at)")

    # --------------------------------------------------------------- messages
    op.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id     VARCHAR(255) NOT NULL,
            role        VARCHAR(32) NOT NULL,
            content     TEXT NOT NULL,
            file_id     UUID REFERENCES files(id) ON DELETE SET NULL,
            metadata_json JSONB,
            embedding   vector(1024),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_messages_user_id    ON messages (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_messages_role        ON messages (role)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_messages_file_id     ON messages (file_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_messages_created_at  ON messages (created_at)")

    # ----------------------------------------------------------------- traces
    op.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id         VARCHAR(255) NOT NULL,
            request_id      VARCHAR(64) NOT NULL,
            step_name       VARCHAR(128) NOT NULL,
            input_text      TEXT,
            output_text     TEXT,
            input_payload   JSONB,
            output_payload  JSONB,
            latency_ms      INTEGER NOT NULL,
            estimated_tokens INTEGER,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            route_decision  VARCHAR(32),
            model_used      VARCHAR(128),
            rejection_reason VARCHAR(64),
            input_hash      VARCHAR(64),
            tool_names      TEXT[]
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_user_id        ON traces (user_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_request_id     ON traces (request_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_step_name      ON traces (step_name)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_created_at     ON traces (created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_route_decision ON traces (route_decision)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_model_used     ON traces (model_used)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_rejection_reason ON traces (rejection_reason)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_traces_input_hash     ON traces (input_hash)")


def downgrade() -> None:
    # Dropping all baseline tables is destructive — intentionally left as no-op.
    pass
