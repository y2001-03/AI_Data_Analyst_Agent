"""initial_schema

Create the three core business tables:

- ``datasets``          — uploaded file metadata
- ``analysis_sessions`` — question-driven analysis runs
- ``analysis_results``  — structured output per session

Revision ID: b33b00a93c3f
Revises:
Create Date: 2026-07-21 14:36:19.307936
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b33b00a93c3f"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── datasets ──────────────────────────────────────────────────
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_path", sa.String(1024), nullable=True),
        sa.Column("file_size", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("file_type", sa.String(16), nullable=False, server_default="csv"),
        sa.Column("row_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("column_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "schema_info",
            postgresql.JSONB,
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ── analysis_sessions ─────────────────────────────────────────
    op.create_table(
        "analysis_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(36),
            sa.ForeignKey("datasets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("question", sa.Text, nullable=True),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("workflow_trace_id", sa.String(36), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # ── analysis_results ──────────────────────────────────────────
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("analysis_sessions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "result",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # ── Indexes ───────────────────────────────────────────────────
    op.create_index("ix_datasets_filename", "datasets", ["filename"])
    op.create_index("ix_datasets_created_at", "datasets", ["created_at"])
    op.create_index("ix_sessions_dataset_id", "analysis_sessions", ["dataset_id"])
    op.create_index("ix_sessions_status", "analysis_sessions", ["status"])
    op.create_index("ix_sessions_created_at", "analysis_sessions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_sessions_created_at")
    op.drop_index("ix_sessions_status")
    op.drop_index("ix_sessions_dataset_id")
    op.drop_index("ix_datasets_created_at")
    op.drop_index("ix_datasets_filename")
    op.drop_table("analysis_results")
    op.drop_table("analysis_sessions")
    op.drop_table("datasets")
