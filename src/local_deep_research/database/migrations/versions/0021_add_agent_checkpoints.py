"""Add encrypted LangGraph checkpoint tables.

Revision ID: 0021
Revises: 0020
Create Date: 2026-07-12
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy_utc import UtcDateTime


revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade():
    inspector = inspect(op.get_bind())

    if not inspector.has_table("agent_checkpoints"):
        op.create_table(
            "agent_checkpoints",
            sa.Column(
                "thread_id",
                sa.String(36),
                sa.ForeignKey("research_history.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "checkpoint_ns",
                sa.String(255),
                primary_key=True,
                nullable=False,
                server_default="",
            ),
            sa.Column("checkpoint_id", sa.String(128), primary_key=True),
            sa.Column("parent_checkpoint_id", sa.String(128)),
            sa.Column("checkpoint_type", sa.String(64), nullable=False),
            sa.Column("checkpoint_blob", sa.LargeBinary(), nullable=False),
            sa.Column("metadata_type", sa.String(64), nullable=False),
            sa.Column("metadata_blob", sa.LargeBinary(), nullable=False),
            sa.Column("created_at", UtcDateTime(), nullable=False),
        )
        op.create_index(
            "ix_agent_checkpoints_thread_latest",
            "agent_checkpoints",
            ["thread_id", "checkpoint_ns", "checkpoint_id"],
        )

    if not inspector.has_table("agent_checkpoint_blobs"):
        op.create_table(
            "agent_checkpoint_blobs",
            sa.Column(
                "thread_id",
                sa.String(36),
                sa.ForeignKey("research_history.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "checkpoint_ns",
                sa.String(255),
                primary_key=True,
                nullable=False,
                server_default="",
            ),
            sa.Column("channel", sa.String(255), primary_key=True),
            sa.Column("version", sa.String(128), primary_key=True),
            sa.Column("value_type", sa.String(64), nullable=False),
            sa.Column("value_blob", sa.LargeBinary(), nullable=False),
        )

    if not inspector.has_table("agent_checkpoint_writes"):
        op.create_table(
            "agent_checkpoint_writes",
            sa.Column(
                "thread_id",
                sa.String(36),
                sa.ForeignKey("research_history.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "checkpoint_ns",
                sa.String(255),
                primary_key=True,
                nullable=False,
                server_default="",
            ),
            sa.Column("checkpoint_id", sa.String(128), primary_key=True),
            sa.Column("task_id", sa.String(128), primary_key=True),
            sa.Column("write_index", sa.Integer(), primary_key=True),
            sa.Column("channel", sa.String(255), nullable=False),
            sa.Column("value_type", sa.String(64), nullable=False),
            sa.Column("value_blob", sa.LargeBinary(), nullable=False),
            sa.Column(
                "task_path", sa.Text(), nullable=False, server_default=""
            ),
        )
        op.create_index(
            "ix_agent_checkpoint_writes_checkpoint",
            "agent_checkpoint_writes",
            ["thread_id", "checkpoint_ns", "checkpoint_id"],
        )


def downgrade():
    inspector = inspect(op.get_bind())
    for table in (
        "agent_checkpoint_writes",
        "agent_checkpoint_blobs",
        "agent_checkpoints",
    ):
        if inspector.has_table(table):
            op.drop_table(table)
