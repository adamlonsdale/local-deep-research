"""Durable LangGraph checkpoints stored in each encrypted user database."""

from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy_utc import UtcDateTime, utcnow

from .base import Base


class AgentCheckpoint(Base):
    """A serialized LangGraph checkpoint for one research agent thread."""

    __tablename__ = "agent_checkpoints"

    thread_id = Column(
        String(36),
        ForeignKey("research_history.id", ondelete="CASCADE"),
        primary_key=True,
    )
    checkpoint_ns = Column(String(255), primary_key=True, default="")
    checkpoint_id = Column(String(128), primary_key=True)
    parent_checkpoint_id = Column(String(128))
    checkpoint_type = Column(String(64), nullable=False)
    checkpoint_blob = Column(LargeBinary, nullable=False)
    metadata_type = Column(String(64), nullable=False)
    metadata_blob = Column(LargeBinary, nullable=False)
    created_at = Column(UtcDateTime, nullable=False, default=utcnow())

    __table_args__ = (
        Index(
            "ix_agent_checkpoints_thread_latest",
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        ),
    )


class AgentCheckpointBlob(Base):
    """Serialized channel value for a particular LangGraph version."""

    __tablename__ = "agent_checkpoint_blobs"

    thread_id = Column(
        String(36),
        ForeignKey("research_history.id", ondelete="CASCADE"),
        primary_key=True,
    )
    checkpoint_ns = Column(String(255), primary_key=True, default="")
    channel = Column(String(255), primary_key=True)
    version = Column(String(128), primary_key=True)
    value_type = Column(String(64), nullable=False)
    value_blob = Column(LargeBinary, nullable=False)


class AgentCheckpointWrite(Base):
    """Pending LangGraph task write associated with a checkpoint."""

    __tablename__ = "agent_checkpoint_writes"

    thread_id = Column(
        String(36),
        ForeignKey("research_history.id", ondelete="CASCADE"),
        primary_key=True,
    )
    checkpoint_ns = Column(String(255), primary_key=True, default="")
    checkpoint_id = Column(String(128), primary_key=True)
    task_id = Column(String(128), primary_key=True)
    write_index = Column(Integer, primary_key=True)
    channel = Column(String(255), nullable=False)
    value_type = Column(String(64), nullable=False)
    value_blob = Column(LargeBinary, nullable=False)
    task_path = Column(Text, nullable=False, default="")

    __table_args__ = (
        Index(
            "ix_agent_checkpoint_writes_checkpoint",
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        ),
    )
