"""LangGraph checkpoint saver backed by a user's encrypted SQLCipher DB."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

from .models import (
    AgentCheckpoint,
    AgentCheckpointBlob,
    AgentCheckpointWrite,
)
from .session_context import get_user_db_session


def _typed_parts(value: tuple[str, bytes]) -> tuple[str, bytes]:
    value_type, value_blob = value
    return str(value_type), bytes(value_blob)


class EncryptedAgentCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph state in the already-encrypted per-user database.

    The public LangGraph SQLite saver is a separate optional dependency and
    would create a plaintext sidecar database. This implementation uses the
    application's SQLAlchemy session boundary instead, so checkpoints inherit
    the same SQLCipher encryption, password lifecycle, backups, and WAL rules
    as the rest of the user's research data.
    """

    def __init__(self, username: str, *, session_factory=None, serde=None):
        super().__init__(serde=serde)
        self.username = username
        self._session_factory = session_factory

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        return get_user_db_session(self.username)

    @staticmethod
    def has_checkpoint(db_session, thread_id: str) -> bool:
        return (
            db_session.query(AgentCheckpoint)
            .filter_by(thread_id=thread_id)
            .first()
            is not None
        )

    def _load_blobs(self, db_session, row, checkpoint: dict) -> dict:
        values = {}
        versions = checkpoint.get("channel_versions", {})
        for channel, version in versions.items():
            blob = db_session.get(
                AgentCheckpointBlob,
                {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "channel": channel,
                    "version": str(version),
                },
            )
            if blob is None or blob.value_type == "empty":
                continue
            values[channel] = self.serde.loads_typed(
                (blob.value_type, bytes(blob.value_blob))
            )
        return values

    def _tuple_from_row(self, db_session, row) -> CheckpointTuple:
        checkpoint = self.serde.loads_typed(
            (row.checkpoint_type, bytes(row.checkpoint_blob))
        )
        writes = (
            db_session.query(AgentCheckpointWrite)
            .filter_by(
                thread_id=row.thread_id,
                checkpoint_ns=row.checkpoint_ns,
                checkpoint_id=row.checkpoint_id,
            )
            .order_by(
                AgentCheckpointWrite.task_id,
                AgentCheckpointWrite.write_index,
            )
            .all()
        )
        config = {
            "configurable": {
                "thread_id": row.thread_id,
                "checkpoint_ns": row.checkpoint_ns,
                "checkpoint_id": row.checkpoint_id,
            }
        }
        parent_config = None
        if row.parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": row.thread_id,
                    "checkpoint_ns": row.checkpoint_ns,
                    "checkpoint_id": row.parent_checkpoint_id,
                }
            }
        return CheckpointTuple(
            config=config,
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(
                    db_session, row, checkpoint
                ),
            },
            metadata=self.serde.loads_typed(
                (row.metadata_type, bytes(row.metadata_blob))
            ),
            parent_config=parent_config,
            pending_writes=[
                (
                    write.task_id,
                    write.channel,
                    self.serde.loads_typed(
                        (write.value_type, bytes(write.value_blob))
                    ),
                )
                for write in writes
            ],
        )

    def get_tuple(self, config) -> CheckpointTuple | None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        with self._session() as db_session:
            query = db_session.query(AgentCheckpoint).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
            )
            if checkpoint_id:
                row = query.filter_by(checkpoint_id=checkpoint_id).first()
            else:
                row = query.order_by(
                    AgentCheckpoint.checkpoint_id.desc()
                ).first()
            return self._tuple_from_row(db_session, row) if row else None

    def list(
        self,
        config,
        *,
        filter: dict[str, Any] | None = None,
        before=None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        with self._session() as db_session:
            query = db_session.query(AgentCheckpoint)
            if config is not None:
                configurable = config["configurable"]
                query = query.filter_by(thread_id=configurable["thread_id"])
                if "checkpoint_ns" in configurable:
                    query = query.filter_by(
                        checkpoint_ns=configurable["checkpoint_ns"]
                    )
                if checkpoint_id := get_checkpoint_id(config):
                    query = query.filter_by(checkpoint_id=checkpoint_id)
            if before is not None and (
                before_id := get_checkpoint_id(before)
            ):
                query = query.filter(
                    AgentCheckpoint.checkpoint_id < before_id
                )
            rows = query.order_by(
                AgentCheckpoint.checkpoint_id.desc()
            ).all()
            tuples = []
            for row in rows:
                item = self._tuple_from_row(db_session, row)
                if filter and not all(
                    item.metadata.get(key) == value
                    for key, value in filter.items()
                ):
                    continue
                tuples.append(item)
                if limit is not None and len(tuples) >= limit:
                    break
        yield from tuples

    def put(self, config, checkpoint, metadata, new_versions):
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_copy = checkpoint.copy()
        channel_values = checkpoint_copy.pop("channel_values")
        checkpoint_type, checkpoint_blob = _typed_parts(
            self.serde.dumps_typed(checkpoint_copy)
        )
        metadata_type, metadata_blob = _typed_parts(
            self.serde.dumps_typed(
                get_checkpoint_metadata(config, metadata)
            )
        )

        with self._session() as db_session:
            for channel, version in new_versions.items():
                key = {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": str(version),
                }
                row = db_session.get(AgentCheckpointBlob, key)
                if channel in channel_values:
                    value_type, value_blob = _typed_parts(
                        self.serde.dumps_typed(channel_values[channel])
                    )
                else:
                    value_type, value_blob = "empty", b""
                if row is None:
                    db_session.add(
                        AgentCheckpointBlob(
                            **key,
                            value_type=value_type,
                            value_blob=value_blob,
                        )
                    )
                else:
                    row.value_type = value_type
                    row.value_blob = value_blob

            key = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
            row = db_session.get(AgentCheckpoint, key)
            if row is None:
                row = AgentCheckpoint(**key)
                db_session.add(row)
            row.parent_checkpoint_id = configurable.get("checkpoint_id")
            row.checkpoint_type = checkpoint_type
            row.checkpoint_blob = checkpoint_blob
            row.metadata_type = metadata_type
            row.metadata_blob = metadata_blob
            db_session.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config["configurable"]
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = configurable["checkpoint_id"]
        with self._session() as db_session:
            for index, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, index)
                key = {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "write_index": write_index,
                }
                row = db_session.get(AgentCheckpointWrite, key)
                if row is not None and write_index >= 0:
                    continue
                value_type, value_blob = _typed_parts(
                    self.serde.dumps_typed(value)
                )
                if row is None:
                    row = AgentCheckpointWrite(**key)
                    db_session.add(row)
                row.channel = channel
                row.value_type = value_type
                row.value_blob = value_blob
                row.task_path = task_path
            db_session.commit()

    def delete_thread(self, thread_id: str) -> None:
        with self._session() as db_session:
            for model in (
                AgentCheckpointWrite,
                AgentCheckpointBlob,
                AgentCheckpoint,
            ):
                db_session.query(model).filter_by(thread_id=thread_id).delete(
                    synchronize_session=False
                )
            db_session.commit()

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config,
        *,
        filter: dict[str, Any] | None = None,
        before=None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(
            config, filter=filter, before=before, limit=limit
        ):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)


def delete_research_checkpoints(username: str, research_id: str) -> None:
    """Remove durable agent state after a research becomes terminal."""
    EncryptedAgentCheckpointSaver(username).delete_thread(research_id)
