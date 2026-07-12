"""Integration tests for encrypted-database LangGraph continuation."""

from contextlib import contextmanager
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_deep_research.database.agent_checkpoint_saver import (
    EncryptedAgentCheckpointSaver,
)
from local_deep_research.database.models import Base, ResearchHistory


class _CounterState(TypedDict):
    value: int


def test_new_saver_instance_continues_interrupted_graph(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'checkpoints.db'}")
    Base.metadata.create_all(engine)
    session_type = sessionmaker(bind=engine)
    with session_type() as session:
        session.add(
            ResearchHistory(
                id="research-1",
                query="query",
                mode="quick",
                status="in_progress",
                created_at="2026-07-12T00:00:00+00:00",
            )
        )
        session.commit()

    @contextmanager
    def session_factory():
        session = session_type()
        try:
            yield session
        finally:
            session.close()

    builder = StateGraph(_CounterState)
    builder.add_node("first", lambda state: {"value": state["value"] + 1})
    builder.add_node("second", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    config = {"configurable": {"thread_id": "research-1"}}

    first_saver = EncryptedAgentCheckpointSaver(
        "alice", session_factory=session_factory
    )
    interrupted = builder.compile(
        checkpointer=first_saver, interrupt_after=["first"]
    )
    assert interrupted.invoke({"value": 0}, config) == {"value": 1}
    assert first_saver.get_tuple(config) is not None

    # A new saver/graph represents a new container process. Supplying None
    # resumes the graph's pending second node instead of replaying the input.
    second_saver = EncryptedAgentCheckpointSaver(
        "alice", session_factory=session_factory
    )
    resumed = builder.compile(checkpointer=second_saver)
    assert resumed.invoke(None, config) == {"value": 2}
    assert len(list(second_saver.list(config))) >= 2

    second_saver.delete_thread("research-1")
    assert second_saver.get_tuple(config) is None
    engine.dispose()
