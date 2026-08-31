"""Regression coverage for restart recovery of interrupted research."""

from datetime import UTC, datetime, timedelta
from contextlib import nullcontext
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_deep_research.constants import ResearchStatus
from local_deep_research.database.models import (
    AgentCheckpoint,
    Base,
    ChatMessage,
    ChatMessageType,
    ChatProgressStep,
    ChatRole,
    ChatSession,
    QueuedResearch,
    QueueStatus,
    ResearchHistory,
    ResearchResource,
    TaskMetadata,
    UserActiveResearch,
)
from local_deep_research.database.queue_service import UserQueueService
from local_deep_research.web.queue.processor_v2 import QueueProcessorV2


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _launch_envelope():
    return {
        "submission": {
            "model_provider": "openai",
            "model": "test-model",
            "custom_endpoint": "https://llm.example.test/v1",
            "search_engine": "wikipedia",
            "max_results": 7,
            "time_period": "m",
            "iterations": 4,
            "questions_per_iteration": 3,
            "strategy": "source-based",
        },
        "settings_snapshot": {
            "llm.provider": {"value": "openai"},
            "llm.model": {"value": "test-model"},
            "search.tool": {"value": "wikipedia"},
        },
    }


def test_recovery_requeues_stranded_direct_research_and_restores_launch_args():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    envelope = _launch_envelope()
    db.add(
        ResearchHistory(
            id="research-1",
            query="original query",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            progress=67,
            progress_log=[{"time": old.isoformat(), "progress": 67}],
            report_content="partial output from crashed attempt",
            research_meta=envelope,
        )
    )
    db.add(
        UserActiveResearch(
            username="alice",
            research_id="research-1",
            status=ResearchStatus.IN_PROGRESS,
            started_at=old,
            settings_snapshot=envelope,
        )
    )
    db.add(
        TaskMetadata(
            task_id="research-1",
            task_type="research",
            status="processing",
        )
    )
    db.add(QueueStatus(active_tasks=1, queued_tasks=0))
    db.add(
        ResearchResource(
            research_id="research-1",
            title="partial source",
            url="https://example.test/partial",
            created_at=old.isoformat(),
        )
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 1

    history = db.get(ResearchHistory, "research-1")
    queued = db.query(QueuedResearch).filter_by(research_id="research-1").one()
    task = db.get(TaskMetadata, "research-1")
    queue_status = db.query(QueueStatus).one()

    assert history.status == ResearchStatus.QUEUED
    assert history.progress == 0
    assert history.report_content is None
    assert history.research_meta["resume_count"] == 1
    assert queued.is_processing is False
    assert queued.settings_snapshot["submission"]["model"] == "test-model"
    assert db.get(UserActiveResearch, ("alice", "research-1")) is None
    assert task.status == ResearchStatus.QUEUED
    assert queue_status.active_tasks == 0
    assert queue_status.queued_tasks == 1
    assert (
        db.query(ResearchResource).filter_by(research_id="research-1").count()
        == 0
    )

    class _Thread:
        ident = 12345

    with patch(
        "local_deep_research.web.queue.processor_v2.start_research_process",
        return_value=_Thread(),
    ) as start:
        processor._start_research(db, "alice", "secret", queued)

    args = start.call_args.args
    kwargs = start.call_args.kwargs
    assert args[:3] == ("research-1", "original query", "quick")
    assert kwargs["username"] == "alice"
    assert kwargs["user_password"] == "secret"
    assert kwargs["model_provider"] == "openai"
    assert kwargs["model"] == "test-model"
    assert kwargs["search_engine"] == "wikipedia"
    assert kwargs["max_results"] == 7
    assert kwargs["iterations"] == 4
    assert kwargs["questions_per_iteration"] == 3
    assert kwargs["settings_snapshot"] == envelope["settings_snapshot"]


def test_recovery_uses_history_snapshot_when_active_row_was_never_committed():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="history-only",
            query="query",
            mode="detailed",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 1

    queued = (
        db.query(QueuedResearch).filter_by(research_id="history-only").one()
    )
    assert queued.query == "query"
    assert queued.settings_snapshot["submission"]["strategy"] == "source-based"


def test_langgraph_recovery_preserves_progress_sources_and_checkpoint():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    envelope = _launch_envelope()
    envelope["submission"]["strategy"] = "langgraph-agent"
    db.add(
        ResearchHistory(
            id="agent-research",
            query="query",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            progress=64,
            progress_log=[{"time": old.isoformat(), "progress": 64}],
            research_meta=envelope,
        )
    )
    db.add(
        ResearchResource(
            research_id="agent-research",
            title="durable source",
            url="https://example.test/durable",
            resource_metadata={
                "original_data": {
                    "title": "durable source",
                    "link": "https://example.test/durable",
                    "index": "1",
                }
            },
            created_at=old.isoformat(),
        )
    )
    db.add(
        AgentCheckpoint(
            thread_id="agent-research",
            checkpoint_ns="",
            checkpoint_id="checkpoint-1",
            checkpoint_type="msgpack",
            checkpoint_blob=b"checkpoint",
            metadata_type="msgpack",
            metadata_blob=b"metadata",
            created_at=old,
        )
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 1

    history = db.get(ResearchHistory, "agent-research")
    queued = (
        db.query(QueuedResearch)
        .filter_by(research_id="agent-research")
        .one()
    )
    assert history.progress == 64
    assert history.research_meta["resume_mode"] == "checkpoint"
    assert "latest saved checkpoint" in history.progress_log[-1]["message"]
    assert queued.settings_snapshot["submission"][
        "resume_from_checkpoint"
    ] is True
    assert (
        db.query(ResearchResource)
        .filter_by(research_id="agent-research")
        .count()
        == 1
    )
    assert (
        db.query(AgentCheckpoint)
        .filter_by(thread_id="agent-research")
        .count()
        == 1
    )


def test_recovery_restores_exact_persisted_chat_context():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    raw_settings = _launch_envelope()["settings_snapshot"]
    research_context = {
        "session_id": "chat-1",
        "original_query": "first question",
        "accumulated_findings": "prior answer",
        "past_findings": "prior source",
        "key_entities": ["Ada"],
        "topics": ["computing"],
        "is_multi_turn": True,
        "turn_count": 2,
        "delegate_strategy": "source-based",
    }
    db.add(
        ChatSession(
            id="chat-1",
            title="Chat",
            status="active",
            accumulated_context={},
            message_count=2,
        )
    )
    db.add(
        ResearchHistory(
            id="chat-research",
            query="chat question",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            chat_session_id="chat-1",
            step_count=1,
            research_meta={
                "submission": {
                    "chat_session_id": "chat-1",
                    "message_id": "message-1",
                    "chat_message_id": "message-1",
                    "research_mode": "quick",
                    "strategy": "enhanced-contextual-followup",
                    "research_context": research_context,
                },
                "settings_snapshot": raw_settings,
            },
        )
    )
    db.add(
        UserActiveResearch(
            username="alice",
            research_id="chat-research",
            status=ResearchStatus.IN_PROGRESS,
            started_at=old,
            settings_snapshot=raw_settings,
        )
    )
    db.add_all(
        [
            ChatMessage(
                id="message-1",
                session_id="chat-1",
                research_id=None,
                role=ChatRole.USER,
                message_type=ChatMessageType.FOLLOWUP,
                content="chat question",
                sequence_number=1,
            ),
            ChatMessage(
                id="partial-response",
                session_id="chat-1",
                research_id="chat-research",
                role=ChatRole.ASSISTANT,
                message_type=ChatMessageType.RESPONSE,
                content="partial output",
                sequence_number=2,
            ),
            ChatProgressStep(
                id="step-1",
                research_id="chat-research",
                session_id="chat-1",
                phase="search",
                content="partial step",
                sequence_number=1,
            ),
        ]
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 1

    queued = (
        db.query(QueuedResearch).filter_by(research_id="chat-research").one()
    )
    submission = queued.settings_snapshot["submission"]
    assert submission["chat_session_id"] == "chat-1"
    assert submission["chat_message_id"] == "message-1"
    assert submission["research_context"] == research_context
    assert queued.settings_snapshot["settings_snapshot"] == raw_settings
    assert db.get(ChatMessage, "message-1") is not None
    assert db.get(ChatMessage, "partial-response") is None
    assert db.query(ChatProgressStep).count() == 0
    assert db.get(ResearchHistory, "chat-research").step_count == 0
    assert db.get(ChatSession, "chat-1").message_count == 1


def test_legacy_contextual_row_without_context_fails_instead_of_changing_semantics():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    raw_settings = _launch_envelope()["settings_snapshot"]
    db.add(
        ResearchHistory(
            id="legacy-chat",
            query="referential follow-up",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            chat_session_id="chat-1",
            research_meta={
                "submission": {
                    "chat_session_id": "chat-1",
                    "message_id": "message-1",
                }
            },
        )
    )
    db.add(
        UserActiveResearch(
            username="alice",
            research_id="legacy-chat",
            status=ResearchStatus.IN_PROGRESS,
            started_at=old,
            settings_snapshot=raw_settings,
        )
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 0

    history = db.get(ResearchHistory, "legacy-chat")
    assert history.status == ResearchStatus.FAILED
    assert "context" in history.research_meta["recovery_error"]
    assert db.query(QueuedResearch).count() == 0


def test_recovery_repairs_queue_row_committed_before_task_metadata():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="queued-without-task",
            query="query",
            mode="quick",
            status=ResearchStatus.QUEUED,
            created_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(
        QueuedResearch(
            username="alice",
            research_id="queued-without-task",
            query="query",
            mode="quick",
            settings_snapshot=_launch_envelope(),
            position=1,
            is_processing=False,
        )
    )
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 0

    assert (
        db.get(TaskMetadata, "queued-without-task").status
        == ResearchStatus.QUEUED
    )
    status = db.query(QueueStatus).one()
    assert status.queued_tasks == 1
    assert status.active_tasks == 0


def test_terminal_tracking_ghost_does_not_consume_recovery_capacity():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="finished-before-crash",
            query="done",
            mode="quick",
            status=ResearchStatus.COMPLETED,
            created_at=old.isoformat(),
            completed_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(
        UserActiveResearch(
            username="alice",
            research_id="finished-before-crash",
            status=ResearchStatus.IN_PROGRESS,
            started_at=old,
            settings_snapshot=_launch_envelope(),
        )
    )
    db.add(
        TaskMetadata(
            task_id="finished-before-crash",
            task_type="research",
            status="processing",
        )
    )
    db.add(
        ResearchHistory(
            id="interrupted",
            query="resume me",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(QueueStatus(active_tasks=2, queued_tasks=0))
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 1

    status = db.query(QueueStatus).one()
    assert status.active_tasks == 0
    assert status.queued_tasks == 1
    assert db.get(ResearchHistory, "finished-before-crash").status == (
        ResearchStatus.COMPLETED
    )
    assert (
        db.get(TaskMetadata, "finished-before-crash").status
        == ResearchStatus.COMPLETED
    )
    assert (
        db.get(UserActiveResearch, ("alice", "finished-before-crash")) is None
    )

    # The stale UserActiveResearch row for the completed run must not be
    # counted again by the dispatcher after QueueStatus was repaired.
    with (
        patch(
            "local_deep_research.web.queue.processor_v2.session_password_store.get_session_password",
            return_value="secret",
        ),
        patch(
            "local_deep_research.web.queue.processor_v2.db_manager.open_user_database",
            return_value=object(),
        ),
        patch(
            "local_deep_research.web.queue.processor_v2.get_user_db_session",
            return_value=nullcontext(db),
        ),
        patch(
            "local_deep_research.settings.manager.SettingsManager.get_setting",
            return_value=1,
        ),
        patch.object(processor, "_start_queued_researches") as start_queued,
    ):
        processor._process_user_queue("alice", "session-1")

    start_queued.assert_called_once()


def test_recovery_skips_current_process_row_and_is_idempotent():
    db = _session()
    processor = QueueProcessorV2(check_interval=1)
    fresh = processor._recovery_cutoff + timedelta(seconds=1)
    db.add(
        ResearchHistory(
            id="fresh",
            query="query",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=fresh.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(
        UserActiveResearch(
            username="alice",
            research_id="fresh",
            status=ResearchStatus.IN_PROGRESS,
            started_at=fresh,
            settings_snapshot=_launch_envelope(),
        )
    )
    db.commit()

    with patch(
        "local_deep_research.web.routes.globals.is_research_thread_alive",
        return_value=False,
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 0
        assert processor.recover_interrupted_researches(db, "alice") == 0

    assert db.get(ResearchHistory, "fresh").status == ResearchStatus.IN_PROGRESS
    assert db.query(QueuedResearch).count() == 0


def test_unrecoverable_legacy_row_becomes_failed_instead_of_staying_in_progress():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="missing-launch-data",
            query="query",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            research_meta={},
        )
    )
    db.add(
        TaskMetadata(
            task_id="missing-launch-data",
            task_type="research",
            status="processing",
        )
    )
    db.add(QueueStatus(active_tasks=1, queued_tasks=0))
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with (
        patch(
            "local_deep_research.web.routes.globals.is_research_thread_alive",
            return_value=False,
        ),
        patch.object(processor, "_build_recovery_payload", return_value=None),
    ):
        assert processor.recover_interrupted_researches(db, "alice") == 0

    history = db.get(ResearchHistory, "missing-launch-data")
    status = db.query(QueueStatus).one()
    assert history.status == ResearchStatus.FAILED
    assert history.research_meta["recovery_error"]
    assert (
        db.get(TaskMetadata, "missing-launch-data").status
        == ResearchStatus.FAILED
    )
    assert status.active_tasks == 0
    assert status.queued_tasks == 0


def test_queue_loop_reclaims_processing_row_before_zero_queued_early_exit():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="claimed-before-crash",
            query="query",
            mode="quick",
            status=ResearchStatus.IN_PROGRESS,
            created_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(
        QueuedResearch(
            username="alice",
            research_id="claimed-before-crash",
            query="query",
            mode="quick",
            settings_snapshot=_launch_envelope(),
            position=1,
            is_processing=True,
        )
    )
    db.add(
        TaskMetadata(
            task_id="claimed-before-crash",
            task_type="research",
            status="processing",
        )
    )
    db.add(QueueStatus(active_tasks=1, queued_tasks=0))
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    with (
        patch(
            "local_deep_research.web.queue.processor_v2.session_password_store.get_session_password",
            return_value="secret",
        ),
        patch(
            "local_deep_research.web.queue.processor_v2.db_manager.open_user_database",
            return_value=object(),
        ),
        patch(
            "local_deep_research.web.queue.processor_v2.get_user_db_session",
            return_value=nullcontext(db),
        ),
        patch(
            "local_deep_research.web.routes.globals.is_research_active",
            return_value=False,
        ),
        patch(
            "local_deep_research.settings.manager.SettingsManager.get_setting",
            return_value=3,
        ),
        patch.object(processor, "_start_queued_researches") as start_queued,
    ):
        processor._process_user_queue("alice", "session-1")

    start_queued.assert_called_once()
    queued = (
        db.query(QueuedResearch)
        .filter_by(research_id="claimed-before-crash")
        .one()
    )
    status = db.query(QueueStatus).one()
    assert queued.is_processing is False
    assert (
        db.get(ResearchHistory, "claimed-before-crash").status
        == ResearchStatus.QUEUED
    )
    assert (
        db.get(TaskMetadata, "claimed-before-crash").status
        == ResearchStatus.QUEUED
    )
    assert status.active_tasks == 0
    assert status.queued_tasks == 1


def test_stopped_recovered_queue_does_not_leak_active_capacity():
    db = _session()
    old = datetime.now(UTC) - timedelta(hours=1)
    db.add(
        ResearchHistory(
            id="stopped-after-recovery",
            query="query",
            mode="quick",
            status=ResearchStatus.SUSPENDED,
            created_at=old.isoformat(),
            research_meta=_launch_envelope(),
        )
    )
    db.add(
        QueuedResearch(
            username="alice",
            research_id="stopped-after-recovery",
            query="query",
            mode="quick",
            settings_snapshot=_launch_envelope(),
            position=1,
            is_processing=False,
        )
    )
    db.add(
        TaskMetadata(
            task_id="stopped-after-recovery",
            task_type="research",
            status=ResearchStatus.QUEUED,
        )
    )
    db.add(QueueStatus(active_tasks=0, queued_tasks=1))
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    processor._start_queued_researches(
        db,
        queue_service=UserQueueService(db),
        username="alice",
        password="secret",
        available_slots=1,
    )

    status = db.query(QueueStatus).one()
    task = db.get(TaskMetadata, "stopped-after-recovery")
    assert db.query(QueuedResearch).count() == 0
    assert task.status == ResearchStatus.SUSPENDED
    assert status.active_tasks == 0
    assert status.queued_tasks == 0


def test_deleted_recovered_history_releases_dispatch_claim_for_retry():
    db = _session()
    db.add(
        QueuedResearch(
            username="alice",
            research_id="deleted-after-recovery",
            query="query",
            mode="quick",
            settings_snapshot=_launch_envelope(),
            position=1,
            is_processing=False,
        )
    )
    db.add(
        TaskMetadata(
            task_id="deleted-after-recovery",
            task_type="research",
            status=ResearchStatus.QUEUED,
        )
    )
    db.add(QueueStatus(active_tasks=0, queued_tasks=1))
    db.commit()

    processor = QueueProcessorV2(check_interval=1)
    processor._start_queued_researches(
        db,
        queue_service=UserQueueService(db),
        username="alice",
        password="secret",
        available_slots=1,
    )

    queued = db.query(QueuedResearch).one()
    task = db.get(TaskMetadata, "deleted-after-recovery")
    status = db.query(QueueStatus).one()
    assert queued.is_processing is False
    assert task.status == ResearchStatus.QUEUED
    assert status.active_tasks == 0
    assert status.queued_tasks == 1
