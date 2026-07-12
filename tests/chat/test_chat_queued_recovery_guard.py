"""Queued restart recovery must preserve per-chat turn ordering."""

import uuid
from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import patch

from local_deep_research.chat.routes import (
    _enforce_chat_session_research_slot,
)
from local_deep_research.chat.service import ChatService
from local_deep_research.constants import ResearchStatus
from local_deep_research.database.models import ChatSession, ResearchHistory


def test_recovered_queued_turn_occupies_chat_session(
    setup_database_for_all_tests,
):
    SessionLocal = setup_database_for_all_tests
    session_id = str(uuid.uuid4())
    research_id = str(uuid.uuid4())

    with SessionLocal() as db:
        db.add(
            ChatSession(
                id=session_id,
                title="Restarted chat",
                status="active",
                accumulated_context={},
            )
        )
        db.add(
            ResearchHistory(
                id=research_id,
                query="resume this turn",
                mode="quick",
                status=ResearchStatus.QUEUED,
                created_at=datetime.now(UTC).isoformat(),
                chat_session_id=session_id,
            )
        )
        db.commit()

        result = _enforce_chat_session_research_slot(db, "alice", session_id)
        with patch(
            "local_deep_research.chat.service.get_user_db_session",
            return_value=nullcontext(db),
        ):
            active_id_on_reload = ChatService(
                username="alice"
            ).get_in_progress_research_id(session_id)

    assert result is not None
    message, status_code = result
    assert status_code == 409
    assert "already in progress" in message.lower()
    assert active_id_on_reload == research_id
