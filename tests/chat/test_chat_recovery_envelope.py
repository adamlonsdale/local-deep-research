"""Chat creation persists the exact envelope needed for restart replay."""

from contextlib import nullcontext
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from local_deep_research.chat.routes import _spawn_chat_research
from local_deep_research.database.models import (
    Base,
    ChatSession,
    ResearchHistory,
)


def test_chat_spawn_persists_exact_context_before_worker_start():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(
        ChatSession(
            id="chat-1",
            title="Chat",
            status="active",
            accumulated_context={},
        )
    )
    db.commit()

    settings = {
        "llm.provider": {"value": "openai"},
        "llm.model": {"value": "test-model"},
        "search.tool": {"value": "wikipedia"},
        "search.iterations": {"value": 4},
        "search.questions_per_iteration": {"value": 2},
        "search.search_strategy": {"value": "source-based"},
    }
    context = {
        "session_id": "chat-1",
        "is_multi_turn": True,
        "accumulated_findings": "earlier answer",
    }
    service = Mock()
    service.insert_message_in_db.return_value = "message-1"

    with (
        patch(
            "local_deep_research.chat.routes.get_user_db_session",
            return_value=nullcontext(db),
        ),
        patch(
            "local_deep_research.chat.routes.get_user_password",
            return_value="secret",
        ),
        patch(
            "local_deep_research.chat.routes.start_research_process"
        ) as start,
    ):
        research_id, message_id = _spawn_chat_research(
            username="alice",
            session_id="chat-1",
            content="What about her later work?",
            settings_snapshot=settings,
            research_context=context,
            messages=[{"role": "assistant", "content": "earlier answer"}],
            service=service,
        )

    history = db.get(ResearchHistory, research_id)
    submission = history.research_meta["submission"]
    assert message_id == "message-1"
    assert history.research_meta["settings_snapshot"] == settings
    assert submission["chat_message_id"] == "message-1"
    assert submission["strategy"] == "enhanced-contextual-followup"
    assert submission["research_context"] == {
        **context,
        "delegate_strategy": "source-based",
    }
    assert (
        start.call_args.kwargs["research_context"]
        == submission["research_context"]
    )
