"""Phase B / API-1 (P1): WebSocket conversation continuity across reconnects.

The /ws endpoint generated a fresh `thread_id = str(uuid4())` on every connection.
LangGraph checkpoints conversation state by thread_id, so a client reconnecting to
the same session got a brand-new thread → the whole conversation history/context was
lost on every reconnect.

The thread_id must be STABLE per (user, session) so reconnects resume — but keyed on
the user too, so a different user presenting the same session_id can NOT resume
another user's conversation (the fresh-uuid was what previously provided that
isolation, per the A.5 WS review).
"""
from __future__ import annotations

from uuid import uuid4

from services.orchestrator_service.src.api import _ws_thread_id


def test_thread_id_is_stable_for_same_user_and_session() -> None:
    """Reconnects (same user + session) resume the same conversation thread."""
    user = str(uuid4())
    session = str(uuid4())
    assert _ws_thread_id(user, session) == _ws_thread_id(user, session)


def test_thread_id_is_isolated_per_user() -> None:
    """A different user with the same session_id must NOT get the same thread (no IDOR resume)."""
    session = str(uuid4())
    assert _ws_thread_id(str(uuid4()), session) != _ws_thread_id(str(uuid4()), session)


def test_thread_id_differs_per_session() -> None:
    user = str(uuid4())
    assert _ws_thread_id(user, str(uuid4())) != _ws_thread_id(user, str(uuid4()))
