"""REV-07 (Sprint A.5): conversation-history must enforce thread ownership.

GET /sessions/{id}/history took a thread_id and returned that thread's stored
messages with no check that the thread belongs to the caller — any authenticated
user could read any other user's conversation by supplying their thread_id
(IDOR). The owner is recorded in the checkpoint state (channel_values.user_id).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from services.orchestrator_service.src.api import get_conversation_history
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser


def _user(uid: str) -> AuthenticatedUser:
    return AuthenticatedUser(user_id=uid, token_type=TokenType.ACCESS)


def _graph_builder_owned_by(owner_id: str):
    checkpoint = {
        "channel_values": {
            "user_id": owner_id,
            "messages": [{"role": "user", "content": "hi"}],
        }
    }
    checkpointer = MagicMock()
    checkpointer.get = MagicMock(return_value=checkpoint)
    gb = MagicMock()
    gb.get_checkpointer = MagicMock(return_value=checkpointer)
    return gb


@pytest.mark.asyncio
async def test_history_owner_can_read() -> None:
    gb = _graph_builder_owned_by("owner-1")
    resp = await get_conversation_history(
        session_id="s1",
        thread_id="t1",
        limit=50,
        current_user=_user("owner-1"),
        graph_builder=gb,
    )
    assert resp.message_count == 1


@pytest.mark.asyncio
async def test_history_other_user_forbidden() -> None:
    """IDOR: a different user must not read another user's thread history."""
    gb = _graph_builder_owned_by("owner-1")
    with pytest.raises(HTTPException) as exc:
        await get_conversation_history(
            session_id="s1",
            thread_id="t1",
            limit=50,
            current_user=_user("attacker-2"),
            graph_builder=gb,
        )
    assert exc.value.status_code == 403
