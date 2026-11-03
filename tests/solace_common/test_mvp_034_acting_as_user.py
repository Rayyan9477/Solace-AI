"""MVP-034 (supporting; WS-14 closes it): request-less work must be able to act as a user for RLS."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from solace_common.request_context import (
    acting_as_user,
    get_current_user_id,
    reset_current_user_id,
    set_current_user_id,
)


def test_acting_as_user_sets_then_restores_previous_value():
    reset_current_user_id()
    outer, inner = uuid4(), uuid4()
    set_current_user_id(str(outer))
    with acting_as_user(inner) as active:
        assert active == str(inner)
        assert get_current_user_id() == str(inner)
    assert get_current_user_id() == str(outer)
    reset_current_user_id()


def test_acting_as_user_restores_on_exception():
    reset_current_user_id()
    with pytest.raises(RuntimeError):
        with acting_as_user(uuid4()):
            raise RuntimeError("boom")
    assert get_current_user_id() is None


def test_acting_as_user_accepts_uuid_strings_and_none():
    reset_current_user_id()
    uid = str(uuid4()).upper()
    with acting_as_user(uid) as active:
        assert active == uid.lower()
        assert get_current_user_id() == uid.lower()
    with acting_as_user(None) as active:
        assert active is None
        assert get_current_user_id() is None


@pytest.mark.parametrize("bad", ["", "user-1", "1; DROP TABLE users", 12345])
def test_acting_as_user_rejects_non_uuid(bad):
    reset_current_user_id()
    with pytest.raises(ValueError):
        with acting_as_user(bad):
            pass
    assert get_current_user_id() is None


def test_acting_as_user_rejects_non_string_non_uuid_types():
    """An int whose decimal digits form 32 hex characters must not become a different UUID (review finding)."""
    reset_current_user_id()
    with pytest.raises(ValueError):
        with acting_as_user(12345678901234567890123456789012):
            pass
    assert get_current_user_id() is None


@pytest.mark.asyncio
async def test_acting_as_user_is_isolated_between_concurrent_tasks():
    reset_current_user_id()
    a, b = uuid4(), uuid4()
    seen: dict[str, str | None] = {}

    async def work(label, uid):
        with acting_as_user(uid):
            await asyncio.sleep(0.01)
            seen[label] = get_current_user_id()

    await asyncio.gather(work("a", a), work("b", b))
    assert seen == {"a": str(a), "b": str(b)}
    assert get_current_user_id() is None
