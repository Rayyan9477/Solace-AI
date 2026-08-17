"""C.2 — request-scoped current-user contextvar (RLS foundation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from solace_common.request_context import (
    get_current_user_id,
    reset_current_user_id,
    set_current_user_id,
)


def test_default_is_none():
    reset_current_user_id()
    assert get_current_user_id() is None


def test_set_and_get():
    reset_current_user_id()
    set_current_user_id("user-1")
    assert get_current_user_id() == "user-1"
    reset_current_user_id()


def test_coerces_to_str():
    reset_current_user_id()
    set_current_user_id(12345)
    assert get_current_user_id() == "12345"
    reset_current_user_id()


def test_reset_with_token_restores_previous():
    reset_current_user_id()
    set_current_user_id("outer")
    tok = set_current_user_id("inner")
    assert get_current_user_id() == "inner"
    reset_current_user_id(tok)
    assert get_current_user_id() == "outer"
    reset_current_user_id()
