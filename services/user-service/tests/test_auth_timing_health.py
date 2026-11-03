"""P2-4 / P2-5 / P2-11: auth timing side-channels + anti-email-bomb throttle.

- P2-11: /auth/login must equalize response latency across the "unknown account",
  "wrong password", and "account blocked (suspended / unverified)" branches by running a
  constant-time dummy password verify on the branches that would otherwise return before
  ever touching the password hasher. Otherwise an attacker distinguishes registered vs
  unregistered (and active vs blocked) addresses by timing alone.
- P2-4: /auth/resend-verification must bound how many verification emails a SINGLE
  address can trigger (per-account throttle), independent of the per-IP gateway bucket —
  a distributed bomb aimed at one victim address is not stopped by per-IP limits. The
  endpoint stays enumeration-safe: over-limit requests still return 204, they just skip
  the send.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api import (
    LoginRequest,
    ResendVerificationRequest,
    login,
    resend_verification,
)
from src.domain.entities import User
from src.domain.value_objects import AccountStatus


class _RecordingPasswordService:
    """Counts hash/verify calls so a test can prove the dummy verify actually ran."""

    def __init__(self) -> None:
        self.verify_calls = 0
        self.hash_calls = 0

    def hash_password(self, password: str) -> str:
        self.hash_calls += 1
        return "$dummy$hash"

    def verify_password(self, password: str, hashed: str):
        self.verify_calls += 1
        return SimpleNamespace(
            is_valid=False, needs_rehash=False, new_hash=None, algorithm_used=None
        )


class _Repo:
    def __init__(self, user: User | None) -> None:
        self._user = user

    async def get_by_email(self, email: str) -> User | None:
        return self._user


@pytest.mark.asyncio
async def test_login_unknown_account_runs_dummy_verify() -> None:
    """Regression (already-fixed path): a missing account still burns a password verify."""
    ps = _RecordingPasswordService()
    repo = _Repo(None)
    req = LoginRequest(email="nobody@example.com", password="Whatever1")

    with pytest.raises(HTTPException) as ei:
        await login(
            req,
            request=object(),  # unused before the raise
            jwt_service=None,
            password_service=ps,  # type: ignore[arg-type]
            user_repository=repo,  # type: ignore[arg-type]
            user_service=None,
            session_manager=None,
        )
    assert ei.value.status_code == 401
    assert ps.verify_calls >= 1, "not-found path must run a constant-time dummy verify"


@pytest.mark.asyncio
async def test_login_blocked_account_runs_dummy_verify() -> None:
    """A suspended/blocked account must NOT short-circuit before the password hasher —
    otherwise its response is measurably faster than a wrong-password attempt on an
    active account, leaking account state by timing."""
    ps = _RecordingPasswordService()
    user = User(email="blocked@example.com", password_hash="Hashed1a", display_name="B")
    user.status = AccountStatus.SUSPENDED
    repo = _Repo(user)
    req = LoginRequest(email="blocked@example.com", password="Whatever1")

    with pytest.raises(HTTPException) as ei:
        await login(
            req,
            request=object(),
            jwt_service=None,
            password_service=ps,  # type: ignore[arg-type]
            user_repository=repo,  # type: ignore[arg-type]
            user_service=None,
            session_manager=None,
        )
    assert ei.value.status_code == 401
    assert ps.verify_calls >= 1, "blocked path must run a constant-time dummy verify"


@pytest.mark.asyncio
async def test_resend_verification_per_account_throttle_bounds_sends() -> None:
    """P2-4: repeated resend requests for the SAME address are bounded per-account, and
    every request still returns 204 (enumeration-safe — over-limit just skips the send)."""
    import src.api as api

    api._reset_resend_throttle()

    class _Svc:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def resend_verification_email_by_email(self, email: str) -> None:
            self.calls.append(email)

    svc = _Svc()
    req = ResendVerificationRequest(email="victim@example.com")

    attempts = api._RESEND_THROTTLE_MAX + 5
    for _ in range(attempts):
        result = await resend_verification(req, svc)  # type: ignore[arg-type]
        assert result is None  # always 204 — no enumeration signal

    assert len(svc.calls) == api._RESEND_THROTTLE_MAX


@pytest.mark.asyncio
async def test_resend_verification_throttle_is_per_account() -> None:
    """The throttle keys on the address, so bombing one victim does not consume another
    account's budget."""
    import src.api as api

    api._reset_resend_throttle()

    class _Svc:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def resend_verification_email_by_email(self, email: str) -> None:
            self.calls.append(email)

    svc = _Svc()
    for _ in range(api._RESEND_THROTTLE_MAX + 3):
        await resend_verification(
            ResendVerificationRequest(email="a@example.com"), svc  # type: ignore[arg-type]
        )
    # A different address still gets its full budget.
    await resend_verification(
        ResendVerificationRequest(email="b@example.com"), svc  # type: ignore[arg-type]
    )
    assert svc.calls.count("a@example.com") == api._RESEND_THROTTLE_MAX
    assert svc.calls.count("b@example.com") == 1
