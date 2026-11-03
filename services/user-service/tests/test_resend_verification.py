"""REV-31: unauthenticated, enumeration-safe resend-verification (deadlock fix).

An unverified production user whose initial verification email failed to send is
blocked from login (prod requires a verified email) and therefore could not reach
the previously auth-gated resend endpoint — a permanent lockout. The resend path is
now reachable by email and enumeration-safe: it regenerates + resends only for a
real, still-unverified account, and is a silent no-op otherwise.
"""
from __future__ import annotations

import pytest

from src.domain.entities import User
from src.domain.service import UserService

from fixtures import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
)


def _service() -> UserService:
    return UserService(
        user_repository=InMemoryUserRepository(),
        preferences_repository=InMemoryUserPreferencesRepository(),
    )


@pytest.mark.asyncio
async def test_resend_by_email_regenerates_and_sends_for_unverified(monkeypatch) -> None:
    svc = _service()
    user = User(email="new@example.com", password_hash="Hashed1a", display_name="New")
    assert user.email_verified is False
    await svc._user_repo.save(user)
    original_token = user.email_verification_token

    sent: list[tuple[str, str]] = []

    async def _fake_send(email, display_name, token):  # noqa: ANN001
        sent.append((email, token))
        return True

    monkeypatch.setattr(svc, "_send_verification_email", _fake_send)

    await svc.resend_verification_email_by_email("new@example.com")

    # A fresh verification email was sent and the token was regenerated.
    assert len(sent) == 1
    assert sent[0][0] == "new@example.com"
    refreshed = await svc._user_repo.get_by_email("new@example.com")
    assert refreshed.email_verification_token is not None
    assert refreshed.email_verification_token != original_token


@pytest.mark.asyncio
async def test_resend_by_email_unknown_address_is_silent_noop(monkeypatch) -> None:
    svc = _service()
    sent: list = []

    async def _fake_send(**kwargs):  # noqa: ANN003
        sent.append(kwargs)
        return True

    monkeypatch.setattr(svc, "_send_verification_email", _fake_send)

    # Must not raise and must not send — no account-enumeration signal.
    await svc.resend_verification_email_by_email("nobody@example.com")
    assert sent == []


@pytest.mark.asyncio
async def test_resend_by_email_already_verified_is_noop(monkeypatch) -> None:
    svc = _service()
    user = User(email="v@example.com", password_hash="Hashed1a", display_name="V")
    user.email_verified = True
    await svc._user_repo.save(user)

    sent: list = []

    async def _fake_send(**kwargs):  # noqa: ANN003
        sent.append(kwargs)
        return True

    monkeypatch.setattr(svc, "_send_verification_email", _fake_send)

    await svc.resend_verification_email_by_email("v@example.com")
    assert sent == []  # no verification email to an already-verified account


@pytest.mark.asyncio
async def test_resend_endpoint_returns_204_without_nameerror() -> None:
    """The unauthenticated endpoint handler must return cleanly (204), not raise a
    NameError from stale current_user/token references (Phase C gate finding)."""
    from src.api import ResendVerificationRequest, resend_verification

    class _FakeService:
        called_with: list[str] = []

        async def resend_verification_email_by_email(self, email: str) -> None:
            self.called_with.append(email)

    svc = _FakeService()
    req = ResendVerificationRequest(email="anybody@example.com")
    # Endpoint returns None (FastAPI maps to 204); must NOT raise.
    result = await resend_verification(req, svc)  # type: ignore[arg-type]
    assert result is None
    assert svc.called_with == ["anybody@example.com"]
