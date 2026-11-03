import sys

p = r"D:\Repo\Contextual-Chatbot\docs\superpowers\plans\2026-10-18-mvp-remediation\WS-08-identity-auth-mfa.md"
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    n = t.count(old)
    if n != 1:
        sys.exit(f"expected 1 match, got {n}: {old[:90]!r}")
    t = t.replace(old, new)


# ---- WS-08.18 ----
rep("**Depends on:** WS-08.17, WS-14:set_password_hash (request R-6, W3), WS-14:send_password_reset_email (request R-6, W3)",
    "**Depends on:** WS-08.17, WS-14.6, WS-07.3, WS-13.12")

rep("""  - `PasswordResetService(store, user_service, password_service, session_revoker)` with `async request_reset(email: str) -> None` and `async reset(token: str, new_password: str) -> UUID`. `reset` raises `InvalidResetToken`, hashes with `hash_password_async`, calls `user_service.set_password_hash(user_id, new_hash, mark_email_verified=True)`, revokes all sessions and records `AUTH_PASSWORD_RESET`.
  - `password_reset_router`: `POST /auth/password/forgot {email}` always returns 202 `{"status": "accepted"}`. The lookup and the email run in a background task, with a per-address throttle of 3 per hour; the email goes through `user_service.send_password_reset_email(user, token, expiry_minutes=30)`.""",
    """  - `PasswordResetMailer(*, settings: ServiceIntegrationSettings | None = None, http_client_factory: Callable[[], httpx.AsyncClient] | None = None)` with `async send(user: User, token: str, *, expiry_minutes: int) -> bool`. It posts IF-09 `POST {settings.notification_service_url}/api/v1/notifications/email` with body `{"to_email": user.email, "template_type": "password_reset", "variables": {"display_name", "reset_link": f"{settings.frontend_url}/reset-password?token=<token>", "expiry_minutes"}, "priority": "high"}` and headers `await service_auth_headers("notification-service")` (user-service holds `service:write:notification`). Any error or non-2xx returns False and logs WARNING `password_reset_email_failed` with `error_type` or `status` only; neither the token nor the address is logged. `ServiceIntegrationSettings` is imported read-only from `src.domain.service` (WS-14.6), so the URLs come from `SERVICE_NOTIFICATION_SERVICE_URL` and `USER_SERVICE_FRONTEND_URL`. The template `password_reset` and its variables `display_name`, `reset_link` and `expiry_minutes` already exist (`services/notification-service/src/domain/templates.py:190-196`).
  - `PasswordResetService(store, user_repository, password_service, session_revoker, mailer)` with `async request_reset(email: str) -> None` and `async reset(token: str, new_password: str) -> UUID`. `reset` raises `InvalidResetToken`, hashes with `hash_password_async`, loads the user, calls `user.set_password_hash(new_hash, mark_email_verified=True)` and `await user_repository.update(user)`, revokes all sessions and records `AUTH_PASSWORD_RESET`. It needs no new `UserService` method.
  - `password_reset_router`: `POST /auth/password/forgot {email}` always returns 202 `{"status": "accepted"}`. The lookup and the email run in a background task, with a per-address throttle of 3 per hour; the email goes through `mailer.send(user, token, expiry_minutes=30)`.""")

rep("""- Modify: `services/user-service/src/api_auth.py` (`auth_router.include_router(password_reset_router)`; the password-strength validator is shared as `validate_password_strength(value: str) -> str`)
- Modify: `services/user-service/src/identity_wiring.py` (Postgres store when `runtime.pg_pool` is set; strict without a pool raises)
- Modify: `services/user-service/tests/identity_testkit.py` (new override `reset_store` (default `InMemoryPasswordResetTokenStore()`), popped before building `password_reset=PasswordResetService(reset_store, user_service, password_service, session_revoker)`)""",
    """- Modify: `services/user-service/src/domain/entities.py` (new `User.set_password_hash(new_hash: str, *, mark_email_verified: bool = False) -> None`: sets `password_hash` and `updated_at` and clears `login_attempts` and `locked_until`; with `mark_email_verified` it sets `email_verified=True`, clears `email_verification_token` and turns `PENDING_VERIFICATION` into `ACTIVE`; a suspended or deleted account keeps its status)
- Modify: `services/user-service/src/api_auth.py` (`auth_router.include_router(password_reset_router)`; the password-strength validator is shared as `validate_password_strength(value: str) -> str`)
- Modify: `services/user-service/src/identity_wiring.py` (Postgres store when `runtime.pg_pool` is set; strict without a pool raises; `PasswordResetMailer()` with default settings)
- Modify: `services/user-service/tests/identity_testkit.py` (new overrides `reset_store` (default `InMemoryPasswordResetTokenStore()`) and `reset_mailer` (default `RecordingResetMailer()`, a helper class with `sent: list[tuple[str, str, int]]` holding email, raw token and expiry, and `async send(user, token, *, expiry_minutes) -> bool`), both popped before building `password_reset=PasswordResetService(reset_store, user_repository, password_service, session_revoker, reset_mailer)`)""")

rep("""- Consumes: `UserService.set_password_hash(user_id: UUID, new_password_hash: str, *, mark_email_verified: bool = False) -> PasswordChangeResult` and `UserService.send_password_reset_email(user: User, token: str, expiry_minutes: int) -> bool` (request R-6); the `password_reset` notification template (exists in notification-service); slot 013 (WS-08.15); `SessionRevoker` (WS-08.4); `audit_auth_event` (WS-08.14).
- Produces (IF-17): `POST /api/v1/auth/password/forgot`, `POST /api/v1/auth/password/reset`; `PasswordResetService`, `PasswordResetTokenStore.delete_for_user` (request R-10); `validate_password_strength`; error code `INVALID_RESET_TOKEN`.""",
    """- Consumes: `UserRepository.get_by_id`, `get_by_email` and `update(user)` (WS-14's repository, called through `api_deps.get_user_repository`, never edited); `ServiceIntegrationSettings.notification_service_url` and `.frontend_url` (WS-14.6); IF-04 `service_auth_headers(target_service, *, on_behalf_of=None)` (WS-07.3); IF-09 `POST /api/v1/notifications/email {to_email, template_type, variables, priority}` accepting a `service:write:notification` token (WS-13.12); the `password_reset` notification template (exists in notification-service); slot 013 (WS-08.15); `SessionRevoker` (WS-08.4); `audit_auth_event` (WS-08.14).
- Produces (IF-17): `POST /api/v1/auth/password/forgot`, `POST /api/v1/auth/password/reset`; `PasswordResetService`, `PasswordResetMailer`, `User.set_password_hash` (WS-08.20 reuses it), `PasswordResetTokenStore.delete_for_user` (request R-10); `validate_password_strength`; error code `INVALID_RESET_TOKEN`; test-kit `RecordingResetMailer`.""")

rep('''from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from identity_testkit import PASSWORD, build_identity_app, client_for, login, seed_user
from src.domain.value_objects import AccountStatus
from src.password_reset import InMemoryPasswordResetTokenStore

FORGOT = "/api/v1/auth/password/forgot"
RESET = "/api/v1/auth/password/reset"
NEW = "Brand-New-Horse-3"


def _mailbox(state) -> list:
    sent: list = []

    async def send_password_reset_email(user, token, expiry_minutes):
        sent.append((user.email, token, expiry_minutes))
        return True

    async def set_password_hash(user_id, new_password_hash, *, mark_email_verified=False):
        user = await state.user_repository.get_by_id(user_id)
        user.password_hash = new_password_hash
        if mark_email_verified and not user.email_verified:
            user.email_verified = True
            user.status = AccountStatus.ACTIVE
        await state.user_repository.update(user)
        return type("R", (), {"success": True})()

    state.user_service.send_password_reset_email = send_password_reset_email
    state.user_service.set_password_hash = set_password_hash
    return sent
''', '''import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx

import src.password_reset as password_reset
from identity_testkit import PASSWORD, build_identity_app, client_for, login, seed_user
from src.domain.service import ServiceIntegrationSettings
from src.domain.value_objects import AccountStatus
from src.password_reset import InMemoryPasswordResetTokenStore, PasswordResetMailer

FORGOT = "/api/v1/auth/password/forgot"
RESET = "/api/v1/auth/password/reset"
NEW = "Brand-New-Horse-3"


def _mailbox(state) -> list:
    return state.reset_mailer.sent  # RecordingResetMailer from the test kit: (email, raw token, expiry_minutes)
''')

rep('''    user = await state.user_repository.get_by_id(squatter.user_id)
    assert (user.status, user.email_verified, attacker.status_code) == (AccountStatus.ACTIVE, True, 401)
```''', '''    user = await state.user_repository.get_by_id(squatter.user_id)
    assert (user.status, user.email_verified, attacker.status_code) == (AccountStatus.ACTIVE, True, 401)


async def test_mvp_099_reset_email_goes_to_notification_with_a_service_token(monkeypatch):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500 if len(requests) > 1 else 202)

    async def fake_headers(target_service, *, on_behalf_of=None):
        return {"Authorization": f"Bearer svc-{target_service}"}

    monkeypatch.setattr(password_reset, "service_auth_headers", fake_headers)
    settings = ServiceIntegrationSettings(notification_service_url="http://notification.test",
                                          frontend_url="https://app.example")
    mailer = PasswordResetMailer(settings=settings,
                                 http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    user = SimpleNamespace(email="alice@example.com", display_name="Alice")
    assert await mailer.send(user, "raw-token-1", expiry_minutes=30) is True
    assert await mailer.send(user, "raw-token-2", expiry_minutes=30) is False
    first = requests[0]
    body = json.loads(first.content)
    assert str(first.url) == "http://notification.test/api/v1/notifications/email"
    assert first.headers["Authorization"] == "Bearer svc-notification-service"
    assert body["to_email"] == "alice@example.com" and body["template_type"] == "password_reset"
    assert body["variables"] == {"display_name": "Alice", "expiry_minutes": 30,
                                 "reset_link": "https://app.example/reset-password?token=raw-token-1"}
```''')

rep("Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.password_reset'`. The defect behind it: no route issues or redeems a reset token, so a user who forgets the password (or is locked out) can never get back into the account (MVP-099).",
    "Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.password_reset'`. The defect behind it: no route issues or redeems a reset token, so a user who forgets the password (or is locked out) can never get back into the account (MVP-099); `generate_password_reset_token` and `verify_password_reset_token` (`services/user-service/src/infrastructure/token_service.py:184` and `:222`) have no caller outside tests.")

# ---- WS-08.20 ----
rep("**Depends on:** WS-08.19, WS-14:set_password_hash (request R-6, W3), WS-07.7",
    "**Depends on:** WS-08.19, WS-08.18, WS-07.7")
rep("  - success clears the counter, hashes with `hash_password_async`, persists with `user_service.set_password_hash`, revokes all sessions and records `AUTH_PASSWORD_CHANGED`;",
    "  - success clears the counter, hashes with `hash_password_async`, persists with `user.set_password_hash(new_hash)` (WS-08.18) and `await user_repository.update(user)`, revokes all sessions and records `AUTH_PASSWORD_CHANGED`;")
rep("- Consumes: `StepUpGuard` (WS-08.8); `UserService.set_password_hash` (request R-6); `verify_password_async`, `hash_password_async` (WS-08.3); the `auth-password-change` route bucket (WS-07.7).",
    "- Consumes: `StepUpGuard` (WS-08.8); `User.set_password_hash` (WS-08.18) and `UserRepository.update` (WS-14's repository, called, not edited); `verify_password_async`, `hash_password_async` (WS-08.3); the `auth-password-change` route bucket (WS-07.7).")
rep('''def _wire_set_hash(state):
    async def set_password_hash(user_id, new_password_hash, *, mark_email_verified=False):
        user = await state.user_repository.get_by_id(user_id)
        user.password_hash = new_password_hash
        await state.user_repository.update(user)
        return type("R", (), {"success": True})()

    state.user_service.set_password_hash = set_password_hash


''', '')
t = t.replace("    _wire_set_hash(state)\n", "")
if "_wire_set_hash" in t:
    sys.exit("leftover _wire_set_hash")
rep("async def test_mvp_226_success_persists_through_set_password_hash_and_revokes_sessions():",
    "async def test_mvp_226_success_persists_through_the_user_repository_and_revokes_sessions():")
rep("The shared-budget test likewise never sees `SESSION_REVOKED`. The success test passes only once WS-14's `set_password_hash` exists, and fails today because the route still calls `UserService.change_password`, which bypasses the spy.",
    "The shared-budget test likewise never sees `SESSION_REVOKED`. The success test is a regression guard: it passes before and after, because today's route persists through `UserService.change_password` (`services/user-service/src/api.py:760-791`), and it must stay green once the route persists through `User.set_password_hash`.")

open(p, "w", encoding="utf-8", newline="\n").write(t)
print("ok")
