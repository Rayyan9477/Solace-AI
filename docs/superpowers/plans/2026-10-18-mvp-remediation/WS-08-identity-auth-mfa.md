# WS-08: Identity: login, sessions, OAuth, re-authentication, MFA and password reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make email/password and Google sign-in safe and usable by real clients: correct lockout and verification gates, session-bound refresh rotation with a retry grace window, safe OAuth linking with a one-time code handoff, re-authentication for destructive actions, TOTP MFA that is mandatory for clinicians and admins, password reset, and every authentication event in the audit trail.

**Architecture:** All work lives in user-service's identity modules, which WS-14 hands over at the W0 barrier (`api_auth.py`, `api_deps.py`, `identity_wiring.py`), plus the identity infrastructure modules this plan owns (JWT, passwords, OAuth, MFA, reset). Shared platform pieces are consumed, never edited: the shared token blacklist and session-revocation key (WS-07, IF-04), `client_ip`, `require_mfa_for_roles`, the PHI codec and the audit runtime (WS-09, IF-13/IF-14), and the runtime bootstrap (WS-04, IF-01). Redis-backed stores (rotation, step-up counters, OAuth codes) are proven on real Redis, and the Postgres stores (oauth_accounts, MFA factors, reset tokens, audit rows) are proven on the migrated schema as `solace_app` under RLS.

**Covers:** 21 register entries (blocker 2, high 3, medium 15, low 1), plus supporting work for MVP-209 (WS-07 closes it) and MVP-100 (WS-14 closes it). **Waves:** 2 (phase B), 3 (phase B), 4 (phase A: slot 013; phase B: the rest). **Depends on:**
- WS-14: WS-14.3 (the W0 split creates `api_auth.py`, `api_deps.py` and `identity_wiring.py` by verbatim move and hands them over at the W0 barrier), WS-14.6 (`verify_email(user_id | None, token)` and `ServiceIntegrationSettings`), WS-14.16 (`delete_user` returning `DeleteUserResult.erasure_request_id`, and the `oauth_accounts` erasure store) and WS-14.17 (`get_consent_claim`). WS-14 owns `UserService` and the user repository (IF-16); the asks no WS-14 task produces yet stay open under Cross-workstream requests.
- WS-07: the shared blacklist with `revoke_session` and the `token_blacklist:sid:<sid>` key, `build_token_blacklist` (WS-07.5), IF-17 claim surfacing and rejection of the `reauth`/`mfa`/`mfa_enroll` token types (WS-07.2), `client_ip` with env-resolved hops (WS-07.7), the strict RS256 rules (WS-07.10) and `require_mfa_for_roles` (WS-07.11).
- WS-04: `is_strict_environment` (WS-04.1), `create_service_app` and `ServiceRuntimeSpec` (WS-04.3), `ServiceRuntime.redis`/`pg_pool` (WS-04.4), the edge error envelope (WS-04.9) and audit wiring (WS-04.10).
- WS-09: the PHI codec for TOTP secrets and email hashes (WS-09.7, IF-13) and the audit runtime (WS-09.9, WS-09.11, IF-14).
- WS-13: WS-13.12 (`/api/v1/notifications/email` accepts a `service:write:notification` token; the reset mailer uses it). The `account_exists` template is an open request (R-4).
- WS-02: slot `013_identity_mfa` on the fixed chain (WS-02.3) and the NULLIF policy rule.
- WS-05: the IF-19 `redis_url` and `pg_migrated` fixtures (WS-05.2).
- WS-01: WS-01.2 (`pyotp`, PyJWT >= 2.13 and `httpx` in the lock, D-16).
- WS-03: WS-03.1 (the IF-02 OAuth rows for user-service) and WS-03.14 (RS256 key files); the strict-required OAuth flags and the operator docs stay in open request R-12.

**Owned paths:** `services/user-service/src/auth.py`, `services/user-service/src/oauth.py`, `services/user-service/src/api_auth.py`, `services/user-service/src/api_deps.py`, `services/user-service/src/identity_wiring.py`, `services/user-service/src/mfa.py`, `services/user-service/src/password_reset.py`, `services/user-service/src/domain/entities.py`, `services/user-service/src/infrastructure/jwt_service.py`, `services/user-service/src/infrastructure/password_service.py`, `services/user-service/src/infrastructure/token_service.py`, `services/user-service/src/infrastructure/oauth_google.py`, `services/user-service/src/infrastructure/oauth_account_repository.py`, `services/user-service/src/infrastructure/mfa_repository.py`, `migrations/versions/013_identity_mfa.py` (`upgrade()`/`downgrade()` only), `docs/api/auth.md`. New test files go under `services/user-service/tests/` (`test_mvp_<id>_<slug>.py` plus the helper `identity_testkit.py`) and `tests/realinfra/identity-auth-mfa/`. **Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

Conventions for every task below:
- **W0 handoff assumptions.** After WS-14's verbatim split, `src.api_auth` exposes `auth_router` (the `/auth/*` routes, `POST /users/me/password`, `DELETE /users/me` and the JWKS route) and the request models; `src.api_deps` exposes `get_jwt_service`, `get_password_service`, `get_user_service`, `get_user_repository`, `get_session_manager`, `get_current_user`, `UserResponse`, `user_to_response` and `TokenResponse`; `src.identity_wiring` exposes `start_identity(runtime, app)` and `stop_identity()`. If the split left `TokenResponse` in `api_auth.py`, WS-08.2 moves it to `api_deps.py` and re-exports it from `api_auth.py`, so `oauth.py`, `mfa.py` and `password_reset.py` import it without a cycle. Route handlers keep their names (`login`, `refresh_token`, `logout`, `revoke_token`, `register`, `verify_email`, `resend_verification`, `change_password`, `delete_current_user`).
- **No edits to WS-14 files.** `main.py`, `api.py`, `domain/service.py`, `config.py`, the repositories and `tests/conftest.py` belong to WS-14. A test that needs a not-yet-landed `UserService` method sets it as an instance attribute on `state.user_service` (a spy), so the WS-08 tests do not wait on WS-14 lane timing. The real method is still a declared dependency, and the W2-W4 E2E suites prove it.
- **Error bodies.** Routes raise `HTTPException(status, detail={"code": <CODE>, "message": <fixed text>})`. The edge envelope (WS-04.9) renders this as `{"error": {"code", "message", "request_id"}}`. No `str(exc)`, token, email or password ever appears in a body, a log line or an audit row.
- **Test kit.** `services/user-service/tests/identity_testkit.py` (created in WS-08.2 and extended task by task) builds the identity stack in process with `create_service_app` (edge spec, rate limiting off), in-memory repositories from `fixtures.py` and a fast Argon2 configuration. Tests drive it with `httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://auth.example")`. `pytest.ini` sets `asyncio_mode = auto`, so async tests need no marker.
- **Real-infra tests** live in `tests/realinfra/identity-auth-mfa/`, carry `pytestmark = [pytest.mark.realinfra, pytest.mark.integration]`, use only the IF-19 fixtures, and always run in their own pytest process (their `conftest.py` puts `services/user-service` first on `sys.path` and refuses a shadowed `src` package).
- Test passwords, secrets and emails below are throwaway synthetic literals that exist only in tests.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/user-service/src/domain/entities.py` | Modify | Lockout state machine, check order, `can_refresh`, shared `normalize_email`/`normalize_locale`, `set_password_hash` | WS-08.1, WS-08.5, WS-08.7, WS-08.18 |
| `services/user-service/src/api_auth.py` | Modify | Login gate, verify-email body, async hashing, sid-bound refresh, rotation, 202 registration, reauth and purpose-bound delete, client IP, cns, loc, audit, MFA step-up, password change guard; aggregates `mfa_router` and `password_reset_router` | WS-08.2 to WS-08.8, WS-08.11, WS-08.13, WS-08.14, WS-08.16, WS-08.18, WS-08.19, WS-08.20, WS-08.21 |
| `services/user-service/src/api_deps.py` | Modify | `TokenResponse` home, `get_session_revoker`, `get_step_up_guard`, 503 mapping for backend errors, `require_mfa_for_roles` for user-service routes | WS-08.2, WS-08.4, WS-08.8, WS-08.16 |
| `services/user-service/src/identity_wiring.py` | Modify | `build_session_security` (shared blacklist, sessions, revoker, rotation store, step-up counters), OAuth component wiring, MFA and reset wiring, `build_jwt_service` with strict RS256 | WS-08.4, WS-08.6, WS-08.8, WS-08.9, WS-08.10, WS-08.12, WS-08.16, WS-08.17, WS-08.18, WS-08.19 |
| `services/user-service/src/auth.py` | Modify | `SessionRevoker`, Redis index TTL on activity, refresh rotation stores and Lua swap, failure counters and `StepUpGuard`, `audit_auth_event`, `consent_claim_for`, `locale_claim_for` (D-29) | WS-08.4, WS-08.5, WS-08.6, WS-08.8, WS-08.13, WS-08.14, WS-08.21 |
| `services/user-service/src/infrastructure/jwt_service.py` | Modify | Shared blacklist re-export, sid/amr/mfa/cns/loc claims, type-before-blacklist, reauth/mfa/enrollment tokens, strict RS256 default | WS-08.4, WS-08.5, WS-08.6, WS-08.8, WS-08.13, WS-08.16, WS-08.19, WS-08.21 |
| `services/user-service/src/infrastructure/password_service.py` | Modify | `hash_password_async`, `verify_password_async`, bounded concurrency | WS-08.3 |
| `services/user-service/src/infrastructure/oauth_google.py` | Modify | Endpoints and issuer from settings, `JWKSCache`, `verify_id_token_async` with nonce, no response-body logging | WS-08.9, WS-08.10 |
| `services/user-service/src/oauth.py` | Modify | Settings from `GOOGLE_OAUTH_*`, redirect allow-list, state cookie binding, one-time code handoff, `/exchange`, link by Google subject, `/link`, MFA step-up, `loc` claim | WS-08.3, WS-08.4, WS-08.9, WS-08.10, WS-08.11, WS-08.12, WS-08.13, WS-08.14, WS-08.16, WS-08.21 |
| `services/user-service/src/infrastructure/oauth_account_repository.py` | Create | `OAuthAccountRepository` (in-memory and Postgres), `delete_for_user` | WS-08.12 |
| `migrations/versions/013_identity_mfa.py` | Modify | `user_mfa_factors`, `mfa_recovery_codes` (RLS, NULLIF) and `password_reset_tokens` (non-RLS) | WS-08.15 |
| `services/user-service/src/infrastructure/mfa_repository.py` | Create | `MFARepository` protocol, in-memory and Postgres implementations | WS-08.16, WS-08.17 |
| `services/user-service/src/mfa.py` | Create | `TOTPVerifier`, `MFAService`, recovery codes, `mfa_router` (verify, enroll start/confirm) | WS-08.16, WS-08.17 |
| `services/user-service/src/password_reset.py` | Create | `PasswordResetService`, token stores (in-memory and Postgres), `password_reset_router` | WS-08.18 |
| `docs/api/auth.md` | Create | The IF-17 client contract (linked from `docs/API-HANDOFF.md`) | WS-08.7, WS-08.8, WS-08.10, WS-08.12, WS-08.16, WS-08.17, WS-08.18, WS-08.20, WS-08.21 |
| `services/user-service/tests/test_mvp_221_locale_claim.py` | Create | The `loc` claim on every issuance path and the grammar parity lock (D-29) | WS-08.21 |
| `services/user-service/tests/identity_testkit.py` | Create | In-process identity stack for WS-08 tests (helper, no tests) | WS-08.2, extended in WS-08.3, .4, .6, .8, .10, .12, .14, .16, .18 |
| `services/user-service/tests/test_auth_timing_health.py` | Modify | Fake password service gains async methods | WS-08.3 |
| `services/user-service/tests/test_refresh_rotation.py` | Modify | Rewritten on the test kit: sid-bound tokens, then grace semantics | WS-08.4, WS-08.6 |
| `services/user-service/tests/test_api.py` | Modify | Login error codes; registration 202 | WS-08.2, WS-08.7 |
| `services/user-service/tests/test_oauth_endpoints.py` | Modify | Registration 202; redirect/code contract; link-by-subject expectations | WS-08.7, WS-08.10, WS-08.12 |
| `services/user-service/tests/test_oauth_google.py` | Modify | Verifier cases move to `verify_id_token_async` with a mocked JWKS | WS-08.9 |
| `services/user-service/tests/test_mvp_*.py` (20 files, named in each task) | Create | Task acceptance tests | every task except WS-08.15 |
| `tests/realinfra/identity-auth-mfa/conftest.py` | Create | `sys.path` setup, shadow guard, `insert_minimal_user` fixture | WS-08.4, WS-08.12 |
| `tests/realinfra/identity-auth-mfa/test_mvp_*.py` (7 files) | Create | Real Redis and Postgres proofs | WS-08.4, .6, .12, .14, .15, .17, .18 |

## Tasks

### Task WS-08.1: Lockout state machine: an expired lock resets the counter and a lock is checked before verification state
**Covers:** MVP-215, MVP-219
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-14.3 (the W0 verbatim split; ownership of `entities.py` passes to WS-08 at the W0 barrier)
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/domain/entities.py` (`User.record_login_attempt`: a failure while the lock is active changes nothing; a failure after `locked_until` has passed first resets `login_attempts` to 0 and `locked_until` to None, then counts and locks at `max_attempts`; `User.can_login`: check order deleted, suspended, locked, pending verification, other status; new `User.lock_expired()`)
- Test: `services/user-service/tests/test_mvp_215_lockout_state.py`
**Interfaces:**
- Consumes: none.
- Produces: `User.record_login_attempt(success: bool, max_attempts: int = 5, lockout_duration_minutes: int = 30) -> None` (same signature, new semantics); `User.can_login() -> tuple[bool, str | None]`, which returns `(False, "Account is locked until <iso>")` before `(False, "Email verification required")`; `User.lock_expired() -> bool` (True when `locked_until` is set and in the past). WS-08.2 and the Google callback rely on this order.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_215_lockout_state.py::test_mvp_215_expired_lock_resets_the_failure_counter` (and the other tests in the file)
```python
"""MVP-215 / MVP-219: lockout resets when it expires, never extends while active, and wins over verification state."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.domain.entities import User
from src.domain.value_objects import AccountStatus


def _user(**fields) -> User:
    base = dict(email="alice@example.com", password_hash="$argon2id$synthetic", display_name="Alice",
                status=AccountStatus.ACTIVE, email_verified=True)
    base.update(fields)
    return User(**base)


def _fail(user: User, times: int = 1) -> None:
    for _ in range(times):
        user.record_login_attempt(success=False, max_attempts=5, lockout_duration_minutes=30)


def _past() -> datetime:
    return datetime.now(timezone.utc) - timedelta(seconds=1)


def test_mvp_215_expired_lock_resets_the_failure_counter():
    user = _user()
    _fail(user, 5)
    assert user.is_locked() and user.login_attempts == 5
    user.locked_until = _past()
    assert user.lock_expired() is True
    _fail(user, 1)
    assert (user.login_attempts, user.is_locked(), user.locked_until) == (1, False, None)


def test_mvp_215_five_new_failures_after_expiry_lock_again():
    user = _user()
    _fail(user, 5)
    user.locked_until = _past()
    _fail(user, 5)
    assert user.is_locked() and user.login_attempts == 5


def test_mvp_215_failures_during_an_active_lock_do_not_extend_it():
    user = _user()
    _fail(user, 5)
    first_lock = user.locked_until
    _fail(user, 3)
    assert (user.locked_until, user.login_attempts) == (first_lock, 5)


def test_mvp_215_success_still_clears_everything():
    user = _user(login_attempts=3)
    user.record_login_attempt(success=True, max_attempts=5, lockout_duration_minutes=30)
    assert (user.login_attempts, user.locked_until) == (0, None) and user.last_login is not None


@pytest.mark.parametrize("status", [AccountStatus.PENDING_VERIFICATION, AccountStatus.ACTIVE])
def test_mvp_219_lock_is_reported_before_verification_state(status):
    user = _user(status=status, email_verified=status is AccountStatus.ACTIVE, login_attempts=5,
                 locked_until=datetime.now(timezone.utc) + timedelta(minutes=10))
    allowed, reason = user.can_login()
    assert allowed is False
    assert reason.startswith("Account is locked")


def test_mvp_219_expired_lock_on_pending_account_reports_verification():
    user = _user(status=AccountStatus.PENDING_VERIFICATION, email_verified=False, login_attempts=5, locked_until=_past())
    assert user.can_login() == (False, "Email verification required")
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_215_lockout_state.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `test_mvp_215_expired_lock_resets_the_failure_counter` gets `AttributeError: 'User' object has no attribute 'lock_expired'`. Without that line it would get `(6, True, <future>)`, because one wrong password after expiry re-locks the account (MVP-215). `..._do_not_extend_it` gets a later `locked_until` and `login_attempts == 8`. `test_mvp_219_lock_is_reported_before_verification_state[PENDING_VERIFICATION]` gets `"Email verification required"`, so the lock is never seen for pending accounts (MVP-219).
- [ ] **Step 3: Implement** `User.lock_expired() -> bool`, the new `record_login_attempt` failure branch and the reordered `can_login` in `services/user-service/src/domain/entities.py`. The success branch is unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_entities.py` lock-after-five and reset-on-success tests are unchanged)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.2: Login refuses unverified accounts in every strict environment and never lets a lock through; verify-email takes the emailed token
**Covers:** MVP-213, MVP-219
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.1, WS-04.1, WS-04.3, WS-04.9, WS-14.3, WS-14.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: staging is as strict as production; every guard uses `is_strict_environment()`)
**Real infra:** no
**Files:**
- Create: `services/user-service/tests/identity_testkit.py` (helper module, full content in Step 1)
- Modify: `services/user-service/src/api_auth.py` (`login`: the verification gate uses `solace_infrastructure.runtime.is_strict_environment()` instead of `ENVIRONMENT == "production"`; order: unknown account, then blocked (`can_login` false for any reason other than verification), then password, then verification gate; error details use codes `INVALID_CREDENTIALS` (401) and `EMAIL_NOT_VERIFIED` (403); `verify_email` accepts a JSON body `VerifyEmailRequest{token: str, user_id: UUID | None = None}` and keeps the legacy `?user_id=&token=` query form; it calls `user_service.verify_email(user_id, token)` with the `user_id` from the body or query when present and `None` otherwise; a failure returns 400 `INVALID_VERIFICATION_TOKEN` without `result.error`)
- Modify: `services/user-service/src/api_deps.py` (`TokenResponse` lives here; moved from `api_auth.py` if the W0 split placed it there, with a re-export left in `api_auth.py`)
- Modify: `services/user-service/tests/test_api.py` (only login assertions that read the old string `detail`; they now read `detail["code"]`)
- Test: `services/user-service/tests/test_mvp_213_login_verification_gate.py`
**Interfaces:**
- Consumes: `is_strict_environment` (IF-01, WS-04.1); `create_service_app`, `ServiceRuntimeSpec` (IF-01, WS-04.3) in the test kit; the edge error envelope (WS-04.9); `UserService.verify_email(user_id: UUID | None, token: str) -> EmailVerificationResult` (WS-14.6: without a `user_id` it consumes the token first, then looks the user up); `User.can_login` order (WS-08.1).
- Produces: `POST /api/v1/auth/verify-email` body `{token}` with optional `user_id` (IF-17), which is the IF-17 conformance WS-14 requests in R-14-8; login error codes `INVALID_CREDENTIALS` and `EMAIL_NOT_VERIFIED`; test-kit names `SECRET`, `PASSWORD`, `FAST_PASSWORDS`, `BASE_URL`, `IdentityTestState`, `build_identity_state(**overrides)`, `build_identity_app(state=None, **overrides) -> tuple[FastAPI, IdentityTestState]`, `client_for(app, *, base_url=BASE_URL) -> httpx.AsyncClient`, `seed_user(state, *, email, password, role, status, verified, **fields) -> User`, `login(client, *, email, password, headers=None) -> dict`, `claims(token) -> dict`, `bearer(token) -> dict[str, str]`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_213_login_verification_gate.py::test_mvp_213_unverified_account_cannot_log_in_in_strict_envs`. First create the helper `services/user-service/tests/identity_testkit.py`:
```python
"""WS-08 identity test kit: the user-service identity stack in process (helper module, holds no tests)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
from fastapi import FastAPI

from fixtures import InMemoryConsentRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository
from solace_infrastructure.runtime import ServiceRuntimeSpec, create_service_app
from src.api_auth import auth_router
from src.auth import SessionConfig, SessionManager
from src.domain.entities import User
from src.domain.service import UserService
from src.domain.value_objects import AccountStatus, UserRole
from src.infrastructure.jwt_service import InMemoryTokenBlacklist, JWTConfig, JWTService
from src.infrastructure.password_service import PasswordConfig, PasswordService
from src.oauth import oauth_router

SECRET = "test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"
PASSWORD = "Correct-Horse-9"
FAST_PASSWORDS = PasswordConfig(argon2_time_cost=1, argon2_memory_cost=8192)
BASE_URL = "https://auth.example"


class IdentityTestState(SimpleNamespace):
    def increment_stat(self, name: str) -> None:
        self.stats[name] = self.stats.get(name, 0) + 1


def build_identity_state(**overrides: Any) -> IdentityTestState:
    blacklist = overrides.pop("token_blacklist", None) or InMemoryTokenBlacklist()
    passwords = overrides.pop("password_service", None) or PasswordService(FAST_PASSWORDS)
    users, prefs, consents = InMemoryUserRepository(), InMemoryUserPreferencesRepository(), InMemoryConsentRepository()
    state = IdentityTestState(
        stats={}, initialized=True, settings=None, token_blacklist=blacklist, password_service=passwords,
        jwt_service=JWTService(JWTConfig(secret_key=SECRET), blacklist=blacklist),
        user_repository=users, preferences_repository=prefs, consent_repository=consents,
        user_service=UserService(user_repository=users, preferences_repository=prefs,
                                 consent_repository=consents, password_service=passwords),
        session_manager=SessionManager(SessionConfig(secret_key=SECRET)),
        oauth_settings=None, oauth_state_store=None, oauth_http_client_factory=None, oauth_id_token_verifier=None,
    )
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def build_identity_app(state: IdentityTestState | None = None, **overrides: Any) -> tuple[FastAPI, IdentityTestState]:
    state = state or build_identity_state(**overrides)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    routers = [(auth_router, "/api/v1")]
    if not any(getattr(r, "path", "").startswith("/auth/oauth/") for r in auth_router.routes):
        routers.append((oauth_router, "/api/v1"))
    spec = ServiceRuntimeSpec(service_name="user-service", environment="test", edge=True)
    app = create_service_app(spec, lifespan=lifespan, routers=routers, title="identity-test")
    app.state.service = state
    return app, state


def client_for(app: FastAPI, *, base_url: str = BASE_URL) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


async def seed_user(state: IdentityTestState, *, email: str = "alice@example.com", password: str = PASSWORD,
                    role: UserRole = UserRole.USER, status: AccountStatus = AccountStatus.ACTIVE,
                    verified: bool = True, **fields: Any) -> User:
    user = User(email=email, password_hash=state.password_service.hash_password(password), display_name="Test User",
                role=role, status=status, email_verified=verified, **fields)
    return await state.user_repository.save(user)


async def login(client: httpx.AsyncClient, *, email: str = "alice@example.com", password: str = PASSWORD,
                headers: dict[str, str] | None = None) -> dict:
    response = await client.post("/api/v1/auth/login", json={"email": email, "password": password}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def claims(token: str) -> dict:
    return jwt.decode(token, options={"verify_signature": False})


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
```
Then the test file:
```python
"""MVP-213 / MVP-219: strict environments refuse unverified logins, and a lock always holds."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from identity_testkit import PASSWORD, build_identity_app, client_for, seed_user
from src.domain.value_objects import AccountStatus

LOGIN = "/api/v1/auth/login"
VERIFY = "/api/v1/auth/verify-email"
PENDING = dict(email="pending@example.com", status=AccountStatus.PENDING_VERIFICATION, verified=False)


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_213_unverified_account_cannot_log_in_in_strict_envs(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    app, state = build_identity_app()
    await seed_user(state, **PENDING)
    async with client_for(app) as c:
        r = await c.post(LOGIN, json={"email": "pending@example.com", "password": PASSWORD})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "EMAIL_NOT_VERIFIED"
    assert "access_token" not in r.text


@pytest.mark.parametrize("env", ["staging", "production"])
async def test_mvp_213_wrong_password_on_unverified_account_is_a_counted_401(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    app, state = build_identity_app()
    user = await seed_user(state, **PENDING)
    async with client_for(app) as c:
        r = await c.post(LOGIN, json={"email": "pending@example.com", "password": "Wrong-Horse-1"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_CREDENTIALS"
    assert (await state.user_repository.get_by_id(user.user_id)).login_attempts == 1


async def test_mvp_213_development_keeps_unverified_login_for_local_work(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    app, state = build_identity_app()
    await seed_user(state, **PENDING)
    async with client_for(app) as c:
        r = await c.post(LOGIN, json={"email": "pending@example.com", "password": PASSWORD})
    assert r.status_code == 200 and r.json()["access_token"]


@pytest.mark.parametrize("env", ["development", "production"])
async def test_mvp_219_locked_pending_account_gets_no_tokens_with_the_right_password(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    app, state = build_identity_app()
    await seed_user(state, **PENDING, login_attempts=5, locked_until=datetime.now(timezone.utc) + timedelta(minutes=20))
    async with client_for(app) as c:
        r = await c.post(LOGIN, json={"email": "pending@example.com", "password": PASSWORD})
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_CREDENTIALS"
    assert "access_token" not in r.text


async def test_mvp_213_verify_email_accepts_the_token_from_the_emailed_link():
    app, state = build_identity_app()
    uid = uuid4()
    seen: list[tuple] = []

    async def verify_email(user_id, token):  # spy with WS-14.6's signature verify_email(user_id: UUID | None, token)
        seen.append((user_id, token))
        return SimpleNamespace(success=token == "tok-123", error="internal detail", error_code="INVALID_TOKEN")

    state.user_service.verify_email = verify_email
    async with client_for(app) as c:
        ok = await c.post(VERIFY, json={"token": "tok-123"})
        bad = await c.post(VERIFY, json={"token": "tok-999"})
        with_id = await c.post(VERIFY, json={"token": "tok-123", "user_id": str(uid)})
        legacy = await c.post(VERIFY, params={"user_id": str(uid), "token": "tok-123"})
    assert (ok.status_code, with_id.status_code, legacy.status_code) == (204, 204, 204)
    assert seen == [(None, "tok-123"), (None, "tok-999"), (uid, "tok-123"), (uid, "tok-123")]
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_VERIFICATION_TOKEN"
    assert "internal detail" not in bad.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_213_login_verification_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL. With `ENVIRONMENT=staging` the unverified login returns 200 with tokens, because the gate only checks `== "production"` (MVP-213). The production case returns 401 `UNAUTHORIZED` instead of 403 `EMAIL_NOT_VERIFIED`. The locked pending account gets 200 in development, because `can_login()` reports verification before the lock and the non-production branch lets that through (MVP-219). Verify-email returns 422 for the two body-only calls and the body call with `user_id`, because the verbatim-moved route still declares `user_id: UUID` and `token: str` as required query parameters (`services/user-service/src/api.py:1005-1011` today), so a `{token}` body is never read; only the legacy query call reaches the spy. The development regression test passes.
- [ ] **Step 3: Implement** the login order and gate, the error-code details and `VerifyEmailRequest` in `services/user-service/src/api_auth.py`. Blocked paths keep the constant-time dummy verify before their 401. The `TokenResponse` move happens only if it is needed.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.3: Argon2 hashing and verification run off the event loop on every password route
**Covers:** MVP-210
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.2
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/password_service.py` (`PasswordConfig.max_concurrent_hashes: int = 4`; `PasswordService.hash_password_async` and `verify_password_async` run the sync methods through `asyncio.to_thread`, holding a per-instance `threading.BoundedSemaphore(max_concurrent_hashes)` inside the worker thread)
- Modify: `services/user-service/src/api_auth.py` (`register`, `login`, `change_password` and the dummy verify use the async variants; `_constant_time_dummy_verify` becomes `async`, and the dummy hash is computed once with `hash_password_async`)
- Modify: `services/user-service/src/oauth.py` (`_link_or_create_user` hashes its random password with `hash_password_async`)
- Modify: `services/user-service/tests/test_auth_timing_health.py` (`_RecordingPasswordService` gains `hash_password_async`/`verify_password_async` with the same counters)
- Modify: `services/user-service/tests/identity_testkit.py` (add `enable_google_oauth(state, *, identity, settings=None)`, `google_round_trip(client) -> httpx.Response` and `ticks_during(awaitable) -> tuple[Any, int]`)
- Test: `services/user-service/tests/test_mvp_210_argon2_off_loop.py`
**Interfaces:**
- Consumes: none.
- Produces: `PasswordService.hash_password_async(password: str) -> str` and `PasswordService.verify_password_async(password: str, password_hash: str) -> PasswordVerificationResult` (awaitables). WS-08.8, WS-08.12, WS-08.18 and WS-08.20 use them, and WS-14 switches `UserService.change_password` to them (request R-5). Test kit: `enable_google_oauth` installs `GoogleOAuthSettings(client_id="test-google-client-id", client_secret="test-google-client-secret", redirect_uri="https://auth.example/api/v1/auth/oauth/google/callback")`, an `InMemoryOAuthStateStore`, a fake token-exchange client and a stub verifier that returns `identity`. `google_round_trip` performs `/start` and then `/callback` with the returned state.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_210_argon2_off_loop.py::test_mvp_210_password_routes_do_not_block_the_event_loop`
```python
"""MVP-210: Argon2 runs off the event loop for register, login, the dummy verify and Google signup."""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from identity_testkit import (FAST_PASSWORDS, PASSWORD, build_identity_app, client_for, enable_google_oauth,
                              google_round_trip, seed_user, ticks_during)
from src.infrastructure.oauth_google import GoogleIdentity
from src.infrastructure.password_service import PasswordConfig, PasswordService


class _SlowPasswords(PasswordService):
    """Every hash and verify holds its thread for 300 ms, like production Argon2 under load."""

    def hash_password(self, password: str) -> str:
        time.sleep(0.3)
        return super().hash_password(password)

    def verify_password(self, password: str, password_hash: str):
        time.sleep(0.3)
        return super().verify_password(password, password_hash)


@pytest.mark.parametrize("route,body", [
    ("/api/v1/auth/login", {"email": "alice@example.com", "password": PASSWORD}),
    ("/api/v1/auth/login", {"email": "nobody@example.com", "password": PASSWORD}),
    ("/api/v1/auth/register", {"email": "new@example.com", "password": PASSWORD, "display_name": "New"}),
])
async def test_mvp_210_password_routes_do_not_block_the_event_loop(route, body):
    app, state = build_identity_app(password_service=_SlowPasswords(FAST_PASSWORDS))
    await seed_user(state)
    async with client_for(app) as c:
        _response, ticks = await ticks_during(c.post(route, json=body))
    assert ticks >= 8


async def test_mvp_210_google_signup_hash_does_not_block_the_event_loop():
    app, state = build_identity_app(password_service=_SlowPasswords(FAST_PASSWORDS))
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-new", email="new@example.com",
                                                       email_verified=True))
    async with client_for(app) as c:
        _response, ticks = await ticks_during(google_round_trip(c))
    assert ticks >= 8


class _CountingPasswords(PasswordService):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def hash_password(self, password: str) -> str:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return "$argon2id$synthetic"


async def test_mvp_210_concurrent_hashes_are_bounded():
    svc = _CountingPasswords(PasswordConfig(argon2_time_cost=1, argon2_memory_cost=8192, max_concurrent_hashes=2))
    await asyncio.gather(*(svc.hash_password_async(PASSWORD) for _ in range(6)))
    assert svc.peak == 2
```
Test-kit helper added in this task:
```python
async def ticks_during(awaitable) -> tuple[Any, int]:
    """Run `awaitable` while a 20 ms ticker counts how often the event loop was free."""
    ticks = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await awaitable
    finally:
        done.set()
        await task
    return result, ticks
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_210_argon2_off_loop.py -o "addopts=" -p no:logging -q`
Expected: FAIL. Each route test sees `ticks` of 0 or 1, because `hash_password`/`verify_password` run synchronously on the only event loop for about 300 ms (MVP-210: register, login, the dummy verify and Google signup). The concurrency test fails with `AttributeError: '_CountingPasswords' object has no attribute 'hash_password_async'`.
- [ ] **Step 3: Implement** `hash_password_async(self, password: str) -> str` and `verify_password_async(self, password: str, password_hash: str) -> PasswordVerificationResult` in `password_service.py`, and switch the call sites in `api_auth.py` and `oauth.py`. Approach: the semaphore is a `threading.BoundedSemaphore` acquired inside the function that `asyncio.to_thread` runs, so it binds no event loop and caps Argon2 memory at `max_concurrent_hashes x 64 MB`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass, including `test_auth_timing_health.py` with its async-capable fake
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.4: Session-bound tokens: every token carries its sid, refresh needs that live session, and logout and password change revoke sessions in the shared blacklist
**Covers:** MVP-212
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.3, WS-07.2, WS-07.5, WS-04.4, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: identity routes fail closed with 503 when the revocation or session backend errors; the shared middleware's own check fails open, per WS-07.5)
**Real infra:** yes
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py`:
  - the local `TokenBlacklist`, `InMemoryTokenBlacklist` and `RedisTokenBlacklist` are deleted and re-exported from `solace_security.auth`;
  - new `IdentityBackendUnavailable(JWTError)`;
  - `TokenPayload` gains `session_id: UUID | None = None`, `amr: tuple[str, ...] = ()`, `mfa: bool = False` and `expires_at: datetime | None = None`, and `TokenPair` gains `session_id: UUID | None = None`;
  - `generate_token_pair(user_id, email, role, *, session_id=None, amr=("pwd",), mfa=False)` writes `sid` (when given), `amr` and `mfa` into both tokens;
  - `verify_token` fills the new fields and, after the existing jti check, raises `TokenInvalidError("Session has been revoked")` when `blacklist.is_session_revoked(sid)`. Any exception from a blacklist call becomes `IdentityBackendUnavailable`, and `revoke_token` does the same.
- Modify: `services/user-service/src/auth.py` (new `SessionRevoker`)
- Modify: `services/user-service/src/api_auth.py`:
  - `login` creates the session before minting and passes `session_id=session.session_id, amr=("pwd",)`;
  - `refresh_token` rejects a refresh token without `session_id` (401 `INVALID_TOKEN`), requires `await session_manager.validate_session(sid)` and a non-revoked sid (401 `SESSION_EXPIRED`), and mints with the same sid, amr and mfa. The existing "revoked" reuse branch stays as it is; WS-08.5 and WS-08.6 replace it.
  - `logout` revokes only the caller's sid and blacklists the presented access token; it falls back to `revoke_all` for a sid-less legacy token;
  - `change_password` and `delete_current_user` call `session_revoker.revoke_all(user_id, reason=...)`;
  - `IdentityBackendUnavailable` becomes 503 `IDENTITY_BACKEND_UNAVAILABLE`.
- Modify: `services/user-service/src/api_deps.py` (`get_session_revoker(request) -> SessionRevoker`; `get_current_user` maps `IdentityBackendUnavailable` to 503 `IDENTITY_BACKEND_UNAVAILABLE`)
- Modify: `services/user-service/src/oauth.py` (the callback mints with `session_id=session.session_id, amr=("google",)`)
- Modify: `services/user-service/src/identity_wiring.py` (new `build_session_security`; `start_identity` uses it with `runtime.redis` and `is_strict_environment(runtime.spec.environment)`, then stores `jwt_service`, `token_blacklist`, `session_manager` and `session_revoker` on `app.state.service`)
- Modify: `services/user-service/tests/test_refresh_rotation.py` (rewritten on the test kit: tokens come from `/auth/login`; its two existing scenarios keep their assertions)
- Modify: `services/user-service/tests/identity_testkit.py` (`build_identity_state` pops the `session_manager` override before building, then adds `session_revoker=SessionRevoker(session_manager, blacklist, access_token_ttl=timedelta(minutes=15))`, so an overridden session manager or blacklist is the one the revoker uses; `SessionRevoker` keeps them as `_session_manager` and `_blacklist`)
- Create: `tests/realinfra/identity-auth-mfa/conftest.py` (new file in this workstream's real-infra slug; it only reads the user-service suite and edits nothing there). It puts the user-service package root and then that service's tests directory at the front of `sys.path`, applies the same test-only `os.environ.setdefault` values as the user-service suite's conftest (copied, not imported), and raises `RuntimeError("tests/realinfra/identity-auth-mfa must run in its own pytest process")` when `sys.modules["src"]` is a package outside the user-service package root.
- Test: `services/user-service/tests/test_mvp_212_session_bound_refresh.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_212_logout_revokes_across_services.py`
**Interfaces:**
- Consumes: `solace_security.auth.TokenBlacklist.revoke_session(session_id: str, until: datetime)`, `.is_session_revoked(session_id: str) -> bool`, `InMemoryTokenBlacklist`, `RedisTokenBlacklist` and the key `token_blacklist:sid:<sid>` (WS-07.5); `solace_security.middleware.build_token_blacklist(redis_client=None, *, strict=None)` (WS-07.5); `TokenPayload.session_id` read from `sid` by the shared validator (WS-07.2); `ServiceRuntime.redis`, `ServiceRuntime.spec` (WS-04.4); fixture `redis_url` (IF-19).
- Produces (IF-17 session part):
  - access and refresh claims `sid`, `amr`, `mfa`;
  - `SessionRevoker(session_manager, blacklist, *, access_token_ttl: timedelta)` with `async revoke_session(session_id: UUID, *, reason: str) -> None`, `async revoke_all(user_id: UUID, *, reason: str) -> int` and `async is_revoked(session_id: UUID) -> bool`. Each revocation deletes the session record and writes the sid key until `now + access_token_ttl + 60 s`; backend errors raise `IdentityBackendUnavailable`. WS-14.17 revokes on consent revocation through `session_manager.revoke_all_user_sessions`, which WS-08.13 routes through the same sid revocation.
  - `build_session_security(*, redis_client: Any | None, strict: bool, access_token_ttl: timedelta, session_config: SessionConfig) -> SessionSecurity` (dataclass `blacklist`, `session_manager`, `session_revoker`). In strict environments without a client it raises `RuntimeError` naming `REDIS_URL`; otherwise it uses `RedisSessionManager` with a client and `SessionManager` with a WARNING without one.
  - error codes `SESSION_EXPIRED`, `IDENTITY_BACKEND_UNAVAILABLE`.

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_212_session_bound_refresh.py::test_mvp_212_stolen_refresh_token_dies_with_its_session` and the real-infra file
```python
"""MVP-212: refresh tokens are bound to their session; logout and password change kill that session everywhere."""
from __future__ import annotations

from datetime import timedelta
from uuid import UUID

import pytest

from identity_testkit import PASSWORD, bearer, build_identity_app, claims, client_for, login, seed_user
from solace_security import auth as shared_auth
from src.auth import SessionConfig
from src.infrastructure import jwt_service as local_jwt

REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"
PASSWORD_CHANGE = "/api/v1/users/me/password"


async def test_mvp_212_stolen_refresh_token_dies_with_its_session():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        stolen = await login(c)
        changed = await c.post(PASSWORD_CHANGE, headers=bearer(stolen["access_token"]),
                               json={"current_password": PASSWORD, "new_password": "Another-Horse-7"})
        assert changed.status_code == 204
        fresh = await login(c, password="Another-Horse-7")
        replay = await c.post(REFRESH, json={"refresh_token": stolen["refresh_token"]})
        assert replay.status_code == 401
        assert (await c.post(REFRESH, json={"refresh_token": fresh["refresh_token"]})).status_code == 200


async def test_mvp_212_logout_ends_only_the_calling_session():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        a, b = await login(c), await login(c)
        assert (await c.post(LOGOUT, headers=bearer(a["access_token"]))).status_code == 204
        assert (await c.post(REFRESH, json={"refresh_token": a["refresh_token"]})).status_code == 401
        assert (await c.post(LOGOUT, headers=bearer(a["access_token"]))).status_code == 401
        assert (await c.post(REFRESH, json={"refresh_token": b["refresh_token"]})).status_code == 200


async def test_mvp_212_tokens_carry_the_session_and_method():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
    access, refresh = claims(tokens["access_token"]), claims(tokens["refresh_token"])
    assert access["sid"] == refresh["sid"] and UUID(access["sid"])
    assert (access["amr"], access["mfa"], access["type"]) == (["pwd"], False, "access")


async def test_mvp_212_logout_writes_the_shared_session_revocation():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        await c.post(LOGOUT, headers=bearer(tokens["access_token"]))
    assert await state.token_blacklist.is_session_revoked(claims(tokens["access_token"])["sid"]) is True


def test_mvp_212_user_service_uses_the_shared_blacklist_classes():
    assert local_jwt.InMemoryTokenBlacklist is shared_auth.InMemoryTokenBlacklist
    assert local_jwt.RedisTokenBlacklist is shared_auth.RedisTokenBlacklist


class _DownBlacklist(shared_auth.InMemoryTokenBlacklist):
    async def is_blacklisted(self, jti):
        raise ConnectionError("redis down")

    async def revoke_session(self, session_id, until):
        raise ConnectionError("redis down")


async def test_mvp_212_revocation_backend_errors_fail_closed():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        state.jwt_service._blacklist = _DownBlacklist()
        state.session_revoker._blacklist = state.jwt_service._blacklist
        r = await c.post(LOGOUT, headers=bearer(tokens["access_token"]))
    assert r.status_code == 503 and r.json()["error"]["code"] == "IDENTITY_BACKEND_UNAVAILABLE"


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_212_strict_identity_requires_redis(monkeypatch, env):
    from src.identity_wiring import build_session_security

    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        build_session_security(redis_client=None, strict=True, access_token_ttl=timedelta(minutes=15),
                               session_config=SessionConfig(secret_key="s" * 40))
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_212_logout_revokes_across_services.py
MVP-212 on real Redis: a user-service logout is seen by every other service through the shared sid key."""
from __future__ import annotations

import pytest
import redis.asyncio as aioredis

from identity_testkit import SECRET, bearer, build_identity_app, claims, client_for, login, seed_user
from solace_security.auth import AuthSettings, JWTManager
from solace_security.middleware import build_token_blacklist
from src.auth import RedisSessionManager, SessionConfig

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_212_logout_in_user_service_is_seen_by_another_service(redis_url):
    issuer_redis = aioredis.from_url(redis_url, decode_responses=True)
    other_redis = aioredis.from_url(redis_url, decode_responses=True)
    try:
        blacklist = build_token_blacklist(issuer_redis)
        sessions = RedisSessionManager(issuer_redis, SessionConfig(secret_key=SECRET))
        app, state = build_identity_app(token_blacklist=blacklist, session_manager=sessions)
        await seed_user(state)
        other_service = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=build_token_blacklist(other_redis))
        async with client_for(app) as c:
            tokens = await login(c)
            assert (await other_service.validate_access_token_async(tokens["access_token"])).success is True
            assert (await c.post("/api/v1/auth/logout", headers=bearer(tokens["access_token"]))).status_code == 204
        result = await other_service.validate_access_token_async(tokens["access_token"])
        assert (result.success, result.error_code) == (False, "TOKEN_REVOKED")
        sid = claims(tokens["access_token"])["sid"]
        assert await other_redis.get(f"token_blacklist:sid:{sid}") == "1"
        assert 0 < await other_redis.ttl(f"token_blacklist:sid:{sid}") <= 16 * 60
    finally:
        await issuer_redis.aclose()
        await other_redis.aclose()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_212_session_bound_refresh.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_212_logout_revokes_across_services.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. The replayed stolen refresh token returns 200 after the victim's new login, because refresh only checks that some session exists (MVP-212). Logout of A kills B's refresh (401). The claims test fails with `KeyError: 'sid'`. The class-identity test fails because user-service defines its own blacklist classes. The fail-closed test fails with `AttributeError` on `state.session_revoker` (and without it gets 204, because logout swallows the blacklist error). The strict test fails at `ImportError: cannot import name 'build_session_security'`. In the real-infra test the other service still accepts the logged-out token, and the sid key is absent.
- [ ] **Step 3: Implement** the jwt_service, `SessionRevoker`, route, dependency, OAuth and wiring changes listed under Files. Approach: `SessionRevoker.revoke_all` reads `get_user_sessions(user_id, active_only=False)` before deleting, so every sid is written to the blacklist.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.5: Refresh checks the token type first, re-reads the account and keeps an active session alive
**Covers:** MVP-228, MVP-214, MVP-330
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`verify_token` validates `type` against `expected_type` immediately after the signature check, before any blacklist or session lookup; a mismatch raises `TokenTypeMismatch(TokenInvalidError)`)
- Modify: `services/user-service/src/domain/entities.py` (new `User.can_refresh() -> tuple[bool, str | None]`: deleted or SUSPENDED/INACTIVE gives False; PENDING_VERIFICATION gives `(False, "Email verification required")`; ACTIVE gives True; a password-guess lock does not end existing sessions)
- Modify: `services/user-service/src/api_auth.py` (`refresh_token`:
  - `TokenTypeMismatch` returns 401 `INVALID_TOKEN_TYPE` with no session side effect;
  - the account is re-read with `user_repository.get_by_id`; when `can_refresh()` is false (pending is allowed outside strict environments), the session is revoked and the route returns 401 `ACCOUNT_UNAVAILABLE`;
  - `await session_manager.update_activity(sid)` runs before minting;
  - the new pair carries the re-read `email` and `role.value`;
  - error bodies use fixed messages, never `str(e)`.)
- Modify: `services/user-service/src/auth.py` (`RedisSessionManager.update_activity` also re-sets `user_sessions:<user_id>` with `ex=session_timeout_minutes*60`)
- Test: `services/user-service/tests/test_mvp_228_refresh_hardening.py`
**Interfaces:**
- Consumes: `SessionRevoker` (WS-08.4); `SessionManager.update_activity`, `RedisSessionManager.update_activity`.
- Produces: `TokenTypeMismatch`; `User.can_refresh()`; refresh error codes `INVALID_TOKEN_TYPE` and `ACCOUNT_UNAVAILABLE`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_228_refresh_hardening.py::test_mvp_228_revoked_access_token_posted_as_refresh_changes_nothing`
```python
"""MVP-228 / MVP-330 / MVP-214: refresh checks type first, re-reads the account and extends the session."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest

from identity_testkit import SECRET, bearer, build_identity_app, claims, client_for, login, seed_user
from src.auth import RedisSessionManager, SessionConfig, SessionManager
from src.domain.value_objects import AccountStatus, UserRole

REFRESH = "/api/v1/auth/refresh"


async def test_mvp_228_revoked_access_token_posted_as_refresh_changes_nothing():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        a, b = await login(c), await login(c)
        await c.post("/api/v1/auth/logout", headers=bearer(a["access_token"]))
        r = await c.post(REFRESH, json={"refresh_token": a["access_token"]})
        assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_TOKEN_TYPE"
        assert (await c.post(REFRESH, json={"refresh_token": b["refresh_token"]})).status_code == 200


async def test_mvp_330_refresh_rereads_the_role():
    app, state = build_identity_app()
    user = await seed_user(state, role=UserRole.PREMIUM)
    async with client_for(app) as c:
        tokens = await login(c)
        user.role = UserRole.USER
        await state.user_repository.update(user)
        r = await c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 200
    assert claims(r.json()["access_token"])["roles"] == ["user"]


async def test_mvp_330_suspended_account_cannot_refresh_and_loses_the_session():
    app, state = build_identity_app()
    user = await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        user.status = AccountStatus.SUSPENDED
        await state.user_repository.update(user)
        r = await c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == 401 and r.json()["error"]["code"] == "ACCOUNT_UNAVAILABLE"
    assert await state.session_manager.get_session(UUID(claims(tokens["refresh_token"])["sid"])) is None


async def test_mvp_214_refresh_keeps_an_active_session_alive():
    app, state = build_identity_app(session_manager=SessionManager(SessionConfig(secret_key=SECRET,
                                                                                 session_timeout_minutes=5)))
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        session = state.session_manager._sessions[UUID(claims(tokens["access_token"])["sid"])]
        session.last_activity -= timedelta(minutes=4)
        first = await c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
        assert first.status_code == 200
        session.last_activity -= timedelta(minutes=4)
        second = await c.post(REFRESH, json={"refresh_token": first.json()["refresh_token"]})
    assert second.status_code == 200


async def test_mvp_214_redis_activity_also_extends_the_user_index():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    manager = RedisSessionManager(redis, SessionConfig(secret_key=SECRET, session_timeout_minutes=5))
    user = type("U", (), {"user_id": uuid4()})()
    session = await manager.create_session(user=user)
    await redis.expire(f"user_sessions:{user.user_id}", 10)
    await manager.update_activity(session.session_id)
    assert await redis.ttl(f"user_sessions:{user.user_id}") > 290
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_228_refresh_hardening.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The revoked access token hits the "revoked" branch before any type check and revokes all of the owner's sessions, so B's refresh returns 401 (MVP-228). The demoted account's refreshed token still carries `["premium"]`, and the suspended account refreshes with 200 (MVP-330); the same staleness lets a demoted clinician keep clinician reads. The second refresh after 8 idle minutes returns 401 `SESSION_EXPIRED`, because refresh never calls `update_activity` (MVP-214). The Redis index TTL stays at 10.
- [ ] **Step 3: Implement** `TokenTypeMismatch`, the type-first order in `verify_token`, `User.can_refresh`, the refresh changes and the index-TTL write. The "revoked" reuse branch is still present for genuine refresh tokens, and WS-08.6 replaces it.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.6: Atomic refresh rotation with a 30-second grace for an exact retry; reuse after the grace revokes only that session
**Covers:** MVP-224
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.5, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Modify: `services/user-service/src/auth.py` (new `REFRESH_GRACE_SECONDS = 30`, `RotationOutcome`, `RefreshRotationStore` protocol, `InMemoryRefreshRotationStore`, `RedisRefreshRotationStore` and `ROTATE_LUA`)
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`generate_token_pair(..., refresh_jti: str | None = None)` uses the given jti for the refresh token; `TokenPair.refresh_jti`)
- Modify: `services/user-service/src/api_auth.py` (`refresh_token` drops the substring "revoke" logic. After the type, sid, live-session and account checks from WS-08.4/WS-08.5 (so a token of a dead session is a plain 401 and consumes nothing), it calls `rotation_store.consume(payload.jti, expires_at=payload.expires_at, child_jti=str(uuid4()))`:
  - `rotate`: mints the pair with `refresh_jti=outcome.child_jti`;
  - `grace`: re-mints the pair with the same `refresh_jti=outcome.child_jti`, so the family never forks;
  - `reuse`: `session_revoker.revoke_session(sid, reason="refresh_reuse")` and 401 `REFRESH_REUSED`;
  - a store error returns 503 `IDENTITY_BACKEND_UNAVAILABLE`.)
- Modify: `services/user-service/src/api_deps.py` (`get_rotation_store(request) -> RefreshRotationStore`)
- Modify: `services/user-service/src/identity_wiring.py` (`SessionSecurity.rotation_store`: Redis with a client, in-memory with a WARNING in development/test; strict without a client raises `RuntimeError("REDIS_URL is required for refresh rotation in strict environments")`)
- Modify: `services/user-service/tests/identity_testkit.py` (`rotation_store=InMemoryRefreshRotationStore()`)
- Modify: `services/user-service/tests/test_refresh_rotation.py` (the replay scenario uses an `InMemoryRefreshRotationStore` whose injected clock is advanced 31 s before the replay, and now expects that only the replayed token's session is revoked while a second session of the user survives)
- Test: `services/user-service/tests/test_mvp_224_refresh_rotation_grace.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_224_atomic_rotation_redis.py`
**Interfaces:**
- Consumes: `SessionRevoker.revoke_session` (WS-08.4); fixture `redis_url` (IF-19).
- Produces (IF-17 rotation): `@dataclass(frozen=True) RotationOutcome(kind: Literal["rotate", "grace", "reuse"], child_jti: str)`; `RefreshRotationStore.consume(jti: str, *, expires_at: datetime, child_jti: str) -> RotationOutcome`; `InMemoryRefreshRotationStore(*, grace_seconds: int = REFRESH_GRACE_SECONDS, clock: Callable[[], float] = time.monotonic)`; `RedisRefreshRotationStore(redis_client, *, grace_seconds: int = REFRESH_GRACE_SECONDS, key_prefix: str = "refresh_rot:")`; Redis key `refresh_rot:<jti>` = `"<child_jti>|<first_use_ms>"` with TTL until the refresh token's `exp`; refresh error code `REFRESH_REUSED`.

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_224_refresh_rotation_grace.py::test_mvp_224_retry_within_grace_returns_the_same_refresh_family` and the real-infra file
```python
"""MVP-224: rotation is atomic, an exact retry within 30 s is idempotent, and later reuse ends only that session."""
from __future__ import annotations

import asyncio

from identity_testkit import build_identity_app, claims, client_for, login, seed_user
from src.auth import InMemoryRefreshRotationStore

REFRESH = "/api/v1/auth/refresh"


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


async def test_mvp_224_retry_within_grace_returns_the_same_refresh_family():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        first = await c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
        retry = await c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
        assert (first.status_code, retry.status_code) == (200, 200)
        assert claims(first.json()["refresh_token"])["jti"] == claims(retry.json()["refresh_token"])["jti"]
        assert (await c.post(REFRESH, json={"refresh_token": first.json()["refresh_token"]})).status_code == 200


async def test_mvp_224_concurrent_refreshes_mint_one_child():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        r1, r2 = await asyncio.gather(c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]}),
                                      c.post(REFRESH, json={"refresh_token": tokens["refresh_token"]}))
    assert (r1.status_code, r2.status_code) == (200, 200)
    assert claims(r1.json()["refresh_token"])["jti"] == claims(r2.json()["refresh_token"])["jti"]


async def test_mvp_224_reuse_after_grace_revokes_only_that_session():
    clock = _Clock()
    app, state = build_identity_app(rotation_store=InMemoryRefreshRotationStore(clock=clock))
    await seed_user(state)
    async with client_for(app) as c:
        a, b = await login(c), await login(c)
        child = (await c.post(REFRESH, json={"refresh_token": a["refresh_token"]})).json()
        clock.now += 31
        replay = await c.post(REFRESH, json={"refresh_token": a["refresh_token"]})
        assert replay.status_code == 401 and replay.json()["error"]["code"] == "REFRESH_REUSED"
        assert (await c.post(REFRESH, json={"refresh_token": child["refresh_token"]})).status_code == 401
        assert (await c.post(REFRESH, json={"refresh_token": b["refresh_token"]})).status_code == 200
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_224_atomic_rotation_redis.py
MVP-224 on real Redis: the Lua swap admits exactly one rotation under concurrency, then grace, then reuse."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from src.auth import RedisRefreshRotationStore

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_224_lua_swap_is_atomic_then_grace_then_reuse(redis_url):
    client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        store = RedisRefreshRotationStore(client, grace_seconds=2)
        jti = str(uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(days=1)
        outcomes = await asyncio.gather(*(store.consume(jti, expires_at=expires_at, child_jti=str(uuid4()))
                                          for _ in range(20)))
        assert Counter(o.kind for o in outcomes) == {"rotate": 1, "grace": 19}
        assert len({o.child_jti for o in outcomes}) == 1
        assert 0 < await client.ttl(f"refresh_rot:{jti}") <= 86400
        await asyncio.sleep(2.3)
        late = await store.consume(jti, expires_at=expires_at, child_jti=str(uuid4()))
        assert late.kind == "reuse" and late.child_jti == outcomes[0].child_jti
    finally:
        await client.aclose()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_224_refresh_rotation_grace.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_224_atomic_rotation_redis.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'InMemoryRefreshRotationStore'`. The defects behind it: a lost-response retry of the same refresh token hits the blacklist "revoked" branch and revokes every session of the user. Rotation is a GET-then-SET blacklist, so two simultaneous refreshes both mint pairs (MVP-224).
- [ ] **Step 3: Implement** the stores and the refresh flow listed under Files. `ROTATE_LUA` (`KEYS[1]` = `refresh_rot:<jti>`, `ARGV` = child jti, TTL seconds, grace milliseconds), using Redis `TIME` so all instances share one clock:
```lua
local t = redis.call('TIME')
local now_ms = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local v = redis.call('GET', KEYS[1])
if not v then
  redis.call('SET', KEYS[1], ARGV[1] .. '|' .. now_ms, 'EX', tonumber(ARGV[2]))
  return {'rotate', ARGV[1]}
end
local sep = string.find(v, '|', 1, true)
local child, first_ms = string.sub(v, 1, sep - 1), tonumber(string.sub(v, sep + 1))
if now_ms - first_ms <= tonumber(ARGV[3]) then return {'grace', child} end
return {'reuse', child}
```
The in-memory store does the same under an `asyncio.Lock` with its injected clock. No token material is stored, only jtis.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.7: Registration answers 202 for every address and validates with one model; the auth contract document
**Covers:** MVP-217, MVP-221
**Severity:** medium
**Wave:** 2
**Phase:** B
**Depends on:** WS-08.6
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-11 (default assumed: the code is the contract and `docs/api/auth.md` documents it)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/domain/entities.py` (new module functions `normalize_email(value: str) -> str`, which uses `email_validator.validate_email(value, check_deliverability=False, allow_smtputf8=False)` and then lower-cases the whole address, and `normalize_locale(value: str) -> str`, which accepts `^[a-z]{2,3}([-_][A-Za-z]{2})?$` case-insensitively and returns `ll` or `ll-CC`; `User.validate_email` and `User.validate_locale` call them; `EMAIL_REGEX` is removed)
- Modify: `services/user-service/src/api_auth.py`:
  - `RegisterRequest` and `ResendVerificationRequest` validate with `normalize_email`/`normalize_locale`;
  - `register` has status 202 and returns `{"status": "accepted"}` for both a new and an existing address. On `EMAIL_EXISTS` it schedules `BackgroundTasks` work that loads the existing user and calls `user_service.send_account_exists_email(user)`. The background work catches every exception (including `AttributeError` while R-3 is unlanded) and logs WARNING `account_exists_notice_failed` with `error_type` only, never the address, so the notice is best-effort and the 202 never depends on it. Any other failure returns 503 `REGISTRATION_UNAVAILABLE` and logs only the error code. The `registrations` stat counts real creations only.
- Create: `docs/api/auth.md` (the W2 contract: register 202, login and its error codes, refresh with sid binding and the 30 s grace, logout per session, revoke-token, verify-email body, resend, password change, token claims `sub type roles sid amr mfa iss aud iat exp`, error envelope)
- Modify: `services/user-service/tests/test_api.py` (`TestRegistration::test_register_success` and `::test_register_duplicate_email` expect 202 `{"status": "accepted"}`)
- Modify: `services/user-service/tests/test_oauth_endpoints.py` (its two `register` calls assert 202)
- Test: `services/user-service/tests/test_mvp_217_registration_contract.py`
- Test: `services/user-service/tests/test_mvp_217_auth_contract_doc.py`
**Interfaces:**
- Consumes: `UserService.send_account_exists_email(user: User) -> bool` (open request R-3, no producing WS-14 task yet); notification template `account_exists` (open request R-4, no producing WS-13 task yet). The tests use a spy, so this task does not wait on either; the register-timing half of MVP-217 needs open request R-2 (send the verification email after commit), which no WS-14 task produces yet.
- Produces: `normalize_email`, `normalize_locale` (`src.domain.entities`); `POST /api/v1/auth/register` -> 202 `{"status": "accepted"}` (IF-17); error code `REGISTRATION_UNAVAILABLE`; `docs/api/auth.md`, which every later task that adds a route must extend (the doc test enumerates the routes).

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_217_registration_contract.py::test_mvp_217_register_gives_the_same_202_for_new_and_existing_addresses` and the doc test
```python
"""MVP-217 / MVP-221: no email-exists oracle; the request model and the entity accept and reject the same input."""
from __future__ import annotations

import pytest

from identity_testkit import PASSWORD, build_identity_app, client_for

REGISTER = "/api/v1/auth/register"


def _body(email: str, **extra) -> dict:
    return {"email": email, "password": PASSWORD, "display_name": "Synthetic", **extra}


async def test_mvp_217_register_gives_the_same_202_for_new_and_existing_addresses():
    app, state = build_identity_app()
    notified: list[str] = []

    async def send_account_exists_email(user):
        notified.append(user.email)
        return True

    state.user_service.send_account_exists_email = send_account_exists_email
    async with client_for(app) as c:
        first = await c.post(REGISTER, json=_body("alice@example.com"))
        second = await c.post(REGISTER, json=_body("alice@example.com", display_name="Someone Else"))
    assert (first.status_code, second.status_code) == (202, 202)
    assert first.json() == second.json() == {"status": "accepted"}
    assert notified == ["alice@example.com"]
    assert await state.user_repository.count() == 1


@pytest.mark.parametrize("email,stored", [("o'brien@example.com", "o'brien@example.com"),
                                          ("first&last@example.com", "first&last@example.com"),
                                          ("Mixed.Case@Example.COM", "mixed.case@example.com")])
async def test_mvp_221_addresses_the_request_model_accepts_can_register(email, stored):
    app, state = build_identity_app()
    async with client_for(app) as c:
        r = await c.post(REGISTER, json=_body(email))
    assert r.status_code == 202
    assert await state.user_repository.get_by_email(stored) is not None


@pytest.mark.parametrize("locale,stored", [("en", "en"), ("en_US", "en-US"), ("en-us", "en-US"), ("pt-BR", "pt-BR")])
async def test_mvp_221_locales_are_normalized_not_rejected(locale, stored):
    app, state = build_identity_app()
    async with client_for(app) as c:
        r = await c.post(REGISTER, json=_body("loc@example.com", locale=locale))
    assert r.status_code == 202
    assert (await state.user_repository.get_by_email("loc@example.com")).locale == stored


@pytest.mark.parametrize("email", ["not-an-email", "jos\u00e9@example.com"])
async def test_mvp_221_invalid_input_is_a_422_without_internal_detail(email):
    app, _state = build_identity_app()
    async with client_for(app) as c:
        r = await c.post(REGISTER, json=_body(email))
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "validation error for User" not in r.text and "Failed to create user" not in r.text
```
```python
"""services/user-service/tests/test_mvp_217_auth_contract_doc.py
IF-17 contract document: every auth route the app serves is documented (starting with the 202 registration contract)."""
from __future__ import annotations

from pathlib import Path

from fastapi.routing import APIRoute

from identity_testkit import build_identity_app

DOC = Path(__file__).resolve().parents[3] / "docs" / "api" / "auth.md"


def test_every_auth_route_is_documented():
    app, _state = build_identity_app()
    paths = sorted({r.path for r in app.routes if isinstance(r, APIRoute)
                    and r.path.startswith(("/api/v1/auth", "/api/v1/users/me"))})
    text = DOC.read_text(encoding="utf-8")
    assert [p for p in paths if p not in text] == []
    assert "202" in text and '"status": "accepted"' in text
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_217_registration_contract.py D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_217_auth_contract_doc.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- The first registration returns 201 and the second 409 `Email already registered`, which is the account-enumeration oracle (MVP-217).
- `o'brien@` and `first&last@` return 400 with `Failed to create user: 1 validation error for User`. `en` and `en_US` are rejected the same way, and `jos\u00e9@` passes the request model but fails inside `User()` (MVP-221).
- The doc test fails with `FileNotFoundError` for `docs/api/auth.md`.
- The `not-an-email` case already passes.
- [ ] **Step 3: Implement** `normalize_email`, `normalize_locale`, the entity validators, the request models, the 202 `register` and `docs/api/auth.md`. Approach: both outcomes return immediately after the password hash and the create attempt; the existing-owner email is sent from a background task, so the response does not wait on the notification call.
- [ ] **Step 4: Run them and confirm they pass**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_entities.py::test_invalid_email_format_raises_error` still raises for `invalid-email`, and `::test_invalid_locale_format_raises_error` still raises for `invalid`)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.8: POST /auth/reauth and a re-authenticated, purpose-bound DELETE /users/me
**Covers:** MVP-229
**Severity:** medium
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.7, WS-14.16
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: deletion is immediate and irreversible after re-authentication and confirmation; no grace window)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`TokenType.REAUTH = "reauth"`; `REAUTH_TOKEN_TTL_SECONDS = 300`; `create_reauth_token(user_id: UUID, session_id: UUID, purpose: str) -> str`; `async verify_reauth_token(token: str, *, user_id: UUID, session_id: UUID, purpose: str) -> str`, which returns the jti and raises `TokenInvalidError` for a wrong type, user, session, an expired token or a blacklisted jti, and `ReauthPurposeMismatch(TokenInvalidError)` for a wrong purpose)
- Modify: `services/user-service/src/auth.py` (new `FailureCounter` protocol with `async incr(key: str, window_seconds: int) -> int` and `async clear(key: str) -> None`; `InMemoryFailureCounter`; `RedisFailureCounter` (INCR, then EXPIRE with NX); `StepUpGuard(counter, revoker, *, max_failures=5, window_seconds=900)` with `async record_failure(session_id: UUID) -> bool` (True once the session was revoked) and `async clear(session_id: UUID) -> None`, key `stepup_fail:<sid>`)
- Modify: `services/user-service/src/api_auth.py`:
  - `ReauthRequest{password: str | None, code: str | None, purpose: Literal["account_delete", "sensitive"] = "sensitive"}`;
  - `POST /auth/reauth` requires an access token and checks the password with `verify_password_async`. Success returns 200 `{"reauth_token", "expires_in": 300}` and clears the counter. Failure returns 401 `INVALID_CREDENTIALS` and counts it; the fifth failure revokes the session and returns 401 `SESSION_REVOKED`. `code` is accepted from WS-08.17 and returns 400 `MFA_NOT_ENROLLED` until then.
  - `DELETE /users/me` requires `X-Reauth-Token` carrying purpose `account_delete` for the caller's user and sid. A missing or invalid token returns 401 `REAUTH_REQUIRED`, and a wrong purpose returns 403 `REAUTH_PURPOSE_MISMATCH`. The jti is blacklisted before `user_service.delete_user(user_id=..., reason="user_request")` runs. The `DeleteUserResult` is mapped as WS-14 asks in R-14-9: `success` runs `session_revoker.revoke_all(user_id, reason="account_deleted")` and returns 202 `{"erasure_request_id": str(result.erasure_request_id)}`; error code `USER_NOT_FOUND` returns 404 `USER_NOT_FOUND`; any other failure (`DELETE_FAILED`) returns 503 `ERASURE_INCOMPLETE`. Sessions are revoked only on success, and `result.error` never reaches the body.
- Modify: `services/user-service/src/api_deps.py` (`get_step_up_guard(request) -> StepUpGuard`)
- Modify: `services/user-service/src/identity_wiring.py` (`SessionSecurity.step_up_guard`: `RedisFailureCounter` with a client, `InMemoryFailureCounter` with a WARNING outside strict environments; strict without a client raises `RuntimeError` naming `REDIS_URL`)
- Modify: `services/user-service/tests/identity_testkit.py` (`failure_counter=InMemoryFailureCounter()`, `step_up_guard=StepUpGuard(...)`)
- Modify: `docs/api/auth.md` (reauth and account deletion sections)
- Test: `services/user-service/tests/test_mvp_229_reauth_and_account_deletion.py`
**Interfaces:**
- Consumes: `UserService.delete_user(user_id: UUID, reason: str | None) -> DeleteUserResult` with `success`, `error_code` and `erasure_request_id: UUID | None` (WS-14.16; IF-16 step 1 returns the id inside the result object, a WS-14 ledger Ruling); `SessionRevoker` (WS-08.4); `verify_password_async` (WS-08.3).
- Produces (IF-17): `POST /api/v1/auth/reauth` -> `{reauth_token, expires_in}` (claims `sub`, `type="reauth"`, `sid`, `purpose`, `jti`, `exp` = 5 min); `DELETE /api/v1/users/me` with header `X-Reauth-Token` -> 202 `{erasure_request_id}`, 404 `USER_NOT_FOUND` or 503 `ERASURE_INCOMPLETE` (IF-16; the delete half of R-14-9, requested by WS-14.16); `StepUpGuard` (WS-08.20 reuses it); error codes `REAUTH_REQUIRED`, `REAUTH_PURPOSE_MISMATCH`, `SESSION_REVOKED`, `MFA_NOT_ENROLLED`, `ERASURE_INCOMPLETE`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_229_reauth_and_account_deletion.py::test_mvp_229_delete_with_a_bare_access_token_is_refused`
```python
"""MVP-229: irreversible erasure needs a fresh, single-use, purpose-bound re-authentication."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from identity_testkit import PASSWORD, bearer, build_identity_app, client_for, login, seed_user

ME = "/api/v1/users/me"
REAUTH = "/api/v1/auth/reauth"


def _spy_delete(state):
    deleted: list[UUID] = []
    request_id = uuid4()

    async def delete_user(user_id, reason=None):  # WS-14.16 shape: DeleteUserResult(success, error, error_code, erasure_request_id)
        deleted.append(user_id)
        return SimpleNamespace(success=True, error=None, error_code=None, erasure_request_id=request_id)

    state.user_service.delete_user = delete_user
    return deleted, request_id


async def test_mvp_229_delete_with_a_bare_access_token_is_refused():
    app, state = build_identity_app()
    await seed_user(state)
    deleted, _ = _spy_delete(state)
    async with client_for(app) as c:
        tokens = await login(c)
        r = await c.delete(ME, headers=bearer(tokens["access_token"]))
    assert r.status_code == 401 and r.json()["error"]["code"] == "REAUTH_REQUIRED"
    assert deleted == []


async def test_mvp_229_reauth_then_delete_returns_the_erasure_request_once():
    app, state = build_identity_app()
    user = await seed_user(state)
    deleted, request_id = _spy_delete(state)
    async with client_for(app) as c:
        tokens = await login(c)
        ra = await c.post(REAUTH, headers=bearer(tokens["access_token"]),
                          json={"password": PASSWORD, "purpose": "account_delete"})
        assert ra.status_code == 200 and ra.json()["expires_in"] == 300
        headers = {**bearer(tokens["access_token"]), "X-Reauth-Token": ra.json()["reauth_token"]}
        r = await c.delete(ME, headers=headers)
        assert r.status_code == 202 and r.json() == {"erasure_request_id": str(request_id)}
        again = await c.delete(ME, headers=headers)
    assert again.status_code == 401 and deleted == [user.user_id]


async def test_mvp_229_reauth_token_is_bound_to_user_session_and_purpose():
    app, state = build_identity_app()
    await seed_user(state)
    await seed_user(state, email="bob@example.com")
    deleted, _ = _spy_delete(state)
    async with client_for(app) as c:
        alice_a, alice_b = await login(c), await login(c)
        bob = await login(c, email="bob@example.com")
        bob_token = (await c.post(REAUTH, headers=bearer(bob["access_token"]),
                                  json={"password": PASSWORD, "purpose": "account_delete"})).json()["reauth_token"]
        other_session = (await c.post(REAUTH, headers=bearer(alice_b["access_token"]),
                                      json={"password": PASSWORD, "purpose": "account_delete"})).json()["reauth_token"]
        sensitive = (await c.post(REAUTH, headers=bearer(alice_a["access_token"]),
                                  json={"password": PASSWORD})).json()["reauth_token"]
        cases = [(bob_token, 401), (other_session, 401), (sensitive, 403)]
        for token, status in cases:
            r = await c.delete(ME, headers={**bearer(alice_a["access_token"]), "X-Reauth-Token": token})
            assert r.status_code == status
    assert deleted == []


async def test_mvp_229_wrong_reauth_passwords_end_the_session_on_the_fifth_try():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        codes = [(await c.post(REAUTH, headers=bearer(tokens["access_token"]),
                               json={"password": "Wrong-Horse-1"})).json()["error"]["code"] for _ in range(5)]
        refresh = await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert codes == ["INVALID_CREDENTIALS"] * 4 + ["SESSION_REVOKED"]
    assert refresh.status_code == 401


@pytest.mark.parametrize("error_code,status,code", [("USER_NOT_FOUND", 404, "USER_NOT_FOUND"),
                                                    ("DELETE_FAILED", 503, "ERASURE_INCOMPLETE")])
async def test_mvp_229_failed_erasure_is_an_error_and_keeps_the_session(error_code, status, code):
    app, state = build_identity_app()
    await seed_user(state)

    async def delete_user(user_id, reason=None):
        return SimpleNamespace(success=False, error="internal detail", error_code=error_code, erasure_request_id=None)

    state.user_service.delete_user = delete_user
    async with client_for(app) as c:
        tokens = await login(c)
        ra = await c.post(REAUTH, headers=bearer(tokens["access_token"]),
                          json={"password": PASSWORD, "purpose": "account_delete"})
        r = await c.delete(ME, headers={**bearer(tokens["access_token"]), "X-Reauth-Token": ra.json()["reauth_token"]})
        refresh = await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r.status_code == status and r.json()["error"]["code"] == code
    assert "internal detail" not in r.text
    assert refresh.status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_229_reauth_and_account_deletion.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `DELETE /users/me` with a bare access token returns 204, and the spy records the erasure, because the handler depends only on `get_current_user` (MVP-229). `/auth/reauth` returns 404, so the other tests fail at the reauth call (`KeyError: 'reauth_token'`); today's route also maps every failed result to 400 with `result.error` as the detail (`services/user-service/src/api.py:984-988`).
- [ ] **Step 3: Implement** the reauth token functions, `FailureCounter`, `StepUpGuard`, the two routes, the dependency, the wiring and the doc sections. The reauth token jti is blacklisted before `delete_user` runs, so a concurrent second DELETE cannot reuse it.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass, including `test_mvp_217_auth_contract_doc.py` with the new route documented
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.9: Google endpoints come from settings, and id_token verification uses a cached async JWKS
**Covers:** MVP-222
**Severity:** medium
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.8, WS-05.12 (stub IdP uses these settings in E2E)
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/oauth_google.py`:
  - `GoogleOAuthSettings` gains `authorize_url`, `token_url`, `jwks_url` and `issuer` (defaults are Google's current values), and `build_authorization_url` and `exchange_code_for_tokens` use them;
  - new `JWKSCache(jwks_url: str, *, client_factory: Callable[[], httpx.AsyncClient], ttl_seconds: int = 3600, timeout_seconds: float = 5.0, min_refresh_interval_seconds: float = 60.0)` with `async get_signing_key(kid: str)`. It honours `Cache-Control: max-age`, refetches on an unknown kid at most once per interval and serializes fetches with an `asyncio.Lock`.
  - new `async verify_id_token_async(id_token: str, settings: GoogleOAuthSettings, *, jwks: JWKSCache, expected_nonce: str | None = None, leeway_seconds: int = 30) -> GoogleIdentity`. It checks RS256, `aud == client_id` and `iss == settings.issuer` (also `accounts.google.com` when the issuer is Google's default), requires `exp`, `iat` and `sub`, requires `email_verified`, and checks `nonce` when one is expected.
  - `exchange_code_for_tokens` logs only the status, never the response body;
  - the per-call `PyJWKClient` path (`_google_signing_key`) is deleted.
- Modify: `services/user-service/src/oauth.py` (`google_oauth_settings_from_env` reads `GOOGLE_OAUTH_AUTHORIZE_URL`, `GOOGLE_OAUTH_TOKEN_URL`, `GOOGLE_OAUTH_JWKS_URL` and `GOOGLE_OAUTH_ISSUER`; the callback awaits the verifier when it returns an awaitable, and the default verifier is `verify_id_token_async` with the state's `oauth_jwks_cache`)
- Modify: `services/user-service/src/identity_wiring.py` (builds one `JWKSCache` per process when OAuth is enabled and stores it as `oauth_jwks_cache`)
- Modify: `services/user-service/tests/test_oauth_google.py` (cases that injected `signing_key_resolver` into the deleted sync verifier move to `verify_id_token_async` with a `JWKSCache` on an `httpx.MockTransport`)
- Test: `services/user-service/tests/test_mvp_222_google_jwks_cache.py`
**Interfaces:**
- Consumes: env names `GOOGLE_OAUTH_ISSUER`, `GOOGLE_OAUTH_AUTHORIZE_URL`, `GOOGLE_OAUTH_TOKEN_URL`, `GOOGLE_OAUTH_JWKS_URL` (IF-02).
- Produces: `JWKSCache`, `verify_id_token_async` and the new `GoogleOAuthSettings` fields; WS-08.10 passes `expected_nonce`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_222_google_jwks_cache.py::test_mvp_222_jwks_is_fetched_once_for_repeated_sign_ins`
```python
"""MVP-222: one cached JWKS per process, fetched asynchronously; endpoints and issuer come from settings."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import jwt
import pytest
import structlog
from cryptography.hazmat.primitives.asymmetric import rsa

from identity_testkit import ticks_during
from src.infrastructure.oauth_google import (GoogleOAuthSettings, JWKSCache, exchange_code_for_tokens,
                                             verify_id_token_async)
from src.oauth import google_oauth_settings_from_env

ISSUER = "https://idp.test"
SETTINGS = GoogleOAuthSettings(client_id="client-1", client_secret="s", redirect_uri="https://auth.example/cb",
                               issuer=ISSUER, jwks_url=f"{ISSUER}/jwks", token_url=f"{ISSUER}/token",
                               authorize_url=f"{ISSUER}/authorize")
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key())), "kid": "k1", "alg": "RS256", "use": "sig"}


def _id_token(kid="k1", iss=ISSUER, nonce=None) -> str:
    now = int(time.time())
    body = {"iss": iss, "aud": "client-1", "sub": "g-sub-1", "email": "user@example.com", "email_verified": True,
            "iat": now, "exp": now + 300}
    if nonce:
        body["nonce"] = nonce
    return jwt.encode(body, KEY, algorithm="RS256", headers={"kid": kid})


def _cache(counter: dict, delay: float = 0.0, **kw) -> JWKSCache:
    async def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        await asyncio.sleep(delay)
        return httpx.Response(200, json={"keys": [JWK]}, headers={"Cache-Control": "public, max-age=3600"})

    return JWKSCache(SETTINGS.jwks_url, client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


async def test_mvp_222_jwks_is_fetched_once_for_repeated_sign_ins():
    counter = {"n": 0}
    cache = _cache(counter)
    for _ in range(3):
        identity = await verify_id_token_async(_id_token(), SETTINGS, jwks=cache)
    assert counter["n"] == 1 and identity.provider_user_id == "g-sub-1"


async def test_mvp_222_slow_jwks_does_not_block_the_event_loop():
    _identity, ticks = await ticks_during(verify_id_token_async(_id_token(), SETTINGS, jwks=_cache({"n": 0}, delay=0.3)))
    assert ticks >= 8


async def test_mvp_222_unknown_kid_refetches_at_most_once_per_interval():
    counter = {"n": 0}
    cache = _cache(counter)
    await verify_id_token_async(_id_token(), SETTINGS, jwks=cache)
    with pytest.raises(ValueError):
        await verify_id_token_async(_id_token(kid="k2"), SETTINGS, jwks=cache)
    assert counter["n"] == 1


async def test_mvp_222_issuer_and_nonce_are_enforced():
    cache = _cache({"n": 0})
    with pytest.raises(ValueError):
        await verify_id_token_async(_id_token(iss="https://evil.test"), SETTINGS, jwks=cache)
    with pytest.raises(ValueError):
        await verify_id_token_async(_id_token(nonce="n-1"), SETTINGS, jwks=cache, expected_nonce="n-2")


def test_mvp_222_endpoints_and_issuer_come_from_env(monkeypatch):
    for name, value in {"GOOGLE_OAUTH_CLIENT_ID": "c", "GOOGLE_OAUTH_CLIENT_SECRET": "s",
                        "GOOGLE_OAUTH_REDIRECT_URI": "https://auth.example/cb", "GOOGLE_OAUTH_ISSUER": ISSUER,
                        "GOOGLE_OAUTH_AUTHORIZE_URL": f"{ISSUER}/authorize", "GOOGLE_OAUTH_TOKEN_URL": f"{ISSUER}/token",
                        "GOOGLE_OAUTH_JWKS_URL": f"{ISSUER}/jwks"}.items():
        monkeypatch.setenv(name, value)
    settings = google_oauth_settings_from_env()
    assert (settings.issuer, settings.authorize_url, settings.token_url, settings.jwks_url) == (
        ISSUER, f"{ISSUER}/authorize", f"{ISSUER}/token", f"{ISSUER}/jwks")


async def test_token_exchange_failure_never_logs_the_response_body():
    async def handler(request):
        return httpx.Response(400, text='{"error":"invalid_grant","id_token":"synthetic-secret-xyz"}')

    with structlog.testing.capture_logs() as logs, pytest.raises(RuntimeError):
        await exchange_code_for_tokens(SETTINGS, code="c", code_verifier="v",
                                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert "synthetic-secret-xyz" not in repr(logs)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_222_google_jwks_cache.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'JWKSCache'`. The defects behind it: `verify_id_token` builds a new `PyJWKClient` per call, so the cache is discarded, and it does a blocking urllib fetch with a 30 s timeout on the event loop (MVP-222). `GoogleOAuthSettings` has no `issuer`/`jwks_url` fields, and the token exchange logs up to 500 characters of the response body.
- [ ] **Step 3: Implement** `JWKSCache`, `verify_id_token_async`, the settings fields and env reads, the callback's awaitable support and the body-free failure log. Approach: the cache holds `{kid: key}` with an expiry, and keys are built with `jwt.PyJWK(jwk).key`, so no network call happens outside the cache's own async client.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass, including the moved `test_oauth_google.py` cases (this lane edits that file because the code under test is `oauth_google.py`)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.10: Google sign-in hands off by an allow-listed 302 with a one-time code, bound to the browser by a state cookie
**Covers:** MVP-104, MVP-223
**Severity:** high
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.9, WS-03.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-11 (default assumed: OAuth returns by 302 with a one-time code that is exchanged by POST)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/oauth.py`:
  - **Settings.** New `OAuthConfigError`. `google_oauth_settings_from_env()` raises it in strict environments when OAuth is enabled but `GOOGLE_OAUTH_REDIRECT_URI` or `OAUTH_ALLOWED_REDIRECT_ORIGINS` is missing; the localhost default applies only in development/test. New `parse_allowed_origins(raw: str) -> frozenset[tuple[str, str]]` and `redirect_allowed(uri: str, allowed) -> bool`: the redirect's (scheme, netloc) must match an allowed origin, and `javascript`, `data`, `file`, userinfo and fragments are refused.
  - **State store.** `PendingAuthorization(code_verifier, redirect_uri, nonce, binding_hash)` replaces the bare verifier in both state stores, with `put(state, pending, *, ttl_seconds)` and `pop(state) -> PendingAuthorization | None`.
  - **Code store.** New `InMemoryOneTimeCodeStore(clock=time.monotonic)` and `RedisOneTimeCodeStore(redis)` with `put(code: str, payload: dict, *, ttl_seconds: int = OAUTH_CODE_TTL_SECONDS)` and `take(code) -> dict | None`. They store the sha256 of the code under `oauth:google:code:<hash>`, and Redis uses GETDEL.
  - **`/start?redirect_uri=`.** Returns 400 `REDIRECT_URI_NOT_ALLOWED` unless the URI is allow-listed. It sends `nonce` to the IdP and sets cookie `solace_oauth_state` to a random binding (`HttpOnly`, `SameSite=Lax`, `Path=/api/v1/auth/oauth/google`, `Max-Age=600`, and `Secure` when the configured callback URI is https); the store keeps only `sha256(binding)`.
  - **`/callback`.** Returns 400 `OAUTH_STATE_MISMATCH` (JSON, no redirect) on an unknown state or a missing or mismatched cookie. It verifies the id_token with `expected_nonce`, then 302-redirects to `redirect_uri?code=<one-time code>` (60 s), or to `redirect_uri?error=<account_unavailable|oauth_failed>`, and clears the cookie. Tokens never appear in the Location, the body or a log line.
  - **`POST /auth/oauth/google/exchange {code}`.** Takes the code once (400 `INVALID_CODE` otherwise), re-reads the user, creates the session and returns `TokenResponse` with `amr=["google"]`.
- Modify: `services/user-service/src/identity_wiring.py` (Redis code and state stores on `app.state.service.oauth_code_store` and `.oauth_state_store`; strict without Redis raises; the allow-list is parsed once at startup into `.oauth_allowed_origins`)
- Modify: `services/user-service/tests/identity_testkit.py` (`enable_google_oauth(..., allowed_origins=("https://app.example",))` also sets `oauth_code_store=InMemoryOneTimeCodeStore()` and `oauth_allowed_origins`; `google_round_trip(client, *, redirect_uri="https://app.example/auth/done")` now sends `redirect_uri` and returns the callback response)
- Modify: `services/user-service/tests/test_oauth_endpoints.py` (start, callback and single-use tests follow the 302/code contract; the suspended and locked regressions expect `error=account_unavailable`)
- Modify: `docs/api/auth.md` (OAuth section: start, callback, exchange; `GOOGLE_OAUTH_REDIRECT_URI`, `OAUTH_ALLOWED_REDIRECT_ORIGINS` and the `GOOGLE_OAUTH_*` endpoint variables; cookie and code semantics; redirect error values)
- Test: `services/user-service/tests/test_mvp_104_oauth_code_handoff.py`
**Interfaces:**
- Consumes: `verify_id_token_async(..., expected_nonce=)` (WS-08.9); env `OAUTH_ALLOWED_REDIRECT_ORIGINS`, `GOOGLE_OAUTH_REDIRECT_URI` (IF-02 OAuth rows, written to the registry and compose by WS-03.1 in W0; the strict-required flags and the STAGING-BRINGUP text stay in open request R-12).
- Produces (IF-17): `GET /api/v1/auth/oauth/google/start?redirect_uri=`; `GET /api/v1/auth/oauth/google/callback` -> 302 `?code=` or `?error=`; `POST /api/v1/auth/oauth/google/exchange {code}` -> token pair; `OAUTH_CODE_TTL_SECONDS = 60`; cookie `solace_oauth_state`; `PendingAuthorization`; the one-time code store (WS-08.12 reuses it for link codes); error codes `REDIRECT_URI_NOT_ALLOWED`, `OAUTH_STATE_MISMATCH`, `INVALID_CODE`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_104_oauth_code_handoff.py::test_mvp_104_callback_redirects_with_a_one_time_code_not_tokens`
```python
"""MVP-104 / MVP-223: allow-listed 302 handoff with a one-time code; the callback must come from the same browser."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from identity_testkit import build_identity_app, client_for, enable_google_oauth
from src.infrastructure.oauth_google import GoogleIdentity
from src.oauth import InMemoryOneTimeCodeStore, OAuthConfigError, google_oauth_settings_from_env

START = "/api/v1/auth/oauth/google/start"
CALLBACK = "/api/v1/auth/oauth/google/callback"
EXCHANGE = "/api/v1/auth/oauth/google/exchange"
APP = "https://app.example/auth/done"
IDENTITY = GoogleIdentity(provider_user_id="g-1", email="new@example.com", email_verified=True)


def _q(url: str) -> dict:
    return parse_qs(urlsplit(url).query)


async def _start(client, redirect_uri=APP):
    return await client.get(START, params={"redirect_uri": redirect_uri})


async def test_mvp_104_callback_redirects_with_a_one_time_code_not_tokens():
    app, state = build_identity_app()
    enable_google_oauth(state, identity=IDENTITY)
    async with client_for(app) as c:
        start = await _start(c)
        assert start.status_code == 302
        cb = await c.get(CALLBACK, params={"code": "g-code", "state": _q(start.headers["location"])["state"][0]})
        assert cb.status_code == 302
        loc = urlsplit(cb.headers["location"])
        assert f"{loc.scheme}://{loc.netloc}{loc.path}" == APP and set(_q(cb.headers["location"])) == {"code"}
        assert "access_token" not in cb.text and "refresh_token" not in cb.text
        code = _q(cb.headers["location"])["code"][0]
        first = await c.post(EXCHANGE, json={"code": code})
        second = await c.post(EXCHANGE, json={"code": code})
    assert first.status_code == 200 and first.json()["access_token"]
    assert second.status_code == 400 and second.json()["error"]["code"] == "INVALID_CODE"


async def test_mvp_104_code_expires_after_60_seconds():
    clock = type("Clock", (), {"now": 0.0, "__call__": lambda self: self.now})()
    app, state = build_identity_app()
    enable_google_oauth(state, identity=IDENTITY)
    state.oauth_code_store = InMemoryOneTimeCodeStore(clock=clock)
    async with client_for(app) as c:
        start = await _start(c)
        cb = await c.get(CALLBACK, params={"code": "g", "state": _q(start.headers["location"])["state"][0]})
        clock.now += 61
        r = await c.post(EXCHANGE, json={"code": _q(cb.headers["location"])["code"][0]})
    assert r.status_code == 400


@pytest.mark.parametrize("redirect", ["https://evil.example/x", "javascript:alert(1)", "https://user@app.example/x", ""])
async def test_mvp_104_start_rejects_unlisted_redirects(redirect):
    app, state = build_identity_app()
    enable_google_oauth(state, identity=IDENTITY)
    async with client_for(app) as c:
        r = await _start(c, redirect)
    assert r.status_code == 400 and r.json()["error"]["code"] == "REDIRECT_URI_NOT_ALLOWED"


@pytest.mark.parametrize("missing", ["GOOGLE_OAUTH_REDIRECT_URI", "OAUTH_ALLOWED_REDIRECT_ORIGINS"])
def test_mvp_104_strict_env_requires_explicit_redirect_settings(monkeypatch, missing):
    monkeypatch.setenv("ENVIRONMENT", "production")
    for name, value in {"GOOGLE_OAUTH_CLIENT_ID": "c", "GOOGLE_OAUTH_CLIENT_SECRET": "s",
                        "GOOGLE_OAUTH_REDIRECT_URI": "https://auth.example/cb",
                        "OAUTH_ALLOWED_REDIRECT_ORIGINS": "https://app.example"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)
    with pytest.raises(OAuthConfigError, match=missing):
        google_oauth_settings_from_env()


async def test_mvp_223_callback_from_another_browser_is_refused():
    app, state = build_identity_app()
    enable_google_oauth(state, identity=IDENTITY)
    async with client_for(app) as attacker, client_for(app) as victim:
        start = await _start(attacker)
        cb = await victim.get(CALLBACK, params={"code": "g", "state": _q(start.headers["location"])["state"][0]})
    assert cb.status_code == 400 and cb.json()["error"]["code"] == "OAUTH_STATE_MISMATCH"
    assert "location" not in cb.headers


async def test_mvp_223_state_cookie_is_httponly_lax_secure_and_scoped_and_nonce_is_bound():
    app, state = build_identity_app()
    seen = {}

    async def verifier(id_token, settings, **kw):
        seen.update(kw)
        return IDENTITY

    enable_google_oauth(state, identity=IDENTITY)
    state.oauth_id_token_verifier = verifier
    async with client_for(app) as c:
        start = await _start(c)
        cookie = start.headers["set-cookie"].lower()
        await c.get(CALLBACK, params={"code": "g", "state": _q(start.headers["location"])["state"][0]})
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" in cookie
    assert "path=/api/v1/auth/oauth/google" in cookie
    assert seen["expected_nonce"] == _q(start.headers["location"])["nonce"][0]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_104_oauth_code_handoff.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'InMemoryOneTimeCodeStore'`. The defects behind it: the callback returns the token pair as a raw JSON body with no redirect or code handoff, and the redirect URI silently defaults to localhost (MVP-104). `/start` sets no cookie, so a callback URL started by an attacker and replayed in a victim's browser validates, which is login CSRF (MVP-223).
- [ ] **Step 3: Implement** the settings guard, allow-list, `PendingAuthorization`, code stores, cookie binding, nonce, the redirecting callback and `/exchange` in `oauth.py`, plus the wiring, test-kit, test-file and doc changes. Approach: the callback compares `sha256(cookie)` with the stored `binding_hash` using `hmac.compare_digest`, and the code payload holds only `{user_id, amr, flow}`; the session is created at `/exchange`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (including `test_mvp_210_argon2_off_loop.py` through the updated `google_round_trip`, and the doc test with `/exchange` documented)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.11: Sessions store the real client IP behind Caddy (consent records follow through request R-9)
**Covers:** MVP-326
**Severity:** medium
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.10, WS-07.7
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-26 (default assumed: `TRUSTED_PROXY_HOPS=1` in strict environments)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/api_auth.py` (`login` creates the session with `ip_address=client_ip(request)`)
- Modify: `services/user-service/src/oauth.py` (`/exchange` uses `client_ip(request)` for the session)
- Test: `services/user-service/tests/test_mvp_326_client_ip_behind_proxy.py`
**Interfaces:**
- Consumes: `solace_security.rate_limit.client_ip(request, trusted_proxy_hops: int | None = None)`, where `None` resolves `TRUSTED_PROXY_HOPS` (WS-07.7); WS-14's `record_consent` using the same function (open request R-9; no WS-14 task produces it yet, so the consent-record half of MVP-326 closes only when WS-14 lands R-9; this task's tests cover the session half).
- Produces: none (behaviour only). WS-08.16 and WS-08.12 create sessions the same way.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_326_client_ip_behind_proxy.py::test_mvp_326_login_session_records_the_forwarded_client_ip`
```python
"""MVP-326 (session half): behind Caddy, sessions keep the real client IP, not the proxy's.
The consent-record half is WS-14's code (api.py record_consent) and is requested as R-9 with its own test."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest

from identity_testkit import build_identity_app, claims, client_for, enable_google_oauth, google_round_trip, login, seed_user
from src.infrastructure.oauth_google import GoogleIdentity

XFF = {"X-Forwarded-For": "198.51.100.7"}


async def _session_ip(state, tokens) -> str:
    session = await state.session_manager.get_session(UUID(claims(tokens["access_token"])["sid"]))
    return session.ip_address


async def test_mvp_326_login_session_records_the_forwarded_client_ip(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c, headers=XFF)
    assert await _session_ip(state, tokens) == "198.51.100.7"


async def test_mvp_326_forwarded_header_is_ignored_without_trusted_hops(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "0")
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c, headers=XFF)
    assert await _session_ip(state, tokens) == "127.0.0.1"


async def test_mvp_326_google_exchange_session_records_the_forwarded_client_ip(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    app, state = build_identity_app()
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-1", email="g@example.com", email_verified=True))
    async with client_for(app) as c:
        cb = await google_round_trip(c)
        code = parse_qs(urlsplit(cb.headers["location"]).query)["code"][0]
        tokens = (await c.post("/api/v1/auth/oauth/google/exchange", json={"code": code}, headers=XFF)).json()
    assert await _session_ip(state, tokens) == "198.51.100.7"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_326_client_ip_behind_proxy.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The login and exchange sessions record `127.0.0.1` (the socket peer, which is the Caddy container in the deploy; today's login passes `request.client.host`, `services/user-service/src/api.py:544`, and the OAuth path does the same, `services/user-service/src/oauth.py:281`) (MVP-326). The no-hops test passes.
- [ ] **Step 3: Implement** the two `client_ip(request)` call sites. MVP-326 is marked fixed only once WS-14 also lands R-9 (consent records, with the test named in R-9); until then the ledger records the session half as done and the consent half as open.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.12: Google accounts link only by verified Google subject; an existing password account links only after proving its password
**Covers:** MVP-102
**Severity:** high
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.11, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `services/user-service/src/infrastructure/oauth_account_repository.py` (`OAuthAccountRepository` protocol: `async get_user_id(provider: str, subject: str) -> UUID | None`, `async link(user_id: UUID, provider: str, subject: str) -> None` (raises `OAuthSubjectAlreadyLinked` when the subject belongs to another user; linking the same pair again is a no-op), `async delete_for_user(user_id: UUID) -> int`; `InMemoryOAuthAccountRepository`; `PostgresOAuthAccountRepository(pool_name: str = "default")`, which uses `ConnectionPoolManager.acquire()` on the non-RLS `oauth_accounts` table, writes `email` as NULL and inserts with `ON CONFLICT (provider, provider_user_id) DO NOTHING` followed by an ownership check)
- Modify: `services/user-service/src/oauth.py`:
  - `_link_or_create_user` is replaced by `resolve_google_user(svc, identity) -> GoogleResolution` (`kind: Literal["user", "link_required", "unverified"]`, `user`, `user_id`). It resolves a linked subject to its user. A different local account with the same email is never mutated and gives `link_required`. An unknown email creates a verified ACTIVE user plus the link row. An unverified Google email gives `unverified`.
  - the callback redirects `link_required` to `?error=account_link_required&link_code=<one-time code>` and `unverified` to `?error=email_not_verified`;
  - new `POST /auth/oauth/google/link {link_code, password}`: takes the code once, then checks the password with `verify_password_async` (a locked account gets 401 without a check). A wrong password is recorded with `record_login_attempt(False)` and returns 401 `INVALID_CREDENTIALS`. A right password records `record_login_attempt(True)` and links the subject; a PENDING account becomes verified and ACTIVE (Google proved the email and the password proved the account); the route creates the session and returns tokens with `amr=["google"]`.
- Modify: `services/user-service/src/identity_wiring.py` (`start_identity` sets `app.state.service.oauth_account_repository`, the name WS-14.16 reads with `getattr(state, "oauth_account_repository", None)` to register erasure store `oauth_accounts` (R-14-10): the Postgres repository when `runtime.pg_pool` is set, in-memory with a WARNING otherwise; strict without a pool raises)
- Modify: `services/user-service/tests/identity_testkit.py` (`oauth_account_repository=InMemoryOAuthAccountRepository()`)
- Modify: `services/user-service/tests/test_oauth_endpoints.py` (`test_callback_links_existing_user_without_duplicate` now expects `error=account_link_required` and no new user)
- Modify: `tests/realinfra/identity-auth-mfa/conftest.py` (add fixture `insert_minimal_user`, which returns `async def insert(conn: asyncpg.Connection) -> UUID`. It fills every NOT NULL `users` column without a default, read from `information_schema.columns`: uuid gets a new uuid, text/varchar gets `"t-<12 hex>"`, boolean gets false, integers get 0, timestamps get now(), jsonb gets `{}`. Tests take the fixture rather than importing `conftest`, so another slug's `conftest` module can never shadow it.)
- Modify: `docs/api/auth.md` (link flow and redirect error values)
- Test: `services/user-service/tests/test_mvp_102_oauth_link_by_subject.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_102_oauth_accounts_real_pg.py`
**Interfaces:**
- Consumes: the `oauth_accounts` table from slot 004 (non-RLS, IF-03); `ConnectionPoolManager.acquire()` (IF-03); the one-time code store (WS-08.10); `verify_password_async` (WS-08.3); fixture `pg_migrated` (IF-19).
- Produces: `OAuthAccountRepository`, `OAuthSubjectAlreadyLinked`, `PostgresOAuthAccountRepository`, `InMemoryOAuthAccountRepository` and `.delete_for_user`, exposed as `app.state.service.oauth_account_repository` with `email` always written as NULL, so no address is stored and IF-13 has nothing to encrypt (this is the R-14-10 request from WS-14.16, which registers it as erasure store `oauth_accounts`); `resolve_google_user`; `POST /api/v1/auth/oauth/google/link`; redirect error values `account_link_required`, `email_not_verified`.

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_102_oauth_link_by_subject.py::test_mvp_102_google_never_takes_over_a_preregistered_unverified_account` and the real-infra file
```python
"""MVP-102: Google sign-in links by verified subject only; an email match never promotes or merges an account."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from identity_testkit import PASSWORD, build_identity_app, claims, client_for, enable_google_oauth, google_round_trip, seed_user
from src.domain.value_objects import AccountStatus
from src.infrastructure.oauth_google import GoogleIdentity

EXCHANGE = "/api/v1/auth/oauth/google/exchange"
LINK = "/api/v1/auth/oauth/google/link"


def _params(response) -> dict:
    return parse_qs(urlsplit(response.headers["location"]).query)


async def test_mvp_102_google_never_takes_over_a_preregistered_unverified_account():
    app, state = build_identity_app()
    squatter = await seed_user(state, email="victim@gmail.com", password="Attacker-Pass-1",
                               status=AccountStatus.PENDING_VERIFICATION, verified=False)
    squatter_hash = squatter.password_hash
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-victim", email="victim@gmail.com",
                                                       email_verified=True))
    async with client_for(app) as c:
        params = _params(await google_round_trip(c))
    assert params["error"] == ["account_link_required"] and "code" not in params
    user = await state.user_repository.get_by_id(squatter.user_id)
    assert (user.status, user.email_verified, user.password_hash) == (
        AccountStatus.PENDING_VERIFICATION, False, squatter_hash)
    assert await state.oauth_account_repository.get_user_id("google", "g-victim") is None


async def test_mvp_102_linked_subject_signs_in_even_after_an_email_change():
    app, state = build_identity_app()
    user = await seed_user(state, email="old@example.com")
    await state.oauth_account_repository.link(user.user_id, "google", "g-1")
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-1", email="new@example.com", email_verified=True))
    async with client_for(app) as c:
        tokens = (await c.post(EXCHANGE, json={"code": _params(await google_round_trip(c))["code"][0]})).json()
    assert claims(tokens["access_token"])["sub"] == str(user.user_id)


async def test_mvp_102_new_identity_creates_a_verified_account_and_a_link_row():
    app, state = build_identity_app()
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-new", email="fresh@example.com", email_verified=True))
    async with client_for(app) as c:
        r = await c.post(EXCHANGE, json={"code": _params(await google_round_trip(c))["code"][0]})
    user = await state.user_repository.get_by_email("fresh@example.com")
    assert r.status_code == 200 and (user.status, user.email_verified) == (AccountStatus.ACTIVE, True)
    assert await state.oauth_account_repository.get_user_id("google", "g-new") == user.user_id


async def test_mvp_102_password_owner_links_by_proving_the_password():
    app, state = build_identity_app()
    owner = await seed_user(state, email="owner@example.com")
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-own", email="owner@example.com", email_verified=True))
    async with client_for(app) as c:
        link_code = _params(await google_round_trip(c))["link_code"][0]
        wrong = await c.post(LINK, json={"link_code": link_code, "password": "Wrong-Horse-1"})
        assert wrong.status_code == 401
        link_code = _params(await google_round_trip(c))["link_code"][0]
        right = await c.post(LINK, json={"link_code": link_code, "password": PASSWORD})
    assert right.status_code == 200 and claims(right.json()["access_token"])["amr"] == ["google"]
    assert await state.oauth_account_repository.get_user_id("google", "g-own") == owner.user_id
    assert (await state.user_repository.get_by_id(owner.user_id)).login_attempts == 0


async def test_mvp_102_unverified_google_email_is_refused():
    app, state = build_identity_app()
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-u", email="u@example.com", email_verified=False))
    async with client_for(app) as c:
        params = _params(await google_round_trip(c))
    assert params["error"] == ["email_not_verified"]
    assert await state.user_repository.get_by_email("u@example.com") is None
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_102_oauth_accounts_real_pg.py
MVP-102 on the migrated schema as solace_app: link rows round-trip, a subject cannot move users, no email is stored."""
from __future__ import annotations

import asyncpg
import pytest

from src.infrastructure.oauth_account_repository import OAuthSubjectAlreadyLinked, PostgresOAuthAccountRepository

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_102_postgres_oauth_link_round_trip(pg_migrated, insert_minimal_user):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        owner, other = await insert_minimal_user(su), await insert_minimal_user(su)
        repo = PostgresOAuthAccountRepository()
        assert await repo.get_user_id("google", "g-rt-1") is None
        await repo.link(owner, "google", "g-rt-1")
        await repo.link(owner, "google", "g-rt-1")
        assert await repo.get_user_id("google", "g-rt-1") == owner
        with pytest.raises(OAuthSubjectAlreadyLinked):
            await repo.link(other, "google", "g-rt-1")
        assert await su.fetchval("SELECT email FROM oauth_accounts WHERE provider_user_id = 'g-rt-1'") is None
        assert await repo.delete_for_user(owner) == 1
        assert await repo.get_user_id("google", "g-rt-1") is None
    finally:
        await su.close()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_102_oauth_link_by_subject.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_102_oauth_accounts_real_pg.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. The callback finds the squatter's row by email, sets `email_verified=True`, activates it and redirects with a login `code`, so the squatter's password now opens the victim's account (MVP-102). The test kit has no `oauth_accounts`, and nothing writes `oauth_accounts`, so no Google subject is ever persisted. The real-infra test fails at `ModuleNotFoundError: src.infrastructure.oauth_account_repository`.
- [ ] **Step 3: Implement** the repository, `resolve_google_user`, the callback outcomes, `/link`, the wiring and the doc section. The callback never calls `user.activate()` or `user_repository.update` on an email-matched account; only `/link` does, after the password check.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.13: Access tokens carry the consent claim `cns`, re-read at every issuance; revoking all of a user's sessions also revokes each sid in the shared blacklist
**Covers:** none (supports MVP-100, owned and closed by WS-14)
**Severity:** high
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.12, WS-14.17
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** D-23 (default assumed: chat requires `cns.ok`; a failed consent lookup mints `ok: false`, so chat stays blocked while login still works)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`generate_token_pair(..., cns: Mapping[str, Any] | None = None)` writes `cns` into the access token only)
- Modify: `services/user-service/src/auth.py`:
  - new `async consent_claim_for(user_service, user_id: UUID) -> dict[str, Any]`, which awaits `user_service.get_consent_claim(user_id)` and returns `{"ok": bool, "version": str}`, or `{"ok": False, "version": "unknown"}` on any error, with a WARNING that carries no user data;
  - `SessionManager` and `RedisSessionManager` gain `attach_revocation_blacklist(blacklist: TokenBlacklist, *, access_token_ttl: timedelta) -> None`. Once attached, `revoke_session(session_id)` and `revoke_all_user_sessions(user_id)` write `blacklist.revoke_session(str(sid), now + access_token_ttl + 60 s)` for every session they remove, before deleting the record, and a blacklist error raises `IdentityBackendUnavailable`. `SessionRevoker.__init__` (WS-08.4) attaches its blacklist, and `SessionRevoker.revoke_all` delegates to `session_manager.revoke_all_user_sessions`, so WS-14.17's direct consent-revocation call and the revoker take one code path (R-14-9).
- Modify: `services/user-service/src/api_auth.py` (`login` and `refresh_token`, including the grace re-mint, pass `cns=await consent_claim_for(...)`)
- Modify: `services/user-service/src/oauth.py` (`/exchange` and `/link` do the same)
- Test: `services/user-service/tests/test_mvp_100_cns_claim.py`
**Interfaces:**
- Consumes: `UserService.get_consent_claim(user_id: UUID) -> dict` returning `{"ok": bool, "version": str}` (IF-08 `cns`; WS-14.17); `TokenBlacklist.revoke_session` (WS-07.5) through the WS-08.4 re-export.
- Produces (IF-17/IF-08): access-token claim `cns: {ok: bool, version: str}` on every issuance path (login, refresh and the grace re-mint, OAuth `/exchange` and `/link`). WS-07.2 surfaces it as `AuthenticatedUser.cns`, and WS-11.20 gates chat on it. `SessionManager.revoke_all_user_sessions` now ends the sessions at every service, not only in user-service. Both are the cns and session halves of the R-14-9 request from WS-14.17.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_100_cns_claim.py::test_cns_claim_reflects_required_consents_at_login_and_refresh`
```python
"""Supports MVP-100: access tokens carry the required-consent claim, re-read on login, refresh and Google sign-in."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from identity_testkit import build_identity_app, claims, client_for, enable_google_oauth, google_round_trip, login, seed_user
from src.infrastructure.oauth_google import GoogleIdentity


def _consents(state, value):
    box = {"value": value}

    async def get_consent_claim(user_id):  # spy with WS-14.17's name and shape
        if isinstance(box["value"], Exception):
            raise box["value"]
        return box["value"]

    state.user_service.get_consent_claim = get_consent_claim
    return box


async def test_cns_claim_reflects_required_consents_at_login_and_refresh():
    app, state = build_identity_app()
    await seed_user(state)
    box = _consents(state, {"ok": False, "version": "2026-10"})
    async with client_for(app) as c:
        tokens = await login(c)
        assert claims(tokens["access_token"])["cns"] == {"ok": False, "version": "2026-10"}
        box["value"] = {"ok": True, "version": "2026-10"}
        refreshed = (await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})).json()
    assert claims(refreshed["access_token"])["cns"] == {"ok": True, "version": "2026-10"}
    assert "cns" not in claims(refreshed["refresh_token"])


async def test_cns_claim_is_minted_on_google_sign_in():
    app, state = build_identity_app()
    _consents(state, {"ok": True, "version": "2026-10"})
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-c", email="c@example.com", email_verified=True))
    async with client_for(app) as c:
        code = parse_qs(urlsplit((await google_round_trip(c)).headers["location"]).query)["code"][0]
        tokens = (await c.post("/api/v1/auth/oauth/google/exchange", json={"code": code})).json()
    assert claims(tokens["access_token"])["cns"]["ok"] is True


async def test_cns_lookup_failure_mints_ok_false_and_login_still_works():
    app, state = build_identity_app()
    await seed_user(state)
    _consents(state, ConnectionError("db down"))
    async with client_for(app) as c:
        tokens = await login(c)
    assert claims(tokens["access_token"])["cns"] == {"ok": False, "version": "unknown"}


async def test_revoke_all_user_sessions_revokes_each_sid_in_the_shared_blacklist():
    app, state = build_identity_app()
    user = await seed_user(state)
    _consents(state, {"ok": True, "version": "2026-10"})
    async with client_for(app) as c:
        first, second = await login(c), await login(c)
    assert await state.session_manager.revoke_all_user_sessions(user.user_id) == 2  # WS-14.17's consent-revocation call
    for tokens in (first, second):
        assert await state.token_blacklist.is_session_revoked(claims(tokens["access_token"])["sid"]) is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_100_cns_claim.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `KeyError: 'cns'` in the three claim tests. No token carries the consent state, so the orchestrator cannot enforce consent (MVP-100). The revocation test fails on `is_session_revoked(...) is True`: `revoke_all_user_sessions` deletes the session records only (`services/user-service/src/auth.py:376-399` in memory, `:555-568` in Redis) and writes no sid key, so a consent revocation leaves the access tokens valid at every other service.
- [ ] **Step 3: Implement** `consent_claim_for`, the `cns` parameter, the four call sites and `attach_revocation_blacklist` with the `SessionRevoker` delegation.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass (the real-infra run re-checks WS-08.4's Redis session and cross-service logout tests after the `RedisSessionManager` change)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.14: Every authentication event is written to the audit trail, with no PHI
**Covers:** MVP-347
**Severity:** blocker
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.13, WS-09.9, WS-09.11, WS-09.7, WS-04.10, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: Postgres `audit_events` with one HMAC chain; rows hold ids and codes only)
**Real infra:** yes
**Files:**
- Modify: `services/user-service/src/auth.py`. New `async audit_auth_event(event_type: AuditEventType, *, outcome: AuditOutcome, action: str, user_id: UUID | None, reason_code: str | None = None, session_id: UUID | None = None, amr: Sequence[str] | None = None, mfa: bool | None = None, email_for_hash: str | None = None, extra: Mapping[str, str | int | bool] | None = None) -> None`. It calls `audit_runtime.audit(...)` through the module attribute, using `actor_id=str(user_id)` or `"anonymous"`, `actor_type="user"`, `resource_type="session"` when a sid is known and `"account"` otherwise. `resource_id` is the sid, or `"emailhash:" + get_phi_codec().search_hash(email, purpose="auth_login_email")` for an unknown account. `details` holds only `reason_code`, `sid`, `amr` (comma-joined), `mfa` and `extra`. It never raises; a codec failure omits the resource id.
- Modify: `services/user-service/src/api_auth.py`. Events by route:
  - `login`: `AUTH_LOGIN_SUCCEEDED`; `AUTH_LOGIN_FAILED` with reason codes `bad_password`, `unknown_account`, `account_unavailable` and `email_not_verified`; `AUTH_LOCKOUT` when the failure locked the account (re-read after `record_login_attempt`);
  - `logout`: `AUTH_LOGOUT`;
  - `refresh_token`: `AUTH_TOKEN_REFRESHED`; `AUTH_TOKEN_REVOKED` with `refresh_reuse`;
  - `revoke_token`: `AUTH_TOKEN_REVOKED` with `user_revoke`;
  - `change_password`: `AUTH_PASSWORD_CHANGED`;
  - `delete_current_user`: `ACCOUNT_DELETED` with `extra={"erasure_request_id": ...}`;
  - `/auth/reauth`: `AUTH_LOGIN_FAILED` with `reauth_failed`.
- Modify: `services/user-service/src/oauth.py` (`AUTH_OAUTH_LOGIN` on `/exchange` and `/link`; `AUTH_LOGIN_FAILED` with `bad_password` on a failed `/link`)
- Modify: `services/user-service/tests/identity_testkit.py` (new `AuditSpy` and `install_audit_spy(monkeypatch) -> AuditSpy`, which patches `solace_security.audit_runtime.audit`; `fake_codec(monkeypatch)` patches `src.auth.get_phi_codec` with a stub whose `search_hash` returns `"h" * 64`)
- Test: `services/user-service/tests/test_mvp_347_auth_audit_events.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_347_auth_audit_rows.py`
**Interfaces:**
- Consumes: `solace_security.audit_runtime.audit(...)`, `configure_service_audit(service_name, pool)`, `verify_chain()`, `reset_audit_runtime()` and `ALLOWED_DETAIL_KEYS` (which already contains `reason_code`, `sid`, `amr`, `mfa` and `erasure_request_id`) (IF-14, WS-09.11); `AuditEventType.AUTH_*` and `ACCOUNT_DELETED` (WS-09.9); `get_phi_codec().search_hash(value, *, purpose)` (IF-13, WS-09.7).
- Produces: `audit_auth_event` (WS-08.16, WS-08.17 and WS-08.18 add `AUTH_MFA_FAILED`, `AUTH_MFA_ENROLLED` and `AUTH_PASSWORD_RESET` through it); test-kit `AuditSpy`, `install_audit_spy`, `fake_codec`.

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_347_auth_audit_events.py::test_mvp_347_login_success_and_failures_are_audited` and the real-infra file
```python
"""MVP-347: login, lockout, logout, refresh, revocation, password change, Google sign-in and deletion are audited."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from identity_testkit import (PASSWORD, bearer, build_identity_app, claims, client_for, fake_codec, install_audit_spy,
                              login, seed_user)
from solace_security.audit import AuditEventType as E

LOGIN = "/api/v1/auth/login"


async def test_mvp_347_login_success_and_failures_are_audited(monkeypatch):
    spy = install_audit_spy(monkeypatch)
    fake_codec(monkeypatch)
    app, state = build_identity_app()
    user = await seed_user(state)
    async with client_for(app) as c:
        await c.post(LOGIN, json={"email": "alice@example.com", "password": "Wrong-Horse-1"})
        await c.post(LOGIN, json={"email": "ghost@example.com", "password": PASSWORD})
        tokens = await login(c)
    assert spy.types() == [E.AUTH_LOGIN_FAILED, E.AUTH_LOGIN_FAILED, E.AUTH_LOGIN_SUCCEEDED]
    known, unknown, ok = spy.calls
    assert (known["subject_user_id"], known["details"]["reason_code"]) == (str(user.user_id), "bad_password")
    assert (unknown["actor_id"], unknown["resource_id"], unknown["details"]["reason_code"]) == (
        "anonymous", "emailhash:" + "h" * 64, "unknown_account")
    assert ok["details"] == {"amr": "pwd", "sid": claims(tokens["access_token"])["sid"], "mfa": False}
    assert "alice@example.com" not in repr(spy.calls) and "ghost@example.com" not in repr(spy.calls)
    assert PASSWORD not in repr(spy.calls)


async def test_mvp_347_lockout_session_and_account_events_are_audited(monkeypatch):
    spy = install_audit_spy(monkeypatch)
    fake_codec(monkeypatch)
    app, state = build_identity_app()
    await seed_user(state, email="bob@example.com")
    await seed_user(state)
    request_id = uuid4()

    async def delete_user(user_id, reason=None):  # WS-14.16 DeleteUserResult shape
        return SimpleNamespace(success=True, error=None, error_code=None, erasure_request_id=request_id)

    state.user_service.delete_user = delete_user
    async with client_for(app) as c:
        for _ in range(5):
            await c.post(LOGIN, json={"email": "bob@example.com", "password": "Wrong-Horse-1"})
        assert spy.types()[-2:] == [E.AUTH_LOGIN_FAILED, E.AUTH_LOCKOUT]
        tokens = await login(c)
        refreshed = (await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})).json()
        await c.post("/api/v1/auth/revoke-token", headers=bearer(tokens["access_token"]))
        await c.post("/api/v1/auth/logout", headers=bearer(refreshed["access_token"]))
        again = await login(c)
        await c.post("/api/v1/users/me/password", headers=bearer(again["access_token"]),
                     json={"current_password": PASSWORD, "new_password": "Another-Horse-7"})
        third = await login(c, password="Another-Horse-7")
        reauth = (await c.post("/api/v1/auth/reauth", headers=bearer(third["access_token"]),
                               json={"password": "Another-Horse-7", "purpose": "account_delete"})).json()
        await c.delete("/api/v1/users/me", headers={**bearer(third["access_token"]),
                                                     "X-Reauth-Token": reauth["reauth_token"]})
    tail = [t for t in spy.types() if t is not E.AUTH_LOGIN_SUCCEEDED][-5:]
    assert tail == [E.AUTH_TOKEN_REFRESHED, E.AUTH_TOKEN_REVOKED, E.AUTH_LOGOUT, E.AUTH_PASSWORD_CHANGED, E.ACCOUNT_DELETED]
    assert spy.calls[-1]["details"]["erasure_request_id"] == str(request_id)


async def test_mvp_347_refresh_reuse_and_google_sign_in_are_audited(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from identity_testkit import enable_google_oauth, google_round_trip
    from src.auth import InMemoryRefreshRotationStore
    from src.infrastructure.oauth_google import GoogleIdentity

    spy = install_audit_spy(monkeypatch)
    fake_codec(monkeypatch)
    clock = type("Clock", (), {"now": 0.0, "__call__": lambda self: self.now})()
    app, state = build_identity_app(rotation_store=InMemoryRefreshRotationStore(clock=clock))
    await seed_user(state)
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-a", email="g@example.com", email_verified=True))
    async with client_for(app) as c:
        tokens = await login(c)
        await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        clock.now += 31
        await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        reuse = spy.calls[-1]
        code = parse_qs(urlsplit((await google_round_trip(c)).headers["location"]).query)["code"][0]
        await c.post("/api/v1/auth/oauth/google/exchange", json={"code": code})
    assert (reuse["event_type"], reuse["details"]["reason_code"]) == (E.AUTH_TOKEN_REVOKED, "refresh_reuse")
    assert spy.calls[-1]["event_type"] is E.AUTH_OAUTH_LOGIN and spy.calls[-1]["details"]["amr"] == "google"
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_347_auth_audit_rows.py
MVP-347 on the migrated schema: a login writes an audit_events row with no PHI and the chain verifies."""
from __future__ import annotations

import asyncpg
import pytest

from identity_testkit import build_identity_app, client_for, login, seed_user
from solace_security import audit_runtime

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_347_login_writes_a_verifiable_audit_row(pg_migrated, monkeypatch):
    monkeypatch.setenv("AUDIT_HMAC_KEY", "k" * 40)
    monkeypatch.setenv("AUDIT_HMAC_RETIRED_KEYS_JSON", "[]")
    audit_runtime.reset_audit_runtime()
    await audit_runtime.configure_service_audit("user-service", pg_migrated.app_pool)
    app, state = build_identity_app()
    user = await seed_user(state)
    async with client_for(app) as c:
        await login(c)
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        row = await su.fetchrow("SELECT event_type, subject_user_id, details::text AS d FROM audit_events "
                                "WHERE event_type = 'auth_login_succeeded' ORDER BY chain_seq DESC LIMIT 1")
    finally:
        await su.close()
    assert str(row["subject_user_id"]) == str(user.user_id) and "alice@example.com" not in row["d"]
    assert (await audit_runtime.verify_chain()).ok is True
    audit_runtime.reset_audit_runtime()
```
Test-kit additions:
```python
class AuditSpy:
    def __init__(self):
        self.calls: list[dict] = []

    async def __call__(self, event_type, **kwargs):
        self.calls.append({"event_type": event_type, **kwargs})

    def types(self):
        return [c["event_type"] for c in self.calls]


def install_audit_spy(monkeypatch) -> AuditSpy:
    from solace_security import audit_runtime

    spy = AuditSpy()
    monkeypatch.setattr(audit_runtime, "audit", spy)
    return spy


def fake_codec(monkeypatch) -> None:
    import src.auth as auth_module

    monkeypatch.setattr(auth_module, "get_phi_codec",
                        lambda: SimpleNamespace(search_hash=lambda value, *, purpose: "h" * 64))
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_347_auth_audit_events.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_347_auth_audit_rows.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. `spy.calls == []` in every unit test and no `auth_login_succeeded` row exists, because authentication events exist only as stdout log lines that rotate away, and an unknown-email failure leaves no account-level trace (MVP-347). `fake_codec` fails first with `AttributeError: module 'src.auth' has no attribute 'get_phi_codec'`.
- [ ] **Step 3: Implement** `audit_auth_event` and the call sites listed under Files. `auth.py` imports `from solace_security import audit_runtime` and `from solace_security.phi_codec import get_phi_codec` at module level. Calls run inline after the response decision, because `audit()` never raises and bounds its own retries.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.15: Slot 013: MFA factor and recovery-code tables under RLS, and the password-reset token table
**Covers:** MVP-344, MVP-099
**Severity:** blocker
**Wave:** 4
**Phase:** A (schema lane, slot order)
**Depends on:** WS-02.3, WS-05.2, WS-08.14 (W3 barrier)
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: TOTP secrets are encrypted with the PHI codec, plus 10 hashed one-time recovery codes)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/013_identity_mfa.py` (`upgrade()` and `downgrade()` only; revision ids unchanged). The PostgreSQL branch creates three tables:
  - `user_mfa_factors`: `user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE`, `factor_type VARCHAR(16) NOT NULL DEFAULT 'totp' CHECK (factor_type = 'totp')`, `secret_ciphertext TEXT NOT NULL`, `encryption_key_id VARCHAR(32) NOT NULL`, `confirmed_at TIMESTAMPTZ NULL`, `last_used_step BIGINT NULL`, `created_at` and `updated_at TIMESTAMPTZ NOT NULL DEFAULT now()`;
  - `mfa_recovery_codes`: `id UUID PRIMARY KEY`, `user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE`, `code_hash CHAR(64) NOT NULL`, `used_at TIMESTAMPTZ NULL`, `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`, `UNIQUE (user_id, code_hash)`, index on `user_id`;
  - `password_reset_tokens`: `token_hash CHAR(64) PRIMARY KEY`, `user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE`, `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`, `expires_at TIMESTAMPTZ NOT NULL`, `used_at TIMESTAMPTZ NULL`, index on `user_id`.

  `ENABLE` and `FORCE ROW LEVEL SECURITY` apply to the first two, with policies `<table>_by_user` using the IF-03 NULLIF predicate for both `USING` and `WITH CHECK`. `password_reset_tokens` stays non-RLS, because the redeem step looks it up by hash before any user is known; it holds no PHI. `downgrade()` drops the policies and the three tables.
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_344_slot_013_schema.py`
**Interfaces:**
- Consumes: the slot scaffold and NULLIF rule (IF-03, WS-02.3); fixture `insert_minimal_user` (WS-08.12 slug conftest); fixture `pg_migrated` (IF-19).
- Produces: the three tables above, used by WS-08.16, WS-08.17 and WS-08.18. WS-14's erasure deletes rows through the repositories (request R-10).

- [ ] **Step 1: Write the failing test** in `tests/realinfra/identity-auth-mfa/test_mvp_344_slot_013_schema.py::test_mvp_344_slot_013_tables_exist_with_the_right_rls`
```python
"""MVP-344 / MVP-099: slot 013 creates the MFA tables under per-user RLS and a context-free reset-token table."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
TABLES = ["user_mfa_factors", "mfa_recovery_codes", "password_reset_tokens"]


async def test_mvp_344_slot_013_tables_exist_with_the_right_rls(pg_migrated):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        rows = await su.fetch("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = ANY($1)", TABLES)
        quals = [r["qual"] for r in await su.fetch(
            "SELECT qual FROM pg_policies WHERE tablename IN ('user_mfa_factors', 'mfa_recovery_codes')")]
    finally:
        await su.close()
    rls = {r["relname"]: (r["relrowsecurity"], r["relforcerowsecurity"]) for r in rows}
    assert rls == {"user_mfa_factors": (True, True), "mfa_recovery_codes": (True, True),
                   "password_reset_tokens": (False, False)}
    assert len(quals) == 2 and all("NULLIF" in q for q in quals)


async def test_mvp_344_factor_rows_are_invisible_and_unwritable_across_users(pg_migrated, insert_minimal_user):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        a, b = await insert_minimal_user(su), await insert_minimal_user(su)
    finally:
        await su.close()
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(a))
        await conn.execute("INSERT INTO user_mfa_factors (user_id, secret_ciphertext, encryption_key_id) "
                           "VALUES ($1, 'enc:v1:k1:synthetic', 'k1')", a)
        await conn.execute("SELECT set_config('app.current_user_id', $1, false)", str(b))
        assert await conn.fetchval("SELECT count(*) FROM user_mfa_factors") == 0
        with pytest.raises(asyncpg.PostgresError):
            await conn.execute("INSERT INTO user_mfa_factors (user_id, secret_ciphertext, encryption_key_id) "
                               "VALUES ($1, 'enc:v1:k1:x', 'k1')", a)
        await conn.execute("SELECT set_config('app.current_user_id', '', false)")
        assert await conn.fetchval("SELECT count(*) FROM user_mfa_factors") == 0


async def test_mvp_099_reset_tokens_work_without_a_user_context(pg_migrated, insert_minimal_user):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        user_id = await insert_minimal_user(su)
    finally:
        await su.close()
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute("SELECT set_config('app.current_user_id', '', false)")
        await conn.execute("INSERT INTO password_reset_tokens (token_hash, user_id, expires_at) VALUES ($1, $2, $3)",
                           "a" * 64, user_id, datetime.now(timezone.utc) + timedelta(minutes=30))
        assert await conn.fetchval("SELECT user_id FROM password_reset_tokens WHERE token_hash = $1", "a" * 64) == user_id
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_344_slot_013_schema.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. `rls == {}` and the inserts raise `UndefinedTableError`, because slot 013 is an empty scaffold: there is nowhere to keep a second factor (MVP-344) or a single-use reset token (MVP-099).
- [ ] **Step 3: Implement** the `upgrade()` and `downgrade()` bodies in `migrations/versions/013_identity_mfa.py`, following the slot file's dialect guard.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** the chain must still upgrade, downgrade and re-upgrade. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\migrations -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass, including `test_chain_real_pg.py` (every policy uses NULLIF; downgrade to base is clean). WS-02 reviews the slot diff.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.16: TOTP step-up at login: enrolled users get an MFA challenge, clinicians and admins without a factor must enroll, and verified tokens carry `mfa=true`
**Covers:** MVP-344
**Severity:** blocker
**Wave:** 4
**Phase:** B
**Depends on:** WS-08.15, WS-07.11, WS-01.2, WS-08.21
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: TOTP per RFC 6238, 30 s, 6 digits, plus or minus 1 step, replay of the last used step rejected; 10 recovery codes; mandatory for CLINICIAN and ADMIN; optional for users)
**Real infra:** no
**Files:**
- Create: `services/user-service/src/infrastructure/mfa_repository.py`:
  - `@dataclass MFAFactor(user_id, secret_ciphertext, encryption_key_id, confirmed_at, last_used_step)`;
  - `MFARepository` protocol: `async get_factor(user_id) -> MFAFactor | None`, `async upsert_pending_factor(user_id, secret_ciphertext, key_id) -> None`, `async confirm_factor(user_id, step: int) -> bool`, `async advance_last_used_step(user_id, step: int) -> bool` (an atomic compare-and-set that succeeds only when `step > last_used_step`), `async add_recovery_codes(user_id, hashes: Sequence[str]) -> None` (replaces any old codes), `async consume_recovery_code(user_id, code_hash) -> bool`, `async delete_for_user(user_id) -> int`;
  - `InMemoryMFARepository` (guarded by an `asyncio.Lock`).
- Create: `services/user-service/src/mfa.py`:
  - `TOTPVerifier(*, period=30, digits=6, window=1)` with `matching_step(secret_b32: str, code: str, *, at: float | None = None) -> int | None`;
  - `hash_recovery_code(code: str) -> str` (sha256 hex of the normalized code);
  - `MFAService(repository, *, codec=None, verifier=None)` with `async has_confirmed_factor(user_id) -> bool`, `async verify_code(user_id, code) -> bool` and `async consume_recovery_code(user_id, code) -> bool`. The secret is decrypted with `get_phi_codec().decrypt(..., field="user_mfa_factors.secret", owner_id=user_id)`.
  - `mfa_router` with `POST /auth/mfa/verify {mfa_token, code | recovery_code}`. It accepts an unexpired, unused `mfa` token, allows at most 5 wrong attempts per token (`FailureCounter` key `mfa_attempts:<jti>`), and records `AUTH_MFA_FAILED` on each failure. Success blacklists the token's jti, creates the session with `client_ip(request)` and returns tokens with `amr=[<first factor>, "otp"]`, `mfa=True` and `loc=locale_claim_for(user)` (WS-08.21, D-29).
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`TokenType.MFA = "mfa"`, `TokenType.MFA_ENROLL = "mfa_enroll"`; `MFA_TOKEN_TTL_SECONDS = 300`, `MFA_ENROLL_TOKEN_TTL_SECONDS = 600`; `create_step_up_token(user_id: UUID, token_type: TokenType, amr: Sequence[str]) -> str`; `verify_token` decodes them only with a matching `expected_type`)
- Modify: `services/user-service/src/api_auth.py`:
  - `login`, after the password and verification gates: a user with a confirmed factor gets 200 `{"mfa_required": true, "mfa_token", "expires_in": 300}`; a CLINICIAN or ADMIN without one gets 200 `{"mfa_enrollment_required": true, "enrollment_token", "expires_in": 600}`; neither case creates a session. Other users get tokens with `mfa=False`.
  - `auth_router.include_router(mfa_router)`.
- Modify: `services/user-service/src/oauth.py` (`/exchange` and `/link` apply the same step-up with `amr=["google"]`)
- Modify: `services/user-service/src/api_deps.py` (new `require_mfa_for_roles(*roles: str)`, the user-service-token analogue of WS-07.11's dependency: 403 `MFA_REQUIRED` when the payload role is one of `roles` and `payload.mfa` is not True)
- Modify: `services/user-service/src/identity_wiring.py` (`mfa_service` over `InMemoryMFARepository` until WS-08.17 switches strict environments to Postgres)
- Modify: `services/user-service/tests/identity_testkit.py` (`mfa_repository=InMemoryMFARepository()`, `mfa_service=MFAService(..., codec=PlainCodec())`; `PlainCodec`, a reversible stand-in for the IF-13 codec: `encrypt` returns `"enc:test:" + base64url(value)`, `decrypt` reverses it, `active_key_id = "k-test"`, so a stored secret never appears verbatim; `async seed_totp(state, user, *, recovery_codes=()) -> str`, which stores a confirmed factor plus the hashes of `recovery_codes` and returns the base32 secret)
- Modify: `docs/api/auth.md` (login challenge shapes, `/auth/mfa/verify`, `amr`/`mfa` values)
- Test: `services/user-service/tests/test_mvp_344_mfa_login_step_up.py`
**Interfaces:**
- Consumes: slot 013 tables (WS-08.15); `require_mfa_for_roles` and the MFA condition in `can_access_subject` for every other service (WS-07.11); `AuthenticatedUser.mfa` (WS-07.2); `FailureCounter` (WS-08.8); `audit_auth_event` (WS-08.14); `client_ip` (WS-07.7); `pyotp` (lock, D-16).
- Produces (IF-17 MFA): the login challenge bodies; `POST /api/v1/auth/mfa/verify`; token types `mfa` and `mfa_enroll`; access claims `amr` in {`["pwd"]`, `["pwd", "otp"]`, `["google"]`, `["google", "otp"]`} and `mfa`; `MFARepository`, `InMemoryMFARepository`, `MFAService`, `TOTPVerifier`, `hash_recovery_code`; `api_deps.require_mfa_for_roles(*roles)` (consumed by WS-14 for admin routes, request R-11); error codes `MFA_INVALID`, `MFA_TOKEN_INVALID`, `MFA_REQUIRED`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_344_mfa_login_step_up.py::test_mvp_344_enrolled_user_gets_an_mfa_challenge_not_tokens`
```python
"""MVP-344: TOTP step-up at login; clinicians and admins cannot obtain tokens without a second factor."""
from __future__ import annotations

import pyotp
import pytest
from fastapi import Depends, FastAPI

from identity_testkit import PASSWORD, bearer, build_identity_app, claims, client_for, seed_totp, seed_user
from src.api_deps import require_mfa_for_roles
from src.domain.value_objects import UserRole
from src.mfa import TOTPVerifier

LOGIN = "/api/v1/auth/login"
VERIFY = "/api/v1/auth/mfa/verify"


async def _challenge(c, email="alice@example.com") -> dict:
    r = await c.post(LOGIN, json={"email": email, "password": PASSWORD})
    assert r.status_code == 200
    return r.json()


async def test_mvp_344_enrolled_user_gets_an_mfa_challenge_not_tokens():
    app, state = build_identity_app()
    user = await seed_user(state)
    secret = await seed_totp(state, user)
    async with client_for(app) as c:
        challenge = await _challenge(c)
        assert challenge["mfa_required"] is True and "access_token" not in challenge
        r = await c.post(VERIFY, json={"mfa_token": challenge["mfa_token"], "code": pyotp.TOTP(secret).now()})
    assert r.status_code == 200
    access = claims(r.json()["access_token"])
    assert (access["amr"], access["mfa"]) == (["pwd", "otp"], True)
    assert access["loc"] == "en-US"                                      # D-29: every issuance path carries loc (WS-08.21)


async def test_mvp_344_a_totp_code_cannot_be_replayed():
    app, state = build_identity_app()
    user = await seed_user(state)
    secret = await seed_totp(state, user)
    code = pyotp.TOTP(secret).now()
    async with client_for(app) as c:
        first = await c.post(VERIFY, json={"mfa_token": (await _challenge(c))["mfa_token"], "code": code})
        second = await c.post(VERIFY, json={"mfa_token": (await _challenge(c))["mfa_token"], "code": code})
    assert first.status_code == 200
    assert second.status_code == 401 and second.json()["error"]["code"] == "MFA_INVALID"


def test_mvp_344_window_is_one_step_either_side():
    secret = pyotp.random_base32()
    now = 1_800_000_000.0
    verifier = TOTPVerifier()
    totp = pyotp.TOTP(secret)
    assert verifier.matching_step(secret, totp.at(now - 30), at=now) == int(now // 30) - 1
    assert verifier.matching_step(secret, totp.at(now + 30), at=now) == int(now // 30) + 1
    assert verifier.matching_step(secret, totp.at(now - 60), at=now) is None


@pytest.mark.parametrize("role", [UserRole.CLINICIAN, UserRole.ADMIN])
async def test_mvp_344_privileged_role_without_a_factor_must_enroll(role):
    app, state = build_identity_app()
    await seed_user(state, email="staff@example.com", role=role)
    async with client_for(app) as c:
        body = await _challenge(c, "staff@example.com")
    assert body["mfa_enrollment_required"] is True and "access_token" not in body
    assert claims(body["enrollment_token"])["type"] == "mfa_enroll"


async def test_mvp_344_plain_user_without_a_factor_logs_in_directly():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        body = await _challenge(c)
    assert claims(body["access_token"])["mfa"] is False


async def test_mvp_344_mfa_token_allows_five_wrong_codes_then_dies():
    app, state = build_identity_app()
    user = await seed_user(state)
    secret = await seed_totp(state, user)
    async with client_for(app) as c:
        token = (await _challenge(c))["mfa_token"]
        for _ in range(5):
            assert (await c.post(VERIFY, json={"mfa_token": token, "code": "000000"})).status_code == 401
        last = await c.post(VERIFY, json={"mfa_token": token, "code": pyotp.TOTP(secret).now()})
    assert last.status_code == 401 and last.json()["error"]["code"] == "MFA_TOKEN_INVALID"


async def test_mvp_344_recovery_code_works_once():
    app, state = build_identity_app()
    user = await seed_user(state)
    await seed_totp(state, user, recovery_codes=["abcde-fghij"])
    async with client_for(app) as c:
        ok = await c.post(VERIFY, json={"mfa_token": (await _challenge(c))["mfa_token"], "recovery_code": "abcde-fghij"})
        again = await c.post(VERIFY, json={"mfa_token": (await _challenge(c))["mfa_token"], "recovery_code": "abcde-fghij"})
    assert (ok.status_code, again.status_code) == (200, 401)


async def test_mvp_344_user_service_routes_can_require_mfa_for_privileged_roles():
    _app, state = build_identity_app()
    probe = FastAPI()
    probe.state.service = state

    @probe.get("/probe")
    async def guarded(user=Depends(require_mfa_for_roles("clinician", "admin"))):
        return {"ok": True}

    clinician = await seed_user(state, email="doc@example.com", role=UserRole.CLINICIAN)
    without = state.jwt_service.generate_token_pair(clinician.user_id, clinician.email, "clinician", mfa=False)
    with_mfa = state.jwt_service.generate_token_pair(clinician.user_id, clinician.email, "clinician", mfa=True)
    async with client_for(probe) as c:
        denied = await c.get("/probe", headers=bearer(without.access_token))
        allowed = await c.get("/probe", headers=bearer(with_mfa.access_token))
    assert denied.status_code == 403 and allowed.status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_344_mfa_login_step_up.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'require_mfa_for_roles' from 'src.api_deps'` and `ModuleNotFoundError: No module named 'src.mfa'`. The defect behind it: login issues tokens after a password check only, and `TokenPurpose.TWO_FACTOR_AUTH` is an unused placeholder, so clinician and admin accounts with cross-user PHI access are password-only (MVP-344).
- [ ] **Step 3: Implement** the repository, `TOTPVerifier`, `MFAService`, `mfa_router`, the step-up tokens, the login and OAuth branches, `require_mfa_for_roles` and the wiring. Approach: `matching_step` compares each candidate code with `hmac.compare_digest`, and `verify_code` succeeds only when `advance_last_used_step` succeeds, so a replayed or concurrent duplicate code fails.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (no existing test logs a clinician or admin in through `/auth/login`; the WS-08 tests that refresh or demote accounts use the `user` and `premium` roles)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.17: MFA enrollment with recovery codes, the Postgres factor store with codec-encrypted secrets, and re-authentication by code
**Covers:** MVP-344
**Severity:** blocker
**Wave:** 4
**Phase:** B
**Depends on:** WS-08.16, WS-09.7, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (same default as WS-08.16)
**Real infra:** yes
**Files:**
- Modify: `services/user-service/src/infrastructure/mfa_repository.py` (new `PostgresMFARepository(pool_name: str = "default")`. Every method runs inside `acting_as_user(user_id)` on `ConnectionPoolManager.acquire()`/`.transaction()`, so a lookup with no request context still finds the user's own rows. `advance_last_used_step` is one `UPDATE ... WHERE user_id = $1 AND confirmed_at IS NOT NULL AND (last_used_step IS NULL OR last_used_step < $2) RETURNING user_id`, and `consume_recovery_code` is one `UPDATE ... SET used_at = now() WHERE user_id = $1 AND code_hash = $2 AND used_at IS NULL RETURNING id`)
- Modify: `services/user-service/src/mfa.py`:
  - `MFAService.begin_enrollment(user_id, account_label: str) -> tuple[str, str]` (otpauth URI with issuer `Solace-AI`, base32 secret). It raises `MFAAlreadyEnrolled` when a confirmed factor exists and stores the pending secret encrypted, with `codec.active_key_id`.
  - `MFAService.confirm_enrollment(user_id, code) -> list[str] | None` (ten codes in the form `xxxxx-xxxxx`; only their hashes are stored);
  - `generate_recovery_codes(n: int = 10) -> list[str]`;
  - routes `POST /auth/mfa/enroll/start` and `/confirm`, which accept an access token or an `mfa_enroll` token. `confirm` with an enrollment token also returns a token pair (`amr=[<first factor>, "otp"]`, `mfa=True`, `loc=locale_claim_for(user)` from WS-08.21, D-29) and records `AUTH_MFA_ENROLLED`. `start` after confirmation returns 409 `MFA_ALREADY_ENROLLED`.
- Modify: `services/user-service/src/api_auth.py` (`/auth/reauth` with `code` verifies through `MFAService.verify_code` for an enrolled user; failures count in the `StepUpGuard`)
- Modify: `services/user-service/src/identity_wiring.py` (`PostgresMFARepository` when `runtime.pg_pool` is set, stored as `app.state.service.mfa_repository`, the name R-10 asks WS-14 to read; strict without a pool raises)
- Modify: `docs/api/auth.md` (enrollment, recovery codes, reauth by code)
- Test: `services/user-service/tests/test_mvp_344_mfa_enrollment.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_344_mfa_repository_real_pg.py`
**Interfaces:**
- Consumes: `get_phi_codec()`, `PHICodec.encrypt/decrypt(value, *, field, owner_id)`, `active_key_id`, `reset_phi_codec()` (IF-13, WS-09.7); `acting_as_user` (WS-07.1); slot 013 (WS-08.15); fixtures `pg_migrated` (IF-19) and `insert_minimal_user` (WS-08.12).
- Produces (IF-17): `POST /api/v1/auth/mfa/enroll/start` -> `{otpauth_uri, secret}`; `POST /api/v1/auth/mfa/enroll/confirm {code}` -> `{recovery_codes: [10]}`, plus `access_token`, `refresh_token`, `token_type` and `expires_in` when called with an enrollment token; `/auth/reauth {code}`; `PostgresMFARepository`; `MFAAlreadyEnrolled`; `MFARepository.delete_for_user` (request R-10).

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_344_mfa_enrollment.py::test_mvp_344_enrollment_shows_the_secret_once_and_confirm_returns_ten_recovery_codes` and the real-infra file
```python
"""MVP-344: users enroll TOTP and get 10 one-time recovery codes; clinicians enroll with their enrollment token."""
from __future__ import annotations

import pyotp

from identity_testkit import PASSWORD, bearer, build_identity_app, claims, client_for, login, seed_user
from src.domain.value_objects import UserRole

START = "/api/v1/auth/mfa/enroll/start"
CONFIRM = "/api/v1/auth/mfa/enroll/confirm"


async def test_mvp_344_enrollment_shows_the_secret_once_and_confirm_returns_ten_recovery_codes():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        auth = bearer((await login(c))["access_token"])
        start = (await c.post(START, headers=auth)).json()
        assert start["otpauth_uri"].startswith("otpauth://totp/") and "issuer=Solace-AI" in start["otpauth_uri"]
        confirm = await c.post(CONFIRM, headers=auth, json={"code": pyotp.TOTP(start["secret"]).now()})
        again = await c.post(START, headers=auth)
        challenge = (await c.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": PASSWORD})).json()
    codes = confirm.json()["recovery_codes"]
    assert confirm.status_code == 200 and len(codes) == 10 and len(set(codes)) == 10
    assert again.status_code == 409 and again.json()["error"]["code"] == "MFA_ALREADY_ENROLLED"
    assert challenge["mfa_required"] is True
    stored = repr(state.mfa_repository.__dict__)
    assert start["secret"] not in stored and not any(code in stored for code in codes)


async def test_mvp_344_wrong_confirmation_code_leaves_mfa_off():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        auth = bearer((await login(c))["access_token"])
        await c.post(START, headers=auth)
        r = await c.post(CONFIRM, headers=auth, json={"code": "000000"})
        assert r.status_code == 401
        assert "access_token" in (await c.post("/api/v1/auth/login",
                                               json={"email": "alice@example.com", "password": PASSWORD})).json()


async def test_mvp_344_clinician_enrolls_with_the_enrollment_token_and_gets_mfa_tokens():
    app, state = build_identity_app()
    await seed_user(state, email="doc@example.com", role=UserRole.CLINICIAN)
    async with client_for(app) as c:
        body = (await c.post("/api/v1/auth/login", json={"email": "doc@example.com", "password": PASSWORD})).json()
        enroll = bearer(body["enrollment_token"])
        assert (await c.post("/api/v1/auth/logout", headers=enroll)).status_code == 401
        start = (await c.post(START, headers=enroll)).json()
        done = (await c.post(CONFIRM, headers=enroll, json={"code": pyotp.TOTP(start["secret"]).now()})).json()
    access = claims(done["access_token"])
    assert (access["roles"], access["mfa"], access["amr"]) == (["clinician"], True, ["pwd", "otp"])
    assert access["loc"] == "en-US"                                      # D-29 (WS-08.21)


async def test_mvp_344_reauth_accepts_a_totp_code_for_enrolled_users():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        auth = bearer((await login(c))["access_token"])
        secret = (await c.post(START, headers=auth)).json()["secret"]
        await c.post(CONFIRM, headers=auth, json={"code": pyotp.TOTP(secret).now()})
        r = await c.post("/api/v1/auth/reauth", headers=auth, json={"code": pyotp.TOTP(secret).at(_next_step())})
    assert r.status_code == 200 and r.json()["reauth_token"]


def _next_step() -> float:
    """A code one step ahead: inside the +-1 window and newer than the step used to confirm."""
    import time

    return time.time() + 30
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_344_mfa_repository_real_pg.py
MVP-344 on the migrated schema as solace_app: secrets are ciphertext at rest, rows are per-user under RLS,
context-less lookups still work and step advancement is atomic."""
from __future__ import annotations

import asyncio
import time

import asyncpg
import pyotp
import pytest

from solace_common.request_context import acting_as_user
from solace_infrastructure.database.connection_manager import ConnectionPoolManager
from solace_security.phi_codec import get_phi_codec, reset_phi_codec
from src.infrastructure.mfa_repository import PostgresMFARepository
from src.mfa import MFAService

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


@pytest.fixture
def codec(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "ab" * 32)
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", "synthetic-salt-for-tests-only-32b")
    monkeypatch.setenv("ENCRYPTION_KEY_ID", "k1")
    reset_phi_codec()
    yield get_phi_codec()
    reset_phi_codec()


@pytest.fixture
async def su(pg_migrated):
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    yield conn
    await conn.close()


async def test_mvp_344_secret_is_ciphertext_at_rest_and_found_without_request_context(su, codec, insert_minimal_user):
    user_id = await insert_minimal_user(su)
    service = MFAService(PostgresMFARepository(), codec=codec)
    _uri, secret = await service.begin_enrollment(user_id, "user@example.com")
    assert await service.confirm_enrollment(user_id, pyotp.TOTP(secret).now()) is not None
    raw = await su.fetchval("SELECT secret_ciphertext FROM user_mfa_factors WHERE user_id = $1", user_id)
    assert raw.startswith("enc:v1:") and secret not in raw
    assert await service.has_confirmed_factor(user_id) is True


async def test_mvp_344_another_users_context_sees_no_factor_and_the_guc_does_not_leak(su, codec, insert_minimal_user):
    owner, other = await insert_minimal_user(su), await insert_minimal_user(su)
    service = MFAService(PostgresMFARepository(), codec=codec)
    _uri, secret = await service.begin_enrollment(owner, "o@example.com")
    await service.confirm_enrollment(owner, pyotp.TOTP(secret).now())
    with acting_as_user(other):
        async with ConnectionPoolManager.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM user_mfa_factors") == 0
    assert await service.has_confirmed_factor(owner) is True
    assert await service.has_confirmed_factor(other) is False


async def test_mvp_344_concurrent_use_of_one_code_succeeds_once(su, codec, insert_minimal_user):
    user_id = await insert_minimal_user(su)
    service = MFAService(PostgresMFARepository(), codec=codec)
    _uri, secret = await service.begin_enrollment(user_id, "c@example.com")
    await service.confirm_enrollment(user_id, pyotp.TOTP(secret).at(time.time() - 30))
    code = pyotp.TOTP(secret).now()
    results = await asyncio.gather(*(service.verify_code(user_id, code) for _ in range(5)))
    assert results.count(True) == 1
    assert await PostgresMFARepository().delete_for_user(user_id) >= 1
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_344_mfa_enrollment.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_344_mfa_repository_real_pg.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. The enrollment routes return 404, so no user can create a factor and no clinician can complete the mandatory enrollment WS-08.16 demands (MVP-344). `/auth/reauth {code}` returns 400 `MFA_NOT_ENROLLED`. The real-infra file fails at `ImportError: cannot import name 'PostgresMFARepository'`.
- [ ] **Step 3: Implement** `PostgresMFARepository`, the enrollment methods and routes, recovery-code generation, reauth by code, the wiring and the doc section. Approach: `confirm_enrollment` calls `confirm_factor(user_id, step)` and `add_recovery_codes` in one `ConnectionPoolManager.transaction()`, so a factor never becomes active without its recovery codes.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.18: Forgot and reset password with a 30-minute, single-use, hashed token that revokes every session
**Covers:** MVP-099
**Severity:** high
**Wave:** 4
**Phase:** B
**Depends on:** WS-08.17, WS-14.6, WS-07.3, WS-13.12, WS-13.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes
**Files:**
- Create: `services/user-service/src/password_reset.py`:
  - `PASSWORD_RESET_TTL_MINUTES = 30`;
  - `PasswordResetTokenStore` protocol: `async issue(user_id, token_hash, expires_at) -> None` (marks the user's older unused tokens used), `async redeem(token_hash) -> UUID | None` (atomic single use, unexpired only), `async delete_for_user(user_id) -> int`;
  - `InMemoryPasswordResetTokenStore(clock=lambda: datetime.now(timezone.utc))`;
  - `PostgresPasswordResetTokenStore(pool_name="default")` on the non-RLS `password_reset_tokens` table (`UPDATE ... SET used_at = now() WHERE token_hash = $1 AND used_at IS NULL AND expires_at > now() RETURNING user_id`);
  - `PasswordResetMailer(*, settings: ServiceIntegrationSettings | None = None, http_client_factory: Callable[[], httpx.AsyncClient] | None = None)` with `async send(user: User, token: str, *, expiry_minutes: int) -> bool`. It posts IF-09 `POST {settings.notification_service_url}/api/v1/notifications/email` with body `{"to_email": user.email, "template_type": "password_reset", "variables": {"display_name", "reset_link": f"{settings.frontend_url}/reset-password?token=<token>", "expiry_minutes"}, "priority": "high"}` and headers `await service_auth_headers("notification-service")` (user-service holds `service:write:notification`). Any error or non-2xx returns False and logs WARNING `password_reset_email_failed` with `error_type` or `status` only; neither the token nor the address is logged. `ServiceIntegrationSettings` is imported read-only from `src.domain.service` (WS-14.6), so the URLs come from `SERVICE_NOTIFICATION_SERVICE_URL` and `USER_SERVICE_FRONTEND_URL`. The `password_reset` template and its variables `display_name`, `reset_link` and `expiry_minutes` are notification-service's (WS-13); WS-13.7 locks them for this mailer (request R-15), and this task edits nothing in notification-service.
  - `PasswordResetService(store, user_repository, password_service, session_revoker, mailer)` with `async request_reset(email: str) -> None` and `async reset(token: str, new_password: str) -> UUID`. `reset` raises `InvalidResetToken`, hashes with `hash_password_async`, loads the user, calls `user.set_password_hash(new_hash, mark_email_verified=True)` and `await user_repository.update(user)`, revokes all sessions and records `AUTH_PASSWORD_RESET`. It needs no new `UserService` method.
  - `password_reset_router`: `POST /auth/password/forgot {email}` always returns 202 `{"status": "accepted"}`. The lookup and the email run in a background task, with a per-address throttle of 3 per hour; the email goes through `mailer.send(user, token, expiry_minutes=30)`. `POST /auth/password/reset {token, new_password}` returns 200 `{"status": "password_reset"}` or 400 `INVALID_RESET_TOKEN`, and the new password must pass the registration strength rules.
- Modify: `services/user-service/src/domain/entities.py` (new `User.set_password_hash(new_hash: str, *, mark_email_verified: bool = False) -> None`: sets `password_hash` and `updated_at` and clears `login_attempts` and `locked_until`; with `mark_email_verified` it sets `email_verified=True`, clears `email_verification_token` and turns `PENDING_VERIFICATION` into `ACTIVE`; a suspended or deleted account keeps its status)
- Modify: `services/user-service/src/api_auth.py` (`auth_router.include_router(password_reset_router)`; the password-strength validator is shared as `validate_password_strength(value: str) -> str`)
- Modify: `services/user-service/src/identity_wiring.py` (Postgres store when `runtime.pg_pool` is set, stored as `app.state.service.password_reset_store`, the name R-10 asks WS-14 to read; strict without a pool raises; `PasswordResetMailer()` with default settings)
- Modify: `services/user-service/tests/identity_testkit.py` (new overrides `reset_store` (default `InMemoryPasswordResetTokenStore()`) and `reset_mailer` (default `RecordingResetMailer()`, a helper class with `sent: list[tuple[str, str, int]]` holding email, raw token and expiry, and `async send(user, token, *, expiry_minutes) -> bool`), both popped before building `password_reset=PasswordResetService(reset_store, user_repository, password_service, session_revoker, reset_mailer)`; the mailer is also kept on the state as `reset_mailer`, which the tests read)
- Modify: `docs/api/auth.md` (forgot and reset)
- Test: `services/user-service/tests/test_mvp_099_password_reset.py`
- Test: `tests/realinfra/identity-auth-mfa/test_mvp_099_reset_tokens_real_pg.py`
**Interfaces:**
- Consumes: `UserRepository.get_by_id`, `get_by_email` and `update(user)` (WS-14's repository, called through `api_deps.get_user_repository`, never edited); `ServiceIntegrationSettings.notification_service_url` and `.frontend_url` (WS-14.6); IF-04 `service_auth_headers(target_service, *, on_behalf_of=None)` (WS-07.3); IF-09 `POST /api/v1/notifications/email {to_email, template_type, variables, priority}` accepting a `service:write:notification` token (WS-13.12); the `password_reset` notification template, whose name, subject `Reset your Solace-AI password`, required variables (`display_name`, `reset_link`, `expiry_minutes`) and verbatim plain-text rendering of `reset_link` WS-13.7 locks (request R-15); slot 013 (WS-08.15); `SessionRevoker` (WS-08.4); `audit_auth_event` (WS-08.14).
- Produces (IF-17): `POST /api/v1/auth/password/forgot`, `POST /api/v1/auth/password/reset`; `PasswordResetService`, `PasswordResetMailer`, `User.set_password_hash` (WS-08.20 reuses it), `PasswordResetTokenStore.delete_for_user` (request R-10); `validate_password_strength`; error code `INVALID_RESET_TOKEN`; test-kit `RecordingResetMailer`.

- [ ] **Step 1: Write the failing tests** in `services/user-service/tests/test_mvp_099_password_reset.py::test_mvp_099_reset_sets_the_password_and_revokes_every_session` and the real-infra file
```python
"""MVP-099: a locked-out or forgetful user can recover the account by email, safely."""
from __future__ import annotations

import json
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


async def test_mvp_099_forgot_answers_the_same_for_known_and_unknown_addresses():
    app, state = build_identity_app()
    await seed_user(state)
    sent = _mailbox(state)
    async with client_for(app) as c:
        known = await c.post(FORGOT, json={"email": "alice@example.com"})
        unknown = await c.post(FORGOT, json={"email": "ghost@example.com"})
    assert (known.status_code, unknown.status_code) == (202, 202) and known.json() == unknown.json()
    assert [(e, m) for e, _t, m in sent] == [("alice@example.com", 30)]


async def test_mvp_099_reset_sets_the_password_and_revokes_every_session():
    app, state = build_identity_app()
    await seed_user(state)
    sent = _mailbox(state)
    async with client_for(app) as c:
        old = await login(c)
        await c.post(FORGOT, json={"email": "alice@example.com"})
        r = await c.post(RESET, json={"token": sent[0][1], "new_password": NEW})
        assert r.status_code == 200 and r.json() == {"status": "password_reset"}
        assert (await c.post("/api/v1/auth/refresh", json={"refresh_token": old["refresh_token"]})).status_code == 401
        assert (await c.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": PASSWORD})).status_code == 401
        assert (await c.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": NEW})).status_code == 200


async def test_mvp_099_token_is_single_use_and_older_tokens_die_on_a_new_request():
    app, state = build_identity_app()
    await seed_user(state)
    sent = _mailbox(state)
    async with client_for(app) as c:
        await c.post(FORGOT, json={"email": "alice@example.com"})
        await c.post(FORGOT, json={"email": "alice@example.com"})
        stale = await c.post(RESET, json={"token": sent[0][1], "new_password": NEW})
        fresh = await c.post(RESET, json={"token": sent[1][1], "new_password": NEW})
        reused = await c.post(RESET, json={"token": sent[1][1], "new_password": NEW})
    assert (stale.status_code, fresh.status_code, reused.status_code) == (400, 200, 400)
    assert reused.json()["error"]["code"] == "INVALID_RESET_TOKEN"


async def test_mvp_099_token_expires_after_30_minutes_and_is_stored_hashed():
    clock = SimpleNamespace(now=datetime.now(timezone.utc))
    store = InMemoryPasswordResetTokenStore(clock=lambda: clock.now)
    app, state = build_identity_app(reset_store=store)
    await seed_user(state)
    sent = _mailbox(state)
    async with client_for(app) as c:
        await c.post(FORGOT, json={"email": "alice@example.com"})
        assert sent[0][1] not in repr(store.__dict__)
        clock.now += timedelta(minutes=31)
        r = await c.post(RESET, json={"token": sent[0][1], "new_password": NEW})
    assert r.status_code == 400


async def test_mvp_099_reset_on_a_pending_account_proves_the_inbox_and_evicts_a_squatter_password():
    app, state = build_identity_app()
    squatter = await seed_user(state, email="victim@example.com", password="Attacker-Pass-1",
                               status=AccountStatus.PENDING_VERIFICATION, verified=False)
    sent = _mailbox(state)
    async with client_for(app) as c:
        await c.post(FORGOT, json={"email": "victim@example.com"})
        await c.post(RESET, json={"token": sent[0][1], "new_password": NEW})
        attacker = await c.post("/api/v1/auth/login", json={"email": "victim@example.com", "password": "Attacker-Pass-1"})
    user = await state.user_repository.get_by_id(squatter.user_id)
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
```
```python
"""tests/realinfra/identity-auth-mfa/test_mvp_099_reset_tokens_real_pg.py
MVP-099 on the migrated schema as solace_app: single use under concurrency, expiry by database time, hash at rest."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest

from src.password_reset import PostgresPasswordResetTokenStore

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_099_postgres_reset_tokens_are_single_use_and_expire(pg_migrated, insert_minimal_user):
    su = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        user_id = await insert_minimal_user(su)
        store = PostgresPasswordResetTokenStore()
        live, dead = hashlib.sha256(b"live-token").hexdigest(), hashlib.sha256(b"dead-token").hexdigest()
        await store.issue(user_id, dead, datetime.now(timezone.utc) - timedelta(seconds=1))
        await store.issue(user_id, live, datetime.now(timezone.utc) + timedelta(minutes=30))
        results = await asyncio.gather(*(store.redeem(live) for _ in range(5)))
        assert results.count(user_id) == 1 and results.count(None) == 4
        assert await store.redeem(dead) is None
        rows = await su.fetch("SELECT token_hash, used_at FROM password_reset_tokens WHERE user_id = $1", user_id)
        assert len(rows) == 2 and all(len(r["token_hash"]) == 64 and r["used_at"] is not None for r in rows)
        assert await store.delete_for_user(user_id) == 2
    finally:
        await su.close()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_099_password_reset.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa\test_mvp_099_reset_tokens_real_pg.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.password_reset'`. The defect behind it: no route issues or redeems a reset token, so a user who forgets the password (or is locked out) can never get back into the account (MVP-099); `generate_password_reset_token` and `verify_password_reset_token` (`services/user-service/src/infrastructure/token_service.py:184` and `:222`) have no caller outside tests.
- [ ] **Step 3: Implement** `password_reset.py`, the router aggregation, the shared strength validator, the wiring and the doc section. Tokens are `secrets.token_urlsafe(32)`; only `sha256(token)` hex is stored, and no log line carries the token or the email.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.19: The user-token issuer is RS256-only in strict environments and fails startup on partial key material
**Covers:** none (supports MVP-209, owned and closed by WS-07)
**Severity:** medium
**Wave:** 4
**Phase:** B
**Depends on:** WS-08.18, WS-07.10, WS-03.14
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: from W4 user tokens are RS256-only and `AUTH_ACCEPT_LEGACY_HS256=false`); D-07 (default assumed: key material arrives through NAME_FILE from the secrets store)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`JWTConfig.accept_legacy_hs256` defaults to False; `JWTService.__init__` raises `IdentityConfigError` instead of logging and continuing when RS256 is configured but the key manager cannot be built)
- Modify: `services/user-service/src/identity_wiring.py` (new `build_jwt_service(*, secret_key: str, access_token_expire_minutes: int, refresh_token_expire_days: int, blacklist, strict: bool) -> JWTService`. It holds the RS256 composition that moved from `main.py`, now without its `except Exception: fall back to HS256`. In strict environments it requires `AuthSettings().algorithm == "RS256"`, a signing key, an active kid and a public keyset containing the active kid, else it raises `IdentityConfigError` naming the missing variable; `start_identity` calls it.)
- Modify: `services/user-service/tests/test_rs256_issuer_migration.py` (only assertions that relied on the dual-accept default now pass `accept_legacy_hs256=True` explicitly)
- Test: `services/user-service/tests/test_mvp_209_user_token_rs256_strict.py`
**Interfaces:**
- Consumes: `solace_security.auth.AuthSettings` with the strict RS256 rule and `KeyManager.from_settings` requiring a keyset in staging (WS-07.10); `generate_rsa_keypair(kid)` (`solace_security.keys`); key files and env names `AUTH_ALGORITHM`, `AUTH_PRIVATE_KEY_PEM`, `AUTH_ACTIVE_KID`, `AUTH_PUBLIC_KEYS_JSON`, `AUTH_ACCEPT_LEGACY_HS256` (IF-02/IF-22, WS-03.14).
- Produces: `build_jwt_service`; `IdentityConfigError`; JWKS at `/.well-known/jwks.json` publishing every kid of the keyset (IF-17).

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_209_user_token_rs256_strict.py::test_mvp_209_strict_issuer_refuses_hs256`
```python
"""Supports MVP-209: user-service signs RS256 only in strict envs, fails startup on partial keys, and rotates by kid."""
from __future__ import annotations

import json
from uuid import uuid4

import jwt
import pytest

from solace_security.auth import InMemoryTokenBlacklist
from solace_security.keys import generate_rsa_keypair
from src.identity_wiring import IdentityConfigError, build_jwt_service
from src.infrastructure.jwt_service import JWTConfig, TokenInvalidError

SECRET = "test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"


def _build(strict=True):
    return build_jwt_service(secret_key=SECRET, access_token_expire_minutes=15, refresh_token_expire_days=30,
                             blacklist=InMemoryTokenBlacklist(), strict=strict)


def _rs256_env(monkeypatch, active="k-new", keyset=("k-old", "k-new")):
    keys = {kid: generate_rsa_keypair(kid)[:2] for kid in keyset}
    monkeypatch.setenv("AUTH_SECRET_KEY", SECRET)
    monkeypatch.setenv("AUTH_ALGORITHM", "RS256")
    monkeypatch.setenv("AUTH_ACCEPT_LEGACY_HS256", "false")
    monkeypatch.setenv("AUTH_ACTIVE_KID", active)
    monkeypatch.setenv("AUTH_PRIVATE_KEY_PEM", keys[active][0])
    monkeypatch.setenv("AUTH_PUBLIC_KEYS_JSON", json.dumps({kid: pub for kid, (_p, pub) in keys.items()}))
    return keys


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_209_strict_issuer_refuses_hs256(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.setenv("AUTH_SECRET_KEY", SECRET)
    monkeypatch.delenv("AUTH_ALGORITHM", raising=False)
    with pytest.raises(IdentityConfigError, match="RS256"):
        _build()


@pytest.mark.parametrize("missing", ["AUTH_PUBLIC_KEYS_JSON", "AUTH_PRIVATE_KEY_PEM", "AUTH_ACTIVE_KID"])
def test_mvp_209_partial_key_material_fails_startup(monkeypatch, missing):
    monkeypatch.setenv("ENVIRONMENT", "production")
    _rs256_env(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(IdentityConfigError, match=missing):
        _build()


async def test_mvp_209_strict_tokens_are_rs256_and_hs256_is_rejected(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    _rs256_env(monkeypatch)
    svc = _build()
    pair = svc.generate_token_pair(uuid4(), "u@example.com", "user")
    header = jwt.get_unverified_header(pair.access_token)
    assert (header["alg"], header["kid"]) == ("RS256", "k-new")
    forged = jwt.encode({"sub": str(uuid4()), "user_id": str(uuid4()), "email": "x@example.com", "role": "admin",
                         "type": "access", "jti": "j", "iss": "solace-ai", "aud": "solace-ai-api",
                         "exp": 9999999999}, SECRET, algorithm="HS256")
    with pytest.raises(TokenInvalidError):
        await svc.verify_token(forged)


async def test_mvp_209_token_signed_with_a_retired_kid_still_verifies(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    keys = _rs256_env(monkeypatch)
    svc = _build()
    uid = uuid4()
    old = jwt.encode({"sub": str(uid), "user_id": str(uid), "email": "u@example.com", "role": "user", "type": "access",
                      "jti": "j-old", "iss": "solace-ai", "aud": "solace-ai-api", "exp": 9999999999},
                     keys["k-old"][0], algorithm="RS256", headers={"kid": "k-old"})
    assert (await svc.verify_token(old)).user_id == uid


def test_mvp_209_legacy_hs256_acceptance_is_off_by_default():
    assert JWTConfig(secret_key=SECRET).accept_legacy_hs256 is False
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_209_user_token_rs256_strict.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'build_jwt_service'`. The defect behind it: the issuer composition that moved from `main.py` falls back to HS256 with a warning on any RS256 problem, and `JWTConfig.accept_legacy_hs256` defaults to True, so the RS256 issuer is inert in every deploy and forged HS256 tokens signed with the shared secret stay valid (MVP-209).
- [ ] **Step 3: Implement** `build_jwt_service`, `IdentityConfigError`, the strict checks and the new defaults. Approach: in strict environments `build_jwt_service` first checks each of `AUTH_ALGORITHM == RS256`, `AUTH_PRIVATE_KEY_PEM`, `AUTH_ACTIVE_KID` and `AUTH_PUBLIC_KEYS_JSON` (the keyset must contain the active kid) and raises `IdentityConfigError` naming the first missing variable; any error from `AuthSettings()` or `KeyManager` is re-raised as `IdentityConfigError` with the variable name only, never a value. Outside strict environments, the HS256 fallback stays and logs a WARNING without key material.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.20: Password change verifies the current password before hashing and ends the session after repeated failures
**Covers:** MVP-226
**Severity:** medium
**Wave:** 4
**Phase:** B
**Depends on:** WS-08.19, WS-08.18, WS-07.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: the per-route limit `auth-password-change` of 5/min keyed on the user comes from WS-07.7; this task adds the per-session failure counter)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/api_auth.py` (`change_password`:
  - loads the user and checks `current_password` with `verify_password_async` before any hashing;
  - a wrong password calls `StepUpGuard.record_failure(sid)` and returns 400 `INVALID_PASSWORD`, or 401 `SESSION_REVOKED` once the guard revoked the session (the account itself is not locked);
  - success clears the counter, hashes with `hash_password_async`, persists with `user.set_password_hash(new_hash)` (WS-08.18) and `await user_repository.update(user)`, revokes all sessions and records `AUTH_PASSWORD_CHANGED`;
  - it no longer calls `UserService.change_password`.)
- Modify: `docs/api/auth.md` (password change semantics; final W4 review of the whole document)
- Test: `services/user-service/tests/test_mvp_226_password_change_guard.py`
**Interfaces:**
- Consumes: `StepUpGuard` (WS-08.8); `User.set_password_hash` (WS-08.18) and `UserRepository.update` (WS-14's repository, called, not edited); `verify_password_async`, `hash_password_async` (WS-08.3); the `auth-password-change` route bucket (WS-07.7).
- Produces: `POST /api/v1/users/me/password` error codes `INVALID_PASSWORD` and `SESSION_REVOKED`; failures shared with `/auth/reauth` through the same `stepup_fail:<sid>` counter.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_226_password_change_guard.py::test_mvp_226_wrong_current_password_never_hashes_the_new_one`
```python
"""MVP-226: a stolen access token cannot brute-force the current password through POST /users/me/password."""
from __future__ import annotations

from identity_testkit import FAST_PASSWORDS, PASSWORD, bearer, build_identity_app, client_for, login, seed_user
from src.infrastructure.password_service import PasswordService

CHANGE = "/api/v1/users/me/password"


class _CountingPasswords(PasswordService):
    def __init__(self, config):
        super().__init__(config)
        self.new_hashes = 0

    async def hash_password_async(self, password):
        self.new_hashes += 1
        return await super().hash_password_async(password)


async def test_mvp_226_wrong_current_password_never_hashes_the_new_one():
    passwords = _CountingPasswords(FAST_PASSWORDS)
    app, state = build_identity_app(password_service=passwords)
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        passwords.new_hashes = 0
        r = await c.post(CHANGE, headers=bearer(tokens["access_token"]),
                         json={"current_password": "Wrong-Horse-1", "new_password": "Another-Horse-7"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_PASSWORD"
    assert passwords.new_hashes == 0


async def test_mvp_226_fifth_wrong_guess_ends_the_session_but_not_the_account():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        codes = [(await c.post(CHANGE, headers=bearer(tokens["access_token"]),
                               json={"current_password": f"Wrong-Horse-{i}", "new_password": "Another-Horse-7"}))
                 .json()["error"]["code"] for i in range(5)]
        refresh = await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        relogin = await c.post("/api/v1/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert codes == ["INVALID_PASSWORD"] * 4 + ["SESSION_REVOKED"]
    assert refresh.status_code == 401 and relogin.status_code == 200


async def test_mvp_226_reauth_and_password_change_failures_share_one_budget():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        auth = bearer((await login(c))["access_token"])
        for _ in range(3):
            await c.post("/api/v1/auth/reauth", headers=auth, json={"password": "Wrong-Horse-1"})
        await c.post(CHANGE, headers=auth, json={"current_password": "Wrong-Horse-1", "new_password": "Another-Horse-7"})
        fifth = await c.post(CHANGE, headers=auth, json={"current_password": "Wrong-Horse-1",
                                                         "new_password": "Another-Horse-7"})
    assert fifth.json()["error"]["code"] == "SESSION_REVOKED"


async def test_mvp_226_success_persists_through_the_user_repository_and_revokes_sessions():
    app, state = build_identity_app()
    await seed_user(state)
    async with client_for(app) as c:
        a, b = await login(c), await login(c)
        r = await c.post(CHANGE, headers=bearer(a["access_token"]),
                         json={"current_password": PASSWORD, "new_password": "Another-Horse-7"})
        assert r.status_code == 204
        assert (await c.post("/api/v1/auth/refresh", json={"refresh_token": b["refresh_token"]})).status_code == 401
        assert (await c.post("/api/v1/auth/login", json={"email": "alice@example.com",
                                                         "password": "Another-Horse-7"})).status_code == 200
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_226_password_change_guard.py -o "addopts=" -p no:logging -q`
Expected: FAIL. On a wrong current password the route hashes the new password first (`new_hashes == 1`). The fifth wrong guess still returns 400 `INVALID_PASSWORD`, because nothing counts failures (MVP-226). The shared-budget test likewise never sees `SESSION_REVOKED`. The success test is a regression guard: it passes before and after, because today's route persists through `UserService.change_password` (`services/user-service/src/api.py:760-791`), and it must stay green once the route persists through `User.set_password_hash`.
- [ ] **Step 3: Implement** the new `change_password` flow and the doc section.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\identity-auth-mfa -m realinfra -o "addopts=" -p no:logging -q`. Expected: all pass (`test_api.py::TestPasswordChange` cases assert 204 and 400 as before; `test_mvp_217_auth_contract_doc.py` passes with every W4 route documented)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.20:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-08.21: Access tokens carry the user's locale claim `loc`, read from the profile at every issuance (D-29)
**Covers:** none (supports D-29: in-band crisis resources for the user's locale with no lookup at screen time)
**Supports:** MVP-221 (closed by WS-08.7; this task pins its locale grammar into the token and against `solace_clinical`)
**Severity:** high
**Wave:** 3
**Phase:** B
**Depends on:** WS-08.13, WS-08.7, WS-12.2
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** D-29 (decided 2026-10-19: several countries, US first. The user's locale must be available at screen time without a database or network call, so the access token carries it as the claim `loc`, issued at login from the profile locale, default `en-US`. A missing claim means en-US to every consumer; a stored value that cannot be normalized is issued as `und`, which consumers serve the generic fallback for, never guessing a country)
**Real infra:** no
**Files:**
- Modify: `services/user-service/src/infrastructure/jwt_service.py` (`generate_token_pair(..., loc: str | None = None)` writes `loc` into the access token only, when given)
- Modify: `services/user-service/src/auth.py` (new `locale_claim_for(user) -> str`: `"en-US"` when `user.locale` is `None` or blank; else `normalize_locale(user.locale)`; on `ValueError` returns `"und"` and logs WARNING `token_locale_claim_invalid` with no user data and no locale text; never raises)
- Modify: `services/user-service/src/api_auth.py` (`login` and `refresh_token`, including the grace re-mint, pass `loc=locale_claim_for(user)`; refresh uses the account it already re-reads (WS-08.5), so a profile change shows up at the next refresh)
- Modify: `services/user-service/src/oauth.py` (`/exchange` and `/link` do the same)
- Modify: `docs/api/auth.md` (access-token claims gain `loc`: canonical `ll` or `ll-CC`, default `en-US`, `und` when unknown; it drives the crisis resources the orchestrator shows)
- Test: `services/user-service/tests/test_mvp_221_locale_claim.py`
**Interfaces:**
- Consumes: `normalize_locale` (WS-08.7, the profile grammar); `User.locale` (the entity default `en-US`; WS-14.22 validates the profile update route); WS-08.13's four issuance call sites; `solace_clinical.canonical_locale` (WS-12.2, test only: the parity check that the issuer and the clinical resolver use one grammar).
- Produces (IF-17 addition, D-29): the access-token claim `loc: str` on every issuance path: login, refresh and the grace re-mint, OAuth `/exchange` and `/link`, and from W4 `/auth/mfa/verify` (WS-08.16) and `/auth/mfa/enroll/confirm` with an enrollment token (WS-08.17). It is never on refresh, `reauth`, `mfa` or `mfa_enroll` tokens. `src.auth.locale_claim_for(user) -> str`. WS-07 surfaces the claim as `AuthenticatedUser.locale` (WS-11's request to WS-07), WS-11.21 passes it to `run_turn(locale=)` with no lookup, and WS-10.5 resolves it with `solace_clinical.resolve_locale`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_221_locale_claim.py::test_d29_loc_claim_is_the_profile_locale_at_login_and_refresh`
```python
"""D-29: access tokens carry the user's profile locale as `loc` (default en-US), re-read at every issuance."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from structlog.testing import capture_logs

from identity_testkit import build_identity_app, claims, client_for, enable_google_oauth, google_round_trip, login, seed_user
from solace_clinical import canonical_locale
from src.auth import locale_claim_for
from src.domain.entities import normalize_locale
from src.infrastructure.oauth_google import GoogleIdentity


async def test_d29_loc_claim_is_the_profile_locale_at_login_and_refresh():
    app, state = build_identity_app()
    user = await seed_user(state)
    async with client_for(app) as c:
        tokens = await login(c)
        assert claims(tokens["access_token"])["loc"] == "en-US"           # the profile default
        stored = await state.user_repository.get_by_id(user.user_id)
        stored.update_profile(locale="fr_fr")                             # canonicalized by the entity (WS-08.7)
        await state.user_repository.update(stored)
        refreshed = (await c.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})).json()
    assert claims(refreshed["access_token"])["loc"] == "fr-FR"
    assert "loc" not in claims(tokens["refresh_token"]) and "loc" not in claims(refreshed["refresh_token"])


async def test_d29_loc_claim_is_minted_on_google_sign_in():
    app, state = build_identity_app()
    enable_google_oauth(state, identity=GoogleIdentity(provider_user_id="g-l", email="l@example.com", email_verified=True))
    async with client_for(app) as c:
        code = parse_qs(urlsplit((await google_round_trip(c)).headers["location"]).query)["code"][0]
        tokens = (await c.post("/api/v1/auth/oauth/google/exchange", json={"code": code})).json()
    assert claims(tokens["access_token"])["loc"] == "en-US"


class _Profile:
    def __init__(self, locale):
        self.locale = locale


@pytest.mark.parametrize("stored,claim", [("en-US", "en-US"), ("pt-BR", "pt-BR"), (None, "en-US"), ("", "en-US"),
                                          ("english", "und")])
def test_d29_locale_claim_for_never_fails_and_never_guesses_a_country(stored, claim):
    with capture_logs() as logs:
        assert locale_claim_for(_Profile(stored)) == claim
    assert all("english" not in str(entry) for entry in logs)


@pytest.mark.parametrize("raw", ["en-US", "en_us", "EN-us", "pt-BR", "en", "fil-PH", "english", "en-USA", "e1-US",
                                 "en--US", ""])
def test_d29_profile_grammar_matches_the_clinical_resolver(raw):
    expected = canonical_locale(raw)
    if expected is None:
        with pytest.raises(ValueError):
            normalize_locale(raw)
    else:
        assert normalize_locale(raw) == expected
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests\test_mvp_221_locale_claim.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'locale_claim_for' from 'src.auth'`. The defect behind it: no token carries the user's locale, so the orchestrator would need a database or network lookup at screen time, which the crisis invariant forbids, and every user would be shown the US resources (D-29). The parity test already passes after WS-08.7 and WS-12.2; it is a lock that keeps the issuer's grammar and `solace_clinical.canonical_locale` identical.
- [ ] **Step 3: Implement** `locale_claim_for`, the `loc` parameter and the four call sites (the same sites WS-08.13 passes `cns` at), and the `docs/api/auth.md` claim row.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_mvp_100_cns_claim.py` and `test_mvp_217_auth_contract_doc.py` included)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-08.21:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| # | Owner WS | Exact change | Files | WS-08 task that depends on it |
|---|---|---|---|---|
| R-1 | WS-14 | Produced by WS-14.3 (W0 phase B): the verbatim split as the skeleton describes: `auth_router` in `api_auth.py` (the `/auth/*` routes, `POST /users/me/password`, `DELETE /users/me` and the JWKS route, with handler names unchanged); the providers, `get_current_user`, `UserResponse`, `user_to_response` and `TokenResponse` in `api_deps.py`; `start_identity(runtime, app)`/`stop_identity()` in `identity_wiring.py`; `main.py` includes `auth_router` once; the existing user-service tests are re-pointed to `src.api_auth`. | `services/user-service/src/{api_auth,api_deps,identity_wiring,main}.py`, `services/user-service/tests/*` | all (WS-08.1 and WS-08.2 name WS-14.3) |
| R-2 | WS-14 | Partly produced. WS-14.6 (W1) delivers `UserService.verify_email(user_id: UUID \| None, token)` with hashed, expiring tokens and the link `{frontend_url}/verify-email?user_id=<uuid>&token=<t>`; WS-08.2 calls it, so the earlier `verify_email_by_token` ask is withdrawn. Still open (no WS-14 task yet): send the verification email after commit from a background task rather than inline in `create_user`, so `register` timing does not reveal whether an account was created (the timing half of MVP-217). | `services/user-service/src/domain/service.py` | WS-08.2 (via WS-14.6); WS-08.7 timing half (open) |
| R-3 | WS-14 | Open (no WS-14 task yet). W2 phase B: `UserService.send_account_exists_email(user: User) -> bool`, posting template `account_exists` with variables `display_name` and `login_link` (`{frontend_url}/login`) through the same service-token notification call as WS-14.6, and logging nothing that identifies the address. Until it lands, WS-08.7's best-effort background call logs `account_exists_notice_failed` and the 202 is unchanged. | `services/user-service/src/domain/service.py` | WS-08.7 (owner notice only; the oracle fix does not wait on it) |
| R-4 | WS-13 | Open (no WS-13 task yet). W2 phase B: `TemplateType.ACCOUNT_EXISTS = "account_exists"` with a default template (subject "Someone tried to create a Solace-AI account with this address", required variables `display_name`, `login_link`; no token, no PHI). | `services/notification-service/src/domain/templates.py` | WS-08.7 (owner notice only) |
| R-5 | WS-14 | Open and optional (no WS-14 task). `UserService.change_password` verifies with `await self._password_service.verify_password_async(...)`. Without it, the password-change route keeps one synchronous verify on the loop in W2-W3; WS-08.20 (W4) stops calling `change_password`, which closes that path of MVP-210 inside this workstream. | `services/user-service/src/domain/service.py` | none (WS-08.20 closes the route path) |
| R-6 | WS-14 | Withdrawn. WS-08.18 persists the new hash through its own entity method `User.set_password_hash` and WS-14's existing `UserRepository.update`, and sends the reset email through its own `PasswordResetMailer` (IF-09 `/api/v1/notifications/email`, `service_auth_headers`, the existing `password_reset` template), so no new `UserService` method is needed. WS-08.20 reuses `User.set_password_hash`. | none | none |
| R-7 | WS-14 | Produced by WS-14.17 (W3 phase B) as `UserService.get_consent_claim(user_id) -> {"ok": bool, "version": str}`; revoking a required consent calls `session_manager.revoke_all_user_sessions(user_id)`. WS-08.13 consumes `get_consent_claim` and makes `revoke_all_user_sessions` write the sid keys, which is R-14-9's session half. | `services/user-service/src/domain/service.py`, `services/user-service/src/api.py` | WS-08.13 |
| R-8 | WS-14 | Produced by WS-14.16 (W3 phase B) in a different shape: `UserService.delete_user(user_id, reason) -> DeleteUserResult(success, error, error_code, erasure_request_id)` (WS-14's ledger Ruling on IF-16 step 1). WS-08.8 maps it as R-14-9 asks (202, 404 `USER_NOT_FOUND`, 503 `ERASURE_INCOMPLETE`). | `services/user-service/src/domain/service.py` | WS-08.8 |
| R-9 | WS-14 | Open (no WS-14 task yet). W3 phase B (MVP-326 consent half): `record_consent` stores `ip_address=client_ip(request)` (`solace_security.rate_limit`, env-resolved hops), with a new test `services/user-service/tests/test_mvp_326_consent_client_ip.py` asserting that `POST /api/v1/users/me/consent` sent with `X-Forwarded-For: 198.51.100.7` under `TRUSTED_PROXY_HOPS=1` returns 201 with `ip_address == "198.51.100.7"`. | `services/user-service/src/api.py` | MVP-326 closure (WS-08.11 covers the session half) |
| R-10 | WS-14 | Partly produced. WS-14.16 (W3) registers erasure store `oauth_accounts` when `state.oauth_account_repository` is present, which WS-08.12 sets (R-14-10). Still open (no WS-14 task yet), W4 phase B: stores `identity_mfa` and `identity_reset_tokens` with deleters calling `MFARepository.delete_for_user` (WS-08.17) and `PasswordResetTokenStore.delete_for_user` (WS-08.18), read from `app.state.service.mfa_repository` and `.password_reset_store`. The users row is soft-deleted, so ON DELETE CASCADE never fires. | `services/user-service/src/domain/erasure.py`, `services/user-service/src/main.py` | WS-08.12 (via WS-14.16); WS-08.17, WS-08.18 erasure completeness (open) |
| R-11 | WS-14 | Produced by WS-14.21 (W4 phase B), which adds the IF-04 `require_mfa_for_roles("admin")` (WS-07.11) to the admin and provisioning routes. Provisioned accounts reach `mfa_enrollment_required` at first login through WS-08.16. | `services/user-service/src/api.py` | MVP-344 closure at the W4 barrier |
| R-12 | WS-03 | Partly produced. WS-03.1 (W0) writes the IF-02 OAuth rows (`GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `GOOGLE_OAUTH_REDIRECT_URI`, `GOOGLE_OAUTH_ISSUER`, `GOOGLE_OAUTH_AUTHORIZE_URL`, `GOOGLE_OAUTH_TOKEN_URL`, `GOOGLE_OAUTH_JWKS_URL`, `OAUTH_ALLOWED_REDIRECT_ORIGINS`) for user-service, which WS-08.10 depends on. Still open, for the next WS-03 phase-A env sync: `GOOGLE_OAUTH_CLIENT_SECRET` from `/run/secrets` (IF-22); `GOOGLE_OAUTH_REDIRECT_URI` and `OAUTH_ALLOWED_REDIRECT_ORIGINS` marked required in strict when the client id is set; `docs/STAGING-BRINGUP.md` documents them (the documentation half of MVP-104, WS-03.15). | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml`, `docs/STAGING-BRINGUP.md` | WS-08.10 |
| R-13 | WS-11 | W3 (API-HANDOFF rewrite): replace the stale Google OAuth "NOT IMPLEMENTED" and token-in-body text with a link to `docs/api/auth.md`. | `docs/API-HANDOFF.md` | WS-08.10 (MVP-104 doc closure) |
| R-14 | WS-05 | W2: `make_user` treats the 202 `{"status": "accepted"}` registration as success and takes the user id from the login token's `sub`. W4: under IF-17, a clinician or admin without a factor never receives an access token (login returns `mfa_enrollment_required`), so `test_clinician_or_admin_without_mfa_is_denied_phi_admin_and_ack` mints its `mfa=false` clinician token with the per-run RS256 key (as `mint_hs256_user_token` does), or enrolls and asserts the step-up instead. | `src/solace_testing/e2e.py`, `tests/e2e/test_w4_enterprise.py` | WS-08.7, WS-08.16 |
| R-15 | WS-13 | W1 (planned in WS-13.7, which already rewrites the template engines): keep and lock the `password_reset` template that WS-08.18's `PasswordResetMailer` posts through IF-09 `/email`: `TemplateType.PASSWORD_RESET = "password_reset"`, subject `Reset your Solace-AI password`, required variables exactly `display_name`, `reset_link`, `expiry_minutes`, and a plain-text body that renders `reset_link` and the display name verbatim (not HTML-escaped) after the text/HTML engine split, with the HTML body still escaped. This change used to be described in WS-08.18's Files line as an existing-template reference. | `services/notification-service/src/domain/templates.py` | WS-08.18 |

## New interfaces

Beyond the skeleton's IF-17 (all in user-service unless noted):
- Access-token claim `loc` (D-29, WS-08.21): the user's profile locale, canonical `ll` or `ll-CC` (the `normalize_locale` grammar, identical to `solace_clinical.canonical_locale`), `en-US` when the profile has none, `und` when the stored value cannot be normalized; on every access-token issuance path and never on refresh or step-up tokens. It is the one locale contract for in-band crisis resources: consumers read it with no lookup, a missing claim means en-US, and `solace_clinical.resolve_locale` decides between the registry entry and the generic fallback.
- Token types `reauth`, `mfa` and `mfa_enroll` (`TokenType.REAUTH`, `.MFA`, `.MFA_ENROLL`), with TTLs `REAUTH_TOKEN_TTL_SECONDS = 300`, `MFA_TOKEN_TTL_SECONDS = 300` and `MFA_ENROLL_TOKEN_TTL_SECONDS = 600`. The reauth claim `purpose` takes the values `account_delete` and `sensitive`, and `POST /auth/reauth` accepts the optional field `purpose` (default `sensitive`). `DELETE /users/me` needs purpose `account_delete`, which is the explicit confirmation step D-24 asks for.
- `DELETE /api/v1/users/me` returns 202 `{"erasure_request_id"}`, 404 `USER_NOT_FOUND` or 503 `ERASURE_INCOMPLETE` (R-14-9).
- Login challenge bodies `{mfa_required, mfa_token, expires_in}` and `{mfa_enrollment_required, enrollment_token, expires_in}`. `POST /auth/mfa/enroll/confirm` also returns a token pair when it is called with an enrollment token.
- `POST /api/v1/auth/oauth/google/link {link_code, password}`. Callback redirect parameters are `code`, or `error` in {`account_link_required` (with `link_code`), `email_not_verified`, `account_unavailable`, `oauth_failed`}. The cookie is `solace_oauth_state`, and the OAuth code TTL is `OAUTH_CODE_TTL_SECONDS = 60`.
- `POST /auth/password/reset` returns 200 `{"status": "password_reset"}`, and `POST /auth/password/forgot` returns 202 `{"status": "accepted"}`. `POST /auth/verify-email` takes a JSON body `{token}`; the legacy query form `?user_id=&token=` is kept.
- Error codes: `INVALID_CREDENTIALS`, `EMAIL_NOT_VERIFIED`, `INVALID_VERIFICATION_TOKEN`, `INVALID_TOKEN`, `INVALID_TOKEN_TYPE`, `SESSION_EXPIRED`, `REFRESH_REUSED`, `ACCOUNT_UNAVAILABLE`, `IDENTITY_BACKEND_UNAVAILABLE`, `REGISTRATION_UNAVAILABLE`, `REAUTH_REQUIRED`, `REAUTH_PURPOSE_MISMATCH`, `SESSION_REVOKED`, `MFA_NOT_ENROLLED`, `REDIRECT_URI_NOT_ALLOWED`, `OAUTH_STATE_MISMATCH`, `INVALID_CODE`, `MFA_INVALID`, `MFA_TOKEN_INVALID`, `MFA_ALREADY_ENROLLED`, `MFA_REQUIRED` (same code as WS-07.11), `INVALID_RESET_TOKEN`, `INVALID_PASSWORD`, `USER_NOT_FOUND` and `ERASURE_INCOMPLETE` (DELETE /users/me, R-14-9).
- Redis keys: `refresh_rot:<jti>`, `stepup_fail:<sid>`, `mfa_attempts:<mfa jti>` and `oauth:google:code:<sha256>`, next to the existing `session:`, `user_sessions:`, `oauth:google:state:` and the WS-07 `token_blacklist:` keys.
- Python names:
  - `src.auth`: `SessionRevoker`, `RotationOutcome`, `RefreshRotationStore`, `InMemoryRefreshRotationStore`, `RedisRefreshRotationStore`, `REFRESH_GRACE_SECONDS`, `FailureCounter`, `InMemoryFailureCounter`, `RedisFailureCounter`, `StepUpGuard`, `audit_auth_event`, `consent_claim_for`, `locale_claim_for` (D-29, WS-08.21), `SessionManager.attach_revocation_blacklist` and `RedisSessionManager.attach_revocation_blacklist`;
  - `src.identity_wiring`: `build_session_security`, `SessionSecurity`, `build_jwt_service`, `IdentityConfigError`;
  - `src.infrastructure.jwt_service`: `IdentityBackendUnavailable`, `TokenTypeMismatch`, `ReauthPurposeMismatch`;
  - `src.api_deps`: `require_mfa_for_roles(*roles)`;
  - `src.domain.entities`: `normalize_email`, `normalize_locale`, `User.lock_expired`, `User.can_refresh`, `User.set_password_hash`;
  - `src.infrastructure.oauth_google`: `JWKSCache`, `verify_id_token_async`;
  - `src.oauth`: `OAuthConfigError`, `PendingAuthorization`, `InMemoryOneTimeCodeStore`, `RedisOneTimeCodeStore`, `resolve_google_user`;
  - `src.infrastructure.oauth_account_repository`: `OAuthAccountRepository` and its implementations;
  - `src.infrastructure.mfa_repository`: `MFARepository` and its implementations;
  - `src.mfa`: `MFAService`, `TOTPVerifier`, `hash_recovery_code`, `generate_recovery_codes`, `MFAAlreadyEnrolled`;
  - `src.password_reset`: `PasswordResetService`, `PasswordResetMailer`, `PasswordResetTokenStore` and its implementations;
  - `src.api_auth`: `validate_password_strength`;
  - `src.infrastructure.password_service`: `PasswordService.hash_password_async`, `.verify_password_async` and `PasswordConfig.max_concurrent_hashes`.
- Audit: `resource_id = "emailhash:<hex>"` from `search_hash(email, purpose="auth_login_email")` for unknown-account login failures.
- Slot 013 tables `user_mfa_factors`, `mfa_recovery_codes` (RLS) and `password_reset_tokens` (non-RLS, holds no PHI). This adds `password_reset_tokens` to the IF-03 list of non-RLS tables, and needs a ledger Ruling naming WS-02.

## Self-review

Coverage (register id to tasks):

| Register id | Severity | Tasks |
|---|---|---|
| MVP-344 | blocker | WS-08.15 (schema), WS-08.16 (step-up, mandatory for clinician/admin), WS-08.17 (enrollment, recovery codes, Postgres store); with WS-07.11 and R-11 for enforcement elsewhere |
| MVP-347 | blocker | WS-08.14 (MFA, reset and enrollment events added in WS-08.16/.17/.18) |
| MVP-099 | high | WS-08.15 (table), WS-08.18 (routes, token store, `User.set_password_hash`, `PasswordResetMailer`) |
| MVP-102 | high | WS-08.12 |
| MVP-104 | high | WS-08.10 (code handoff, strict settings, doc); R-12, R-13 for operator docs |
| MVP-210 | medium | WS-08.3; the password-change route path closes in WS-08.20, which stops calling `UserService.change_password` (R-5 is optional) |
| MVP-212 | medium | WS-08.4 |
| MVP-213 | medium | WS-08.2 |
| MVP-214 | medium | WS-08.5 |
| MVP-215 (absorbs MVP-227) | medium | WS-08.1 |
| MVP-217 | medium | WS-08.7 (the oracle); the timing half needs open request R-2, and the owner notice needs open requests R-3 and R-4 |
| MVP-219 | medium | WS-08.1 (entity order), WS-08.2 (route) |
| MVP-221 | medium | WS-08.7 |
| MVP-222 | medium | WS-08.9 |
| MVP-223 | medium | WS-08.10 |
| MVP-224 | medium | WS-08.6 |
| MVP-226 | medium | WS-08.20 (route throttle from WS-07.7) |
| MVP-228 | medium | WS-08.5 |
| MVP-229 | medium | WS-08.8 |
| MVP-326 | medium | WS-08.11 (sessions); the consent-record half needs open request R-9 (WS-14's `api.py`, with the test named there) |
| MVP-330 | low | WS-08.5 |
| MVP-100 (WS-14, supporting) | high | WS-08.13 |
| MVP-209 (WS-07, supporting) | medium | WS-08.19 |

Interface consistency:
- WS-07 references are honoured with the ids it expects:
  - WS-08.4 writes `token_blacklist:sid:<sid>` through `TokenBlacklist.revoke_session`, re-exports the shared blacklist classes and makes identity routes fail closed;
  - WS-08.11 uses `client_ip(request)` with env-resolved hops;
  - WS-08.16 mints the `mfa` claim and relies on `require_mfa_for_roles`;
  - WS-08.19 relies on the WS-07.10 strict RS256 rules.
- The IF-17 claim names `sid`, `amr`, `mfa` and `cns` and the token types `reauth`, `mfa` and `mfa_enroll` match WS-07.2, which rejects the step-up types at every other service. The D-29 claim `loc` is an addition; WS-07 surfaces it as `AuthenticatedUser.locale` through the request in WS-11's plan.
- IF-16 is consumed as WS-14.16 delivers it: `DELETE /api/v1/users/me` requires `X-Reauth-Token` and calls `delete_user(user_id, reason="user_request")`, which returns `DeleteUserResult` carrying `erasure_request_id` (WS-14's ledger Ruling on IF-16 step 1).
- IF-14 calls use only keys in WS-09.11's `ALLOWED_DETAIL_KEYS` (`reason_code`, `sid`, `amr`, `mfa`, `erasure_request_id`) and the IF-14 event names.
- IF-13 is used for TOTP secrets (`field="user_mfa_factors.secret"`, owner = user) and for the login email hash.
- IF-03: RLS tables are accessed only through `ConnectionPoolManager.acquire()`/`transaction()` inside `acting_as_user`, and the slot edits stay inside `upgrade()`/`downgrade()`.
- IF-02 names: `GOOGLE_OAUTH_*`, `OAUTH_ALLOWED_REDIRECT_ORIGINS`, `TRUSTED_PROXY_HOPS`, `REDIS_URL` and `AUTH_*`. No new environment variable is introduced; every constant is fixed by IF-17.
- Phase rules: WS-08.15 is the only phase-A task (W4 schema lane) and depends only on earlier-wave or phase-A work. Every other task is phase B and depends on phase-A tasks of the same or earlier waves, or on WS-14 phase-B work of earlier waves. The same-wave phase-B dependencies on WS-14 (WS-14.16 for WS-08.8 and WS-14.17 for WS-08.13, both W3 phase B) are ordered by the controller ahead of the dependent WS-08 task and recorded in the ledger; no phase-A task depends on phase-B work. The WS-08 tests themselves use spies with the WS-14 signatures, so they never block on lane timing.
- Shared test directory: `services/user-service/tests` is stewarded by WS-14, and its protocol lets the owner of the code under test edit an existing file and add `test_mvp_<id>_<slug>.py` files. The existing files WS-08 edits test only WS-08-owned code after the W0 split: `test_refresh_rotation.py` (refresh route and `auth.py`), `test_oauth_endpoints.py` (`oauth.py`), `test_oauth_google.py` (`oauth_google.py`), `test_auth_timing_health.py` (the login timing path), `test_rs256_issuer_migration.py` (`jwt_service.py`) and, in `test_api.py`, only the login, registration and password-change cases that exercise `api_auth.py` routes. WS-14 edits those files only in W0 (WS-14.3, import re-pointing) and W1 (WS-14.8, the on-call routing test), so no wave has two writers. `identity_testkit.py` is a new helper created by WS-08.2 and edited only by WS-08 tasks. WS-08 never edits `services/user-service/tests/conftest.py` or `fixtures.py`; WS-08.4's real-infra conftest lives in `tests/realinfra/identity-auth-mfa/`.
- `docs/api/auth.md` is created by WS-08.7 and edited only by WS-08 tasks. WS-11.15 edits `docs/API-HANDOFF.md` and only links to `docs/api/auth.md` (WS-11 plan, WS-11.15 Files), so there is no second writer.

Review focus (numbered 1-5 as listed in the skeleton):
- 1 (crisis false negatives) and 2 (alert-path failures) fall outside this workstream's code.
- 3 (foreign-but-valid identifiers and context-less RLS work): `test_mvp_344_mfa_repository_real_pg.py` (WS-08.17) proves a context-less MFA lookup still finds the factor, another user's context sees nothing, and the GUC does not leak. If the lookup silently read nothing, MFA would be skipped. WS-08.8 (`test_mvp_229_reauth_token_is_bound_to_user_session_and_purpose`) proves that another user's or another session's reauth token is refused, and WS-08.12 proves a Google subject cannot move to another user.
- 4 (PHI in unplanned sinks): `test_mvp_347_auth_audit_events.py` (WS-08.14) asserts that no email or password reaches an audit call. WS-08.9 asserts the token-exchange body is never logged. WS-08.10 asserts that no token appears in the Location header or body. WS-08.2 asserts that internal error text never reaches a response.
- 5 (strict boots with partial configuration): `test_mvp_209_user_token_rs256_strict.py` (WS-08.19) covers each of the three RS256 variables missing and old-kid rotation. WS-08.4 and WS-08.6 cover strict mode without Redis, and WS-08.10 covers strict mode without a redirect URI or allow-list. WS-08.13's revocation test keeps a consent revocation from silently leaving tokens valid at other services.

Open risks:
- This revision aligns with WS-14's written plan: WS-14.3 (split), WS-14.6 (`verify_email(user_id | None, token)`), WS-14.16 (`DeleteUserResult.erasure_request_id`) and WS-14.17 (`get_consent_claim`). R-2 (timing half), R-3, R-4, R-5, R-9 and the W4 half of R-10 have no producing task in WS-14 or WS-13 yet. Until they land: the register timing differs slightly, an existing owner gets no notice email, consent records keep the proxy IP, and the MFA and reset-token rows survive an erasure. Each is recorded in the ledger as open at the barrier that needs it.
- WS-08 has no W1 lane. Until WS-08.2 lands in W2, verification works only through WS-14's W1 link fix with the legacy query form, and the strict-profile W1 E2E depends on that fix.
- A victim who clicks a verification link that an unsolicited squatter registration sent to them still activates the squatter's password. WS-08.18 (reset proves the inbox and replaces the password) is the recovery path, and the 202 registration no longer resends links on re-registration.
- The one-time OAuth code and the rotation and step-up state live in Redis. A Redis outage makes identity routes return 503 (D-26, fail closed), so sign-in is unavailable during the outage.
- `tests/realinfra/identity-auth-mfa` must run in its own pytest process, because user-service's package is named `src` and so is the shared root package. The slug `conftest.py` refuses a shadowed import loudly.
- An access token minted before a sid revocation stays valid at other services for at most its remaining lifetime when their revocation check fails open during a Redis outage (WS-07.5 policy).
- `password_reset_tokens` is a new non-RLS table and needs a ledger Ruling extending the IF-03 non-RLS list (WS-02 reviews it in WS-08.15).

REVIEW_FOCUS: 3=WS-08.17,WS-08.8,WS-08.12; 4=WS-08.14,WS-08.2,WS-08.10; 5=WS-08.19,WS-08.4,WS-08.10

## Review responses

- Error: WS-08.4 edits `services/user-service/tests`, which WS-14 owns: fixed. The checker read backticked paths inside the description of the one file WS-08.4 creates, `tests/realinfra/identity-auth-mfa/conftest.py`; WS-08.4 never edited the user-service suite. The line now names only the created file and describes the `sys.path` entries in words.
- Error: Wave 2 `services/user-service/tests/conftest.py` edited by WS-08.4 and WS-14.10: fixed by the same rewording; no WS-08 task edits that conftest, so WS-14.10 is its only writer.
- Error: Wave 2 `docs/api/auth.md` edited by WS-08.7 and WS-11.15: rejected: WS-11.15 modifies `docs/API-HANDOFF.md` and creates `docs/api/orchestrator.md`, and only links to `docs/api/auth.md` (WS-11 plan line 1424, "auth links to `docs/api/auth.md`"); the skeleton's `docs/API-HANDOFF.md` protocol says the auth contract "lives in docs/api/auth.md (WS-08) and is linked, not copied".
- Error: plan a task for WS-14's request R-14-8 (verify-email accepts `token` alone, passing `user_id` when present): fixed. It joins WS-08.2: `VerifyEmailRequest{token, user_id: UUID | None = None}` plus the legacy query form, calling WS-14.6's `verify_email(user_id | None, token)`; the spy test records `(None, token)` for body calls and `(uuid, token)` for the body-with-id and legacy calls; Produces names R-14-8; Depends on names WS-14.6.
- Note (same kind, not in the findings file): WS-14's R-14-9 joins WS-08.8 (DELETE maps `DeleteUserResult` to 202, 404 `USER_NOT_FOUND` or 503 `ERASURE_INCOMPLETE`, sessions revoked only on success, `reason="user_request"`, new failure-mapping test) and WS-08.13 (`cns` from `get_consent_claim`; `revoke_all_user_sessions` writes every sid to the shared blacklist, new test); R-14-10 joins WS-08.12 (`app.state.service.oauth_account_repository`, `email` written as NULL): fixed.
- Warning: WS-08.1 names `WS-14:W0-split` in Depends on: fixed (WS-14.3).
- Warning: WS-08.2 names `WS-14:W0-split` and `WS-14:verify_email_by_token` in Depends on: fixed (WS-14.3 and WS-14.6; the by-token method is withdrawn because WS-14.6's `verify_email(None, token)` consumes the token first).
- Warning: WS-08.7 names `WS-14:send_account_exists_email`, `WS-13:account_exists-template` and `WS-14:verification-send-off-request-path`: fixed. No WS-14 or WS-13 task produces them, so Depends on is WS-08.6 only and they stay open requests R-3, R-4 and R-2 (timing half); the owner notice is best-effort (exceptions caught, logged by type only), so the 202 never depends on them.
- Warning: WS-08.8 names `WS-14:delete_user-returns-erasure-id`: fixed (WS-14.16, consumed in the `DeleteUserResult` shape it returns).
- Warning: WS-08.10 names `WS-03:oauth-env-sync`: fixed (WS-03.1 writes the IF-02 OAuth rows in W0; the strict-required flags and the STAGING-BRINGUP text stay in R-12).
- Warning: WS-08.11 names `WS-14:consent-client-ip`: fixed. No WS-14 task produces it, so R-9 stays open; the consent-record test, which exercises WS-14's `api.py`, moved into R-9, and WS-08.11 now covers the session half only.
- Warning: WS-08.12 names `WS-14:erasure-registers-identity-stores`: fixed. WS-14.16 registers `oauth_accounts` from `state.oauth_account_repository`, an output of WS-08.12 rather than an input, so it is not a dependency; the W4 MFA and reset-token stores stay open in R-10.
- Warning: WS-08.13 names `WS-14:consent_claim`: fixed (WS-14.17, method `get_consent_claim`).
- Warning: WS-08.16 names `WS-01:pyotp-lock (D-16)`: fixed (WS-01.2, whose D-16 lock adds pyotp).
- Warning: WS-08.18 names `WS-14:set_password_hash` and `WS-14:send_password_reset_email`: fixed. No WS-14 task produces them, so WS-08.18 persists through its own `User.set_password_hash` (entities.py is WS-08's) and WS-14's existing `UserRepository.update`, and mails through its own `PasswordResetMailer` (IF-09 `/email`, `service_auth_headers`, the existing `password_reset` template, new mailer test); Depends on WS-14.6, WS-07.3, WS-13.12; R-6 withdrawn.
- Warning: WS-08.20 names `WS-14:set_password_hash`: fixed (depends on WS-08.18 for `User.set_password_hash` and persists through `UserRepository.update`).
- Warning: WS-08.2 edits `services/user-service/tests/test_api.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; the cases edited here exercise the `api_auth.py` routes WS-08 owns after the W0 split (login, registration); WS-14.5 itself attributes the `detail` assertion at `test_api.py:189` to the register route in `api_auth.py`. WS-14 edits it only in W0 (WS-14.3 import re-pointing) and W1 (WS-14.8, the on-call routing case), so no wave has two writers.
- Warning: WS-08.2 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict (WS-08.2 creates it).
- Warning: WS-08.3 edits `services/user-service/tests/test_auth_timing_health.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests the login dummy-verify and resend-verification throttle, both `/auth/*` routes in `api_auth.py` (WS-08). WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.3 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.4 edits `services/user-service/tests/conftest.py`: fixed (misparse removed, see the first error).
- Warning: WS-08.4 edits `services/user-service/tests/test_refresh_rotation.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests `refresh_token` and `jwt_service.py` rotation, both WS-08's. WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.4 edits `services/user-service`, which no workstream owns: fixed (same misparse; the path was a `sys.path` entry in a description).
- Warning: WS-08.4 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.6 edits `services/user-service/tests/test_refresh_rotation.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests `refresh_token` and `jwt_service.py` rotation, both WS-08's. WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.6 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.7 edits `services/user-service/tests/test_oauth_endpoints.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests `oauth.py` start, callback and registration flows, which WS-08 owns. WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.7 edits `services/user-service/tests/test_api.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; the cases edited here exercise the `api_auth.py` routes WS-08 owns after the W0 split (login, registration); WS-14.5 itself attributes the `detail` assertion at `test_api.py:189` to the register route in `api_auth.py`. WS-14 edits it only in W0 (WS-14.3 import re-pointing) and W1 (WS-14.8, the on-call routing case), so no wave has two writers.
- Warning: WS-08.8 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.9 edits `services/user-service/tests/test_oauth_google.py`: rejected: it tests `infrastructure/oauth_google.py`, which WS-08 owns, and the shared_files protocol lets the owner of the code under test edit an existing file; no WS-14 task edits it.
- Warning: WS-08.10 edits `services/user-service/tests/test_oauth_endpoints.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests `oauth.py` start, callback and registration flows, which WS-08 owns. WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.10 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.12 edits `services/user-service/tests/test_oauth_endpoints.py`: rejected: the shared_files protocol for `services/user-service/tests` lets the owner of the code under test edit an existing file; it tests `oauth.py` start, callback and registration flows, which WS-08 owns. WS-14 edits it only in W0 (WS-14.3 import re-pointing), so no wave has two writers.
- Warning: WS-08.12 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.14 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.16 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.18 edits `services/user-service/tests/identity_testkit.py`: rejected: `identity_testkit.py` is a new helper created by WS-08.2 for WS-08's own tests; no WS-14 task creates or edits it (the WS-14 plan never names it), so there is no existing-file conflict.
- Warning: WS-08.19 edits `services/user-service/tests/test_rs256_issuer_migration.py`: rejected: it tests `infrastructure/jwt_service.py` RS256 issuance and verification, which WS-08 owns, and the shared_files protocol lets the owner of the code under test edit an existing file; no WS-14 task edits it.
- Warning: WS-08.4 lists `services/user-service` under Create: fixed (misparse removed; only `tests/realinfra/identity-auth-mfa/conftest.py` is created).
- Warning: WS-08.4 lists `services/user-service/tests` under Create: fixed (same).
- Warning: WS-08.4 lists `services/user-service/tests/conftest.py` under Create: fixed (same).
- Reconciliation 8 (WS-08.18 named WS-13's `services/notification-service/src/domain/templates.py`): fixed. WS-08.18 only relied on the existing `password_reset` template, but its Files line cited the file. The path is gone from WS-08.18; the dependency is new request R-15 to WS-13, planned in WS-13.7 (W1), which already rewrites the template engines and now pins the template's name, subject, required variables and verbatim plain-text `reset_link` with `test_ws08_password_reset_template_keeps_its_contract` (RED today: the shared autoescape renders `O&#39;Brien` in the text body). WS-13.7's Produces line names WS-08.18, and WS-08.18 depends on WS-13.7. R-4 (`account_exists`) is unchanged and still open.
- decision 2026-10-19 D-29: new task WS-08.21 (W3 phase B, after WS-08.13) adds the access-token claim `loc`, read from the profile at every issuance through `locale_claim_for(user)`. The default is `en-US`; a value that cannot be normalized is issued as `und`, so no country is guessed. The claim is never on refresh or step-up tokens. RED tests cover login, refresh after a profile change, Google sign-in, the helper's failure cases, and a parity lock that `normalize_locale` (WS-08.7) and `solace_clinical.canonical_locale` (WS-12.2) accept exactly the same tags. This gives the orchestrator the user's locale at screen time with no database or network call. The file map, New interfaces (IF-17 addition) and Interface consistency are updated.
- decision 2026-10-19 D-29: WS-08.16 (`/auth/mfa/verify`) and WS-08.17 (`/auth/mfa/enroll/confirm` with an enrollment token) also mint `loc=locale_claim_for(user)`, each with one added assertion `access["loc"] == "en-US"`, and WS-08.16 now depends on WS-08.21, so every issuance path carries the claim. No new env variable and no new cross-workstream request: WS-07's `AuthenticatedUser.locale` is requested from WS-11's plan, and the profile field's validation is WS-14.22.