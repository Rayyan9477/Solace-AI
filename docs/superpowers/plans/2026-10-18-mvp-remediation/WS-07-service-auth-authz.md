# WS-07: Inter-service credentials, authorization, revocation and rate limiting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every service one inter-service credential (short-lived service JWTs that may act on behalf of a user), resolve service permissions on the receiving side, share token revocation through Redis, rate-limit on the real client with a safe failure policy, scope clinician access to assigned patients, and move user tokens to RS256-only and service tokens to per-service keys.

**Architecture:** All work lives in the shared libraries (`src/solace_security`, `src/solace_common`, `infrastructure/api_gateway/rate_limiting.py`), so every task is a phase-A platform task. The libraries expose the IF-04 functions (`service_auth_headers`, `forwarded_auth_headers`, `install_rate_limiting`, `build_token_blacklist`, `can_access_subject`, `require_mfa_for_roles`); WS-04 wires them into the runtime bootstrap and the service lanes adopt them at their routes, so no `main.py` is edited here. Supporting edits in other workstreams' files are listed under "Cross-workstream requests".

**Covers:** 10 register entries (blocker 0, high 1, medium 6, low 3), plus supporting work for MVP-034 (WS-14), MVP-021 (WS-11) and MVP-344 (WS-08). **Waves:** 0, 1, 3, 4. **Depends on:** WS-01 (PyJWT >= 2.13, redis, cryptography and prometheus-client pins; lazy `infrastructure/api_gateway/__init__.py` so the limiter imports without Kong), WS-04 (IF-01 `is_strict_environment` and `ServiceRuntimeSpec` in W0; it wires this workstream's functions into `bootstrap.py` in W1 phase A2), WS-05 (IF-19 `redis_url` and `pg_migrated` fixtures for the real-infra tests), WS-02 (the migrated chain the subject-scoping checker reads), WS-13 (`crisis_alerts` columns from slot 015 for the D-21 72-hour acknowledgement rule), WS-08 (the `sid`, `amr`, `mfa` and `cns` claims it mints). **Owned paths:** `src/solace_security/auth.py`, `src/solace_security/middleware.py`, `src/solace_security/service_auth.py`, `src/solace_security/keys.py`, `src/solace_security/rate_limit.py`, `src/solace_security/authorization.py`, `src/solace_security/__init__.py`, `src/solace_security/py.typed`, `src/solace_common`, `infrastructure/api_gateway/rate_limiting.py`, `scripts/generate_jwt_keys.py`; new test files under `tests/solace_security`, `tests/solace_common`, `tests/security`, `infrastructure/api_gateway/tests` and `tests/realinfra/service-auth-authz/`. **Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

Conventions for every task below:
- Commands use absolute paths, so they work from any directory. One pytest process per suite.
- Real-infra tests carry `pytestmark = [pytest.mark.realinfra, pytest.mark.integration]`, use only the IF-19 fixtures from `tests/realinfra/conftest.py` (WS-05), and run with `$env:SOLACE_REQUIRE_REALINFRA="1"` so a missing container is a failure, not a skip.
- `src/solace_security/__init__.py` gets no new eager re-exports; consumers import new names by full module path.
- Never log a token, a password, a Redis URL or an email address. Test secrets below are throwaway literals that exist only in tests.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `src/solace_common/request_context.py` | Modify | `acting_as_user()` for request-less RLS work | WS-07.1 |
| `src/solace_security/auth.py` | Modify | `TokenPayload` identity claims, malformed-claim mapping, `extra_metadata` on service tokens, session revocation API and fail-open revocation check, strict RS256-only user tokens and per-service service-token keys | WS-07.2, WS-07.3, WS-07.5, WS-07.10 |
| `src/solace_security/middleware.py` | Modify | `AuthenticatedUser`/`AuthenticatedService` fields, obo acceptance, receiver-side service permissions, `require_service_permission`, `find_unguarded_service_routes`, `build_token_blacklist`/`configure_token_blacklist`, subject scoping, `require_mfa_for_roles` | WS-07.2, WS-07.4, WS-07.5, WS-07.8, WS-07.11 |
| `src/solace_security/service_auth.py` | Modify | Identities, permissions and grants, obo tokens, `service_auth_headers`, `forwarded_auth_headers`, receiver-side `validate_service_token`, per-service signing | WS-07.3, WS-07.10 |
| `src/solace_security/authorization.py` | Modify | `SubjectAccessChecker` protocol and `PostgresSubjectAccessChecker` | WS-07.8 |
| `src/solace_security/keys.py` | Modify | Strict-environment keyset requirement, `ServiceKeySettings`, `ServiceKeyring` | WS-07.10 |
| `src/solace_security/rate_limit.py` | Modify | `resolve_trusted_proxy_hops`, `classify_route`, `build_rate_limiter`, `install_rate_limiting`, D-26 backend failure policy and metric | WS-07.7 |
| `infrastructure/api_gateway/rate_limiting.py` | Modify | Bounded, self-sweeping `RateLimitStore`; strict-environment check in `create_solace_rate_limiter` | WS-07.6, WS-07.7 |
| `scripts/generate_jwt_keys.py` | Modify | Writes user and per-service keysets to files, prints names only | WS-07.10 |
| `tests/solace_common/test_mvp_034_acting_as_user.py` | Create | acting_as_user tests | WS-07.1 |
| `tests/solace_security/test_mvp_317_malformed_claims_and_identity_claims.py` | Create | 401 mapping and IF-17 claim surfacing | WS-07.2 |
| `tests/solace_security/test_mvp_314_service_token_model.py` | Create | Permission matrix, obo tokens, header helpers | WS-07.3 |
| `tests/integration/test_service_integration.py` | Modify | Only `TestPermissionMatrix::test_user_service_has_no_permissions` (user-service now holds `service:write:notification`) | WS-07.3 |
| `tests/solace_security/test_mvp_314_obo_and_receiver_permissions.py` | Create | obo acceptance, receiver-side permissions, route scan | WS-07.4 |
| `tests/solace_security/test_mvp_186_shared_revocation.py` | Create | Strict Redis requirement, URL never logged, session revocation, fail-open | WS-07.5 |
| `tests/realinfra/service-auth-authz/test_mvp_186_cross_service_revocation.py` | Create | Revocation propagates between processes over real Redis; pins the key format | WS-07.5 |
| `infrastructure/api_gateway/tests/test_mvp_311_rate_limit_store_bounds.py` | Create | Store sweep, bound and hot-key survival | WS-07.6 |
| `tests/solace_security/test_mvp_101_install_rate_limiting.py` | Create | Real-client keying, auth-route policies, D-26 policy, strict Redis requirement | WS-07.7 |
| `tests/realinfra/service-auth-authz/test_mvp_101_shared_redis_limits.py` | Create | Two instances share one Redis bucket | WS-07.7 |
| `tests/solace_security/test_mvp_120_subject_scoping.py` | Create | can_access_subject, authorize_subject, set_subject_context | WS-07.8 |
| `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py` | Create | Checker SQL against the migrated schema as solace_app | WS-07.8 |
| `tests/security/test_mvp_314_service_routes_guarded.py` | Create | Acceptance: no service-token route in the nine apps lacks a permission | WS-07.9 |
| `tests/solace_security/test_mvp_209_rs256_and_service_keys.py` | Create | RS256-only in strict, per-service keys, key script output | WS-07.10 |
| `tests/solace_security/test_mvp_344_require_mfa_for_roles.py` | Create | MFA gate for CLINICIAN/ADMIN and in subject scoping | WS-07.11 |
| `tests/solace_security/test_mvp_120_subject_scoping.py` | Modify | Principals built with `mfa=True` once MFA gates clinician scoping | WS-07.11 |

## Tasks

### Task WS-07.1: acting_as_user and the correlation-id context for request-less work and outbound hops
**Covers:** MVP-034, MVP-356
**Severity:** blocker
**Wave:** 0
**Depends on:** none
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-02 (default assumed: every service connects as NOBYPASSRLS solace_app from W0, so work without a request must set the RLS user explicitly)
**Real infra:** no
**Files:**
- Modify: `src/solace_common/request_context.py` (add `acting_as_user`, `set_correlation_ids`, `get_correlation_headers` and `clear_correlation_ids`; `set_current_user_id`, `get_current_user_id` and `reset_current_user_id` keep their semantics)
- Test: `tests/solace_common/test_mvp_034_acting_as_user.py`
- Test: `tests/solace_common/test_mvp_356_correlation_context.py`
**Interfaces:**
- Consumes: `set_current_user_id`, `reset_current_user_id`, `get_current_user_id` (same module).
- Produces:
  - `acting_as_user(user_id: str | UUID | None) -> ContextManager[str | None]` (IF-03/IF-04). It sets the contextvar to the canonical lower-case UUID string (or None), yields that value, and restores the previous value on exit, including on exception. A value that is not a UUID raises `ValueError` before anything is set. Supporting piece of MVP-034: WS-14's registration writes `user_preferences` inside it in W0 phase B; WS-06 consumers, WS-10, WS-12, WS-15, WS-17, WS-14's erasure helper and CLIs use it later. WS-14 closes MVP-034.
  - `set_correlation_ids(*, request_id: str, correlation_id: str, turn_id: str) -> None`, `get_correlation_headers() -> dict[str, str]` (keys `X-Request-ID`, `X-Correlation-ID`, `X-Turn-ID`, only those set and non-empty) and `clear_correlation_ids() -> None`, on contextvars, exactly as WS-04.3 requested: its request-context middleware sets them per request, and WS-07.3's outbound header helpers read them. Supporting piece of MVP-356 (WS-11 closes it).

- [ ] **Step 1: Write the failing tests** in `tests/solace_common/test_mvp_034_acting_as_user.py::test_acting_as_user_sets_then_restores_previous_value` (and the other tests in the file) and `tests/solace_common/test_mvp_356_correlation_context.py`
```python
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
```
```python
"""tests/solace_common/test_mvp_356_correlation_context.py
MVP-356 (supporting; WS-11 closes it): correlation ids live in one request-scoped context that WS-04.3 sets."""
from __future__ import annotations

import asyncio

import pytest

from solace_common.request_context import clear_correlation_ids, get_correlation_headers, set_correlation_ids


def test_headers_reflect_only_the_ids_that_are_set():
    clear_correlation_ids()
    assert get_correlation_headers() == {}
    set_correlation_ids(request_id="r-1", correlation_id="c-1", turn_id="t-1")
    assert get_correlation_headers() == {"X-Request-ID": "r-1", "X-Correlation-ID": "c-1", "X-Turn-ID": "t-1"}
    clear_correlation_ids()
    assert get_correlation_headers() == {}


def test_empty_values_are_omitted():
    clear_correlation_ids()
    set_correlation_ids(request_id="r-2", correlation_id="", turn_id="")
    assert get_correlation_headers() == {"X-Request-ID": "r-2"}
    clear_correlation_ids()


@pytest.mark.asyncio
async def test_ids_are_isolated_between_concurrent_requests():
    clear_correlation_ids()
    seen = {}

    async def request(label):
        set_correlation_ids(request_id=f"r-{label}", correlation_id=f"c-{label}", turn_id=f"t-{label}")
        await asyncio.sleep(0.01)
        seen[label] = get_correlation_headers()["X-Request-ID"]

    await asyncio.gather(asyncio.create_task(request("a")), asyncio.create_task(request("b")))
    assert seen == {"a": "r-a", "b": "r-b"}
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_common\test_mvp_034_acting_as_user.py -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_common\test_mvp_356_correlation_context.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'acting_as_user' from 'solace_common.request_context'` and `cannot import name 'clear_correlation_ids'`: there is no supported way to set the RLS user for registration, consumers or CLIs, and no shared place holds the ids an outbound hop must carry.
- [ ] **Step 3: Implement** in `src/solace_common/request_context.py`: `acting_as_user` as a `contextlib.contextmanager` (normalise with `str(UUID(str(user_id)))`, `ValueError` for anything else, `None` passes through; `token = set_current_user_id(value)`, yield, `reset_current_user_id(token)` in `finally`); three module-level `ContextVar[str | None]` for the correlation ids with the three helpers.
- [ ] **Step 4: Run them and confirm they pass**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suites. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_common -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.2: Malformed-claim tokens give 401, and the IF-17 identity claims reach AuthenticatedUser
**Covers:** MVP-317
**Severity:** low
**Wave:** 1
**Depends on:** none
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `src/solace_security/auth.py` (`TokenPayload`: `session_id` also reads the `sid` claim through `Field(default=None, validation_alias=AliasChoices("session_id", "sid"))`; new fields `amr: list[str] = Field(default_factory=list)`, `mfa: bool = False`, `cns: dict[str, Any] | None = None`. `JWTManager.decode_token_sync`: the `TokenPayload(**decoded)` call (line 563) is wrapped so `pydantic.ValidationError` returns `AuthenticationResult.fail("INVALID_TOKEN", "Malformed token claims")`)
- Modify: `src/solace_security/middleware.py` (`AuthenticatedUser`: new fields `amr: list[str] = []`, `mfa: bool = False`, `cns: dict | None = None`, `via_service: str | None = None`; `from_payload` maps `session_id`, `amr`, `mfa` and `cns`)
- Test: `tests/solace_security/test_mvp_317_malformed_claims_and_identity_claims.py`
**Interfaces:**
- Consumes: none.
- Produces: `AuthenticatedUser.session_id` (from `sid` or `session_id`), `.amr`, `.mfa`, `.cns`, `.via_service` (IF-04, IF-17). WS-11 reads `.cns` for the consent gate (W3), WS-07.5 reads `TokenPayload.session_id` for session revocation, WS-07.11 reads `.mfa`. A signed token whose `type` is not a `TokenType` value (WS-08's `reauth`, `mfa` and `mfa_enroll` tokens) or that lacks `exp` is rejected with `INVALID_TOKEN`.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_317_malformed_claims_and_identity_claims.py::test_malformed_but_signed_token_is_a_failure_result_not_an_exception`
```python
"""MVP-317: validly signed tokens with a non-enum type or no exp give 401, not 500.
Also surfaces the IF-17 identity claims (sid, amr, mfa, cns) that WS-08 mints from W2."""
from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager

SECRET = "test-secret-key-32-bytes-long!!!"


class _FakeURL:
    path = "/api/v1/x"


class _FakeRequest:
    url = _FakeURL()


def _manager() -> JWTManager:
    return JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())


def _sign(**claims) -> str:
    now = int(time.time())
    base = {"sub": "u-1", "jti": "j-1", "iat": now, "exp": now + 600, "iss": "solace-ai", "aud": "solace-ai-api"}
    base.update(claims)
    return jwt.encode({k: v for k, v in base.items() if v is not None}, SECRET, algorithm="HS256")


@pytest.mark.parametrize("claims", [{"type": "password_reset"}, {"type": "reauth"}, {"type": "access", "exp": None}])
def test_malformed_but_signed_token_is_a_failure_result_not_an_exception(claims):
    result = _manager().validate_access_token(_sign(**claims))
    assert result.success is False
    assert result.error_code == "INVALID_TOKEN"


@pytest.mark.asyncio
async def test_get_current_user_maps_malformed_claims_to_401(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=_sign(type="password_reset"))
    with pytest.raises(HTTPException) as exc:
        await mw.get_current_user(_FakeRequest(), creds)
    assert exc.value.status_code == 401
    assert exc.value.detail == "Malformed token claims"  # fixed text: no ValidationError text, no claim value
    assert "password_reset" not in str(exc.value.detail) and "validation error" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_identity_claims_are_surfaced(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    token = _sign(type="access", roles=["clinician"], sid="5b1d7c2e-0000-4000-8000-000000000001",
                  amr=["pwd", "otp"], mfa=True, cns={"ok": True, "version": "2026-10"})
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    user = await mw.get_current_user(_FakeRequest(), creds)
    assert user.session_id == "5b1d7c2e-0000-4000-8000-000000000001"
    assert user.amr == ["pwd", "otp"]
    assert user.mfa is True
    assert user.cns == {"ok": True, "version": "2026-10"}
    assert user.via_service is None


def test_service_token_round_trip_is_unchanged():
    manager = _manager()
    assert manager.decode_token_sync(manager.create_service_token("safety-service")).success is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_317_malformed_claims_and_identity_claims.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The parametrized test and the 401 test raise `pydantic_core._pydantic_core.ValidationError` (1 validation error for TokenPayload, `type` or `exp`), which is the 500 the register describes; its message embeds the offending claim value, which is why the 401 test also pins the fixed detail that `_handle_auth_failure` (middleware.py:207-227) copies from `result.error_message`; the claims test fails with `session_id is None` and `AttributeError: 'AuthenticatedUser' object has no attribute 'amr'`.
- [ ] **Step 3: Implement** the `TokenPayload` fields, the `ValidationError` mapping in `JWTManager.decode_token_sync` and the four `AuthenticatedUser` fields with the `from_payload` mapping. Import `AliasChoices` and `ValidationError` from `pydantic`. `model_dump()` keeps field names, so service tokens minted here keep `session_id`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\integration -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.3: Service-token permission model, on-behalf-of tokens and outbound header helpers
**Covers:** MVP-314, MVP-021
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-07.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: short-lived obo service JWTs, HS256 with AUTH_SECRET_KEY in W1-W3, per-service RS256 keys from W4); D-28 (default assumed: notification-service is the only pager, so safety-service loses `service:write:notification`)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/service_auth.py` (`ServiceIdentity.ANALYTICS = "analytics-service"`; `ServicePermission.ACT_FOR_USER = "service:act_for_user"`, `READ_ONCALL_ROSTER = "service:read:oncall"`, `WRITE_ANALYTICS = "service:write:analytics"`; `SERVICE_PERMISSIONS`: orchestrator gains `ACT_FOR_USER` and `WRITE_PERSONALITY`, safety loses `WRITE_NOTIFICATION`, user-service becomes `[WRITE_NOTIFICATION]`, notification gains `READ_ONCALL_ROSTER`, analytics is `[]`; `OBO_TOKEN_TTL_SECONDS = 300`; new `resolve_service_permissions`; `ServiceTokenManager.service_name` property; `create_service_token(..., *, on_behalf_of=None)`; `validate_service_token` checks permissions against the matrix, not the claims; new `correlation_headers`, `service_auth_headers`, `forwarded_auth_headers`)
- Modify: `src/solace_security/auth.py` (`JWTManager.create_service_token(service_name, permissions=None, expire_minutes=60, *, extra_metadata: dict[str, str] | None = None)` merges `extra_metadata` into `metadata` after `service_name`, which it may not override)
- Modify: `tests/integration/test_service_integration.py` (only `TestPermissionMatrix::test_user_service_has_no_permissions`: rename to `test_user_service_may_only_send_notifications` and assert `SERVICE_PERMISSIONS[ServiceIdentity.USER] == [ServicePermission.WRITE_NOTIFICATION]`)
- Test: `tests/solace_security/test_mvp_314_service_token_model.py`
**Interfaces:**
- Consumes: `AuthSettings`, `JWTManager` (auth.py); `solace_common.request_context.get_correlation_headers()` (WS-07.1, set per request by WS-04.3's middleware); `structlog.contextvars.get_contextvars()` keys `request_id`, `correlation_id` and `turn_id` (WS-10's `run_turn` re-binds `turn_id` per WebSocket message).
- Produces (IF-04):
  - `resolve_service_permissions(service_name: str) -> list[str]`: the values in `SERVICE_PERMISSIONS[ServiceIdentity(service_name)]`; raises `ValueError` for an unknown identity.
  - `ServiceTokenManager.create_service_token(service_name, target_service=None, additional_permissions=None, expire_minutes=None, *, on_behalf_of: str | None = None) -> ServiceCredentials`. With `on_behalf_of` (must parse as a UUID, else `ValueError`), the token carries `metadata.obo_sub`, lives at most `OBO_TOKEN_TTL_SECONDS` (300) and is never written to the cache. Claims stay `sub='service:<name>'`, `type='service'`, `iss='solace-ai'`, `aud='solace-ai-api'`.
  - `ServiceTokenManager.service_name -> str | None` (from `ServiceAuthSettings.service_name`).
  - `correlation_headers() -> dict[str, str]`: `get_correlation_headers()` overridden, key by key, by the structlog contextvars `request_id`, `correlation_id` and `turn_id` when bound (the more specific binding wins, so a WebSocket frame's turn id is the one sent).
  - `async def service_auth_headers(target_service: str, *, on_behalf_of: str | None = None) -> dict[str, str]`: `Authorization: Bearer <token>` from the process singleton plus `correlation_headers()`. Raises `RuntimeError("service auth is not initialized ...")` when `initialize_service_auth()` has not run with a `service_name`, and lets token-minting errors propagate (WS-10's safety client maps a raise to its error path). Plain tokens come from `get_or_create_token`; obo tokens are minted per call.
  - `forwarded_auth_headers(request: Request) -> dict[str, str]`: the inbound `Authorization` header unchanged plus `correlation_headers()`; used for same-user hops such as therapy to memory (WS-15).
  - Grants consumed by WS-11 (`ACT_FOR_USER`), WS-13 (`READ_ONCALL_ROSTER`), WS-14 (`WRITE_NOTIFICATION`) and WS-18 (`WRITE_PERSONALITY` for `/detect`, `WRITE_ANALYTICS` held by no service).

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_314_service_token_model.py::test_permission_matrix_matches_if04`
```python
"""MVP-314 / MVP-021 (supporting): one inter-service credential with receiver-side permissions (IF-04)."""
from __future__ import annotations

from uuid import uuid4

import jwt
import pytest
import structlog
from starlette.requests import Request

from solace_security import service_auth as sa
from solace_security.auth import AuthSettings
from solace_security.service_auth import (
    SERVICE_PERMISSIONS, ServiceAuthSettings, ServiceIdentity, ServicePermission, ServiceTokenManager,
    forwarded_auth_headers, initialize_service_auth, resolve_service_permissions, service_auth_headers,
)

SECRET = "test-secret-key-32-bytes-long!!!"


def _manager(name: str) -> ServiceTokenManager:
    return ServiceTokenManager(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name=name))


def _claims(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=["HS256"], audience="solace-ai-api", issuer="solace-ai")


def _holders(permission) -> set:
    return {identity for identity, perms in SERVICE_PERMISSIONS.items() if permission in perms}


def test_permission_matrix_matches_if04():
    assert ServicePermission.ACT_FOR_USER.value == "service:act_for_user"
    assert ServicePermission.READ_ONCALL_ROSTER.value == "service:read:oncall"
    assert _holders(ServicePermission.ACT_FOR_USER) == {ServiceIdentity.ORCHESTRATOR}
    assert _holders(ServicePermission.READ_ONCALL_ROSTER) == {ServiceIdentity.NOTIFICATION}
    assert _holders(ServicePermission.WRITE_NOTIFICATION) == {ServiceIdentity.USER}
    assert _holders(ServicePermission.WRITE_ANALYTICS) == set()
    assert ServicePermission.READ_SAFETY in SERVICE_PERMISSIONS[ServiceIdentity.ORCHESTRATOR]
    assert ServicePermission.WRITE_PERSONALITY in SERVICE_PERMISSIONS[ServiceIdentity.ORCHESTRATOR]
    assert SERVICE_PERMISSIONS[ServiceIdentity("analytics-service")] == []


def test_resolve_service_permissions_comes_from_the_matrix_and_rejects_unknown():
    assert "service:escalate" not in resolve_service_permissions("memory-service")
    assert "service:escalate" in resolve_service_permissions("safety-service")
    with pytest.raises(ValueError):
        resolve_service_permissions("config-service")


def test_obo_token_claims_and_ttl():
    uid = uuid4()
    creds = _manager("orchestrator-service").create_service_token(
        "orchestrator-service", "therapy-service", on_behalf_of=str(uid))
    claims = _claims(creds.token)
    assert claims["sub"] == "service:orchestrator-service"
    assert claims["type"] == "service"
    assert claims["metadata"]["service_name"] == "orchestrator-service"
    assert claims["metadata"]["obo_sub"] == str(uid)
    assert claims["exp"] - claims["iat"] <= sa.OBO_TOKEN_TTL_SECONDS == 300


def test_obo_tokens_are_never_cached():
    manager = _manager("orchestrator-service")
    a = manager.create_service_token("orchestrator-service", "memory-service", on_behalf_of=str(uuid4()))
    b = manager.create_service_token("orchestrator-service", "memory-service", on_behalf_of=str(uuid4()))
    assert a.token != b.token
    assert manager.get_cached_token("orchestrator-service", "memory-service") is None


@pytest.mark.parametrize("bad", ["not-a-uuid", "", "123"])
def test_obo_requires_a_uuid(bad):
    with pytest.raises(ValueError):
        _manager("orchestrator-service").create_service_token("orchestrator-service", on_behalf_of=bad)


def test_validate_service_token_ignores_claimed_permissions():
    manager = _manager("orchestrator-service")
    forged = manager._get_jwt_manager().create_service_token("memory-service", permissions=["service:escalate"])
    assert manager.validate_service_token(forged, required_permissions=["service:escalate"]).authenticated is False


@pytest.mark.asyncio
async def test_service_auth_headers_carry_identity_and_correlation(monkeypatch):
    monkeypatch.setattr(sa, "_service_token_manager", None)
    initialize_service_auth(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name="notification-service"))
    structlog.contextvars.bind_contextvars(request_id="r-1", correlation_id="c-1", turn_id="t-1")
    try:
        headers = await service_auth_headers("user-service")
    finally:
        structlog.contextvars.clear_contextvars()
        monkeypatch.setattr(sa, "_service_token_manager", None)
    assert headers["Authorization"].startswith("Bearer ")
    assert _claims(headers["Authorization"][7:])["sub"] == "service:notification-service"
    assert (headers["X-Request-ID"], headers["X-Correlation-ID"], headers["X-Turn-ID"]) == ("r-1", "c-1", "t-1")


@pytest.mark.asyncio
async def test_request_context_ids_are_used_and_structlog_bindings_win(monkeypatch):
    from solace_common.request_context import clear_correlation_ids, set_correlation_ids
    monkeypatch.setattr(sa, "_service_token_manager", None)
    initialize_service_auth(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name="orchestrator-service"))
    set_correlation_ids(request_id="req-ctx", correlation_id="corr-ctx", turn_id="turn-request")
    structlog.contextvars.bind_contextvars(turn_id="turn-frame")
    try:
        headers = await service_auth_headers("safety-service", on_behalf_of=str(uuid4()))
    finally:
        structlog.contextvars.clear_contextvars()
        clear_correlation_ids()
        monkeypatch.setattr(sa, "_service_token_manager", None)
    assert (headers["X-Request-ID"], headers["X-Correlation-ID"], headers["X-Turn-ID"]) == (
        "req-ctx", "corr-ctx", "turn-frame")


@pytest.mark.asyncio
async def test_service_auth_headers_require_an_initialized_identity(monkeypatch):
    monkeypatch.setattr(sa, "_service_token_manager", None)
    with pytest.raises(RuntimeError, match="service auth"):
        await service_auth_headers("user-service")


def test_forwarded_auth_headers_copy_the_inbound_authorization():
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer inbound-token")]})
    assert forwarded_auth_headers(request)["Authorization"] == "Bearer inbound-token"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_314_service_token_model.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'forwarded_auth_headers'` (and `resolve_service_permissions`, `service_auth_headers`): there is no obo credential and no receiver-side permission resolution today, and `validate_service_token` trusts the `permissions` claim.
- [ ] **Step 3: Implement** the identities, permissions, grants and helpers listed under Files. `create_service_token` passes `extra_metadata={"obo_sub": <uuid>}` and `expire_minutes=5` to the JWT manager for obo tokens and returns before the cache write; `validate_service_token` replaces its claim check with `resolve_service_permissions(service_name)`. Update the one test in `tests/integration/test_service_integration.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\integration -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.4: get_current_user accepts orchestrator obo tokens; receivers resolve service permissions
**Covers:** MVP-314, MVP-021
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-07.2, WS-07.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: only services granted `service:act_for_user`, today the orchestrator alone, may act for a user; obo principals get `roles=['user']` only)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/middleware.py` (`get_current_user` (lines 230-272): decode once with `decode_token_async(token)` and no expected type; `ACCESS` gives the user as today; `SERVICE` is accepted only when `resolve_service_permissions(<name from sub>)` contains `service:act_for_user`, `metadata.obo_sub` parses as a UUID and `exp - iat <= OBO_TOKEN_TTL_SECONDS + clock skew`, yielding `AuthenticatedUser(user_id=obo_sub, token_type=SERVICE, roles=['user'], permissions=[], metadata={'via_service': name}, via_service=name)`; every other type or failure is 401; the RLS user is set to the resolved `user_id`. `get_current_service` (lines 298-332): permissions from `resolve_service_permissions` (unknown identity gives 401), `on_behalf_of` filled and the RLS user set when the token is obo. New `AuthenticatedService.on_behalf_of: str | None = None`. `require_service_permission` (lines 408-444): the inner dependency declares `service: AuthenticatedService = Depends(get_current_service)` so test overrides of `get_current_service` flow through, and it carries the attribute `__solace_service_permissions__ = permissions`. New `find_unguarded_service_routes`.)
- Test: `tests/solace_security/test_mvp_314_obo_and_receiver_permissions.py`
**Interfaces:**
- Consumes: `resolve_service_permissions`, `OBO_TOKEN_TTL_SECONDS`, `ServiceTokenManager.create_service_token(..., on_behalf_of=)` (WS-07.3); `JWTManager.create_service_token(..., extra_metadata=)` (WS-07.3); `AuthenticatedUser.via_service` (WS-07.2).
- Produces (IF-04; public names and signatures unchanged): obo acceptance in `get_current_user`, receiver-side `get_current_service`, `AuthenticatedService.on_behalf_of`, override-friendly `require_service_permission(*permissions: str)`, and `find_unguarded_service_routes(app: FastAPI) -> list[str]`, which returns sorted `"<METHODS> <path>"` strings for every HTTP or WebSocket route whose dependency tree contains `get_current_service` but no dependency carrying `__solace_service_permissions__`. Service lanes (WS-12, WS-13, WS-14, WS-15, WS-16, WS-17, WS-18) call it in their own tests and WS-07.9 runs it over all nine apps.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_314_obo_and_receiver_permissions.py::test_get_current_user_accepts_orchestrator_obo_token`
```python
"""MVP-314 / MVP-021 (supporting): obo acceptance and receiver-side service permissions (IF-04)."""
from __future__ import annotations

import time
from uuid import uuid4

import jwt
import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from solace_common.request_context import get_current_user_id, reset_current_user_id
from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager, TokenType
from solace_security.middleware import (
    AuthenticatedService, find_unguarded_service_routes, get_current_service, require_service_permission,
)
from solace_security.service_auth import (
    SERVICE_PERMISSIONS, ServiceAuthSettings, ServiceIdentity, ServicePermission, ServiceTokenManager,
)

SECRET = "test-secret-key-32-bytes-long!!!"


class _URL:
    path = "/api/v1/x"


class _Req:
    url = _URL()


@pytest.fixture
def manager(monkeypatch) -> JWTManager:
    m = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: m)
    return m


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _obo(uid=None, service="orchestrator-service") -> str:
    tm = ServiceTokenManager(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name=service))
    return tm.create_service_token(service, on_behalf_of=str(uid or uuid4())).token


@pytest.mark.asyncio
async def test_get_current_user_accepts_orchestrator_obo_token(manager):
    reset_current_user_id()
    uid = uuid4()
    user = await mw.get_current_user(_Req(), _creds(_obo(uid)))
    assert user.user_id == str(uid)
    assert user.via_service == "orchestrator-service"
    assert user.roles == ["user"]
    assert user.token_type == TokenType.SERVICE
    assert get_current_user_id() == str(uid)
    reset_current_user_id()


@pytest.mark.asyncio
async def test_obo_from_a_service_without_act_for_user_is_rejected(manager):
    token = manager.create_service_token("therapy-service", extra_metadata={"obo_sub": str(uuid4())})
    with pytest.raises(HTTPException) as exc:
        await mw.get_current_user(_Req(), _creds(token))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_plain_service_token_and_long_lived_obo_are_rejected(manager):
    plain = manager.create_service_token("orchestrator-service")
    long_lived = manager.create_service_token(
        "orchestrator-service", expire_minutes=60, extra_metadata={"obo_sub": str(uuid4())})
    for token in (plain, long_lived):
        with pytest.raises(HTTPException) as exc:
            await mw.get_current_user(_Req(), _creds(token))
        assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_service_resolves_permissions_receiver_side(manager):
    forged = manager.create_service_token("memory-service", permissions=["service:escalate"])
    service = await mw.get_current_service(_Req(), _creds(forged))
    assert service.service_name == "memory-service"
    assert sorted(service.permissions) == sorted(p.value for p in SERVICE_PERMISSIONS[ServiceIdentity.MEMORY])
    assert "service:escalate" not in service.permissions


@pytest.mark.asyncio
async def test_unknown_service_identity_is_rejected(manager):
    now = int(time.time())
    token = jwt.encode({"sub": "service:config-service", "type": "service", "jti": "j", "iat": now, "exp": now + 300,
                        "iss": "solace-ai", "aud": "solace-ai-api", "metadata": {"service_name": "config-service"}},
                       SECRET, algorithm="HS256")
    with pytest.raises(HTTPException) as exc:
        await mw.get_current_service(_Req(), _creds(token))
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_obo_service_principal_carries_on_behalf_of(manager):
    uid = uuid4()
    service = await mw.get_current_service(_Req(), _creds(_obo(uid)))
    assert service.on_behalf_of == str(uid)


def _app() -> FastAPI:
    app = FastAPI()

    @app.post("/escalate")
    async def escalate(s: AuthenticatedService = Depends(
            require_service_permission(ServicePermission.TRIGGER_ESCALATION.value))):
        return {"service": s.service_name}

    @app.post("/check")
    async def check(s: AuthenticatedService = Depends(require_service_permission(ServicePermission.READ_SAFETY.value))):
        return {"service": s.service_name}

    @app.get("/open-status")
    async def open_status(s: AuthenticatedService = Depends(get_current_service)):
        return {"ok": True}

    return app


def test_claimed_but_ungranted_permission_is_denied(manager):
    forged = manager.create_service_token("memory-service", permissions=["service:escalate"])
    response = TestClient(_app()).post("/escalate", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 403


def test_require_service_permission_honours_dependency_overrides():
    app = _app()
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="orchestrator-service", permissions=[ServicePermission.READ_SAFETY.value])
    client = TestClient(app)
    assert client.post("/check").status_code == 200
    assert client.post("/escalate").status_code == 403


def test_find_unguarded_service_routes_lists_only_unguarded_routes():
    assert find_unguarded_service_routes(_app()) == ["GET /open-status"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_314_obo_and_receiver_permissions.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'find_unguarded_service_routes'`. With a temporary stub of that one function the defect tests also fail as the register states: the obo token gets 401 `INVALID_TOKEN_TYPE` (no credential works across services, MVP-021), the memory token claiming `service:escalate` passes `/escalate` with 200 and keeps the claim in `permissions` (MVP-314), and the override test gets 401 because `require_service_permission` bypasses `Depends(get_current_service)`.
- [ ] **Step 3: Implement** the middleware changes listed under Files. Read the service name from `sub` (`service:<name>`), reject a token whose `metadata.service_name` disagrees with it, and keep `_handle_auth_failure` for every rejection so the body never carries exception text. The route scan walks `route.dependant.dependencies` recursively.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** every service suite imports `middleware.py`, so run the security suites plus the consumer suites that override `get_current_service`. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\safety_service\tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\personality_service\tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\orchestrator_service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.5: One shared revocation store: Redis in strict environments, URL never logged, no 500 on a Redis error, session revocation
**Covers:** MVP-186, MVP-188
**Severity:** medium
**Wave:** 1
**Depends on:** WS-07.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: a limiter-backend failure fails closed on auth routes and open elsewhere, with a metric and an alert; the shared revocation check runs only on non-auth service routes, so it fails open with a metric and a WARNING, while user-service's own identity routes fail closed in WS-08.4; recorded as an owner Ruling); D-14 (strict means staging and production)
**Real infra:** yes
**Files:**
- Modify: `src/solace_security/auth.py` (`TokenBlacklist`: concrete `async def revoke_session(self, session_id: str, until: datetime) -> None` (stores `add(f"sid:{session_id}", until)`), `async def is_session_revoked(self, session_id: str) -> bool` and `def is_session_revoked_sync(self, session_id: str) -> bool`, so subclasses that implement only `add`/`is_blacklisted` keep working; `JWTManager.decode_token_async` checks the jti and, when `payload.session_id` is set, the session; any exception from those lookups is caught once per call, logged as `token_revocation_check_unavailable` (no token, jti or URL) and counted in `solace_token_revocation_check_errors_total`, and the token is treated as not revoked; `decode_token_sync` also checks `is_session_revoked_sync`)
- Modify: `src/solace_security/middleware.py` (replace `_build_token_blacklist` (lines 123-168) with public `build_token_blacklist`; add `configure_token_blacklist`; `_get_jwt_manager` uses the configured blacklist when one is set; the `url=redis_url` log field (line 153) is removed)
- Test: `tests/solace_security/test_mvp_186_shared_revocation.py`
- Test: `tests/realinfra/service-auth-authz/test_mvp_186_cross_service_revocation.py`
**Interfaces:**
- Consumes: `TokenPayload.session_id` populated from the `sid` claim (WS-07.2); `solace_infrastructure.runtime.is_strict_environment` (IF-01, WS-04 W0), imported inside the function to avoid an import cycle.
- Produces (IF-04 blacklist contract):
  - `build_token_blacklist(redis_client: Any | None = None, *, strict: bool | None = None) -> TokenBlacklist`. Uses the given client (WS-04 passes `runtime.redis`), else a lazy `redis.asyncio.from_url(REDIS_URL, decode_responses=True, socket_timeout=REDIS_TIMEOUT or 1.0)`. With `strict` (default `is_strict_environment()`), a missing `REDIS_URL` with no client, or `AUTH_BLACKLIST_BACKEND=memory`, raises `RuntimeError("REDIS_URL is required for token revocation in strict environments")`. In development/test it falls back to `InMemoryTokenBlacklist` with a WARNING. It never logs the URL.
  - `configure_token_blacklist(blacklist: TokenBlacklist | None) -> None`: sets (or clears) the process blacklist and clears the `_get_jwt_manager` cache.
  - Session revocation key: Redis `token_blacklist:sid:<session_id>`, value `"1"`, TTL until the given `until`. WS-08 writes it on logout, password change and reset; every service reads it through `get_current_user`.
  - Metric `solace_token_revocation_check_errors_total` (counter, no labels; the scrape target names the service).

- [ ] **Step 1: Write the failing tests** in `tests/solace_security/test_mvp_186_shared_revocation.py::test_strict_environment_without_redis_url_fails_loud` and the real-infra file
```python
"""MVP-186 / MVP-188: one shared revocation store (Redis in strict), URL never logged, errors never 500."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
import structlog
from fastapi.security import HTTPAuthorizationCredentials
from prometheus_client import REGISTRY

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager, RedisTokenBlacklist

SECRET = "test-secret-key-32-bytes-long!!!"
SID = "0b6f8a52-1111-4000-8000-000000000001"


class _URL:
    path = "/api/v1/x"


class _Req:
    url = _URL()


def _access(sid=SID, jti="j-1") -> str:
    now = int(time.time())
    return jwt.encode({"sub": "u-1", "type": "access", "jti": jti, "sid": sid, "iat": now, "exp": now + 600,
                       "iss": "solace-ai", "aud": "solace-ai-api"}, SECRET, algorithm="HS256")


@pytest.fixture(autouse=True)
def _isolated_manager(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", SECRET)
    monkeypatch.delenv("AUTH_BLACKLIST_BACKEND", raising=False)
    mw._get_jwt_manager.cache_clear()
    yield
    mw.configure_token_blacklist(None)
    mw._get_jwt_manager.cache_clear()


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_environment_without_redis_url_fails_loud(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        mw._get_jwt_manager()


def test_memory_override_is_refused_in_strict(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("REDIS_URL", "redis://:pw-override@redis:6379/0")
    monkeypatch.setenv("AUTH_BLACKLIST_BACKEND", "memory")
    with patch("redis.asyncio.from_url", return_value=MagicMock()), pytest.raises(RuntimeError):
        mw.build_token_blacklist()


def test_redis_url_is_never_logged(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://:s3cr3t-pw-188@redis:6379/0")
    with patch("redis.asyncio.from_url", return_value=MagicMock()), structlog.testing.capture_logs() as logs:
        blacklist = mw.build_token_blacklist()
    assert isinstance(blacklist, RedisTokenBlacklist)
    assert "s3cr3t-pw-188" not in repr(logs)


@pytest.mark.asyncio
async def test_revoking_a_session_rejects_every_token_of_that_session_only():
    blacklist = InMemoryTokenBlacklist()
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=blacklist)
    await blacklist.revoke_session(SID, datetime.now(timezone.utc) + timedelta(minutes=15))
    for token in (_access(jti="a"), _access(jti="b")):
        result = await manager.validate_access_token_async(token)
        assert (result.success, result.error_code) == (False, "TOKEN_REVOKED")
    other = await manager.validate_access_token_async(_access(sid="0b6f8a52-2222-4000-8000-000000000002", jti="c"))
    assert other.success is True


class _DownRedis:
    async def get(self, key):
        raise ConnectionError("redis down")

    async def set(self, *args, **kwargs):
        raise ConnectionError("redis down")


@pytest.mark.asyncio
async def test_revocation_backend_error_fails_open_with_one_metric_increment(monkeypatch):
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=RedisTokenBlacklist(_DownRedis()))
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    before = REGISTRY.get_sample_value("solace_token_revocation_check_errors_total") or 0.0
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=_access())
    user = await mw.get_current_user(_Req(), creds)
    assert user.user_id == "u-1"
    assert REGISTRY.get_sample_value("solace_token_revocation_check_errors_total") == before + 1


def test_configured_blacklist_is_used_by_the_middleware():
    shared = InMemoryTokenBlacklist()
    mw.configure_token_blacklist(shared)
    assert mw._get_jwt_manager()._blacklist is shared
```
```python
"""tests/realinfra/service-auth-authz/test_mvp_186_cross_service_revocation.py
MVP-186 on real Redis: a revocation written by one service is seen by every other; pins the sid key."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import redis.asyncio as aioredis

from solace_security.auth import AuthSettings, JWTManager
from solace_security.middleware import build_token_blacklist

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
SECRET = "test-secret-key-32-bytes-long!!!"
SID = "7d2c9e10-3333-4000-8000-000000000003"


def _token(jti: str) -> str:
    now = int(time.time())
    return jwt.encode({"sub": "u-1", "type": "access", "jti": jti, "sid": SID, "iat": now, "exp": now + 600,
                       "iss": "solace-ai", "aud": "solace-ai-api"}, SECRET, algorithm="HS256")


@pytest.mark.asyncio
async def test_revocation_propagates_between_two_service_processes(redis_url):
    a_client = aioredis.from_url(redis_url, decode_responses=True)
    b_client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        service_a = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=build_token_blacklist(a_client))
        service_b = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=build_token_blacklist(b_client))
        assert (await service_b.validate_access_token_async(_token("j-1"))).success is True
        await service_a.revoke_token(_token("j-1"))
        assert (await service_b.validate_access_token_async(_token("j-1"))).error_code == "TOKEN_REVOKED"
        await service_a._blacklist.revoke_session(SID, datetime.now(timezone.utc) + timedelta(minutes=15))
        assert (await service_b.validate_access_token_async(_token("j-2"))).error_code == "TOKEN_REVOKED"
        assert await b_client.get(f"token_blacklist:sid:{SID}") == "1"
        assert 0 < await b_client.ttl(f"token_blacklist:sid:{SID}") <= 15 * 60
    finally:
        await a_client.aclose()
        await b_client.aclose()
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_186_shared_revocation.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\service-auth-authz\test_mvp_186_cross_service_revocation.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL. The strict test gets a JWTManager with a per-process `InMemoryTokenBlacklist` and no error (the register's "logout never reaches the other services"); `build_token_blacklist`, `configure_token_blacklist` and `revoke_session` are missing (`AttributeError`); the outage test raises `ConnectionError` out of `get_current_user` (the 500); the real-infra test fails on `build_token_blacklist` import.
- [ ] **Step 3: Implement** the blacklist API, the fail-open check with its counter and the two middleware functions listed under Files. Register the counter through a guarded helper that returns the existing collector from `prometheus_client.REGISTRY` if the name is already registered, so re-imports do not raise.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (the existing `test_middleware.py` fallback tests still pass because they run outside strict environments)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.6: Bounded, self-sweeping in-memory rate-limit store
**Covers:** MVP-311
**Severity:** low
**Wave:** 1
**Depends on:** none
**Implementer model:** sonnet
**Review:** security-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `infrastructure/api_gateway/rate_limiting.py` (`RateLimitStore.__init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, sweep_every: int = 1000)` with class constant `DEFAULT_MAX_ENTRIES = 100_000`; `_counters` becomes an `OrderedDict` with move-to-end on every touch and least-recently-used eviction once `max_entries` is exceeded; `increment` calls `cleanup_expired()` every `sweep_every` increments; the unused `_expiry` dict (line 128) is removed; `RedisRateLimitStore._sync_fallback` uses the bounded default)
- Test: `infrastructure/api_gateway/tests/test_mvp_311_rate_limit_store_bounds.py`
**Interfaces:**
- Consumes: none.
- Produces: `RateLimitStore(max_entries: int = 100_000, sweep_every: int = 1000)`; behaviour of `increment`, `get_count`, `reset` and `cleanup_expired` is otherwise unchanged. WS-07.7 builds dev/test limiters on it.

- [ ] **Step 1: Write the failing test** in `infrastructure/api_gateway/tests/test_mvp_311_rate_limit_store_bounds.py::test_expired_counters_are_swept_during_normal_increments`
```python
"""MVP-311: the in-memory rate-limit store must not grow without bound."""
from __future__ import annotations

import infrastructure.api_gateway.rate_limiting as rl


def _policy(limit: int = 5) -> rl.RateLimitPolicy:
    return rl.RateLimitPolicy(name="bounded", limit=limit, window=rl.RateLimitWindow.MINUTE)


def test_expired_counters_are_swept_during_normal_increments(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(rl.time, "time", lambda: clock["t"])
    store = rl.RateLimitStore(sweep_every=100)
    for i in range(5000):
        store.increment(_policy(), f"ip:{i}")
    clock["t"] += 3 * 60
    for i in range(100):
        store.increment(_policy(), f"fresh:{i}")
    assert len(store._counters) <= 100


def test_store_is_bounded_and_keeps_the_most_recent_keys():
    store = rl.RateLimitStore(max_entries=100)
    for i in range(500):
        store.increment(_policy(), f"ip:{i}")
    assert len(store._counters) <= 100
    assert store.get_count(_policy(), "ip:499") == 1
    assert store.get_count(_policy(), "ip:0") == 0


def test_a_hot_key_survives_eviction_pressure():
    store = rl.RateLimitStore(max_entries=50)
    for i in range(300):
        store.increment(_policy(limit=1000), "attacker")
        store.increment(_policy(limit=1000), f"noise:{i}")
    assert store.get_count(_policy(limit=1000), "attacker") == 300


def test_redis_store_sync_fallback_is_bounded():
    assert rl.RedisRateLimitStore(object())._sync_fallback._max_entries == rl.RateLimitStore.DEFAULT_MAX_ENTRIES
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\infrastructure\api_gateway\tests\test_mvp_311_rate_limit_store_bounds.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `TypeError: RateLimitStore.__init__() got an unexpected keyword argument 'sweep_every'` (and `'max_entries'`): the store has no bound and `cleanup_expired()` has no caller, so 5,000 callers leave 5,000 counters.
- [ ] **Step 3: Implement** the bounded store in `infrastructure/api_gateway/rate_limiting.py` as listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\infrastructure\api_gateway\tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_rate_limit_middleware.py -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.7: install_rate_limiting: real-client keying behind Caddy, auth-route policies, Redis backend and the D-26 failure policy
**Covers:** MVP-101, MVP-185
**Severity:** high
**Wave:** 1
**Depends on:** WS-07.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: the app-level Redis limiter keyed on user id or the real client IP with TRUSTED_PROXY_HOPS=1 is the control; on a backend failure auth routes return 503 and other routes pass, with a metric and an alert)
**Real infra:** yes
**Files:**
- Modify: `src/solace_security/rate_limit.py` (new `resolve_trusted_proxy_hops`, `classify_route`, `AUTH_ROUTE_LIMITS`, `build_rate_limiter`, `install_rate_limiting`; `client_ip(request, trusted_proxy_hops: int | None = None)` where `None` means `resolve_trusted_proxy_hops()`; `RateLimitMiddleware` gains `auth_route_predicate: Callable[[str], bool] | None = None`, catches any limiter exception, increments `solace_rate_limit_backend_errors_total{service}`, logs `rate_limit_backend_unavailable` without the identifier, and returns 503 `{"error": {"code": "RATE_LIMIT_UNAVAILABLE", "message": "Authentication is temporarily unavailable. Please retry shortly."}}` with `Retry-After: 5` on auth routes or passes the request through elsewhere)
- Modify: `infrastructure/api_gateway/rate_limiting.py` (`create_solace_rate_limiter`: the Redis requirement (lines 367-374) uses `is_strict_environment()` instead of `env == "production"`)
- Test: `tests/solace_security/test_mvp_101_install_rate_limiting.py`
- Test: `tests/realinfra/service-auth-authz/test_mvp_101_shared_redis_limits.py`
**Interfaces:**
- Consumes: `RateLimitStore` (WS-07.6); `create_solace_rate_limiter`, `RateLimitPolicy`; `_get_jwt_manager().validate_access_token` (signature-checked keying only; the route still authenticates); `solace_infrastructure.runtime.is_strict_environment` and a `ServiceRuntimeSpec`-shaped `spec` with `service_name` and `environment` (IF-01, WS-04 W0), imported lazily.
- Produces (IF-04 rate limiting):
  - `resolve_trusted_proxy_hops(environ: Mapping[str, str] | None = None) -> int`: `TRUSTED_PROXY_HOPS` when set (an int >= 0), else 1 in strict environments and 0 otherwise.
  - `classify_route(path: str) -> tuple[str | None, bool]`: route name and whether it is an auth route. Auth routes are `/api/v1/auth/*` and `/api/v1/users/me/password`. Named routes and per-minute limits (`AUTH_ROUTE_LIMITS`): `auth-login` 20, `auth-register` 10, `auth-refresh` 30, `auth-resend` 5, `auth-verify` 20, `auth-forgot` 5, `auth-reset` 10, `auth-reauth` 10, `auth-mfa` 10 (`/mfa/verify`, `/mfa/enroll/*`), `auth-oauth` 30 (`/oauth/google/*`), `auth-session` 30 (`/logout`, `/revoke-token`), `auth-password-change` 5; `/api/v1/orchestrator/chat` maps to the existing `chat-message` policy and is not an auth route.
  - `build_rate_limiter(service_name: str, *, redis_client: Any | None) -> RateLimiter`: `create_solace_rate_limiter` plus the route policies, with limits overridden by the optional JSON object `RATE_LIMIT_OVERRIDES_JSON` (route name to per-minute limit).
  - `install_rate_limiting(app: FastAPI, spec: ServiceRuntimeSpec, *, force: bool = False, redis_client: Any | None = None) -> None`: installs `RateLimitMiddleware` keyed `user:<sub>` for a signature-valid bearer, else `ip:<client_ip>`; Redis from `redis_client`, else a lazy `redis.asyncio.from_url(REDIS_URL, socket_timeout=0.5, socket_connect_timeout=0.5)`; strict with neither raises `RuntimeError("REDIS_URL is required for rate limiting in strict environments")`; development/test without Redis uses the bounded in-memory store with a WARNING. Unless `force`, it does nothing when the environment (spec or `ENVIRONMENT`) starts with `test` and is not strict, so suites that share a module-level app do not bleed counters. Call it before CORS is added so CORS stays outermost.
  - Metric `solace_rate_limit_backend_errors_total{service}`.

- [ ] **Step 1: Write the failing tests** in `tests/solace_security/test_mvp_101_install_rate_limiting.py::test_login_bucket_is_per_real_client_not_per_proxy` and the real-infra file
```python
"""MVP-101 / MVP-185: limiter keyed on the real client behind Caddy, Redis-backed, D-26 failure policy."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from solace_security.rate_limit import classify_route, install_rate_limiting, resolve_trusted_proxy_hops

LOGIN = "/api/v1/auth/login"


def _spec(env: str = "development"):
    return SimpleNamespace(service_name="user-service", environment=env, rate_limited=True, edge=True)


def _app(spec, **kwargs) -> FastAPI:
    app = FastAPI()

    @app.post(LOGIN)
    async def login():
        return {"ok": False}

    @app.get("/api/v1/users/me")
    async def me():
        return {"ok": True}

    @app.get("/live")
    async def live():
        return {"status": "alive"}

    install_rate_limiting(app, spec, force=True, **kwargs)
    return app


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("RATE_LIMIT_OVERRIDES_JSON", raising=False)
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")


def test_login_bucket_is_per_real_client_not_per_proxy():
    with TestClient(_app(_spec())) as client:
        attacker = {"x-forwarded-for": "198.51.100.7"}
        assert [client.post(LOGIN, headers=attacker).status_code for _ in range(20)] == [200] * 20
        assert client.post(LOGIN, headers=attacker).status_code == 429
        assert client.post(LOGIN, headers={"x-forwarded-for": "203.0.113.9"}).status_code == 200


@pytest.mark.parametrize("environ,expected", [
    ({"ENVIRONMENT": "production"}, 1), ({"ENVIRONMENT": "staging"}, 1),
    ({"ENVIRONMENT": "development"}, 0), ({"ENVIRONMENT": "development", "TRUSTED_PROXY_HOPS": "2"}, 2)])
def test_trusted_proxy_hops_resolution(environ, expected):
    assert resolve_trusted_proxy_hops(environ) == expected


def test_auth_routes_are_classified():
    assert classify_route(LOGIN) == ("auth-login", True)
    assert classify_route("/api/v1/auth/register") == ("auth-register", True)
    assert classify_route("/api/v1/auth/oauth/google/exchange") == ("auth-oauth", True)
    assert classify_route("/api/v1/users/me/password") == ("auth-password-change", True)
    assert classify_route("/api/v1/orchestrator/chat") == ("chat-message", False)
    assert classify_route("/api/v1/users/me") == (None, False)


class _DownRedis:
    async def eval(self, *args, **kwargs):
        raise ConnectionError("redis down")

    async def ttl(self, *args, **kwargs):
        raise ConnectionError("redis down")


def test_backend_failure_fails_closed_on_auth_routes_and_open_elsewhere():
    labels = {"service": "user-service"}
    before = REGISTRY.get_sample_value("solace_rate_limit_backend_errors_total", labels) or 0.0
    with TestClient(_app(_spec(), redis_client=_DownRedis())) as client:
        denied = client.post(LOGIN)
        assert denied.status_code == 503
        assert denied.json()["error"]["code"] == "RATE_LIMIT_UNAVAILABLE"
        assert denied.headers["Retry-After"] == "5"
        assert client.get("/api/v1/users/me").status_code == 200
    assert REGISTRY.get_sample_value("solace_rate_limit_backend_errors_total", labels) == before + 2


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_environment_requires_redis(env):
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        _app(_spec(env))


def test_named_limits_can_be_overridden(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_OVERRIDES_JSON", '{"auth-login": 2}')
    with TestClient(_app(_spec())) as client:
        headers = {"x-forwarded-for": "192.0.2.1"}
        assert [client.post(LOGIN, headers=headers).status_code for _ in range(3)] == [200, 200, 429]


def test_liveness_probe_is_exempt():
    with TestClient(_app(_spec())) as client:
        assert all(client.get("/live").status_code == 200 for _ in range(50))
```
```python
"""tests/realinfra/service-auth-authz/test_mvp_101_shared_redis_limits.py
MVP-101 on real Redis: two instances of a service share one login bucket per real client."""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_security.rate_limit import install_rate_limiting

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
LOGIN = "/api/v1/auth/login"


def _instance() -> FastAPI:
    app = FastAPI()

    @app.post(LOGIN)
    async def login():
        return {"ok": False}

    install_rate_limiting(app, SimpleNamespace(service_name="user-service", environment="development",
                                               rate_limited=True, edge=True), force=True)
    return app


def test_two_instances_share_one_login_bucket(redis_url, monkeypatch):
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    headers = {"x-forwarded-for": f"198.51.100.{random.randint(1, 250)}"}
    with TestClient(_instance()) as a, TestClient(_instance()) as b:
        codes = [a.post(LOGIN, headers=headers).status_code for _ in range(10)]
        codes += [b.post(LOGIN, headers=headers).status_code for _ in range(11)]
    assert codes[:20] == [200] * 20
    assert codes[20] == 429
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_101_install_rate_limiting.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\service-auth-authz\test_mvp_101_shared_redis_limits.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'install_rate_limiting'`. The defects behind it: user-service keys every client on the socket peer (`client_ip(request)` with hops 0), so all clients behind Caddy share one 20/min login bucket (MVP-101), and `RateLimitMiddleware.dispatch` (lines 115-120) has no handling for a limiter exception, so a Redis outage turns every login into a 500 (MVP-185).
- [ ] **Step 3: Implement** the functions listed under Files in `src/solace_security/rate_limit.py` and the strict check in `create_solace_rate_limiter`. The principal resolver tries the bearer only when an `Authorization: Bearer` header is present and degrades to the IP key on any failure.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\infrastructure\api_gateway\tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\orchestrator_service\tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.8: Clinician subject scoping: can_access_subject, authorize_subject and set_subject_context
**Covers:** MVP-120, MVP-141
**Severity:** medium
**Wave:** 3
**Depends on:** WS-07.1, WS-07.4, WS-13.3, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21 (default assumed: a clinician reads or writes a user's PHI only when assigned to that user in clinician_patient_assignments or when they acknowledged a crisis episode for that user within the last 72 h; admins have no PHI read path; the RLS user is set to the data owner only after the check passes; no BYPASSRLS role)
**Real infra:** yes
**Files:**
- Modify: `src/solace_security/authorization.py` (new `SubjectAccessChecker` protocol, `PostgresSubjectAccessChecker`, `CLINICIAN_ACK_ACCESS_HOURS = 72`)
- Modify: `src/solace_security/middleware.py` (new `can_access_subject`, `set_subject_context`, `authorize_subject`)
- Create: `tests/solace_security/test_mvp_120_subject_scoping.py` (test; WS-07.11 extends it in W4)
- Create: `tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py` (test)
**Interfaces:**
- Consumes: `AuthenticatedUser` with `via_service` and `roles` (WS-07.2, WS-07.4); `set_current_user_id` (solace_common); `ConnectionPoolManager.get_pool("default")` (IF-03, WS-02), resolved lazily when no pool is injected; tables `clinician_patient_assignments(clinician_id, patient_id)` (001) and `crisis_alerts(user_id, acknowledged_by, acknowledged_at)` (IF-09, WS-13 slot 015), both non-RLS and readable by solace_app.
- Produces (IF-04 W3; async because the decision reads the database):
  - `class SubjectAccessChecker(Protocol)`: `async def is_assigned(self, clinician_id: str, subject_user_id: str) -> bool`; `async def acknowledged_recent_episode(self, clinician_id: str, subject_user_id: str, *, within_hours: int = 72) -> bool`.
  - `PostgresSubjectAccessChecker(pool: asyncpg.Pool | None = None)` implementing it with parameterised `SELECT 1 ... LIMIT 1` queries.
  - `async def can_access_subject(principal: AuthenticatedUser, subject_user_id: str | UUID, *, checker: SubjectAccessChecker | None = None) -> bool`: False for a non-UUID subject; True for self, including an obo principal acting for that same user, without a lookup; for role `clinician` (not via a service), the assignment or the 72-hour acknowledgement; False for every other role.
  - `def set_subject_context(subject_user_id: str | UUID) -> contextvars.Token`: sets the RLS user to the canonical UUID string; `ValueError` for a non-UUID.
  - `async def authorize_subject(principal, subject_user_id, *, checker=None) -> None`: raises `HTTPException(403, detail={"code": "SUBJECT_FORBIDDEN", "message": "Not permitted for this user"})` without touching the context, else calls `set_subject_context`. Adopted in W3 phase B by WS-14 (`GET /users/{uuid}`), WS-15, WS-16 and WS-17 at their clinician paths (Cross-workstream requests).
  - Satisfies request R-13-6 from WS-13 (raised against WS-13.3's slot 015): `crisis_alerts` has the CHECK constraints `max_level IN ('HIGH','CRITICAL')` and `status IN ('open','acknowledged','resolved')` and `max_level` has no default, so the real-infra test passes `max_level="HIGH"` and `status="acknowledged"` explicitly on every `crisis_alerts` row and never lets the generic filler choose them.

- [ ] **Step 1: Write the failing tests** in `tests/solace_security/test_mvp_120_subject_scoping.py::test_unassigned_clinician_is_denied_and_assigned_one_allowed` and the real-infra file
```python
"""MVP-120 / MVP-141: clinicians reach only assigned patients; the RLS user becomes the data owner only after authorization."""
from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import HTTPException

from solace_common.request_context import get_current_user_id, reset_current_user_id, set_current_user_id
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, authorize_subject, can_access_subject, set_subject_context


class _Checker:
    def __init__(self, assigned=(), acknowledged=()):
        self.assigned, self.acknowledged, self.calls = set(assigned), set(acknowledged), []

    async def is_assigned(self, clinician_id, subject_user_id):
        self.calls.append(("assigned", clinician_id, subject_user_id))
        return (clinician_id, subject_user_id) in self.assigned

    async def acknowledged_recent_episode(self, clinician_id, subject_user_id, *, within_hours=72):
        self.calls.append(("ack", clinician_id, subject_user_id, within_hours))
        return (clinician_id, subject_user_id) in self.acknowledged


def _p(uid, *roles, via=None) -> AuthenticatedUser:
    return AuthenticatedUser(user_id=str(uid), token_type=TokenType.SERVICE if via else TokenType.ACCESS,
                             roles=list(roles), via_service=via)


@pytest.mark.asyncio
async def test_self_access_is_allowed_without_a_lookup():
    uid, checker = uuid4(), _Checker()
    assert await can_access_subject(_p(uid, "user"), uid, checker=checker) is True
    assert checker.calls == []


@pytest.mark.asyncio
async def test_unassigned_clinician_is_denied_and_assigned_one_allowed():
    clinician, patient, other = uuid4(), uuid4(), uuid4()
    checker = _Checker(assigned={(str(clinician), str(patient))})
    assert await can_access_subject(_p(clinician, "clinician"), patient, checker=checker) is True
    assert await can_access_subject(_p(clinician, "clinician"), other, checker=checker) is False


@pytest.mark.asyncio
async def test_clinician_who_acknowledged_an_episode_within_72h_is_allowed():
    clinician, patient = uuid4(), uuid4()
    checker = _Checker(acknowledged={(str(clinician), str(patient))})
    assert await can_access_subject(_p(clinician, "clinician"), patient, checker=checker) is True
    assert ("ack", str(clinician), str(patient), 72) in checker.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("roles", [("admin",), ("superadmin",), ("system",), ("user",), ("therapist",)])
async def test_other_roles_never_reach_another_users_phi(roles):
    assert await can_access_subject(_p(uuid4(), *roles), uuid4(), checker=_Checker()) is False


@pytest.mark.asyncio
async def test_obo_principal_reaches_only_its_own_user():
    uid = uuid4()
    principal = _p(uid, "user", via="orchestrator-service")
    assert await can_access_subject(principal, uid, checker=_Checker()) is True
    assert await can_access_subject(principal, uuid4(), checker=_Checker()) is False


@pytest.mark.asyncio
async def test_non_uuid_subject_is_denied():
    assert await can_access_subject(_p(uuid4(), "clinician"), "not-a-uuid", checker=_Checker()) is False


@pytest.mark.asyncio
async def test_authorize_subject_sets_the_rls_user_to_the_owner_only_after_success():
    clinician, patient, other = uuid4(), uuid4(), uuid4()
    checker = _Checker(assigned={(str(clinician), str(patient))})
    set_current_user_id(str(clinician))
    with pytest.raises(HTTPException) as exc:
        await authorize_subject(_p(clinician, "clinician"), other, checker=checker)
    assert exc.value.status_code == 403
    assert get_current_user_id() == str(clinician)
    await authorize_subject(_p(clinician, "clinician"), patient, checker=checker)
    assert get_current_user_id() == str(patient)
    reset_current_user_id()


def test_set_subject_context_requires_a_uuid():
    with pytest.raises(ValueError):
        set_subject_context("not-a-uuid")
```
```python
"""tests/realinfra/service-auth-authz/test_mvp_120_subject_access_checker.py
MVP-120: the checker's SQL matches the migrated schema and runs as solace_app."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg
import pytest

from solace_security.authorization import PostgresSubjectAccessChecker

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]

_PLACEHOLDERS = {"uuid": lambda: uuid4(), "boolean": lambda: False, "integer": lambda: 0, "bigint": lambda: 0,
                 "timestamp with time zone": lambda: datetime.now(timezone.utc), "jsonb": lambda: "{}",
                 "ARRAY": lambda: []}


async def _insert(conn: asyncpg.Connection, table: str, **known) -> None:
    """Insert one row; every NOT NULL column without a default gets a typed placeholder."""
    columns = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = $1 AND is_nullable = 'NO' AND column_default IS NULL", table)
    values = dict(known)
    for column in columns:
        if column["column_name"] not in values:
            values[column["column_name"]] = _PLACEHOLDERS.get(column["data_type"], lambda: f"x-{uuid4().hex[:12]}")()
    names = ", ".join(values)
    params = ", ".join(f"${i}" for i in range(1, len(values) + 1))
    await conn.execute(f"INSERT INTO {table} ({names}) VALUES ({params})", *values.values())


@pytest.mark.asyncio(loop_scope="session")  # app_pool belongs to the session-scoped IF-19 fixture
async def test_checker_reads_assignments_and_recent_acknowledgements_as_solace_app(pg_migrated):
    clinician, patient, acked, stale = uuid4(), uuid4(), uuid4(), uuid4()
    superuser = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        for uid, role in ((clinician, "clinician"), (patient, "user"), (acked, "user"), (stale, "user")):
            await _insert(superuser, "users", id=uid, email=f"{uid.hex}@test.invalid", role=role)
        await _insert(superuser, "clinician_patient_assignments", id=uuid4(), clinician_id=clinician,
                      patient_id=patient, assigned_at=datetime.now(timezone.utc))
        now = datetime.now(timezone.utc)
        await _insert(superuser, "crisis_alerts", alert_id=uuid4(), user_id=acked, acknowledged_by=clinician,
                      acknowledged_at=now - timedelta(hours=10), status="acknowledged", max_level="HIGH")
        await _insert(superuser, "crisis_alerts", alert_id=uuid4(), user_id=stale, acknowledged_by=clinician,
                      acknowledged_at=now - timedelta(hours=80), status="acknowledged", max_level="HIGH")
    finally:
        await superuser.close()
    checker = PostgresSubjectAccessChecker(pool=pg_migrated.app_pool)
    assert await checker.is_assigned(str(clinician), str(patient)) is True
    assert await checker.is_assigned(str(clinician), str(acked)) is False
    assert await checker.acknowledged_recent_episode(str(clinician), str(acked)) is True
    assert await checker.acknowledged_recent_episode(str(clinician), str(stale)) is False
```
- [ ] **Step 2: Run them and confirm they fail**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_120_subject_scoping.py -o "addopts=" -p no:logging -q; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\realinfra\service-auth-authz\test_mvp_120_subject_access_checker.py -m realinfra -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'authorize_subject'` and `cannot import name 'PostgresSubjectAccessChecker'`: nothing consults `clinician_patient_assignments` today (MVP-120) and the RLS user is always the caller (middleware.py:266-270, MVP-141). If the real-infra test fails instead with `CheckViolationError` on `crisis_alerts`, an explicit `max_level`/`status` was dropped (R-13-6); that is a test defect, not the RED.
- [ ] **Step 3: Implement** the protocol, the checker and the three middleware functions as listed. The clinician branch requires `"clinician" in principal.roles` and `principal.via_service is None`; nothing in this task trusts a role from a request body.
- [ ] **Step 4: Run them and confirm they pass**
Run: same two commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.9: Acceptance: no service-token route in the nine MVP apps lacks a permission check
**Covers:** MVP-314
**Severity:** low
**Wave:** 4
**Depends on:** WS-07.4
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: every service-only route enforces `require_service_permission` with receiver-side permissions)
**Real infra:** no
**Files:**
- Create: `tests/security/test_mvp_314_service_routes_guarded.py`
- Test: `tests/security/test_mvp_314_service_routes_guarded.py`
**Interfaces:**
- Consumes: `find_unguarded_service_routes(app)` (WS-07.4); the supporting route edits landed in W1 to W3 by WS-12, WS-13, WS-14, WS-15, WS-16, WS-17 and WS-18 (Cross-workstream requests); the app import paths of IF-21 (`services.<pkg>.src.main` for underscore services, `src.main` from `services/user-service` and `services/notification-service`, `main` from `services/analytics-service/src`).
- Produces: the closing proof for MVP-314. No production code changes.

- [ ] **Step 1: Write the test** in `tests/security/test_mvp_314_service_routes_guarded.py::test_every_service_token_route_enforces_a_permission`
```python
"""MVP-314 acceptance: every route that accepts a service token enforces a receiver-side permission."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
APPS = [
    ("orchestrator-service", [REPO], "services.orchestrator_service.src.main"),
    ("safety-service", [REPO], "services.safety_service.src.main"),
    ("therapy-service", [REPO], "services.therapy_service.src.main"),
    ("diagnosis-service", [REPO], "services.diagnosis_service.src.main"),
    ("memory-service", [REPO], "services.memory_service.src.main"),
    ("personality-service", [REPO], "services.personality_service.src.main"),
    ("user-service", [REPO / "services" / "user-service"], "src.main"),
    ("notification-service", [REPO / "services" / "notification-service"], "src.main"),
    ("analytics-service", [REPO / "services" / "analytics-service" / "src"], "main"),
]
_PROBE = ("import importlib, json, sys; sys.path[:0] = {paths!r}; app = importlib.import_module({module!r}).app; "
          "from solace_security.middleware import find_unguarded_service_routes; "
          "print('UNGUARDED=' + json.dumps(find_unguarded_service_routes(app)))")


@pytest.mark.parametrize("service,paths,module", APPS, ids=[a[0] for a in APPS])
def test_every_service_token_route_enforces_a_permission(service, paths, module):
    env = {**os.environ, "PYTHONPATH": f"{REPO};{REPO / 'src'}", "ENVIRONMENT": "test",
           "AUTH_SECRET_KEY": "mvp-314-test-secret-key-32-bytes-long!"}
    code = _PROBE.format(paths=[str(p) for p in paths], module=module)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [line for line in out.stdout.splitlines() if line.startswith("UNGUARDED=")][-1]
    assert json.loads(line[len("UNGUARDED="):]) == [], f"{service} has service-token routes without a permission"
```
- [ ] **Step 2: Run it and confirm it fails on the pre-adoption tree**
Find the head tree of WS-07.4 in `D:\tmp\mvp-exec\ledger\progress.md` (`Task WS-07.4: complete (trees <base>..<head>)`; services were not yet guarded then). Run: `git -C D:\Repo\Contextual-Chatbot archive --format=zip -o D:\tmp\mvp-exec\agy\mvp314-pre.zip <WS-07.4 head tree>; Expand-Archive D:\tmp\mvp-exec\agy\mvp314-pre.zip D:\tmp\mvp-exec\agy\mvp314-pre -Force; Copy-Item D:\Repo\Contextual-Chatbot\tests\security\test_mvp_314_service_routes_guarded.py D:\tmp\mvp-exec\agy\mvp314-pre\tests\security\; $env:PYTHONPATH="D:\tmp\mvp-exec\agy\mvp314-pre;D:\tmp\mvp-exec\agy\mvp314-pre\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\tmp\mvp-exec\agy\mvp314-pre\tests\security\test_mvp_314_service_routes_guarded.py -o "addopts=" -p no:logging -q`
Expected: FAIL for safety-service (listing `POST /api/v1/safety/escalate` among others), personality-service (`POST /api/v1/personality/detect`), analytics-service (`POST /api/v1/analytics/events/ingest`) and the therapy/diagnosis/memory `/status` routes: the routes accepted any service token.
- [ ] **Step 3: Implement:** no production code changes. If any app still lists an unguarded route on the current tree, stop and record a ledger finding naming the route and its owning workstream; do not edit their files.
- [ ] **Step 4: Run it and confirm it passes on the current tree**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\security\test_mvp_314_service_routes_guarded.py -o "addopts=" -p no:logging -q`. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\security -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.10: RS256-only user tokens in strict environments and per-service RS256 service-token keys
**Covers:** MVP-209
**Severity:** medium
**Wave:** 4
**Depends on:** WS-07.3, WS-07.5
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-03 (default assumed: from W4 user tokens are RS256-only with AUTH_ACCEPT_LEGACY_HS256=false, and each service signs its own service tokens with SERVICE_TOKEN_PRIVATE_KEY_PEM, kid = service name); D-07 (decided 2026-10-19, changed: SOPS + age replace the default's cloud secrets store; the consumer contract is unchanged: key material is read through `NAME_FILE` from root-only tmpfs `/run/secrets` files that `deploy/render_secrets.py` renders from the per-environment SOPS file `deploy/secrets/<env>.sops.yaml`, encrypted to that environment's age recipient)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/auth.py` (`AuthSettings.accept_legacy_hs256` defaults to `False`; a `model_post_init` check raises `ValueError` in strict environments unless `algorithm == "RS256"` and `accept_legacy_hs256 is False`; `JWTManager` loads `ServiceKeyring.from_env(...)`; `_encode` signs SERVICE tokens with the keyring when it can sign (RS256, `kid` = its own service name) and refuses HS256 service tokens in strict environments; `decode_token_sync`: an RS256 header whose `kid` is in the service keyset verifies with that key and requires `type == "service"` and `sub == "service:" + kid`; a `kid` in the user keyset requires `type != "service"`; an HS256 service token is rejected once a service keyset is configured or in strict environments)
- Modify: `src/solace_security/keys.py` (`KeyManager.from_settings`: the keyset requirement (lines 145-151) uses `is_strict_environment()` and never generates ephemeral keys there; new `ServiceKeySettings(BaseSettings)` with `env_prefix="SERVICE_TOKEN_"` and fields `private_key_pem: SecretStr | None`, `public_keys_json: str | None`; new `ServiceKeyring`)
- Modify: `src/solace_security/service_auth.py` (`ServiceTokenManager` passes its own `service_name` to the JWT manager's keyring; `create_service_token` raises `ValueError` naming the manager's service when asked to sign another identity while a keyring is configured)
- Modify: `scripts/generate_jwt_keys.py` (new `--out-dir DIR` and `--services all|<name,...>`: writes `AUTH_PRIVATE_KEY_PEM`, `AUTH_ACTIVE_KID`, `AUTH_PUBLIC_KEYS_JSON`, one `SERVICE_TOKEN_PRIVATE_KEY_PEM.<service>` per `ServiceIdentity` and `SERVICE_TOKEN_PUBLIC_KEYS_JSON` into the directory with owner-only permissions; prints only file names; refuses to overwrite without `--rotate`)
- Test: `tests/solace_security/test_mvp_209_rs256_and_service_keys.py`
**Interfaces:**
- Consumes: `ServiceTokenManager`, `ServiceIdentity` (WS-07.3); `is_strict_environment` (IF-01).
- Produces (IF-04 W4, IF-22 key names): `ServiceKeyring.from_env(service_name: str | None) -> ServiceKeyring` with `can_sign: bool`, `sign(claims: dict) -> str`, `public_pem_for(kid: str | None) -> str | None`; env `SERVICE_TOKEN_PRIVATE_KEY_PEM` (each service its own) and `SERVICE_TOKEN_PUBLIC_KEYS_JSON` (`{service_name: public_pem}`, all services); key-generation files named exactly after the IF-22 secret names. WS-08.19 consumes the strict RS256 rules for user-service's own issuer and verifier.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_209_rs256_and_service_keys.py::test_hs256_service_token_rejected_once_service_keys_are_configured`
```python
"""MVP-209: user tokens RS256-only in strict envs; service tokens signed with per-service keys (kid = service name)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jwt
import pytest

from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager, TokenType
from solace_security.keys import KeyManager, generate_rsa_keypair
from solace_security.service_auth import ServiceAuthSettings, ServiceTokenManager

SECRET = "test-secret-key-32-bytes-long!!!"
REPO = Path(__file__).resolve().parents[2]


def _now() -> int:
    return int(time.time())


@pytest.fixture
def keys():
    return {name: generate_rsa_keypair(name)[:2] for name in ("orchestrator-service", "safety-service")}


@pytest.fixture
def act_as(monkeypatch, keys):
    monkeypatch.setenv("SERVICE_TOKEN_PUBLIC_KEYS_JSON", json.dumps({n: pub for n, (_priv, pub) in keys.items()}))

    def use(name: str) -> None:
        monkeypatch.setenv("SERVICE_TOKEN_PRIVATE_KEY_PEM", keys[name][0])
    return use


def _verifier() -> JWTManager:
    return JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())


def test_accept_legacy_hs256_defaults_to_false():
    assert AuthSettings(secret_key=SECRET).accept_legacy_hs256 is False


@pytest.mark.parametrize("env", ["staging", "production"])
def test_strict_environment_refuses_hs256_and_the_legacy_flag(monkeypatch, env):
    monkeypatch.setenv("ENVIRONMENT", env)
    with pytest.raises(ValueError, match="RS256"):
        AuthSettings(secret_key=SECRET, algorithm="HS256")
    _priv, pub, kid = generate_rsa_keypair("u1")
    with pytest.raises(ValueError, match="AUTH_ACCEPT_LEGACY_HS256"):
        AuthSettings(secret_key=SECRET, algorithm="RS256", public_keys_json=json.dumps({kid: pub}),
                     accept_legacy_hs256=True)


def test_keyset_is_required_in_staging_not_only_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    settings = SimpleNamespace(algorithm="RS256", private_key_pem=None, active_kid=None, public_keys_json=None)
    with pytest.raises(RuntimeError, match="AUTH_PUBLIC_KEYS_JSON"):
        KeyManager.from_settings(settings)


def test_hs256_service_token_rejected_once_service_keys_are_configured(act_as):
    act_as("safety-service")
    legacy = jwt.encode({"sub": "service:orchestrator-service", "type": "service", "jti": "j", "iat": _now(),
                         "exp": _now() + 300, "iss": "solace-ai", "aud": "solace-ai-api",
                         "metadata": {"service_name": "orchestrator-service"}}, SECRET, algorithm="HS256")
    assert _verifier().decode_token_sync(legacy, TokenType.SERVICE).success is False


def test_service_tokens_are_rs256_with_kid_equal_to_service_name(act_as):
    act_as("orchestrator-service")
    manager = ServiceTokenManager(AuthSettings(secret_key=SECRET),
                                  ServiceAuthSettings(service_name="orchestrator-service"))
    token = manager.create_service_token("orchestrator-service").token
    header = jwt.get_unverified_header(token)
    assert (header["alg"], header["kid"]) == ("RS256", "orchestrator-service")
    assert _verifier().decode_token_sync(token, TokenType.SERVICE).success is True


def test_a_service_cannot_sign_another_services_identity(act_as):
    act_as("safety-service")
    manager = ServiceTokenManager(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name="safety-service"))
    with pytest.raises(ValueError, match="safety-service"):
        manager.create_service_token("orchestrator-service")


def test_kid_sub_mismatch_is_rejected(act_as, keys):
    act_as("safety-service")
    forged = jwt.encode({"sub": "service:orchestrator-service", "type": "service", "jti": "j", "iat": _now(),
                         "exp": _now() + 300, "iss": "solace-ai", "aud": "solace-ai-api",
                         "metadata": {"service_name": "orchestrator-service"}},
                        keys["safety-service"][0], algorithm="RS256", headers={"kid": "safety-service"})
    assert _verifier().decode_token_sync(forged, TokenType.SERVICE).success is False


def test_generate_jwt_keys_writes_files_and_prints_no_key_material(tmp_path):
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "generate_jwt_keys.py"), "--out-dir", str(tmp_path),
                          "--services", "all"], capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "-----BEGIN" not in out.stdout
    keyset = json.loads((tmp_path / "SERVICE_TOKEN_PUBLIC_KEYS_JSON").read_text())
    assert {"orchestrator-service", "user-service", "notification-service", "analytics-service"} <= set(keyset)
    private = (tmp_path / "SERVICE_TOKEN_PRIVATE_KEY_PEM.orchestrator-service").read_text()
    assert private.startswith("-----BEGIN PRIVATE KEY-----")
    assert (tmp_path / "AUTH_PRIVATE_KEY_PEM").exists() and (tmp_path / "AUTH_PUBLIC_KEYS_JSON").exists()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_209_rs256_and_service_keys.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `accept_legacy_hs256` is `True` (auth.py:55); strict settings with HS256 or the legacy flag are accepted; `KeyManager.from_settings` generates ephemeral keys under staging (keys.py:145-154 checks only `production`); the HS256 service token signed with the shared secret verifies (any container can mint any service identity); service tokens are HS256 with no `kid`; the key script prints the private key to stdout (`--out-dir` is not recognised).
- [ ] **Step 3: Implement** the changes listed under Files. User-token and service-token keysets stay disjoint: a `kid` is looked up in the service keyset first, then the user keyset, and the token type must match the keyset it was found in.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\integration -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\services\user-service\tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_rs256_jwks.py::test_service_token_stays_hs256_and_verifies_even_with_flag_off` still holds because no service keyring is configured there)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.11: require_mfa_for_roles and MFA in clinician subject scoping
**Covers:** MVP-344
**Severity:** blocker
**Wave:** 4
**Depends on:** WS-07.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: TOTP MFA is mandatory for CLINICIAN and ADMIN; no PHI, admin or acknowledgement API without mfa=true; optional for end users; the one-time acknowledgement link stays usable without login)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/middleware.py` (new `require_mfa_for_roles`; `can_access_subject`'s clinician branch also requires `principal.mfa is True`)
- Modify: `tests/solace_security/test_mvp_120_subject_scoping.py` (this workstream's own file from WS-07.8: `_p` builds principals with `mfa=True`, so the assigned-clinician cases keep modelling an MFA-verified clinician)
- Test: `tests/solace_security/test_mvp_344_require_mfa_for_roles.py`
**Interfaces:**
- Consumes: `AuthenticatedUser.mfa` (WS-07.2) populated from the `mfa` claim WS-08.16 mints; `can_access_subject` (WS-07.8).
- Produces (IF-04 W4): `require_mfa_for_roles(*roles: str) -> Callable[..., Awaitable[AuthenticatedUser]]`, a dependency on `get_current_user` that raises `HTTPException(403, detail={"code": "MFA_REQUIRED", "message": "Multi-factor authentication is required for this account"})` when the principal holds any of `roles` and `mfa` is not `True`; obo principals (role `user`) pass. Consumed in W4 phase B by WS-13 (acknowledge/resolve), WS-14 (admin routes and provisioning) and every clinician PHI path through `can_access_subject`. This is the supporting piece of MVP-344; WS-08 closes it.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_344_require_mfa_for_roles.py::test_mfa_is_required_for_privileged_roles`
```python
"""MVP-344 (supporting; WS-08 closes it): CLINICIAN/ADMIN tokens without mfa=true are refused (D-08)."""
from __future__ import annotations

import time
from uuid import uuid4

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager, TokenType
from solace_security.middleware import AuthenticatedUser, can_access_subject, require_mfa_for_roles
from solace_security.service_auth import ServiceAuthSettings, ServiceTokenManager

SECRET = "test-secret-key-32-bytes-long!!!"


def _token(roles, mfa) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(uuid4()), "type": "access", "jti": str(uuid4()), "iat": now, "exp": now + 600,
                       "iss": "solace-ai", "aud": "solace-ai-api", "roles": roles, "mfa": mfa,
                       "amr": ["pwd", "otp"] if mfa else ["pwd"]}, SECRET, algorithm="HS256")


@pytest.fixture
def client(monkeypatch):
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    app = FastAPI()

    @app.get("/phi")
    async def phi(user: AuthenticatedUser = Depends(require_mfa_for_roles("clinician", "admin"))):
        return {"user": user.user_id}

    return TestClient(app)


@pytest.mark.parametrize("roles,mfa,status", [(["clinician"], False, 403), (["admin"], False, 403),
                                              (["clinician"], True, 200), (["user"], False, 200)])
def test_mfa_is_required_for_privileged_roles(client, roles, mfa, status):
    response = client.get("/phi", headers={"Authorization": f"Bearer {_token(roles, mfa)}"})
    assert response.status_code == status
    if status == 403:
        assert response.json()["detail"]["code"] == "MFA_REQUIRED"


def test_obo_principal_is_not_blocked(client):
    manager = ServiceTokenManager(AuthSettings(secret_key=SECRET),
                                  ServiceAuthSettings(service_name="orchestrator-service"))
    obo = manager.create_service_token("orchestrator-service", on_behalf_of=str(uuid4())).token
    assert client.get("/phi", headers={"Authorization": f"Bearer {obo}"}).status_code == 200


class _AssignedChecker:
    async def is_assigned(self, clinician_id, subject_user_id):
        return True

    async def acknowledged_recent_episode(self, clinician_id, subject_user_id, *, within_hours=72):
        return True


@pytest.mark.asyncio
async def test_assigned_clinician_without_mfa_cannot_reach_the_subject():
    clinician = AuthenticatedUser(user_id=str(uuid4()), token_type=TokenType.ACCESS, roles=["clinician"], mfa=False)
    assert await can_access_subject(clinician, uuid4(), checker=_AssignedChecker()) is False
    verified = clinician.model_copy(update={"mfa": True})
    assert await can_access_subject(verified, uuid4(), checker=_AssignedChecker()) is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_344_require_mfa_for_roles.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'require_mfa_for_roles'`; with a stub, the scoping test returns `True` for the clinician without MFA (W3 behaviour), which is the password-only clinician path the register describes.
- [ ] **Step 3: Implement** `require_mfa_for_roles(*roles: str)` and the `mfa` condition in `can_access_subject` in `src/solace_security/middleware.py`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q`. Expected: all pass, including `test_mvp_120_subject_scoping.py` with its principals now built with `mfa=True`
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-07.12: The access token's `loc` claim reaches AuthenticatedUser.locale unchanged (D-29)
**Covers:** none (supports D-29: the user's locale is available at crisis-screen time with no lookup)
**Supports:** MVP-317 (the IF-17 claim surfacing of WS-07.2, extended by one claim)
**Severity:** high
**Wave:** 3
**Depends on:** WS-07.2, WS-07.4
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** D-29 (decided 2026-10-19: several countries, US first. The one locale contract is the access-token claim `loc`, issued by WS-08.21 from the profile locale (default `en-US`, `und` when the stored value cannot be normalized). The shared middleware carries it unchanged; `solace_clinical.resolve_locale` (WS-12.2, called by WS-10.5) owns all defaulting and validation, so a missing claim means en-US there, never here)
**Real infra:** no
**Files:**
- Modify: `src/solace_security/auth.py` (`TokenPayload.loc: str | None = None`, read from the `loc` claim; nothing else changes, and service tokens never set it)
- Modify: `src/solace_security/middleware.py` (`AuthenticatedUser.locale: str | None = None`; `from_payload` maps `payload.loc` unchanged, with no defaulting or validation; the obo path of `get_current_user` (WS-07.4) builds the principal from the service token, so `locale` is `None` there)
- Create: `tests/solace_security/test_mvp_317_locale_claim.py`
- Test: `tests/solace_security/test_mvp_317_locale_claim.py`
**Interfaces:**
- Consumes: the `loc` claim (IF-17 addition, WS-08.21); `AuthenticatedUser.from_payload` and the obo acceptance in `get_current_user` (WS-07.2, WS-07.4).
- Produces: `AuthenticatedUser.locale: str | None` (IF-04 addition, additive under the `middleware.py` shared-file protocol) and `TokenPayload.loc: str | None`, on REST through `get_current_user` and on the WebSocket path through `from_payload`; obo service tokens yield `None`. This lands WS-11's request for WS-11.21 and WS-15's R15 for WS-15.23; both depend on this task. WS-08.21 (the issuer) needs no dependency either way: it mints the claim in user-service, and this task reads whatever a verified token carries.

- [ ] **Step 1: Write the failing test** in `tests/solace_security/test_mvp_317_locale_claim.py`
```python
"""D-29: the access token's `loc` claim reaches AuthenticatedUser.locale unchanged; service tokens carry none."""
from __future__ import annotations

import time

import jwt
import pytest
from fastapi.security import HTTPAuthorizationCredentials

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager, TokenPayload, TokenType

SECRET = "test-secret-key-32-bytes-long!!!"


class _FakeURL:
    path = "/api/v1/x"


class _FakeRequest:
    url = _FakeURL()


def _manager() -> JWTManager:
    return JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())


def _sign(**claims) -> str:
    now = int(time.time())
    base = {"sub": "u-1", "jti": "j-1", "iat": now, "exp": now + 600, "iss": "solace-ai", "aud": "solace-ai-api",
            "type": "access"}
    base.update(claims)
    return jwt.encode({k: v for k, v in base.items() if v is not None}, SECRET, algorithm="HS256")


@pytest.mark.asyncio
@pytest.mark.parametrize("loc", ["en-US", "fr-FR", "und", "not-a-locale"])
async def test_d29_loc_claim_reaches_the_principal_unchanged(monkeypatch, loc):
    manager = _manager()
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    user = await mw.get_current_user(_FakeRequest(), HTTPAuthorizationCredentials(scheme="Bearer", credentials=_sign(loc=loc)))
    assert user.locale == loc  # no defaulting or validation here: solace_clinical.resolve_locale owns the policy


@pytest.mark.asyncio
async def test_d29_missing_loc_claim_is_none_not_a_default(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    user = await mw.get_current_user(_FakeRequest(), HTTPAuthorizationCredentials(scheme="Bearer", credentials=_sign()))
    assert user.locale is None


def test_d29_from_payload_maps_loc_for_the_websocket_path():
    payload = TokenPayload(sub="u-1", type=TokenType.ACCESS, exp=int(time.time()) + 600, loc="pt-BR")
    assert mw.AuthenticatedUser.from_payload(payload).locale == "pt-BR"
    assert mw.AuthenticatedUser(user_id="u-1", token_type=TokenType.ACCESS).locale is None


def test_d29_service_tokens_carry_no_locale():
    manager = _manager()
    result = manager.decode_token_sync(manager.create_service_token("orchestrator-service"))
    assert result.success is True and result.payload.loc is None
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security\test_mvp_317_locale_claim.py -o "addopts=" -p no:logging -q`
Expected: FAIL. The REST and WebSocket tests fail with `AttributeError: 'AuthenticatedUser' object has no attribute 'locale'`, and the service-token test with `AttributeError: 'TokenPayload' object has no attribute 'loc'`: `TokenPayload` ignores unknown claims (pydantic's default `extra="ignore"`), so the `loc` claim WS-08.21 mints is dropped at verification, and the orchestrator could only learn the user's locale through a lookup at screen time, which the crisis invariant forbids (D-29).
- [ ] **Step 3: Implement** the `TokenPayload.loc` field and the `AuthenticatedUser.locale` field with its `from_payload` mapping. Add nothing that validates, normalizes or defaults the value.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\solace_security -o "addopts=" -p no:logging -q; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest D:\Repo\Contextual-Chatbot\tests\integration -o "addopts=" -p no:logging -q`. Expected: all pass (`test_mvp_317_malformed_claims_and_identity_claims.py` included)
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-07.12:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Owner WS | Exact change | Files | Needed by |
|---|---|---|---|
| WS-04 | W1 phase A2: in `start_service_runtime`, call `initialize_service_auth(service_settings=ServiceAuthSettings(service_name=spec.service_name))` for every service, and `configure_token_blacklist(build_token_blacklist(runtime.redis, strict=is_strict_environment(spec.environment)))` so revocation uses the runtime's Redis and fails at boot in strict; in `create_service_app`, call `install_rate_limiting(app, spec)` when `spec.rate_limited`, before CORS. `ServiceRuntime.outbound_headers()` should reuse `solace_security.service_auth.correlation_headers()` (contextvar keys `request_id`, `correlation_id`, `turn_id`) | `src/solace_infrastructure/runtime/bootstrap.py`, `src/solace_infrastructure/runtime/http.py` | WS-07.5, WS-07.7 |
| WS-04 | W1 phase A2: `ProductionGuards` requires `REDIS_URL` for every service in strict environments (the shared blacklist needs it even where `uses_redis` is false) | `src/solace_security/production_guards.py` | WS-07.5 |
| WS-04 | W1: alert rules `RateLimitBackendErrors` (`increase(solace_rate_limit_backend_errors_total[5m]) > 0`) and `TokenRevocationCheckErrors` (`increase(solace_token_revocation_check_errors_total[5m]) > 0`), promtool-tested and routed like the other W1 rules | `infrastructure/prometheus/rules/solace.yml` | WS-07.5, WS-07.7 |
| WS-03 | W1 phase A env sync: `REDIS_URL` (secret, required in strict) on all nine app services; `TRUSTED_PROXY_HOPS` on orchestrator and user-service (1 in strict); new optional non-secret `RATE_LIMIT_OVERRIDES_JSON` (no default; only the test overlay sets it). W4: `AUTH_ALGORITHM=RS256` and `AUTH_ACCEPT_LEGACY_HS256=false` in strict; `AUTH_PRIVATE_KEY_PEM` only on user-service; `SERVICE_TOKEN_PRIVATE_KEY_PEM` per container from `/run/secrets` (each service its own file, rendered by `deploy/render_secrets.py` from `deploy/secrets/<env>.sops.yaml`, into which the operator encrypts the files that `scripts/generate_jwt_keys.py --out-dir` writes to a temporary owner-only directory, D-07); `AUTH_PUBLIC_KEYS_JSON` and `SERVICE_TOKEN_PUBLIC_KEYS_JSON` on all services; `docs/STAGING-BRINGUP.md` stops telling operators to put AUTH_* keys in the shared `.env.prod`. Planned: the W1 part in WS-03.6's W1 sync, the STAGING-BRINGUP part in WS-03.8, the W4 part in WS-03.14, which depends on WS-07.10 (it documents and uses the `--out-dir` CLI), so WS-07.10 takes no reverse dependency | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml`, `docs/STAGING-BRINGUP.md` | WS-07.5, WS-07.7, WS-07.10 |
| WS-05 | W1: `docker-compose.test.yml` sets `RATE_LIMIT_OVERRIDES_JSON` (for example `{"auth-login": 500, "auth-register": 500, "auth-refresh": 500}`) so `make_user` loops in e2e suites do not hit the 20/min and 10/min auth buckets that all test traffic shares behind Caddy | `docker-compose.test.yml` | WS-07.7 |
| WS-12 | W1 phase B (supporting MVP-314): `/check`, `/detect-crisis`, `/assess`, `/filter-output` use `require_service_permission(ServicePermission.READ_SAFETY.value)`, `/escalate` uses `require_service_permission(ServicePermission.TRIGGER_ESCALATION.value)`; when `service.on_behalf_of` is set, the body `user_id` must equal it (403 otherwise); test overrides grant those permission values | `services/safety_service/src/api.py`, `services/safety_service/tests/test_api.py` | WS-07.9 |
| WS-13 | W1 phase B (supporting MVP-314, already in IF-09): `/send` and `/email` use `require_service_permission(ServicePermission.WRITE_NOTIFICATION.value)` | `services/notification-service/src/api.py` | WS-07.9 |
| WS-13 | W1 slot 015: `crisis_alerts` keeps the IF-09 columns `user_id`, `acknowledged_by` (the acknowledging clinician's user id, or NULL for an anonymous link acknowledgement) and `acknowledged_at`, readable by solace_app | `migrations/versions/015_notification.py` | WS-07.8 |
| WS-14 | W1 phase B (supporting MVP-314, already in IF-08): internal roster and care-team routes use `require_service_permission(ServicePermission.READ_ONCALL_ROSTER.value)` | `services/user-service/src/internal_api.py` | WS-07.9 |
| WS-18 | W2 phase B (supporting MVP-314): personality `/detect` and `/profile/update` use `require_service_permission(ServicePermission.WRITE_PERSONALITY.value)`, `/style`, `/adapt` and `/status` use `READ_PERSONALITY`, and the body `user_id` must equal `service.on_behalf_of` when set; analytics `/events/ingest` uses `require_service_permission(ServicePermission.WRITE_ANALYTICS.value)`, which no service holds, or is removed | `services/personality_service/src/api.py`, `services/analytics-service/src/api.py` and their tests | WS-07.9 |
| WS-15, WS-16, WS-17 | W2 phase B (supporting MVP-314): the `GET /status` routes use `require_service_permission` with `READ_THERAPY`, `READ_DIAGNOSIS` and `READ_MEMORY` respectively | `services/therapy_service/src/api.py`, `services/diagnosis_service/src/api.py`, `services/memory_service/src/api.py` | WS-07.9 |
| WS-14, WS-15, WS-16, WS-17 | W3 phase B (supporting MVP-120 and MVP-141): every handler that lets a clinician act on another user's data calls `await authorize_subject(principal, subject_user_id)` before any database work (diagnosis api.py:55-243, memory api.py:55-214, user-service `GET /users/{user_id}`, therapy clinician paths), replacing the role-only bypass | `services/diagnosis_service/src/api.py`, `services/memory_service/src/api.py`, `services/user-service/src/api.py`, `services/therapy_service/src/api.py` | MVP-120 closure at the W3 isolation e2e (WS-05) |

## New interfaces

Beyond the skeleton's IF-04, this plan adds:
- `AuthenticatedUser.amr: list[str]`, `.mfa: bool`, `.cns: dict | None` and `TokenPayload.session_id` read from the `sid` claim (IF-17 claims surfaced to every service).
- `AuthenticatedUser.locale: str | None` and `TokenPayload.loc: str | None` (D-29, WS-07.12): the access token's `loc` claim, unchanged and never defaulted here; `None` for obo service principals and tokens without the claim.
- `AuthenticatedService.on_behalf_of: str | None`.
- `find_unguarded_service_routes(app: FastAPI) -> list[str]` and the marker attribute `__solace_service_permissions__` on `require_service_permission` dependencies.
- `resolve_service_permissions(service_name: str) -> list[str]`, `correlation_headers() -> dict[str, str]`, `ServiceTokenManager.service_name`, `OBO_TOKEN_TTL_SECONDS = 300`, `JWTManager.create_service_token(..., *, extra_metadata=)`.
- `ServiceIdentity.ANALYTICS = "analytics-service"`, `ServicePermission.WRITE_ANALYTICS = "service:write:analytics"` (held by no service) and the extra grant of `WRITE_PERSONALITY` to orchestrator-service (needed for `/detect` under receiver-side checks).
- `TokenBlacklist.revoke_session(session_id, until)`, `.is_session_revoked(session_id)`, `.is_session_revoked_sync(session_id)`; Redis key `token_blacklist:sid:<session_id>`; `build_token_blacklist(redis_client=None, *, strict=None)`, `configure_token_blacklist(blacklist)`; metric `solace_token_revocation_check_errors_total`.
- `resolve_trusted_proxy_hops(environ=None)`, `classify_route(path)`, `AUTH_ROUTE_LIMITS`, `build_rate_limiter(service_name, *, redis_client)`, and the keyword arguments `force` and `redis_client` on `install_rate_limiting`; env `RATE_LIMIT_OVERRIDES_JSON`; 503 body code `RATE_LIMIT_UNAVAILABLE`; `RateLimitMiddleware(auth_route_predicate=...)`.
- `RateLimitStore(max_entries=100_000, sweep_every=1000)`.
- `can_access_subject` is `async` (the skeleton lists it as returning bool; it reads the database); `authorize_subject(principal, subject_user_id, *, checker=None)`; `SubjectAccessChecker`, `PostgresSubjectAccessChecker(pool=None)`, `CLINICIAN_ACK_ACCESS_HOURS = 72`.
- `ServiceKeySettings` (env prefix `SERVICE_TOKEN_`), `ServiceKeyring.from_env(service_name)`; `scripts/generate_jwt_keys.py --out-dir DIR --services all|<names> [--rotate]`.

## Self-review

Coverage (register id to tasks):

| Register id | Severity | Tasks |
|---|---|---|
| MVP-101 | high | WS-07.7 |
| MVP-120 | medium | WS-07.8 (primitive), W3 phase B adoption by WS-14/15/16/17 (requested), proven by the WS-05 W3 isolation matrix |
| MVP-141 | medium | WS-07.8 |
| MVP-185 | medium | WS-07.7 |
| MVP-186 | medium | WS-07.5 |
| MVP-188 | medium | WS-07.5 |
| MVP-209 | medium | WS-07.10 (library), WS-08.19 (user-service issuer and verifier) |
| MVP-311 | low | WS-07.6 |
| MVP-314 | low | WS-07.3, WS-07.4, WS-07.9 (acceptance) |
| MVP-317 | low | WS-07.2 |
| MVP-034 (WS-14, supporting) | blocker | WS-07.1 |
| MVP-021 (WS-11, supporting) | blocker | WS-07.3, WS-07.4 |
| MVP-344 (WS-08, supporting) | blocker | WS-07.11 |

Interface consistency:
- IF-04 names are used verbatim: `ServiceTokenManager.create_service_token(..., *, on_behalf_of=)`, `service_auth_headers(target_service, *, on_behalf_of=None)`, `forwarded_auth_headers(request)`, `ACT_FOR_USER='service:act_for_user'`, `READ_ONCALL_ROSTER='service:read:oncall'`, `WRITE_NOTIFICATION` granted to user-service, `get_current_user`/`get_current_service`/`require_service_permission`, `AuthenticatedUser.via_service`, `can_access_subject`, `set_subject_context`, `require_mfa_for_roles`, `install_rate_limiting(app, spec)`, `acting_as_user(user_id)`, `TRUSTED_PROXY_HOPS`, `REDIS_URL`, `solace_rate_limit_backend_errors_total`, `AUTH_ACCEPT_LEGACY_HS256`, `SERVICE_TOKEN_PRIVATE_KEY_PEM`, `SERVICE_TOKEN_PUBLIC_KEYS_JSON`.
- With WS-08: the session-revocation key (`token_blacklist:sid:<sid>`) and `TokenBlacklist.revoke_session` are what WS-08.4 writes; WS-08.4 re-exports the shared blacklist classes so both sides use one implementation; the IF-17 claim names `sid`, `amr`, `mfa`, `cns` match what WS-08 mints; WS-08's `reauth`, `mfa` and `mfa_enroll` token types are rejected by the shared middleware (WS-07.2); WS-08.11 uses `client_ip(request)` with the env-resolved hops (WS-07.7); WS-08.16 relies on `require_mfa_for_roles` and the MFA condition in `can_access_subject` (WS-07.11); WS-08.19 relies on the strict RS256 rules (WS-07.10).
- Phase rules: every WS-07 task is phase A (platform paths). No task depends on a same-wave phase-B task; WS-07.9 runs in W4 so all supporting route edits (W1 to W3) are behind a barrier. WS-07.8's dependencies WS-13.3 (W1 slot 015) and WS-05.2 (W0 fixtures) are earlier-wave phase-A tasks.
- Shared test file: WS-07.3 edits one existing case in `tests/integration/test_service_integration.py` (`TestPermissionMatrix::test_user_service_has_no_permissions`, lines 173-176), which asserts `SERVICE_PERMISSIONS` from `src/solace_security/service_auth.py`. The shared_files protocol for `tests/integration` lets the owner of the code under test edit an existing file, and WS-07 owns `service_auth.py`. No other plan edits that file (WS-01.8 only stops CI ignoring it).

Open risks:
- The revocation check fails open on a Redis error (D-26 by analogy, recorded as an owner Ruling). During a Redis outage a token revoked before the outage can be replayed for at most its remaining lifetime (15 minutes for user tokens); the counter and the WS-04 alert make the window visible. A fail-closed policy would take the crisis path down with Redis.
- In W1-W3 every service still shares AUTH_SECRET_KEY, so any compromised container can mint any service or user token until WS-07.10 and WS-08.19 land in W4.
- `RATE_LIMIT_OVERRIDES_JSON` can loosen limits in any environment; only the test overlay should set it. A strict-environment guard on it is a possible WS-04 follow-up.
- The real-infra subject checker test fills unknown NOT NULL columns generically. The known `crisis_alerts` CHECK columns (`max_level`, `status`) are passed explicitly (R-13-6); a CHECK constraint added later by WS-13 or WS-14 on a column the test does not pass would surface as `CheckViolationError` and needs an explicit value in the test.
- The obo TTL bound relies on reasonably synchronised clocks; the check allows the configured 30 s skew.

Review focus (numbered 1-5 in skeleton order):
- 1 (crisis false negatives) and 2 (alert-path failures) are outside this workstream's code; WS-07 adds no detection or paging logic.
- 3 (foreign-but-valid identifiers and context-less work): WS-07.1 pins that `acting_as_user` sets the RLS user for request-less work and always restores it, on exit, on exception and per concurrent task, so a pooled worker never keeps another user's id. WS-07.8 pins that an unassigned clinician is denied, that an obo principal reaches only its own user, and that the RLS user becomes the data owner only after the check passes; its real-infra test runs the checker's SQL as `solace_app` on the migrated schema.
- 4 (PHI in unplanned sinks): WS-07.2 pins that a malformed signed token gives a 401 with fixed detail text, never pydantic's error text with the claim values. WS-07.5 additionally pins that the Redis URL (a secret) never reaches a log.
- 5 (strict boots with partial configuration): WS-07.5 (strict environment without `REDIS_URL` fails loud; the in-memory override is refused in strict), WS-07.7 (strict rate limiting requires Redis) and WS-07.10 (strict refuses HS256 and the legacy flag, the keyset is required in staging as well as production, and HS256 service tokens are refused once service keys are configured).

REVIEW_FOCUS: 3=WS-07.1,WS-07.8; 4=WS-07.2; 5=WS-07.5,WS-07.7,WS-07.10

## Review responses

- Error: WS-07.11 modifies `tests/solace_security/test_mvp_120_subject_scoping.py`, which no earlier-wave task or dependency creates: fixed. WS-07.8 (W3, a declared dependency of WS-07.11) creates it, but its Files listed it as `Test:`, which the checker does not count as a creation; WS-07.8 now lists both of its test files as `Create:`.
- Error: plan a task for WS-13's request R-13-6 (the `crisis_alerts` CHECK constraints; the generic filler must pass `max_level='HIGH'`): fixed. It joins WS-07.8: the real-infra test already passes `max_level="HIGH"` and `status="acknowledged"` on both `crisis_alerts` inserts, and WS-07.8's Produces now names R-13-6 and WS-13.3, its Depends on adds WS-13.3 (slot 015, W1) and WS-05.2 (the `pg_migrated` fixture, W0), and Step 2 says that a `CheckViolationError` is a test defect, not the RED. Verified against WS-13.3's DDL: `max_level varchar(16) NOT NULL CHECK IN ('HIGH','CRITICAL')` has no default and `status` defaults to `'open'`.
- Warning: WS-07.3 edits `tests/integration/test_service_integration.py`, owned by WS-05: rejected: the shared_files protocol for `tests/integration` says an existing file is edited only by the owner of the code it tests. The one edited case, `TestPermissionMatrix::test_user_service_has_no_permissions` (`tests/integration/test_service_integration.py:173-176`), asserts `SERVICE_PERMISSIONS[ServiceIdentity.USER]` from `src/solace_security/service_auth.py`, which WS-07 owns, and no other plan edits the file. Recorded in the Self-review.
- Added the missing REVIEW_FOCUS line. To pin failure mode 4 in WS-07's own code, WS-07.2's 401 test now asserts the fixed detail and that no claim value or validation text reaches the body.
- decision 2026-10-19 integration: new task WS-07.12 (W3 phase A, depends on WS-07.2 and WS-07.4) adds `TokenPayload.loc` and `AuthenticatedUser.locale` for D-29, landing WS-11's request (WS-11.21) and WS-15's R15 (WS-15.23), which now depend on it; the claim is carried unchanged and `solace_clinical.resolve_locale` owns the policy. New interfaces lists the field.
- decision 2026-10-19 integration: the WS-03 request row now records where each part is planned (W1 in WS-03.6, STAGING-BRINGUP in WS-03.8, W4 in WS-03.14); WS-03.14 depends on WS-07.10 for the `--out-dir` CLI, so WS-07.10 takes no reverse dependency.
- decision 2026-10-19 D-07: WS-07.10's gate now names SOPS + age: `/run/secrets` files are rendered by `deploy/render_secrets.py` from `deploy/secrets/<env>.sops.yaml`, and no cloud secrets service is involved. The WS-03 request row now says the per-service `SERVICE_TOKEN_PRIVATE_KEY_PEM` files are rendered from the SOPS file, into which the operator encrypts the output of `scripts/generate_jwt_keys.py --out-dir`; they are not written into `/run/secrets` directly. The `NAME_FILE` consumer contract, WS-07.10's code and its tests are unchanged.
