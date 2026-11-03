# WS-14: User accounts, clinician roster, consent and erasure fan-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** User accounts persist correctly as `solace_app` under RLS. Clinicians are provisioned only by an audited CLI onto a service-only roster. Verification email works with expiring, hashed tokens. User PII is encrypted with a blind index. Consent gates chat. Account deletion erases the user in every service through a retry-safe, transactional fan-out that reports completion.

**Architecture:** user-service adopts the IF-01 runtime in W0. Its identity code (routes, dependencies, JWT/session/OAuth composition) is split verbatim into `api_auth.py`, `api_deps.py` and `identity_wiring.py`, which pass to WS-08 at the W0 barrier. Account, roster, consent and erasure logic stays in WS-14's `domain/service.py`, the repositories, `api.py` and the new service-only `internal_api.py` (IF-08). WS-14 also owns two shared pieces: the platform helper `solace_infrastructure.gdpr.consumer` (IF-16), which every PHI service uses to erase a user on `user.deleted` and report `user.erasure.completed`, and slot `012_user_accounts`, which is amended in W1, W2 and W3.

**Covers:** 16 register entries (blocker 3, high 5, medium 4, low 4).

**Waves:**
- 0, phase B: WS-14.1 to WS-14.3.
- 1, phase A: WS-14.4. Phase B: WS-14.5 to WS-14.9.
- 2, phase A: WS-14.11. Phase B: WS-14.10, WS-14.12.
- 3, phase A: WS-14.13, WS-14.14. Phase B: WS-14.15 to WS-14.20, WS-14.22 (D-29 profile locale).
- 4, phase B: WS-14.21.

**Depends on:**
- WS-01: WS-01.7 import-root guard (events.py case `-k WS-14`). The user-service image must carry `scripts/` (request R-14-1).
- WS-02: WS-02.3 creates the slot scaffold `012_user_accounts` and the NULLIF policies. WS-02.6 provides the one `default` pool with legacy aliases.
- WS-04: WS-04.4/WS-04.5 runtime and conformance suite (W0). WS-04.6 bootstrap wiring of service tokens, event plane and rate limiting (W1). WS-04.8 codec warm-up (W2). WS-04.10 audit wiring (W3).
- WS-05: WS-05.2 IF-19 `pg_migrated` fixture (W0). WS-05.10 REST contract framework (W2). WS-05.14 erasure event contracts (W3).
- WS-06: WS-06.2 bare bridge imports (W0). WS-06.8 `ServiceEventPlane` with `publish(conn=)` (W1). WS-06.11 `erase_user_events` and `UserErasureCompletedKafkaEvent` (W3).
- WS-07: WS-07.1 `acting_as_user` (W0). WS-07.3 service tokens and `service_auth_headers` (W1). WS-07.4 `require_service_permission` (W1). WS-07.7 `install_rate_limiting` (W1). WS-07.8 `authorize_subject` (W3). WS-07.11 `require_mfa_for_roles` (W4).
- WS-09: WS-09.7/WS-09.8 PHI codec and key rotation (W2). WS-09.11/WS-09.12 audit runtime and `audit_phi_access` (W3).

**Owned paths:**
- user-service: `services/user-service/src/{main,config,api,internal_api,events,__init__}.py`, `services/user-service/src/domain/{service,consent,erasure,value_objects,__init__}.py`, `services/user-service/src/infrastructure/{postgres_repository,repository,encryption_service,__init__}.py`, `services/user-service/README.md`, `services/user-service/tests` (steward of the shared test dir).
- Platform and migration: `src/solace_infrastructure/gdpr`, `migrations/versions/012_user_accounts.py`.
- Scripts and docs: `scripts/provision_clinician.py`, `docs/runbooks/clinician-onboarding.md`.
- Created in W0 and handed to WS-08 at the W0 barrier: `services/user-service/src/api_auth.py`, `api_deps.py`, `identity_wiring.py`.
- New real-infra tests: `tests/realinfra/accounts-roster-consent-erasure/`. New platform helper tests: `tests/solace_infrastructure/test_mvp_*.py`.

**Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

**Conventions every task uses** (repeated in each task's commands, so each task stands alone):
- Unit command: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest <path> -o "addopts=" -p no:logging -q`.
- Real-infra command: the same with `$env:SOLACE_REQUIRE_REALINFRA="1";` added before the python call.
- User-service tests import the service as `src.*` (`services/user-service/tests/conftest.py` puts the service root first on `sys.path`) and in-memory doubles from `fixtures`.
- Real-infra tests under `tests/realinfra/accounts-roster-consent-erasure/` load the service as the alias package `user_service_src` through that directory's `conftest.py`. This avoids colliding with the repository-root `src` package and with notification-service. They are marked `pytestmark = [pytest.mark.realinfra, pytest.mark.integration]`.
- Real-infra tests never select users by email in SQL, because email is ciphertext from W2. They select by id or count rows.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/user-service/src/main.py` | Modify | IF-01 SPEC and lifespan; composition `build_user_service_state`; routers; event-plane subscriptions | 14.1, 14.2, 14.3, 14.6, 14.7, 14.10, 14.16 |
| `services/user-service/src/config.py` | Modify | Drop USER_DB_*/USER_REDIS_* and the unused field key; `env` accepts `test` | 14.2, 14.10 |
| `services/user-service/src/events.py` | Modify | Bare `solace_events` import; `user.created` without contact data | 14.2, 14.16 |
| `services/user-service/src/api_auth.py` | Create (verbatim move, then WS-08's) | `/auth/*`, `POST /users/me/password`, `DELETE /users/me`, JWKS, aggregated OAuth router | 14.3 |
| `services/user-service/src/api_deps.py` | Create (verbatim move, then WS-08's) | Dependency providers, local `get_current_user`, `UserResponse`, `user_to_response`, `TokenResponse` | 14.3 |
| `services/user-service/src/identity_wiring.py` | Create (verbatim move, then WS-08's) | `start_identity(runtime, app)`, `stop_identity()` | 14.3 |
| `services/user-service/src/internal_api.py` | Create, then Modify | IF-08 service-only roster and care-team routes | 14.3, 14.8, 14.18 |
| `services/user-service/src/api.py` | Modify | Profile (including the D-29 locale), preferences, consent, account setup, user lookup and admin routes | 14.3, 14.5, 14.8, 14.9, 14.16, 14.17, 14.18, 14.19, 14.20, 14.21, 14.22 |
| `services/user-service/src/domain/service.py` | Modify | UserService: registration, verification, roster, provisioning, erasure, consent | 14.1, 14.5, 14.6, 14.8, 14.9, 14.15, 14.16, 14.17, 14.18, 14.19 |
| `services/user-service/src/domain/erasure.py` | Modify | Local erasure stores, expected services, erasure status types | 14.15, 14.16 |
| `services/user-service/src/domain/consent.py` | Modify | Consent policy: required set, document version, `evaluate_consent_status` | 14.17 |
| `services/user-service/src/domain/value_objects.py` | Modify | `ClinicianContact`, `CareTeam`, new consent types, `REQUIRED_CONSENTS` | 14.8, 14.17 |
| `services/user-service/src/domain/__init__.py` | Modify | Exports after ConsentService removal | 14.17 |
| `services/user-service/src/__init__.py` | Modify | Usage docstring without ConsentService | 14.17 |
| `services/user-service/src/infrastructure/repository.py` | Modify | Abstract repositories and the factory | 14.1, 14.5, 14.6, 14.8, 14.16 |
| `services/user-service/src/infrastructure/postgres_repository.py` | Modify | SQL that matches the migrated schema, transactions, codec, tokens, roster, erasure requests | 14.1, 14.5, 14.6, 14.8, 14.10, 14.12, 14.16 |
| `services/user-service/README.md` | Modify | Encryption claims match the code | 14.10 |
| `src/solace_infrastructure/gdpr/consumer.py` | Create | IF-16 `register_user_erasure_consumer` | 14.13 |
| `migrations/versions/012_user_accounts.py` | Modify (scaffold by WS-02.3) | W1 roster/tokens, W2 PII widths/blind index/consent version, W3 erasure tables | 14.4, 14.11, 14.14 |
| `scripts/provision_clinician.py` | Create, then Modify | IF-08 provisioning CLI | 14.9, 14.18 |
| `docs/runbooks/clinician-onboarding.md` | Create | Operator runbook for provisioning and on-call | 14.9 |
| `services/user-service/tests/conftest.py` | Modify | Test env: drop the USER_DB override (W0), a loopback notification URL so tests never wait on DNS (W1), codec keys (W2) | 14.2, 14.6, 14.10 |
| `services/user-service/tests/test_service.py` | Modify | Verification tests read the issued token from the token repository | 14.6 |
| `services/user-service/tests/fixtures.py` | Modify | In-memory doubles for the new repository methods | 14.5, 14.6, 14.8, 14.16 |
| `services/user-service/tests/runtime_fakes.py` | Create | `fake_runtime(...)` test double (not collected) | 14.2 |
| `services/user-service/tests/test_config.py` | Modify | Remove DB/Redis config tests | 14.2 |
| `services/user-service/tests/test_api.py`, `test_oauth_endpoints.py`, `test_auth_timing_health.py`, `test_refresh_rotation.py`, `test_guc_context.py` | Modify | Import lines after the verbatim split; the on-call routing test (14.8) | 14.3, 14.8 |
| `services/user-service/tests/test_resend_verification.py` | Modify | Imports after the split; hashed tokens | 14.3, 14.6 |
| `services/user-service/tests/test_rate_limiting.py` | Modify (delete) | Retired local limiter | 14.7 |
| `services/user-service/tests/test_gdpr_erasure.py` | Modify | New erasure store order | 14.15 |
| `services/user-service/tests/test_consent.py` | Modify (delete) | Retired ConsentService | 14.17 |
| `services/user-service/tests/test_mvp_*.py` | Create | One file per task (named in each task) | all phase-B tasks |
| `services/user-service/tests/test_mvp_221_profile_locale.py` | Create | Profile locale canonical, default en-US, malformed is 422 (D-29) | 14.22 |
| `tests/realinfra/accounts-roster-consent-erasure/conftest.py` | Create, then Modify | Alias loader, `cpm_default`, `superuser_conn`, `seed_user`, `build_repo`; autouse codec keys from W2 | 14.1, 14.10 |
| `tests/realinfra/accounts-roster-consent-erasure/test_mvp_*.py` | Create | Real-infra proofs (named in each task) | 14.1, 14.4, 14.5, 14.6, 14.8, 14.9, 14.10, 14.11, 14.12, 14.14, 14.15, 14.16 |
| `tests/solace_infrastructure/test_mvp_029_erasure_consumer.py` | Create | IF-16 helper | 14.13 |

## Tasks

### Task WS-14.1: Registration writes satisfy the migrated schema through the default pool and under the new user's RLS context

**Covers:** MVP-034
**Severity:** blocker
**Wave:** 0
**Depends on:** WS-02.3, WS-02.6, WS-05.2, WS-07.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the schema of record; repository SQL changes to match it). D-02 (default assumed: the service connects only as `solace_app`, NOBYPASSRLS).
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`:
  - The three repositories take `__init__(self, schema: str = "public", *, pool_name: str = "default")`.
  - `_acquire()` always returns `ConnectionPoolManager.acquire(self._pool_name)`. The `FeatureFlags` branch and the legacy-client path go.
  - Module constant `PLAINTEXT_KEY_ID = "none"`.
  - users INSERT supplies `encryption_key_id`.
  - `user_preferences` INSERT supplies `id` (`uuid4()`). `notification_channels` is bound as a Python list in save and update (no `json.dumps`).
  - consent INSERT supplies `created_at`/`updated_at` (both `consent.granted_at`) and no longer binds the str version to the integer `version` lock column.
  - assignment INSERT supplies `id`.
- Modify: `services/user-service/src/infrastructure/repository.py`. `RepositoryFactory(config: RepositoryConfig | None = None)` builds the Postgres repositories when `config.use_postgres` and raises `RepositoryError` otherwise. The `postgres_client` parameter goes.
- Modify: `services/user-service/src/domain/service.py`. `create_user` and the default-create branch of `update_preferences` save preferences inside `acting_as_user(user_id)`.
- Modify: `services/user-service/src/main.py`. Delete the `PostgresClient` block and its `disconnect()`; build `RepositoryFactory(RepositoryConfig(use_postgres=True))`. This is R-02.8: one pool per service.
- Create: `tests/realinfra/accounts-roster-consent-erasure/conftest.py`, with:
  - `UserServicePackage.mod(dotted)`, which imports `user_service_src.<dotted>` after loading the user-service source directory (read only, never edited here) as the alias package `user_service_src` with `importlib.util.spec_from_file_location(..., submodule_search_locations=[...])`.
  - Fixture `us`.
  - Fixture `superuser_conn`: an asyncpg connection to `pg_migrated.dsn_superuser`.
  - Fixture `cpm_default`: sets `POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE` from `urlsplit(pg_migrated.dsn_app)`, plus `POSTGRES_SSL_MODE=disable` and `ENVIRONMENT=test`; calls `ConnectionPoolManager.close_all_pools()`, then `get_pool("default")`, and closes all pools on teardown.
  - Fixture `seed_user`: an async factory `seed_user(**columns) -> UUID` that inserts as superuser with `id, created_at, updated_at, encryption_key_id='none', email, password_hash, display_name` plus any overrides.
  - Helper `build_repo(cls, **kw)`: returns `cls(**kw)`. On `TypeError` it returns `cls(_LegacyClient(), **kw)`, where `_LegacyClient.acquire()` returns `ConnectionPoolManager.acquire("default")`. This lets the RED run reach today's SQL.
- Test: `services/user-service/tests/test_mvp_034_repository_pool.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_034_registration_schema.py` (create)

**Interfaces:**
- Consumes:
  - IF-03 `ConnectionPoolManager.acquire(pool_name="default")`, which sets `app.current_user_id` from the contextvar at acquire time.
  - `acting_as_user(user_id)` (WS-07.1).
  - IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)` (WS-05.2).
- Produces:
  - `PostgresUserRepository(schema="public", *, pool_name="default")`, and the same signature for the preferences and consent repositories.
  - `PLAINTEXT_KEY_ID`.
  - The slug conftest fixtures `us`, `superuser_conn`, `cpm_default`, `seed_user` and the helper `build_repo`, reused by every later WS-14 real-infra task.
  - Satisfies R-02.8 for WS-02.3/WS-02.6/WS-02.8 (registration preferences under `acting_as_user`; no `USER_DB_*` client).
  - Satisfies WS-06.13's JSONB request ahead of its W2 deadline.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_034_repository_pool.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_034_registration_schema.py`
```python
# services/user-service/tests/test_mvp_034_repository_pool.py
"""MVP-034 / R-02.8: repositories use the one default pool; registration writes preferences as the new user."""
from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from solace_common.request_context import get_current_user_id
from solace_infrastructure.database.connection_manager import ConnectionPoolManager
import src.infrastructure.postgres_repository as pg_repo
from src.domain.service import UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository


@pytest.mark.asyncio
async def test_mvp_034_repositories_acquire_from_the_default_pool(monkeypatch) -> None:
    seen: list[str] = []

    @asynccontextmanager
    async def _acquire(pool_name="default", timeout=10.0):
        seen.append(pool_name)
        yield object()

    monkeypatch.setattr(ConnectionPoolManager, "acquire", _acquire)
    for cls in (pg_repo.PostgresUserRepository, pg_repo.PostgresUserPreferencesRepository,
                pg_repo.PostgresConsentRepository):
        async with cls()._acquire():
            pass
    assert seen == ["default", "default", "default"]


@pytest.mark.asyncio
async def test_mvp_034_registration_saves_preferences_as_the_new_user() -> None:
    class _RecordingPrefs(InMemoryUserPreferencesRepository):
        def __init__(self) -> None:
            super().__init__()
            self.contexts: list[str | None] = []

        async def save(self, preferences):
            self.contexts.append(get_current_user_id())
            return await super().save(preferences)

    prefs = _RecordingPrefs()
    svc = UserService(user_repository=InMemoryUserRepository(), preferences_repository=prefs)
    result = await svc.create_user(email="ctx@example.com", password_hash="h", display_name="Ctx")
    assert result.success is True
    assert prefs.contexts == [str(result.user.user_id)]
    assert get_current_user_id() is None
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_034_registration_schema.py
"""MVP-034: signup, consent and assignment INSERTs succeed on the migrated schema as solace_app."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

from solace_common.request_context import acting_as_user
from conftest import build_repo

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _service(us):
    repo, svc = us.mod("infrastructure.postgres_repository"), us.mod("domain.service")
    return svc.UserService(
        user_repository=build_repo(repo.PostgresUserRepository),
        preferences_repository=build_repo(repo.PostgresUserPreferencesRepository),
        consent_repository=build_repo(repo.PostgresConsentRepository),
        integration_settings=svc.ServiceIntegrationSettings(notification_service_url="http://127.0.0.1:9"),
    )


async def test_mvp_034_registration_persists_user_and_preferences(us, cpm_default, superuser_conn) -> None:
    result = await _service(us).create_user(
        email=f"reg-{uuid4().hex[:8]}@test.invalid", password_hash="argon2-synthetic", display_name="Reg User")
    assert result.success is True, result.error_code
    uid = result.user.user_id
    row = await superuser_conn.fetchrow("SELECT encryption_key_id, role, status FROM users WHERE id = $1", uid)
    assert (row["encryption_key_id"], row["role"], row["status"]) == ("none", "user", "pending_verification")
    prefs = await superuser_conn.fetchrow(
        "SELECT id, jsonb_typeof(notification_channels) AS kind FROM user_preferences WHERE user_id = $1", uid)
    assert prefs is not None and prefs["id"] is not None and prefs["kind"] == "array"


async def test_mvp_034_consent_and_assignment_inserts_satisfy_the_schema(us, cpm_default, superuser_conn) -> None:
    repo, ent, vo = (us.mod("infrastructure.postgres_repository"), us.mod("domain.entities"),
                     us.mod("domain.value_objects"))
    users = build_repo(repo.PostgresUserRepository)
    clinician = await users.save(ent.User(email=f"c-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                          display_name="Clin", role=vo.UserRole.CLINICIAN))
    patient = await users.save(ent.User(email=f"p-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                        display_name="Pat"))
    with acting_as_user(patient.user_id):
        await build_repo(repo.PostgresConsentRepository).save(vo.ConsentRecord(
            consent_id=uuid4(), user_id=patient.user_id, consent_type=vo.ConsentType.TERMS_OF_SERVICE,
            granted=True, version="1.0"))
    assert await users.assign_patient_to_clinician(clinician.user_id, patient.user_id) is True
    assert await superuser_conn.fetchval(
        "SELECT count(*) FROM consent_records WHERE user_id = $1", patient.user_id) == 1
    assert await superuser_conn.fetchval(
        "SELECT count(*) FROM clinician_patient_assignments WHERE clinician_id = $1 AND patient_id = $2",
        clinician.user_id, patient.user_id) == 1


async def test_mvp_034_guard_preferences_without_user_context_are_rejected(us, cpm_default) -> None:
    """Guard (passes before and after): the fix relies on acting_as_user, not on a policy hole."""
    repo, ent = us.mod("infrastructure.postgres_repository"), us.mod("domain.entities")
    user = await build_repo(repo.PostgresUserRepository).save(
        ent.User(email=f"g-{uuid4().hex[:8]}@test.invalid", password_hash="x", display_name="G"))
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        await build_repo(repo.PostgresUserPreferencesRepository).save(ent.UserPreferences(user_id=user.user_id))
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_034_repository_pool.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_034_registration_schema.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- Unit file:
  - `TypeError: __init__() missing 1 required positional argument: 'client'`. Repositories cannot be built without the per-service `USER_DB_*` client (R-02.8).
  - `contexts == [None]`. Registration writes `user_preferences` with no RLS user.
- Real-infra file:
  - `result.success is False` with `CREATE_FAILED`. The users INSERT omits `encryption_key_id` (NOT NULL, no default).
  - The consent save raises `NotNullViolationError` on `created_at`, or `DataError` for `'1.0'` bound to the integer `version`.
  - The assignment INSERT raises `NotNullViolationError` on `id`.
  - The guard test passes.
- [ ] **Step 3: Implement** the edits listed under Files. Every repository method keeps its name and return type. Preferences SQL binds `preferences.notification_channels` as a list, relying on the pool's JSONB codec (IF-03). `_row_to_entity` keeps decoding a `str` value for rows written before the fix.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass (`test_repository.py::TestRepositoryFactory` still raises without `use_postgres`).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.2: Adopt the IF-01 runtime: fail-loud datastores, one Redis client, no /status, docs off in strict

**Covers:** MVP-107, MVP-328
**Severity:** high
**Wave:** 0
**Depends on:** WS-14.1, WS-04.4, WS-04.5, WS-06.2, WS-01.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: strict means ENVIRONMENT in {staging, production}; the pilot runs production)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/main.py`:
  - Add module-level `SPEC = ServiceRuntimeSpec(service_name="user-service", environment=ServiceConfig().env, log_level=ServiceConfig().log_level, uses_postgres=True, uses_redis=True, event_plane="publish_consume", handles_phi=True, audited=True, rate_limited=True, edge=True)`.
  - Add `async def build_user_service_state(runtime, app) -> ServiceState`. The token blacklist, session manager and OAuth state store all use `runtime.redis`. When `runtime.redis is None`: in strict, raise `StartupDependencyError("user-service identity stores require REDIS_URL")`; otherwise log one WARNING `identity_redis_unavailable` and use in-memory stores. Repositories come from `RepositoryFactory(RepositoryConfig(use_postgres=True))` and `UserService` is built as today.
  - Lifespan:
    - first statement `runtime = await start_service_runtime(app, SPEC)`;
    - then `build_user_service_state`;
    - then an unguarded `initialize_event_bridge(postgres_pool=runtime.pg_pool)` with `runtime.register_shutdown("user_event_bridge", bridge.stop)`;
    - last statement `await stop_service_runtime(runtime)`.
  - Add `jwks_router = APIRouter()` holding the existing `/.well-known/jwks.json` handler.
  - Add `ROUTERS = [(api_router, "/api/v1"), (oauth_router, "/api/v1"), (jwks_router, "")]`.
  - Add `def create_app() -> FastAPI`: `create_service_app(SPEC, lifespan=lifespan, routers=ROUTERS, title="Solace-AI User Service")`, then `_register_rate_limiting(app, os.environ.get("REDIS_URL", "redis://localhost:6379/0"))`. Then `app = create_app()`.
  - `_register_rate_limiting` keeps its name and signature. Its fail-loud flag uses `is_strict_environment(SPEC.environment)`.
  - Remove `structlog.configure`, the `ProductionGuards` call, the PHI-processor block, `CORSMiddleware`, all four `@app.exception_handler`s, `/health`, `/ready` and `/status`.
- Modify: `services/user-service/src/config.py`:
  - Delete `DatabaseConfig`, `RedisConfig`, the `database` and `redis` fields of `UserServiceSettings`, and the database-password check in `validate_nested_secrets`.
  - `ServiceConfig.env` becomes `Literal["development", "test", "staging", "production"]`.
- Modify: `services/user-service/src/events.py` (`to_kafka_event` imports `from solace_events.schemas import ...`; R-01.7).
- Modify: `services/user-service/tests/conftest.py` (remove the `USER_DB_PASSWORD` override).
- Modify: `services/user-service/tests/test_config.py` (delete `TestDatabaseConfig` and `TestRedisConfig`; `TestUserServiceSettings` drops `USER_DB_PASSWORD`; record the test-count drop in the ledger).
- Create: `services/user-service/tests/runtime_fakes.py`:
  - `fake_runtime(*, redis=None, pg_pool=None, event_plane=None, phi_codec=None, environment="test") -> SimpleNamespace`, with `spec.environment`, `spec.service_name="user-service"`, and dicts `readiness_checks`, `shutdowns`, `background_tasks`;
  - `register_readiness_check(name, check, *, critical=True)`, `register_shutdown(name, closer)`, `register_background_task(name, task)`;
  - `outbound_headers() -> {}`.
- Test: `services/user-service/tests/test_mvp_107_runtime_adoption.py` (create)

**Interfaces:**
- Consumes: IF-01 `ServiceRuntimeSpec`, `create_service_app`, `start_service_runtime`, `stop_service_runtime`, `is_strict_environment`, `StartupDependencyError`, `ServiceRuntime.redis/pg_pool/register_shutdown` (WS-04.4); `initialize_event_bridge(kafka_settings=None, use_mock=False, postgres_pool=None)` (WS-06.2, names stable).
- Produces:
  - `src.main.SPEC`, `src.main.ROUTERS: list[tuple[APIRouter, str]]`, `async def build_user_service_state(runtime, app: FastAPI) -> ServiceState`, `def create_app() -> FastAPI`;
  - `runtime_fakes.fake_runtime`.
  - Satisfies WS-04.5 (`tests/bootstrap/test_service_conformance.py -k user-service`), WS-01.7 (`tests/build -k WS-14`) and the R-02.8 part of main.py.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_107_runtime_adoption.py`
```python
"""MVP-107 / MVP-328: user-service runs on the IF-01 runtime; strict boots never fall back to in-memory identity stores."""
from __future__ import annotations

import dataclasses
import pathlib
from datetime import datetime, timezone
from uuid import uuid4

import fakeredis.aioredis
import pytest
import structlog
from fastapi import FastAPI

from solace_infrastructure.runtime import StartupDependencyError, create_service_app
import src.main as main
from src.config import ServiceConfig, UserServiceSettings
from src.events import UserDeletedEvent, to_kafka_event
from runtime_fakes import fake_runtime

MAIN_SRC = pathlib.Path(main.__file__).read_text(encoding="utf-8")


def test_mvp_107_spec_declares_every_platform_dependency() -> None:
    spec = main.SPEC
    assert spec.service_name == "user-service" and spec.event_plane == "publish_consume"
    assert (spec.uses_postgres, spec.uses_redis, spec.handles_phi, spec.audited, spec.rate_limited, spec.edge) == (
        True, True, True, True, True, True)
    assert spec.docs_enabled is None


def test_mvp_107_main_builds_no_private_datastore_clients_or_handlers() -> None:
    for needle in ("PostgresClient", "aioredis.from_url", "ProductionGuards", "structlog.configure(",
                   "exception_handler", "USER_DB_"):
        assert needle not in MAIN_SRC, needle


def test_mvp_107_settings_no_longer_carry_private_datastores() -> None:
    assert not {"database", "redis"} & set(UserServiceSettings.model_fields)
    assert ServiceConfig(env="test").env == "test"


@pytest.mark.asyncio
async def test_mvp_107_strict_boot_without_redis_is_fatal() -> None:
    with pytest.raises(StartupDependencyError):
        await main.build_user_service_state(fake_runtime(redis=None, environment="production"), FastAPI())


@pytest.mark.asyncio
async def test_mvp_107_one_runtime_redis_client_backs_blacklist_and_sessions() -> None:
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    state = await main.build_user_service_state(fake_runtime(redis=client), FastAPI())
    assert state.jwt_service._blacklist._redis is client
    assert state.session_manager._redis is client


@pytest.mark.asyncio
async def test_mvp_107_dev_without_redis_warns_and_uses_memory() -> None:
    with structlog.testing.capture_logs() as logs:
        state = await main.build_user_service_state(fake_runtime(redis=None, environment="development"), FastAPI())
    assert type(state.session_manager).__name__ == "SessionManager"
    assert any(e["event"] == "identity_redis_unavailable" and e["log_level"] == "warning" for e in logs)


def test_mvp_328_status_is_gone_and_docs_follow_the_runtime() -> None:
    paths = {getattr(r, "path", None) for r in main.app.routes}
    assert "/status" not in paths
    assert {"/live", "/ready", "/health", "/metrics", "/.well-known/jwks.json"} <= paths
    strict = create_service_app(dataclasses.replace(main.SPEC, environment="production"),
                                lifespan=main.lifespan, routers=main.ROUTERS, title="probe")
    assert not {"/docs", "/redoc", "/openapi.json", "/status"} & {getattr(r, "path", None) for r in strict.routes}


def test_mvp_107_domain_events_map_to_the_canonical_schema() -> None:
    from solace_events.schemas import UserDeletedKafkaEvent

    event = to_kafka_event(UserDeletedEvent(aggregate_id=uuid4(), deleted_at=datetime.now(timezone.utc),
                                            reason="user_request"))
    assert isinstance(event, UserDeletedKafkaEvent)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_107_runtime_adoption.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- `AttributeError: module 'src.main' has no attribute 'SPEC'`, and the same for `build_user_service_state` and `ROUTERS`.
- The source test finds `aioredis.from_url`, `ProductionGuards`, `structlog.configure(` and `exception_handler`. These are MVP-107's silent in-memory fallbacks and swallowed failures.
- `database`/`redis` are still settings fields.
- `/status` is served (MVP-328).
- `to_kafka_event` returns `None`: the `src.solace_events` import fails with the service root first on `sys.path`.
- [ ] **Step 3: Implement** the edits listed under Files. `ServiceState` keeps its fields and `stats` property. `create_app()` must build a fresh app each call, because `test_oauth_endpoints.py` calls it. The RS256 key-manager block moves into `build_user_service_state` unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite, the conformance case and the import-root case.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap/test_service_conformance.py -k user-service -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/build -k WS-14 -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.3: Verbatim split of identity code into api_auth, api_deps and identity_wiring

**Covers:** none (structural W0 split the skeleton assigns to WS-14; supports MVP-107; hands three files to WS-08)
**Severity:** high
**Wave:** 0
**Depends on:** WS-14.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** no

**Files:**
- Create: `services/user-service/src/api_deps.py`. Verbatim from api.py: `get_jwt_service`, `get_password_service`, `get_user_service`, `get_user_repository`, `get_session_manager`, `get_current_user`, `UserResponse`, `user_to_response` and `TokenResponse` (WS-08's R-1 places `TokenResponse` here, so the OAuth and MFA modules import it without importing `api_auth`, which itself includes `src.oauth.oauth_router`).
- Create: `services/user-service/src/api_auth.py`, verbatim, with handler names unchanged (`register`, `login`, `refresh_token`, `logout`, `revoke_token`, `verify_email`, `resend_verification`, `change_password`, `delete_current_user`, `jwks`):
  - From api.py: `_mask_email`, the timing equalizer, the resend throttle helpers and `_reset_resend_throttle`; `RegisterRequest`, `LoginRequest`, `ResendVerificationRequest`, `RefreshRequest`, `PasswordChangeRequest` (`TokenResponse` is imported from `.api_deps`); the routes `/auth/register`, `/auth/login`, `/auth/refresh`, `/auth/logout`, `/auth/revoke-token`, `/auth/verify-email`, `/auth/resend-verification`, `POST /users/me/password` and `DELETE /users/me`, all on an inner router mounted at `/api/v1`.
  - From main.py: `jwks_router` (created by WS-14.2).
  - The single export `auth_router = APIRouter()`, which includes the inner router, `src.oauth.oauth_router` at `/api/v1`, and the JWKS route at the root.
- Create: `services/user-service/src/identity_wiring.py`. Verbatim from `build_user_service_state`: the JWT key-manager, blacklist, password, token, session, auth-service and OAuth-state composition, each Redis-backed store built from `runtime.redis` as WS-14.2 left it, as `async def start_identity(runtime, app: FastAPI) -> None` (fills `app.state.service`) and `async def stop_identity() -> None`.
- Create: `services/user-service/src/internal_api.py` (`internal_router = APIRouter()`, empty until WS-14.8).
- Modify: `services/user-service/src/api.py`. Remove the moved symbols. Import the dependencies and `TokenResponse` from `.api_deps`, so `src.api.get_current_user` stays importable and line 36 of user-service's `oauth.py` (`from .api import TokenResponse`, a WS-08 file that this task does not edit) keeps resolving until WS-08 re-points it.
- Modify: `services/user-service/src/main.py`:
  - `ROUTERS = [(auth_router, ""), (api_router, "/api/v1"), (internal_router, "/api/v1/internal")]`;
  - `build_user_service_state` calls `await start_identity(runtime, app)`;
  - `runtime.register_shutdown("identity", stop_identity)`.
- Modify (import lines only, no assertion changes):
  - `test_api.py`: add `from src.api_auth import auth_router` and `app.include_router(auth_router)`.
  - `test_oauth_endpoints.py`: include `auth_router` in place of the separate `oauth_router` include.
  - `test_auth_timing_health.py` and `test_refresh_rotation.py`: import the moved names from `src.api_auth`, and `import src.api_auth as api` for the throttle helpers.
  - `test_resend_verification.py`: `from src.api_auth import ResendVerificationRequest, resend_verification`.
  - `test_guc_context.py`: `from src.api_deps import get_current_user`.
  - All of these are under `services/user-service/tests/`. WS-14 owns every moved symbol until the W0 barrier.
- Test: `services/user-service/tests/test_mvp_107_identity_split.py` (create)

**Interfaces:**
- Consumes: `build_user_service_state`, `ROUTERS` (WS-14.2).
- Produces:
  - `src.api_auth.auth_router` (one router; WS-08 later adds the MFA and OAuth-exchange routers to it, so main.py never changes for identity);
  - `src.api_deps` names as listed;
  - `src.identity_wiring.start_identity(runtime, app) -> None` and `stop_identity() -> None`;
  - `src.internal_api.internal_router`.
  - Ownership of `api_auth.py`, `api_deps.py` and `identity_wiring.py` passes to WS-08 at the W0 barrier (shared_files protocol).
  - This is the W0 phase-B split requested by WS-08 as R-1 (needed by every WS-08 task, from WS-08's first W2 task on): `auth_router` with handler names unchanged, `TokenResponse` in `api_deps`, `start_identity`/`stop_identity` built from `runtime.redis`, `auth_router` included once, existing tests re-pointed in this task.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_107_identity_split.py`
```python
"""W0 split (supports MVP-107): identity routes, dependencies and composition move verbatim; the route table is unchanged."""
from __future__ import annotations

import inspect
import pathlib

from fastapi.routing import APIRoute

AUTH = {("POST", "/api/v1/auth/register"), ("POST", "/api/v1/auth/login"), ("POST", "/api/v1/auth/refresh"),
        ("POST", "/api/v1/auth/logout"), ("POST", "/api/v1/auth/revoke-token"),
        ("POST", "/api/v1/auth/verify-email"), ("POST", "/api/v1/auth/resend-verification"),
        ("POST", "/api/v1/users/me/password"), ("DELETE", "/api/v1/users/me"), ("GET", "/.well-known/jwks.json"),
        ("GET", "/api/v1/auth/oauth/google/start"), ("GET", "/api/v1/auth/oauth/google/callback")}
PROFILE = {("GET", "/users/me"), ("PUT", "/users/me"), ("GET", "/users/me/preferences"),
           ("PUT", "/users/me/preferences"), ("GET", "/users/me/consent"), ("POST", "/users/me/consent"),
           ("GET", "/users/{user_id:uuid}"), ("GET", "/users/on-call-clinicians"),
           ("PUT", "/users/{user_id}/on-call")}


def _routes(routes) -> list[tuple[str, str]]:
    return [(m, r.path) for r in routes if isinstance(r, APIRoute) for m in r.methods]


def test_split_routers_own_their_routes() -> None:
    # Subset checks, so later additive tasks (WS-14.8/14.9 admin routes, WS-08 MFA routes) keep this green.
    from fastapi import APIRouter
    from src.api import router as api_router
    from src.api_auth import auth_router
    from src.internal_api import internal_router

    assert AUTH <= set(_routes(auth_router.routes))
    profile = set(_routes(api_router.routes))
    assert PROFILE <= profile
    assert not [r for r in profile if r[1].startswith("/auth/")]
    assert ("DELETE", "/users/me") not in profile and ("POST", "/users/me/password") not in profile
    assert isinstance(internal_router, APIRouter)


def test_split_app_serves_each_route_once() -> None:
    import src.main as main

    served = _routes(main.app.routes)
    expected = AUTH | {(m, "/api/v1" + p) for m, p in PROFILE}
    assert expected <= set(served) and len(served) == len(set(served))


def test_split_dependencies_and_identity_wiring_exist() -> None:
    import src.api as api
    import src.api_deps as deps
    import src.identity_wiring as wiring
    import src.main as main

    for name in ("get_jwt_service", "get_password_service", "get_user_service", "get_user_repository",
                 "get_session_manager", "get_current_user", "UserResponse", "user_to_response", "TokenResponse"):
        assert hasattr(deps, name), name
    assert api.get_current_user is deps.get_current_user
    assert api.TokenResponse is deps.TokenResponse  # src/oauth.py still imports it from .api
    assert inspect.iscoroutinefunction(wiring.start_identity) and inspect.iscoroutinefunction(wiring.stop_identity)
    text = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert "JWTService(" not in text and "RedisSessionManager(" not in text


def test_split_keeps_handler_names() -> None:
    from src.api_auth import auth_router

    names = {r.endpoint.__name__ for r in auth_router.routes if isinstance(r, APIRoute)}
    assert {"register", "login", "refresh_token", "logout", "revoke_token", "verify_email",
            "resend_verification", "change_password", "delete_current_user", "jwks"} <= names
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_107_identity_split.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.api_auth'` (and `src.api_deps`, `src.identity_wiring`, `src.internal_api`). Today identity and account code share one router and main.py composes the JWT, session and OAuth stores inline.
- [ ] **Step 3: Implement** the move. Move each function, model and constant without editing its body, apart from import paths. Run a `git diff --stat` comparison against the WS-14.2 checkpoint tree. The only body differences allowed are the import lines.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the conformance case.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap/test_service_conformance.py -k user-service -o "addopts=" -p no:logging -q
```
Expected: all pass, with the same test count as after WS-14.2.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.4: Slot 012 (W1 body): backup on-call flag and the account_tokens table

**Covers:** MVP-105, MVP-230
**Severity:** high
**Wave:** 1
**Depends on:** WS-02.3, WS-05.2
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-01 (default assumed: owners edit only `upgrade()`/`downgrade()`; slot bodies may be amended in place until the first real deployment). D-13 (default assumed: backup on-call clinicians exist).
**Real infra:** yes

**Files:**
- Modify: `migrations/versions/012_user_accounts.py`. Only `upgrade()` and `downgrade()` change.
  - `upgrade()`:
    - `op.add_column("users", sa.Column("is_backup", sa.Boolean(), nullable=False, server_default=sa.text("false")))`;
    - `op.create_table("account_tokens", token_hash String(64) PK, user_id UUID NOT NULL FK users.id ON DELETE CASCADE, purpose String(32) NOT NULL, created_at timestamptz NOT NULL server_default now(), expires_at timestamptz NOT NULL, used_at timestamptz NULL, CheckConstraint("purpose IN ('email_verification','account_setup')", name="ck_account_tokens_purpose"))`;
    - `op.create_index("ix_account_tokens_user_purpose", "account_tokens", ["user_id", "purpose"])`.
  - `downgrade()` reverses the three in reverse order.
  - No RLS: the table is read before authentication. Grants come from 009's default privileges.
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_slot012_accounts.py` (create)

**Interfaces:**
- Consumes: the slot-012 scaffold (revision `012_user_accounts`, down revision `011_audit_trail`) and 009 default privileges (WS-02.3); IF-19 `pg_migrated`; this directory's `superuser_conn` and `seed_user` (WS-14.1).
- Produces: `users.is_backup`, and table `account_tokens(token_hash, user_id, purpose, created_at, expires_at, used_at)` with purposes `email_verification` and `account_setup`. Consumed by WS-14.6, WS-14.8 and WS-14.9. The task runs in the W1 schema lane after slot 010 and before slots 014 and 015.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_slot012_accounts.py`
```python
"""Slot 012 (W1): users.is_backup and account_tokens exist, are usable by solace_app and check their purpose."""
from __future__ import annotations

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_slot012_adds_backup_flag_and_account_tokens(superuser_conn) -> None:
    users = {r["column_name"]: r for r in await superuser_conn.fetch(
        "SELECT column_name, is_nullable, column_default FROM information_schema.columns WHERE table_name = 'users'")}
    assert users["is_backup"]["is_nullable"] == "NO" and "false" in users["is_backup"]["column_default"]
    cols = {r["column_name"] for r in await superuser_conn.fetch(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'account_tokens'")}
    assert cols == {"token_hash", "user_id", "purpose", "created_at", "expires_at", "used_at"}
    assert await superuser_conn.fetchval("SELECT relrowsecurity FROM pg_class WHERE relname = 'account_tokens'") is False


async def test_slot012_solace_app_uses_account_tokens_and_purpose_is_checked(pg_migrated, seed_user) -> None:
    uid = await seed_user()
    conn = await asyncpg.connect(pg_migrated.dsn_app)
    try:
        await conn.execute("INSERT INTO account_tokens (token_hash, user_id, purpose, expires_at) "
                           "VALUES ($1, $2, 'account_setup', now() + interval '1 hour')", "a" * 64, uid)
        assert await conn.fetchval("UPDATE account_tokens SET used_at = now() WHERE token_hash = $1 "
                                   "AND used_at IS NULL RETURNING user_id", "a" * 64) == uid
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute("INSERT INTO account_tokens (token_hash, user_id, purpose, expires_at) "
                               "VALUES ($1, $2, 'password_reset', now())", "b" * 64, uid)
    finally:
        await conn.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_slot012_accounts.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `KeyError: 'is_backup'` and `UndefinedTableError: relation "account_tokens" does not exist`. There is no place to store a backup on-call flag (MVP-105) or an expiring, hashed one-time token (MVP-230).
- [ ] **Step 3: Implement** `upgrade()` and `downgrade()` as listed. Use `postgresql.UUID(as_uuid=True)`, as 001 does.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the chain test (upgrade head, downgrade base, upgrade head) and this directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.5: Transactional signup, fixed error messages and no PII or exception text in results or logs

**Covers:** MVP-106, MVP-327
**Severity:** high
**Wave:** 1
**Depends on:** WS-14.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/infrastructure/repository.py`. New abstract `UserRepository.save_with_preferences(user: User, preferences: UserPreferences, *, preferences_repository: UserPreferencesRepository) -> User`. In-memory implementations write the preferences through `preferences_repository`; SQL implementations write both rows in one transaction and ignore it.
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`:
  - Module constants `USER_INSERT_SQL` and `PREFERENCES_INSERT_SQL` (with a `{schema}` placeholder), used by `save`, the preferences `save` and the new `save_with_preferences`.
  - `save_with_preferences` runs both INSERTs in one `ConnectionPoolManager.transaction("default")`.
  - A unique violation becomes `DuplicateEntityError`.
- Modify: `services/user-service/src/domain/service.py`:
  - `create_user` calls `save_with_preferences` inside `acting_as_user(user.user_id)`.
  - Every `except Exception` in `create_user`, `update_user`, `delete_user`, `change_password`, `verify_email`, `update_preferences`, `record_consent` and `set_on_call_status` returns a fixed message (the `FIXED` map in the test) and logs `error_type=type(e).__name__` only.
  - No log call in the module passes `email=`.
- Modify: `services/user-service/src/api.py`. Routes raise `HTTPException(status, detail={"code": result.error_code, "message": <fixed text>})` instead of passing `result.error` through.
- Modify: `services/user-service/tests/fixtures.py` (`InMemoryUserRepository.save_with_preferences`: saves the user, then calls `preferences_repository.save`, and removes the user if that raises)
- Test: `services/user-service/tests/test_mvp_106_transactional_signup.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_106_signup_transaction_pg.py` (create)

**Interfaces:**
- Consumes: `acting_as_user` (WS-07.1); `ConnectionPoolManager.transaction("default")` (IF-03); slug fixtures (WS-14.1).
- Produces:
  - `UserRepository.save_with_preferences(...)`, `USER_INSERT_SQL` and `PREFERENCES_INSERT_SQL`.
  - The fixed result messages: "Registration failed", "Profile update failed", "Account deletion failed", "Password change failed", "Email verification failed", "Preferences update failed", "Consent could not be recorded", "On-call update failed".
  - WS-08's `api_auth.py` routes keep passing `result.error`, which is now always one of these.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_106_transactional_signup.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_106_signup_transaction_pg.py`
```python
# services/user-service/tests/test_mvp_106_transactional_signup.py
"""MVP-106 / MVP-327: a failed signup leaves no account; results and logs never carry exception text or email."""
from __future__ import annotations

from uuid import uuid4

import pytest
import structlog

from src.domain.service import UserService
from src.domain.value_objects import ConsentType
from src.infrastructure.password_service import PasswordConfig, PasswordService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository

SECRET_TEXT = "db-internal:5432 password=hunter2"
FIXED = {"update_user": "Profile update failed", "change_password": "Password change failed",
         "delete_user": "Account deletion failed", "update_preferences": "Preferences update failed",
         "record_consent": "Consent could not be recorded", "set_on_call_status": "On-call update failed",
         "verify_email": "Email verification failed"}


class _BrokenPrefs(InMemoryUserPreferencesRepository):
    async def save(self, preferences):
        raise RuntimeError('null value in column "id" of relation "user_preferences" violates not-null constraint')


class _Exploding:
    def __getattr__(self, name):
        async def _boom(*args, **kwargs):
            raise RuntimeError(f"connection to {SECRET_TEXT} failed")
        return _boom


@pytest.mark.asyncio
async def test_mvp_106_failed_preferences_leave_no_account_and_retry_succeeds() -> None:
    users = InMemoryUserRepository()
    failed = await UserService(user_repository=users, preferences_repository=_BrokenPrefs()).create_user(
        email="tx@example.com", password_hash="h", display_name="Tx")
    assert (failed.success, failed.error_code, failed.error) == (False, "CREATE_FAILED", "Registration failed")
    assert await users.get_by_email("tx@example.com") is None
    retry = await UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository()
                              ).create_user(email="tx@example.com", password_hash="h", display_name="Tx")
    assert retry.success is True


@pytest.mark.asyncio
@pytest.mark.parametrize("op", sorted(FIXED))
async def test_mvp_327_service_errors_never_echo_exception_text(op) -> None:
    svc = UserService(user_repository=_Exploding(), preferences_repository=_Exploding(),
                      consent_repository=_Exploding(), password_service=PasswordService(PasswordConfig()))
    uid = uuid4()
    calls = {"update_user": lambda: svc.update_user(uid, display_name="X"),
             "change_password": lambda: svc.change_password(uid, "Old1pass", "hash"),
             "delete_user": lambda: svc.delete_user(uid, reason="user_request"),
             "update_preferences": lambda: svc.update_preferences(uid, theme="dark"),
             "record_consent": lambda: svc.record_consent(uid, ConsentType.TERMS_OF_SERVICE, True, "1.0"),
             "set_on_call_status": lambda: svc.set_on_call_status(uid, True),
             "verify_email": lambda: svc.verify_email(uid, "token")}
    with structlog.testing.capture_logs() as logs:
        result = await calls[op]()
    assert result.success is False and result.error == FIXED[op]
    assert not [entry for entry in logs if "hunter2" in repr(entry) or "db-internal" in repr(entry)]


@pytest.mark.asyncio
async def test_mvp_327_signup_logs_carry_no_email_address() -> None:
    svc = UserService(user_repository=InMemoryUserRepository(), preferences_repository=InMemoryUserPreferencesRepository())
    with structlog.testing.capture_logs() as logs:
        await svc.create_user(email="phi-log@example.com", password_hash="h", display_name="Phi")
    assert not [entry for entry in logs if "phi-log@example.com" in repr(entry)]
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_106_signup_transaction_pg.py
"""MVP-106: users and user_preferences are written in one transaction on Postgres."""
from __future__ import annotations

from uuid import uuid4

import pytest

from conftest import build_repo

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_106_failed_preferences_insert_leaves_no_users_row(us, cpm_default, superuser_conn) -> None:
    repo, svc_mod = us.mod("infrastructure.postgres_repository"), us.mod("domain.service")
    svc = svc_mod.UserService(user_repository=build_repo(repo.PostgresUserRepository),
                              preferences_repository=build_repo(repo.PostgresUserPreferencesRepository),
                              integration_settings=svc_mod.ServiceIntegrationSettings(
                                  notification_service_url="http://127.0.0.1:9"))
    email = f"tx-{uuid4().hex[:8]}@test.invalid"
    before = await superuser_conn.fetchval("SELECT count(*) FROM users")
    await superuser_conn.execute("REVOKE INSERT ON user_preferences FROM solace_app")
    try:
        failed = await svc.create_user(email=email, password_hash="argon2-synthetic", display_name="Tx")
    finally:
        await superuser_conn.execute("GRANT INSERT ON user_preferences TO solace_app")
    assert failed.success is False and failed.error == "Registration failed"
    assert await superuser_conn.fetchval("SELECT count(*) FROM users") == before
    assert (await svc.create_user(email=email, password_hash="argon2-synthetic", display_name="Tx")).success is True
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_106_transactional_signup.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_106_signup_transaction_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- The orphan user survives (`get_by_email` returns it). The retry returns `EMAIL_EXISTS`. The error is `'Failed to create user: null value in column "id" ...'` (MVP-106).
- Each parametrized op returns `"Failed to ...: connection to db-internal:5432 password=hunter2 failed"` and logs `error=<that text>` (MVP-327).
- The signup log carries `email=phi-log@example.com`.
- On Postgres the users count grows by one after the failed signup: two independent connections.
- [ ] **Step 3: Implement** the edits listed under Files. `create_user` maps `DuplicateEntityError` to `EMAIL_EXISTS` (the existing behaviour) and every other exception to `CREATE_FAILED`. Log lines keep `user_id`. They drop `email`, or use the existing `_mask_email` style only where an operator needs it, which none of these do.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass. The only existing `detail` assertion, `test_api.py` line 189 "already registered", belongs to the register route in `api_auth.py`, which still passes the unchanged `EMAIL_EXISTS` message.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.6: Verification email reaches notification with a service token and a complete link; tokens are hashed, single-use and expiring

**Covers:** MVP-035, MVP-230
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-14.4, WS-14.5, WS-07.3, WS-04.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-14 (default assumed: the localhost link guard applies in staging and production)
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/domain/service.py`:
  - `ServiceIntegrationSettings` (gains `populate_by_name=True`):
    - `notification_service_url` reads `SERVICE_NOTIFICATION_SERVICE_URL`, default `http://notification-service:8003`;
    - `frontend_url` reads `USER_SERVICE_FRONTEND_URL`, default `http://localhost:3000`;
    - new `validate_for_environment(environment: str) -> None` raises `ValueError` naming `USER_SERVICE_FRONTEND_URL` when strict and the host is `localhost`, `127.0.0.1` or empty.
  - `UserService.__init__` gains `account_token_repository: AccountTokenRepository | None = None` (stored as `self._account_tokens`) and `http_client_factory: Callable[[], httpx.AsyncClient] | None = None`.
  - Token flow:
    - `create_user` and `resend_verification_email` issue `self._account_tokens.issue(user_id, "email_verification", timedelta(hours=self._verification_token_expiry_hours))` and set no `email_verification_token` on the entity. When `self._account_tokens is None` (unit tests only; main always passes the Postgres repository), they issue nothing, send nothing and log WARNING `verification_tokens_unavailable`.
    - `_send_verification_email(email, display_name, token, *, user_id)` posts to `{notification_service_url}/api/v1/notifications/email` with `await service_auth_headers("notification-service")` and the link `{frontend_url}/verify-email?user_id=<uuid>&token=<token>`.
    - `verify_email(user_id: UUID | None, token: str)`:
      - With a `user_id`: an unknown user is `INVALID_TOKEN` (no oracle); an already-verified user is `ALREADY_VERIFIED`; otherwise `consume(token, "email_verification", user_id=user_id)`.
      - Without one: `consume` first, then the user lookup.
      - `None` from `consume` is `INVALID_TOKEN`, with no expired/unknown distinction.
    - Remove `_get_service_auth_headers`'s use for notification.
- Modify: `services/user-service/src/infrastructure/repository.py`. New ABC `AccountTokenRepository`:
  - `issue(user_id: UUID, purpose: Literal["email_verification", "account_setup"], ttl: timedelta) -> str` returns the raw token (`secrets.token_urlsafe(32)`) and stores only `sha256(token).hexdigest()`;
  - `consume(token: str, purpose: str, *, user_id: UUID | None = None) -> UUID | None` is atomic and single-use; it refuses expired or used tokens and, when `user_id` is given, a token of another user, which it then leaves unconsumed;
  - `revoke(user_id, purpose) -> int`;
  - `delete_for_user(user_id) -> int`.
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`. New `PostgresAccountTokenRepository(*, pool_name="default")`. `consume` is one `UPDATE ... SET used_at = now() WHERE token_hash = $1 AND purpose = $2 AND used_at IS NULL AND expires_at > now() [AND user_id = $3] RETURNING user_id`.
- Modify: `services/user-service/src/main.py`. `build_user_service_state` passes `PostgresAccountTokenRepository()` and `ServiceIntegrationSettings()`, and calls `validate_for_environment(SPEC.environment)`. A strict `ValueError` becomes `StartupDependencyError`.
- Modify: `services/user-service/tests/fixtures.py`. New `InMemoryAccountTokenRepository(now: Callable[[], datetime] = lambda: datetime.now(timezone.utc))` with the same semantics, plus test helpers `rows()` (returning `{token_hash: row}`) and `issued: list[tuple[UUID, str, str]]` (user id, purpose, raw token).
- Modify: `services/user-service/tests/test_resend_verification.py`. `test_resend_by_email_regenerates_and_sends_for_unverified` asserts a new hashed token row in `InMemoryAccountTokenRepository` instead of `email_verification_token`. The fakes take the `user_id` keyword.
- Modify: `services/user-service/tests/test_service.py`:
  - The `user_service` fixture passes `account_token_repository=account_tokens`, a new `InMemoryAccountTokenRepository` fixture.
  - `test_create_user_success` asserts `email_verification_token is None` and one issued `email_verification` token.
  - `test_verify_email_success` and `test_verify_email_already_verified` take the raw token from `account_tokens.issued[-1][2]`.
- Modify: `services/user-service/tests/conftest.py` (`setdefault("SERVICE_NOTIFICATION_SERVICE_URL", "http://127.0.0.1:9")`, so suite runs never wait on DNS for `notification-service`)
- Test: `services/user-service/tests/test_mvp_035_verification_email.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_230_account_tokens_pg.py` (create)

**Interfaces:**
- Consumes:
  - IF-04 `service_auth_headers(target_service, *, on_behalf_of=None) -> dict[str, str]` (WS-07.3); user-service holds `service:write:notification`; the bootstrap initialises service auth (WS-04.6).
  - IF-09 `POST /api/v1/notifications/email {to_email, template_type, variables, priority}` with a service token (WS-13).
  - IF-02 `SERVICE_NOTIFICATION_SERVICE_URL`.
- Produces:
  - `AccountTokenRepository`, `PostgresAccountTokenRepository` and `InMemoryAccountTokenRepository`.
  - The verification link format `{USER_SERVICE_FRONTEND_URL}/verify-email?user_id=<uuid>&token=<token>`.
  - `UserService.verify_email(user_id: UUID | None, token: str)`, which WS-08's route calls (IF-17 `{token}` body).
  - `ServiceIntegrationSettings.validate_for_environment`.
  - New env var `USER_SERVICE_FRONTEND_URL` (R-14-3).

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_035_verification_email.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_230_account_tokens_pg.py`
```python
# services/user-service/tests/test_mvp_035_verification_email.py
"""MVP-035 / MVP-230: the verification email reaches notification-service with a service token and a usable,
expiring, single-use, hashed link."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import httpx
import pytest

import fixtures
import src.domain.service as service_module
from src.domain.service import ServiceIntegrationSettings, UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository

REAL_CLIENT = httpx.AsyncClient


def _service() -> UserService:
    svc = UserService(user_repository=InMemoryUserRepository(), preferences_repository=InMemoryUserPreferencesRepository())
    token_repo = getattr(fixtures, "InMemoryAccountTokenRepository", None)  # created by this task; absent on today's tree
    if token_repo is not None:
        svc._account_tokens = token_repo()
    return svc


@pytest.fixture
def outbox(monkeypatch):
    sent: list[httpx.Request] = []
    transport = httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(202, json={}))
    monkeypatch.setattr(service_module.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=transport, **kw))

    async def _headers(target: str, *, on_behalf_of=None) -> dict[str, str]:
        assert target == "notification-service" and on_behalf_of is None
        return {"Authorization": "Bearer svc-token", "X-Request-ID": "req-1"}

    monkeypatch.setattr(service_module, "service_auth_headers", _headers, raising=False)
    monkeypatch.setenv("SERVICE_NOTIFICATION_SERVICE_URL", "http://notification-service:8003")
    monkeypatch.setenv("USER_SERVICE_FRONTEND_URL", "https://app.example.test")
    return sent


@pytest.mark.asyncio
async def test_mvp_035_verification_email_reaches_notification_with_a_service_token(outbox) -> None:
    svc = _service()
    result = await svc.create_user(email="verify@example.com", password_hash="h", display_name="Ver")
    assert result.success is True
    [request] = outbox
    assert str(request.url) == "http://notification-service:8003/api/v1/notifications/email"
    assert request.headers["Authorization"] == "Bearer svc-token"
    body = json.loads(request.content)
    assert (body["to_email"], body["template_type"]) == ("verify@example.com", "email_verification")
    link = body["variables"]["verification_link"]
    query = parse_qs(urlsplit(link).query)
    assert link.startswith("https://app.example.test/verify-email?")
    assert query["user_id"] == [str(result.user.user_id)] and len(query["token"][0]) >= 43


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


@pytest.mark.asyncio
async def test_mvp_230_verification_token_is_hashed_single_use_and_expiring(outbox) -> None:
    from fixtures import InMemoryAccountTokenRepository

    clock, users = _Clock(), InMemoryUserRepository()
    tokens = InMemoryAccountTokenRepository(now=clock)
    svc = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository(),
                      account_token_repository=tokens)
    first = (await svc.create_user(email="one@example.com", password_hash="h", display_name="One")).user
    token = parse_qs(urlsplit(json.loads(outbox[-1].content)["variables"]["verification_link"]).query)["token"][0]
    assert list(tokens.rows()) == [hashlib.sha256(token.encode()).hexdigest()]
    assert (await users.get_by_id(first.user_id)).email_verification_token is None
    assert (await svc.verify_email(UUID(int=0), token)).error_code == "INVALID_TOKEN"  # unknown user, token kept
    assert (await svc.verify_email(None, token)).success is True
    assert (await svc.verify_email(first.user_id, token)).error_code == "ALREADY_VERIFIED"

    second = (await svc.create_user(email="two@example.com", password_hash="h", display_name="Two")).user
    late = parse_qs(urlsplit(json.loads(outbox[-1].content)["variables"]["verification_link"]).query)["token"][0]
    clock.now += timedelta(hours=25)
    assert (await svc.verify_email(second.user_id, late)).error_code == "INVALID_TOKEN"
    assert (await users.get_by_id(second.user_id)).email_verified is False


@pytest.mark.parametrize("env", ["staging", "production"])
def test_mvp_035_strict_rejects_a_localhost_link_base(monkeypatch, env) -> None:
    monkeypatch.setenv("USER_SERVICE_FRONTEND_URL", "http://localhost:3000")
    with pytest.raises(ValueError, match="USER_SERVICE_FRONTEND_URL"):
        ServiceIntegrationSettings().validate_for_environment(env)
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_230_account_tokens_pg.py
"""MVP-230: account tokens are stored hashed and consumed exactly once, never after expiry."""
from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta

import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_230_token_is_hashed_and_consumed_once_under_concurrency(us, cpm_default, superuser_conn, seed_user) -> None:
    uid = await seed_user()
    tokens = us.mod("infrastructure.postgres_repository").PostgresAccountTokenRepository()
    raw = await tokens.issue(uid, "email_verification", timedelta(hours=24))
    stored = await superuser_conn.fetchrow("SELECT token_hash FROM account_tokens WHERE user_id = $1", uid)
    assert stored["token_hash"] == hashlib.sha256(raw.encode()).hexdigest()
    results = await asyncio.gather(*(tokens.consume(raw, "email_verification") for _ in range(5)))
    assert results.count(uid) == 1 and results.count(None) == 4
    assert await tokens.consume(raw, "account_setup") is None


async def test_mvp_230_expired_token_is_refused(us, cpm_default, superuser_conn, seed_user) -> None:
    uid = await seed_user()
    tokens = us.mod("infrastructure.postgres_repository").PostgresAccountTokenRepository()
    raw = await tokens.issue(uid, "email_verification", timedelta(hours=24))
    await superuser_conn.execute("UPDATE account_tokens SET expires_at = now() - interval '1 second' WHERE user_id = $1", uid)
    assert await tokens.consume(raw, "email_verification") is None
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_035_verification_email.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_230_account_tokens_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- MVP-035:
  - The request goes to `http://localhost:8005/api/v1/notifications/email`, because `USER_SERVICE_NOTIFICATION_SERVICE_URL` defaults to localhost and `SERVICE_NOTIFICATION_SERVICE_URL` is ignored.
  - The `Authorization` header is absent or not the IF-04 service token.
  - The link is `http://localhost:3000/verify-email?token=...` with no `user_id`.
  - `validate_for_environment` does not exist.
- MVP-230:
  - `ImportError: InMemoryAccountTokenRepository`, then `TypeError` on `account_token_repository`.
  - Today the raw token sits on `users.email_verification_token` and never expires.
  - `PostgresAccountTokenRepository` is missing.
- [ ] **Step 3: Implement** the edits listed under Files. `resend_verification_email` revokes the user's previous `email_verification` tokens before issuing a new one. No log line carries the token, the link or the email address.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass. The E2E proof (mailpit link, then verify, then login) is WS-05's `make_user` once R-14-6 lands.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.7: Retire the private rate limiter once the runtime installs the shared one

**Covers:** MVP-107
**Severity:** high
**Wave:** 1
**Depends on:** WS-14.3, WS-04.6, WS-07.7
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-26 (default assumed: the shared Redis limiter keyed on user or real client IP is the only control; it fails closed for auth routes)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/main.py`. Delete `_rate_limiting_disabled` and `_register_rate_limiting`, and the call in `create_app()`. The runtime's `install_rate_limiting(app, SPEC)`, driven by `SPEC.rate_limited=True`, is the only limiter.
- Modify: `services/user-service/tests/test_rate_limiting.py` (delete the file; it tests the removed function; record the count drop in the ledger)
- Test: `services/user-service/tests/test_mvp_107_single_rate_limiter.py` (create)

**Interfaces:**
- Consumes: IF-01/IF-04 `create_service_app` installing `solace_security.rate_limit.install_rate_limiting(app, spec)` when `spec.rate_limited` (WS-04.6, WS-07.7).
- Produces: user-service builds no Redis client at import time and keys no limiter on the proxy IP.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_107_single_rate_limiter.py`
```python
"""MVP-107 (closure): user-service has exactly one rate limiter, the runtime's; no private Redis client at import."""
from __future__ import annotations

import pathlib


def test_mvp_107_no_private_limiter_code_remains() -> None:
    import src.main as main

    text = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert not hasattr(main, "_register_rate_limiting") and not hasattr(main, "_rate_limiting_disabled")
    assert "create_solace_rate_limiter" not in text and "RateLimitMiddleware" not in text


def test_mvp_107_development_app_has_exactly_one_limiter(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("USER_SERVICE_ENV", "development")
    import src.main as main

    limiters = [m for m in main.create_app().user_middleware if m.cls.__name__ == "RateLimitMiddleware"]
    assert len(limiters) == 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_107_single_rate_limiter.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `_register_rate_limiting` still exists. In development the app carries two `RateLimitMiddleware` layers: the runtime's, and the private one keyed on the proxy IP with its own Redis client.
- [ ] **Step 3: Implement** the deletions.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the conformance case.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/bootstrap/test_service_conformance.py -k user-service -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.8: Service-only roster and care-team API (IF-08); admin-only on-call and assignment management

**Covers:** MVP-103, MVP-105
**Severity:** high
**Wave:** 1
**Depends on:** WS-14.1, WS-14.4, WS-14.5, WS-07.3, WS-07.4
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:**
- D-13 (default assumed: ADMIN manages on-call and assignments; no self-service on-call);
- D-12/D-23 (default assumed: the care-team display name and contact fields are null unless the user's latest `crisis_contact` consent is granted).
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/internal_api.py`. Both routes depend on `require_service_permission(ServicePermission.READ_ONCALL_ROSTER.value)`:
  - `GET /clinicians/on-call` returns `{clinicians: [{user_id, display_name, email, phone_e164, role, is_backup}]}`;
  - `GET /users/{user_id}/care-team` returns `{user_id, display_name, contact_email, contact_phone_e164, crisis_contact_consent, clinicians}`, or 404 `{"code": "USER_NOT_FOUND"}`.
- Modify: `services/user-service/src/api.py`:
  - `GET /users/on-call-clinicians` and `PUT /users/{user_id}/on-call` depend on `require_roles("admin")` from `solace_security.middleware`.
  - New `PUT /admin/clinicians/{user_id}/on-call {on_call: bool, backup: bool}` returns 204, 404 `USER_NOT_FOUND` or 400 `INVALID_ROLE`.
  - New `POST /admin/assignments {clinician_id, user_id}` returns 201 `{clinician_id, user_id}` or 404 `NOT_FOUND`.
  - Both admin routes depend on `require_roles("admin")`.
- Modify: `services/user-service/src/domain/value_objects.py`:
  - `ConsentType.CRISIS_CONTACT = "crisis_contact"` (optional consent);
  - `@dataclass(frozen=True) ClinicianContact(user_id: UUID, display_name: str, email: str, phone_e164: str | None, role: str, is_backup: bool)`;
  - `@dataclass(frozen=True) CareTeam(user_id: UUID, display_name: str | None, contact_email: str | None, contact_phone_e164: str | None, crisis_contact_consent: bool, clinicians: tuple[ClinicianContact, ...])`.
- Modify: `services/user-service/src/domain/service.py`:
  - `get_on_call_roster() -> list[ClinicianContact]`;
  - `get_care_team(user_id: UUID) -> CareTeam | None`, which reads the `crisis_contact` consent inside `acting_as_user(user_id)`;
  - `set_on_call_status(user_id, is_on_call, *, backup: bool = False) -> UpdateUserResult`, which writes through `set_on_call_flags`.
- Modify: `services/user-service/src/infrastructure/repository.py`. New abstract methods:
  - `UserRepository.list_on_call_roster() -> list[ClinicianContact]`: ACTIVE, CLINICIAN, on call, not deleted; ordered with primaries first;
  - `UserRepository.list_care_team(patient_id: UUID) -> list[ClinicianContact]`: assigned, ACTIVE, CLINICIAN, not deleted;
  - `UserRepository.set_on_call_flags(user_id: UUID, *, on_call: bool, backup: bool) -> bool`;
  - `ConsentRepository.has_active_consent(user_id: UUID, consent_type: ConsentType) -> bool` (latest record granted).
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`. The four methods above in SQL. `has_active_consent` selects only `granted` (`ORDER BY granted_at DESC LIMIT 1`), so it does not depend on the W2 version mapping.
- Modify: `services/user-service/tests/fixtures.py` (in-memory versions; `InMemoryUserRepository` keeps a `_backup: set[UUID]`)
- Modify: `services/user-service/tests/test_api.py` (`TestOnCallCliniciansRouting::test_on_call_clinicians_route_not_shadowed` uses an admin token and also asserts a `user` token gets 403, never 422)
- Test: `services/user-service/tests/test_mvp_103_roster_api.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_103_roster_queries_pg.py` (create)

**Interfaces:**
- Consumes:
  - IF-04 `require_service_permission`, `get_current_service`, `require_roles`, `find_unguarded_service_routes` and `AuthenticatedService` (WS-07.4);
  - `ServicePermission.READ_ONCALL_ROSTER = "service:read:oncall"`, granted to notification-service (WS-07.3);
  - `acting_as_user` (WS-07.1).
- Produces:
  - IF-08 routes `GET /api/v1/internal/clinicians/on-call` and `GET /api/v1/internal/users/{user_id}/care-team`, with the exact field names above; `PUT /api/v1/admin/clinicians/{user_id}/on-call`; `POST /api/v1/admin/assignments`.
  - `ClinicianContact`, `CareTeam`, `ConsentType.CRISIS_CONTACT`.
  - Satisfies WS-07.9's request (internal routes guarded by `READ_ONCALL_ROSTER`).
  - Consumed by WS-13 (recipients) and WS-05 (`--assign` flows).

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_103_roster_api.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_103_roster_queries_pg.py`
```python
# services/user-service/tests/test_mvp_103_roster_api.py
"""MVP-103 / MVP-105: the roster is service-only, the care team honours crisis-contact consent, admins manage on-call."""
from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager
from solace_security.middleware import find_unguarded_service_routes
from solace_security.service_auth import ServiceAuthSettings, ServiceTokenManager
from src.api import router as api_router
from src.internal_api import internal_router
from src.domain.entities import User
from src.domain.service import UserService
from src.domain.value_objects import AccountStatus, ConsentRecord, ConsentType, UserRole
from fixtures import InMemoryConsentRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository

SECRET = "test-secret-key-32-bytes-long!!!"


def _user_token(roles: list[str]) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(uuid4()), "type": "access", "jti": str(uuid4()), "iat": now, "exp": now + 600,
                       "iss": "solace-ai", "aud": "solace-ai-api", "roles": roles, "mfa": True,
                       "amr": ["pwd", "otp"]}, SECRET, algorithm="HS256")


def _service_token(service: str) -> str:
    manager = ServiceTokenManager(AuthSettings(secret_key=SECRET), ServiceAuthSettings(service_name=service))
    return manager.create_service_token(service).token


@pytest.fixture
def world(monkeypatch):
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    users, consents = InMemoryUserRepository(), InMemoryConsentRepository()
    svc = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository(),
                      consent_repository=consents)
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(internal_router, prefix="/api/v1/internal")
    app.state.service = SimpleNamespace(user_service=svc, user_repository=users)
    return SimpleNamespace(app=app, client=TestClient(app), users=users, consents=consents, svc=svc)


async def _clinician(users, *, on_call=True) -> User:
    return await users.save(User(email=f"clin-{uuid4().hex[:6]}@test.invalid", password_hash="x", display_name="Dr On",
                                 role=UserRole.CLINICIAN, status=AccountStatus.ACTIVE, is_on_call=on_call,
                                 phone_number="+15555550100", email_verified=True))


async def test_mvp_103_roster_is_service_only_and_permission_checked(world) -> None:
    clinician = await _clinician(world.users)
    ok = world.client.get("/api/v1/internal/clinicians/on-call",
                          headers={"Authorization": f"Bearer {_service_token('notification-service')}"})
    assert ok.status_code == 200
    assert ok.json() == {"clinicians": [{"user_id": str(clinician.user_id), "display_name": "Dr On",
                                         "email": clinician.email, "phone_e164": "+15555550100",
                                         "role": "clinician", "is_backup": False}]}
    other = world.client.get("/api/v1/internal/clinicians/on-call",
                             headers={"Authorization": f"Bearer {_service_token('memory-service')}"})
    assert other.status_code == 403
    user = world.client.get("/api/v1/internal/clinicians/on-call",
                            headers={"Authorization": f"Bearer {_user_token(['user'])}"})
    assert user.status_code in (401, 403) and "clinicians" not in user.text
    assert find_unguarded_service_routes(world.app) == []


async def test_mvp_103_care_team_contact_fields_need_crisis_contact_consent(world) -> None:
    clinician = await _clinician(world.users)
    patient = await world.users.save(User(email="pat@test.invalid", password_hash="x", display_name="Pat",
                                          status=AccountStatus.ACTIVE, phone_number="+15555550199"))
    await world.users.assign_patient_to_clinician(clinician.user_id, patient.user_id)
    headers = {"Authorization": f"Bearer {_service_token('notification-service')}"}
    url = f"/api/v1/internal/users/{patient.user_id}/care-team"
    hidden = world.client.get(url, headers=headers).json()
    assert (hidden["crisis_contact_consent"], hidden["display_name"], hidden["contact_email"],
            hidden["contact_phone_e164"]) == (False, None, None, None)
    assert [c["user_id"] for c in hidden["clinicians"]] == [str(clinician.user_id)]
    await world.consents.save(ConsentRecord(consent_id=uuid4(), user_id=patient.user_id,
                                            consent_type=ConsentType.CRISIS_CONTACT, granted=True, version="1.0"))
    shown = world.client.get(url, headers=headers).json()
    assert (shown["crisis_contact_consent"], shown["display_name"], shown["contact_email"],
            shown["contact_phone_e164"]) == (True, "Pat", "pat@test.invalid", "+15555550199")
    assert world.client.get(f"/api/v1/internal/users/{uuid4()}/care-team", headers=headers).status_code == 404


async def test_mvp_103_public_on_call_routes_are_admin_only(world) -> None:
    clinician = await _clinician(world.users, on_call=False)
    user = {"Authorization": f"Bearer {_user_token(['user'])}"}
    admin = {"Authorization": f"Bearer {_user_token(['admin'])}"}
    assert world.client.get("/api/v1/users/on-call-clinicians", headers=user).status_code == 403
    assert world.client.get("/api/v1/users/on-call-clinicians", headers=admin).status_code == 200
    assert world.client.put(f"/api/v1/users/{clinician.user_id}/on-call", json={"is_on_call": True},
                            headers=user).status_code == 403


async def test_mvp_105_admin_sets_on_call_backup_and_assignments(world) -> None:
    clinician = await _clinician(world.users, on_call=False)
    patient = await world.users.save(User(email="p2@test.invalid", password_hash="x", display_name="P2"))
    admin = {"Authorization": f"Bearer {_user_token(['admin'])}"}
    r = world.client.put(f"/api/v1/admin/clinicians/{clinician.user_id}/on-call",
                         json={"on_call": True, "backup": True}, headers=admin)
    assert r.status_code == 204
    roster = await world.svc.get_on_call_roster()
    assert [(c.user_id, c.is_backup) for c in roster] == [(clinician.user_id, True)]
    r = world.client.post("/api/v1/admin/assignments",
                          json={"clinician_id": str(clinician.user_id), "user_id": str(patient.user_id)}, headers=admin)
    assert r.status_code == 201
    assert await world.users.is_patient_assigned_to_clinician(clinician.user_id, patient.user_id)
    not_clinician = world.client.put(f"/api/v1/admin/clinicians/{patient.user_id}/on-call",
                                     json={"on_call": True, "backup": False}, headers=admin)
    assert not_clinician.status_code == 400
    user = {"Authorization": f"Bearer {_user_token(['user'])}"}
    assert world.client.post("/api/v1/admin/assignments", json={"clinician_id": str(clinician.user_id),
                             "user_id": str(patient.user_id)}, headers=user).status_code == 403
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_103_roster_queries_pg.py
"""MVP-103 / MVP-105: roster and care-team queries are correct as solace_app, including RLS-protected consent."""
from __future__ import annotations

from uuid import uuid4

import pytest

from solace_common.request_context import acting_as_user, get_current_user_id

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _service(us):
    repo = us.mod("infrastructure.postgres_repository")
    return us.mod("domain.service").UserService(
        user_repository=repo.PostgresUserRepository(), preferences_repository=repo.PostgresUserPreferencesRepository(),
        consent_repository=repo.PostgresConsentRepository())


async def test_mvp_103_care_team_reads_consent_without_request_context(us, cpm_default, seed_user) -> None:
    vo = us.mod("domain.value_objects")
    clinician = await seed_user(role="clinician", status="active", is_on_call=True, email_verified=True,
                                phone_number="+15555550100")
    patient = await seed_user(status="active")
    svc = _service(us)
    await svc._user_repo.assign_patient_to_clinician(clinician, patient)
    with acting_as_user(patient):
        await svc._consent_repo.save(vo.ConsentRecord(consent_id=uuid4(), user_id=patient,
                                                      consent_type=vo.ConsentType.CRISIS_CONTACT, granted=True,
                                                      version="1.0"))
    assert get_current_user_id() is None
    team = await svc.get_care_team(patient)
    assert team.crisis_contact_consent is True and team.contact_email is not None
    assert [c.user_id for c in team.clinicians] == [clinician] and team.clinicians[0].phone_e164 == "+15555550100"


async def test_mvp_105_roster_lists_only_active_on_call_clinicians(us, cpm_default, seed_user, superuser_conn) -> None:
    on = await seed_user(role="clinician", status="active", is_on_call=True)
    off = await seed_user(role="clinician", status="active", is_on_call=False)
    gone = await seed_user(role="clinician", status="active", is_on_call=True)
    await superuser_conn.execute("UPDATE users SET deleted_at = now() WHERE id = $1", gone)
    plain = await seed_user(role="user", status="active", is_on_call=True)
    svc = _service(us)
    assert await svc._user_repo.set_on_call_flags(on, on_call=True, backup=True) is True
    roster = {c.user_id: c for c in await svc.get_on_call_roster()}
    assert on in roster and roster[on].is_backup is True
    assert not {off, gone, plain} & set(roster)
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_103_roster_api.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_103_roster_queries_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- `ConsentType.CRISIS_CONTACT` is missing (`AttributeError`).
- The internal routes are 404 (empty router).
- `GET /api/v1/users/on-call-clinicians` returns 200 to a plain `user` token, with every on-call clinician's email and phone (MVP-103).
- The admin routes are 404.
- `get_care_team`, `get_on_call_roster` and `set_on_call_flags` do not exist (MVP-105).
- [ ] **Step 3: Implement** the edits listed under Files. `get_care_team` returns `None` for an unknown or deleted user. When consent is absent, the display name and contact fields are `None` and the clinicians list is still returned. Responses never include clinicians who are not ACTIVE.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.9: Clinician provisioning CLI and the one-time account-setup link

**Covers:** MVP-105
**Severity:** high
**Wave:** 1
**Depends on:** WS-14.6, WS-14.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-13 (default assumed: only an operator creates CLINICIAN/ADMIN accounts with this CLI inside the user-service container; at least two on-call clinicians and the fallback mailbox before go-live)
**Real infra:** yes

**Files:**
- Create: `scripts/provision_clinician.py`, with:
  - `build_parser() -> argparse.ArgumentParser`, flags `--email`, `--display-name`, `--phone` (E.164 `^\+[1-9]\d{7,14}$`), `--on-call`, `--backup`, `--role {CLINICIAN,ADMIN}` (default CLINICIAN, case-insensitive), `--assign USER_ID`;
  - `async def provision(args, service: UserService, *, frontend_url: str) -> tuple[UUID, str]`;
  - `async def default_service_factory() -> tuple[UserService, Callable[[], Awaitable[None]]]`: `ConnectionPoolManager.get_pool("default")` from `POSTGRES_*`, the three Postgres repositories, `PostgresAccountTokenRepository`, `PasswordService(PasswordConfig())`; the closer closes all pools;
  - `def main(argv: Sequence[str] | None = None, *, service_factory=None, configure_logging: bool = False) -> int`, with exit codes 0 ok, 1 error, 2 usage, 3 account exists;
  - stdout is exactly `user_id=<uuid>` and `set_password_url=<frontend>/set-password?token=<token>`; logs go to stderr through the PHI sanitizer only when `configure_logging`;
  - `if __name__ == "__main__": raise SystemExit(main(configure_logging=True))`.
- Create: `docs/runbooks/clinician-onboarding.md`. Covers the provisioning command, the set-password link handling (72 h, single use), on-call/backup and assignment through the admin endpoints, MFA enrollment at first login from W4, and the go-live minimum of two on-call clinicians.
- Modify: `services/user-service/src/domain/service.py`:
  - `provision_privileged_account(*, email, display_name, phone_e164, role: UserRole, on_call=False, backup=False) -> ProvisionResult`. `@dataclass ProvisionResult(user_id: UUID, setup_token: str)`. Error codes `INVALID_ROLE` and `EMAIL_EXISTS`. It creates an ACTIVE, verified account with an unusable random password hash, saves it with `save_with_preferences` under `acting_as_user`, sets the flags and issues an `account_setup` token for 72 h.
  - `complete_account_setup(token: str, new_password_hash: str) -> PasswordChangeResult` (`INVALID_TOKEN` on failure).
- Modify: `services/user-service/src/api.py`. New `POST /users/account-setup {token, new_password}` returns 204, or 400 `{"code": "INVALID_TOKEN"}`. It is unauthenticated. The password policy is `PasswordPolicy(require_special=False)`. The route hashes with `get_password_service`.
- Test: `services/user-service/tests/test_mvp_105_provision_cli.py` (create)
- Create: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py` (real-infra test; WS-14.10 later adds the `ENCRYPTION_*` variables to its `_env`)

**Interfaces:**
- Consumes: `AccountTokenRepository` (WS-14.6); `save_with_preferences` (WS-14.5); `set_on_call_flags`, `assign_patient` (WS-14.8); `USER_SERVICE_FRONTEND_URL` (R-14-3).
- Produces:
  - IF-08 CLI `python -m scripts.provision_clinician --email E --display-name N --phone +1XXXXXXXXXX [--on-call] [--backup] [--role CLINICIAN|ADMIN] [--assign USER_ID]`, printing exactly two lines, `user_id=<uuid>` and `set_password_url=<url>`. This is WS-05's request 5, needed by WS-05.8 (the `provision_clinician(stack, ...)` E2E helper). It runs in the user-service image once WS-01.5 copies `scripts/` there (R-14-1); the subprocess test below runs the same module entry point.
  - `POST /api/v1/users/account-setup {token, new_password}`, which WS-05's `provision_clinician` helper calls (R-14-6).
  - `UserService.provision_privileged_account` and `complete_account_setup`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_105_provision_cli.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py`
```python
# services/user-service/tests/test_mvp_105_provision_cli.py
"""MVP-105: only an operator CLI creates clinicians and admins; it prints exactly the id and a one-time setup link."""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import provision_clinician as cli
from src.api import router as api_router
from src.domain.service import UserService
from src.domain.value_objects import AccountStatus, UserRole
from src.infrastructure.password_service import PasswordConfig, PasswordService
from fixtures import InMemoryAccountTokenRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("USER_SERVICE_FRONTEND_URL", "https://app.example.test")
    users, tokens = InMemoryUserRepository(), InMemoryAccountTokenRepository()
    svc = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository(),
                      account_token_repository=tokens, password_service=PasswordService(PasswordConfig()))

    async def factory():
        async def _close() -> None:
            return None
        return svc, _close

    return SimpleNamespace(users=users, tokens=tokens, svc=svc, factory=factory)


def _run(env, *argv) -> tuple[int, list[str]]:
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = cli.main(list(argv), service_factory=env.factory)
    return rc, out.getvalue().splitlines()


def test_mvp_105_cli_creates_a_verified_on_call_clinician(env) -> None:
    rc, lines = _run(env, "--email", "clin-1@test.invalid", "--display-name", "Dr One", "--phone", "+15555550100",
                     "--on-call", "--backup")
    assert rc == 0 and len(lines) == 2
    assert re.fullmatch(r"user_id=[0-9a-f-]{36}", lines[0])
    assert lines[1].startswith("set_password_url=https://app.example.test/set-password?token=")
    user = asyncio.run(env.users.get_by_id(UUID(lines[0].split("=", 1)[1])))
    assert (user.role, user.status, user.email_verified, user.is_on_call, user.phone_number) == (
        UserRole.CLINICIAN, AccountStatus.ACTIVE, True, True, "+15555550100")
    assert [c.is_backup for c in asyncio.run(env.svc.get_on_call_roster())] == [True]
    assert PasswordService(PasswordConfig()).verify_password("guess", user.password_hash).is_valid is False


@pytest.mark.parametrize("argv,code", [
    (["--email", "a@test.invalid", "--display-name", "A", "--phone", "5555550100"], 2),
    (["--email", "a@test.invalid", "--display-name", "A", "--phone", "+15555550100", "--role", "ADMIN", "--on-call"], 2),
    (["--email", "a@test.invalid", "--display-name", "A", "--phone", "+15555550100", "--role", "USER"], 2),
])
def test_mvp_105_cli_rejects_bad_input_without_output(env, argv, code) -> None:
    rc, lines = _run(env, *argv)
    assert rc == code and lines == []


def test_mvp_105_cli_refuses_an_existing_email_and_assigns_patients(env) -> None:
    patient = asyncio.run(env.svc.create_user(email="pat@test.invalid", password_hash="h", display_name="Pat")).user
    rc, lines = _run(env, "--email", "clin-2@test.invalid", "--display-name", "Dr Two", "--phone", "+15555550101",
                     "--assign", str(patient.user_id))
    clinician_id = UUID(lines[0].split("=", 1)[1])
    assert rc == 0 and asyncio.run(env.users.is_patient_assigned_to_clinician(clinician_id, patient.user_id))
    rc, lines = _run(env, "--email", "clin-2@test.invalid", "--display-name", "Dr Two", "--phone", "+15555550101")
    assert rc == 3 and lines == []


def test_mvp_105_setup_link_sets_a_password_once(env) -> None:
    _, lines = _run(env, "--email", "clin-3@test.invalid", "--display-name", "Dr Three", "--phone", "+15555550102")
    token = parse_qs(urlsplit(lines[1].split("=", 1)[1]).query)["token"][0]
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.service = SimpleNamespace(user_service=env.svc, password_service=PasswordService(PasswordConfig()))
    client = TestClient(app)
    first = client.post("/api/v1/users/account-setup", json={"token": token, "new_password": "NewPassw0rd"})
    again = client.post("/api/v1/users/account-setup", json={"token": token, "new_password": "NewPassw0rd"})
    assert (first.status_code, again.status_code) == (204, 400)
    assert again.json()["detail"]["code"] == "INVALID_TOKEN"
    user = asyncio.run(env.users.get_by_id(UUID(lines[0].split("=", 1)[1])))
    assert PasswordService(PasswordConfig()).verify_password("NewPassw0rd", user.password_hash).is_valid is True
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py
"""MVP-105: the CLI, run as it runs in the user-service image, provisions an on-call clinician as solace_app."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import UUID, uuid4

import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
REPO = Path(__file__).resolve().parents[3]


def _env(pg_migrated) -> dict[str, str]:
    parts = urlsplit(pg_migrated.dsn_app)
    env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
    env.update(PYTHONPATH=os.pathsep.join([str(REPO / "services" / "user-service"), str(REPO), str(REPO / "src")]),
               ENVIRONMENT="test", POSTGRES_HOST=parts.hostname, POSTGRES_PORT=str(parts.port),
               POSTGRES_USER=unquote(parts.username), POSTGRES_PASSWORD=unquote(parts.password),
               POSTGRES_DATABASE=parts.path.lstrip("/"), POSTGRES_SSL_MODE="disable",
               USER_SERVICE_FRONTEND_URL="https://app.example.test")
    return env


async def test_mvp_105_cli_provisions_an_on_call_clinician(pg_migrated, superuser_conn) -> None:
    email = f"clin-{uuid4().hex[:8]}@test.invalid"
    out = subprocess.run([sys.executable, "-m", "scripts.provision_clinician", "--email", email, "--display-name",
                          "Dr Provisioned", "--phone", "+15555550123", "--on-call", "--backup"],
                         cwd=REPO, env=_env(pg_migrated), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    lines = out.stdout.splitlines()
    assert len(lines) == 2 and email not in out.stdout + out.stderr
    uid = UUID(lines[0].removeprefix("user_id="))
    token = parse_qs(urlsplit(lines[1].removeprefix("set_password_url=")).query)["token"][0]
    row = await superuser_conn.fetchrow(
        "SELECT role, status, email_verified, is_on_call, is_backup FROM users WHERE id = $1", uid)
    assert dict(row) == {"role": "clinician", "status": "active", "email_verified": True, "is_on_call": True,
                         "is_backup": True}
    tok = await superuser_conn.fetchrow("SELECT purpose, expires_at - created_at AS ttl FROM account_tokens "
                                        "WHERE token_hash = $1", hashlib.sha256(token.encode()).hexdigest())
    assert tok["purpose"] == "account_setup" and timedelta(hours=71) < tok["ttl"] <= timedelta(hours=72)
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_105_provision_cli.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.provision_clinician'`, and the subprocess exits 1 with the same error. No supported path creates a CLINICIAN or stores a phone number (MVP-105).
- [ ] **Step 3: Implement** the files listed. `provision` rejects `--on-call`, `--backup` or `--assign` unless the role is CLINICIAN. It validates `--assign` as a UUID of an existing account. All usage errors print one line to stderr without echoing the email or phone. The setup link base comes from `ServiceIntegrationSettings().frontend_url`, after `validate_for_environment(os.environ.get("ENVIRONMENT", "development"))`.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass. WS-05's W1 crisis E2E proves the in-image run once R-14-1 lands.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.11: Slot 012 (W2 body): ciphertext-width PII columns, the email blind index and the consent document version

**Covers:** MVP-218, MVP-220
**Severity:** medium
**Wave:** 2
**Depends on:** WS-14.4, WS-02.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-01 (default assumed: the slot body is amended in place; no deployed database exists)
**Real infra:** yes

**Files:**
- Modify: `migrations/versions/012_user_accounts.py`. Only `upgrade()` and `downgrade()` change.
  - `upgrade()` adds:
    - `users.email`, `users.display_name` and `users.phone_number` become `sa.Text()`;
    - drop the unique index on `users.email` (SQLAlchemy named it `ix_users_email` for `index=True, unique=True` in 001; confirm with `pg_indexes`);
    - `op.add_column("users", sa.Column("email_hash", sa.String(64), nullable=True))` and `op.create_index("uq_users_email_hash", "users", ["email_hash"], unique=True)`;
    - `op.add_column("consent_records", sa.Column("document_version", sa.String(20), nullable=False, server_default="1.0"))` and `op.add_column("consent_records", sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True))`.
  - `downgrade()` reverses them: restore `String(255)`/`String(100)`/`String(20)` and the unique email index.
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_slot012_pii_columns.py` (create)

**Interfaces:**
- Consumes: the W1 body of slot 012 (WS-14.4); IF-03 chain test.
- Produces: `users.email_hash VARCHAR(64)` with unique index `uq_users_email_hash`; text-typed PII columns; `consent_records.document_version VARCHAR(20) NOT NULL DEFAULT '1.0'` and `expires_at`. Consumed by WS-14.10 and WS-14.12. Runs first in the W2 schema lane; WS-11.6 (slot 016) waits on it.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_slot012_pii_columns.py`
```python
"""Slot 012 (W2): PII columns hold codec ciphertext, email is found by a unique blind index, consent stores its document version."""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def _columns(conn, table):
    rows = await conn.fetch("SELECT column_name, data_type, character_maximum_length AS width, is_nullable, "
                            "column_default FROM information_schema.columns WHERE table_name = $1", table)
    return {r["column_name"]: r for r in rows}


async def test_slot012_users_pii_columns_fit_ciphertext_and_email_has_a_blind_index(superuser_conn) -> None:
    users = await _columns(superuser_conn, "users")
    assert {c: users[c]["data_type"] for c in ("email", "display_name", "phone_number", "bio")} == {
        "email": "text", "display_name": "text", "phone_number": "text", "bio": "text"}
    assert (users["email_hash"]["data_type"], users["email_hash"]["width"]) == ("character varying", 64)
    indexes = {r["indexname"]: r["indexdef"] for r in await superuser_conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'users'")}
    assert "CREATE UNIQUE INDEX" in indexes["uq_users_email_hash"]
    assert not [d for d in indexes.values() if d.startswith("CREATE UNIQUE INDEX") and d.endswith("(email)")]


async def test_slot012_consent_records_store_the_document_version(superuser_conn) -> None:
    consent = await _columns(superuser_conn, "consent_records")
    assert consent["document_version"]["is_nullable"] == "NO" and "1.0" in consent["document_version"]["column_default"]
    assert consent["expires_at"]["is_nullable"] == "YES"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_slot012_pii_columns.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- `email`, `display_name` and `phone_number` are `character varying` of 255/100/20. An `enc:v1:` envelope of a long email does not fit, and `users.email` is unique on its own value, which ciphertext cannot keep (MVP-218).
- `KeyError: 'email_hash'`.
- `KeyError: 'document_version'`: there is no column for the consent document version (MVP-220).
- [ ] **Step 3: Implement** `upgrade()` and `downgrade()` as listed.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the chain test and this directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.10: User PII through the IF-13 codec with an email blind index; retire the Fernet field-key exemption

**Covers:** MVP-218
**Severity:** medium
**Wave:** 2
**Depends on:** WS-14.11, WS-14.6, WS-14.8, WS-14.9, WS-09.7, WS-09.8, WS-04.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** none
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`:
  - `from solace_security.phi_codec import get_phi_codec`.
  - `USER_PII_FIELDS = ("email", "display_name", "phone_number", "bio")`. The AAD field name is `f"users.{column}"` and the owner is `users.id`.
  - `def email_search_hash(email: str, codec) -> str` returns `codec.search_hash(email.strip().lower(), purpose="users.email")`.
  - Every users write sets the encrypted fields, `email_hash` and `encryption_key_id = codec.active_key_id`. This replaces `PLAINTEXT_KEY_ID`, which is removed.
  - Every read decrypts, including `list_on_call_roster`, `list_care_team`, `list_users`, `find_by_role` and `find_on_call_clinicians`.
  - `get_by_email` looks up `WHERE email_hash = $1 AND deleted_at IS NULL`.
  - A unique violation on `uq_users_email_hash` becomes `DuplicateEntityError`.
- Modify: `services/user-service/src/main.py`. Remove the REV-12 exemption comment, the `FERNET_FIELD_KEY` requirement and `create_encryption_service`. `FERNET_TOKEN_KEY` stays, because WS-08's token service uses it.
- Modify: `services/user-service/src/config.py` (remove `UserServiceSettings.field_encryption_key` and its validator entry)
- Modify: `services/user-service/README.md`. PII is AES-256-GCM through `solace_security.phi_codec`, with the email found by HMAC blind index. The Fernet `EncryptionService` is not used for persistence.
- Modify: `services/user-service/tests/conftest.py`. `setdefault` `ENCRYPTION_MASTER_KEY` (`"ab" * 32`), `ENCRYPTION_SEARCH_HASH_SALT` and `ENCRYPTION_KEY_ID=test-k1`. These are synthetic, test-only values.
- Modify: `tests/realinfra/accounts-roster-consent-erasure/conftest.py`. New autouse fixture `phi_codec`: sets the same three variables, deletes `ENCRYPTION_RETIRED_KEYS_JSON`, calls `reset_phi_codec()`, yields `get_phi_codec()`, and resets on teardown.
- Modify: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py` (`_env` adds the three `ENCRYPTION_*` variables)
- Test: `services/user-service/tests/test_mvp_218_user_pii_codec.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_pii_at_rest_pg.py` (create)

**Interfaces:**
- Consumes: IF-13 `get_phi_codec()`, `PHICodec.encrypt/decrypt/search_hash`, `active_key_id`, `PHIDecryptError`, `reset_phi_codec()`, envelope `enc:v1:<key_id>:...`, `ENCRYPTION_KEY_ID`, `ENCRYPTION_RETIRED_KEYS_JSON` (WS-09.7, WS-09.8); the bootstrap codec warm-up (WS-04.8).
- Produces:
  - `USER_PII_FIELDS`, `email_search_hash`.
  - This is WS-14's part of WS-09's request CR-10, needed by WS-09.15 (the W3 phase-B ADOPTERS lock): `services/user-service/src/infrastructure/postgres_repository.py` imports `solace_security.phi_codec`; user-service calls neither `get_phi_encryptor` nor `configure_phi_encryption` (today `main.py:190-195` only mentions the latter in the REV-12 comment, which this task removes); every users write stores `codec.active_key_id` in `encryption_key_id`. The ADOPTERS path for user-service needs no amendment.
  - `FERNET_FIELD_KEY` and `USER_SERVICE_FIELD_ENCRYPTION_KEY` are no longer read (R-14-4).
  - The three synthetic `ENCRYPTION_*` setdefaults in `services/user-service/tests/conftest.py`, which WS-08's real-infra conftest mirrors (R-14-13). WS-14.10 is the only W2 editor of that conftest.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_218_user_pii_codec.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_pii_at_rest_pg.py`
```python
# services/user-service/tests/test_mvp_218_user_pii_codec.py
"""MVP-218: user PII goes through the IF-13 codec; the misleading Fernet exemption is gone."""
from __future__ import annotations

import pathlib

from solace_security.phi_codec import PHICodec


def test_mvp_218_pii_fields_and_blind_index_normalisation() -> None:
    import src.infrastructure.postgres_repository as pg_repo

    codec = PHICodec(active_key_id="k1", keys={"k1": bytes(range(32))}, search_salt=b"s" * 32, allow_plaintext=False)
    assert pg_repo.USER_PII_FIELDS == ("email", "display_name", "phone_number", "bio")
    assert pg_repo.email_search_hash(" Alice@Example.TEST ", codec) == codec.search_hash(
        "alice@example.test", purpose="users.email")


def test_mvp_218_no_fernet_field_exemption_remains() -> None:
    import src.config as config
    import src.main as main

    text = pathlib.Path(main.__file__).read_text(encoding="utf-8")
    assert "FERNET_FIELD_KEY" not in text and "REV-12" not in text and "create_encryption_service" not in text
    assert "field_encryption_key" not in config.UserServiceSettings.model_fields
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_pii_at_rest_pg.py
"""MVP-218: user PII is ciphertext at rest, found by blind index, bound to its row and readable after key rotation."""
from __future__ import annotations

import json
from uuid import uuid4

import pytest

from solace_security.phi_codec import PHIDecryptError, reset_phi_codec

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


def _user(us, email):
    return us.mod("domain.entities").User(email=email, password_hash="argon2-synthetic", display_name="Alice Example",
                                          phone_number="+15555550111", bio="free text about me")


async def test_mvp_218_pii_is_ciphertext_and_found_by_blind_index(us, cpm_default, phi_codec, superuser_conn) -> None:
    repo = us.mod("infrastructure.postgres_repository")
    users = repo.PostgresUserRepository()
    email = f"Alice-{uuid4().hex[:8]}@Test.Invalid"
    user = await users.save(_user(us, email))
    row = dict(await superuser_conn.fetchrow("SELECT email, email_hash, display_name, phone_number, bio, "
                                             "encryption_key_id FROM users WHERE id = $1", user.user_id))
    for column in ("email", "display_name", "phone_number", "bio"):
        assert row[column].startswith("enc:v1:test-k1:"), column
    assert "alice" not in repr(row).lower() and "5555550111" not in repr(row)
    assert row["email_hash"] == phi_codec.search_hash(email.lower(), purpose="users.email")
    assert row["encryption_key_id"] == "test-k1"
    found = await users.get_by_email(email.upper())
    assert (found.user_id, found.display_name, found.phone_number) == (user.user_id, "Alice Example", "+15555550111")
    with pytest.raises(repo.DuplicateEntityError):
        await users.save(_user(us, email.lower()))


async def test_mvp_218_ciphertext_cannot_move_between_users(us, cpm_default, phi_codec, superuser_conn) -> None:
    users = us.mod("infrastructure.postgres_repository").PostgresUserRepository()
    a = await users.save(_user(us, f"a-{uuid4().hex[:8]}@test.invalid"))
    b = await users.save(_user(us, f"b-{uuid4().hex[:8]}@test.invalid"))
    await superuser_conn.execute("UPDATE users SET display_name = (SELECT display_name FROM users WHERE id = $1) "
                                 "WHERE id = $2", a.user_id, b.user_id)
    with pytest.raises(PHIDecryptError):
        await users.get_by_id(b.user_id)


async def test_mvp_218_rows_under_a_retired_key_still_decrypt(us, cpm_default, phi_codec, superuser_conn, monkeypatch) -> None:
    users = us.mod("infrastructure.postgres_repository").PostgresUserRepository()
    user = await users.save(_user(us, f"r-{uuid4().hex[:8]}@test.invalid"))
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "cd" * 32)
    monkeypatch.setenv("ENCRYPTION_KEY_ID", "test-k2")
    monkeypatch.setenv("ENCRYPTION_RETIRED_KEYS_JSON", json.dumps({"test-k1": "ab" * 32}))
    reset_phi_codec()
    got = await users.get_by_id(user.user_id)
    assert got.display_name == "Alice Example"
    got.display_name = "Rotated"
    await users.update(got)
    row = await superuser_conn.fetchrow("SELECT display_name, encryption_key_id FROM users WHERE id = $1", user.user_id)
    assert row["encryption_key_id"] == "test-k2" and row["display_name"].startswith("enc:v1:test-k2:")
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_218_user_pii_codec.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_218_pii_at_rest_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- `AttributeError: USER_PII_FIELDS`.
- The source test finds `FERNET_FIELD_KEY`, `REV-12` and `create_encryption_service`: the exemption cites a Fernet service that never encrypts anything.
- On Postgres, `email` and `display_name` are plaintext (`startswith("enc:v1:")` fails) and `email_hash` is NULL (MVP-218).
- The fixture `phi_codec` is missing until the conftest change.
- [ ] **Step 3: Implement** the edits listed under Files. Decryption uses the row's own `id` as owner. A decrypt failure propagates as `PHIDecryptError`, never as a silent `None`. `soft_delete`'s anonymised email is encrypted and hashed like any other.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass, including the W0/W1 real-infra tests, which now run with the autouse codec keys.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.12: Consent records store and return their document version; notification's contracts verify against the internal API

**Covers:** MVP-220
**Severity:** medium
**Wave:** 2
**Depends on:** WS-14.11, WS-14.10, WS-14.8, WS-05.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`:
  - `PostgresConsentRepository.save` writes `document_version = consent.version` and `expires_at`. It never binds the integer `version` lock.
  - `_row_to_entity` reads `version=row["document_version"]` and `expires_at`.
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_220_consent_version_pg.py` (create)
- Test: `services/user-service/tests/test_mvp_357_user_provider_contracts.py` (create). This is the W2 provider verification of `tests/contracts/notification__user/*.json`. The care-team contract's `crisis_contact_consent` field is computed from these consent records.

**Interfaces:**
- Consumes: slot 012 `document_version` (WS-14.11); IF-20 `solace_testing.contracts.verify_provider(app, contracts_dir, *, auth_headers)` (WS-05.10); the notification__user contract files written by WS-13 (R-14-11).
- Produces: `ConsentRecord.version` round-trips as a `str`. The provider check satisfies WS-05.10's coverage guard for `notification__user` (WS-05 request 10).

- [ ] **Step 1: Write the failing test** in `tests/realinfra/accounts-roster-consent-erasure/test_mvp_220_consent_version_pg.py` and `services/user-service/tests/test_mvp_357_user_provider_contracts.py`
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_220_consent_version_pg.py
"""MVP-220: the consent document version is stored in its own column and read back as text, per user under RLS."""
from __future__ import annotations

from uuid import uuid4

import pytest

from solace_common.request_context import acting_as_user

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_220_document_version_round_trips(us, cpm_default, seed_user, superuser_conn) -> None:
    repo, vo = us.mod("infrastructure.postgres_repository"), us.mod("domain.value_objects")
    consents, owner, other = repo.PostgresConsentRepository(), await seed_user(), await seed_user()
    with acting_as_user(owner):
        await consents.save(vo.ConsentRecord(consent_id=uuid4(), user_id=owner,
                                             consent_type=vo.ConsentType.PRIVACY_POLICY, granted=True, version="2.1"))
        records = await consents.get_by_user_id(owner)
    assert [(r.consent_type.value, r.version) for r in records] == [("privacy_policy", "2.1")]
    row = await superuser_conn.fetchrow("SELECT version, document_version FROM consent_records WHERE user_id = $1", owner)
    assert (row["document_version"], row["version"]) == ("2.1", 1)
    with acting_as_user(other):
        assert await consents.get_by_user_id(owner) == []
```
```python
# services/user-service/tests/test_mvp_357_user_provider_contracts.py
"""MVP-357 (provider side, WS-05 request 10): notification__user contracts verify against the real internal router."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from fastapi import FastAPI

from solace_security.middleware import AuthenticatedService, get_current_service
from solace_security.service_auth import ServicePermission
from solace_testing.contracts import load_contracts, verify_provider
from src.domain.entities import User
from src.domain.service import UserService
from src.domain.value_objects import AccountStatus, UserRole
from src.internal_api import internal_router
from fixtures import InMemoryConsentRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository

CONTRACTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "notification__user"


async def test_notification__user_contracts_verify_against_the_internal_api() -> None:
    users = InMemoryUserRepository()
    svc = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository(),
                      consent_repository=InMemoryConsentRepository())
    for contract in load_contracts(CONTRACTS):
        request = contract.request
        path = request["path"] if isinstance(request, dict) else request.path
        for part in path.split("/"):
            try:
                await users.save(User(user_id=UUID(part), email=f"u{part[:8]}@test.invalid", password_hash="x",
                                      display_name="Seed", status=AccountStatus.ACTIVE))
            except ValueError:
                continue
    await users.save(User(email="clin@test.invalid", password_hash="x", display_name="Dr On", role=UserRole.CLINICIAN,
                          status=AccountStatus.ACTIVE, is_on_call=True))
    app = FastAPI()
    app.include_router(internal_router, prefix="/api/v1/internal")
    app.state.service = SimpleNamespace(user_service=svc)
    app.dependency_overrides[get_current_service] = lambda: AuthenticatedService(
        service_name="notification-service", permissions=[ServicePermission.READ_ONCALL_ROSTER.value])
    results = verify_provider(app, CONTRACTS, auth_headers=lambda interaction: {})
    assert len(results) >= 2
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_220_consent_version_pg.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="0"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_357_user_provider_contracts.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- `get_by_user_id` raises `ValidationError: version Input should be a valid string`. `_row_to_entity` passes the integer lock `1`, and the saved document version is lost: `document_version` keeps its default `'1.0'` (MVP-220).
- The provider check fails on the care-team body if the consent mapping is broken, or on any field drift from the IF-08 shapes that WS-13 wrote down.
- If the contract directory is missing, the test fails on `load_contracts`, which blocks the barrier until R-14-11 lands.
- [ ] **Step 3: Implement** the two repository edits.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite, the real-infra directory and the contract suite.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/contracts -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass, or XFAIL as listed in WS-05's contract manifest.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.13: IF-16 erasure consumer helper for every PHI service

**Covers:** MVP-029
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-06.8, WS-06.11, WS-09.11, WS-07.1
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: PHI is hard-deleted per service, the service's own event rows are erased except the erasure-protocol events, audit rows are kept)
**Real infra:** no

**Files:**
- Create: `src/solace_infrastructure/gdpr/consumer.py`:
  - `USER_EVENTS_TOPIC = "solace.users"`;
  - `def erasure_group_id(service_name: str) -> str`, returning `f"solace-group-erasure-{service_name}"`;
  - `async def handle_user_deleted(event, *, plane, service_name, erasure, purge_caches=None) -> UserErasureCompletedKafkaEvent | None`;
  - `def register_user_erasure_consumer(plane, *, service_name: str, erasure: UserDataErasure, purge_caches: Callable[[UUID], Awaitable[None]] | None = None) -> None`, which calls `plane.subscribe(USER_EVENTS_TOPIC, ["user.deleted"], <handler>, group_id=erasure_group_id(service_name), max_attempts=5)`.
  - Module-level imports: `from solace_events.service_plane import erase_user_events`, `from solace_security import audit_runtime`, `from solace_common.request_context import acting_as_user`. The handler calls `erase_user_events(...)` and `audit_runtime.audit(...)` through these names, so tests patch `solace_infrastructure.gdpr.consumer.erase_user_events` and `solace_security.audit_runtime.audit`. The existing package init module of the gdpr package is not edited (it keeps exporting only the erasure orchestrator); callers import the submodule by full path.
- Test: `tests/solace_infrastructure/test_mvp_029_erasure_consumer.py` (create)

**Interfaces:**
- Consumes:
  - IF-05 `ServiceEventPlane.subscribe(topic, event_types, handler, *, group_id, max_attempts=5)` and `.publish(event, *, conn=None)`; `RetryableEventError`, `NonRetryableEventError`, `EventPublishError` (WS-06.8);
  - `erase_user_events(user_id: UUID) -> int` and `UserErasureCompletedKafkaEvent(user_id, service_name, erasure_request_id, stores, completed_at)` (WS-06.11);
  - IF-14 `audit(...)` with `AuditEventType.ACCOUNT_ERASED_IN_SERVICE` and details keys from `ALLOWED_DETAIL_KEYS` (WS-09.11);
  - `UserDataErasure`, `ErasureIncomplete` (this package).
- Produces IF-16 `register_user_erasure_consumer`, with this contract:
  - the erasure runs under `acting_as_user(event.user_id)`;
  - `erasure_request_id = event.metadata.correlation_id`;
  - `stores = report.deleted | {"event_tables": <erase_user_events count>}`;
  - completion is published with `metadata.correlation_id = erasure_request_id`, `causation_id = event.metadata.event_id` and `source_service = service_name`;
  - `ErasureIncomplete`, `EventPublishError` or a failing `purge_caches` raises `RetryableEventError` whose message lists store names only;
  - a service with no registered stores raises `NonRetryableEventError`;
  - other event types return `None`.
- Satisfies WS-17.11's request and serves WS-11, WS-12, WS-13, WS-15, WS-16 and WS-18.

- [ ] **Step 1: Write the failing test** in `tests/solace_infrastructure/test_mvp_029_erasure_consumer.py`
```python
"""MVP-029 / IF-16: a PHI service erases a deleted user as that user, audits it and reports completion."""
from __future__ import annotations

from uuid import uuid4

import pytest

from solace_common.request_context import get_current_user_id
from solace_events.errors import EventPublishError, NonRetryableEventError, RetryableEventError
from solace_events.schemas import (ConsentChangedKafkaEvent, EventMetadata, UserDeletedKafkaEvent,
                                   UserErasureCompletedKafkaEvent)
from solace_infrastructure.gdpr.consumer import register_user_erasure_consumer
from solace_infrastructure.gdpr.erasure import UserDataErasure
from solace_security.audit import AuditEventType


class _Plane:
    def __init__(self, fail_publish: bool = False) -> None:
        self.subs: list[dict] = []
        self.published: list = []
        self.fail_publish = fail_publish

    def subscribe(self, topic, event_types, handler, *, group_id, max_attempts=5) -> None:
        self.subs.append(dict(topic=topic, event_types=tuple(event_types), handler=handler, group_id=group_id))

    async def publish(self, event, *, conn=None):
        if self.fail_publish:
            raise EventPublishError("broker down")
        self.published.append(event)
        return event.metadata.event_id


@pytest.fixture
def audits(monkeypatch):
    calls: list = []

    async def _audit(event_type, **kwargs) -> None:
        calls.append((event_type, kwargs))

    async def _erase_events(user_id) -> int:
        return 2

    monkeypatch.setattr("solace_security.audit_runtime.audit", _audit)
    monkeypatch.setattr("solace_infrastructure.gdpr.consumer.erase_user_events", _erase_events)
    return calls


def _deleted(uid, req):
    return UserDeletedKafkaEvent(user_id=uid, reason="user_request",
                                 metadata=EventMetadata(correlation_id=req, source_service="user-service"))


async def test_mvp_029_consumer_erases_as_the_user_audits_and_reports(audits) -> None:
    plane, seen, purged = _Plane(), [], []
    erasure = UserDataErasure()

    async def _rows(uid: str) -> int:
        seen.append(get_current_user_id())
        return 3

    async def _purge(uid) -> None:
        purged.append(uid)

    erasure.register("memory_records", _rows)
    register_user_erasure_consumer(plane, service_name="memory-service", erasure=erasure, purge_caches=_purge)
    [sub] = plane.subs
    assert (sub["topic"], sub["event_types"], sub["group_id"]) == (
        "solace.users", ("user.deleted",), "solace-group-erasure-memory-service")
    uid, req = uuid4(), uuid4()
    await sub["handler"](_deleted(uid, req))
    assert seen == [str(uid)] and get_current_user_id() is None and purged == [uid]
    [done] = plane.published
    assert isinstance(done, UserErasureCompletedKafkaEvent)
    assert (done.user_id, done.service_name, done.erasure_request_id) == (uid, "memory-service", req)
    assert done.stores == {"memory_records": 3, "event_tables": 2} and done.metadata.correlation_id == req
    [(event_type, kw)] = audits
    assert event_type == AuditEventType.ACCOUNT_ERASED_IN_SERVICE and kw["actor_type"] == "service"
    assert kw["subject_user_id"] == str(uid) and kw["details"]["erasure_request_id"] == str(req)


async def test_mvp_029_incomplete_erasure_is_retryable_and_reports_nothing(audits) -> None:
    plane, erasure = _Plane(), UserDataErasure()

    async def _down(uid: str) -> int:
        raise RuntimeError("store down for alice@example.test")

    erasure.register("therapy_sessions", _down)
    register_user_erasure_consumer(plane, service_name="therapy-service", erasure=erasure)
    uid = uuid4()
    with pytest.raises(RetryableEventError) as exc:
        await plane.subs[0]["handler"](_deleted(uid, uuid4()))
    assert "therapy_sessions" in str(exc.value)
    assert str(uid) not in str(exc.value) and "alice" not in str(exc.value)
    assert plane.published == [] and audits == []


async def test_mvp_029_a_service_with_no_stores_dead_letters(audits) -> None:
    plane = _Plane()
    register_user_erasure_consumer(plane, service_name="safety-service", erasure=UserDataErasure())
    with pytest.raises(NonRetryableEventError):
        await plane.subs[0]["handler"](_deleted(uuid4(), uuid4()))


async def test_mvp_029_completion_publish_failure_is_retryable(audits) -> None:
    plane, erasure = _Plane(fail_publish=True), UserDataErasure()

    async def _zero(uid: str) -> int:
        return 0

    erasure.register("personality_profiles", _zero)
    register_user_erasure_consumer(plane, service_name="personality-service", erasure=erasure)
    with pytest.raises(RetryableEventError):
        await plane.subs[0]["handler"](_deleted(uuid4(), uuid4()))


async def test_mvp_029_other_user_events_are_ignored(audits) -> None:
    plane, erasure = _Plane(), UserDataErasure()

    async def _zero(uid: str) -> int:
        return 0

    erasure.register("x", _zero)
    register_user_erasure_consumer(plane, service_name="diagnosis-service", erasure=erasure)
    await plane.subs[0]["handler"](ConsentChangedKafkaEvent(user_id=uuid4(), consent_type="privacy_policy",
                                                            action="revoked"))
    assert plane.published == [] and audits == []
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_infrastructure/test_mvp_029_erasure_consumer.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'solace_infrastructure.gdpr.consumer'`. Nothing in any service consumes `user.deleted`, so a deleted user's PHI survives everywhere except user-service (MVP-029).
- [ ] **Step 3: Implement** `src/solace_infrastructure/gdpr/consumer.py`.
  - Order inside `acting_as_user(event.user_id)`: `erasure.erase(str(event.user_id))`, then `erase_user_events(event.user_id)`, then `purge_caches(event.user_id)`.
  - Then outside the context: the audit call, then the completion publish.
  - The `RuntimeError` that `UserDataErasure.erase` raises for zero stores becomes `NonRetryableEventError`.
  - Audit details: `{"service": service_name, "stores": ",".join(sorted(stores)), "count": sum(stores.values()), "erasure_request_id": str(request_id)}`.
  - Logs carry `service`, `erasure_request_id` and store names only.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the platform suite and the event suite.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_infrastructure -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/solace_events -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.14: Slot 012 (W3 body): erasure requests and per-service completions

**Covers:** MVP-029
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-14.11, WS-02.3
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** D-24 (default assumed: erasure bookkeeping rows hold ids, service names and counts only, and are retained like audit rows)
**Real infra:** yes

**Files:**
- Modify: `migrations/versions/012_user_accounts.py`. Only `upgrade()` and `downgrade()` change.
  - `upgrade()` adds:
    - `erasure_requests`: `id` UUID PK; `user_id` UUID NOT NULL, indexed, no FK; `reason` String(32) NOT NULL; `status` String(16) NOT NULL with `CHECK (status IN ('fanout','complete'))`; `expected_services` `ARRAY(String(64))` NOT NULL; `local_stores` JSONB NOT NULL default `'{}'`; `requested_at` timestamptz NOT NULL default now(); `completed_at` timestamptz NULL.
    - Partial unique index `uq_erasure_requests_open_user ON erasure_requests (user_id) WHERE status = 'fanout'`.
    - `erasure_service_completions`: `request_id` UUID FK `erasure_requests.id` ON DELETE CASCADE; `service_name` String(64); `stores` JSONB NOT NULL default `'{}'`; `completed_at` timestamptz NOT NULL; PK (`request_id`, `service_name`).
  - `downgrade()` drops both tables.
  - Neither table has RLS.
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_slot012_erasure_tables.py` (create)

**Interfaces:**
- Consumes: slot 012 W2 body (WS-14.11); the slug fixtures.
- Produces: tables `erasure_requests` and `erasure_service_completions`, as listed. Consumed by WS-14.16. Runs first in the W3 schema lane, before slots 013 to 021.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_slot012_erasure_tables.py`
```python
"""Slot 012 (W3): erasure bookkeeping tables, one open request per user, completions keyed per service."""
from __future__ import annotations

from uuid import uuid4

import asyncpg
import pytest

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
INSERT_REQ = ("INSERT INTO erasure_requests (id, user_id, reason, status, expected_services) "
              "VALUES ($1, $2, 'user_request', $3, ARRAY['memory-service'])")


async def test_slot012_erasure_tables_exist_without_rls(superuser_conn) -> None:
    for table in ("erasure_requests", "erasure_service_completions"):
        assert await superuser_conn.fetchval("SELECT relrowsecurity FROM pg_class WHERE relname = $1", table) is False


async def test_slot012_one_open_request_per_user_and_completion_keys(pg_migrated, seed_user) -> None:
    uid = await seed_user()
    conn = await asyncpg.connect(pg_migrated.dsn_app)
    try:
        first, done = uuid4(), uuid4()
        await conn.execute(INSERT_REQ, first, uid, "fanout")
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(INSERT_REQ, uuid4(), uid, "fanout")
        await conn.execute(INSERT_REQ, done, uid, "complete")
        await conn.execute("INSERT INTO erasure_service_completions (request_id, service_name, stores, completed_at) "
                           "VALUES ($1, 'memory-service', $2, now())", first, {"memory_records": 3})
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute("INSERT INTO erasure_service_completions (request_id, service_name, completed_at) "
                               "VALUES ($1, 'memory-service', now())", first)
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(INSERT_REQ, uuid4(), uid, "pending")
    finally:
        await conn.close()
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_slot012_erasure_tables.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `relrowsecurity` is `None` and the INSERT raises `UndefinedTableError`. No durable record of an erasure request or of which services have finished exists (MVP-029).
- [ ] **Step 3: Implement** `upgrade()` and `downgrade()`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS. The test uses a raw asyncpg connection, which has no JSONB codec. If the dict parameter is rejected, bind `json.dumps(...)` in this test only; the repository code always binds objects through the pool codec.
- [ ] **Step 5: Regression:** run the chain test and this directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.15: Retry-safe local erasure: the users row is anonymised last and only after every store succeeds

**Covers:** MVP-216
**Severity:** medium
**Wave:** 3
**Depends on:** WS-14.10, WS-14.6
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: consent records and the anonymised users row are retained; everything else of the user's is hard-deleted)
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/domain/erasure.py`. `build_user_data_erasure(user_repository, preferences_repository, *, account_token_repository=None)` registers `user_preferences`, `clinician_assignments` and, when given, `account_tokens`. `user_profile` is no longer a store. The docstring records why: the soft delete is the final step in `UserService.delete_user`.
- Modify: `services/user-service/src/domain/service.py`. `delete_user` order:
  1. Live user lookup.
  2. `erasure.erase(...)` inside `acting_as_user(user_id)`.
  3. `ErasureIncomplete` gives `DELETE_FAILED` and leaves the users row untouched.
  4. On success only: `user.soft_delete()` and `update(user)`.
- Modify: `services/user-service/tests/test_gdpr_erasure.py`:
  - `test_registry_registers_expected_store_names` expects `("user_preferences", "clinician_assignments")`;
  - the erase tests drop the `user_profile` key;
  - `test_erase_fails_loud_when_a_store_raises` asserts the users row is still live and not anonymised.
- Test: `services/user-service/tests/test_mvp_216_erasure_retry.py` (create)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_216_local_erasure_pg.py` (create)

**Interfaces:**
- Consumes: `UserDataErasure` (this lane's package); `AccountTokenRepository.delete_for_user` (WS-14.6); `acting_as_user` (WS-07.1).
- Produces: `build_user_data_erasure(..., *, account_token_repository=None)`, and the delete contract: the users row changes only after full success, so a retry finds the live user and finishes.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_216_erasure_retry.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_216_local_erasure_pg.py`
```python
# services/user-service/tests/test_mvp_216_erasure_retry.py
"""MVP-216: a failed erasure leaves the account live; the retry finishes the job."""
from __future__ import annotations

import pytest

from solace_common.request_context import get_current_user_id
from src.domain.entities import User, UserPreferences
from src.domain.erasure import build_user_data_erasure
from src.domain.service import UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository


class _FlakyPrefs(InMemoryUserPreferencesRepository):
    def __init__(self) -> None:
        super().__init__()
        self.calls, self.contexts = 0, []

    async def delete(self, user_id) -> bool:
        self.calls += 1
        self.contexts.append(get_current_user_id())
        if self.calls == 1:
            raise RuntimeError("transient")
        return await super().delete(user_id)


@pytest.mark.asyncio
async def test_mvp_216_failed_erasure_keeps_the_account_and_the_retry_completes() -> None:
    users, prefs = InMemoryUserRepository(), _FlakyPrefs()
    user = await users.save(User(email="erase@example.com", password_hash="h", display_name="Erase Me"))
    await prefs.save(UserPreferences(user_id=user.user_id))
    svc = UserService(user_repository=users, preferences_repository=prefs)
    first = await svc.delete_user(user.user_id, reason="user_request")
    assert (first.success, first.error_code) == (False, "DELETE_FAILED")
    live = await users.get_by_id(user.user_id)
    assert live is not None and live.email == "erase@example.com" and live.deleted_at is None
    second = await svc.delete_user(user.user_id, reason="user_request")
    assert second.success is True
    assert await prefs.get_by_user_id(user.user_id) is None and await users.get_by_id(user.user_id) is None
    assert prefs.contexts == [str(user.user_id), str(user.user_id)]


def test_mvp_216_users_row_is_not_an_erasure_store() -> None:
    erasure = build_user_data_erasure(InMemoryUserRepository(), InMemoryUserPreferencesRepository())
    assert erasure.store_names == ("user_preferences", "clinician_assignments")
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_216_local_erasure_pg.py
"""MVP-216: local erasure removes RLS-protected rows even with no request context, then anonymises the users row."""
from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from solace_common.request_context import get_current_user_id

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def test_mvp_216_delete_user_erases_rls_rows_without_request_context(us, cpm_default, superuser_conn) -> None:
    repo, svc_mod = us.mod("infrastructure.postgres_repository"), us.mod("domain.service")
    tokens = repo.PostgresAccountTokenRepository()
    svc = svc_mod.UserService(user_repository=repo.PostgresUserRepository(),
                              preferences_repository=repo.PostgresUserPreferencesRepository(),
                              account_token_repository=tokens,
                              integration_settings=svc_mod.ServiceIntegrationSettings(notification_service_url="http://127.0.0.1:9"))
    uid = (await svc.create_user(email=f"del-{uuid4().hex[:8]}@test.invalid", password_hash="argon2-synthetic",
                                 display_name="Del")).user.user_id
    await tokens.issue(uid, "account_setup", timedelta(hours=1))
    assert get_current_user_id() is None
    assert (await svc.delete_user(uid, reason="user_request")).success is True
    assert await superuser_conn.fetchval("SELECT count(*) FROM user_preferences WHERE user_id = $1", uid) == 0
    assert await superuser_conn.fetchval("SELECT count(*) FROM account_tokens WHERE user_id = $1", uid) == 0
    row = await superuser_conn.fetchrow("SELECT deleted_at, status FROM users WHERE id = $1", uid)
    assert row["deleted_at"] is not None and row["status"] == "inactive"
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_216_erasure_retry.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_216_local_erasure_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- After the first, failed delete, `get_by_id` returns `None` (the profile was anonymised first), and the retry returns `USER_NOT_FOUND`. The remaining stores are never erased (MVP-216).
- `contexts == [None]`.
- `store_names` starts with `user_profile`.
- On Postgres, the `account_tokens` row survives (it is not a registered store). With no request context, the preferences delete runs with no RLS user and silently removes 0 rows while reporting success.
- [ ] **Step 3: Implement** the edits listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.16: user.deleted in the same transaction as the erasure request; per-service completion tracking and the admin status view

**Covers:** MVP-029
**Severity:** blocker
**Wave:** 3
**Depends on:** WS-14.13, WS-14.14, WS-14.15, WS-06.8, WS-06.11, WS-05.14
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: complete when every expected PHI service reported; target 15 min, at most 24 h)
**Real infra:** yes

**Files:**
- Modify: `services/user-service/src/domain/erasure.py`:
  - `EXPECTED_ERASURE_SERVICES = ("orchestrator-service", "safety-service", "therapy-service", "diagnosis-service", "memory-service", "personality-service", "notification-service")`;
  - `ERASURE_TRACKER_GROUP = "solace-group-erasure-user-service"`;
  - `@dataclass(frozen=True) ServiceCompletion(service_name: str, completed_at: datetime, stores: dict[str, int])`;
  - `@dataclass(frozen=True) ErasureStatus(request_id: UUID, user_id: UUID, status: Literal["fanout", "complete"], requested_at: datetime, completed_at: datetime | None, services: dict[str, ServiceCompletion | None])`;
  - `build_user_data_erasure(..., erase_events: Callable[[UUID], Awaitable[int]] | None = None, oauth_account_repository=None)` registers `event_tables` and `oauth_accounts` when given.
- Modify: `services/user-service/src/infrastructure/repository.py`. New ABC `ErasureRequestRepository`:
  - `finalize(user: User, *, request_id: UUID, reason: str, local_stores: Mapping[str, int], expected_services: Sequence[str], publish: Callable[[Any], Awaitable[None]] | None) -> None`: updates the users row, inserts the request and calls `publish(conn)`, all in one transaction;
  - `record_completion(request_id: UUID, service_name: str, stores: Mapping[str, int], completed_at: datetime) -> ErasureStatus | None`: `None` for an unknown request; an idempotent upsert; marks `complete` when every expected service has reported; serialised with `SELECT ... FOR UPDATE`;
  - `get_status(request_id: UUID) -> ErasureStatus | None`.
- Modify: `services/user-service/src/infrastructure/postgres_repository.py`:
  - `PostgresErasureRequestRepository(*, pool_name="default")`;
  - `PostgresUserRepository.update(user, *, conn: asyncpg.Connection | None = None)`, which uses the given connection inside the caller's transaction.
- Modify: `services/user-service/src/domain/service.py`:
  - `UserService.__init__` gains `erasure_request_repository: ErasureRequestRepository | None = None` and `event_plane: ServiceEventPlane | None = None`.
  - `DeleteUserResult.erasure_request_id: UUID | None = None`.
  - `delete_user`'s final step calls `erasure_requests.finalize(...)`, whose `publish` runs `plane.publish(UserDeletedKafkaEvent(user_id=user_id, reason=<reason code>, metadata=EventMetadata(correlation_id=request_id, source_service="user-service")), conn=conn)`.
  - With no request repository or plane: in strict, return `DELETE_FAILED`; in dev/test, soft-delete and log WARNING `erasure_fanout_unavailable`.
  - Reason codes are `user_request` and `admin_request`.
  - New `async def handle_erasure_completed(event: BaseEvent) -> None`: ignores other types; an unknown request raises `NonRetryableEventError`; a service outside the expected set is logged and ignored.
  - New `get_erasure_status(request_id) -> ErasureStatus | None`.
- Modify: `services/user-service/src/main.py`:
  - pass `PostgresErasureRequestRepository()` and `runtime.event_plane`;
  - pass `erase_events=erase_user_events` when the plane exists, and `oauth_account_repository=getattr(state, "oauth_account_repository", None)`;
  - `runtime.event_plane.subscribe("solace.users", ["user.erasure.completed"], user_service.handle_erasure_completed, group_id=ERASURE_TRACKER_GROUP)`;
  - strict with `runtime.event_plane is None` raises `StartupDependencyError`.
- Modify: `services/user-service/src/api.py`. New `GET /admin/erasure-requests/{request_id}` with `require_roles("admin")`. It returns `{request_id, user_id, status, requested_at, completed_at, services: [{service_name, status: "complete"|"pending", completed_at, stores}]}`, or 404 `ERASURE_REQUEST_NOT_FOUND`.
- Modify: `services/user-service/src/events.py`. `to_kafka_event(UserCreatedEvent)` sets `email=""` and `display_name=""`, which satisfies WS-06.12's request.
- Modify: `services/user-service/tests/fixtures.py`. New `InMemoryErasureRequestRepository`. `finalize` applies the user update only if `publish` succeeds and records `last_conn` (a sentinel object).
- Test: `services/user-service/tests/test_mvp_029_erasure_fanout.py` (create)
- Test: `services/user-service/tests/test_mvp_357_user_event_contracts.py` (create; `verify_event_producer` over `tests/contracts/events/user__*/user.deleted.json`)
- Test: `tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_erasure_fanout_pg.py` (create)

**Interfaces:**
- Consumes:
  - IF-05 `ServiceEventPlane.publish(event, *, conn=None)` and `.subscribe(...)` (WS-06.8);
  - `erase_user_events` and `UserErasureCompletedKafkaEvent` (WS-06.11);
  - IF-20 `load_event_contracts` and `verify_event_producer` (WS-05.14 contracts);
  - slot-012 erasure tables (WS-14.14); retry-safe local erasure (WS-14.15).
- Produces:
  - IF-16 steps 2, 3 and 5: `UserService.delete_user(user_id, reason) -> DeleteUserResult` carrying `erasure_request_id`, which WS-08's DELETE route returns (R-14-9). This answers WS-08's request R-8 for WS-08.8 (W3 phase B, depends on this task): on success `erasure_request_id` is the request UUID; failures return `error_code` `USER_NOT_FOUND` or `DELETE_FAILED` instead of raising, matching WS-08.8's 404/503 mapping (ledger Ruling on the IF-16 wrapper); `user.deleted` published on `solace.users` with `metadata.correlation_id = erasure_request_id`; completion tracking.
  - `GET /api/v1/admin/erasure-requests/{request_id}`.
  - `EXPECTED_ERASURE_SERVICES`, `ERASURE_TRACKER_GROUP`, `ErasureStatus`, `ServiceCompletion` and `ErasureRequestRepository`.
  - WS-05.15's erasure E2E reads the status endpoint.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_029_erasure_fanout.py`, `services/user-service/tests/test_mvp_357_user_event_contracts.py` and `tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_erasure_fanout_pg.py`
```python
# services/user-service/tests/test_mvp_029_erasure_fanout.py
"""MVP-029: deletion publishes user.deleted with the erasure request id, in the same unit of work, and tracks completion."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from solace_events.errors import EventPublishError, NonRetryableEventError
from solace_events.schemas import UserErasureCompletedKafkaEvent
from src.domain.entities import User
from src.domain.erasure import EXPECTED_ERASURE_SERVICES
from src.domain.service import UserService
from src.events import UserCreatedEvent, to_kafka_event
from fixtures import InMemoryErasureRequestRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository


class _Plane:
    def __init__(self, fail: bool = False) -> None:
        self.published, self.fail = [], fail

    async def publish(self, event, *, conn=None):
        if self.fail:
            raise EventPublishError("broker down")
        self.published.append((event, conn))
        return event.metadata.event_id


async def _setup(plane):
    users, requests = InMemoryUserRepository(), InMemoryErasureRequestRepository()
    user = await users.save(User(email="gone@example.com", password_hash="h", display_name="Gone"))
    svc = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository(),
                      erasure_request_repository=requests, event_plane=plane)
    return svc, users, requests, user


def _completed(uid, req, service):
    return UserErasureCompletedKafkaEvent(user_id=uid, service_name=service, erasure_request_id=req,
                                          stores={"rows": 1}, completed_at=datetime.now(timezone.utc))


@pytest.mark.asyncio
async def test_mvp_029_delete_publishes_user_deleted_inside_the_finalize_transaction() -> None:
    plane = _Plane()
    svc, users, requests, user = await _setup(plane)
    result = await svc.delete_user(user.user_id, reason="user_request")
    assert result.success is True and isinstance(result.erasure_request_id, UUID)
    [(event, conn)] = plane.published
    assert (event.event_type, event.user_id, event.reason) == ("user.deleted", user.user_id, "user_request")
    assert event.metadata.correlation_id == result.erasure_request_id
    assert event.metadata.source_service == "user-service" and conn is requests.last_conn
    status = await svc.get_erasure_status(result.erasure_request_id)
    assert status.status == "fanout" and set(status.services) == set(EXPECTED_ERASURE_SERVICES)


@pytest.mark.asyncio
async def test_mvp_029_publish_failure_rolls_back_the_anonymisation() -> None:
    svc, users, requests, user = await _setup(_Plane(fail=True))
    result = await svc.delete_user(user.user_id, reason="user_request")
    assert (result.success, result.error_code) == (False, "DELETE_FAILED")
    assert (await users.get_by_id(user.user_id)).email == "gone@example.com"


@pytest.mark.asyncio
async def test_mvp_029_every_expected_service_must_report() -> None:
    svc, _, _, user = await _setup(_Plane())
    req = (await svc.delete_user(user.user_id, reason="user_request")).erasure_request_id
    for service in EXPECTED_ERASURE_SERVICES[:-1]:
        await svc.handle_erasure_completed(_completed(user.user_id, req, service))
    await svc.handle_erasure_completed(_completed(user.user_id, req, EXPECTED_ERASURE_SERVICES[0]))
    await svc.handle_erasure_completed(_completed(user.user_id, req, "analytics-service"))
    assert (await svc.get_erasure_status(req)).status == "fanout"
    await svc.handle_erasure_completed(_completed(user.user_id, req, EXPECTED_ERASURE_SERVICES[-1]))
    done = await svc.get_erasure_status(req)
    assert done.status == "complete" and done.completed_at is not None
    with pytest.raises(NonRetryableEventError):
        await svc.handle_erasure_completed(_completed(user.user_id, uuid4(), "memory-service"))


def test_mvp_029_user_created_event_carries_no_contact_details() -> None:
    event = to_kafka_event(UserCreatedEvent(aggregate_id=uuid4(), email="a@b.test", display_name="Al", role="user",
                                            status="pending_verification", timezone="UTC", locale="en-US"))
    assert (event.email, event.display_name) == ("", "")
```
```python
# services/user-service/tests/test_mvp_357_user_event_contracts.py
"""MVP-357 (producer side, WS-05.14): user.deleted matches every user__<svc> consumer contract."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from solace_events.schemas import EventMetadata, UserDeletedKafkaEvent
from solace_testing.contracts import load_event_contracts, verify_event_producer

EVENTS = Path(__file__).resolve().parents[3] / "tests" / "contracts" / "events"
USER_CONTRACTS = [c for c in load_event_contracts(EVENTS) if c.producer == "user-service"]


@pytest.mark.parametrize("contract", USER_CONTRACTS, ids=lambda c: f"user__{c.consumer}:{c.event_type}")
def test_user_deleted_matches_the_consumer_contract(contract) -> None:
    produced = UserDeletedKafkaEvent(user_id=uuid4(), reason="user_request",
                                     metadata=EventMetadata(correlation_id=uuid4(), source_service="user-service"))
    assert verify_event_producer(contract, produced) == []


def test_every_phi_service_has_a_user_deleted_contract() -> None:
    assert len(USER_CONTRACTS) == 7
```
```python
# tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_erasure_fanout_pg.py
"""MVP-029: the anonymised users row, the erasure request and the user.deleted outbox row commit together."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from solace_events.schemas import UserErasureCompletedKafkaEvent
from solace_events.service_plane import shutdown_service_event_plane, start_service_event_plane

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]


async def _svc(us, plane):
    repo, svc_mod = us.mod("infrastructure.postgres_repository"), us.mod("domain.service")
    return svc_mod.UserService(user_repository=repo.PostgresUserRepository(),
                               preferences_repository=repo.PostgresUserPreferencesRepository(),
                               erasure_request_repository=repo.PostgresErasureRequestRepository(), event_plane=plane,
                               integration_settings=svc_mod.ServiceIntegrationSettings(notification_service_url="http://127.0.0.1:9"))


async def test_mvp_029_request_and_outbox_row_commit_with_the_soft_delete(us, cpm_default, superuser_conn) -> None:
    plane = await start_service_event_plane("user-service", pool=cpm_default, strict=False, use_mock=True)
    try:
        svc = await _svc(us, plane)
        uid = (await svc.create_user(email=f"f-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                     display_name="F")).user.user_id
        result = await svc.delete_user(uid, reason="user_request")
        assert result.success is True
        req = await superuser_conn.fetchrow("SELECT user_id, status, expected_services FROM erasure_requests WHERE id = $1",
                                            result.erasure_request_id)
        assert (req["user_id"], req["status"], len(req["expected_services"])) == (uid, "fanout", 7)
        assert await superuser_conn.fetchval("SELECT count(*) FROM event_outbox WHERE event_type = 'user.deleted' "
                                             "AND aggregate_id = $1 AND source_service = 'user-service'", uid) == 1
        expected = us.mod("domain.erasure").EXPECTED_ERASURE_SERVICES
        for service in expected:
            await svc.handle_erasure_completed(UserErasureCompletedKafkaEvent(
                user_id=uid, service_name=service, erasure_request_id=result.erasure_request_id,
                stores={"rows": 0}, completed_at=datetime.now(timezone.utc)))
        assert await superuser_conn.fetchval("SELECT status FROM erasure_requests WHERE id = $1",
                                             result.erasure_request_id) == "complete"
    finally:
        await shutdown_service_event_plane("user-service")


async def test_mvp_029_publish_failure_leaves_no_partial_commit(us, cpm_default, superuser_conn, monkeypatch) -> None:
    plane = await start_service_event_plane("user-service", pool=cpm_default, strict=False, use_mock=True)
    try:
        svc = await _svc(us, plane)
        uid = (await svc.create_user(email=f"g-{uuid4().hex[:8]}@test.invalid", password_hash="x",
                                     display_name="G")).user.user_id

        async def _boom(event, *, conn=None):
            await conn.execute("SELECT 1")
            raise RuntimeError("crash after the domain write")

        monkeypatch.setattr(plane, "publish", _boom)
        assert (await svc.delete_user(uid, reason="user_request")).success is False
        assert await superuser_conn.fetchval("SELECT deleted_at FROM users WHERE id = $1", uid) is None
        assert await superuser_conn.fetchval("SELECT count(*) FROM erasure_requests WHERE user_id = $1", uid) == 0
    finally:
        await shutdown_service_event_plane("user-service")
```
- [ ] **Step 2: Run it and confirm it fails**
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_029_erasure_fanout.py -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_357_user_event_contracts.py -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure/test_mvp_029_erasure_fanout_pg.py -o "addopts=" -p no:logging -q
```
Expected: FAIL.
- `ImportError: InMemoryErasureRequestRepository`, `EXPECTED_ERASURE_SERVICES`; then `TypeError` on `event_plane`/`erasure_request_repository`.
- Today `delete_user` publishes nothing: `publish_user_deleted` is defined nowhere and no publisher is injected.
- There is no request id, no status and no completion handler (MVP-029).
- `user.created` carries the email address.
- The producer contract test fails: no user-service code builds the contracted `user.deleted`.
- On Postgres, no `erasure_requests` row and no `user.deleted` outbox row exist.
- [ ] **Step 3: Implement** the edits listed under Files. `record_completion` runs in one transaction: `SELECT expected_services, status FROM erasure_requests WHERE id = $1 FOR UPDATE`; `INSERT ... ON CONFLICT DO NOTHING`; count the completions of expected services; `UPDATE status = 'complete', completed_at = now()` when all are present. The request's `local_stores` stores the local `report.deleted` counts. No log line in this path carries anything but ids, service names and counts.
- [ ] **Step 4: Run it and confirm it passes**
Run: the same commands. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite, the contract suite and the real-infra directory.
Run:
```
$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/contracts -o "addopts=" -p no:logging -q
$env:SOLACE_REQUIRE_REALINFRA="1"
& "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/accounts-roster-consent-erasure -o "addopts=" -p no:logging -q
```
Expected: all pass. Full fan-out across the seven services is WS-05.15's strict E2E (`test_w3_privacy.py`). It needs every PHI lane's `register_user_erasure_consumer` call.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.16:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.17: Consent gate inputs: required versioned consents, the cns claim, revocation ends sessions, consent_changed events

**Covers:** MVP-100
**Severity:** high
**Wave:** 3
**Depends on:** WS-14.12, WS-14.16, WS-06.8
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-23 (default assumed: chat requires terms_of_service, privacy_policy and ai_companion_disclosure at the current document version, plus data_processing, which is already required today and is kept so nothing weakens; crisis_contact is optional and recorded separately; revoking a required consent ends sessions)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/domain/value_objects.py`:
  - `ConsentType.AI_COMPANION_DISCLOSURE = "ai_companion_disclosure"`;
  - `REQUIRED_CONSENTS: tuple[ConsentType, ...] = (TERMS_OF_SERVICE, PRIVACY_POLICY, DATA_PROCESSING, AI_COMPANION_DISCLOSURE)`;
  - `ConsentType.is_required()` returns membership in `REQUIRED_CONSENTS`.
- Modify: `services/user-service/src/domain/consent.py`. Replace the unused `ConsentService` with:
  - `ConsentPolicySettings(BaseSettings)` with `document_version: str = "1.0"` read from `USER_SERVICE_CONSENT_DOCUMENT_VERSION`;
  - `@dataclass(frozen=True) ConsentStatus(ok: bool, version: str, missing: tuple[str, ...])`;
  - `def evaluate_consent_status(records: Iterable[ConsentRecord], *, document_version: str, now: datetime | None = None) -> ConsentStatus`. Per required type it takes the latest record by `granted_at`; on a tie the revocation wins. That record must be granted, at `document_version`, and not expired.
- Modify: `services/user-service/src/domain/__init__.py` (exports `ConsentStatus`, `ConsentPolicySettings`, `evaluate_consent_status`, `REQUIRED_CONSENTS`; drops the ConsentService names) and `services/user-service/src/__init__.py` (usage docstring)
- Modify: `services/user-service/src/domain/service.py`:
  - `UserService(..., consent_policy: ConsentPolicySettings | None = None)`;
  - `get_consent_status(user_id) -> ConsentStatus` and `get_consent_claim(user_id) -> dict[str, object]`, which returns `{"ok": bool, "version": str}`;
  - `record_consent` refuses `granted=True` for a required type at another version with `CONSENT_VERSION_MISMATCH`;
  - after a successful save it publishes `ConsentChangedKafkaEvent(user_id, consent_type, action)` through `event_plane`, when present. A publish failure is logged with ids only and does not undo the record.
- Modify: `services/user-service/src/api.py`. `POST /users/me/consent` calls `await session_manager.revoke_all_user_sessions(user_id)` when a required consent is revoked (`granted=False`).
- Modify: `services/user-service/tests/test_consent.py` (delete; it tests the removed `ConsentService`, whose "cannot revoke required consent" rule contradicts D-23; record the count drop)
- Test: `services/user-service/tests/test_mvp_100_consent_gate.py` (create)

**Interfaces:**
- Consumes: `ConsentRecord.version` as the document version (WS-14.12); the `event_plane` keyword of `UserService` (WS-14.16); IF-05 `publish` (WS-06.8); `get_session_manager` (api_deps).
- Produces:
  - IF-08 `cns` inputs: `UserService.get_consent_claim(user_id) -> {"ok": bool, "version": str}`, which WS-08 mints into access tokens (R-14-9) and WS-11.20 enforces;
  - `REQUIRED_CONSENTS`, `ConsentStatus`, `evaluate_consent_status`, `ConsentType.AI_COMPANION_DISCLOSURE`;
  - env var `USER_SERVICE_CONSENT_DOCUMENT_VERSION` (R-14-5);
  - `user.consent_changed` events from user-service.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_100_consent_gate.py`
```python
"""MVP-100: required consents gate chat through the cns claim; revoking one ends sessions and emits an event."""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import router as api_router
from src.api_deps import get_current_user, get_session_manager
from src.domain.service import UserService
from src.domain.value_objects import REQUIRED_CONSENTS, ConsentType
from fixtures import InMemoryConsentRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository


class _Plane:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, event, *, conn=None):
        self.published.append(event)
        return event.metadata.event_id


def _svc(plane=None) -> UserService:
    return UserService(user_repository=InMemoryUserRepository(), preferences_repository=InMemoryUserPreferencesRepository(),
                       consent_repository=InMemoryConsentRepository(), event_plane=plane)


@pytest.mark.asyncio
async def test_mvp_100_claim_is_ok_only_with_every_required_consent_at_the_current_version() -> None:
    svc, uid = _svc(), uuid4()
    status = await svc.get_consent_status(uid)
    assert status.ok is False and set(status.missing) == {c.value for c in REQUIRED_CONSENTS}
    assert {c.value for c in REQUIRED_CONSENTS} == {"terms_of_service", "privacy_policy", "data_processing",
                                                    "ai_companion_disclosure"}
    stale = await svc.record_consent(uid, ConsentType.TERMS_OF_SERVICE, True, "0.9")
    assert stale.error_code == "CONSENT_VERSION_MISMATCH"
    for consent in REQUIRED_CONSENTS:
        assert (await svc.record_consent(uid, consent, True, "1.0")).success is True
    assert await svc.get_consent_claim(uid) == {"ok": True, "version": "1.0"}
    await svc.record_consent(uid, ConsentType.PRIVACY_POLICY, False, "1.0")
    assert await svc.get_consent_claim(uid) == {"ok": False, "version": "1.0"}


@pytest.mark.asyncio
async def test_mvp_100_consent_changes_emit_an_event_without_free_text() -> None:
    plane, uid = _Plane(), uuid4()
    await _svc(plane).record_consent(uid, ConsentType.PRIVACY_POLICY, False, "1.0")
    [event] = plane.published
    assert (event.event_type, event.user_id, event.consent_type, event.action) == (
        "user.consent_changed", uid, "privacy_policy", "revoked")


def test_mvp_100_revoking_a_required_consent_ends_sessions_but_optional_does_not() -> None:
    uid, revoked = uuid4(), []

    class _Sessions:
        async def revoke_all_user_sessions(self, user_id) -> int:
            revoked.append(user_id)
            return 1

    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.service = SimpleNamespace(user_service=_svc())
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uid, role="user")
    app.dependency_overrides[get_session_manager] = lambda: _Sessions()
    client = TestClient(app)
    optional = client.post("/api/v1/users/me/consent", json={"consent_type": "marketing_communications",
                                                             "granted": False, "version": "1.0"})
    assert optional.status_code == 201 and revoked == []
    required = client.post("/api/v1/users/me/consent", json={"consent_type": "privacy_policy", "granted": False,
                                                             "version": "1.0"})
    assert required.status_code == 201 and revoked == [uid]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_100_consent_gate.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- `ImportError: REQUIRED_CONSENTS`.
- With a stub, `get_consent_status` is missing.
- A stale version is accepted.
- No consent event is published.
- Revoking `privacy_policy` returns 201 and leaves every session alive (MVP-100: consent is recorded but never enforced or propagated).
- [ ] **Step 3: Implement** the edits listed under Files.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass. The `test_value_objects.py` `is_required` assertions still hold. The chat-side gate is WS-11.20, proven end to end by WS-05.16.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.17:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.18: Audit trail for provisioning, on-call and assignment changes, consent changes, account deletion and PHI reads

**Covers:** MVP-105, MVP-100
**Severity:** high
**Wave:** 3
**Depends on:** WS-14.8, WS-14.9, WS-14.16, WS-14.17, WS-09.11, WS-09.12, WS-04.10
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-10 (default assumed: one HMAC-chained `audit_events` store; rows hold ids and codes only)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/api.py`:
  - `from solace_security import audit_runtime`;
  - the admin on-call route audits `ADMIN_ONCALL_CHANGED`, with details `{"on_call", "backup"}` and the clinician as subject;
  - the assignments route audits `ADMIN_ASSIGNMENT_CHANGED` (details `{"role": "clinician"}`, the patient as subject);
  - `POST /users/me/consent` audits `CONSENT_CHANGED` (details `{"consent_type", "granted", "consent_version"}`);
  - `GET /users/me`, `GET /users/me/consent` and `GET /users/me/preferences` gain `Depends(audit_phi_access("user_profile"))`, `"consent_record"` and `"user_preferences"`.
- Modify: `services/user-service/src/internal_api.py`. The care-team route audits `PHI_READ` with `actor_type="service"`, `actor_id=<service_name>`, `resource_type="care_team"` and the user as subject.
- Modify: `services/user-service/src/domain/service.py`. A successful `delete_user` audits `ACCOUNT_DELETED` with details `{"erasure_request_id", "stores", "count"}`.
- Modify: `scripts/provision_clinician.py`:
  - `default_service_factory` calls `await audit_runtime.configure_service_audit("user-service", pool)`;
  - a successful provision audits `ADMIN_CLINICIAN_PROVISIONED` with `actor_type="system"`, `actor_id="provision_clinician_cli"` and details `{"role", "on_call", "backup"}`;
  - `--assign` audits `ADMIN_ASSIGNMENT_CHANGED`.
- Test: `services/user-service/tests/test_mvp_105_audit_events.py` (create)

**Interfaces:**
- Consumes: IF-14 `audit(...)`, `audit_phi_access(resource_type, *, subject=None)`, `configure_service_audit(service_name, pool)` and `ALLOWED_DETAIL_KEYS` (WS-09.11, WS-09.12); `AuditEventType.ADMIN_CLINICIAN_PROVISIONED`, `ADMIN_ONCALL_CHANGED`, `ADMIN_ASSIGNMENT_CHANGED`, `CONSENT_CHANGED`, `ACCOUNT_DELETED` and `PHI_READ` (WS-09.9); bootstrap audit configuration (WS-04.10).
- Produces: audit rows for every privileged account action in user-service. These satisfy IF-08's "writes an audit event from W3" and WS-09's note that WS-14.18 uses these event types.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_105_audit_events.py`
```python
"""MVP-105 / MVP-100 (with IF-14): privileged account actions, consent changes, deletion and PHI reads are audited without PHI."""
from __future__ import annotations

import asyncio
import contextlib
import io
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts import provision_clinician as cli
from solace_security.audit import AuditEventType
from solace_security.audit_runtime import ALLOWED_DETAIL_KEYS
from solace_security.auth import TokenType
from solace_security.middleware import AuthenticatedUser, get_current_user as shared_user
from src.api import router as api_router
from src.api_deps import get_current_user
from src.domain.service import UserService
from src.infrastructure.password_service import PasswordConfig, PasswordService
from fixtures import (InMemoryAccountTokenRepository, InMemoryConsentRepository, InMemoryErasureRequestRepository,
                      InMemoryUserPreferencesRepository, InMemoryUserRepository)


@pytest.fixture
def audits(monkeypatch):
    calls: list = []

    async def _audit(event_type, **kwargs) -> None:
        calls.append((event_type, kwargs))

    monkeypatch.setattr("solace_security.audit_runtime.audit", _audit)
    monkeypatch.setenv("USER_SERVICE_FRONTEND_URL", "https://app.example.test")
    return calls


class _Plane:
    async def publish(self, event, *, conn=None):
        return event.metadata.event_id


def _svc():
    return UserService(user_repository=InMemoryUserRepository(), preferences_repository=InMemoryUserPreferencesRepository(),
                       consent_repository=InMemoryConsentRepository(), account_token_repository=InMemoryAccountTokenRepository(),
                       erasure_request_repository=InMemoryErasureRequestRepository(), event_plane=_Plane(),
                       password_service=PasswordService(PasswordConfig()))


def _clean(calls):
    for _, kw in calls:
        assert set(kw.get("details") or {}) <= ALLOWED_DETAIL_KEYS
        assert "@" not in repr(kw.get("details")) and "Dr Audit" not in repr(kw)


def test_mvp_105_cli_provisioning_and_assignment_are_audited(audits) -> None:
    svc = _svc()
    patient = asyncio.run(svc.create_user(email="pa@test.invalid", password_hash="h", display_name="Pa")).user

    async def factory():
        async def _close() -> None:
            return None
        return svc, _close

    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert cli.main(["--email", "clin-a@test.invalid", "--display-name", "Dr Audit", "--phone", "+15555550150",
                         "--on-call", "--assign", str(patient.user_id)], service_factory=factory) == 0
    clinician = UUID(out.getvalue().splitlines()[0].split("=", 1)[1])
    types = [t for t, _ in audits]
    assert types == [AuditEventType.ADMIN_CLINICIAN_PROVISIONED, AuditEventType.ADMIN_ASSIGNMENT_CHANGED]
    assert audits[0][1]["actor_type"] == "system" and audits[0][1]["subject_user_id"] == str(clinician)
    assert audits[0][1]["details"] == {"role": "clinician", "on_call": True, "backup": False}
    _clean(audits)


def test_mvp_100_consent_change_and_profile_read_are_audited(audits) -> None:
    uid, svc = uuid4(), _svc()
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.service = SimpleNamespace(user_service=svc)
    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(user_id=uid, role="user")
    app.dependency_overrides[shared_user] = lambda: AuthenticatedUser(user_id=str(uid), token_type=TokenType.ACCESS,
                                                                       roles=["user"])
    client = TestClient(app)
    client.post("/api/v1/users/me/consent", json={"consent_type": "privacy_policy", "granted": True, "version": "1.0"})
    client.get("/api/v1/users/me/consent")
    consent = [kw for t, kw in audits if t == AuditEventType.CONSENT_CHANGED]
    assert consent and consent[0]["details"] == {"consent_type": "privacy_policy", "granted": True,
                                                 "consent_version": "1.0"}
    reads = [kw for t, kw in audits if t == AuditEventType.PHI_READ]
    assert reads and reads[0]["resource_type"] == "consent_record" and reads[0]["subject_user_id"] == str(uid)
    _clean(audits)


@pytest.mark.asyncio
async def test_mvp_105_account_deletion_is_audited(audits) -> None:
    svc = _svc()
    user = (await svc.create_user(email="del@test.invalid", password_hash="h", display_name="Del")).user
    result = await svc.delete_user(user.user_id, reason="user_request")
    [kw] = [kw for t, kw in audits if t == AuditEventType.ACCOUNT_DELETED]
    assert kw["subject_user_id"] == str(user.user_id) and kw["details"]["erasure_request_id"] == str(result.erasure_request_id)
    _clean(audits)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_105_audit_events.py -o "addopts=" -p no:logging -q`
Expected: FAIL. `audits == []` in every test. Clinician provisioning, assignment, consent changes, deletion and PHI reads write no audit row (IF-08 requires an audit event from W3; MVP-091 class).
- [ ] **Step 3: Implement** the audit calls listed under Files. Every call site goes through `audit_runtime.audit(...)` and never awaits the result inside a `try` that hides the response. `audit()` never raises, by IF-14. Details carry only ids, booleans and codes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.18:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.19: Low cleanup: profile URL and bio validation; dead therapy-progress client removed

**Covers:** MVP-329, MVP-331
**Severity:** low
**Wave:** 3
**Depends on:** WS-14.6, WS-14.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/api.py`:
  - `UpdateProfileRequest.avatar_url` must be an absolute `https://` URL with a host, no whitespace, no credentials and at most 500 characters;
  - `bio` (at most 500) rejects control characters other than `\n` and `\t`, and rejects `<` and `>`;
  - violations give a 422 from the request model.
- Modify: `services/user-service/src/domain/service.py`. Remove `UserProgress`, `get_progress`, `_fetch_therapy_progress`, `ServiceIntegrationSettings.therapy_service_url`, and `_get_service_auth_headers` with its `ServiceTokenManager` import, which have no caller after WS-14.6.
- Test: `services/user-service/tests/test_mvp_329_profile_fields.py` (create)

**Interfaces:**
- Consumes: none.
- Produces: the validated profile fields. `USER_SERVICE_THERAPY_SERVICE_URL` is no longer read (R-14-5).

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_329_profile_fields.py`
```python
"""MVP-329 / MVP-331: profile URLs and bios are safe to render later; the dead therapy-progress client is gone."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api import UpdateProfileRequest


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html;base64,PHNjcmlwdD4=",
                                 "http://tracker.example/p.gif", "https://exa mple.test/a.png",
                                 "//evil.example/x.png", "https://user:pw@cdn.example.test/a.png"])
def test_mvp_329_avatar_url_must_be_plain_https(url) -> None:
    with pytest.raises(ValidationError):
        UpdateProfileRequest(avatar_url=url)


@pytest.mark.parametrize("bio", ["<script>alert(1)</script>", "bad\x00byte", "\x1b[31mred"])
def test_mvp_329_bio_rejects_markup_and_control_characters(bio) -> None:
    with pytest.raises(ValidationError):
        UpdateProfileRequest(bio=bio)


def test_mvp_329_plain_values_are_accepted() -> None:
    request = UpdateProfileRequest(avatar_url="https://cdn.example.test/a.png", bio="Line one\nLine two\tend")
    assert request.avatar_url == "https://cdn.example.test/a.png"


def test_mvp_331_dead_therapy_progress_client_is_gone() -> None:
    from src.domain import service

    assert not hasattr(service.UserService, "get_progress")
    assert not hasattr(service.UserService, "_fetch_therapy_progress")
    assert not hasattr(service, "UserProgress")
    assert "therapy_service_url" not in service.ServiceIntegrationSettings.model_fields
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_329_profile_fields.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- Every hostile `avatar_url` and `bio` is accepted: the only check is `max_length` (MVP-329).
- `get_progress`, `UserProgress` and `therapy_service_url` still exist. This is dead code with the wrong port, path and auth that returns made-up defaults (MVP-331).
- [ ] **Step 3: Implement** the validators (`field_validator`) and the deletions.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.19:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.20: User lookup by id is scoped to self or an assigned clinician (supporting MVP-120)

**Covers:** none (supporting task for MVP-120, owned by WS-07, per WS-07's W3 request)
**Severity:** medium
**Wave:** 3
**Depends on:** WS-07.8, WS-14.18
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-21 (default assumed: a clinician reads a user's data only when assigned or after acknowledging a crisis episode within 72 h; admins have no PHI read endpoint)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/api.py`:
  - `GET /users/{user_id:uuid}` depends on the shared `solace_security.middleware.get_current_user`;
  - it calls `await authorize_subject(principal, user_id, checker=get_subject_checker(request))` before any repository call;
  - the role-only `admin/clinician/system` check is removed;
  - new `def get_subject_checker(request) -> SubjectAccessChecker | None` reads `request.app.state.service.subject_checker` (`None` means WS-07's Postgres checker).
- Test: `services/user-service/tests/test_mvp_120_user_lookup_scoping.py` (create)

**Interfaces:**
- Consumes: IF-04 W3 `authorize_subject(principal, subject_user_id, *, checker=None)` and `SubjectAccessChecker` (WS-07.8).
- Produces: `get_subject_checker`. Satisfies WS-07's request for `GET /users/{user_id}`. The `unassigned_clinician` case of WS-05.15's isolation matrix becomes green.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_120_user_lookup_scoping.py`
```python
"""Supports MVP-120: only the user or an assigned clinician can read a user by id; admins and other users cannot."""
from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager
from src.api import router as api_router
from src.domain.entities import User
from src.domain.service import UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository

SECRET = "test-secret-key-32-bytes-long!!!"


def _token(sub, roles) -> dict[str, str]:
    now = int(time.time())
    token = jwt.encode({"sub": str(sub), "type": "access", "jti": str(uuid4()), "iat": now, "exp": now + 600,
                        "iss": "solace-ai", "aud": "solace-ai-api", "roles": roles, "mfa": True, "amr": ["pwd", "otp"]},
                       SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


class _Checker:
    def __init__(self, assigned) -> None:
        self.assigned = assigned

    async def is_assigned(self, clinician_id, subject_user_id) -> bool:
        return (clinician_id, subject_user_id) in self.assigned

    async def acknowledged_recent_episode(self, clinician_id, subject_user_id, *, within_hours=72) -> bool:
        return False


@pytest.fixture
def world(monkeypatch):
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    users = InMemoryUserRepository()
    patient = User(email="pat@test.invalid", password_hash="x", display_name="Pat")
    import asyncio
    asyncio.run(users.save(patient))
    assigned, stranger = uuid4(), uuid4()
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.service = SimpleNamespace(
        user_service=UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository()),
        subject_checker=_Checker({(str(assigned), str(patient.user_id))}))
    return SimpleNamespace(client=TestClient(app), patient=patient, assigned=assigned, stranger=stranger)


@pytest.mark.parametrize("who,roles,status", [("self", ["user"], 200), ("assigned", ["clinician"], 200),
                                              ("stranger", ["clinician"], 403), ("stranger", ["admin"], 403),
                                              ("stranger", ["user"], 403)])
def test_mvp_120_user_lookup_is_subject_scoped(world, who, roles, status) -> None:
    sub = {"self": world.patient.user_id, "assigned": world.assigned, "stranger": world.stranger}[who]
    response = world.client.get(f"/api/v1/users/{world.patient.user_id}", headers=_token(sub, roles))
    assert response.status_code == status
    if status == 403:
        assert response.json()["detail"]["code"] == "SUBJECT_FORBIDDEN" and "pat@" not in response.text
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_120_user_lookup_scoping.py -o "addopts=" -p no:logging -q`
Expected: FAIL.
- The unassigned clinician and the admin get 200 with the user's email: the role-only bypass (MVP-120).
- `self` gets 403 today, because the route admits only admin, clinician and system.
- The route uses the local JWT dependency, so the shared-token requests may also fail with 401.
- [ ] **Step 3: Implement** the route change and `get_subject_checker`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.20:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.21: Admin roster and erasure endpoints require an MFA-verified admin

**Covers:** MVP-103
**Severity:** high
**Wave:** 4
**Depends on:** WS-07.11, WS-14.8, WS-14.16
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-08 (default assumed: TOTP MFA is mandatory for CLINICIAN and ADMIN; tokens without `mfa=true` get 403 on admin endpoints)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/api.py`. Add `Depends(require_mfa_for_roles("admin"))` alongside `require_roles("admin")` on `GET /users/on-call-clinicians`, `PUT /users/{user_id}/on-call`, `PUT /admin/clinicians/{user_id}/on-call`, `POST /admin/assignments` and `GET /admin/erasure-requests/{request_id}`.
- Test: `services/user-service/tests/test_mvp_103_admin_mfa.py` (create)

**Interfaces:**
- Consumes: IF-04 W4 `require_mfa_for_roles(*roles)`, which raises 403 `{"code": "MFA_REQUIRED"}` (WS-07.11). WS-08 forces enrollment for provisioned accounts at first login (IF-17 `mfa_enrollment_required`).
- Produces: IF-08's "ADMIN-only (MFA-verified from W4)". The provisioning CLI's accounts carry no factor, so WS-08's login forces enrollment; no CLI change is needed.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_103_admin_mfa.py`
```python
"""MVP-103 (W4): admin roster and erasure endpoints refuse an admin token without MFA."""
from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from solace_security import middleware as mw
from solace_security.auth import AuthSettings, InMemoryTokenBlacklist, JWTManager
from src.api import router as api_router
from src.domain.service import UserService
from fixtures import InMemoryErasureRequestRepository, InMemoryUserPreferencesRepository, InMemoryUserRepository

SECRET = "test-secret-key-32-bytes-long!!!"
CLIN, PAT, REQ = uuid4(), uuid4(), uuid4()
ROUTES = [("PUT", f"/api/v1/admin/clinicians/{CLIN}/on-call", {"on_call": True, "backup": False}),
          ("POST", "/api/v1/admin/assignments", {"clinician_id": str(CLIN), "user_id": str(PAT)}),
          ("GET", f"/api/v1/admin/erasure-requests/{REQ}", None),
          ("GET", "/api/v1/users/on-call-clinicians", None),
          ("PUT", f"/api/v1/users/{CLIN}/on-call", {"is_on_call": True})]


def _admin(mfa: bool) -> dict[str, str]:
    now = int(time.time())
    token = jwt.encode({"sub": str(uuid4()), "type": "access", "jti": str(uuid4()), "iat": now, "exp": now + 600,
                        "iss": "solace-ai", "aud": "solace-ai-api", "roles": ["admin"], "mfa": mfa,
                        "amr": ["pwd", "otp"] if mfa else ["pwd"]}, SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch):
    manager = JWTManager(AuthSettings(secret_key=SECRET), token_blacklist=InMemoryTokenBlacklist())
    monkeypatch.setattr(mw, "_get_jwt_manager", lambda: manager)
    app = FastAPI()
    app.include_router(api_router, prefix="/api/v1")
    app.state.service = SimpleNamespace(user_service=UserService(
        user_repository=InMemoryUserRepository(), preferences_repository=InMemoryUserPreferencesRepository(),
        erasure_request_repository=InMemoryErasureRequestRepository()))
    return TestClient(app)


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_mvp_103_admin_routes_require_mfa(client, method, path, body) -> None:
    denied = client.request(method, path, json=body, headers=_admin(False))
    assert denied.status_code == 403 and denied.json()["detail"]["code"] == "MFA_REQUIRED"
    allowed = client.request(method, path, json=body, headers=_admin(True))
    assert allowed.status_code != 403
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_103_admin_mfa.py -o "addopts=" -p no:logging -q`
Expected: FAIL for every route. A password-only admin token passes `require_roles("admin")`, so the denied request returns 204, 201, 404 or 200 instead of 403 `MFA_REQUIRED`. IF-08 requires MFA-verified admins from W4.
- [ ] **Step 3: Implement** the added dependency on the five routes.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass. The W1 admin tests already mint `mfa: true` tokens.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.21:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-14.22: The profile locale is canonical BCP-47-subset text, defaults to en-US and feeds the `loc` claim (D-29)

**Covers:** none (supports D-29: the user's locale decides which crisis resources are shown)
**Supports:** MVP-221 (closed by WS-08.7 for registration; this task applies the same locale validation to the profile route)
**Severity:** medium
**Wave:** 3
**Depends on:** WS-14.19, WS-08.7
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** D-29 (decided 2026-10-19: several countries, US first. WS-14 owns the profile locale field. It is validated against the BCP-47 subset `ll` or `ll-CC` (WS-08.7's `normalize_locale`, the same grammar as `solace_clinical.canonical_locale`) and defaults to `en-US`. WS-08.21 issues it as the access-token claim `loc`, so screening never looks it up. Admitting a user whose locale has no populated resources is an operator and cohort decision (D-31); engineering guarantees the generic fallback and its alert)
**Real infra:** no

**Files:**
- Modify: `services/user-service/src/api.py`: `UpdateProfileRequest.locale` gets a `field_validator` that returns `normalize_locale(value)` for a non-`None` value and leaves `None` (unchanged) alone. A malformed tag gives a 422 from the request model, never a stored value and never a 400 carrying validator text.
- Test: `services/user-service/tests/test_mvp_221_profile_locale.py` (create)

**Interfaces:**
- Consumes: `normalize_locale` (`src.domain.entities`, WS-08.7); `UserService.update_user` and `User.update_profile` (the entity validator also calls `normalize_locale` after WS-08.7).
- Produces (D-29): the profile locale field. `User.locale` is always canonical `ll` or `ll-CC`, or the default `en-US`; `PUT /api/v1/users/me {locale}` canonicalizes it and rejects malformed tags with 422. WS-08.21 reads `User.locale` into the `loc` claim at every issuance, so a change takes effect at the next refresh (at most the 15-minute access-token lifetime). WS-08.21's parity test keeps this grammar identical to `solace_clinical.canonical_locale`.

- [ ] **Step 1: Write the failing test** in `services/user-service/tests/test_mvp_221_profile_locale.py`
```python
"""D-29: the profile locale is canonical BCP-47-subset text, defaults to en-US and is what the `loc` claim carries."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api import UpdateProfileRequest
from src.domain.entities import User
from src.domain.service import UserService
from fixtures import InMemoryUserPreferencesRepository, InMemoryUserRepository


@pytest.mark.parametrize("raw,stored", [("en_us", "en-US"), ("EN-us", "en-US"), ("fr-FR", "fr-FR"), ("pt-br", "pt-BR")])
def test_d29_profile_locale_is_canonicalized_at_the_route_model(raw, stored) -> None:
    assert UpdateProfileRequest(locale=raw).locale == stored


@pytest.mark.parametrize("raw", ["english", "en-USA", "e1-US", "en--US", "<b>en</b>", ""])
def test_d29_malformed_profile_locale_is_a_422_not_a_stored_value(raw) -> None:
    with pytest.raises(ValidationError):
        UpdateProfileRequest(locale=raw)


def test_d29_omitted_locale_leaves_the_profile_alone_and_new_users_default_to_en_us() -> None:
    assert UpdateProfileRequest().locale is None
    assert User(email="a@example.com", password_hash="x", display_name="A").locale == "en-US"


async def test_d29_profile_update_persists_the_canonical_locale() -> None:
    users = InMemoryUserRepository()
    user = User(email="b@example.com", password_hash="x", display_name="B")
    await users.save(user)
    service = UserService(user_repository=users, preferences_repository=InMemoryUserPreferencesRepository())
    result = await service.update_user(user.user_id, locale="fr_fr")
    assert result.success is True and (await users.get_by_id(user.user_id)).locale == "fr-FR"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests/test_mvp_221_profile_locale.py -o "addopts=" -p no:logging -q`
Expected: FAIL on the route-model cases. `UpdateProfileRequest.locale` is a bare `str | None`, so `"en_us"` comes back unchanged, and `"english"`, `"<b>en</b>"` and `""` raise nothing. The PUT then reaches the entity, whose validator error comes back as a 400 that carries the validator text. The entity-default and service-level cases are locks: they pass once WS-08.7's `normalize_locale` is in the entity validator. Before WS-08.7, `update_user(locale="fr_fr")` failed because the old check accepted only `xx-XX`.
- [ ] **Step 3: Implement** the `field_validator` on `UpdateProfileRequest.locale`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** run the owning suite. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/user-service/tests -o "addopts=" -p no:logging -q`. Expected: all pass (`test_mvp_329_profile_fields.py` from WS-14.19 included).
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-14.22:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| # | Owner WS | Exact change | Files | Needed by |
|---|---|---|---|---|
| R-14-1 | WS-01 | W0 (WS-01.5): the user-service image copies `scripts/` to `/app/scripts/` (`COPY --chown=appuser:appuser scripts/ ./scripts/`), so `python -m scripts.provision_clinician` runs in the container with `PYTHONPATH=/app:/app/src`. The directory exists at W0; the CLI file arrives in W1. | `services/user-service/Dockerfile` | WS-14.9 (in-image proof through WS-05.8) |
| R-14-2 | WS-03 | W0 phase-A env sync (WS-03.2): after WS-14.1/WS-14.2, user-service reads only `POSTGRES_*` and `REDIS_URL`. Mark `USER_DB_HOST/PORT/NAME/USER/PASSWORD` and `USER_REDIS_HOST/PORT/DB/PASSWORD` retired for user-service. | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-14.2 |
| R-14-3 | WS-03 | W1 phase-A env sync (WS-03.6): add the registry entry `USER_SERVICE_FRONTEND_URL` (services `[user-service]`, not secret, required_in_strict true, dev_default `http://localhost:3000`, note "base of verification and set-password links; never localhost in strict"). Retire `USER_SERVICE_NOTIFICATION_SERVICE_URL`; user-service reads `SERVICE_NOTIFICATION_SERVICE_URL`. | `deploy/env/env-contract.yaml`, `docker-compose.yml`, `docker-compose.prod.yml` | WS-14.6, WS-14.9 |
| R-14-4 | WS-03 | W3 phase A (WS-03 has no W2 lane): mark `FERNET_FIELD_KEY` and `USER_SERVICE_FIELD_ENCRYPTION_KEY` retired, since WS-14.10 stops reading them in W2, and drop them from the IF-22 secret list. `FERNET_TOKEN_KEY` stays. | `deploy/env/env-contract.yaml`, `deploy/render_secrets.py` | none (cleanup after WS-14.10) |
| R-14-5 | WS-03 | W3 phase-A env sync: add the registry entry `USER_SERVICE_CONSENT_DOCUMENT_VERSION` (user-service, not secret, not required, dev_default `"1.0"`). Retire `USER_SERVICE_THERAPY_SERVICE_URL`. | `deploy/env/env-contract.yaml` | WS-14.17, WS-14.19 |
| R-14-6 | WS-05 | W1 (WS-05.8 helpers and overlay): `docker-compose.test.yml` sets `USER_SERVICE_FRONTEND_URL=http://app.e2e.invalid` in both profiles. `provision_clinician(...)` parses `set_password_url`, calls `POST /api/v1/users/account-setup {token, new_password}` through the edge, then logs in. `make_user` extracts `<base>/verify-email?user_id=<uuid>&token=<t>` from mailpit and calls `POST /api/v1/auth/verify-email?user_id=..&token=..`; the `verify_email_via_sql` shortcut then expires. | `docker-compose.test.yml`, `src/solace_testing/e2e.py` | WS-14.6, WS-14.9 (E2E proof) |
| R-14-7 | WS-05 | W3 (WS-05.16): the consent journey accepts `terms_of_service`, `privacy_policy`, `data_processing` and `ai_companion_disclosure` at version `"1.0"` through `POST /api/v1/users/me/consent`, then refreshes or logs in to obtain `cns.ok=true`. | `tests/e2e/test_w3_consent_oauth.py`, `src/solace_testing/e2e.py` | WS-14.17 (E2E proof) |
| R-14-8 | WS-08 | W2 phase B: `POST /api/v1/auth/verify-email` accepts `token` alone (IF-17), passing `user_id` when present; `UserService.verify_email(user_id, token)`, with `user_id` a `UUID` or `None`, supports both since WS-14.6. | `services/user-service/src/api_auth.py` | none of WS-14's (IF-17 conformance) |
| R-14-9 | WS-08 | W3 phase B. Access tokens minted at login, refresh and the OAuth exchange carry `cns = await user_service.get_consent_claim(user_id)`. `DELETE /api/v1/users/me` (with `X-Reauth-Token`) maps the result: success gives 202 `{"erasure_request_id": result.erasure_request_id}`; `USER_NOT_FOUND` gives 404; `DELETE_FAILED` gives 503 `{error:{code:'ERASURE_INCOMPLETE'}}`. It revokes sessions only on success, and passes `reason="user_request"`. `SessionManager.revoke_all_user_sessions(user_id)` also revokes the user's sids in the shared blacklist, so a consent revocation reaches every service's per-frame check. | `services/user-service/src/api_auth.py`, `services/user-service/src/auth.py`, `services/user-service/src/infrastructure/jwt_service.py` | WS-14.16, WS-14.17 (closure of MVP-029 and MVP-100) |
| R-14-10 | WS-08 | W3 phase B: `identity_wiring.start_identity` sets `app.state.service.oauth_account_repository`, which exposes `async delete_for_user(user_id: UUID) -> int` over `oauth_accounts`. `oauth_accounts.email` goes through IF-13 with owner `user_id`. WS-14.16 registers it as erasure store `oauth_accounts` when present. | `services/user-service/src/infrastructure/oauth_account_repository.py`, `services/user-service/src/identity_wiring.py` | WS-14.16 (store coverage) |
| R-14-11 | WS-13 | W1: write the consumer contracts `tests/contracts/notification__user/on_call_roster.json` and `care_team.json` with the IF-08 shapes, and call the internal routes with a `READ_ONCALL_ROSTER` service token. | `tests/contracts/notification__user/*.json`, `services/notification-service/src/consumers.py` | WS-14.12 |
| R-14-12 | WS-12 | W1 (D-28): safety-service stops calling `GET /api/v1/users/on-call-clinicians`. WS-14.8 makes it ADMIN-only, so the ClinicianRegistry would otherwise receive 403. | `services/safety_service/src/infrastructure/clinician_registry.py` | WS-14.8 (no regression on the safety side) |
| R-14-13 | WS-08 | W2 phase B: WS-08.4 does not edit `services/user-service/tests/conftest.py` (WS-14 stewards it; WS-14.10 is its only W2 editor), and its Files entry should name that file only as a read source, without a path, so the plan checker stops counting it as an edit. Because WS-14.10 makes every users write call `get_phi_codec()`, WS-08.4 depends on WS-14.10, and the setdefault values that WS-08.4 copies into `tests/realinfra/identity-auth-mfa/conftest.py` include `ENCRYPTION_MASTER_KEY` = `"ab" * 32`, `ENCRYPTION_SEARCH_HASH_SALT` and `ENCRYPTION_KEY_ID=test-k1`, the values WS-14.10 adds. | `tests/realinfra/identity-auth-mfa/conftest.py` | WS-14.10 (no same-wave conftest conflict; WS-08's real-infra suite keeps working after the codec adoption) |

## New interfaces

Beyond IF-08 and IF-16 as written in the skeleton:

**user-service runtime and split**
- `src.main.SPEC`, `src.main.ROUTERS`, `build_user_service_state(runtime, app) -> ServiceState` and `create_app() -> FastAPI` (WS-14.2).
- `src.api_auth.auth_router` (one aggregated router, mounted at prefix `""`). The `src.api_deps` names. `src.identity_wiring.start_identity(runtime, app)` / `stop_identity()`. `src.internal_api.internal_router`, mounted at `/api/v1/internal` (WS-14.3).
- `ServiceIntegrationSettings`: reads `SERVICE_NOTIFICATION_SERVICE_URL` and `USER_SERVICE_FRONTEND_URL`, has `validate_for_environment(environment) -> None`, and sets `populate_by_name=True`.

**Repositories**
- `PostgresUserRepository(schema="public", *, pool_name="default")`, with the same signature for the preferences and consent repositories.
- `UserRepository.save_with_preferences(user, preferences, *, preferences_repository) -> User`; module constants `USER_INSERT_SQL` and `PREFERENCES_INSERT_SQL`.
- `PostgresUserRepository.update(user, *, conn=None)`.
- `UserRepository.list_on_call_roster()`, `list_care_team(patient_id)` and `set_on_call_flags(user_id, *, on_call, backup)`; `ConsentRepository.has_active_consent(user_id, consent_type) -> bool`.
- `AccountTokenRepository` with `issue`, `consume`, `revoke` and `delete_for_user`; `PostgresAccountTokenRepository`; token purposes `email_verification` and `account_setup`; tokens stored as sha256 hex.
- `ErasureRequestRepository` with `finalize`, `record_completion` and `get_status`; `PostgresErasureRequestRepository`.
- `USER_PII_FIELDS = ("email", "display_name", "phone_number", "bio")`; AAD field `users.<column>` with owner `users.id`; `email_search_hash(email, codec)` uses purpose `users.email` on the stripped, lower-cased address.

**Domain types and UserService**
- `ClinicianContact`, `CareTeam`, `ConsentType.CRISIS_CONTACT`, `ConsentType.AI_COMPANION_DISCLOSURE`, `REQUIRED_CONSENTS`.
- `ConsentPolicySettings` (`USER_SERVICE_CONSENT_DOCUMENT_VERSION`), `ConsentStatus`, `evaluate_consent_status`.
- `EXPECTED_ERASURE_SERVICES` (the seven PHI services); `ERASURE_TRACKER_GROUP = "solace-group-erasure-user-service"`, which follows the IF-05 `solace-group-erasure-<service>` pattern and subscribes to `user.erasure.completed`; `ErasureStatus`; `ServiceCompletion`.
- `build_user_data_erasure(user_repository, preferences_repository, *, account_token_repository=None, erase_events=None, oauth_account_repository=None)`. The users row is not a store; its anonymisation is the final transactional step.
- UserService:
  - constructor keywords `account_token_repository`, `http_client_factory`, `erasure_request_repository`, `event_plane` and `consent_policy`;
  - methods `verify_email(user_id: UUID | None, token)`, `get_on_call_roster()`, `get_care_team(user_id)`, `set_on_call_status(user_id, is_on_call, *, backup=False)`, `provision_privileged_account(...) -> ProvisionResult(user_id, setup_token)`, `complete_account_setup(token, new_password_hash)`, `handle_erasure_completed(event)`, `get_erasure_status(request_id)`, `get_consent_status(user_id)` and `get_consent_claim(user_id)`;
  - `DeleteUserResult.erasure_request_id: UUID | None`.

**Routes, links, CLI and events**
- D-29 profile locale (WS-14.22): `PUT /api/v1/users/me {locale}` canonicalizes to `ll` or `ll-CC` with `normalize_locale` and answers 422 for a malformed tag; `User.locale` defaults to `en-US` and is the source of WS-08.21's `loc` access-token claim.
- Routes: `POST /api/v1/users/account-setup {token, new_password}`; `PUT /api/v1/admin/clinicians/{user_id}/on-call`; `POST /api/v1/admin/assignments`; `GET /api/v1/admin/erasure-requests/{request_id}`.
- Link formats: `{USER_SERVICE_FRONTEND_URL}/verify-email?user_id=<uuid>&token=<t>` and `{USER_SERVICE_FRONTEND_URL}/set-password?token=<t>` (72 h, single use).
- CLI exit codes: 0 ok, 1 error, 2 usage, 3 account exists. `main(argv=None, *, service_factory=None, configure_logging=False)`.
- `user.deleted` `reason` codes: `user_request`, `admin_request`. `user.created` carries `email=""` and `display_name=""`.
- Slot 012 tables and columns: `users.is_backup`, `account_tokens`, `users.email_hash` (`uq_users_email_hash`), text PII columns, `consent_records.document_version` and `expires_at`, `erasure_requests` (with `uq_erasure_requests_open_user`), `erasure_service_completions`.

**IF-16 helper and tests**
- IF-16 helper details: `USER_EVENTS_TOPIC`, `erasure_group_id(service_name)`, `handle_user_deleted(...)`, and the retry and dead-letter rules in WS-14.13.
- Test helpers: `services/user-service/tests/runtime_fakes.fake_runtime(...)`. The real-infra slug conftest provides `us`, `cpm_default`, `superuser_conn`, `seed_user`, `build_repo` and the autouse `phi_codec`.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-029 | blocker | WS-14.13 (IF-16 helper), WS-14.14 (erasure tables), WS-14.16 (transactional publish, completion tracking, status endpoint); closure needs every PHI lane's consumer call (WS-11, WS-12, WS-13, WS-15, WS-16, WS-17, WS-18) and R-14-9, proven by WS-05.15 |
| MVP-034 | blocker | WS-14.1 |
| MVP-035 | blocker | WS-14.6; E2E proof through R-14-6 |
| MVP-100 | high | WS-14.17, WS-14.18 (consent audit); chat gate WS-11.20; token claim R-14-9 |
| MVP-103 | high | WS-14.8, WS-14.21 |
| MVP-105 | high | WS-14.4, WS-14.8, WS-14.9, WS-14.18 |
| MVP-106 | high | WS-14.5 |
| MVP-107 | high | WS-14.2, WS-14.7 |
| MVP-216 | medium | WS-14.15 |
| MVP-218 | medium | WS-14.11, WS-14.10 |
| MVP-220 | medium | WS-14.11, WS-14.12 |
| MVP-230 | medium | WS-14.4, WS-14.6 |
| MVP-327 | low | WS-14.5 |
| MVP-328 | low | WS-14.2 |
| MVP-329 | low | WS-14.19 |
| MVP-331 | low | WS-14.19 |

**Incoming requests, each planned no later than the requesting task's wave**
- WS-01.7 → WS-14.2 (W0).
- R-02.8 for WS-02.3/WS-02.6/WS-02.8 → WS-14.1 (preferences under `acting_as_user`, no `USER_DB_*` client; W0).
- WS-04.5 → WS-14.2 (W0).
- WS-05.8 (WS-05's request 5: the two-line CLI with `--assign` and `--role`) → WS-14.9 (W1).
- WS-08 R-1 (the W0 split, `TokenResponse` in `api_deps`, handler names unchanged) → WS-14.3 (W0).
- WS-08 R-8 (`delete_user` return value, for WS-08.8) → WS-14.16 (W3), with the `DeleteUserResult` wrapper that WS-08.8 consumes.
- WS-09 CR-10 (codec adoption, for WS-09.15) → WS-14.10 (W2).
- WS-06.13 JSONB list → WS-14.1 (W0, earlier than its W2 deadline).
- WS-06.12 `user.created` → WS-14.16 (W3).
- WS-07.9 → WS-14.8 (W1).
- WS-07 `authorize_subject` → WS-14.20 (W3).
- WS-09.15 → WS-14.10 (W2).
- WS-17.11 → WS-14.13 (W3, phase A).

**Interface consistency check**
- IF-08 field names (`user_id, display_name, email, phone_e164, role, is_backup`; care-team `contact_email, contact_phone_e164, crisis_contact_consent, clinicians`), paths, the `service:read:oncall` guard, the CLI flags and the two-line output match the skeleton. Two documented readings:
  - IF-08 says user tokens get 403 on the roster. The shared `get_current_service` answers a user access token with 401, so WS-14.8 asserts "401 or 403 and no roster data".
  - The care-team display name is nulled without crisis-contact consent, following D-12's "display name and contact only with consent".
- IF-16 step 1 says `delete_user(user_id, reason) -> UUID`, and WS-08's request R-8 repeats it. WS-14.16 returns `DeleteUserResult(success, error, error_code, erasure_request_id)`, which is exactly what WS-08.8's Interfaces line consumes (`success`, `error_code`, `erasure_request_id: UUID | None`, mapped to 202/404/503). This keeps the verbatim-moved `DELETE /users/me` in `api_auth.py` (WS-08) working between WS-14.16 and WS-08's W3 route change (R-14-9). The id is the same value and only the wrapper differs. This is recorded for a ledger Ruling.
- IF-16 helper signature `register_user_erasure_consumer(plane, *, service_name, erasure, purge_caches=None) -> None` matches the skeleton and the calls written in WS-11, WS-12, WS-15 and WS-17 (`from solace_infrastructure.gdpr import consumer` / `...gdpr.consumer.register_user_erasure_consumer`). The group `solace-group-erasure-<service_name>` matches WS-11's `solace-group-erasure-orchestrator-service`.
- IF-13 names (`get_phi_codec`, `search_hash(value, *, purpose)`, `active_key_id`, `PHIDecryptError`, `reset_phi_codec`, envelope `enc:v1:<key_id>:`) match WS-09.7/WS-09.8. WS-14.10 is the task id WS-09.15 depends on.
- WS-09's prose names WS-14.13 (`ACCOUNT_ERASED_IN_SERVICE`) and WS-14.18 (the ADMIN_*, CONSENT_CHANGED and ACCOUNT_DELETED types). Those are exactly these tasks. Audit details use only `ALLOWED_DETAIL_KEYS` members (`service`, `stores`, `count`, `erasure_request_id`, `role`, `on_call`, `backup`, `consent_type`, `consent_version`, `granted`).
- IF-05: `publish(event, *, conn=None)`, `subscribe(topic, event_types, handler, *, group_id, max_attempts=5)`, `UserErasureCompletedKafkaEvent(user_id, service_name, erasure_request_id, stores, completed_at)`, `erase_user_events(user_id) -> int` and topic `solace.users` are used verbatim. The real-infra test starts the plane as WS-06.11's test does (`use_mock=True`).
- IF-03: every RLS table (`user_preferences`, `consent_records`) is touched only inside `acting_as_user` or a request context, through `ConnectionPoolManager.acquire/transaction("default")`. The new tables are non-RLS service tables, like `users`.

**Review focus (numbered 1 to 5) and the tests that pin them in WS-14 code**
1. Crisis phrasing false negatives: not in WS-14 code; the roster and care team carry no detection logic.
2. Partial failures on the alert/erasure paths:
   - WS-14.8: the roster returns only ACTIVE on-call clinicians, and the care team still lists clinicians when consent is absent, so pages are never stranded by an empty list.
   - WS-14.6: tokens are consumed exactly once under concurrency.
   - WS-14.16: a crash after the users UPDATE but before the outbox row leaves no partial commit, and duplicate completions are idempotent.
3. Context-less work and foreign ids under `solace_app`:
   - WS-14.1: a preferences insert without context is rejected, and registration sets the context.
   - WS-14.8: the care-team consent is read with no request context.
   - WS-14.13: the consumer erases as the user and restores the context.
   - WS-14.15: deletion without a request context really removes the RLS rows.
   - WS-14.20: an unassigned clinician and an admin are denied.
4. PHI in unplanned sinks:
   - WS-14.5: no exception text in results or logs, and no email in logs.
   - WS-14.13: the retry error carries no user id or text.
   - WS-14.16: `user.created` carries no contact data.
   - WS-14.18: audit details carry no PHI.
   - WS-14.9: the CLI never echoes the email.
5. Strict boots with partial configuration:
   - WS-14.2: missing Redis is fatal in strict.
   - WS-14.6: a localhost link base is refused in strict.
   - WS-14.10: ciphertext under a retired key still decrypts.

REVIEW_FOCUS: 2=WS-14.6,WS-14.8,WS-14.16; 3=WS-14.1,WS-14.8,WS-14.13,WS-14.15,WS-14.20; 4=WS-14.5,WS-14.9,WS-14.13,WS-14.16,WS-14.18; 5=WS-14.2,WS-14.6,WS-14.10

**Ownership notes**
- WS-14.3 creates `services/user-service/src/api_auth.py`, `api_deps.py` and `identity_wiring.py`, which the skeleton assigns to WS-08. The shared-files protocol for each of the three names WS-14 as the W0 phase-B creator (a verbatim move) and moves ownership to WS-08 at the W0 barrier. No later WS-14 task edits them.

**Open risks**
- WS-08 has no W0 or W1 lane. WS-14.3 edits import lines of six existing test files during the verbatim move, as the owner of every moved symbol until the W0 barrier. The W1 verification-link format relies on WS-05's harness change (R-14-6).
- The `EXPECTED_ERASURE_SERVICES` set follows IF-16 and WS-05.14 (seven PHI services). analytics-service is not expected. If WS-18 adds an analytics erasure consumer, the tuple and WS-05.14's contracts must change together.
- Nothing alerts on an erasure request stuck in `fanout` past the D-24 SLA. A gauge plus a WS-04 rule is a W5 candidate; the admin status endpoint is the manual check.
- The account-setup route is unauthenticated. It sits outside the `/auth/*` rate-limit bucket, which matches only `/auth/*` paths. Its tokens are 256-bit, single-use and expire after 72 h, so guessing is infeasible. A path entry in WS-07's auth policies can follow by ledger Ruling.
- `bio` rejects `<` and `>` (for example "I <3 dogs"). This is a deliberate over-restriction until a client renders profiles.
- Real-infra tests load user-service under the alias package `user_service_src`. A future absolute `src.` import inside user-service would break only those tests; WS-01.7's import-root guard forbids `src.solace_*`.
- The `notification__user` provider check in WS-14.12 depends on WS-13 writing the contract files (R-14-11). If they are missing at the W2 barrier, the test fails visibly; it does not skip.
- If WS-06.8's plane does not accept `subscribe` after start (R-17-05), the completion subscription in `main.py` must move before the runtime starts the plane. That would be a ledger Ruling with WS-04/WS-06.
- R-14-13 asks WS-08.4 to depend on WS-14.10 and to copy the `ENCRYPTION_*` test values. If WS-08 declines, WS-08's real-infra identity suite fails once WS-14.10 lands, because every users write then needs the codec keys.

## Review responses

- Error: Wave 2, `services/user-service/tests/conftest.py` edited by WS-08.4 and WS-14.10: rejected as a WS-14 change, with a request added. WS-14 stewards that file (skeleton shared_files, `services/user-service/tests`), and WS-14.10 is its only W2 editor. WS-08.4 does not edit it: WS-08.4's Files entry creates `tests/realinfra/identity-auth-mfa/conftest.py`, which "only reads the user-service suite and edits nothing there", and WS-08's header says `tests/conftest.py` belongs to WS-14. The checker counted a path that WS-08.4 names only as the source of copied values. New request R-14-13 asks WS-08 to drop the path from WS-08.4's Files, to depend on WS-14.10, and to copy the three `ENCRYPTION_*` values that WS-14.10 adds, since `get_phi_codec()` then runs on every users write.
- Error: WS-14.10 modifies `tests/realinfra/accounts-roster-consent-erasure/test_mvp_105_provision_cli_pg.py`, said not to be created earlier: fixed. The finding was partly wrong: WS-14.9 (wave 1) already created the file, and WS-14.10 depends on WS-14.9. The parser counts only "Create:" lines, though, and WS-14.9 listed the file as "Test: ... (create)". WS-14.9's line now reads `- Create: ... (real-infra test)`, so the parsed index shows who creates it.
- Error: plan a task for WS-05's request 5 (two-line `scripts.provision_clinician` output, `--assign`, `--role CLINICIAN/ADMIN`, runs in the user-service image): fixed. The task already existed: WS-14.9 produces exactly that CLI, and its tests check both stdout lines, `--assign` and `--role`. The checker could not see the match because the request's Files cell parsed as empty (the escaped `\|` in `CLINICIAN\|ADMIN` split the table row). WS-14.9's Produces line now names "WS-05's request 5, needed by WS-05.8" and the in-image dependency R-14-1.
- Error: plan a task for WS-08's R-1 (W0 split): fixed. The split was already planned as WS-14.3; the checker parsed the request's Files cell as empty (brace and wildcard paths). Fixed one real gap it pointed to: R-1 puts `TokenResponse` in `api_deps.py`, and WS-14.3 had moved it into `api_auth.py`, which would break line 36 of user-service's `oauth.py` (`from .api import TokenResponse`, a WS-08 file). WS-14.3 now moves `TokenResponse` to `api_deps.py`, re-imports it in `api.py`, keeps the handler names (new test `test_split_keeps_handler_names`) and names R-1 in its Produces line.
- Error: plan a task for WS-08's R-8 (`delete_user(...) -> UUID`, errors raise, for WS-08.8): rejected as a signature change. WS-14.16 already supplies the erasure id for WS-08.8. WS-08.8's own Interfaces line consumes `DeleteUserResult` with `success`, `error_code` and `erasure_request_id: UUID | None` and maps them to 202/404/503 (WS-08 plan, WS-08.8 Interfaces and the `DELETE /users/me` bullet). Switching to a bare UUID that raises would break WS-08.8 as written. The IF-16 wrapper difference stays recorded for a ledger Ruling, and WS-14.16's Produces line now names R-8 and WS-08.8. The checker could not match the request because the `str \| None` in its cell split the row.
- Error: plan a task for WS-09's CR-10 (codec adoption for WS-09.15): fixed. The adoption was already planned as WS-14.10; the request's Files cell ("the eight ADOPTERS files") holds no path. WS-14.10's Produces line now names CR-10 and WS-09.15 and states all three conditions: it imports `solace_security.phi_codec`, user-service calls neither legacy function (the only mention is the REV-12 comment at `main.py:190-195`, which this task removes), and it writes `codec.active_key_id` to `encryption_key_id`, as `test_mvp_218_pii_is_ciphertext_and_found_by_blind_index` asserts.
- Warning: WS-14.1 edits `services/user-service/src`, which no workstream owns: fixed (false positive). The path was the load location of the alias package, named in a sub-bullet of the conftest Create line, not an edit. It is reworded without the path.
- Warning: WS-14.1 lists `services/user-service/src` under Create although it exists: fixed by the same rewording; WS-14.1 creates only the slug conftest and its two test files.
- Warning: WS-14.3 edits `api_auth.py`, owned by WS-08: rejected (sanctioned). The shared-files protocol for this file says WS-14 creates it in W0 phase B as a verbatim move and hands it to WS-08 at the W0 barrier (Self-review, Ownership notes).
- Warning: WS-14.3 edits `api_deps.py`, owned by WS-08: rejected (sanctioned). Its shared-files protocol says WS-14 creates it in W0 as a verbatim move, and WS-08 owns it after the barrier.
- Warning: WS-14.3 edits `identity_wiring.py`, owned by WS-08: rejected (sanctioned). Its shared-files protocol says WS-14 creates it in W0 from main.py's composition, and WS-08 owns it after the barrier. The shared-files protocol for each file makes WS-14 the W0 phase-B creator through a verbatim move, and ownership passes to WS-08 at the W0 barrier.
- Warning: WS-14.13 lists `src/solace_infrastructure/gdpr/__init__.py` under Create although it exists: fixed (false positive). The plan already said the module is unchanged. The sentence is reworded without the path, so only `consumer.py` is a Create.
- Review focus: the one-line focus summary is added under the existing numbered review-focus list. Focus area 1 has no WS-14 code.
- Additional (found while checking the table): R-14-8's cell held an escaped pipe (`UUID` or `None` written with a pipe), which split the row and hid its Files cell from the checker. It is reworded without the pipe.
- decision 2026-10-19 D-29: new task WS-14.22 (W3 phase B, after WS-14.19, which edits the same request model) makes WS-14 the owner of the profile locale field. `UpdateProfileRequest.locale` is validated and canonicalized with `normalize_locale` (WS-08.7), the BCP-47 subset `ll` or `ll-CC` that `solace_clinical.canonical_locale` shares, and a malformed tag is a 422. New users default to `en-US`, and the stored value feeds WS-08.21's `loc` claim, so the orchestrator never looks the locale up at screen time. The new RED test file is `test_mvp_221_profile_locale.py`. The Waves list, file map and New interfaces are updated. No cross-workstream request: the entity validator is WS-08.7's and the claim is WS-08.21's.
