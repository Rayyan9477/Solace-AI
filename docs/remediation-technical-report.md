# Solace-AI Technical Remediation Report

**Date:** January 2026
**Codebase:** Solace-AI Mental Health Platform
**Scale:** 554 Python files, ~185,700 LOC, 8 microservices

---

## Table of Contents

1. [Phase 1: SQL Injection + Credential Hardcoding](#phase-1-sql-injection--credential-hardcoding)
2. [Phase 2: Authentication + Session Security](#phase-2-authentication--session-security)
3. [Phase 3: Encryption + Audit Store Hardening](#phase-3-encryption--audit-store-hardening)
4. [Phase 4: Service Auth + API Security](#phase-4-service-auth--api-security)
5. [Phase 5: Portkey ML Consolidation](#phase-5-portkey-ml-consolidation)
6. [Phase 6: Infrastructure Layer Hardening](#phase-6-infrastructure-layer-hardening)
7. [Phase 7: Service Implementation Completeness](#phase-7-service-implementation-completeness)
8. [Phase 8: Testing Library Rebuild](#phase-8-testing-library-rebuild)
9. [Phase 9: Deployment Infrastructure](#phase-9-deployment-infrastructure)
10. [Phase 10: Code Quality + Standards](#phase-10-code-quality--standards)

---

## Phase 1: SQL Injection + Credential Hardcoding

### 1.1 SQL Injection in PostgresRepository

**File:** `src/solace_infrastructure/postgres.py`
**Severity:** Critical
**Issue:** Column names from `data.keys()` were interpolated directly into SQL strings in `insert()` and `update()` without validation, allowing SQL injection through crafted dictionary keys.

**Fix:** Added `_is_valid_identifier()` validation (regex: `^[a-zA-Z_][a-zA-Z0-9_]*$`) to all methods constructing SQL from external column names. Also validated `_schema` and `_table` in `PostgresRepository.__init__()`.

```python
# Before (vulnerable):
columns = ", ".join(data.keys())
query = f"INSERT INTO {self.qualified_table} ({columns}) VALUES ..."

# After (safe):
for col in data.keys():
    if not self._is_valid_identifier(col):
        raise ValueError(f"Invalid column name: {col}")
```

### 1.2 Hardcoded Credentials Removed

| File | Line | Before | After |
|------|------|--------|-------|
| `database/migrations_runner.py` | 54 | `default="postgresql+asyncpg://solace:solace@localhost:5432/solace"` | No default; fails fast if unset |
| `database/seed_data.py` | 65 | Same hardcoded URL | No default; environment-variable-only |
| `postgres.py` | 34-35 | `SecretStr("solace")` default password | No default |

### 1.3 Gemini API Key in URL

**File:** `src/solace_ml/gemini.py`
**Issue:** API key passed as `?key=` URL query parameter, visible in logs, browser history, and proxy servers.

**Fix:** Moved to `x-goog-api-key` HTTP header:
```python
# Before:
url = f"...?key={self._settings.api_key.get_secret_value()}"

# After:
headers = {"x-goog-api-key": self._settings.api_key.get_secret_value()}
response = await self._client.post(url, json=payload, headers=headers)
```

### 1.4 Schema Registry Password

**File:** `src/solace_infrastructure/kafka/schemas.py`
**Fix:** Changed `password: str | None` to `password: SecretStr | None` with `.get_secret_value()` at point of use.

### 1.5 AlertManager Secrets

**File:** `src/solace_infrastructure/observability/alerting_rules.py`
**Fix:** Changed `slack_api_url`, `pagerduty_service_key`, `opsgenie_api_key` from `str` to `SecretStr`.

---

## Phase 2: Authentication + Session Security

### 2.1 JWT Token Revocation

**File:** `src/solace_security/auth.py`
**Issue:** No way to invalidate issued JWT tokens. Stolen tokens remained valid until expiry.

**Fix:** Added Redis-backed JTI blacklist:
- `revoke_token(jti: str, ttl: int)` stores JTI in Redis with auto-expiry
- `decode_token()` checks blacklist before returning success
- `SessionManager.invalidate_session()` blacklists all session JWTs

### 2.2 Unknown Service Privilege Escalation

**File:** `src/solace_security/service_auth.py`
**Severity:** Critical
**Issue:** `_get_service_identity()` returned `ServiceIdentity.ORCHESTRATOR` for unknown service names, granting full orchestrator permissions.

**Fix:** Unknown service names now raise `ValueError` with explicit error message:
```python
# Before:
return ServiceIdentity.ORCHESTRATOR  # Dangerous fallback

# After:
raise ValueError(
    f"Unknown service identity: '{service_name}'. "
    f"Known services: {[s.value for s in ServiceIdentity]}"
) from None
```

### 2.3 Redis-Backed Sessions

**File:** `src/solace_security/auth.py`
**Issue:** Sessions stored in `dict` — lost on process restart, no TTL, not thread-safe.

**Fix:** Replaced with Redis hash operations:
- Session data stored as Redis hashes with configurable TTL
- Thread-safe via Redis atomicity
- Sessions survive process restarts

### 2.4 Account Lockout

**File:** `src/solace_security/auth.py`
**Issue:** `max_failed_attempts` and `lockout_duration_minutes` settings existed but were never enforced.

**Fix:** Wired into login flow:
- Failed attempts tracked in Redis with auto-expire
- Account locked after N consecutive failures
- Lockout duration configurable via `AuthSettings`

### 2.5 Refresh Token Rotation

**File:** `src/solace_security/auth.py`
**Issue:** `refresh_access_token()` reused the same refresh token indefinitely.

**Fix:** Issues new refresh token on each refresh and invalidates the old one via JTI blacklist.

### 2.6 RBAC Resolution in Middleware

**File:** `src/solace_security/middleware.py`
**Issue:** `AuthenticatedUser.has_permission()` only checked JWT-embedded permissions, ignoring role-based permissions from `ROLE_PERMISSIONS`.

**Fix:** Permission check now resolves through `ROLE_PERMISSIONS` mapping in `authorization.py`.

---

## Phase 3: Encryption + Audit Store Hardening

### 3.1 Silent Decryption Failures

**File:** `src/solace_security/encryption.py`
**Severity:** Critical (HIPAA)
**Issue:** `decrypt_dict()` caught `(ValueError, Exception)` and returned silently with `pass`, masking data corruption.

**Fix:** Explicit error logging with structured fields and re-raise of the exception.

### 3.2 Random Salt on Decrypt

**File:** `src/solace_security/encryption.py`
**Issue:** If `encrypted.salt` was `None`, a random salt was generated on the decrypt path — guaranteed decryption failure with no useful error.

**Fix:** Raises `ValueError("Cannot decrypt: missing salt")` immediately.

### 3.3 Hardcoded Search Hash Salt

**File:** `src/solace_security/encryption.py`
**Issue:** `b"solace-search-salt"` hardcoded — if discovered, allows rainbow table attacks on hashed PHI.

**Fix:** Salt loaded from `EncryptionSettings.search_hash_salt` (environment variable).

### 3.4 PostgreSQL Audit Store

**File:** `src/solace_security/audit.py`
**Issue:** `InMemoryAuditStore` lost all audit data on restart — HIPAA requires 6-year retention.

**Fix:** Created `PostgresAuditStore` with:
- Persistent storage in dedicated `audit_logs` table
- `AuditLogBase` added to `base_models.py`
- Configurable via `create_audit_store(backend="postgres")`

### 3.5 Production Guard

**Files:** `encryption.py`, `auth.py`
**Issue:** `for_development()` class methods created insecure test instances with no guard against production use.

**Fix:** Raises `ConfigurationError` if `ENVIRONMENT=production`.

### 3.6 Key Rotation

**File:** `src/solace_security/encryption.py`
**Fix:** Version-tagged keys (`v1`, `v2`, ...); ciphertext tagged with key version; decrypt selects correct version; rotate on schedule.

---

## Phase 4: Service Auth + API Security

### 4.1 Auth Middleware on All Endpoints

**Files modified:**
- `services/therapy_service/src/api.py`
- `services/safety_service/src/api.py`
- `services/orchestrator_service/src/api.py`
- `services/personality_service/src/api.py`
- `services/user-service/src/api.py`
- `services/analytics-service/src/api.py`
- `services/orchestrator_service/src/websocket.py`

**Fix:** Applied `Depends(get_current_user)` to user-facing endpoints and `Depends(verify_service_token)` to internal endpoints. Created shared `ServiceBase` class for consistent auth bootstrap.

### 4.2 Duplicate Service Deletion

**Deleted:** `services/user_service/` (underscore variant, 8 files)
**Kept:** `services/user-service/` (hyphen variant, full implementation)
**Also deleted:** `tests/user_service/` (4 test files for deleted service)

### 4.3 PHI Filtering on LLM Responses

**Files:** `therapy_service/src/domain/service.py`, `safety_service/src/domain/service.py`
**Fix:** All LLM-generated responses passed through `PHIDetector.detect().masked_text` before returning to users.

### 4.4 Authorization Denial Audit Logging

**File:** `src/solace_security/middleware.py`
**Fix:** `require_roles`/`require_permissions` denials now call `AuditLogger.log_authorization()` with denial details.

---

## Phase 5: Portkey ML Consolidation

### 5.1 Provider Consolidation

**Deleted raw provider implementations:** `openai.py`, `anthropic.py`, `deepseek.py`, `gemini.py`, `xai.py`, `minimax.py`
**Kept:** `llm_client.py` (interfaces), `inference.py` (parsers), `embeddings.py`

All inference now routes through `UnifiedLLMClient` in `services/shared/infrastructure/llm_client.py` via Portkey gateway.

### 5.2 Task-Type Auto-Parameters

```python
TASK_PROFILES = {
    "crisis":     {"temperature": 0.2, "top_p": 0.9},   # Focused, reliable
    "therapy":    {"temperature": 0.7, "top_p": 0.95},  # Warm, natural
    "diagnosis":  {"temperature": 0.3, "top_p": 0.85},  # Precise
    "creative":   {"temperature": 0.9, "top_p": 1.0},   # Journaling prompts
    "structured": {"temperature": 0.0},                   # JSON output
}
```

### 5.3 PHI Content Safety Layer

Pre-send: Scans user messages for PHI patterns (SSN, email, phone, credit card), masks before sending to external API.
Post-receive: Scans LLM response for accidentally generated PHI.

---

## Phase 6: Infrastructure Layer Hardening

### 6.1 Connection Retry with Backoff

**Files:** `postgres.py`, `redis.py`, `weaviate.py`
**Fix:** `connect()` retries up to 3 times with exponential backoff (`1s`, `2s`, `4s`) on transient errors (`OSError`, `TimeoutError`, `ConnectionError`).

### 6.2 Foreign Key Constraints

**File:** `src/solace_infrastructure/database/base_models.py`
**Fix:** Added proper foreign key relationships:
- `UserProfileBase.user_id` → `ForeignKey("users.id")`
- `ConsentRecordBase.user_id` → `ForeignKey("users.id")`
- Added `AuditLogBase` SQLAlchemy model

### 6.3 Deprecated asyncio API Replacement

12 instances of `asyncio.get_event_loop()` replaced with `asyncio.get_running_loop()` across:
- `health.py`
- `migrations_runner.py`
- `seed_data.py`

### 6.4 Singleton Thread Safety

**Files:** `observability_core.py`, `service_auth.py`
**Fix:** Double-checked locking pattern:
```python
_lock = threading.Lock()

def get_singleton():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = create_instance()
    return _instance
```

### 6.5 Tracer Memory Leak

**File:** `observability_core.py`
**Issue:** `_completed_spans` list grew without bound.
**Fix:** Replaced with `collections.deque(maxlen=10000)` ring buffer.

### 6.6 Kafka Mock Fallback

**Files:** `kafka/monitoring.py`, `kafka/topics.py`, `kafka/schemas.py`
**Issue:** When `aiokafka` was not installed, mock classes returned fake healthy data.
**Fix:** Return `UNAVAILABLE` status with explicit warning.

### 6.7 SchemaCache O(n) Eviction

**File:** `kafka/schemas.py`
**Fix:** Replaced `dict` + linear scan with `collections.OrderedDict` for O(1) LRU eviction.

### 6.8 Alembic Migrations

**Created:**
- `migrations/env.py` — Alembic environment configuration
- `migrations/script.py.mako` — Migration script template
- `migrations/versions/001_initial_schema.py` — Initial schema with all tables from `base_models.py`

---

## Phase 7: Service Implementation Completeness

### 7.1 Exception Handling Cleanup

Replaced ~66 instances of `except Exception: pass` across all services with proper patterns:
- Log error with structured context
- Raise domain-specific exception OR return explicit error result
- Ensure callers can distinguish transient vs permanent failures

### 7.2 PostgreSQL Repository Implementations

**Created:** `services/therapy_service/src/infrastructure/postgres_repository.py` (897 lines)
Four full repository implementations:
- `PostgresTreatmentPlanRepository` — 28-column UPSERT, goals stored as JSONB
- `PostgresTherapySessionRepository` — 22-column UPSERT, interventions/homework as JSONB
- `PostgresTechniqueRepository` — Immutable value objects with dynamic search
- `PostgresOutcomeMeasureRepository` — Clinical scores with subscale JSONB

**Created:** `services/user-service/src/infrastructure/postgres_repository.py` (501 lines)
- `PostgresUserRepository`
- `PostgresUserPreferencesRepository`
- `PostgresConsentRepository`

### 7.3 Repository Factory Pattern

All services use `create_unit_of_work(backend="postgres"|"memory")` for backend-swappable persistence:
```python
def create_unit_of_work(backend: str = "memory", **kwargs) -> UnitOfWork:
    if backend == "postgres":
        from .postgres_repository import PostgresTreatmentPlanRepository, ...
        client = kwargs["postgres_client"]
        return UnitOfWork(
            treatment_plans=PostgresTreatmentPlanRepository(client),
            ...
        )
```

### Service Backend Status

| Service | Backend | Notes |
|---------|---------|-------|
| therapy_service | PostgreSQL (new) | Full UPSERT with JSONB |
| user-service | PostgreSQL (new) | Full CRUD with consent tracking |
| safety_service | PostgreSQL | Already had PostgreSQL backend |
| memory_service | PostgreSQL + Redis | Already had persistent backends |
| diagnosis_service | PostgreSQL | Already had PostgreSQL + factory |
| personality_service | PostgreSQL | Already had PostgreSQL + factory |
| orchestrator_service | In-memory | Appropriate for LangGraph state |
| notification-service | N/A | Infrastructure directory not needed |

---

## Phase 8: Testing Library Rebuild

### 8.1 Contract Test Fix

**File:** `src/solace_testing/contracts.py`
**Issue:** `ProviderContractTest.verify_contract()` always returned `PASSED` without testing.

**Fix:** Now performs actual HTTP request via `api_client.request()` and verifies response against contract:
```python
response = api_client.request(**request_kwargs)
if asyncio.iscoroutine(response):
    response = asyncio.get_event_loop().run_until_complete(response)
result = self.verifier.verify_api_contract(contract, response)
```

### 8.2 API Test Client Fix

**File:** `src/solace_testing/integration.py`
**Issue:** `APITestClient` returned hardcoded mock data (always `{"status": "ok"}`).

**Fix:** Replaced with real `httpx.AsyncClient`:
```python
class APITestClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        import httpx
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def request(self, method, url, headers=None, params=None, json_data=None):
        response = await self._client.request(method=method, url=url, ...)
        return {"status": response.status_code, "headers": dict(response.headers), "json": body}
```

### 8.3 MockPostgres Transaction Fix

**File:** `src/solace_testing/mocks.py`
**Issue:** `begin_transaction()` used shallow copy for savepoints — nested dict mutations leaked between transactions.

**Fix:** `copy.deepcopy(self._tables)` for full isolation.

### 8.4 Security Test Suite

**Created:** `tests/solace_security/test_security_suite.py` (308 lines)

Five test classes:
- `TestSQLInjectionPrevention` — 10 parametrized injection payloads, boundary tests
- `TestAuthenticationSecurity` — JWT secret requirements, password hashing, token tampering
- `TestEncryptionSecurity` — Roundtrip, different-key rejection, plaintext-not-in-ciphertext
- `TestPHISafety` — SSN/email/phone/credit card detection, masking, false positive checks
- `TestInputValidation` — Sanitization, truncation, email/UUID validation, HMAC verification

---

## Phase 9: Deployment Infrastructure

### 9.1 Dockerfiles Created

| Service | Port | File |
|---------|------|------|
| user-service | 8001 | `services/user-service/Dockerfile` |
| notification-service | 8003 | `services/notification-service/Dockerfile` |
| diagnosis_service | 8004 | `services/diagnosis_service/Dockerfile` |
| memory_service | 8005 | `services/memory_service/Dockerfile` |
| therapy_service | 8006 | `services/therapy_service/Dockerfile` |
| personality_service | 8007 | `services/personality_service/Dockerfile` |
| orchestrator_service | 8000 | `services/orchestrator_service/Dockerfile` |

All follow the same template:
- Multi-stage build (`python:3.12-slim`)
- `uv` for fast dependency installation
- Non-root user (`appuser`)
- Health check via `curl`
- Uvicorn CMD with configurable workers

### 9.2 GitHub Actions CI/CD

**Created:** `.github/workflows/ci.yml`

| Job | Tool | Description |
|-----|------|-------------|
| `lint` | ruff | Format check + lint rules |
| `typecheck` | mypy | Strict type checking |
| `test` | pytest | Unit + integration tests with PostgreSQL + Redis service containers |
| `security-scan` | bandit | SAST scan (high severity, high confidence) |
| `docker-build` | docker | Matrix build of all 7 service Dockerfiles |

---

## Phase 10: Code Quality + Standards

### 10.1 OwnershipPolicy DELETE Fix

**File:** `src/solace_security/authorization.py`
**Issue:** Resource owners could delete clinical records (therapy sessions, treatment plans) without explicit DELETE permission.

**Fix:** Removed `Permission.DELETE` from ownership-allowed actions:
```python
# Before:
allowed_actions = {Permission.READ, Permission.WRITE, Permission.DELETE}

# After:
allowed_actions = {Permission.READ, Permission.WRITE}
```

Deletion now requires explicit RBAC permission through therapist/clinician/admin roles.

### 10.2 Async Context Managers

Added `__aenter__`/`__aexit__` to all three infrastructure clients:

| Client | File |
|--------|------|
| `PostgresClient` | `src/solace_infrastructure/postgres.py` |
| `RedisClient` | `src/solace_infrastructure/redis.py` |
| `WeaviateClient` | `src/solace_infrastructure/weaviate.py` |

Pattern:
```python
async def __aenter__(self) -> ClientType:
    await self.connect()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    await self.disconnect()
```

### 10.3 Domain Exception Hierarchy

**File:** `src/solace_common/exceptions.py` (already existed, no changes needed)

```
SolaceError
├── DomainError
│   ├── ValidationError
│   ├── EntityNotFoundError
│   ├── EntityConflictError
│   ├── ConcurrencyError
│   └── BusinessRuleViolationError
├── ApplicationError
│   ├── AuthenticationError
│   ├── AuthorizationError
│   ├── RateLimitExceededError
│   └── SafetyError
└── InfrastructureError
    ├── DatabaseError
    ├── CacheError
    ├── ExternalServiceError
    └── ConfigurationError
```

### 10.4 Singleton Thread Safety

**File:** `src/solace_security/service_auth.py`
**Fix:** Added `threading.Lock()` with double-checked locking to `get_service_token_manager()`.

Other singletons (`observability_core.py` MetricsRegistry + Tracer, `audit.py`) were already fixed in Phase 6.

### 10.5 Ruff Lint Pass

| Category | Count | Fix Method |
|----------|-------|------------|
| Import sorting (I001) | 5 | Auto-fixed |
| `typing` → `collections.abc` (UP035) | 3 | Auto-fixed |
| Unused imports (F401) | 8 | Auto-fixed |
| `timezone.utc` → `UTC` (UP017) | 5 | Auto-fixed |
| Quoted annotations (UP037) | 3 | Auto-fixed |
| `asyncio.TimeoutError` → `TimeoutError` (UP041) | 1 | Auto-fixed |
| Other auto-fixable | 20 | Auto-fixed |
| Missing `from e` on re-raise (B904) | 14 | Manual fix |
| Unnecessary set comprehension (C416) | 1 | Manual fix |
| Missing `ClassVar` annotation (RUF012) | 1 | Manual fix |
| SQL injection false positives (S608) | 8 | Accepted (table names are class constants) |
| **Total** | **69** | **59 fixed, 8 accepted, 2 pre-existing** |
