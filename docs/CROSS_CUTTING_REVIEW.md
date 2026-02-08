# Solace-AI: Cross-Cutting Architecture & Data Flow Review

**Review Date:** 2026-02-08
**Reviewed By:** Senior AI Engineer
**Scope:** System-wide architectural review covering service integration, data flow, security architecture, event bus, configuration, deployment, missing implementations, and dead code
**Method:** Cross-cutting analysis across all services, shared packages, and infrastructure

---

## Executive Summary

This review supplements the line-by-line Phase 1-10 reviews (401 issues) with a system-wide architectural analysis. While individual files may appear internally consistent, the cross-cutting view reveals **fundamental integration failures** that render the platform non-functional as a distributed system.

| Category | Critical | High | Medium | Total |
|----------|----------|------|--------|-------|
| Service Integration & Data Flow | 5 | 8 | 6 | 19 |
| Missing Implementations & Dead Code | 7 | 12 | 7 | 26 |
| Configuration & Deployment | 8 | 13 | 7 | 28 |
| Security Architecture (End-to-End) | 7 | 6 | 2 | 15 |
| Event Bus & Async Flow | 4 | 7 | 6 | 17 |
| **TOTAL (deduplicated)** | **~30** | **~40** | **~25** | **~95** |

> Note: Some findings overlap across categories. The deduplicated count removes duplicates where the same root cause was found by multiple agents. The raw total across all agents is 134 findings.

**The Three Most Damaging Systemic Issues:**

1. **Authentication is fundamentally broken end-to-end.** Three incompatible JWT implementations (gateway, user-service, solace_security) use different issuers and audiences. Tokens created by the user-service login flow will be rejected by the `solace_security` middleware that all other services import. Even if tokens worked, 30+ endpoints across 7 services have no auth at all.

2. **The event-driven architecture is non-functional.** Only 2 of 7+ services have Kafka event bridges. The analytics consumer isn't connected to Kafka. The memory service's event publisher will crash at runtime (swapped arguments). Event bridges that exist only convert a fraction of event types (2 of 6 for safety, 3 of 18 for therapy). The transactional outbox, DLQ, and consumer infrastructure are well-designed but carry almost no real events.

3. **Everything is in-memory.** Sessions, token revocation, rate limiting, metrics, memory tiers, orchestrator state, event outbox, DLQ, analytics aggregations -- all stored in Python dicts that evaporate on restart. Six of the seven planned centralized entity modules don't exist yet.

### Resolution Status (Updated 2026-02-08)

**Significant progress on systemic issues.** Of the three most damaging systemic issues identified above:
1. **Authentication** — Largely resolved (Tiers 0-1): JWT unified, auth fallbacks removed, 30+ endpoints secured, in-memory stores replaced with Redis
2. **Event-driven architecture** — Mostly open (Tiers 3-4): Memory event publisher fixed, but 5 services still lack bridges
3. **In-memory stores** — Partially resolved (Tier 2): Sessions, tokens, rate limits moved to Redis; entity modules created; memory service wired to persistent storage; orchestrator state persistence added

#### Service Integration (INT-*)

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| INT-CRIT-01 | ServiceAuthenticatedClient never used | OPEN | Tier 4 (T4.9) |
| INT-CRIT-02 | Crisis notifications route to non-existent emails | OPEN | Tier 3 (T3.1) |
| INT-CRIT-03 | SafetyAssessment domain vs DB field mismatches | **RESOLVED** | T2.6 |
| INT-CRIT-04 | Safety event bridge drops 4 of 6 event types | OPEN | Tier 3 (T3.5) |
| INT-CRIT-05 | Three incompatible crisis/risk level enums | OPEN | Tier 3 (T3.4) |
| INT-HIGH-01 | Kong gateway only configures 3 of 8+ services | OPEN | Backlog |
| INT-HIGH-02 | Gateway routes reference non-existent services | OPEN | Backlog |
| INT-HIGH-03 | Orchestrator missing user/notification clients | OPEN | Backlog |
| INT-HIGH-04 | solace_common and solace_ml imported by zero services | OPEN | Partial (dead providers deleted) |
| INT-HIGH-05 | Therapy event bridge drops critical events | OPEN | Tier 3 (T3.5) |
| INT-HIGH-06 | Three independent JWT/auth implementations | **RESOLVED** | T1.1 |
| INT-HIGH-07 | All services degrade to no-auth on ImportError | **RESOLVED** | T1.2 |
| INT-HIGH-08 | `any` vs `Any` type bug in safety_entities.py | **RESOLVED** | T0.7 |

#### Missing Implementations (IMPL-*)

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| IMPL-CRIT-01 | Six centralized entity modules never created | **RESOLVED** | T2.1 |
| IMPL-CRIT-02 | Memory service uses in-memory dicts for all tiers | **RESOLVED** | T2.4 |
| IMPL-CRIT-03 | Safety LLM assessor falls back to mock responses | OPEN | Tier 3 (T3.9) |
| IMPL-CRIT-04 | In-memory repos remain default for 4 services | **RESOLVED** | T2.10 |
| IMPL-CRIT-05 | Therapy `_acquire()` infinite recursion | **RESOLVED** | T0.1 |
| IMPL-CRIT-06 | Zero test coverage for PostgreSQL repositories | OPEN | Tier 5 (T5.2) |
| IMPL-CRIT-07 | langgraph-checkpoint-postgres installed but never wired | **RESOLVED** | T2.2 |
| IMPL-HIGH-01 | Entire solace_ml package is dead code | OPEN | Partial (dead providers deleted) |
| IMPL-HIGH-02 | solace_testing only tested by own tests | OPEN | Backlog |
| IMPL-HIGH-03 | MockPostgresConnection always returns [] | OPEN | Tier 5 (T5.3) |
| IMPL-HIGH-04 | Two competing LLM architectures | OPEN | Tier 7 (T7.1) |
| IMPL-HIGH-05 | Memory semantic filter is substring matching | OPEN | Tier 7 (T7.4) |
| IMPL-HIGH-06 | ~30 unused packages in requirements.txt | OPEN | Tier 5 (T5.4) |
| IMPL-HIGH-07 | No integration tests for cross-service communication | OPEN | Tier 5 (T5.2) |
| IMPL-HIGH-08 | Safety LLM assessor test mocks away actual LLM | OPEN | Backlog |
| IMPL-HIGH-09 | Push notification google-auth undeclared | OPEN | Backlog |
| IMPL-HIGH-10 | Missing report generators (2 of 6 types) | OPEN | Tier 7 (T7.6) |
| IMPL-HIGH-11 | Analytics absolute import should be relative | **RESOLVED** | T0.10 |
| IMPL-HIGH-12 | All 5 LLM health_check() return None not False | OPEN | Backlog |

#### Configuration & Deployment (CFG-*)

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| CFG-CRIT-01 | Massive port conflicts | **RESOLVED** | T0.6 |
| CFG-CRIT-02 | SAFETY_ env prefix collision | OPEN | Tier 5 (T5.8) |
| CFG-CRIT-03 | All CI/CD disabled except linting | OPEN | Tier 5 (T5.1) |
| CFG-CRIT-04 | Database name mismatch across services | **RESOLVED** | T2.7 |
| CFG-CRIT-05 | Hardcoded default secrets in multiple files | **RESOLVED** | T0.4 |
| CFG-CRIT-06 | .env file exists on disk with test JWT secret | **RESOLVED** | T0.4 |
| CFG-CRIT-07 | Analytics/Config service no Docker entries | OPEN | Tier 5 (T5.9) |
| CFG-CRIT-08 | Docker naming inconsistency | OPEN | Tier 5 (T5.9) |
| CFG-HIGH-01 | Therapy config port collides with Memory | **RESOLVED** | T0.6 |
| CFG-HIGH-02 | User service __main__ hardcodes port | **RESOLVED** | T0.6 |
| CFG-HIGH-03 to 13 | Various env prefix/config inconsistencies | OPEN | Tier 5 (T5.8-T5.10) |

#### Security Architecture (SEC-*)

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| SEC-CRIT-01 | Three JWT implementations incompatible | **RESOLVED** | T1.1 |
| SEC-CRIT-02 | 30+ unauthenticated endpoints exposing PHI | **RESOLVED** | T1.3 |
| SEC-CRIT-03 | Unauthenticated WebSocket for therapy chat | **RESOLVED** | T1.3 |
| SEC-CRIT-04 | All token/session/brute-force tracking in-memory | **RESOLVED** | T1.4 |
| SEC-CRIT-05 | Fernet keys regenerated on restart | **RESOLVED** | T0.3 |
| SEC-CRIT-06 | User-service has no token revocation | OPEN | Backlog |
| SEC-CRIT-07 | Config service API key validation is no-op | **RESOLVED** | T1.17 |
| SEC-HIGH-01 | Silent auth fallback stubs on ImportError | **RESOLVED** | T1.2 |
| SEC-HIGH-02 | String-based role checks bypass PolicyEngine | **RESOLVED** | T1.12 |
| SEC-HIGH-03 | PHI logged in structured logs; sanitizer unused | OPEN | Tier 6 (T6.9) |
| SEC-HIGH-04 | Status endpoint leaks JWT config | OPEN | Backlog |
| SEC-HIGH-05 | EncryptedFieldMixin stores metadata only | **RESOLVED** | T2.5 |
| SEC-HIGH-06 | User-service get_current_user incompatible | OPEN | Backlog |

#### Event Bus & Async Flow (EVT-*)

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| EVT-CRIT-01 | Memory service event publisher crashes at runtime | **RESOLVED** | T0.2 |
| EVT-CRIT-02 | Orchestrator EventBus calls handlers synchronously | OPEN | Tier 4 (T4.5) |
| EVT-CRIT-03 | Dual divergent event schemas for safety | OPEN | Tier 4 (T4.6) |
| EVT-CRIT-04 | Dual divergent event schemas for memory | OPEN | Tier 4 (T4.6) |
| EVT-HIGH-01 | In-memory DLQ store loses records on restart | OPEN | Tier 4 (T4.4) |
| EVT-HIGH-02 | In-memory outbox loses events on restart | OPEN | Tier 4 (T4.2) |
| EVT-HIGH-03 | 5 of 7 services have no Kafka bridge | OPEN | Tier 4 (T4.1) |
| EVT-HIGH-04 | Analytics consumer not connected to Kafka | OPEN | Backlog |
| EVT-HIGH-05 | WebSocket no buffering/reconnection/heartbeat | OPEN | Backlog |
| EVT-HIGH-06 | Session/working memory entirely in-memory | **RESOLVED** | T2.4 |
| EVT-HIGH-07 | Orchestrator state falls back to MemoryStateStore | **RESOLVED** | T2.2 |

---

## 1. SERVICE INTEGRATION & DATA FLOW

### Critical Findings

**INT-CRIT-01: ServiceAuthenticatedClient Never Used -- All Inter-Service Calls Unauthenticated**
- `src/solace_security/service_auth.py` lines 376-468 implements a full HTTP client with automatic JWT injection for service-to-service calls
- Zero services import or use it. All HTTP clients use plain `httpx`
- The entire `SERVICE_PERMISSIONS` matrix (lines 61-102) is dead code
- **Impact:** Any network actor can impersonate any service

**INT-CRIT-02: Crisis Notifications Route to Non-Existent Email Addresses**
- Safety service `escalation.py:186` uses `f"clinician-{clinician_id}@solace-ai.com"` (placeholder)
- Notification service `consumers.py:316,354,365,400` falls back to `oncall@solace-ai.com`, `escalations@solace-ai.com`, `monitoring@solace-ai.com`
- No actual clinician contact lookup succeeds end-to-end
- **Impact:** Crisis alerts never reach real clinicians

**INT-CRIT-03: SafetyAssessment Domain vs DB Entities Have 15+ Field Mismatches**
- Domain entity (`safety_service/src/domain/entities.py:175-197`): `crisis_level`, `trigger_indicators`, `detection_layers_triggered`, `requires_escalation`, `detection_time_ms`
- SQLAlchemy entity (`solace_infrastructure/database/entities/safety_entities.py:74-196`): `assessment_notes`, `immediate_actions_required`, `assessor_id`, `assessment_method`, `next_assessment_due`
- No mapper layer exists between them
- **Impact:** Safety assessments cannot be correctly persisted/retrieved

**INT-CRIT-04: Safety Event Bridge Drops 4 of 6 Registered Event Types**
- `safety_service/src/infrastructure/event_bridge.py:140-150` only converts `CrisisDetectedEvent` and `EscalationTriggeredEvent`
- `ESCALATION_ACKNOWLEDGED`, `ESCALATION_RESOLVED`, `INCIDENT_CREATED`, `RISK_LEVEL_CHANGED` silently return `None`
- **Impact:** Downstream services never learn about escalation resolutions or risk changes

**INT-CRIT-05: Three Incompatible Crisis/Risk Level Enums**
- Kafka schemas: `CrisisLevel` = `NONE, LOW, ELEVATED, HIGH, CRITICAL`
- Safety domain: `crisis_level: str` (free-form string)
- SQLAlchemy: `RiskLevel` = `MINIMAL, LOW, MODERATE, HIGH, CRITICAL`
- `ELEVATED` vs `MODERATE` vs `MINIMAL` cannot be mapped between systems
- **Impact:** Validation errors or silent data loss at integration boundaries

### High Findings

- **INT-HIGH-01:** Kong gateway only configures 3 of 8+ services (orchestrator, user, safety). Therapy, diagnosis, personality, memory, notification, analytics unreachable through gateway
- **INT-HIGH-02:** Gateway routes reference non-existent `session-service` and `admin-service`
- **INT-HIGH-03:** Orchestrator has no `UserServiceClient` or `NotificationServiceClient` despite config URLs for both
- **INT-HIGH-04:** `solace_common` and `solace_ml` packages imported by zero services -- entirely dead shared code
- **INT-HIGH-05:** Therapy event bridge drops `RiskLevelElevated` and 4 other critical events before they reach Kafka
- **INT-HIGH-06:** Three independent JWT/auth implementations (gateway, user-service, solace_security) with divergent role definitions, issuers, and audiences
- **INT-HIGH-07:** All services silently degrade to no-auth when `solace_security` is unavailable (ImportError fallback)
- **INT-HIGH-08:** `any` vs `Any` type bug across 12 JSONB columns in safety_entities.py

---

## 2. MISSING IMPLEMENTATIONS & DEAD CODE

### Critical Findings

**IMPL-CRIT-01: Six Centralized Entity Modules Declared But Never Created**
- `solace_infrastructure/database/entities/__init__.py:28-33` declares `user_entities`, `therapy_entities`, `diagnosis_entities`, `memory_entities`, `notification_entities`, `analytics_entities` with TODO comments
- Only `safety_entities.py` exists
- **Impact:** Schema Registry and centralized database migration cannot proceed. This blocks Phase 1 remediation.

**IMPL-CRIT-02: Memory Service Uses In-Memory Dicts for All Five Tiers**
- `memory_service/src/domain/service.py:37-42` stores all five memory tiers in Python dicts
- No PostgreSQL, no Weaviate, no Redis persistence for any tier
- **Impact:** All patient memory/context data lost on restart. HIPAA violation.

**IMPL-CRIT-03: Safety LLM Assessor Falls Back to Mock Responses**
- `safety_service/src/ml/llm_assessor.py:135-143` sets `self._use_langchain = False` when LangChain unavailable
- Returns hardcoded mock safety assessments with no indicator to the caller
- **Impact:** Risk assessments in production could be fake without anyone knowing

**IMPL-CRIT-04: In-Memory Repos Remain Default for Therapy, Personality, Diagnosis, User**
- User service `RepositoryConfig.use_postgres` defaults to `False`
- Therapy/personality have no feature flag; default to in-memory
- Production guard (`os.getenv("ENVIRONMENT") == "production"`) is the only safety net
- **Impact:** All patient data in non-production environments is ephemeral

**IMPL-CRIT-05: Therapy `_acquire()` Infinite Recursion in 4 Repository Classes**
- Lines 76, 349, 652, 831 in `therapy_service/src/infrastructure/postgres_repository.py`
- `return self._acquire()` instead of `return self._client.acquire()`
- All 4 repository classes affected: TreatmentPlan, TherapySession, Homework, Outcome
- **Impact:** All therapy database operations crash when using legacy client path

**IMPL-CRIT-06: Zero Test Coverage for PostgreSQL Repositories**
- All service tests only test in-memory implementations
- ~3000+ lines of PostgreSQL code with zero tests
- The infinite recursion bug proves tests never exercise the Postgres path
- **Impact:** No confidence in any database code correctness

**IMPL-CRIT-07: `langgraph-checkpoint-postgres` Installed But Never Wired**
- Package in requirements.txt, `MemorySaver()` used instead
- Orchestrator `StatePersistenceManager._create_store()` falls back to `MemoryStateStore()` for ALL backends including "postgres"
- **Impact:** LangGraph conversation state lost on every restart

### High Findings

- **IMPL-HIGH-01:** Entire `solace_ml` package (~1800 lines, 6 LLM providers) is dead code -- no service imports it
- **IMPL-HIGH-02:** `solace_testing` package (~1000 lines) only tested by its own tests, never consumed
- **IMPL-HIGH-03:** `MockPostgresConnection` always returns `[]` -- tests validate zero real behavior
- **IMPL-HIGH-04:** Two competing LLM architectures: `solace_ml` (unused) vs Portkey `UnifiedLLMClient` (partial)
- **IMPL-HIGH-05:** Memory service `_semantic_filter()` is substring matching, not vector search
- **IMPL-HIGH-06:** ~30 unused packages in requirements.txt adding ~2+ GB install weight (torch, transformers, numpy, pandas, scikit-learn, etc.)
- **IMPL-HIGH-07:** No integration tests exist for any cross-service communication
- **IMPL-HIGH-08:** Safety LLM assessor test mocks away the actual LLM -- safety-critical component untested
- **IMPL-HIGH-09:** Push notification `google-auth` dependency undeclared in requirements
- **IMPL-HIGH-10:** Missing report generators: `ENGAGEMENT_METRICS` and `COMPLIANCE_AUDIT` defined but not implemented
- **IMPL-HIGH-11:** Analytics service `from models import` (absolute) should be `from .models import` (relative)
- **IMPL-HIGH-12:** All 5 LLM client `health_check()` methods return `None` on failure instead of `False`

---

## 3. CONFIGURATION & DEPLOYMENT

### Critical Findings

**CFG-CRIT-01: Massive Port Conflicts**
- 4 services default to port 8001 (safety, orchestrator, user config, user main)
- 6 services default to port 8003 (notification config, notification main, diagnosis config, diagnosis main, personality main, memory config)
- Docker Compose assigns different ports but code defaults clash when running outside Docker
- **Impact:** Only one service can start at a time outside Docker

**CFG-CRIT-02: `SAFETY_` Env Prefix Collision**
- Safety service `SafetyServiceConfig` uses `env_prefix="SAFETY_"`
- Orchestrator `SafetySettings` also uses `env_prefix="SAFETY_"`
- Same env vars affect two different config schemas
- **Impact:** Silent misconfiguration when services share an environment

**CFG-CRIT-03: All CI/CD Disabled Except Linting**
- `.github/workflows/ci.yml`: Type checking (mypy), tests (pytest), security scanning (bandit), and Docker builds are ALL commented out
- Only `ruff check` and `ruff format --check` run
- **Impact:** No automated testing, no security scanning, no build verification

**CFG-CRIT-04: Database Name Mismatch Across Services**
- User: `solace_users`; Therapy: `solace_therapy`; Personality: `personality_db`; Diagnosis: `diagnosis_db`; Memory: `solace_memory`
- Docker Compose creates ONE database: `solace`
- **Impact:** Services expect separate databases that don't exist in deployment

**CFG-CRIT-05: Hardcoded Default Secrets in Multiple Files**
- API Gateway: `"your-secret-key-change-in-production"` (plain string)
- Config Service: `"changeme"` (database password), `"change-in-production"` (JWT secret)
- Docker Compose: `"dev-jwt-secret-key-minimum-32-chars"` and `"dev-encryption-key-32-characters"`
- **Impact:** Predictable secrets if env vars not explicitly set

**CFG-CRIT-06:** `.env` file exists on disk with test JWT secret key

**CFG-CRIT-07:** Analytics service and Config service have no Docker/Compose entries

**CFG-CRIT-08:** Docker build matrix uses inconsistent naming (`user-service` hyphen vs `safety_service` underscore)

### High Findings

- **CFG-HIGH-01:** Therapy config port 8002 collides with Memory main port 8002
- **CFG-HIGH-02:** User service `__main__` hardcodes port 8007 ignoring config
- **CFG-HIGH-03:** `KAFKA_` env prefix collision across 4 services with different schemas
- **CFG-HIGH-04:** `REDIS_` env prefix collision between memory and user services
- **CFG-HIGH-05:** `DB_` vs `POSTGRES_` env prefixes inconsistent across services
- **CFG-HIGH-06:** Different default DB users: `postgres`, `solace`, `personality_user`, `diagnosis_user`
- **CFG-HIGH-07:** Kafka bootstrap servers inconsistent: some `localhost:9092`, some `localhost:29092`
- **CFG-HIGH-08:** `pytest-asyncio==1.3.0` pinned in some services (version doesn't exist)
- **CFG-HIGH-09:** No Kubernetes/Helm manifests despite k8s probe implementations
- **CFG-HIGH-10:** Config service exists in code but has no deployment config
- **CFG-HIGH-11:** Grafana default credentials `admin/admin` in docker-compose
- **CFG-HIGH-12:** Per-service requirements use `==` pins while root uses `>=`
- **CFG-HIGH-13:** `OBSERVABILITY_` env prefix collision between analytics and notification

---

## 4. SECURITY ARCHITECTURE (END-TO-END)

### Critical Findings

**SEC-CRIT-01: Three JWT Implementations with Incompatible Issuers/Audiences**
- Gateway: issuer `"solace-ai"`, audience `"solace-ai-api"`
- User-service: issuer `"solace-ai-user-service"`, audience `"solace-ai-api"`
- solace_security: issuer `"solace-ai"`, audience `"solace-api"`
- Tokens from user-service login fail validation in solace_security middleware
- **Impact:** End-to-end authentication is broken at the protocol level

**SEC-CRIT-02: 30+ Unauthenticated Endpoints Exposing PHI Across 7 Services**
- Therapy: 5 endpoints (session state, treatment plan, homework, delete, progress)
- Safety: 4 of 5 operational endpoints (detect-crisis, escalate, assess, filter-output)
- Orchestrator: 4 endpoints (create session, history, batch, WebSocket)
- Personality: 5 endpoints (style, adapt, profile read/write, traits)
- User Service: on-call clinicians
- Analytics: ALL endpoints (dashboard, metrics, reports, event ingestion)
- Config Service: ALL read endpoints + write endpoints with no-op API key validation

**SEC-CRIT-03: Unauthenticated WebSocket for Real-Time Therapy Chat**
- `orchestrator/api.py:261` accepts connections with no token, no auth dependency
- `user_id` taken from query string, never validated
- Attacker can read/inject messages into active therapy sessions
- **Impact:** Most direct path to real-time PHI exposure

**SEC-CRIT-04: All Token Revocation, Sessions, Brute-Force Tracking In-Memory**
- Gateway token blacklist: `set()` lost on restart
- solace_security token blacklist: `dict()` lost on restart
- User-service sessions: `dict()` lost on restart
- Login attempt tracker: `dict()` lost on restart
- No Redis, no database persistence for any of these

**SEC-CRIT-05: Fernet Keys Regenerated on Restart (Data Loss)**
- `user-service/main.py:163-164` calls `Fernet.generate_key()` each startup
- All pending verification tokens, encrypted fields permanently lost

**SEC-CRIT-06: User-Service Has No Token Revocation -- 30-Day Refresh Tokens**
- `JWTService` can create/verify tokens but cannot revoke them
- Refresh tokens valid 30 days, no blacklist mechanism
- Compromised refresh token = 30 days unrevocable access

**SEC-CRIT-07: Config Service API Key Validation is No-Op**
- `_verify_api_key()` checks header presence, never validates value
- Any non-empty `X-API-Key: anything` passes
- Feature flags, secrets, configs modifiable by anyone

### High Findings

- **SEC-HIGH-01:** Silent auth fallback stubs on ImportError -- `get_current_user_optional()` returns `None`
- **SEC-HIGH-02:** String-based role checks bypass existing `PolicyEngine` RBAC system
- **SEC-HIGH-03:** PHI (emails) logged in structured logs; `PHISanitizer` exists but unused
- **SEC-HIGH-04:** Status endpoint leaks JWT algorithm, expiration config
- **SEC-HIGH-05:** `EncryptedFieldMixin` stores metadata only -- no actual field encryption wired
- **SEC-HIGH-06:** User-service `get_current_user()` incompatible with `solace_security` middleware

---

## 5. EVENT BUS & ASYNC DATA FLOW

### Critical Findings

**EVT-CRIT-01: Memory Service Event Publisher Crashes at Runtime**
- `memory_service/src/events.py:162` calls `publish(MEMORY_TOPIC, event.to_dict())`
- `EventPublisher.publish()` signature is `publish(event: BaseEvent, topic: str)`
- Arguments are swapped -- will raise `TypeError` on first publish

**EVT-CRIT-02: Orchestrator EventBus Calls Handlers Synchronously**
- `orchestrator/src/events.py:149` defines `EventHandler = Callable[[Event], None]`
- `publish()` at line 193 calls `handler(event)` synchronously
- If any handler is async, the coroutine is silently dropped (never awaited)

**EVT-CRIT-03: Dual Divergent Event Schemas for Safety Events**
- Shared Kafka `CrisisDetectedEvent`: `detection_layer` (int), `confidence` (Decimal)
- Local safety `CrisisDetectedEvent`: `detection_layers` (list[int]), `risk_score` (Decimal)
- Bridge maps `detection_layers[0]` -> `detection_layer` (lossy) and `risk_score` -> `confidence` (semantically different)
- Escalation bridge drops `escalation_id`, `crisis_level`, `notification_channels`, `estimated_response_minutes`

**EVT-CRIT-04: Dual Divergent Event Schemas for Memory Events**
- Shared: `memory_id`, `memory_tier` (enum), `retention_category` (enum), `embedding_generated`, `ttl_hours`
- Local: `record_id`, `tier` (str), `importance_score`, `storage_backend`, `storage_time_ms`
- No event bridge exists for memory service -- combined with EVT-CRIT-01, memory events never reach Kafka

### High Findings

- **EVT-HIGH-01:** In-memory DLQ store loses all dead letter records on restart
- **EVT-HIGH-02:** In-memory outbox store loses unpublished events on restart (defeats transactional outbox pattern)
- **EVT-HIGH-03:** 5 of 7 services have NO Kafka event bridge (diagnosis, user, personality, orchestrator, notification) -- events never leave the process
- **EVT-HIGH-04:** Analytics consumer uses internal `asyncio.Queue`, NOT connected to Kafka -- topic subscriptions are decorative
- **EVT-HIGH-05:** WebSocket has no message buffering, no reconnection protocol, no heartbeat
- **EVT-HIGH-06:** Session memory and working memory (Tiers 1-3) entirely in-memory, no persistence
- **EVT-HIGH-07:** Orchestrator state always falls back to `MemoryStateStore()` even when postgres backend configured

---

## Architectural Verdict

The Solace-AI platform has **well-designed individual components** -- the event infrastructure, RBAC system, encryption library, connection pool manager, and schema registry are all architecturally sound. However, **almost none of them are wired together**:

| Component | Designed | Implemented | Integrated | Actually Works End-to-End |
|-----------|----------|-------------|------------|--------------------------|
| JWT Authentication | 3 implementations | 3 implementations | 0 of 3 interoperate | NO |
| RBAC/PolicyEngine | Full design | Full code | 0 services use it | NO |
| Kafka Event Bus | 7 topics, 19 schemas | Publishers + consumers | 2 of 7 services bridged | PARTIALLY (safety+therapy only) |
| Transactional Outbox | Pattern designed | In-memory store only | Not durable | NO |
| Connection Pool Manager | Singleton pattern | Working code | Feature-flagged | PARTIALLY |
| Schema Registry | Decorator pattern | Working code | 1 of 7 entity modules | NO |
| Encryption at Rest | EncryptedFieldMixin + Encryptor | Both exist | Never wired together | NO |
| PHI Protection | PHIDetector + PHIMasker + PHISanitizer | Full code | Not in logging pipeline | NO |
| Service Auth | ServiceAuthenticatedClient | Full code | 0 services use it | NO |
| Memory Persistence | 5-tier design | In-memory dicts | No DB/vector store | NO |
| LangGraph Checkpointing | postgres checkpoint installed | MemorySaver used | Falls back to memory | NO |

**The platform is architecturally a monolith disguised as microservices.** Services exist as separate processes but share no authenticated communication, no persistent event bus, no distributed state, and no common authentication. The sophisticated infrastructure code (solace_security, solace_events, solace_infrastructure) is largely unused by the services it was designed to support.

---

## Grand Total: All Reviews Combined

| Review | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| Phase 1-2 | 13 | 17 | 21 | 14 | 65 |
| Phase 3-4 | 12 | 24 | 28 | 16 | 80 |
| Phase 5-6 | 12 | 36 | 26 | 5 | 79 |
| Phase 7-8 | 21 | 26 | 35 | 16 | 98 |
| Phase 9-10 | 21 | 27 | 21 | 10 | 79 |
| Cross-Cutting (deduplicated) | ~30 | ~40 | ~25 | 0 | ~95 |
| **ESTIMATED GRAND TOTAL** | **~109** | **~170** | **~156** | **~61** | **~496** |

> Note: Cross-cutting findings overlap with some phase-specific findings (e.g., hardcoded JWT secret, Fernet keys, unauthenticated endpoints were found in both). True unique total is likely ~450 after full deduplication.
