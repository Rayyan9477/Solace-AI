# Solace-AI src/ Deep Audit Report

> **Date**: 2026-04-11
> **Scope**: `/home/rayyan9477/Data/Repo/Solace-AI/src/` -- 66 Python files, 22,542 lines, 5 packages
> **Method**: Line-by-line code review of every file in all 5 shared library packages

---

## Executive Summary

The `src/` directory contains 5 shared library packages that form the foundation for all 10 microservices. The DDD domain layer (`solace_common`) and event infrastructure (`solace_events`) are **clean and well-engineered**. The security layer (`solace_security`) has **critical runtime bugs** that block HIPAA compliance. The infrastructure layer (`solace_infrastructure`) has **excellent ORM design** but PHI encryption is never activated. The testing library (`solace_testing`) is **adequate for unit tests** but insufficient for integration testing.

| Package | Files | Lines | Verdict |
|---------|-------|-------|---------|
| `solace_common` | 8 | 2,134 | CLEAN -- solid DDD foundation |
| `solace_events` | 7 | 2,670 | CLEAN -- prior criticals all fixed |
| `solace_security` | 9 | 4,098 | NEEDS WORK -- HMAC crash, InMemory blacklist, key validation |
| `solace_infrastructure` | ~30 | 11,683 | NEEDS WORK -- PHI encryption unwired, RLS missing |
| `solace_testing` | 6 | 1,957 | ACCEPTABLE -- gaps in mock fidelity, not blocking |

---

## Package 1: solace_common (8 files, 2,134 lines)

### Verdict: CLEAN

Well-structured DDD foundation. No bugs found.

### File Inventory

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `__init__.py` | 158 | Package exports | Clean |
| `enums.py` | 147 | CrisisLevel, SeverityLevel canonical enums | Clean |
| `exceptions.py` | 327 | 13+ structured exception types with correlation tracking | Clean |
| `utils.py` | 381 | DateTime, Crypto, Validation, Retry, Collection utilities | Clean |
| `domain/__init__.py` | 75 | Domain layer exports | Clean |
| `domain/value_object.py` | 383 | ValueObject, EmailAddress, Score, Percentage, etc. | Clean |
| `domain/entity.py` | 273 | Entity, MutableEntity, EntityMetadata with optimistic locking | Clean |
| `domain/aggregate.py` | 391 | AggregateRoot, DomainEvent, EventStore, InMemoryEventStore | Clean |

### Key Design Strengths
- `CrisisLevel.from_string()` with comprehensive alias mapping (moderate -> ELEVATED, etc.)
- `CrisisLevel.from_score()` with Decimal-based thresholds preventing float imprecision
- `InMemoryEventStore` blocks production use via environment check
- `retry_async` properly fixed -- regular def returning async wrapper (C-02 resolved)
- Full `__all__` exports on every module

---

## Package 2: solace_events (7 files, 2,670 lines)

### Verdict: CLEAN

All prior critical issues (C-03, C-04, H-35, H-36, H-42, H-44, H-59, H-61) are resolved.

### File Inventory

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `__init__.py` | 236 | Package exports -- all 42 event types exported | Clean |
| `schemas.py` | 574 | 42 event types + EVENT_REGISTRY + topic routing | Clean |
| `publisher.py` | 421 | Transactional outbox publisher with Kafka adapter | Clean |
| `consumer.py` | 510 | Consumer with offset tracking, handler registry, DLQ | Clean |
| `config.py` | 242 | 12 Kafka topics, settings, consumer groups | Clean |
| `dead_letter.py` | 362 | DLQ handler with 4 retry strategies | Clean |
| `postgres_stores.py` | 325 | PostgreSQL-backed outbox + DLQ stores | Clean |

### Key Verified Fixes
- `_TOPIC_MAP` now includes `"user."`, `"notification."`, `"homework."`, `"treatment."` prefixes
- `SolaceTopic` has all 12 topics: SESSIONS, ASSESSMENTS, THERAPY, SAFETY, MEMORY, ANALYTICS, PERSONALITY, MESSAGES, SYSTEM, NOTIFICATIONS, AUDIT, USERS
- `TherapyModality` includes SFBT
- `create_publisher()` blocks InMemoryOutboxStore in production
- `FOR UPDATE SKIP LOCKED` in PostgresOutboxStore for safe concurrent polling

---

## Package 3: solace_security (9 files, 4,098 lines)

### Verdict: NEEDS WORK -- 2 Critical, 4 High, 2 Medium bugs

### File Inventory

| File | Lines | Purpose | Bugs |
|------|-------|---------|------|
| `__init__.py` | 195 | Package exports | Clean |
| `auth.py` | 806 | JWT, password hashing, sessions | 0 (C-01 fix is in middleware) |
| `authorization.py` | 371 | RBAC, policies, authorizer | 1 HIGH (NEW-02) |
| `middleware.py` | 384 | FastAPI auth dependencies | 1 CRITICAL (C-01 partial) |
| `encryption.py` | 352 | AES-256-GCM, field encryption | 2 HIGH (H-38, NEW-06) |
| `audit.py` | 792 | Audit logging with HMAC chain | 1 CRITICAL (NEW-01), 1 MITIGATED (H-40) |
| `phi_protection.py` | 438 | PHI detection & masking | 3 MEDIUM (NEW-07/08/09) |
| `service_auth.py` | 535 | Service-to-service auth | 1 HIGH (NEW-05) |
| `production_guards.py` | 225 | Production safety checks | 0 (H-39 is infra-side) |

### Bug Details

**CRITICAL: audit.py:152 -- HMAC digestmod crash (NEW-01)**
```python
# BROKEN: passes string instead of hashlib function
return hmac_mod.new(hmac_key.encode(), canonical.encode(), algorithm).hexdigest()
# FIX:
import hashlib
return hmac_mod.new(hmac_key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
```
Impact: Audit chain integrity verification crashes when HMAC key is configured. HIPAA audit trail non-functional.

**CRITICAL: middleware.py:132 -- Per-worker token blacklist (C-01 remaining)**
```python
# CURRENT: InMemoryTokenBlacklist -- per-process only
return JWTManager(settings, token_blacklist=InMemoryTokenBlacklist())
# FIX: Use RedisTokenBlacklist
redis_client = ...  # get from app state
return JWTManager(settings, token_blacklist=RedisTokenBlacklist(redis_client))
```
Impact: Token revocation broken in multi-worker deployments.

**HIGH: authorization.py:109 -- Permission enum vs string (NEW-02)**
```python
# BROKEN: Permission enum compared to list[str]
def has_permission(self, permission: Permission) -> bool:
    if permission in self.permissions:  # self.permissions is list[str]
        return True
# FIX: Compare .value
    if permission.value in self.permissions:
        return True
```

**HIGH: encryption.py:86 -- Char length vs byte length (H-38)**
```python
# BROKEN: len() counts characters, not bytes
if len(key_value) != KEY_SIZE:
# FIX:
if len(key_value.encode('utf-8')) != KEY_SIZE:
```

**HIGH: encryption.py:81 -- Dev key wrong length (NEW-06)**
```python
# BROKEN: 33 characters
cls(master_key=SecretStr("dev-only-insecure-key-32-bytes!!"))
# FIX: 32 characters
cls(master_key=SecretStr("dev-only-insecure-key-32bytes!!"))
```

**HIGH: service_auth.py:502 -- Missing Header() dependency (NEW-05)**
```python
# BROKEN: FastAPI won't inject
async def _verify_service(authorization: str | None = None):
# FIX:
from fastapi import Header
async def _verify_service(authorization: str | None = Header(None)):
```

---

## Package 4: solace_infrastructure (~30 files, 11,683 lines)

### Verdict: NEEDS WORK -- 1 Critical, 3 High, 2 Medium

### Critical Findings

**CRITICAL: PHI Encryption Never Activated (H-57)**
- `base_models.py:623` defines `configure_phi_encryption(field_encryptor)` 
- `base_models.py:642-664` defines SQLAlchemy event listeners for auto-encrypt/decrypt
- All three listeners check `if _global_field_encryptor` -- which is always None
- `configure_phi_encryption()` is never called in any service lifespan
- Result: All PHI stored in plaintext despite full encryption infrastructure

**HIGH: DiagnosisSession missing messages in __phi_fields__ (NEW-03)**
- `diagnosis_entities.py:76`: `__phi_fields__ = ["summary"]`
- `diagnosis_entities.py:95-97`: `messages` column stores full conversation history as JSONB
- Conversation content (PHI) not included in auto-encryption

**HIGH: Hypothesis has no __phi_fields__ (NEW-04)**
- `diagnosis_entities.py:208`: `class Hypothesis(ClinicalBase)` with no `__phi_fields__` declaration
- Supporting evidence, contra-evidence, challenge results all unencrypted

**HIGH: SSL Enforcement Disabled by Default (H-39)**
- `postgres.py:94-131`: SSL context method properly implemented
- `feature_flags.py:120`: `enforce_database_ssl` flag exists but disabled
- SSL available but not enforced in production

### ORM Entity Coverage

| Entity File | Tables Defined | PHI Marked | Status |
|-------------|---------------|-----------|--------|
| `user_entities.py` | User, UserPreferences, ConsentRecord, ClinicianPatientAssignment | Yes | Clean |
| `safety_entities.py` | SafetyAssessment, SafetyPlan, RiskFactor, ContraindicationCheck/Rule | Yes | Clean |
| `therapy_entities.py` | TreatmentPlan, TherapySession, Intervention, HomeworkAssignment | Yes | Clean |
| `diagnosis_entities.py` | DiagnosisSession, Symptom, Hypothesis, DiagnosisRecord | Partial | Missing fields |
| `memory_entities.py` | MemoryRecord, MemoryUserProfile, SessionSummary | Yes | Clean |
| `personality_entities.py` | PersonalityProfile, TraitAssessment, ProfileSnapshot | Yes | Clean |
| `notification_entities.py` | Notification, DeliveryAttempt, NotificationPreferences, Batch | N/A | Clean |

---

## Package 5: solace_testing (6 files, 1,957 lines)

### Verdict: ACCEPTABLE -- adequate for unit tests, gaps for integration

### File Inventory

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| `__init__.py` | 156 | Package exports | Clean |
| `factories.py` | 333 | Test data factories (User, Session, Message, Diagnosis, Safety, Vector, LLM) | Good |
| `fixtures.py` | 364 | In-memory DB/cache/queue fixtures | Adequate |
| `mocks.py` | 401 | MockPostgresClient, MockRedisClient, MockWeaviateClient, MockLLMClient | Good |
| `contracts.py` | 343 | Consumer-driven contract testing framework | Good |
| `integration.py` | 355 | Integration test orchestration | Framework only |

### Key Gaps (not blocking MVP)
- Factories don't generate `encryption_key_id` (required by ClinicalBase entities)
- Factories use `str` IDs; real models use `uuid.UUID`
- `MockPostgresClient` handles basic CRUD only (no JOINs, CTEs, complex WHERE)
- `integration.py` services don't actually start (mock-only orchestration)
- No authentication fixtures (JWT, service tokens)
- No field-level encryption mocking

---

## Cross-Package Dependency Map

```
solace_common         (zero external deps except pydantic/structlog)
    |
    v
solace_events         (depends on: solace_common.enums, aiokafka, asyncpg)
    |
    v
solace_security       (depends on: solace_common, python-jose, cryptography, fastapi)
    |
    v
solace_infrastructure (depends on: solace_common, solace_security, asyncpg, redis, weaviate-client, sqlalchemy)
    |
    v
solace_testing        (depends on: all above, httpx, factory-boy patterns)
    |
    v
services/*            (depends on: all above + domain-specific logic)
```

---

## Recommended Fix Priority for MVP

### Immediate (blocks HIPAA / core functionality)
1. **NEW-01**: Fix HMAC digestmod in `audit.py:152` (1 line)
2. **NEW-06**: Fix dev key length in `encryption.py:81` (1 line)
3. **H-57**: Call `configure_phi_encryption()` in service lifespans
4. **NEW-03/04**: Add missing `__phi_fields__` to diagnosis entities
5. **NEW-02**: Fix Permission enum comparison in `authorization.py:109`

### Short-term (blocks multi-worker / production)
6. **C-01**: Replace `InMemoryTokenBlacklist` with `RedisTokenBlacklist` in middleware
7. **H-38**: Fix key length validation to use byte count
8. **NEW-05**: Add `Header()` to service auth dependency
9. **H-39**: Enable SSL enforcement for production

### Deferrable (quality improvements)
10. **NEW-07/08/09**: PHI phone masking, SSN pattern, confidence threshold
11. **NEW-10**: Connection pool race condition
12. **H-56**: Row-Level Security policies
