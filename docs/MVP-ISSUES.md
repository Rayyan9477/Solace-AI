# Solace-AI MVP Issues Registry

> **Generated**: 2026-03-20 | **Last Audit**: 2026-04-11
> **Scope**: Full codebase review (10 parallel agents, ~270 Python files) + targeted `src/` deep audit
> **Methodology**: Design-spec-aware deep code review against SYSTEM-DESIGN-SUMMARY.md
> **Original Totals**: 24 Critical, 62 High, 68 Medium, 35 Low = **192 issues**
> **Audit Delta**: 2 Critical fixed, 1 partially fixed, ~10 High fixed/partially fixed, 1 new Critical, 5 new High, 4 new Medium
> **Current Open**: ~22 Critical, ~57 High, ~72 Medium, ~35 Low = **~186 open issues**

---

## Table of Contents

1. [CRITICAL Issues](#critical-issues)
2. [HIGH Issues](#high-issues)
3. [MEDIUM Issues](#medium-issues)
4. [LOW Issues](#low-issues)
5. [New Issues (April 2026 Audit)](#new-issues-april-2026-audit)
6. [Resolution Log](#resolution-log)
7. [Summary by Module](#summary-by-module)

---

## CRITICAL Issues

### Foundation & Infrastructure

**C-01 | LIB-01 | Middleware JWT — multi-worker token revocation broken** `[PARTIALLY FIXED]`
- Location: `src/solace_security/middleware.py:132`
- Original: `JWTManager(settings)` without `token_blacklist` — crashed every endpoint.
- Fix applied: Now passes `InMemoryTokenBlacklist()` — endpoints work.
- **Remaining**: `InMemoryTokenBlacklist` is per-process. In multi-worker deployments (uvicorn --workers N), tokens revoked in worker A remain valid in worker B. Must replace with `RedisTokenBlacklist`. Same issue in `service_auth.py:175`.

~~**C-02 | LIB-02 | `retry_async` function signature broken**~~ `[FIXED]`
- Location: `src/solace_common/utils.py:315`
- `retry_async` is now a regular `def` returning an `async` wrapper. Retry logic executes correctly.

~~**C-03 | LIB-03 | User events route to wrong Kafka topic**~~ `[FIXED]`
- Location: `src/solace_events/schemas.py:548`
- `_TOPIC_MAP` now includes `"user.": "solace.users"` entry.

~~**C-04 | LIB-04 | Notification events reference non-existent topic**~~ `[FIXED]`
- Location: `src/solace_events/config.py:71-73`
- `SolaceTopic` now has `NOTIFICATIONS`, `AUDIT`, `USERS` members with proper `TOPIC_CONFIGS`.

**C-05 | DB-01 | Only 7 reference tables in migration — 22+ domain tables missing** `[OPEN]`
- Location: `migrations/versions/001_initial_schema.py`
- Migration creates only `users`, `audit_logs`, `system_configurations`, `safety_resources`, `clinical_references`, `therapy_techniques`, `safety_events`. All domain tables missing.

**C-06 | DB-02 | Migration `users` table schema mismatches ORM entity** `[OPEN]`
- Location: `migrations/versions/001_initial_schema.py:21-39` vs `database/entities/user_entities.py:56-122`
- Completely different schemas between migration and ORM.

**C-07 | DB-03 | No `alembic.ini` file exists** `[OPEN]`
- No schema deployment path. `alembic upgrade head` fails.

**C-08 | DB-04 | Memory service defines divergent table schemas, bypasses Alembic** `[OPEN]`
- Location: `services/memory_service/src/infrastructure/postgres_repo.py:27-98`

**C-09 | DB-05+06 | PK column name mismatches across service repos** `[OPEN]`
- Location: personality_service uses `profile_id`, user-service uses `user_id`; ORM uses `id`.

**C-10 | SUPP-02 | Missing Dockerfiles for analytics-service and config-service** `[OPEN]`

**C-11 | SUPP-03 | User-service requires FERNET keys not supplied by docker-compose** `[OPEN]`

### Safety Service

**C-12 | SAFETY-01 | Layer 1 regex patterns bypassed when ML KeywordDetector active** `[OPEN]`
- Location: `services/safety_service/src/domain/crisis_detector.py:679-686`

**C-13 | SAFETY-02 | Event publisher never wired — zero audit trail** `[OPEN]`
- Location: `services/safety_service/src/domain/service.py`

### Diagnosis Service

**C-14 | DIAG-01 | Domain events never dispatched** `[OPEN]`

**C-15 | DIAG-02 | Confidence thresholds don't match spec — no Escalate tier** `[OPEN]`
- Code: 0.85+=VERY_HIGH, 0.70+=HIGH, 0.50+=MEDIUM, <0.50=LOW.
- Spec: 0.70+=High, 0.50-0.70=Moderate, 0.30-0.50=Low, <0.30=Escalate.

### Therapy Service

**C-16 | THER-01 | Treatment response remission branch unreachable** `[OPEN]`

~~**C-17 | THER-02 | TreatmentPlanner, HomeworkManager, ProgressTracker all dead code**~~ `[FIXED]`
- Wired in commit `2405c3a`.

### Memory Service

**C-18 | MEM-01 | All 5 tiers use plain Python dicts** `[PARTIALLY FIXED]`
- Redis integration added for working memory per commit `3732606`. Tier 2/3 partially wired.
- Tiers 4/5 (Episodic/Semantic via Weaviate) still dict-backed.

**C-19 | MEM-02 | Redis infrastructure exists but never initialized** `[PARTIALLY FIXED]`
- Redis cache now instantiated for working memory. Full tier integration incomplete.

**C-20 | MEM-03 | Decay formula double-compounds** `[OPEN]`

**C-21 | MEM-04 | Postgres batch decay uses linear subtraction** `[OPEN]`

### Integration

**C-22 | INTG-01 | Notification service crisis URL points to wrong port** `[OPEN]`

### NEW Critical (April 2026 Audit)

~~**NEW-01 | LIB-11 | Audit HMAC chain crashes at runtime**~~ `[NOT A BUG - VERIFIED 2026-04-12]`
- Location: `src/solace_security/audit.py:152`
- Original claim: `hmac.new(key, msg, "sha256")` crashes at runtime because digestmod should be callable.
- **Verification**: `hmac.new()` has accepted a string digestmod since Python 3.4 (PEP 247). Verified with `python3 -c "import hmac; print(hmac.new(b'k', b'm', 'sha256').hexdigest())"` — works correctly on Python 3.13.
- Status: REMOVED from open issues. My April audit was incorrect.

---

## HIGH Issues

### Safety Service (6)
- **H-01 | SAFETY-03**: Escalation manager shutdown never called `[OPEN]`
- ~~**H-02 | SAFETY-04**: Layer 2 recommended_action not recalculated~~ `[FIXED]` per commit `9ea7730`
- **H-03 | SAFETY-05**: Medium escalation claims "supervisor notified" but sends nothing `[OPEN]`
- **H-04 | SAFETY-06**: Escalation state in-memory `[PARTIALLY FIXED]` — InMemoryEscalationRepository added (commit `fc2403b`), needs PostgreSQL
- **H-05 | SAFETY-07**: LLM assessor cache ignores user_id `[OPEN]`
- **H-06 | SAFETY-08**: Protective factors never reduce risk score `[OPEN]`

### Diagnosis Service (5)
- **H-07 | DIAG-03**: Same challenge adjustment applied to ALL hypotheses `[OPEN]`
- **H-08 | DIAG-04**: Bayesian calibrator receives empty symptom list `[OPEN]`
- **H-09 | DIAG-05**: PHQ-9 MODERATELY_SEVERE maps to score 2 not 3 `[OPEN]`
- **H-10 | DIAG-07**: PCL-5 only 10 items with 20-item thresholds `[OPEN]`
- **H-11 | DIAG-08**: Safety flags never trigger CRISIS phase `[OPEN]`

### Therapy Service (8)
- **H-12 | THER-03**: Technique selection weights don't match spec `[OPEN]`
- **H-13 | THER-04**: "harm"/"danger" trigger crisis in benign contexts `[OPEN]`
- **H-14 | THER-05**: Homework only during CLOSING phase `[OPEN]`
- **H-15 | THER-06**: Severe patients get almost no techniques `[OPEN]`
- **H-16 | THER-07**: TreatmentPlanDTO phase type mismatch `[OPEN]`
- **H-17 | THER-08**: Session state machine allows ANY transition `[OPEN]`
- **H-18 | THER-09**: get_user_progress only queries active sessions `[OPEN]`
- **H-19 | THER-10**: Progress references `start_time` not `started_at` `[OPEN]`

### Orchestrator + Personality (7)
- **H-20 | ORCH-01**: Assessment and Emotion agents are stubs `[PARTIALLY FIXED]` — AssessmentAgent wired per commit `3732606`
- **H-21 | PERS-01**: RoBERTa model never called `[OPEN]`
- **H-22 | PERS-02**: Ensemble weights don't match spec `[OPEN]`
- **H-23 | PERS-03**: Multimodal fusion unused `[OPEN]`
- **H-24 | PERS-04**: MoEL empathy never called `[OPEN]`
- **H-25 | PERS-08**: Personality agent wrong endpoint paths `[OPEN]`
- **H-26 | XSVC-01**: Personality agent lacks service auth `[OPEN]`

### Memory Service (8)
- **H-27 | MEM-05**: Weaviate collections with `Vectors.none()` `[OPEN]`
- **H-28 | MEM-06**: No embedding generation `[OPEN]`
- **H-29 | MEM-07**: Context assembler uses keyword matching `[OPEN]`
- **H-30 | MEM-09**: Consolidation creates fresh DecayManager `[OPEN]`
- **H-31 | MEM-10**: Session count not persisted `[PARTIALLY FIXED]` — recovery method added per commit `3732606`
- **H-32 | MEM-11**: store_session_summary missing columns `[OPEN]`
- **H-33 | MEM-12**: Four tier-specific managers are dead code `[OPEN]`
- **H-34 | MEM-08**: Token count uses word-split `[OPEN]`

### Shared Libraries (6)
- ~~**H-35 | LIB-05**: Event schemas not exported~~ `[FIXED]`
- ~~**H-36 | LIB-06**: TherapyModality missing SFBT~~ `[FIXED]`
- **H-37 | LIB-07**: Bare module imports require install mode `[OPEN]`
- **H-38 | LIB-08**: Encryption key validates char length not byte length `[OPEN]`
- **H-39 | LIB-09**: PostgreSQL SSL never enforced `[OPEN]` — SSL infra exists but feature flag disabled
- **H-40 | LIB-10**: Audit HMAC key defaults to empty string `[MITIGATED]` — blocked in production, logs CRITICAL in dev

### Integration (6)
- **H-41 | INTG-02**: TherapyModality case mismatch `[OPEN]`
- ~~**H-42 | INTG-03**: SFBT missing from Kafka enum~~ `[FIXED]` — same as H-36
- **H-43 | INTG-04**: Safety check case mismatch `[OPEN]`
- ~~**H-44 | INTG-05**: User events route to analytics~~ `[FIXED]` — same as C-03
- **H-45 | INTG-06**: Memory event schemas incompatible `[OPEN]`
- **H-46 | INTG-07**: Safety plan events excluded from bridge `[OPEN]`

### Supporting + Infrastructure (8)
- ~~**H-47 | SUPP-04**: No CI/CD pipeline~~ `[PARTIALLY FIXED]` — CI workflow added per commit `fc2403b`
- **H-48 | SUPP-05**: In-memory session storage in user-service `[OPEN]`
- **H-49 | SUPP-06**: Consent repository wrong field names `[OPEN]`
- **H-50 | SUPP-07**: Notification service hard-imports solace_security `[OPEN]`
- **H-51 | SUPP-08**: Analytics service hard-imports solace_security `[OPEN]`
- **H-52 | SUPP-09**: Missing infrastructure directories `[OPEN]`
- **H-53 | SUPP-10**: RepositoryFactory defaults crash `[OPEN]`
- **H-54 | SUPP-11**: Notification not on data network `[OPEN]`

### Database (6)
- **H-55 | DB-07**: Diagnosis stores symptoms as JSON blobs `[OPEN]`
- **H-56 | DB-08**: No Row-Level Security policies `[OPEN]`
- **H-57 | DB-09**: PHI encryption never wired `[OPEN]` — `configure_phi_encryption()` never called
- **H-58 | DB-10**: Weaviate missing TherapeuticInsight + CrisisEvent `[OPEN]`
- ~~**H-59 | DB-11**: Kafka registers 7 topics not 9~~ `[FIXED]` — now 12 topics
- **H-60 | DB-12**: Safety queries non-existent contraindication_rules table `[OPEN]`

### Architecture Alignment (2)
- ~~**H-61 | ALIGN-08**: solace.audit topic missing~~ `[FIXED]` — now in SolaceTopic enum
- **H-62 | ALIGN-10**: No Alembic migrations for domain tables `[OPEN]`

### NEW High Issues (April 2026 Audit)

**NEW-02 | LIB-12 | Authorization permission check always fails** `[OPEN]`
- Location: `src/solace_security/authorization.py:109`
- `has_permission()` compares `Permission` enum against `list[str]`. Type mismatch means comparison always returns False. Permission checks silently bypassed.

**NEW-03 | DB-13 | DiagnosisSession missing PHI field declaration for messages** `[OPEN]`
- Location: `src/solace_infrastructure/database/entities/diagnosis_entities.py:76`
- `__phi_fields__ = ["summary"]` but `messages` JSONB column contains full conversation history (PHI). Not encrypted at rest.

**NEW-04 | DB-14 | Hypothesis entity has no __phi_fields__** `[OPEN]`
- Location: `src/solace_infrastructure/database/entities/diagnosis_entities.py:208`
- `Hypothesis` inherits `ClinicalBase` but declares no PHI fields. Supporting evidence and challenge results stored unencrypted.

**NEW-05 | LIB-13 | Service auth header never extracted by FastAPI** `[OPEN]`
- Location: `src/solace_security/service_auth.py:502`
- `_verify_service(authorization: str | None = None)` missing `Header()` dependency. FastAPI won't inject the Authorization header.

**NEW-06 | LIB-14 | Encryption dev key fails own validation** `[OPEN]`
- Location: `src/solace_security/encryption.py:81`
- `"dev-only-insecure-key-32-bytes!!"` is 33 characters. `EncryptionSettings` validation at line 86 requires exactly 32. `for_development()` factory is broken.

---

## MEDIUM Issues (68 + 4 new = 72)

### Safety (6): M-01 through M-06
Duplicate resources, in-memory dedup, L4 misses CRITICAL deterioration, clinician workload drift, fallback_oncall_email no default, config divergence.

### Diagnosis (5): M-07 through M-11
Session messages never populated, delete_user_data skips repo, PHQ-15 only 13 items, SQL syntax, challenge API bypasses Devil's Advocate.

### Therapy (10): M-12 through M-21
SFBT missing from TechniqueSelector, division-by-zero, direct session mutation, in-memory treatment plans, ownership checks missing (2), flexible transitions default, ModalityRegistry unused, progress trend inverted, phase criteria never evaluated.

### Orchestrator + Personality (8): M-22 through M-29
Missing aggregator priorities, supervisor priority map, personality over-injection, style applicator ignores 5 params, emotion agent unreachable, LIWC processor unused, LLM detector redundant, no domain events.

### Memory (6): M-30 through M-35
Crisis content non-permanent, semantic filter single-collection, decay rates inconsistent, consolidation missing fields, token budget ignored, long_term decay skipped.

### Shared Libraries (7): M-36 through M-42
LLM returns empty on error, double timestamp serialization, feature flags test leak, encrypted field rename breaks, private attribute access, wrong env var name, empty ENVIRONMENT bypass.

### Integration (8): M-43 through M-50
Memory events not in registry, session ended missing data, null session_ids in Kafka, dual safety clients, session lifecycle gaps, diagnosis safety flags dropped.

### Supporting (8): M-51 through M-58
Email verification bypass, ServiceIdentity crash, Python version mismatch, no Makefile, CORS invalid, readiness returns 200, network segregation, missing env vars.

### Database (6): M-59 through M-64
Missing join table, SQL syntax error, singleton state, Weaviate vectorizer gap, dual-write gap, Redis TTL too short.

### Alignment (4): M-65 through M-68
Stub agents, in-memory tiers, RoBERTa unwired, no K8s/CI.

### NEW Medium Issues (April 2026 Audit)

**NEW-07 | LIB-15 | Phone masking strips formatting** `[OPEN]`
- Location: `src/solace_security/phi_protection.py:218-222`
- `_mask_phone()` extracts digits, masks, then returns unformatted digits. `"(555) 123-4567"` becomes `"****4567"` instead of `"(***) ***-4567"`.

**NEW-08 | LIB-16 | SSN-without-dashes pattern too broad** `[OPEN]`
- Location: `src/solace_security/phi_protection.py:165`
- `\b\d{9}\b` at confidence 0.80 matches any 9-digit number (ZIP+area codes, product IDs). High false positive rate.

**NEW-09 | LIB-17 | MIN_CONFIDENCE_THRESHOLD defined but unused** `[OPEN]`
- Location: `src/solace_security/phi_protection.py:96`
- `MIN_CONFIDENCE_THRESHOLD = 0.80` defined but never applied in detection logic.

**NEW-10 | INFRA-01 | Connection pool TOCTOU race condition** `[OPEN]`
- Location: `src/solace_infrastructure/database/connection_manager.py:205-223`
- First pool existence check (line 205) is outside the lock. Between check and lock acquisition, another coroutine could remove the pool.

---

## LOW Issues (35)

Safety(3), Diagnosis(3), Therapy(4), Orchestrator+Personality(4), Memory(4), Shared(4), Integration(2), Supporting(5), Database(3), Alignment(3).

Key items: dead cache code, unauthenticated endpoints, broad keyword matching, formatting artifacts, duplicate types, missing health_check method, port conflicts, missing reports.py module, invalid Fernet test key, partition counts.

---

## Resolution Log

| Issue | Status | Resolution | Date | Commit |
|-------|--------|-----------|------|--------|
| C-02 | FIXED | `retry_async` signature corrected | ~2026-03 | -- |
| C-03 | FIXED | `_TOPIC_MAP` user prefix added | ~2026-03 | -- |
| C-04 | FIXED | SolaceTopic NOTIFICATIONS/AUDIT/USERS added | ~2026-03 | -- |
| C-17 | FIXED | TreatmentPlanner/HomeworkManager/ProgressTracker wired | 2026-03 | `2405c3a` |
| H-02 | FIXED | recommended_action recalculated after L2 | 2026-03 | `9ea7730` |
| H-35 | FIXED | All event schemas exported | ~2026-03 | -- |
| H-36 | FIXED | SFBT added to TherapyModality | ~2026-03 | `fa44e26` |
| H-42 | FIXED | Same as H-36 | ~2026-03 | `fa44e26` |
| H-44 | FIXED | Same as C-03 | ~2026-03 | -- |
| H-59 | FIXED | SolaceTopic now 12 topics | ~2026-03 | -- |
| H-61 | FIXED | Same as H-59 | ~2026-03 | -- |
| C-01 | PARTIAL | InMemoryTokenBlacklist passed (per-worker only) | ~2026-03 | -- |
| C-18 | PARTIAL | Redis wired for working memory | 2026-03 | `3732606` |
| C-19 | PARTIAL | Redis cache instantiated | 2026-03 | `3732606` |
| H-04 | PARTIAL | InMemoryEscalationRepository added | 2026-03 | `fc2403b` |
| H-20 | PARTIAL | AssessmentAgent wired | 2026-03 | `3732606` |
| H-31 | PARTIAL | Session count recovery added | 2026-03 | `3732606` |
| H-47 | PARTIAL | CI workflow created | 2026-03 | `fc2403b` |

---

## Summary by Module

| Module | Critical | High | Medium | Low | Fixed | Open |
|--------|----------|------|--------|-----|-------|------|
| Foundation (Libs+DB+Infra) | 10 | 20 | 21 | 14 | 8 | **57** |
| Safety Service | 2 | 6 | 6 | 3 | 1 | **16** |
| Diagnosis Service | 2 | 5 | 5 | 3 | 0 | **15** |
| Therapy Service | 2 | 8 | 10 | 4 | 1 | **23** |
| Orchestrator+Personality | 0 | 7 | 8 | 4 | 1 | **18** |
| Memory Service | 4 | 8 | 6 | 4 | 3 | **19** |
| Integration+Data Flows | 1 | 6 | 8 | 2 | 3 | **14** |
| Supporting Services | 3 | 2 | 8 | 5 | 1 | **17** |
| **New (Apr audit)** | 1 | 5 | 4 | 0 | 0 | **10** |
| **TOTAL** | **25** | **67** | **76** | **35** | **18** | **186** |

### Revised Top 10 Fix-First Issues (updated 2026-04-12 after false-positive correction)

**Confirmed false positives (NOT open):**
- ~~NEW-01~~: `hmac.new()` accepts string digestmod in Python 3.4+
- ~~NEW-02~~: authorization.py permission check is correct (both sides are Permission enums)
- ~~NEW-06~~: dev key is exactly 32 chars
- ~~C-05, C-07, H-47, H-57~~: Already fixed in subsequent commits (migrations have 36 tables, alembic.ini exists, CI workflow live, configure_phi_encryption called in all services)

**Actually-open top priorities:**

1. **NEW-03 + NEW-04 + encrypt_phi_fields str-only limitation**: DiagnosisSession.messages and Hypothesis evidence fields are JSONB lists, not strings. Adding to `__phi_fields__` won't encrypt them until `encrypt_phi_fields` handles list/dict via JSON serialization. Sprint 1 Day 2.
2. **C-01 remaining**: Replace InMemoryTokenBlacklist with RedisTokenBlacklist in middleware (pattern exists in user-service). Sprint 1 Day 1.
3. **NEW-05**: service_auth.py:499 missing `Header()` dep — authorization header not extracted by FastAPI. Sprint 1 Day 1.
4. **C-12**: Safety Layer 1 regex bypassed when ML active. Sprint 2 Day 1.
5. **C-13 + C-14**: Safety + Diagnosis event dispatchers never wired. Sprints 2-3.
6. **C-15**: Diagnosis confidence thresholds don't match spec (no ESCALATE tier <0.30). Sprint 3.
7. **C-16**: Therapy remission classification unreachable branch. Sprint 4 Day 1.
8. **C-20 + C-21**: Memory decay formula wrong in both Python and SQL. Sprint 5 Day 1.
9. **H-25 + H-26**: Personality agent wrong endpoint paths + missing service auth. Sprint 6 Day 1.
10. **H-38**: Encryption key validates char length not byte length (latent for non-ASCII keys). Sprint 1 Day 2.
