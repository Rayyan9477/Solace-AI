# Solace-AI MVP Issues Registry

> **Generated**: 2026-03-20 | **Scope**: Full codebase review (10 parallel agents, ~270 Python files)
> **Methodology**: Design-spec-aware deep code review against SYSTEM-DESIGN-SUMMARY.md
> **Totals**: 24 Critical, 62 High, 68 Medium, 35 Low = **192 issues**

---

## Table of Contents

1. [CRITICAL Issues (24)](#critical-issues-24)
2. [HIGH Issues (62)](#high-issues-62)
3. [MEDIUM Issues (68)](#medium-issues-68)
4. [LOW Issues (35)](#low-issues-35)
5. [Summary by Module](#summary-by-module)

---

## CRITICAL Issues (24)

### Foundation & Infrastructure

**C-01 | LIB-01 | Middleware JWT crashes every authenticated endpoint**
- Location: `src/solace_security/middleware.py:131`
- `_get_jwt_manager()` creates `JWTManager(settings)` without `token_blacklist` param. `JWTManager.__init__` raises `ValueError("token_blacklist is required")`. Every `get_current_user`/`get_current_service` dependency fails.
- Impact: ALL authenticated API endpoints across ALL 10 services are broken.

**C-02 | LIB-02 | `retry_async` function signature broken**
- Location: `src/solace_common/utils.py:314-340`
- Declared as `async def` but returns wrapper synchronously. Callers get a coroutine object instead of a wrapped function. Retry logic never executes.

**C-03 | LIB-03 | User events route to wrong Kafka topic**
- Location: `src/solace_events/schemas.py:541-546`
- `_TOPIC_MAP` has no `"user."` prefix entry. User lifecycle events (creation, deletion, GDPR consent) route to `solace.analytics` instead of `solace.users`.

**C-04 | LIB-04 | Notification events reference non-existent topic**
- Location: `src/solace_events/schemas.py:544`
- `"notification."` maps to `"solace.notifications"` but `SolaceTopic` enum has no `NOTIFICATIONS` member. Topic never created.

**C-05 | DB-01 | Only 7 reference tables in migration — 22+ domain tables missing**
- Location: `migrations/versions/001_initial_schema.py`
- Migration creates `users`, `audit_logs`, `system_configurations`, `safety_resources`, `clinical_references`, `therapy_techniques`, `safety_events`. Missing: `diagnosis_sessions`, `diagnosis_symptoms`, `diagnosis_hypotheses`, `memory_records`, `session_summaries`, `personality_profiles`, `trait_assessments`, `treatment_plans`, `therapy_sessions`, `homework_assignments`, `consent_records`, `notifications`, etc.

**C-06 | DB-02 | Migration `users` table schema mismatches ORM entity**
- Location: `migrations/versions/001_initial_schema.py:21-39` vs `database/entities/user_entities.py:56-122`
- Migration: `username`, `is_active`, `is_verified`, `roles`(JSONB). ORM: `display_name`, `role`(String), `status`, `phone_number`, `is_on_call`, `timezone`, `email_verified`, `login_attempts`, etc. Completely different schemas.

**C-07 | DB-03 | No `alembic.ini` file exists**
- Location: Project root
- Migrations cannot be executed. `alembic upgrade head` fails. No schema deployment path.

**C-08 | DB-04 | Memory service defines divergent table schemas, bypasses Alembic**
- Location: `services/memory_service/src/infrastructure/postgres_repo.py:27-98`
- Defines own `Table` objects with different column names than centralized ORM (e.g., `record_id` vs `id`). Calls `metadata.create_all` bypassing migrations.

**C-09 | DB-05+06 | PK column name mismatches across all service repos**
- Location: `personality_service/.../postgres_repository.py` (uses `profile_id`), `user-service/.../postgres_repository.py` (uses `user_id`)
- ORM entities use `id` as PK. Every `WHERE profile_id = $1` / `WHERE user_id = $1` query fails with column-not-found.

**C-10 | SUPP-02 | Missing Dockerfiles for analytics-service and config-service**
- `docker-compose.yml` references Dockerfiles that don't exist. `docker-compose up` fails for these services.

**C-11 | SUPP-03 | User-service requires FERNET keys not supplied by docker-compose**
- Location: `services/user-service/src/main.py:213-219`
- Requires `FERNET_TOKEN_KEY` and `FERNET_FIELD_KEY` env vars. Docker-compose sets different variable names. Service crashes on startup.

### Safety Service

**C-12 | SAFETY-01 | Layer 1 regex patterns bypassed when ML KeywordDetector active**
- Location: `services/safety_service/src/domain/crisis_detector.py:679-686`
- When ML `_keyword_detector` is present, inline regex patterns for suicidal ideation, self-harm, plan indicators, farewell messages are completely skipped. Defense-in-depth broken.

**C-13 | SAFETY-02 | Event publisher never wired — zero audit trail**
- Location: `services/safety_service/src/domain/service.py` (entire file)
- `SafetyEventPublisher`, `AuditEventHandler`, all event classes exist but are never instantiated or called. Zero events emitted from safety checks. HIPAA audit compliance fails.

### Diagnosis Service

**C-14 | DIAG-01 | Domain events never dispatched**
- Location: `services/diagnosis_service/src/domain/service.py` (entire file)
- EventDispatcher/EventFactory never called. Kafka bridge starts but receives nothing. Downstream services never learn about diagnosis results.

**C-15 | DIAG-02 | Confidence thresholds don't match spec — no Escalate tier**
- Location: `services/diagnosis_service/src/domain/confidence.py:297-305`
- Code: 0.85+=VERY_HIGH, 0.70+=HIGH, 0.50+=MEDIUM, <0.50=LOW. Spec: 0.70+=High, 0.50-0.70=Moderate, 0.30-0.50=Low, <0.30=Escalate. No escalation for dangerously uncertain cases.

### Therapy Service

**C-16 | THER-01 | Treatment response remission branch unreachable**
- Location: `services/therapy_service/src/domain/treatment_planner.py:543-561`
- Remission check (`current <= 4`) at line 558 is unreachable. Percentage-based branches always return first. Users achieving remission never classified as such.

**C-17 | THER-02 | TreatmentPlanner, HomeworkManager, ProgressTracker all dead code**
- Location: `services/therapy_service/src/domain/service.py:56-67`
- `TherapyOrchestrator` never instantiates these. Treatment plans created via `_create_mock_treatment_plan` (hardcoded moderate/CBT). Stepped care, phase advancement, goal tracking, outcome measurement — all non-functional.

### Memory Service

**C-18 | MEM-01 | All 5 tiers use plain Python dicts**
- Location: `services/memory_service/src/domain/service.py:47-52`
- Spec: T1=in-memory, T2=Redis, T3=Redis+PostgreSQL, T4=PostgreSQL+Weaviate, T5=Weaviate+PostgreSQL. Reality: ALL five tiers are `dict[UUID, ...]`. All memory lost on restart.

**C-19 | MEM-02 | Redis infrastructure exists but never initialized**
- Location: `services/memory_service/src/main.py:80-142`
- `RedisCache` fully implemented in `redis_cache.py` but never instantiated in lifespan. `MemoryService` has no `redis_cache` parameter.

**C-20 | MEM-03 | Decay formula double-compounds — memories decay too fast**
- Location: `services/memory_service/src/domain/service.py:654-657`
- Multiplies `e^(-λt)` by already-decayed `retention_strength` instead of initial stability `S`. Each cycle compounds.

**C-21 | MEM-04 | Postgres batch decay uses linear subtraction instead of exponential**
- Location: `services/memory_service/src/infrastructure/postgres_repo.py:350-354`
- `retention_strength = max(0.0, retention_strength - decay_factor)` — linear. Two incompatible decay models in same system.

### Integration

**C-22 | INTG-01 | Notification service crisis URL points to wrong port**
- Location: `services/notification-service/src/consumers.py:78`
- Default `user_service_url` = `http://localhost:8006` (therapy service port). User service is on 8001. Crisis clinician lookups go to wrong service.

---

## HIGH Issues (62)

### Safety Service (6)
- **H-01 | SAFETY-03**: Escalation manager shutdown never called — httpx client leaks (`service.py:121-125`)
- **H-02 | SAFETY-04**: Layer 2 recommended_action not recalculated after score bump (`crisis_detector.py:476`)
- **H-03 | SAFETY-05**: Medium escalation claims "supervisor notified" but sends nothing (`escalation.py:461-469`)
- **H-04 | SAFETY-06**: All escalation state, risk history, conversation history in-memory (`escalation.py:533`)
- **H-05 | SAFETY-07**: LLM assessor cache ignores user_id — cross-user collisions (`llm_assessor.py:257-261`)
- **H-06 | SAFETY-08**: Protective factors identified but never reduce risk score (`service.py:310-328`)

### Diagnosis Service (5)
- **H-07 | DIAG-03**: Same challenge adjustment applied to ALL hypotheses (`service.py:196-199`)
- **H-08 | DIAG-04**: Bayesian calibrator receives empty symptom list `[]` (`service.py:198`)
- **H-09 | DIAG-05**: PHQ-9 maps MODERATELY_SEVERE to score 2 instead of 3 (`severity.py:246-252`)
- **H-10 | DIAG-07**: PCL-5 only 10 items but thresholds for 20-item scale (`severity.py:329`)
- **H-11 | DIAG-08**: Safety flags detected but never trigger CRISIS phase (`service.py:232-241`)

### Therapy Service (8)
- **H-12 | THER-03**: Technique selection weights don't match spec (`technique_selector.py:120-248`)
- **H-13 | THER-04**: "harm"/"danger" trigger crisis in benign contexts (`service.py:313-315`)
- **H-14 | THER-05**: Homework only assigned during CLOSING phase (15+ real min) (`service.py:222-227`)
- **H-15 | THER-06**: Severe patients get almost no techniques (≤12 min filter) (`technique_selector.py:150-152`)
- **H-16 | THER-07**: TreatmentPlanDTO phase type mismatch (int vs enum) (`schemas.py:155`)
- **H-17 | THER-08**: Session state machine allows ANY transition when flexible=True (`session_manager.py:327`)
- **H-18 | THER-09**: `get_user_progress` only queries active sessions (`service.py:537-601`)
- **H-19 | THER-10**: Progress references `start_time` but attribute is `started_at` (`service.py:554`)

### Orchestrator + Personality (7)
- **H-20 | ORCH-01**: Assessment and Emotion agents are hardcoded stubs (`assessment_agent.py`, `emotion_agent.py`)
- **H-21 | PERS-01**: RoBERTa model never called — 2-source ensemble (`trait_detector.py:247-260`)
- **H-22 | PERS-02**: Ensemble weights don't match spec (`trait_detector.py:27-29`)
- **H-23 | PERS-03**: Multimodal fusion (1408-dim) fully implemented but unused (`ml/multimodal.py`)
- **H-24 | PERS-04**: MoEL empathy (32 emotions) never called (`ml/empathy.py`)
- **H-25 | PERS-08**: Personality agent calls wrong endpoint paths — 404 (`personality_agent.py:183`)
- **H-26 | XSVC-01**: Personality agent lacks service auth headers — 401 (`personality_agent.py:176`)

### Memory Service (8)
- **H-27 | MEM-05**: Weaviate collections created with `Vectors.none()` (`weaviate_repo.py:143`)
- **H-28 | MEM-06**: No embedding generation anywhere (`service.py:116-132`)
- **H-29 | MEM-07**: Context assembler uses keyword matching not relevance formula (`context_assembler.py:264`)
- **H-30 | MEM-09**: Consolidation creates fresh DecayManager each call (`consolidation.py:283`)
- **H-31 | MEM-10**: Session count not persisted — resets on restart (`service.py:224`)
- **H-32 | MEM-11**: `store_session_summary` missing required columns (`postgres_repo.py:271`)
- **H-33 | MEM-12**: Four tier-specific managers are dead code (`working_memory.py`, etc.)
- **H-34 | MEM-08**: Token count uses word-split not tokens (`service.py:214`)

### Shared Libraries (6)
- **H-35 | LIB-05**: 23 event schemas not exported from `__init__` (`solace_events/__init__.py`)
- **H-36 | LIB-06**: TherapyModality enum missing SFBT (`schemas.py:266`)
- **H-37 | LIB-07**: Bare module imports require specific install mode (`solace_infrastructure/__init__.py`)
- **H-38 | LIB-08**: Encryption key validates char length not byte length (`encryption.py:85`)
- **H-39 | LIB-09**: PostgreSQL SSL never passed to asyncpg — PHI unencrypted (`postgres.py:170`)
- **H-40 | LIB-10**: Audit HMAC key defaults to empty string (`audit.py:65`)

### Integration (6)
- **H-41 | INTG-02**: TherapyModality case mismatch (uppercase vs lowercase)
- **H-42 | INTG-03**: SFBT missing from Kafka TherapyModality enum
- **H-43 | INTG-04**: Orchestrator sends `FULL_ASSESSMENT` uppercase, safety expects lowercase — 422
- **H-44 | INTG-05**: User events route to analytics instead of `solace.users`
- **H-45 | INTG-06**: Memory event schemas incompatible with canonical schemas
- **H-46 | INTG-07**: Safety plan + trajectory events excluded from Kafka bridge

### Supporting + Infrastructure (8)
- **H-47 | SUPP-04**: No CI/CD pipeline
- **H-48 | SUPP-05**: In-memory session storage in user-service
- **H-49 | SUPP-06**: Consent repository wrong field names (`consent_version` vs `version`)
- **H-50 | SUPP-07**: Notification service hard-imports `solace_security`
- **H-51 | SUPP-08**: Analytics service hard-imports `solace_security`
- **H-52 | SUPP-09**: Missing infrastructure directories (init-db, prometheus, grafana)
- **H-53 | SUPP-10**: RepositoryFactory defaults crash on repo creation
- **H-54 | SUPP-11**: Notification not on `data` network — can't reach Kafka

### Database (6)
- **H-55 | DB-07**: Diagnosis stores symptoms as JSON blobs, bypassing relational tables
- **H-56 | DB-08**: No Row-Level Security policies
- **H-57 | DB-09**: PHI encryption never wired — `configure_phi_encryption()` never called
- **H-58 | DB-10**: Weaviate missing TherapeuticInsight + CrisisEvent in memory service
- **H-59 | DB-11**: Kafka registers 7 topics not 9 (missing notifications, audit)
- **H-60 | DB-12**: Safety queries non-existent `contraindication_rules` table

### Architecture Alignment (2)
- **H-61 | ALIGN-08**: `solace.audit` topic missing from Kafka registry
- **H-62 | ALIGN-10**: No Alembic migrations for domain tables

---

## MEDIUM Issues (68)

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

---

## LOW Issues (35)

Safety(3), Diagnosis(3), Therapy(4), Orchestrator+Personality(4), Memory(4), Shared(4), Integration(2), Supporting(5), Database(3), Alignment(3).

Key items: dead cache code, unauthenticated endpoints, broad keyword matching, formatting artifacts, duplicate types, missing health_check method, port conflicts, missing reports.py module, invalid Fernet test key, partition counts.

---

## Summary by Module

| Module | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| Foundation (Libs+DB+Infra) | 10 | 20 | 21 | 14 | **65** |
| Safety Service | 2 | 6 | 6 | 3 | **17** |
| Diagnosis Service | 2 | 5 | 5 | 3 | **15** |
| Therapy Service | 2 | 8 | 10 | 4 | **24** |
| Orchestrator+Personality | 0 | 7 | 8 | 4 | **19** |
| Memory Service | 4 | 8 | 6 | 4 | **22** |
| Integration+Data Flows | 1 | 6 | 8 | 2 | **17** |
| Supporting Services | 3 | 2 | 8 | 5 | **13** |
| **TOTAL** | **24** | **62** | **68** | **35** | **192** |

### Top 10 Fix-First Issues

1. **C-01**: JWT middleware crashes all auth — blocks everything
2. **C-05+C-06+C-07**: Missing DB migrations + schema mismatch + no alembic.ini
3. **C-11**: User-service missing FERNET keys — won't start
4. **C-17**: Therapy TreatmentPlanner/HomeworkManager/ProgressTracker dead code
5. **C-18+C-19**: Memory tiers plain dicts + Redis never connected
6. **C-12**: Safety Layer 1 regex bypassed — crisis detection gap
7. **C-13+C-14**: Safety + Diagnosis events never dispatched — event architecture dead
8. **H-25+H-26**: Personality agent wrong URLs + no auth — integration broken
9. **H-43**: Safety check case mismatch — orchestrator safety checks fail 422
10. **C-20+C-21**: Decay formula wrong in both Python and SQL
