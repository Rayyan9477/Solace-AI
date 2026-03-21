# Solace-AI MVP Fix Plan — Prioritized Task List

> **Date**: 2026-03-20
> **Source**: 192 issues from 10-agent deep review against system design spec
> **Goal**: Fix all Critical + High issues to achieve a demo-ready MVP
> **Strategy**: Foundation first (unblock everything), then services (clinical core), then polish

---

## Phase 0: Foundation Unblock (MUST DO FIRST — blocks all other work)

These issues block the entire platform from starting. Fix before anything else.

### 0.1 Fix JWT Authentication (C-01) — Unblocks all API endpoints
- [ ] In `src/solace_security/middleware.py:131`, change `JWTManager(settings)` to `JWTManager(settings, token_blacklist=InMemoryTokenBlacklist())`
- [ ] Add import for `InMemoryTokenBlacklist` from `auth.py`
- [ ] Verify all services can authenticate requests

### 0.2 Fix Database Migrations (C-05, C-06, C-07, C-09)
- [ ] Create `alembic.ini` at project root with proper config
- [ ] Rewrite `001_initial_schema.py` to match actual ORM entities
- [ ] Generate migration for all 22+ domain tables using `alembic revision --autogenerate`
- [ ] Fix PK column name mismatches: personality repo (`profile_id`→`id`), user repo (`user_id`→`id`)
- [ ] Remove `metadata.create_all` from memory service postgres_repo — use migrations instead
- [ ] Fix memory service table schemas to match centralized ORM entities (C-08)

### 0.3 Fix Docker Compose (C-10, C-11, H-52)
- [ ] Create Dockerfiles for `analytics-service` and `config_service`
- [ ] Add `FERNET_TOKEN_KEY` and `FERNET_FIELD_KEY` to user-service env in docker-compose.yml
- [ ] Create `infrastructure/init-db/` with init SQL scripts
- [ ] Create `infrastructure/prometheus/prometheus.yml` with scrape configs
- [ ] Create `infrastructure/grafana/provisioning/` with dashboard provisioning
- [ ] Update `.env.example` with all required env vars

### 0.4 Fix `retry_async` (C-02)
- [ ] Remove `async` from `retry_async` function signature in `src/solace_common/utils.py:314`

### 0.5 Fix Event Topic Routing (C-03, C-04, H-44, H-59, H-61)
- [ ] Add `"user.": "solace.users"` to `_TOPIC_MAP` in `src/solace_events/schemas.py`
- [ ] Add `NOTIFICATIONS = "solace.notifications"` to `SolaceTopic` enum + `TOPIC_CONFIGS`
- [ ] Add `AUDIT = "solace.audit"` to `SolaceTopic` enum with retention=-1 (infinite)
- [ ] Add `USERS = "solace.users"` to `SolaceTopic` enum + `TOPIC_CONFIGS`
- [ ] Add SFBT to `TherapyModality` enum: `SFBT = "SFBT"` (H-36, H-42)
- [ ] Export all 23 missing event classes from `solace_events/__init__.py` (H-35)

---

## Phase 1: Safety Service (PRIORITY — patient safety)

### 1.1 Fix Crisis Detection Gap (C-12)
- [ ] In `crisis_detector.py:679-686`, when ML KeywordDetector is active, ALSO run `self._layer1.detect()` for regex patterns
- [ ] Merge both results (ML keyword score + regex pattern score) before fusion

### 1.2 Wire Safety Event Publisher (C-13)
- [ ] In `SafetyService.initialize()`, create `SafetyEventPublisher` and `AuditEventHandler`
- [ ] Emit `SafetyCheckCompletedEvent` from `check_safety()`
- [ ] Emit `CrisisDetectedEvent` from `detect_crisis()` when crisis_level >= HIGH
- [ ] Emit `EscalationTriggeredEvent` from `escalate()`

### 1.3 Fix Notification Crisis URL (C-22)
- [ ] Change `user_service_url` default from `http://localhost:8006` to `http://localhost:8001`

### 1.4 Fix Escalation Issues (H-02, H-03, H-04)
- [ ] Recalculate `recommended_action` from new crisis_level after L2 adjustment (H-02)
- [ ] In medium escalation workflow, send actual notification or remove false audit claim (H-03)
- [ ] Add PostgreSQL persistence for escalation records via EscalationRepository (H-04)

### 1.5 Fix LLM Assessor Cache (H-05)
- [ ] Include `user_id` in cache key
- [ ] Set TTL=0 for HIGH/CRITICAL assessments (never cache crisis results)

### 1.6 Apply Protective Factors to Risk Score (H-06)
- [ ] After identifying protective factors, compute weighted adjustment
- [ ] Subtract from risk score (with floor — never suppress CRITICAL)

### 1.7 Fix Safety Bridge (H-46)
- [ ] Add `SAFETY_PLAN_CREATED`, `SAFETY_PLAN_ACTIVATED`, `SAFETY_PLAN_UPDATED`, `OUTPUT_FILTERED`, `TRAJECTORY_ALERT` to `_BRIDGED_EVENT_TYPES`

### 1.8 Fix L4 Trajectory (M-03)
- [ ] Include `CrisisLevel.CRITICAL` in deterioration check condition

---

## Phase 2: Diagnosis Service (clinical reasoning core)

### 2.1 Wire Domain Events (C-14)
- [ ] Inject `EventDispatcher` into `DiagnosisService.__init__`
- [ ] Emit events at each step: symptom extracted, hypothesis generated, safety flag raised, diagnosis recorded

### 2.2 Fix Confidence Thresholds (C-15)
- [ ] Align to spec: 0.70+=HIGH, 0.50-0.70=MEDIUM, 0.30-0.50=LOW, <0.30=ESCALATE
- [ ] Add escalation action when confidence < 0.30
- [ ] Unify thresholds across `confidence.py`, `entities.py`, `value_objects.py`

### 2.3 Fix Bayesian Calibration (H-07, H-08)
- [ ] Pass actual symptoms from step 1 to `calibrate()` instead of empty list (H-08)
- [ ] Track per-hypothesis challenge adjustments in step 3, apply individually in step 4 (H-07)

### 2.4 Fix Assessment Scoring (H-09, H-10)
- [ ] Change PHQ-9 MODERATELY_SEVERE mapping from score 2 to score 3 (H-09)
- [ ] Either add remaining 10 PCL-5 items or halve thresholds for 10-item version (H-10)

### 2.5 Wire Safety Flags to CRISIS Phase (H-11)
- [ ] In `_determine_next_phase`, check safety flags before confidence-based routing
- [ ] If `suicidal_ideation` or `self_harm` present, force `DiagnosisPhase.CRISIS`

### 2.6 Fix Session Messages (M-07)
- [ ] Append user message and system response to `session.messages` in `_update_session`

### 2.7 Fix GDPR Delete (M-08)
- [ ] Call `self._repository.delete_user_data(user_id)` in `delete_user_data`

---

## Phase 3: Therapy Service (treatment delivery)

### 3.1 Wire TreatmentPlanner, HomeworkManager, ProgressTracker (C-17)
- [ ] Instantiate `TreatmentPlanner`, `HomeworkManager`, `ProgressTracker`, `InterventionDeliveryService` in lifespan
- [ ] Pass to `TherapyOrchestrator` constructor
- [ ] Replace `_create_mock_treatment_plan` with `TreatmentPlanner.create_plan()`
- [ ] Replace inline homework with `HomeworkManager.assign_homework()`

### 3.2 Fix Remission Classification (C-16)
- [ ] Move remission check (`current <= 4`) to TOP of `_evaluate_treatment_response`

### 3.3 Fix False-Positive Crisis Detection (H-13)
- [ ] Replace bare "harm"/"danger" regex with contextual patterns ("harm myself", "in danger")
- [ ] Or move to soft-alert category that doesn't force CLOSING phase

### 3.4 Fix Homework Phase Restriction (H-14)
- [ ] Allow homework assignment during WORKING phase, not just CLOSING

### 3.5 Fix Technique Duration Filter (H-15)
- [ ] Increase threshold from ≤12 to ≤20 min for severe patients, or use soft penalty

### 3.6 Fix Progress Endpoint (H-18, H-19)
- [ ] Query persistent repository for completed sessions, not just active in-memory
- [ ] Fix attribute names: `start_time` → `started_at`

### 3.7 Fix Session State Machine (H-17)
- [ ] Default `enable_flexible_transitions` to False, or restrict to forward-only transitions

### 3.8 Add SFBT to TechniqueSelector (M-12)
- [ ] Add Miracle Question, Scaling Questions, Exception Finding to technique library

### 3.9 Wire InterventionDeliveryService (M-19)
- [ ] Call `ModalityProvider.generate_response()` when LLM unavailable

### 3.10 Fix Phase Advancement (M-21)
- [ ] Implement evaluation for actual criteria keys: `alliance_rating`, `goals_set`, `skills_acquired`, etc.

---

## Phase 4: Memory Service (context persistence)

### 4.1 Wire Redis for T2/T3 (C-18, C-19)
- [ ] Instantiate `RedisCache` in lifespan
- [ ] Pass to `MemoryService` and use for working memory (T2) and session memory (T3)

### 4.2 Fix Decay Formula (C-20, C-21)
- [ ] Track `stability` (S) separately from `retention_strength`
- [ ] Formula: `retention_strength = e^(-λ*t) * stability` (stability only modified by reinforcement)
- [ ] Fix Postgres batch decay: replace linear subtraction with exponential formula
- [ ] Unify decay config rates across config.py, decay_manager.py, consolidation.py (M-32)

### 4.3 Fix Weaviate Vector Storage (H-27, H-28)
- [ ] Replace `Configure.Vectors.none()` with proper vector config for bring-your-own-vectors
- [ ] Integrate embedding generation (text-embedding-3-small or equivalent) into store pipeline
- [ ] Add TherapeuticInsight + CrisisEvent collections to memory service (H-58/DB-10)

### 4.4 Implement Relevance Scoring (H-29)
- [ ] Replace keyword matching with spec formula: `Semantic×0.4 + Recency×0.3 + Importance×0.2 + Authority×0.1`

### 4.5 Fix Consolidation Pipeline (H-30)
- [ ] Inject shared `DecayManager` instance into `ConsolidationPipeline`

### 4.6 Fix Safety Override (M-30)
- [ ] Detect crisis keywords in `add_message`/`store_memory` and force `retention_category="permanent"`

### 4.7 Wire Tier-Specific Managers or Remove (H-33)
- [ ] Either integrate `WorkingMemoryManager`, `SessionMemoryManager`, etc. into `MemoryService`
- [ ] Or remove dead code to reduce confusion

---

## Phase 5: Orchestrator + Personality (integration glue)

### 5.1 Fix Personality Agent Endpoints (H-25, H-26)
- [ ] Update `PersonalityServiceClient` URLs to include `/api/v1/personality` prefix
- [ ] Use `ServiceClientFactory` from infrastructure/clients.py instead of raw httpx (adds service auth)

### 5.2 Fix Safety Check Case Mismatch (H-43)
- [ ] Change infrastructure `SafetyServiceClient` default from `"FULL_ASSESSMENT"` to `"full_assessment"`
- [ ] Consolidate dual SafetyServiceClient implementations (M-47)

### 5.3 Wire RoBERTa into TraitDetector (H-21, H-22)
- [ ] Import and instantiate `RoBERTaPersonalityDetector` in `TraitDetector.__init__`
- [ ] Route results through `MultimodalFusion.fuse()` or internal ensemble
- [ ] Update weights to spec: RoBERTa=0.5, LLM=0.3, LIWC=0.2

### 5.4 Wire MoEL Empathy (H-24)
- [ ] Connect `MoELEmpathyGenerator` to `PersonalityOrchestrator.adapt_response()`

### 5.5 Upgrade Style Applicator (M-25)
- [ ] Use all 6 style parameters (warmth, structure, complexity, directness, energy, validation)
- [ ] Call personality service `/adapt` endpoint or port `StyleAdapter.adapt_response()` logic

### 5.6 Wire Full LIWC Processor (M-27)
- [ ] Replace inline `LIWCFeatureExtractor` with proper `LIWCProcessor` from ml/liwc_features.py

### 5.7 Fix Stub Agents (H-20)
- [ ] Assessment agent: forward to diagnosis service `/assess` endpoint
- [ ] Emotion agent: use personality service emotion detection or basic keyword classification
- [ ] Add emotion agent to supervisor routing for EMOTIONAL_SUPPORT intent (M-26)

### 5.8 Wire Personality Domain Events (M-29)
- [ ] Emit ProfileCreated, AssessmentCompleted, TraitChanged events

### 5.9 Fix Session Lifecycle (M-48, M-49)
- [ ] Add session initialization call to memory service on session creation
- [ ] Add REST session end endpoint that triggers memory consolidation

---

## Phase 6: Integration + Events (cross-service reliability)

### 6.1 Fix TherapyModality Case Mismatch (H-41)
- [ ] Normalize modality to uppercase before Kafka enum construction: `TherapyModality(modality_str.upper())`

### 6.2 Fix Memory Event Schema Compatibility (H-45)
- [ ] Align memory service event field names with canonical schemas (`record_id`→`memory_id`, `tier`→`memory_tier`)

### 6.3 Fix Session Ended Event Data (M-44)
- [ ] Include `duration_seconds` and `message_count` in `EventFactory.session_ended()` payload

### 6.4 Fix Null Session IDs (M-45, M-46)
- [ ] Extract session_id from `event.aggregate_id` or payload in therapy and personality `to_kafka_event`

### 6.5 Bridge Diagnosis Safety Flags (M-50)
- [ ] Add `SafetyFlagRaisedEvent` mapping in diagnosis `to_kafka_event()`

### 6.6 Auto-Subscribe Orchestrator Kafka Bridge (L-23)
- [ ] Use `subscribe_all()` pattern for orchestrator event bus → Kafka bridge

---

## Phase 7: Supporting Services + Infrastructure

### 7.1 Fix User-Service Repository (H-49, H-53)
- [ ] Fix consent field names: `consent_version` → `version`
- [ ] Wire PostgreSQL client into RepositoryFactory with `use_postgres=True`

### 7.2 Fix Import Fallbacks (H-50, H-51)
- [ ] Add `solace_security` to requirements.txt for notification and analytics services
- [ ] Or add try/except fallback imports

### 7.3 Fix Docker Networking (H-54, M-57)
- [ ] Add notification-service to `data` network
- [ ] Verify all services that need Kafka are on `data` network

### 7.4 Wire Redis Sessions in User-Service (H-48)
- [ ] Instantiate `RedisSessionManager` in lifespan when Redis available

### 7.5 Fix Safety Contraindication Table (H-60/DB-12)
- [ ] Create migration for `contraindication_rules`, `rule_alternatives`, `rule_prerequisites`
- [ ] Or load rules from seed data into existing tables

### 7.6 Add Security Essentials (H-39, H-40, H-56, H-57)
- [ ] Pass SSL context to asyncpg `create_pool()` (H-39)
- [ ] Make HMAC key required in production guards (H-40)
- [ ] Wire `configure_phi_encryption()` at startup (H-57)
- [ ] Add RLS policies to PHI-containing tables (H-56) — can defer to post-MVP

### 7.7 Create CI/CD Pipeline (H-47)
- [ ] Create `.github/workflows/ci.yml`: lint (ruff), type check (mypy), test (pytest), Docker build

---

## Phase 8: Polish (Medium/Low — post-MVP or time permitting)

### 8.1 Safety Polish
- [ ] Deduplicate crisis resources (M-01)
- [ ] Distributed escalation dedup via Redis (M-02)
- [ ] Fix clinician workload tracking (M-04)
- [ ] Add fallback_oncall_email default (M-05)

### 8.2 Diagnosis Polish
- [ ] Fix PHQ-15 item count (M-09)
- [ ] Wire challenge_hypothesis API to Devil's Advocate (M-11)
- [ ] Add reasoning timeout enforcement (L-06)

### 8.3 Therapy Polish
- [ ] Fix division-by-zero when baseline=0 (M-13)
- [ ] Add ownership checks to delete_session and get_treatment_plan APIs (M-16, M-17)
- [ ] Fix progress trend direction for PHQ-9/GAD-7 (M-20)
- [ ] Fix GAD-7 clinical cutoff consistency (L-09)

### 8.4 Memory Polish
- [ ] Fix context assembler deduplication (L-15)
- [ ] Wire HybridSearchEngine into MemoryService (L-17)
- [ ] Increase Redis working memory TTL to 4 hours (M-64)

### 8.5 Infrastructure Polish
- [ ] Create Makefile for dev workflow (M-54)
- [ ] Fix pre-commit Python version to 3.12 (M-53)
- [ ] Fix analytics CORS configuration (M-55)
- [ ] Fix config service readiness to return 503 (M-56)
- [ ] Add runtime dependencies to pyproject.toml (L-29)

---

## Execution Order Summary

| Phase | Issues Fixed | Estimated Scope | Prerequisite |
|-------|------------|-----------------|-------------|
| **0: Foundation** | C-01 through C-11 + H-35,36,44,52,59,61 | ~20 issues, 2-3 days | None |
| **1: Safety** | C-12,C-13,C-22 + 7 HIGHs + 1 MED | ~11 issues, 2 days | Phase 0 |
| **2: Diagnosis** | C-14,C-15 + 5 HIGHs + 2 MEDs | ~9 issues, 1-2 days | Phase 0 |
| **3: Therapy** | C-16,C-17 + 7 HIGHs + 3 MEDs | ~12 issues, 2-3 days | Phase 0 |
| **4: Memory** | C-18-C-21 + 7 HIGHs + 2 MEDs | ~13 issues, 2-3 days | Phase 0 |
| **5: Orch+Pers** | 7 HIGHs + 4 MEDs | ~11 issues, 2 days | Phases 1-4 |
| **6: Integration** | 1 HIGH + 5 MEDs + 1 LOW | ~7 issues, 1 day | Phases 1-5 |
| **7: Supporting** | 7 HIGHs + 2 MEDs | ~9 issues, 1-2 days | Phase 0 |
| **8: Polish** | Remaining MEDs + LOWs | ~20+ issues, ongoing | All above |

**Total Critical Path**: Phases 0-6 = ~72 issues in ~14-18 working days
