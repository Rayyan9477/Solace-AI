# Solace-AI Repository Review — Final Synthesis

Multi-agent review of `d:/Repo/Contextual-Chatbot` (docs review + code mapping + adversarial claim verification). Date of review: 2026-07-21. All file:line pointers are from verified evidence.

Method: 31 agents — 5 documentation reviewers, 15 code mappers, 10 adversarial claim verifiers, 1 synthesizer.

---

## 1. System map

**What actually exists:** 10 FastAPI microservices under `services/`, 5 shared libraries under `src/` (`solace_common`, `solace_events`, `solace_infrastructure`, `solace_security`, `solace_testing`), a shared LLM gateway in `services/shared/` (Portkey `UnifiedLLMClient`, `llm_client.py:98-100`), Alembic migrations 001-004 (36 tables + RLS + escalations + oauth_accounts), docker-compose dev/prod + Caddy, and ~4,800 tests. There is **no** `libs/` directory, **no** `solace_ml`, **no** api-gateway deployment (the Kong package in `infrastructure/api_gateway/` is dead code), and the root README describes an entirely different legacy Streamlit app.

| Service | Port | Role | Runtime reality |
|---|---|---|---|
| orchestrator | 8000 | LangGraph multi-agent chat pipeline (REST + WS) | Routes are `/api/v1/orchestrator/chat` and `/ws/{session_id}?token=` — not the documented paths; checkpointing always InMemorySaver (`main.py:149-152`) |
| user-service | 8001 | Auth/JWT/OAuth/consent/clinician registry | Register/login/refresh/logout confirmed; OAuth routes absent; on-call-clinicians route shadowed (`api.py:811`) |
| safety | 8002 | 4-layer crisis detection + escalation | Layers/endpoints exist; event publisher never injected (`main.py:119-123`); escalation repo in-memory by default |
| notification | 8003 | Email/SMS/Push + Kafka safety consumer | Kafka consumer off by default and off in compose; SMTP TLS mode broken for default port |
| diagnosis | 8004 | AMIE 4-step chain, Devil's Advocate, confidence calibration | Chain works; persistence broken (NOT NULL `encryption_key_id`, RLS never satisfied); event dispatcher never injected |
| memory | 8005 | 5-tier memory, Ebbinghaus decay, hybrid search | Tier managers unwired; three divergent decay implementations; config split-brain vs compose |
| therapy | 8006 | 6-modality sessions, stepped care, homework | All live state in-memory; Kafka bridge dead via TypeError at startup |
| personality | 8007 | OCEAN ensemble + style adaptation | LLM detector crashes on every call (KeyError); "RoBERTa" is seeded random noise at weight 0.5 |
| config | 8008/8010 | Central config/secrets/flags | An island — nothing calls it; port/healthcheck mismatches make it unreachable under compose |
| analytics | 8009 | Event aggregation + reports | Window-type mismatch means dashboards/reports read zeros; Kafka path fails on every event |

**Integration reality vs. design:**
- **Confirmed:** HTTP/REST + HS256 JWT (not gRPC/RS256/mTLS) — `src/solace_security/auth.py:43`, no `.proto` files. Weaviate 4.10, not ChromaDB — `requirements.txt:35`; no chromadb anywhere. 42 event schemas, 12 topics, outbox/DLQ code all exist in `src/solace_events`. Crisis event schema + notification-service consumer wiring exist (`schemas.py:116-124`, `notification-service/src/consumers.py:133-176`).
- **Diverges critically:** the event plane is **structurally complete but functionally dead**. Every service's bridge uses `use_outbox=True`, but no service starts `OutboxPoller` or calls `flush_outbox` (`src/solace_events/publisher.py:301,333-374` — zero usages in `services/`), and per-service wiring is independently broken (safety: no publisher injected; therapy: `initialize_event_bridge` TypeError swallowed at `main.py:188/196-197`; diagnosis: dispatcher never injected `main.py:128`; orchestrator: nothing publishes to its EventBus; personality/user/notification: events never raised). **No domain event from any service reaches Kafka in practice.** The crisis→notification pipeline works only via the direct HTTP path — which itself is broken by user-service route shadowing.
- **Deployment diverges:** every service Dockerfile COPYs a nonexistent `requirements*.txt` and omits the shared `src/solace_*` + `services/shared` packages, so no container builds/starts as written. No deployment path runs Alembic (`alembic.ini:32` hardcodes wrong DSN; compose/CI never invoke it).

---

## 2. State of the docs

| Cluster | Verdict | Strongest evidence |
|---|---|---|
| **README.md** | **Fiction** — describes a legacy Streamlit/Gemini monolith | `app.py`, `chat.py`, `launch_api.py`, `start.py`, `config.yaml` all absent from repo; pins nonexistent packages (`gemini-api==1.0.1`) |
| **API-HANDOFF.md** | **Aspirational/incorrect on key contracts** | Documented `/api/v1/chat/message`, `/api/v1/ws/chat`, header-based WS auth, OAuth routes, response envelope, rate limits, and WS replay are all contradicted by code (§3 below). A frontend built from this doc would fail on nearly every advanced feature |
| **SYSTEM-DESIGN-SUMMARY.md** | **Good structure, stale status table** (dated 2026-04-11) | Says "Escalation in-memory / event publisher not wired" — PostgresEscalationRepository now exists (`escalation.py:562`, commit 894c3c7) but the in-memory-default part is *still true* (`main.py:115-118`); self-catalogs 8 internal contradictions in §14, plus uncataloged 1h-vs-15min token TTL conflict |
| **AUDIT-REPORT.md** | **Superseded, dangerous if used alone** | Still presents retracted false positives NEW-01/02/06 with "FIX" snippets; its NEW-06 fix is a 31-char key that would *introduce* the bug it claims to fix |
| **BUG-BACKLOG.md** | **Accurate as a 2026-04-12 snapshot, never updated** | Lists C-01, NEW-03/04/05, H-38/39 as open; code verification confirms all were subsequently fixed (`middleware.py:123-168`, `service_auth.py:512-513`, `encryption.py:95-96`, `feature_flags.py:122-130`, `diagnosis_entities.py:78,222-228`) |
| **KNOWN-LIMITATIONS.md** | **Mostly accurate**, one overclaim | "PHI encryption wired in all 8 services" is false — only 6 (missing: user-service, notification-service, analytics-service, config_service). Broken cross-ref: the 3 test flakes are not tracked under DISC- IDs anywhere |
| **MVP-ISSUES.md / MVP-TODOS.md** | **Badly stale** — statuses reflect pre-sprint April | Marks C-12/C-15, NEW-03/04/05 [OPEN]; all verified fixed in code. TODOS still lists fix tasks for retracted false positives (0.6, 0.7, 5.10). Internal count contradictions (~186 vs 203-18) |
| **POST-MVP-BACKLOG.md** | **Most accurate current-state doc** | H-04-full (escalation not wired as default) and C-01-full (Redis not forced in prod) both confirmed still-open in code. DISC-03 partially misleading: argon2-cffi IS in `services/user-service/requirements.txt:12`, missing only from root |
| **MVP-RETROSPECTIVE.md** | **Directionally accurate, overclaims E2E scope** | Sprint fixes verified in code, but "crisis fires escalation event + audit chain, escalations persist via Postgres" is only partially true: E2E test uses InMemoryEscalationRepository (`test_safety_service_e2e.py:85`) and never touches the hash-chained audit log; numeric tensions (3636 vs 582+139 tests; 5 vs 3-4 tracked flakes) |
| **CLINICAL-VALIDATION.md** | **Strongest doc** — clinical thresholds verified | PHQ-9 (`severity.py:432-442`), PCL-5 halved 16/11/9 (`severity.py:461-480` + pinning tests), REMISSION-first (`treatment_planner.py:543-546`), protective-factor cap (`service.py:171-179`), decay fix (`decay_manager.py:111-118`) all confirmed. Misattributes H-13 regexes to safety Layer-1 (they live in therapy `service.py:333`); "ESCALATE blocks insight generation" references code that doesn't exist |
| **Final-System-Architecture / IMPLEMENTATION-PLAN / system-design README** | **Aspirational blueprints, mutually contradictory** | ChromaDB vs Weaviate, 3/4/5 safety layers, three crisis taxonomies, `libs/` vs `src/`, solace-ml (deleted), Kong/Istio/K8s (absent). "Production-Ready" with an all-TBD performance table and unchecked HIPAA checklist |

---

## 3. Doc–code drift (contradicted / partially-true claims — prime bug-hunt leads)

### Contradicted

1. **Orchestrator API surface** — Doc: `POST /api/v1/chat/message`, WS `/api/v1/ws/chat` with `Authorization: Bearer` header. Code: prefix `/api/v1/orchestrator` (`main.py:222`), `POST /chat` (`api.py:99`), WS `/ws/{session_id}` with `token` **query param** (`api.py:249,253`). Header-based WS auth is also un-implementable from browsers.
2. **Google OAuth end-to-end** — No `/auth/oauth/google/start|callback` routes exist anywhere (zero grep hits in `user-service/src/api.py`). `oauth_google.py` is pure helpers; `parse_id_token_payload` decodes **without signature verification** (lines 168-172); no link-or-create, no JWT issuance, no signed cookie. Migration 004 + unit tests exist, but the documented HTTP flow is unbuilt.
3. **WS reconnect session-tail replay** — No replay/resume/buffer code exists; WS handler generates fresh thread/connection IDs per connection (`api.py:272-273`), disconnect just logs (`api.py:351-352`).
4. **Rate limiting (100/user, 10/session, 1000/IP + Retry-After)** — Entirely unimplemented. `Caddyfile:42` is only a comment (and Caddy has no built-in rate_limit); no limiter middleware or Retry-After anywhere in services.
5. **README monolith** — none of the described entry points exist.
6. **ChromaDB as vector store** — code is exclusively Weaviate (`requirements.txt:35`, `weaviate_repo.py`, `weaviate_schema.py`); no chromadb dependency.

### Partially true

7. **Response envelope** — All 7 error codes exist in `src/solace_common/exceptions.py` (:127-:306), but the `{status, data, meta}` envelope is applied nowhere; endpoints return bare Pydantic models and plain `HTTPException`s (`user-service api.py:412-414`); `SolaceError.to_dict` produces a different shape (`exceptions.py:102-105`).
8. **Safety 4 layers / thresholds** — Endpoints and all 4 ML detector files confirmed; but CRITICAL threshold is 0.9 not 0.85 (`value_objects.py:285`), and the code's layers are Input-Gate/Processing-Guard/Output-Filter/Continuous-Monitor, not the doc's L1-keyword…L4-LLM mapping (keyword+pattern fused in Layer 1, `crisis_detector.py:183,232-256`).
9. **Devil's Advocate 6 bias types** — 6 defined (`advocate.py:104-137`) but only 4 ever emitted at runtime (`advocate.py:312-336`); base_rate_neglect and attribution_error are dead metadata.
10. **PHI encryption "all 8 services"** — Only 6 call `configure_phi_encryption` (diagnosis:94, memory:94, orchestrator:100, personality:99, safety:97, therapy:98); **user-service, notification-service, analytics-service, config_service do not**.
11. **argon2-cffi missing (DISC-03)** — missing from root `requirements.txt` but present in `services/user-service/requirements.txt:12` where it's used.
12. **H-13 contextual crisis regexes** — the exact quoted patterns exist verbatim only in the **therapy** service (`therapy_service/src/domain/service.py:333`), not in safety Layer-1 (`crisis_detector.py:129-221`, which uses different patterns achieving similar outcome).
13. **ESCALATE tier "blocks insight generation"** — tiers unified and correct (confidence.py:297-309, entities.py:132-145, value_objects.py:96-111), but no insight-generation code exists to be blocked; ESCALATE is a label with no gating behavior.
14. **Bayesian calibration uses step-1 symptoms** — H-08 fix applied (`service.py:292-297`), but `session.symptoms` merges current-turn symptoms only **after** the chain completes (`service.py:362-365`), so calibration uses prior-turn symptoms.
15. **Crisis E2E (C-13/H-03/H-04)** — E2E test exercises escalation events but with `InMemoryEscalationRepository` (`test_safety_service_e2e.py:85`); the "audit trail" asserted is the event stream, not the HMAC hash-chained audit (`safety_service/src/events.py:420-448` is a plain list); MEDIUM email is unit-tested only.
16. **Microservice decomposition** — services exist with mostly-matching ports, but no api-gateway dir; analytics-service and config_service have no `events.py`; analytics is flat (no domain/ml/infrastructure layers).
17. **Six shared libs under libs/** — five live under `src/` with underscores; `solace_ml` does not exist.
18. **Diagnosis calibration N=3 / "Uncertain"** — `calibration_samples` defaults to 5 (`confidence.py:26`); no "<60% → Uncertain" label anywhere.
19. **Corrective RAG "<3 docs → structured error"** — grading + 2-rephrase cap exist (`rag_pipeline.py:21,190-251`), but FAILED only on zero docs (:244-246); 1-2 relevant docs return SUCCESS.
20. **Memory "diagnoses never decay"** — PERMANENT override works (`decay_manager.py:123-124,214-215`), but clinical content only gets a 0.5 decay modifier (:31,:133-135) and still decays unless explicitly marked permanent.
21. **Roles USER/CLINICIAN/ADMIN/SYSTEM** — actual roles are USER/CLINICIAN/ADMIN/SUPERADMIN/SERVICE (`authorization.py:42-47`); no SYSTEM role.
22. **Safety <10ms keyword-gate latency SLO** — no timing/latency enforcement or tests in `keyword_detector.py`; also `DetectionResult.detection_time_ms` always 0 (`crisis_detector.py:828-837`).
23. **Portkey multi-provider fallback** — implemented, but `enable_fallback` defaults False (`llm_client.py:38`) and ImportError falls back to a None client, so fallback is opt-in, not the described default posture.

---

## 4. Already-known issues (do not re-report in the bug hunt)

**Formally retracted false positives — never re-report:**
- NEW-01: string digestmod in `hmac.new` (`audit.py:152`) — valid since Python 3.4; verified.
- NEW-02: `authorization.py` permission type mismatch — both sides are Permission enums (`authorization.py:112-118`); verified.
- NEW-06: dev key `'dev-only-insecure-key-32-bytes!!'` — exactly 32 chars.
(MVP-TODOS 0.6/0.7/5.10 still list these as tasks — the tasks are stale, not the retractions.)

**Verified fixed (docs may still list as open):** C-12 (Layer-1 regex fusion, `crisis_detector.py:688-696`), C-15 (ESCALATE tier), C-16 (REMISSION-first), C-17, C-20 (decay stability), H-02, H-03 (MEDIUM email path exists), H-06 (protective cap), H-07/H-08 (per-hypothesis + real symptoms, with §3.14 caveat), H-10 (PCL-5 halved), H-22, H-38 (byte-length key check), H-39 (SSL default-on in prod/staging), H-57 (partially — 6 of 8+ services), NEW-03/NEW-04 (JSONB PHI encryption, `base_models.py:448-489`), NEW-05 (Header dependency, `service_auth.py:512-513`), C-01-partial (Redis-preferred blacklist, `middleware.py:123-168`), C-05/C-07/H-47-partial (36-table migration, alembic.ini, CI exists).

**Acknowledged open / deferred (Horizon A/B):** C-01-full (Redis not forced in prod; service tokens hardcode InMemory blacklist `service_auth.py:175,182`); H-04-full (PostgresEscalationRepository not the runtime default); H-56 (RLS on only 3 of ~15 PHI tables, migration 002 docstring); H-47-full (CI lacks gating integration tests/docker); RS256/mTLS/Vault/K8s absent; Apple Sign-In deferred; no Loki/uptime/synthetic canary; no BAA/DPA — explicitly not for live patient onboarding.

**Intentional scope cuts / deviations — not bugs:** PCL-5 10-item screener with halved 16/11/9 cutoffs; no voice input (multimodal voice embedding is mocked); Google-only OAuth (and even that unwired — see §3.2, which IS reportable); Hofstede cultural adaptation unimplemented; single-VPS docker-compose; secrets in `.env.prod`; HS256 shared secret; 100% trace sampling; Kafka data not backed up; cache silent-degrade; frontend out of scope; Q1-Q4 roadmap items (dashboard, FHIR, multilingual etc.) unbuilt by design.

**Known test debt:** 3-5 pre-existing flakes (safety `test_safety_check_crisis_content`, `test_get_crisis_resources`, `test_days_until_review`; therapy `test_request_tracking_headers`; orchestrator `test_memory_node_function`; ordering-dependent `test_delete_user_data`); Kafka integration tests require live broker, excluded from regression; DISC-01 (langgraph pinned 0.2.76), DISC-02 (PyJWT re-verify), DISC-03 (root requirements argon2 — see §3.11), DISC-04 (vignette test over-stubs step 1); hyphenated dirs break repo-wide mypy; accepted lint debt (E701 in severity.py, B904, B017); 2,470-ruff-error baseline is a stale Sprint-0 metric.

**Doc-acknowledged design leftovers:** emotion agent stub; MoEL empathy unwired; multimodal fusion test-only; memory tier managers unwired ("post-MVP" docstring, `memory service.py:38-44`); seed script + DEMO-DATA.md nonexistent.

---

## 5. Top 10 areas of concern (bug-hunt seeds, ranked)

**1. The entire Kafka event plane is dead — including crisis events.**
Outbox is filled but never drained: no service starts `OutboxPoller`/`flush_outbox` (`src/solace_events/publisher.py:301,333-374`). Independently: safety builds `SafetyService` with no publisher (`safety_service/src/main.py:119-123`); therapy's bridge init raises a swallowed TypeError (`therapy_service/src/main.py:188,196-197` vs `event_bridge.py:112-116`); diagnosis never injects its dispatcher (`diagnosis_service/src/main.py:128` vs `:153`); orchestrator/personality/user/notification never publish to their buses (`user-service event_bridge.py:64` never called). Downstream, notification's Kafka consumer is off by default and off in compose (`main.py:204-221`), and analytics' consumer TypeErrors on every event (`consumer.py:86` — `fromisoformat` on a datetime from `model_dump()`). Net effect: `safety.crisis.detected` and the whole audit/analytics event trail never happen. Fail-loud production guards are defeated by broad try/excepts in every lifespan.

**2. Crisis scoring can classify explicit suicidal statements as ELEVATED.**
Fusion divides by the full weight sum even when sentiment/pattern/history are absent, so a lone 0.95 critical keyword yields ~0.62-0.67 — below high (0.7) and critical (0.9) thresholds, no auto-escalation (`safety_service/src/domain/crisis_detector.py:721-740`); LLM averaging (:783) can lower it further; no max-severity override. Compounded by: protective-factor adjustment mutating risk_score after crisis_level derivation (`service.py:171-179`), sentiment neutral-floor bias (`sentiment_analyzer.py:525-539`), negation checking only the first keyword occurrence (:274-288), and per-process risk history (`service.py:120-122`).

**3. Crisis HTTP notification path broken end-to-end by route shadowing.**
`GET /users/{user_id}` registered before `/users/on-call-clinicians` (`user-service/src/api.py:811` vs `:940`) → 422 on every clinician lookup from safety (`clinician_registry.py:146`) and notification (`consumers.py:482`). With #1 killing the Kafka path, **no crisis alert reaches a clinician by either path** except the hardcoded fallback email. Also: escalation `notification_timeout_seconds=300` per attempt × 4 retries blocks `/check` for minutes (`escalation.py:140,213-252`); escalation rows saved once pre-workflow, permanently stale, in-memory repo default (`escalation.py:735,817-832`, `main.py:115-118`).

**4. Diagnosis persistence silently writes nothing; RLS lockout waiting.**
`save_record` omits NOT-NULL `encryption_key_id` → every insert fails, swallowed as a warning (`postgres_repository.py:206`, `service.py:453-454`); `save_session` writes columns that don't exist in migration 001 (:85-88). Migration 002 FORCEs RLS keyed on `app.current_user_id` but **no service code ever sets that GUC** (repo-wide grep) — once applied, reads on diagnosis_sessions/therapy_sessions/memory_records return zero rows and inserts fail WITH CHECK (`002_enable_rls_clinical_tables.py:55-77`). And no deployment path runs Alembic at all (`alembic.ini:32` wrong DSN, env.py ignores DATABASE_URL, compose/CI never migrate).

**5. Authn/authz holes across services (IDOR + revocation).**
Token revocation not enforced on the request path: `decode_token_sync` uses `is_blacklisted_sync`, which the Redis blacklist doesn't override — returns False (`src/solace_security/auth.py:205`, `middleware.py:171-188`). IDOR: orchestrator `GET /sessions/{id}/history` reads any thread_id with no ownership check (`orchestrator api.py:196-229`); diagnosis `end_session`/`challenge` leak another user's differential and file it under the attacker (`diagnosis service.py:427`, `api.py:248`); therapy homework endpoint unowned (`therapy api.py:333`); WS attaches any token to any session_id (`orchestrator api.py:249-290`). Logout never blacklists JWTs; access tokens live 15 min post-logout/deletion (`user-service api.py:538-554,264-299`). Unauthenticated `/status` endpoints on safety, diagnosis, memory, therapy, personality, analytics leak operational stats.

**6. No container in the fleet can build or run.**
Every service Dockerfile `COPY requirements*.txt` matches nothing and copies only its own `src/`, omitting `solace_security/solace_events/solace_infrastructure/services.shared` (e.g. `orchestrator Dockerfile:6,17,22`; identical pattern in safety/therapy/diagnosis/memory/personality/user/notification/analytics). notification/analytics pin `solace-security>=1.0.0` — not the local package, a dependency-confusion risk (`notification requirements.txt:18`). Prod compose interpolates secrets from shell not `.env.prod` — dev JWT secret in prod (`docker-compose.prod.yml:44`); mailhog+kafka-ui host ports stay open in prod; Weaviate server 1.22.4 incompatible with client 4.10 (no gRPC port); config-service listens on 8010 while compose maps/healthchecks 8008 (`config Dockerfile:22`); Prometheus is on the wrong network and scrapes nothing.

**7. Personality service actively corrupts data and crashes.**
`LLMBasedDetector._DETECTION_PROMPT.format()` KeyErrors on every call (unescaped braces, `trait_detector.py:169` — verified by execution), 500ing `/detect` under default settings; `POST /profile/update` discards submitted scores and EMA-drags the profile toward 0.5 via a 40-char placeholder below min_text_length (`api.py:186-193`); "RoBERTa" scores are seeded-random projections given the largest ensemble weight 0.5 (`roberta_model.py:128-146`); `save_profile` wipes assessment_history to `[]` on every upsert (`service.py:176-197`).

**8. Clinical/session state is process-local everywhere; GDPR deletion incomplete.**
Therapy sessions/plans/homework/progress (`session_manager.py:65`), diagnosis sessions (`service.py:69` — `_update_session` silently no-ops on unknown IDs :353-355), safety histories, orchestrator checkpoints (InMemorySaver always; `AsyncPostgresSaver.from_conn_string` misused as a saver, `graph_builder.py:384-402`), memory T1/T2 (`service.py:69-77` — `start_session` wipes working memory :307). GDPR: memory delete never touches Redis (`service.py:545-584`), Weaviate delete can't fail loudly (`weaviate_repo.py:321-334`); diagnosis delete partial-fails with no event (`service.py:542`). Plus memory decay unit inconsistency can wipe long-term memories in days (0.05/hour SQL vs per-day DecayManager rates; `postgres_repo.py:367-394` vs `decay_manager.py:22-25`).

**9. PHI-at-rest and audit-chain gaps behind the "fixed" headlines.**
Encryption not activated in user/notification/analytics/config services (§3.10). Entity-level gaps: SafetyPlan warning_signs/coping_strategies/emergency_contacts, RiskFactor.factor_description, MemoryUserProfile documented "(encrypted)" but absent from `__phi_fields__` (`safety_entities.py:355,261-279,447`; `memory_entities.py:163-195`); encrypted values overflow String(200/300/500) columns (`therapy_entities.py:123`). The audit-chain subsystem is unwired at runtime (no service calls `configure_audit_logger`; `audit.py:741-786` unexported), hash covers only a field subset, and per-process `_last_hash` breaks the chain across restarts/replicas (`audit.py:134,574,699`). Crisis emails embed raw user/session IDs and indicators over external SMTP/SMS (`notification consumers.py:196-255`). `ProductionGuards` validates env vars (`SECRET_KEY`/`ENCRYPTION_KEY`/`JWT_SECRET`) that no settings class reads (`production_guards.py:55`).

**10. Analytics is non-functional and CI can't catch any of this.**
All metrics recorded into MINUTE windows while every report/dashboard queries HOUR/DAY with no rollup → dashboards and all six reports read zeros (`aggregations.py:206-258` vs `reports.py:165-544`); compliance report queries metric names that are never recorded while hardcoding `data_retention_compliant: True` (`reports.py:527-583`) — a misleading regulatory artifact. API-ingested events nest payloads handlers can't see (`api.py:458-471`). Meanwhile CI runs nearly everything with `continue-on-error || true`; orchestrator, user, notification, analytics, config, solace_events/infrastructure, and system suites never run at all; safety `test_api.py` is `--ignore`'d (`.github/workflows/ci.yml:72-101`) — safety-critical regressions merge green.

**Runner-ups worth a pass:** user-service Redis session `TypeError` on every login when Redis is up (`auth.py:476-490` — `ttl=` vs `ex=`); email-verification deadlock for new prod accounts (`service.py:694-696` + `api.py:875-878,417-430`); inconsistent downstream agent URL paths guaranteeing silent 404→canned-fallback (`therapy_agent.py:148` vs `clients.py:294`; `diagnosis_agent.py:159` vs `assessment_agent.py:48`); LangGraph concurrent-write crash on the crisis path (`graph_builder.py:449-460`); dual import roots (`solace_events` vs `src.solace_events`) creating split module identities (`notification consumers.py:37` vs `event_bridge.py`); no LLM request timeout despite the setting (`llm_client.py:281-308`); Postgres outbox `FOR UPDATE SKIP LOCKED` without a transaction (`postgres_stores.py:110`); inconsistent `ENVIRONMENT` vs `{SVC}_ENVIRONMENT` gates silently bypassing fail-loud protections across services.
