# Solace-AI — Functional Completeness Review

**What is needed to make every function of the software work, full and final.**

- **Date:** 2026-08-24
- **Method:** 9 parallel fast agents (Sonnet) reading the actual code, each verifying the prior Fable review's findings (confirm / reject) and locating precise gaps + fixes. Every claim below is grounded in `file:line` evidence.
- **Scope:** FUNCTIONAL / engineering completeness only. **Clinical accuracy and legal/BAA/clinician items are excluded by request** (separate department, not a priority now).
- **Relationship to `READINESS-REVIEW-FABLE.md`:** this doc supersedes it for functional truth — it **corrects several Fable false-positives** (see §5) and pins each real gap to an exact fix.

---

## 1. Verdict

**Not functionally complete.** The core business logic in most services is genuinely implemented and wired (not stubbed) — but the system **cannot build today**, several services **do not boot** under a production/staging env, and the **crisis pipeline, durable state, and cross-service data flow are not wired end-to-end**. The good news: almost everything is a *wiring/provisioning* gap with a small, known fix — not a rewrite.

Counts (deduped, clinical/legal excluded): **~19 BLOCKERS · ~12 MAJOR · handful MINOR.**

---

## 2. Per-service functional status

| Service | Status | One-line |
|---|---|---|
| orchestrator | **partial** | Agent nodes + crisis routing + rate-limit are real; but conversation state is in-memory-only, it never stores memory turns, and its event-bridge init has no prod fail-loud. |
| user | **partial** | Persistence/JWT/refresh/rate-limit real; Argon2 blocks the event loop, no reset-password flow, and a compose env bug crash-loops it on staging. |
| safety | **partial** | Crisis detect/escalate/publish real; contraindication engine is dead code (3 orphaned copies); sentiment sync-call landmine. |
| notification | **partial** | Real email/SMS/push delivery, fail-loud on zero-delivery; but durable DLQ not injected and it stores no delivery history. |
| diagnosis | **partial** | Endpoint→domain→Postgres + Kafka bridge solid; but 500s on the two core write endpoints whenever Redis is down. |
| memory | **partial** | `/store` persists to Postgres, but the orchestrator never calls it, `/context` reads in-process only, and RAG/vector search are dead code. |
| therapy | **partial** | Crisis handling + Redis durability real; but the permanent record drops session content, treatment plans are never persisted, outcome tracking is orphaned, **and therapy crises emit no event**. |
| personality | **works** | Detection→persistence, RLS GUC, and GDPR erasure genuinely wired end-to-end. Minor gaps only. |
| config | **broken** | Cannot start in production (missing config/secrets), "set" path is dead code, all stores in-memory, and nothing else in the system calls it. |
| analytics | **partial** | Kafka path drops 100% of events (datetime bug); ClickHouse + erasure orphaned. REST ingest + reports + `/metrics` work. |

---

## 3. BLOCKERS — functions that do not work until fixed

### 3A. Build & deploy — gates everything (this is the path to AWS staging)
| # | Issue | Location | Fix |
|---|---|---|---|
| B1 | `requirements-runtime.txt` is COPY'd + pip-installed by all 10 Dockerfiles but **exists nowhere** → every image build fails | `services/*/Dockerfile` (e.g. `user-service/Dockerfile:16,20`) | Create `requirements-runtime.txt` at repo root (root `requirements.txt` is the source content); verify each per-service `requirements.txt` layers on top. |
| B2 | `USER_SERVICE_ENV: development` is a **non-overridable literal** (siblings use `${...}`) → under `ENVIRONMENT=staging` the REV-32 guard raises → user-service crash-loops | `docker-compose.yml:206` | `USER_SERVICE_ENV: ${USER_SERVICE_ENV:-development}` + add `USER_SERVICE_ENV=staging` to `docs/STAGING-BRINGUP.md`. |
| B3 | `infrastructure/` package (imported at runtime) is not COPY'd into orchestrator/user images → rate-limit silently off; hard-crash under literal `production` | `orchestrator/src/main.py:334`, `user-service/src/main.py:497`; no Dockerfile copies it | `COPY infrastructure/ ./infrastructure/` in those images, or relocate `api_gateway/` under `src/solace_security`. |
| B4 | CI docker-build uses a per-service build context; Dockerfiles need repo-root context → CI build gate tests nothing real | `.github/workflows/ci.yml:152-184` | Build from root: `docker build -f services/<x>/Dockerfile .`. |
| B5 | Migrations never applied automatically (no Postgres in CI, no alembic step in compose) — only a manual MOTD line | `.github/workflows/ci.yml`, `docker-compose.yml`, `infrastructure/aws-staging/solace-staging.yaml:138` | Add an init container / CI job running `alembic upgrade head` before services start. |

### 3B. Crisis / event pipeline — safety-critical, currently does not flow end-to-end
| # | Issue | Location | Fix |
|---|---|---|---|
| E1 | Crisis/event pipeline **OFF by default** — `kafka_enabled=False`, compose sets no enable flags for safety/therapy/notification | `safety/main.py:37`, `therapy/main.py:39`, `notification/main.py:79`; `docker-compose.yml` | Set `SAFETY_KAFKA_ENABLED`/`THERAPY_KAFKA_ENABLED`/`KAFKA_ENABLED=true` + bootstrap servers. |
| E2 | **Therapy detects a crisis but emits no event** (no `publish()` and no `to_kafka_event` mapping) → therapy crises never reach a clinician | `therapy/src/domain/service.py:406-442`; `events.py:178-241` | Publish `RiskLevelElevatedEvent`/equivalent in `_handle_crisis` + add the missing `to_kafka_event` branch mapping to `CrisisDetectedEvent`. |
| E3 | Safety crisis events ride a **non-durable in-memory `asyncio.Queue`** with a swallow-all `except` → event lost on crash/dispatch failure | `safety/src/events.py:351,393-417` | Persist durably (DB/outbox) at publish time; stop swallowing dispatch exceptions. |
| E4 | **4 of 7 event bridges never start an `OutboxPoller`** → outbox rows never drain to Kafka | `event_bridge.py` in notification, orchestrator, personality, user-service | Instantiate + start `OutboxPoller`, mirroring safety/diagnosis/therapy. |
| E5 | `EventConsumer` claims idempotency **before** handlers run + offset is committed on FAILED → a transient handler failure **permanently de-dupes** a legit redelivery | `src/solace_events/consumer.py:427-450,385-394` | Claim after successful handling (or two-phase reserve→finalize/release). |
| E6 | **No DLQ reaper anywhere** (`process_retriable` never called) → DLQ'd crisis events park forever | `src/solace_events/dead_letter.py:296`; no callers | Background task calling `DeadLetterHandler.process_retriable()` per service. |
| E7 | Notification's **durable Postgres DLQ store is never injected** → in-memory default → failed crisis alert lost on restart | `notification/src/main.py:267-271` (no `dlq_store=`) | Pass `dlq_store=PostgresDLQStore(_event_pool)` into `initialize_safety_consumer`. |
| E8 | The "transactional outbox" insert acquires a **separate connection** from the domain write → not atomic | `src/solace_events/postgres_stores.py:120-136`, `publisher.py:452-462` | Pass the domain txn/connection into the outbox insert, or document the at-least-once tradeoff. |
| E9 | Orchestrator Kafka-bridge init is a **blanket `try/except` with no prod fail-loud** (unlike its siblings) → event plane silently disabled if Kafka is down | `orchestrator/src/main.py:182-199` | Add `if settings.environment == "production": raise ...` to the except (audit other services for the same). |

### 3C. Data & state — content and state silently not persisted
| # | Issue | Location | Fix |
|---|---|---|---|
| D1 | Orchestrator **never calls `MemoryServiceClient.store_memory`** → no conversation turn is ever stored (memory-write orphaned at the integration seam) | `orchestrator/src/infrastructure/clients.py:364-371` (zero callers) | Call `store_memory` (or `/session/message`) after each turn for user msg + assistant reply. |
| D2 | Memory `/context` (the LLM-facing endpoint) reads **only in-process dicts**, never Postgres/Weaviate → empty context after restart/other worker; `RAGPipeline` dead code; vector search unused | `memory/src/domain/context_assembler.py:271+`, `service.py:264-271`; `rag_pipeline.py` (no callers) | Wire persisted tiers + real semantic search into context assembly; wire or delete `RAGPipeline`. |
| D3 | Orchestrator **always uses `InMemorySaver`** (`main.py:161` passes no DSN) and the Postgres checkpointer is **doubly broken** (`from_conn_string` used without `async with`; `.setup()` never called) → state lost on restart, not shared across workers | `orchestrator/src/main.py:161`, `langgraph/graph_builder.py:372-402,390` | Pass a DSN into `OrchestratorGraphBuilder`; `async with AsyncPostgresSaver.from_conn_string(url) as saver: await saver.setup()`, hold for process lifetime. |
| D4 | Therapy **treatment plans never persisted** (in-process dict; durable repo only used by the erase path) → cross-worker "Treatment plan not found" | `therapy/src/domain/service.py:87,199,294` | Route plan reads/writes through `unit_of_work.treatment_plans` (read-through cache, like `_load_session`). |
| D5 | Therapy **session content not persisted** — the entity has no transcript field; `end_session` persists only session#/phase/timestamps/skills; full content lives only in Redis (24h TTL, deleted at end) | `therapy/src/domain/entities.py:113-138`, `service.py:376-402` | Populate the full entity from `SessionState`; add a transcript column if long-term storage is required. |
| D6 | Diagnosis **500s on `/assess` and `/session/start` whenever Redis is down** (`try_save_session` raises; caller only catches conflict) despite its own "graceful no-op" doc | `diagnosis/src/infrastructure/redis_session_repository.py:174-180`, `domain/service.py:127-151` | No-op/return False when uninitialized, or gate write-through on `_initialized`. |
| D7 | **GDPR erasure never fans out** — `user.deleted` has no consumers; each downstream service has `erase_user_data` only as an HTTP endpoint → PHI survives deletion everywhere else | `user-service/src/domain/service.py:498`; no subscribers repo-wide | Add a `user.deleted` consumer (or synchronous HTTP fan-out) that calls each service's existing `erase_user_data`. |
| D8 | Memory Postgres write failures are **swallowed** (`except: log`) → returns 201 "stored" even if the durable write failed → silent message loss | `memory/src/domain/service.py:160-161,393-394` | Re-raise or return a `stored=False` signal (match diagnosis's re-raise pattern). |
| D9 | **config_service cannot start in production** — no config files exist, env ingestion returns only `environment`, so secrets stay default → `_validate_production_secrets()` raises | `config_service/src/settings.py:229-250,321-335` | Wire real env/secret ingestion into `_load_from_environment()` or ship `config.production.json`. |
| D10 | **Analytics drops 100% of Kafka events** — raw `model_dump()` passes a `datetime` to `datetime.fromisoformat()` → `TypeError`, silently swallowed; live in the shipped config | `analytics/src/main.py:287-288`, `consumer.py:86,433-436`; `docker-compose.yml:489` | Use `event.model_dump(mode="json")` (or `event.to_dict()`). |
| D11 | Analytics **ClickHouse repo + GDPR erasure are orphaned** (never called) → even with D10 fixed, events only reach the ephemeral in-memory store | `analytics/src/repository.py`, `erasure.py` (test-only callers) | Wire `insert_event`/`insert_events_batch` into the consumer path; wire erasure to a `user.deleted` handler/admin route. |

### 3D. Security / PHI wiring (code activation only — legal excluded)
| # | Issue | Location | Fix |
|---|---|---|---|
| S1 | The tamper-evident **audit chain is instantiated by zero services** → no PHI-access trail exists at runtime (and `AUDIT_HMAC_KEY` is provisioned for dead code) | `src/solace_security/audit.py` (only tests/docs reference it) | `configure_async_audit_logger(PostgresAuditStore(...))` + `.initialize()` in each PHI service lifespan; instrument PHI read/write/export sites. |
| S2 | **RS256 unprovisioned** in every deploy config → prod signs HS256 with one shared symmetric secret injected into every container | `docker-compose.prod.yml:54-57`; no `AUTH_ALGORITHM`/keys anywhere | Generate a keypair; set `AUTH_ALGORITHM=RS256` + `AUTH_PRIVATE_KEY_PEM`/`AUTH_ACTIVE_KID`/`AUTH_PUBLIC_KEYS_JSON` in `.env.prod` + `x-app-secrets`; flip `AUTH_ACCEPT_LEGACY_HS256=false` after rollout. |
| S3 | **No forgot/reset-password flow** — token machinery exists but no routes; `change_password` needs a valid token so it can't help a locked-out user | `user-service/src/infrastructure/token_service.py:184,222` (no callers) | Add `POST /auth/forgot-password` + `POST /auth/reset-password` wired to the existing token functions. |

---

## 4. MAJOR — works but degraded / incomplete

- **M1 — Argon2/bcrypt hashing runs synchronously on the user-service event loop** (9 call sites: register, login incl. timing-equalizer, change-password, OAuth) → serializes all requests per worker under load. Fix: wrap in `asyncio.to_thread`. (`user-service/src/api.py:448,516,524,775`, `password_service.py:92,180,227`)
- **M2 — Clinician/admin cross-user reads return zero rows** — RLS GUC is scoped to the caller with no carve-out; the app layer permits the read but RLS hides the target's rows. The erase path already does target-scoping — apply the same to reads (or grant a BYPASSRLS role). (`migrations/006:91-93`, `memory/src/api.py:232-244`)
- **M3 — Notification stores no delivery history** — no `notifications`/`deliveries` table; you cannot audit "was this crisis alert delivered?" after the fact. Fix: add a table + repository, record each `DeliveryResult`.
- **M4 — Safety contraindication engine is dead code** — a full rules engine plus **two** duplicate DB repos, none instantiated; only therapy's cruder list-check runs. Fix: wire one, delete the duplicates. (`safety/src/ml/contraindication.py`, `db/contraindication_db.py`, `infrastructure/database.py:49`)
- **M5 — Therapy outcome tracking orphaned** — `ProgressTracker` + `PostgresOutcomeMeasureRepository` built but never invoked despite `enable_outcome_tracking=True`.
- **M6 — Sentiment sync-inference landmine (currently inert)** — `CrisisDetector.detect` calls the analyzer synchronously; harmless today only because `transformers`/`torch` aren't installed (`requirements-ml.txt` missing) so it runs lexicon-only. Fix before any ML deps land: offload + decide on deps. (`safety/src/domain/crisis_detector.py:720`)
- **M7 — config_service stores are ephemeral** — feature flags + secrets (default `LOCAL`) are in-memory; the "set config" path is unreachable dead code. (Vault/AWS secret providers are real if selected.)
- **M8 — Observability is decorative** — only `analytics` exposes a working `/metrics`; the other 9 declare `enable_metrics=True` but never mount an endpoint. Fix: add `prometheus-fastapi-instrumentator` per service.
- **M9 — No backup / restore / DR / rollback** for the Postgres PHI datastore (plain local Docker volume, no snapshots/PITR).
- **M10 — Memory semantic search unused** — embeddings are generated + stored, but only BM25 keyword search is reached; `search_similar`/`hybrid_search` are dead code.
- **M11 — Analytics aggregation is in-memory only** (`MetricsStore` resets on restart) — see D11.
- **M12 — config_service is entirely orphaned** — nothing else in the system calls it at runtime; decide integrate-or-drop.

## 4b. MINOR
- Personality RoBERTa detector is a seeded-random stub (weight pinned to `0.0`, so it doesn't corrupt output).
- Personality repository routes to the `"default"` pool instead of `"personality_db"` (pool-name bypass; likely harmless).
- `Caddyfile` has a stale `# rate_limit` comment but no directive (stock Caddy image lacks the module); app-level limiting covers it.
- Stale docs claim a `/metrics` that doesn't exist (safety README).

---

## 5. Corrected Fable false-positives — DO NOT spend effort on these

The earlier Fable review flagged these as broken; direct code reads prove they are **fine**:
- ✅ **PHI encryption IS wired** and fail-loud at startup in every PHI-persisting service (not a "dead guard").
- ✅ **Orchestrator `settings.redis_url` IS defined** (`orchestrator/src/config.py:80`) — not a boot failure.
- ✅ **Orchestrator agent nodes are real and crisis routing is wired end-to-end** (keyword precheck + safety-agent path → crisis handler → Kafka outbox) — not stubbed/dead.
- ✅ **Memory-service `/store` genuinely persists** to Postgres — the real gap is D1 (orchestrator never calls it), not "nothing stores anything".
- ✅ **Personality** detection, persistence, RLS enforcement, and GDPR erasure are all genuinely wired.
- ✅ **Diagnosis** endpoint→domain→Postgres chain is solid and re-raises on persist failure (no silent loss).
- ✅ **Notification** email/SMS/push channels are real and fail-loud on zero delivery.
- ✅ **Analytics `/metrics` and report generation** are real; **rate limiting** on orchestrator + user-service is correctly wired and fail-loud.

---

## 6. Ordered roadmap — build-first, dependency-sequenced

**Week 1 — Make it build and every service boot.** B1 (requirements-runtime), B2 (USER_SERVICE_ENV), B3 (infrastructure/ copy), B4 (CI context), B5 (auto-migrate), D9 (config boot), D6 (diagnosis-Redis 500). → the stack builds and comes up green on staging (needs the AWS grant to actually run).

**Week 2 — Make a crisis reach a clinician end-to-end.** Do the true blockers first: E2 (therapy emits an event), E5 (idempotency ordering), E3 (safety durable fan-out) — then E1 (enable Kafka), E4 (outbox pollers), E7 (durable notification DLQ), E6 (DLQ reaper), E9 (orchestrator fail-loud). E8/atomicity as a design decision.

**Week 3 — Make data and state persist.** D1 (store conversation turns), D2 (context from persistence), D3 (Postgres checkpointer), D4/D5 (therapy plans + session content), D8 (stop swallowing memory writes), D10/D11 (analytics ingest + ClickHouse), D7 (erasure fan-out).

**Week 4 — Activate security wiring + harden.** S1 (audit chain), S2 (RS256 provisioning), S3 (reset-password), M1 (Argon2 offload), M2 (clinician RLS reads), M3 (notification delivery history), M8 (`/metrics` everywhere), M9 (backup/DR).

### Top 10 to fix first
1. **B1** create `requirements-runtime.txt` — unblocks all builds (nothing else can proceed).
2. **B2** fix `USER_SERVICE_ENV` literal — else user-service crash-loops on staging.
3. **E2** therapy must emit a crisis event — else therapy crises never reach a clinician.
4. **E5** fix idempotency claim-before-handler — else a failed crisis event self-dedupes forever.
5. **D1** orchestrator must store conversation turns — else memory/context is empty.
6. **D3** wire the Postgres checkpointer — else all conversation state is lost on restart.
7. **E1 + E4** turn Kafka on + start the outbox pollers — else no events flow at all.
8. **E9** orchestrator event-bridge must fail loud in prod — else the plane silently dies.
9. **D10** analytics `model_dump(mode="json")` — else 100% of analytics events drop.
10. **S1 + S2** wire the audit chain + provision RS256 — the two security features that exist but aren't active.

*(Produced 2026-08-24 by 9 parallel verification agents; clinical/legal items excluded per request. Each finding is code-cited above; false-positives from the prior Fable pass are corrected in §5.)*
