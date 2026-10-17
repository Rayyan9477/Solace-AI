# Solace-AI — Enterprise Readiness Assessment

- **Date:** 2026-08-24
- **Scope:** Engineering + production-readiness ONLY. Clinical-accuracy sign-off and legal/BAA/DPA are **out of scope** (separate department). Security/compliance **technical** controls (encryption, audit, RLS, PII-in-logs, secrets) are in scope.
- **Target:** production-grade readiness for a **public enterprise launch** of a HIPAA-scope mental-health AI companion (10 FastAPI microservices, LangGraph, Kafka, PostgreSQL/RLS, Redis, Weaviate, ClickHouse).
- **Method:** 8-dimension specialist deep-dig (each finding independently verified with `file:line`), reconciled against the prior authoritative reviews so this doc **extends** rather than repeats them: `docs/FUNCTIONAL-COMPLETENESS.md` (functional truth, ~19 blockers), `docs/READINESS-REVIEW-FABLE.md` (broad pass, ~1-in-3 false-positive rate), `docs/PHASE-C-SECURITY-FINDINGS.md`, `docs/BUG-BACKLOG.md`, `docs/KNOWN-LIMITATIONS.md`, `docs/SYSTEM-DESIGN-SUMMARY.md`.

---

## 1. Executive summary

**Verdict: NOT enterprise/public-launch ready.** This is not a close call. The business logic inside most services is genuinely implemented (the Functional-Completeness review corrected Fable's false-positives on that point), but the system **cannot build today**, **has never been exercised against real Postgres/Redis/Kafka/Weaviate**, and the surrounding production substrate — supply chain, horizontal scale, HA/DR, alerting, release gating, and several HIPAA-technical controls — is either absent or exists only as **unwired code**.

The dominant failure mode is consistent across every dimension: **capability exists in the repo but is not wired into the deployed system.** The crisis/event pipeline is off by default and never drained end-to-end; the tamper-evident audit chain and GDPR erasure fan-out are instantiated by **zero** services; RS256 is implemented but unprovisioned; the LangGraph checkpointer and durable DLQ default to in-memory; and a hard build blocker (`requirements-runtime.txt` missing) means **no image has ever built**. On top of the functional gaps, the enterprise substrate is missing outright: a single-EC2/single-AZ topology with no autoscaling and process-local session/websocket/checkpoint state (so it physically cannot run more than one replica), no Postgres backup/PITR (RPO effectively infinite for PHI), **no alerting of any kind** on a crisis-safety product, no lockfile or vuln scanning, and RLS multi-tenant PHI isolation that has never been tested against a real database.

**Overall readiness posture: ~1.3 / 5 (early-prototype against an enterprise bar).** The good news is that most functional blockers are small, known wiring/provisioning fixes rather than rewrites; the bad news is that the enterprise dimensions (HA/DR, horizontal scale, supply chain, observability/alerting, release gating) are structural and multi-week. A public HIPAA-scope launch is blocked until both the functional-wiring set **and** the enterprise-substrate set are closed and proven on real infrastructure.

---

## 2. Readiness scorecard

| # | Dimension | Score (0–5) | State (one line) | BLOCKER / HIGH |
|---|---|:--:|---|:--:|
| 1 | Functional completeness & deployment wiring | **1** | Core logic real, but nothing builds, several services don't boot, crisis/state/data flows not wired E2E. | ~19 / ~12 |
| 2 | Dependencies, CVEs & supply chain | **1** | No lockfile/hash-pinning, missing runtime manifest breaks all builds, no vuln scan/SBOM, known-CVE lib on auth path. | 2 / 3 |
| 3 | Container & IaC hardening | **2** | Non-root only; no runtime confinement, plaintext env-var secrets, floating unscanned base images, IMDSv1. | 0 / 3 |
| 4 | Scalability & performance | **1** | Single-instance/single-worker/single-EC2; in-memory state defeats multi-replica; PG connection ceiling; no load test. | 3 / 3 |
| 5 | Reliability, HA & DR | **1** | Single-AZ one failure domain; no PG backup/PITR; health probes don't check deps; false durability promises. | 2 / 3 |
| 6 | Observability & SRE operations | **1** | **Zero alerting**; tracing non-functional in deploy; 9/10 services expose no metrics; DLQ/lag metrics dead code. | 1 / 6 |
| 7 | Testing & release-gating maturity | **1** | RLS/PHI isolation never tested on real PG; zero load/soak/chaos; CI builds an image it never runs; no coverage gate. | 2 / 4 |
| 8 | Data lifecycle & privacy technical controls | **2** | Erasure never fans out; audit trail wired nowhere; regex-only PHI log sanitizer; RLS excludes highest-sensitivity tables. | 2 / 6 |

Composite ≈ **1.3 / 5**. Dimensions 3 and 8 score 2 because real substrate exists (non-root images, encryption-at-rest, partial RLS) but is incomplete for a public HIPAA launch; everything else sits at 1 (prototype-grade).

---

## 3. Launch-blocker register — the definitive "cannot launch until fixed" list

Deduplicated across the enterprise deep-dig and the functional review. Where an item was independently found by both, the dedup note names the twin. **Effort:** S ≤1d · M ≤1wk · L ≤1mo · XL >1mo.

### 3A. Build & boot — gates everything else
| ID | Dim | Title | File:line | Impact | Fix | Effort |
|---|---|---|---|---|---|---|
| BLK-01 | Func/Deps | `requirements-runtime.txt` COPY'd by all 10 Dockerfiles but exists nowhere → every image build fails (= dep finding "runtime manifest absent") | `services/user-service/Dockerfile:16,20` | Nothing has ever built; entire stack undeployable | Create root `requirements-runtime.txt` (+ `requirements-ml.txt`); verify per-service layering | S |
| BLK-02 | Func | `USER_SERVICE_ENV: development` non-overridable literal → REV-32 guard raises under staging → user-service crash-loops | `docker-compose.yml:206` | User-service (auth) never boots on staging/prod | `${USER_SERVICE_ENV:-development}` + set staging value | S |
| BLK-03 | Func | `infrastructure/` package imported at runtime but not COPY'd into orchestrator/user images → rate-limit silently off; hard-crash under literal `production` | `orchestrator/src/main.py:334`, `user-service/src/main.py:497` | Rate limiting absent; boot crash in prod | `COPY infrastructure/` or relocate under `src/` | S |
| BLK-04 | Func/Test | CI `docker-build` uses per-service context but Dockerfiles need repo-root context; job also never runs the image it builds | `.github/workflows/ci.yml:156-184` | Build gate yields zero signal; crash-on-boot passes green | Build from root `docker build -f services/<x>/Dockerfile .`; add `docker run` + health probe | S |
| BLK-05 | Func/Data | Migrations never auto-applied (no Postgres in CI, no alembic step in compose); audit/outbox/DLQ tables live outside Alembic | `docker-compose*.yml` (no alembic); `src/solace_security/audit.py:328` | Schema/RLS unprovisioned at deploy; migration reversibility unproven | Init container / CI job `alembic upgrade head` before services start | M |
| BLK-06 | Func | `config_service` cannot start in production — no config files, env ingestion returns only `environment`, secrets stay default → `_validate_production_secrets()` raises | `config_service/src/settings.py:229-250,321-335` | Config service dead in prod (D9) | Wire real env/secret ingestion or ship `config.production.json` | M |

### 3B. Crisis & event pipeline — safety-critical, does not flow end-to-end
| ID | Dim | Title | File:line | Impact | Fix | Effort |
|---|---|---|---|---|---|---|
| BLK-07 | Func | Crisis/event pipeline **OFF by default** — `kafka_enabled=False`, compose sets no enable flags | `safety/main.py:37`, `therapy/main.py:39`, `notification/main.py:79` | No events flow at all in shipped config | Set `*_KAFKA_ENABLED=true` + bootstrap servers | S |
| BLK-08 | Func | **Therapy detects a crisis but emits no event** (no `publish()`, no `to_kafka_event` mapping) → therapy crises never reach a clinician | `therapy/src/domain/service.py:406-442`; `events.py:178-241` | Silent loss of a life-safety signal | Publish crisis event in `_handle_crisis` + add mapping branch | M |
| BLK-09 | Func | Safety crisis events ride a non-durable in-memory `asyncio.Queue` with swallow-all `except` → lost on crash/dispatch failure | `safety/src/events.py:351,393-417` | Crisis event lost before it reaches the outbox | Persist at publish; stop swallowing dispatch errors | M |
| BLK-10 | Func | **4 of 7 event bridges never start an `OutboxPoller`** → outbox rows never drain to Kafka | `event_bridge.py` in notification/orchestrator/personality/user | Events persist but never publish | Instantiate + start `OutboxPoller` (mirror safety/diagnosis) | M |
| BLK-11 | Func/Rel | `EventConsumer` claims idempotency **before** handlers run + commits offset on FAILED → a transient failure **permanently de-dupes** a legit redelivery (at-most-once on crisis events) | `src/solace_events/consumer.py:427-450,385-394` | Crisis/notification event silently dropped on transient error | Two-phase reserve→finalize/release; DLQ before commit | M |
| BLK-12 | Func/Rel | **No DLQ reaper anywhere** (`process_retriable` never called); poller hot-loops on failure with no backoff | `src/solace_events/dead_letter.py:296`; `publisher.py:622-631` | DLQ'd crisis events park forever; CPU/log storm on Kafka outage | Periodic `DeadLetterHandler.process_retriable()`; exp backoff + jitter | M |
| BLK-13 | Func | Notification's durable Postgres DLQ store never injected → in-memory default → failed crisis alert lost on restart | `notification/src/main.py:267-271` | Crisis alert lost across worker restart (E7) | Inject `dlq_store=PostgresDLQStore(_event_pool)` | S |
| BLK-14 | Func | Orchestrator Kafka-bridge init is a blanket `try/except` with no prod fail-loud → event plane silently disabled if Kafka down | `orchestrator/src/main.py:182-199` | Silent event-plane death in prod (E9) | Fail-loud on `environment=="production"` | S |

### 3C. Data & state persistence
| ID | Dim | Title | File:line | Impact | Fix | Effort |
|---|---|---|---|---|---|---|
| BLK-15 | Func | Orchestrator **never calls `store_memory`** → no conversation turn is ever stored | `orchestrator/src/infrastructure/clients.py:364-371` (zero callers) | Memory write path orphaned (D1) | Call `store_memory`/`/session/message` per turn | M |
| BLK-16 | Func/Scale | Orchestrator **always uses `InMemorySaver`**; Postgres checkpointer doubly broken (`from_conn_string` w/o `async with`, `.setup()` never called) → state lost on restart, not shared across workers | `orchestrator/src/main.py:161`, `graph_builder.py:372-402` | Conversation state non-durable; **defeats multi-replica** (= SCALE in-memory-state blocker) | Pass DSN; `async with AsyncPostgresSaver.from_conn_string(...) as s: await s.setup()`, hold for process lifetime | M |
| BLK-17 | Func | Memory `/context` reads **only in-process dicts**, never Postgres/Weaviate → empty context after restart/other worker; RAG + vector search dead code | `memory/context_assembler.py:271+`, `service.py:264-271` | LLM gets empty long-term context in real deploy (D2) | Wire persisted tiers + semantic search; wire/delete `RAGPipeline` | L |
| BLK-18 | Func | Diagnosis **500s on `/assess` and `/session/start` whenever Redis is down** (write-through raises; caller catches only conflict) | `diagnosis/redis_session_repository.py:174-180`, `service.py:127-151` | Core diagnosis endpoints fail on Redis blip (D6) | No-op/return False when uninitialized; gate on `_initialized` | S |
| BLK-19 | Func | Memory Postgres write failures **swallowed** (`except: log`) → returns 201 "stored" even when durable write failed | `memory/service.py:160-161,393-394` | Silent message loss with success response (D8) | Re-raise / return `stored=False` (match diagnosis) | S |
| BLK-20 | Func | **Analytics drops 100% of Kafka events** — `model_dump()` passes `datetime` to `fromisoformat()` → `TypeError`, swallowed | `analytics/main.py:287-288`; `docker-compose.yml:489` | All analytics telemetry lost (D10) | `model_dump(mode="json")` | S |
| BLK-21 | Func | Therapy **treatment plans + session content never persisted** (in-process dict; entity has no transcript field) → cross-worker "plan not found"; content lost at session end | `therapy/service.py:87,199,294`, `entities.py:113-138` | Therapy record incomplete/lost (D4+D5) | Route through `unit_of_work.treatment_plans`; add transcript column | M |

### 3D. Security & privacy technical controls (code activation; legal excluded)
| ID | Dim | Title | File:line | Impact | Fix | Effort |
|---|---|---|---|---|---|---|
| BLK-22 | Data/Sec | **GDPR/HIPAA right-to-erasure never fans out** — `user.deleted` has zero consumers; PHI survives in 7 of 10 services | `src/solace_infrastructure/gdpr/erasure.py:23-25`; `user-service/service.py:491` | Deletion leaves PHI everywhere but user-service (D7) | Add `user.deleted` consumer / HTTP fan-out calling each `erase_user_data` | M |
| BLK-23 | Data/Sec | **HIPAA audit trail instantiated by zero services** — logger + `log_data_access`/`log_phi_export` exist but never configured or called on any PHI path | `src/solace_security/audit.py:672-687`; services (0 call sites) | No §164.312(b) runtime PHI-access trail (S1) | `configure_async_audit_logger(...)` + `.initialize()` in each PHI lifespan; instrument read/write/export | L |
| BLK-24 | Sec | **RS256 unprovisioned** in every deploy config → prod signs HS256 with one shared symmetric secret in every container (1 container compromise = forge any token) | `docker-compose.prod.yml:54-57` | Auth boundary has no asymmetric protection in prod (S2) | Provision keypair; `AUTH_ALGORITHM=RS256` + keys in `.env.prod`; flip `accept_legacy_hs256=false` post-rollout | M |
| BLK-25 | Func | **No forgot/reset-password flow** — token machinery exists but no routes; locked-out users cannot recover | `user-service/token_service.py:184,222` (no callers) | Account-recovery impossible for real users (S3) | Add `POST /auth/forgot-password` + `/reset-password` | M |

### 3E. Scale, reliability & release-gating — enterprise substrate (NEW; not in functional review)
| ID | Dim | Title | File:line | Impact | Fix | Effort |
|---|---|---|---|---|---|---|
| BLK-26 | Scale | **Single-instance/single-worker/single-EC2 topology** — no ASG/ALB, every Dockerfile CMD a single uvicorn, no `deploy.replicas` | `solace-staging.yaml:95-113`; `services/*/Dockerfile`; `docker-compose.prod.yml` | No horizontal scaling or autoscaling anywhere | ASG + ALB across ≥2 AZs; gunicorn/uvicorn `--workers`; replicas | XL |
| BLK-27 | Scale | **In-memory session/websocket/connection state** defeats multi-instance (code comment admits "deploy with a single worker") | `orchestrator/websocket.py:129-130`; `graph_builder.py:402` | Cannot run >1 replica without breaking sessions/routing (co-req to BLK-16) | Redis pub/sub backplane for WS; externalize connection state | L |
| BLK-28 | Scale | **Postgres connection ceiling** — ~9 services × asyncpg pool(max 20) vs single `max_connections=100`, no PgBouncer/RDS Proxy | `src/solace_infrastructure/postgres.py:48-49`; `docker-compose.yml:17-27` | Pool exhaustion at baseline; horizontal scale makes it worse | PgBouncer/RDS Proxy; tune `max_connections`; right-size pools | M |
| BLK-29 | Rel | **No Postgres backup / PITR / WAL archiving**; staging EBS is `DeleteOnTermination=true` | `solace-staging.yaml:110`; repo-wide (no pgbackrest/wal-g/barman) | RPO effectively infinite; `compose down -v` destroys all PHI, no recovery | Managed RDS/Aurora w/ automated backups + PITR, or WAL→S3; dedicated encrypted PHI volume; rehearsed restore | L |
| BLK-30 | Rel | **Single-EC2 / single-AZ** runs the entire stack — one correlated failure domain including the crisis path | `solace-staging.yaml:95-113` | Any host/AZ loss = 100% outage incl. crisis detection; no SLA achievable | Managed multi-AZ statefuls (RDS Multi-AZ, ElastiCache, MSK 3-broker); ≥2 stateless replicas behind ALB | XL |
| BLK-31 | Obs | **No alerting mechanism exists** — zero Prometheus `rule_files`, no Alertmanager; docs falsely claim it's built | `prometheus.yml` (scrape only); no alertmanager in compose | Every failure mode (Kafka outage, DLQ backlog, dropped crisis alert) is silent to humans | Add rule_files + Alertmanager (or managed); page on crisis-path SLOs | M |
| BLK-32 | Test | **RLS / multi-tenant PHI isolation never tested against a real Postgres** — fixtures default to a mock connection; CI has no `services: postgres`; RLS is a no-op on the mock/SQLite | `src/solace_testing/fixtures.py:25,98,117`; `tests/.../test_rls_guc.py:3` | Core cross-user PHI-isolation guarantee has zero automated proof | Testcontainers/CI Postgres; assert real `CREATE POLICY USING(...)` denies cross-user reads | M |
| BLK-33 | Test/Scale | **Zero load, soak, or chaos testing** anywhere; declared perf/resilience markers used by 0 files; documented latency SLOs unenforced | repo-wide (no k6/Locust/JMeter/toxiproxy) | Concurrent-user ceiling unquantified; cannot set autoscaling/SLA; crisis-latency SLO unproven | Add load harness (k6/Locust) + crisis-path latency gate in CI | M |

> **Blocker note on E8 (outbox not atomic — separate connection from the domain write, `postgres_stores.py:120-136`):** real correctness gap but classed **HIGH**, not a launch blocker, because the current claim is documentable at-least-once; fixing atomicity is a design decision, not a boot/safety gate.

**Deduplicated launch-blocker count: 33.** (Twins collapsed: BLK-01 = dep "runtime manifest absent"; BLK-16 absorbs the SCALE in-memory-checkpointer blocker + functional D3; BLK-22 = data-lifecycle erasure blocker + functional D7; BLK-23 = data-lifecycle audit blocker + functional S1.)

---

## 4. HIGH-priority register — must-fix before public, not strictly launch-gating

| ID | Dim | Title | File:line | Fix |
|---|---|---|---|---|
| H-01 | Deps | No dependency-vuln scan, SBOM, or Dependabot; the only image gate is broken | `.github/workflows/ci.yml` | Add pip-audit/Trivy/syft blocking gate + Dependabot |
| H-02 | Deps | Three divergent dependency definitions (pyproject floors vs root `~=` vs per-service pins) → CI tests a different set than prod ships; crypto versions differ per service | `pyproject.toml`, `requirements.txt`, `analytics/user-service requirements` | Single lockfile (uv/pip-compile) with `--require-hashes` across CI + images |
| H-03 | Deps | Known-CVE `python-jose 3.3.0` (CVE-2024-33664 JWE DoS) live on the API-gateway JWT verification path | `infrastructure/api_gateway/auth_plugin.py:20,188`; `requirements.txt:52` | Migrate gateway to PyJWT (already primary) or patched jose |
| H-04 | Container | No runtime confinement on any prod container (no `cap_drop`, `no-new-privileges`, `read_only`, `pids_limit`) — non-root UID is the only barrier | `docker-compose.prod.yml:59-282` | Add hardening via a shared YAML anchor |
| H-05 | Container | Base images float on mutable `python:3.12-slim` + 3rd-party `:latest`, built on the deploy host, no scan/sign/registry gate | all 10 Dockerfiles; `docker-compose*.yml` | Pin by `@sha256`; move builds to CI (Trivy/cosign/ECR immutable); host pulls digest |
| H-06 | Container/Data | Secrets injected as plaintext env vars from hand-created `.env.prod` — PHI field-encryption master key + audit-HMAC key readable via `docker inspect`/`/proc/environ`; no rotation; violates org Secrets-Manager mandate | `docker-compose.prod.yml:54-112` | AWS Secrets Manager or Docker secrets (`*_FILE`); per-secret KMS CMK; document rotation |
| H-07 | Scale | Argon2id password hashing runs **synchronously** on the single-worker event loop (login-storm DoS) | `user-service/api.py:448,524,775`; `password_service.py:92` | `asyncio.to_thread` all 9 hash/verify call sites |
| H-08 | Scale/Rel | Kafka cannot scale or provide durability — 1-partition auto-created topics, strictly sequential single-consumer loop, single broker so RF=3/minISR=2 config is a false promise (`acks=all` acks after 1 replica) | `docker-compose.yml:87-91`; `consumer.py:380-394` | MSK 3-broker; disable auto-create, provision RF=3/minISR=2; add partitions; alert on under-replicated |
| H-09 | Scale | LLM calls have no concurrency limit/backpressure; large sequential timeout budget per turn (30s × retries × sequential nodes) | `services/shared/infrastructure/llm_client.py:35-36` | Bounded `Semaphore` + per-turn deadline across nodes |
| H-10 | Rel | Readiness/health probes never check datastore connectivity; `/health` (Docker's sole restart signal) is a static string → dead-pool service stays in rotation | `diagnosis/main.py:292-316`; `notification/main.py:336` | `/ready` + healthcheck call `check_health()` + Redis PING + Kafka liveness → 503 |
| H-11 | Rel | Orchestrator LangGraph agents bypass the existing `CircuitBreaker` and open a fresh httpx client per request → no load-shedding on brownout; false resilience | `orchestrator/agents/safety_agent.py:167`; `clients.py:67` | Route via pooled `BaseServiceClient` + per-downstream breaker + bulkhead |
| H-12 | Obs | Distributed tracing non-functional end-to-end — no `OTEL_EXPORTER_OTLP_ENDPOINT` in compose, safety defaults to `localhost:4317`, analytics OTel default off, no ingress instrumentor → Jaeger gets zero spans despite docs | `safety/telemetry.py:46`; compose | Set OTLP endpoint to jaeger; add `FastAPIInstrumentor`; propagate context |
| H-13 | Obs | Cross-service correlation broken — httpx calls pass no `X-Request-ID`/`X-Correlation-ID`; each service mints its own ingress ID | `orchestrator/agents/diagnosis_agent.py:~170,204` | Forward correlation headers on every service hop |
| H-14 | Obs/Data | PHI log sanitizer is **regex-only** — patterns cover SSN/phone/email/MRN/etc. but **no NAME/ADDRESS/DIAGNOSIS/MEDICATION or free-text**, the primary PHI a mental-health chatbot logs | `phi_protection.py:99-174,382-397` | Add name/diagnosis/medication key redaction + free-text handling (NER/allowlist logging) |
| H-15 | Obs | DLQ-backlog + Kafka consumer-lag metrics exist but are **fully unwired dead code** — the exact "lost crisis alert" mode has no metric or endpoint | `get_dlq_stats`/`KafkaMonitor` (0 callers) | Instantiate monitors; expose + alert |
| H-16 | Obs | 9 of 10 services' `enable_metrics=True` is dead — `/metrics`/Instrumentator only in analytics → no RED/USE signals incl. the crisis safety-service | `services/*/main.py` | Add `prometheus-fastapi-instrumentator` per service |
| H-17 | Test | `config_service` has **zero tests**, silently omitted from the CI matrix (no red build) | `services/config_service/` (no tests); `ci.yml:94-104` | Add tests + a completeness check so omissions fail |
| H-18 | Test | Consumer-driven contract framework exists (343 lines) but is used by nothing → schema-drift across 10 services reaches prod silently | `src/solace_testing/contracts.py` (only self-test imports) | Add producer/consumer contract tests for crisis event chain + REST boundaries |
| H-19 | Data/Sec | Audit hash-chain is single-writer-only — read-then-write on in-process `_last_hash`; two replicas both write `previous_hash=H`, then `verify_chain` reports false tampering | `audit.py:609-641,712-748` | DB-side serialization (advisory lock / append-only sequence) for chain tip |
| H-20 | Data/Sec | No operable key rotation — `enable_key_rotation`/`key_rotation_days` decorative; single master key; `decrypt` ignores `key_id`; no KMS/envelope/re-encrypt job | `encryption.py:60-61,145-187,235-251` | Envelope encryption w/ KMS-managed DEKs; key_id-selective decrypt; rotation job |
| H-21 | Data/Sec | Inconsistent PHI crypto — user-service uses **Fernet/AES-128** while platform standard is AES-256-GCM; `create_encryption_service(None)` fail-opens to a generated key with only a warning | `user-service/encryption_service.py:36,57,254-259` | Standardize on AES-256-GCM; make missing-key fail-loud |
| H-22 | Data/Sec | RLS **excludes the highest-sensitivity PHI tables** (`safety_*`/risk profiles) and background-written tables because their writers never set the GUC | `migrations/versions/006_extend_rls_phi_tables.py:15-35` | Set GUC in service-to-service/worker paths; extend policies; prove on real PG |
| H-23 | Rel | E8 — "transactional outbox" insert acquires a separate connection from the domain write → not atomic | `postgres_stores.py:120-136`; `publisher.py:452-462` | Pass domain txn/connection into outbox insert |

Plus MAJOR functional items (works-but-degraded) that should be treated as HIGH before public: **M2** clinician/admin cross-user reads return zero rows under caller-scoped GUC (`migrations/006:91-93`); **M3** notification stores no delivery history (cannot audit "was this crisis alert delivered?"); **M8** observability decorative (= H-16); **D11** analytics ClickHouse + erasure orphaned (aggregation in-memory only).

---

## 5. Per-dimension detail

### Dimension 1 — Functional completeness & deployment wiring — **score 1/5**
**Framework:** does every advertised function actually work end-to-end in a deployed configuration. **Authoritative source:** `docs/FUNCTIONAL-COMPLETENESS.md` (9-agent code-verified, supersedes Fable).
**Top gaps:** the whole 3A/3B/3C/3D blocker set above (BLK-01…25) — nothing builds (B1–B5), crisis pipeline off and not drained (E1–E9), conversation/memory/therapy state not persisted (D1–D6, D8), analytics drops 100% of events (D10), security features unwired (S1–S3).
**Notable MAJOR/MINOR:** M1 Argon2 sync-blocking (H-07); M4 safety contraindication engine is dead code (3 orphaned copies); M5 therapy outcome tracking orphaned; M6 sentiment sync-inference landmine (inert only because ML deps absent); M7/M12 config_service ephemeral + entirely orphaned; M10 memory semantic search dead code. MINOR: personality RoBERTa is a seeded-random stub (weight pinned 0.0, harmless); personality repo routes to `"default"` pool; stale `# rate_limit` Caddyfile comment.
**Corrected Fable false-positives (do NOT rework):** PHI encryption IS wired + fail-loud; orchestrator `redis_url` IS defined; agent nodes + crisis routing ARE wired; memory `/store` DOES persist; personality fully wired; diagnosis re-raises on persist failure; notification channels real + fail-loud.

### Dimension 2 — Dependencies, CVEs & supply chain — **score 1/5**
**Framework:** SLSA / reproducible-build hygiene. **Top gaps:** BLK-01 (missing runtime manifest), H-01 (no scan/SBOM/Dependabot), H-02 (three divergent dependency definitions), H-03 (known-CVE python-jose on the auth path).
**MEDIUM/LOW:** base images float on mutable `python:3.12-slim` with unsnapshotted apt (MEDIUM, = H-05 container view); unbounded specifiers on security-critical libs — `cryptography>=42` no upper bound, `aiohttp~=3.11`, `PyJWT~=2.9` (MEDIUM); pinned-old `langgraph~=0.2.76` deferred-upgrade CVE debt (LOW — the "langchain-community in the crisis path" claim was a **misread**; it is not imported anywhere).

### Dimension 3 — Container & IaC hardening — **score 2/5**
**Framework:** CIS Docker Benchmark + AWS Well-Architected Security. **Top gaps:** H-04 (no runtime confinement), H-05 (unscanned floating base images built on host), H-06 (plaintext env-var secrets incl. PHI master key).
**MEDIUM:** IMDSv1 enabled on staging host, SSM-role credential theft via SSRF (`solace-staging.yaml:95-113`); unrestricted egress `0.0.0.0/0` (`:87-90`); several services (notification/analytics/config/zookeeper) have no resource limits, and no service sets `cpus`/`pids_limit` (`:185-215`); docker-compose plugin curl'd in UserData with no checksum verify (`:124-126`); Redis password not fail-loud (no `:?` guard, inherits `${REDIS_PASSWORD:+...}` — boots unauthenticated if unset), PG/Redis no in-transit TLS (`:229-236`); unhealthy containers never auto-restarted under plain compose — HEALTHCHECK is cosmetic (`:46-47`).
**LOW:** staging host in a public subnet w/ auto-assigned public IP for an SSM-only workload (SG has zero inbound, so low practical risk); runtime images install `curl` solely for HEALTHCHECK (SSRF/exfil tool + CVE surface); EBS root uses AWS-managed key not a CMK, no termination protection/snapshot.
*Note: the staging IaC is explicitly ephemeral synthetic-data-only, which is why several AWS items were downgraded HIGH→MEDIUM/LOW — but it is the **only** launch IaC pattern present, so each becomes real once it seeds prod.*

### Dimension 4 — Scalability & performance — **score 1/5**
**Framework:** horizontal-scale readiness + capacity engineering. **Top gaps:** BLK-26 (single-instance topology), BLK-27/BLK-16 (in-memory state defeats multi-replica), BLK-28 (PG connection ceiling), plus H-07 (Argon2 event-loop block), H-08 (Kafka can't scale), H-09 (no LLM backpressure).
**MEDIUM:** LangGraph Postgres checkpointer will cause write-amplification once wired (full-state serialize per node, `graph_builder.py:384-402` — forward-looking); no load/capacity/SLO baseline (= BLK-33); 9/10 services expose no `/metrics` (= H-16) so autoscaling has no signals; no distributed read cache / read replica on hot paths (infra half confirmed single-primary); containers cap memory but not CPU → noisy-neighbor tail latency on one 4-vCPU host.

### Dimension 5 — Reliability, HA & DR — **score 1/5**
**Framework:** availability SLO + RPO/RTO. **Top gaps:** BLK-29 (no backup/PITR), BLK-30 (single-AZ failure domain), H-10 (probes ignore deps), H-11 (agents bypass circuit breaker), H-08 (false Kafka durability).
**MEDIUM:** crisis escalation blocks the response synchronously ~40-50s with a false circuit-breaker docstring (`safety/escalation.py:216-269`); OutboxPoller hot-loops on failure, no DLQ reaper, no leader election for >1 replica (`publisher.py:622-631`, = BLK-12); liveness can't detect a dead background task (poller/consumer) — `/live` is static (`orchestrator/main.py:454-457`); Kafka consumer advances offset on FAILED, best-effort DLQ only logs (`consumer.py:385-394`, = BLK-11).
**LOW:** circuit breaker has no half-open concurrency limit → thundering herd on recovery (`clients.py:98-106`).

### Dimension 6 — Observability & SRE operations — **score 1/5**
**Framework:** RED/USE + incident-readiness. **Top gap (BLOCKER):** BLK-31 — **no alerting at all** on a crisis-safety product.
**HIGH:** H-12 tracing non-functional in deploy (correction to specialist evidence: safety DOES have full OTel code wired into ML components — `safety/telemetry.py`, `keyword_detector.py:201` — so "only analytics has OTel" is false; the operational conclusion still holds, it's a misconfig so downgraded BLOCKER→HIGH); H-13 broken cross-service correlation; H-14 PHI sanitizer no open-vocab coverage; H-15 DLQ/lag metrics dead code; H-16 9/10 services no `/metrics`; plus service `/health` are unconditional liveness stubs used as Docker's sole restart signal.
**MEDIUM:** Grafana provisions a datasource but ships **zero dashboards** (`infrastructure/grafana/provisioning`); Prometheus scrapes postgres/redis/kafka **raw app ports** with no exporter sidecars → 3 permanently-down jobs (`prometheus.yml:53-63`); no synthetic crisis-pipeline canary (the single check that would have caught REV-41/REV-42 in prod).

### Dimension 7 — Testing & release-gating maturity — **score 1/5**
**Framework:** test pyramid + release-gate integrity. **Top gaps (BLOCKER):** BLK-32 (RLS/PHI isolation never tested on real PG), BLK-33 (zero load/soak/chaos). **HIGH:** H-17 (config_service untested + silently CI-excluded), BLK-04 (CI builds an image it never runs), H-18 (contract framework unused).
**MEDIUM:** pytest marker taxonomy 100% aspirational + two competing pytest configs (root `pytest.ini` wins, `pyproject` block inert) → `-m compliance` shows "0 ran" not a failure; 3 flaky tests permanently muted with `xfail(strict=False)` (zero regression signal); **no coverage threshold** enforced anywhere (`--cov` wired, no `--cov-fail-under`); the only cross-service smoke suite covers 3/10 services and CI never runs it (`tests/system/`); DB migrations never exercised in CI.
**LOW:** a 454-line mocked resilience suite is CI-excluded on a stale "needs real infra" rationale though it is entirely mocked.

### Dimension 8 — Data lifecycle & privacy technical controls — **score 2/5**
**Framework:** HIPAA §164.312 technical safeguards + GDPR data-lifecycle (legal/BAA excluded). **Top gaps (BLOCKER):** BLK-22 (erasure never fans out), BLK-23 (audit trail wired nowhere). **HIGH:** H-14/H-19/H-20/H-21/H-22 (regex-only PHI sanitizer, single-writer audit chain, no key rotation, Fernet/AES-128 vs AES-256-GCM split, RLS excludes safety_*).
**MEDIUM:** migrations never auto-applied + audit/outbox/DLQ tables outside Alembic (owner-bypassable immutability, `audit.py:328`); Weaviate erasure silently skipped when Weaviate down at startup → "complete" erase leaves vectors (`memory/main.py:138-144`); no retention automation — decay unscheduled, audit 6-yr retention unenforced, Kafka retention unapplied; PBKDF2 600k iterations per field on **both** encrypt and decrypt, no DEK caching → N-field read = N×600k SHA256 (DoS/latency, `encryption.py:27,226,248`); `ENCRYPTION_MASTER_KEY` validator **rejects its own documented key-gen command** (`token_bytes(32).hex()` = 64 bytes fails the `==32` check → forces a low-entropy passphrase, `encryption.py:84-103`).
**LOW:** audit chain rehydration failure silently starts a new segment (best-effort, no fail-loud, `audit.py:612-619`).

---

## 6. Prior reviews — already covered vs. NEW here (no double-counting)

**Already covered by prior reviews (this doc reconciles, does not re-discover):**
- `docs/FUNCTIONAL-COMPLETENESS.md` — the entire functional blocker set B1–B5, E1–E9, D1–D11, S1–S3 (mapped here to BLK-01…25). This is the authoritative functional record; this assessment **adopts** it and de-dups against it.
- `docs/PHASE-C-SECURITY-FINDINGS.md` — RS256 provisioning (S2/BLK-24), erase-GUC scoping, cross-user clinician RLS (M2), env-gate REV-32 (BLK-02), Redis/PG TLS floors. Most P0/P1 code items are marked resolved there; the **provisioning/wiring** remainder is what surfaces here as blockers.
- `docs/KNOWN-LIMITATIONS.md` — single-VPS deployment (§3.1 → BLK-30), no Vault (§3.2 → H-06), HS256 default (§3.4 → BLK-24), RLS on 3 tables (§3.5 → H-22), REV-32 env var (§3.7 → BLK-02), 100% trace sampling (§4.1), no synthetic monitoring (§4.3). KNOWN-LIMITATIONS frames these as accepted **demo** cuts; this assessment re-scores them against a **public enterprise** bar where they become blockers/HIGH.
- `docs/READINESS-REVIEW-FABLE.md` — broad "built but not wired" theme; **corrected** for its ~1-in-3 false-positive rate (redis_url, agent-nodes, memory-store all fine).

**NEW in this assessment (not in the prior docs):**
- **Supply chain, quantified:** no lockfile/hash-pinning anywhere (SLSA-L1 fail), three divergent dependency definitions, known-CVE python-jose on the auth boundary, floating/unscanned base images built on the deploy host (BLK-01, H-01…H-03, H-05).
- **Horizontal-scale physics:** in-memory websocket/connection/checkpoint state + PG connection ceiling as concrete multi-replica blockers, not just "single VPS" (BLK-27, BLK-28); Argon2 event-loop DoS; no LLM backpressure (H-07, H-09).
- **HA/DR specifics:** no PG backup/PITR with `DeleteOnTermination=true`; probes ignore datastore health; agents bypass the existing circuit breaker; false RF=3 durability on one broker (BLK-29, H-08, H-10, H-11).
- **Observability, verified end-to-end broken:** zero alerting (not "alerts only" as KNOWN-LIMITATIONS 4.3 implies — there are none); tracing gets zero spans in any deployed config; DLQ/lag metrics are dead code; Grafana ships no dashboards; Prometheus scrapes ports with no exporters (BLK-31, H-12, H-13, H-15).
- **Release-gating integrity:** RLS never tested on real PG; CI builds an image it never runs; contract framework unused; no coverage gate; aspirational marker taxonomy (BLK-32, BLK-04, H-18).
- **Crypto/data-lifecycle depth:** single-writer audit-chain fork under replicas; no operable key rotation; Fernet/AES-128 vs AES-256-GCM inconsistency; PBKDF2 600k per-field DoS; the self-contradicting master-key validator (H-19…H-21, and the MEDIUM crypto items).

---

## 7. Themes & root causes

1. **"Code exists but is not wired into the deployed system."** The single most pervasive pattern. Audit chain, GDPR erasure fan-out, RS256, durable DLQ, Postgres checkpointer, DLQ/lag metrics, contract tests, OTel tracing, per-service `/metrics`, the outbox pollers on 4 bridges — all present in the repo, all inert at runtime. Fixes are mostly small wiring/provisioning tasks, but the aggregate is a system that has **never run end-to-end on real infrastructure**.
2. **No CI gate actually builds, boots, or tests what deploys.** The image build fails from the wrong context and is never run; there is no real Postgres/Redis/Kafka in CI; RLS, migrations, and multi-service contracts are only exercised against mocks; no coverage or perf gate. Green CI therefore proves almost nothing about the deployable artifact — which is why the missing `requirements-runtime.txt` survived.
3. **In-memory process-local state architecturally forbids horizontal scale.** Websocket connections, LangGraph checkpoints, therapy/diagnosis session caches, analytics aggregation, and the audit chain tip all live in per-process memory. The system is not merely deployed single-instance for cost — it **cannot** run more than one replica without breaking sessions, forking the audit chain, and losing state. HA and scale are blocked on the same refactor.
4. **A single, ephemeral, synthetic-data topology is the only deployment pattern that exists.** One EC2, one AZ, one broker, one Postgres on a local volume with `DeleteOnTermination=true`, hand-built images, hand-written `.env.prod` secrets. This is fine as the demo it was scoped to be, but it is the only IaC in the repo, so every "acceptable for demo" cut (no backup, no multi-AZ, plaintext secrets, IMDSv1, open egress) becomes a launch blocker the moment it seeds production.
5. **Safety-critical paths inherit the weakest guarantees.** The crisis pipeline is off by default, rides a non-durable queue, self-de-dupes on transient failure, blocks synchronously on notification, has no durable DLQ or reaper, no alerting, and no synthetic canary — precisely the events that need at-least-once, monitored, fail-loud delivery get at-most-once, silent, best-effort handling.
6. **Documentation asserts controls that the code does not run.** SYSTEM-DESIGN-SUMMARY claims Alertmanager rules, working tracing, and coded dashboards; KNOWN-LIMITATIONS claims audit-chain and encryption coverage that later corrections walk back. Attestations run ahead of runtime reality, which is itself a governance risk for a regulated launch — trust the code, not the docs.

---

*Compiled 2026-08-24 by the lead-architect assessment pass. Read-only; every finding is `file:line`-cited above and reconciled against the prior review record. Clinical-accuracy and legal/BAA/DPA items are excluded by scope and tracked separately.*
