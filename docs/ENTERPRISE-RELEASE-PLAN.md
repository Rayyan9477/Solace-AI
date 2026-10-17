# Solace-AI — Enterprise Release Program Plan

> **For agentic workers:** this is the PROGRAM plan (strategy, phases, gates). Each workstream has (or will have) a detailed task-by-task plan under `docs/superpowers/plans/`. Execute those with `superpowers:subagent-driven-development`. Steps use `- [ ]` checkboxes.

**Goal:** Take Solace-AI from ~1.3/5 prototype to a production-grade, publicly-launchable, HIPAA-scope enterprise service — every function working end-to-end on real infrastructure, with the reliability/security/observability substrate a public launch requires.

**Spec (the "what"):** `docs/ENTERPRISE-READINESS-ASSESSMENT.md` (33 blockers, 23 high, 8 scored dimensions) + `docs/FUNCTIONAL-COMPLETENESS.md`. This plan argues from those; executors read both.

**Scope:** Engineering + production-readiness. **OUT of scope** (separate department, per stakeholder): clinical-accuracy sign-off, HIPAA BAA/DPA legal execution. Security/compliance **technical** controls are IN.

**Tech stack:** Python 3.12 / FastAPI ×10, LangGraph, Kafka (transactional outbox), PostgreSQL + RLS, Redis, Weaviate, ClickHouse, Docker Compose → AWS (target: managed statefuls + ASG/ALB multi-AZ).

## Global constraints (apply to every task)
- **TDD, always:** RED (failing test) → GREEN (minimal impl) → regression. No code without a test that proves it.
- **Prove on real infrastructure:** an item is not "done" until verified against real Postgres/Redis/Kafka/Weaviate — not a mock. This is the single biggest lesson from the assessment.
- **Fail-loud in production:** every guard, encryption, audit, and event path must raise (not warn) when `ENVIRONMENT=production` and a required control is absent.
- **No silent drops on the crisis path:** any change touching safety/crisis must preserve at-least-once delivery + a durable trail.
- **Adversarial verify:** every batch of findings/fixes is independently re-checked before it counts as closed.

---

## 1. Definition of "enterprise-ready" — the release gate (exit criteria)

Launch is authorized only when ALL of these are green (this is the production-readiness review, PRR):

1. **Builds & deploys:** all 10 images build in CI from the correct context, the full stack boots on real infra, migrations auto-apply, and CI runs a build→boot→health gate on every PR.
2. **Crisis integrity:** a synthetic CRITICAL crisis flows detection → escalation → clinician-notification end-to-end on real Kafka, proven by an automated E2E test; no transient failure can silently drop it; DLQ is reaped; alerting pages a human on any crisis-path failure.
3. **Durable state:** conversation turns, session state, therapy records, and analytics survive a restart and are correct across ≥2 replicas.
4. **Security controls active:** RS256 in prod, PHI-access audit trail produced on every PHI read/write, GDPR erasure verified to remove PHI from all services on real PG, secrets in a manager (not plaintext env), PHI log-sanitizer covers names/diagnoses/free-text, no known-CVE deps on the auth path.
5. **Observability:** metrics on all services, working distributed tracing, alerting with runbooks, SLOs defined and measured (esp. crisis-path latency).
6. **Scale & resilience:** ≥2 stateless replicas behind an ALB across ≥2 AZs; managed Postgres with automated backup + PITR + a *rehearsed* restore; Redis/Kafka HA; a load test proves capacity + the crisis-latency SLO; RLS multi-tenant PHI isolation proven on a real database in CI.
7. **Supply chain:** single lockfile with hashes, vuln scan (pip-audit/Trivy) + SBOM as a blocking CI gate, base images pinned by digest.

---

## 2. Phased roadmap

Phases are ordered by dependency. **Phase 0 gates everything.** After it, Phases 1–4 parallelize heavily (different subsystems); Phases 5–6 (scale/HA) need real cloud and run partly in parallel with design starting early. Each phase ends with a **gate** (full regression + adversarial review + the phase's exit criteria) before the next begins.

### Phase 0 — Foundation: build, boot & CI truth  ·  *gates everything*  ·  ~1–1.5 wk
**Objective:** the stack builds, every service boots on real infra, and CI proves what deploys.
**Closes:** BLK-01 (requirements-runtime.txt), BLK-02 (USER_SERVICE_ENV), BLK-03 (infrastructure/ copy), BLK-04 (CI build context + run-the-image), BLK-05 (auto-migrate), BLK-06 (config_service boot), BLK-20 (analytics `mode="json"` — trivial, fold in), H-02 (single lockfile), H-01 (vuln-scan gate).
**Workstreams:** (0a) dependency manifests + lockfile; (0b) Dockerfile/compose env fixes; (0c) CI build→boot→migrate→health gate.
**Exit:** `docker compose ... build && up -d` succeeds on staging; `alembic upgrade head` runs automatically; CI fails on a build/boot/health regression; `pip-audit` gate is green.
**Prereq:** a Docker host — the AWS staging grant (per `infrastructure/aws-staging/`) or a local daemon.

### Phase 1 — Crisis pipeline integrity  ·  safety-critical  ·  ~1–2 wk
**Objective:** a crisis reliably reaches a clinician, end-to-end, with no silent loss.
**Closes:** BLK-07 (enable Kafka), BLK-08 (therapy emits crisis event), BLK-09 (safety durable fan-out), BLK-10 (OutboxPoller ×4), BLK-11 (idempotency two-phase), BLK-12 (DLQ reaper + backoff), BLK-13 (durable notification DLQ), BLK-14 (orchestrator fail-loud). HIGH: H-15 (DLQ/lag metrics).
**Exit:** an automated E2E test drives a synthetic CRITICAL case through detection → escalation → clinician notification on real Kafka; a fault-injection test proves a transient handler failure does NOT drop the event (it retries via DLQ); therapy-originated crises reach the pipeline.

### Phase 2 — Durable data & state  ·  ~1–2 wk  ·  co-req with scale-out (Phase 5)
**Objective:** nothing is silently lost; state is correct across restarts and replicas.
**Closes:** BLK-15 (orchestrator stores turns), BLK-16 (Postgres checkpointer, durable + `.setup()`), BLK-17 (memory /context reads persistence), BLK-18 (diagnosis survives Redis down), BLK-19 (no swallowed memory writes), BLK-21 (therapy plans + transcript persisted).
**Exit:** conversation turns are stored and retrieved across a restart and a second worker; checkpointer is Postgres-backed; therapy record is complete; diagnosis degrades gracefully without Redis; a failed write never returns 201.

### Phase 3 — Security & privacy technical controls  ·  ~2 wk  ·  parallel with 2
**Objective:** the security features that exist in code are active and proven in the deployment.
**Closes:** BLK-22 (erasure fan-out), BLK-23 (audit chain wired + instrumented), BLK-24 (RS256 provisioned), BLK-25 (reset-password flow). HIGH: H-03 (drop `python-jose` CVE), H-06 (secrets manager), H-14 (PHI sanitizer: names/diagnoses/free-text).
**Exit:** PHI-access audit rows are produced on real PG; a deleted user's PHI is verifiably gone from all services; prod tokens are RS256; secrets load from a manager; the sanitizer redacts names/diagnoses; no CVE deps on the auth path.

### Phase 4 — Observability & operability  ·  ~1–2 wk  ·  parallel with 1–3
**Objective:** humans can see and are paged on failure.
**Closes:** BLK-31 (alerting: rules + Alertmanager). HIGH: H-10 (health probes check deps), H-11 (circuit-breaker/bulkhead on inter-service calls), H-12 (tracing works E2E), H-13 (correlation headers), H-15 (DLQ metrics), H-16 (/metrics on all services).
**Exit:** metrics + traces flow from every service; alerts page on crisis-path SLO breach, DLQ backlog, Kafka lag, dead-pool health; runbooks exist for each alert.

### Phase 5 — Scalability & performance  ·  ~2–3 wk  ·  needs Phase 2 (state externalized)
**Objective:** run ≥2 replicas correctly and prove capacity.
**Closes:** BLK-27 (WS/connection state → Redis backplane), BLK-28 (PgBouncer/RDS Proxy + pool tuning), BLK-33 (load/soak/chaos harness). HIGH: H-07 (Argon2 off event loop), H-08 (Kafka partitions/HA), H-09 (LLM backpressure), H-11 (bulkheads).
**Exit:** ≥2 stateless replicas serve traffic without breaking sessions; a k6/Locust load test establishes the concurrent-user ceiling and enforces a crisis-path latency SLO in CI; blocking calls are off the event loop.

### Phase 6 — Reliability, HA & DR  ·  ~3–4 wk (XL)  ·  the long pole; needs cloud spend
**Objective:** no single failure domain; recoverable PHI.
**Closes:** BLK-26 (ASG + ALB), BLK-29 (managed Postgres backup/PITR + rehearsed restore), BLK-30 (multi-AZ managed statefuls: RDS Multi-AZ, ElastiCache, MSK 3-broker).
**Exit:** any single host/AZ loss is survivable; automated backups + a *rehearsed* restore meet stated RTO/RPO; the crisis path has no single point of failure.

### Phase 7 — Hardening & production-readiness review  ·  ~1–2 wk
**Objective:** close the residue and run the launch gate.
**Closes:** H-04 (runtime confinement), H-05 (pinned/scanned/signed images), BLK-32 (RLS proven on real PG in CI), H-17 (config_service tests + CI-matrix completeness check), remaining MEDIUM/LOW.
**Exit:** the §1 PRR checklist is fully green; security scan + chaos + load all pass. → **Launch authorized.**

---

## 3. Critical path, timeline & parallelization

```
Phase 0 (build/boot) ─┬─> Phase 1 (crisis)      ─┐
   [GATES ALL]        ├─> Phase 2 (data/state)  ─┤
                      ├─> Phase 3 (security)     ─┼─> Phase 5 (scale) ─> Phase 6 (HA/DR) ─> Phase 7 (PRR) ─> LAUNCH
                      └─> Phase 4 (observability)─┘        (needs P2)        (long pole)
```

- **Honest timeline.** The *functional + safety + security core* (Phases 0–4) is achievable in **~4–6 weeks** with parallel teams. Full **enterprise HA/DR (Phases 5–6)** adds **~4–6 weeks** and real cloud spend — it is structural, not a wiring fix. So: **~1 month to "works correctly and securely on real infra," ~2–3 months to "public-enterprise HA/DR ready."** (This corrects the "~1 month for everything" expectation — the HA/DR set is genuinely multi-week.)
- **Parallelization:** after Phase 0, Phases 1/2/3/4 touch mostly disjoint subsystems → run them as **concurrent workstreams** (a team of agents per workstream). Phase 5 can begin design immediately but *executes* after Phase 2 externalizes state. Phase 6 procurement/IaC can start in parallel with Phases 1–4.
- **Fastest path to a safe closed beta:** Phases 0 → 1 → 2 → 3 (+ Phase 4 alerting on the crisis path) gets to a correct, secure, observable single-region deployment suitable for a **limited/controlled beta** — before the full multi-AZ HA build.

---

## 4. Execution model (how we run it — team of agents)

- **Per workstream:** a detailed `docs/superpowers/plans/*.md` task plan (bite-sized TDD steps, exact files, code). Executed via `superpowers:subagent-driven-development` — a fresh subagent per task + two-stage review.
- **Parallel fan-out:** independent workstreams run as concurrent agents (the pattern used throughout this project). File-disjoint clustering; integration + adversarial verify after each batch.
- **Per-phase gate (industry-standard):** full regression (all suites green) + adversarial review + the phase's exit criteria proven on real infra, before advancing. Mirrors a staged production-readiness review.
- **No git commits** until the stakeholder lifts that constraint (current standing rule); progress tracked in these docs + memory.

---

## 5. Top risks & mitigations
- **No real infra yet** → Phases 1–6 verification is blocked until staging stands up. *Mitigation:* Phase 0's exit is literally "stack runs on real infra"; unblock via the AWS grant (`infrastructure/aws-staging/`) or a local Docker host first.
- **HA/DR cost & time** (Phase 6, XL) → *Mitigation:* stage it; run a correct single-region beta (Phases 0–4) while HA/DR is built; use managed services (RDS/ElastiCache/MSK) to buy reliability instead of hand-rolling.
- **Crisis-path regressions during refactor** → *Mitigation:* the Phase 1 E2E + fault-injection tests become permanent CI gates before any later phase touches that path.
- **False "done" from mock-only tests** → *Mitigation:* the global constraint — nothing counts until proven on real Postgres/Redis/Kafka.

---

## 6. Immediate next actions
1. **Approve this plan** (or adjust phase ordering / timeline).
2. **Generate the detailed per-workstream task plans** (bite-sized TDD, writing-plans format) — starting with Phase 0 and Phase 1, in parallel. *(Ready to launch on your go.)*
3. **Unblock real infra** — the AWS IAM grant (`infrastructure/aws-staging/AWS-STAGING-IAM-SETUP.pdf`) or a local Docker daemon — so Phase 0's "runs on real infra" exit can actually be met.

*(Program plan produced 2026-08-24 from the enterprise readiness assessment. Detailed executable task plans follow per workstream under docs/superpowers/plans/.)*
