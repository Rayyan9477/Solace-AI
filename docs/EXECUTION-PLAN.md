# Solace-AI — Execution Plan to MVP & Enterprise Launch

> **Status**: Approved 2026-07-21 · durable in-repo source of truth (exported from the planning session).
> **Update cadence**: refresh the `REV-` status column in [BUG-BACKLOG.md](BUG-BACKLOG.md) and the sprint checkboxes here at each Phase Gate (step G7).
> **Companion docs**: [SYSTEM-REVIEW-2026-07-21.md](SYSTEM-REVIEW-2026-07-21.md) (full findings) · [ENTERPRISE-READINESS-REMEDIATION-PLAN.md](ENTERPRISE-READINESS-REMEDIATION-PLAN.md) (workstreams) · [BUG-BACKLOG.md](BUG-BACKLOG.md) (`REV-` per-issue detail).

## Context

A 2026-07-21 multi-agent review found Solace-AI is **not launch-ready**: the crisis-safety pipeline can't deliver end-to-end, clinical persistence silently no-ops, HIPAA controls are partially wired, and nothing builds/deploys as written. The 31 verified-open issues (`REV-01..31`) and their fixes are compiled in `docs/BUG-BACKLOG.md` and grouped into 11 workstreams in `docs/ENTERPRISE-READINESS-REMEDIATION-PLAN.md`. All seven P0s were re-verified against current code on 2026-07-21 (line numbers below reflect the current tree).

This plan turns that into an **executable, solo, 1-week-sprint sequence**, driven by TDD and the installed skills/plugins/MCPs, with two milestones:
- **Milestone 1 — MVP (safe clinician demo):** trustworthy CI + all P0 crisis-safety and clinical-data-integrity fixes, verified at unit/integration level.
- **Milestone 2 — Enterprise launch:** deployability, service correctness, truthful API, clinical-accuracy reconciliation, and enterprise/compliance posture (RS256/mTLS/Vault/HA/DR/RLS-full/BAA), verified end-to-end on a real staging stack stood up in this phase.

**Execution assumptions:** 1 engineer, strictly sequential critical path, 1-week sprints, **TDD throughout, and no git commits** — progress is tracked via TodoWrite + status updates in `docs/BUG-BACKLOG.md` and this plan, not via commits/branches. Verification is unit + mocked-integration until Sprint B.3 provisions staging (Docker is not on the dev machine today; fixtures mock PG/Redis/Kafka/Weaviate).

---

## Execution method — skills, plugins & TDD discipline (applies to every ticket)

This is the working loop. Each ticket in every sprint runs through it. Skills are invoked by name; **process skills first (they set the approach), then implementation.** No git commits at any step — "done" is proven by passing tests + verification + independent review, and recorded in docs/TodoWrite.

**Per-ticket loop:**
1. **Scope & impact** — `mcp__gitnexus__impact` (or `/gitnexus-impact-analysis`) on the target symbol for blast radius; if >10 callers, route to refactor not a point fix. Use `/gitnexus-exploring` / `understand-anything` to trace the flow first. Verify any external-library usage against **Context7 MCP** (`resolve-library-id` → `query-docs`) — FastAPI, SQLAlchemy/asyncpg, Alembic, LangGraph, aiokafka, Weaviate, Caddy, PyJWT, cryptography — never fix from memory on library semantics.
2. **Brainstorm (design-heavy tickets only)** — `superpowers:brainstorming` before writing code for anything with a design choice (event wiring, RLS middleware, auth path, rate limiting).
3. **RED** — `superpowers:test-driven-development` (primary) / `agent-skills:test-driven-development`: write the failing test encoding the acceptance criterion. For bug fixes use the **Prove-It pattern** — a regression test that fails reproducing the bug first. `agent-skills:spec` for tickets that need a written contract before code (API, OAuth).
4. **GREEN** — minimal implementation to pass. `agent-skills:incremental-implementation` / `agent-skills:build` for the build-test-verify rhythm (commit step skipped).
5. **Debug when stuck** — `superpowers:systematic-debugging` (hypothesis → repro → minimal fix). Escalate stubborn failures to the `bug-analyzer-reproducer`, `error-detective`, or `code-logic-analyzer` agents, or `codex:rescue` for a second diagnosis pass.
6. **REFACTOR** — `/simplify` or `agent-skills:code-simplify` across the changed files; `code-refactoring-expert` agent if a file produced ≥3 fixes (hot-file rule).
7. **VERIFY (evidence before claims)** — `superpowers:verification-before-completion` + the `/verify` skill: exercise the affected flow and observe behavior, not just unit tests. Run `make test`. Never claim done without command output.
8. **REVIEW (independent, not the author)** — `superpowers:requesting-code-review` → `code-reviewer` agent + `/code-review`. For any auth/PHI/IO/crypto change, add `/security-review` + the `security-vulnerability-scanner` / `security-auditor` agent. Apply feedback via `superpowers:receiving-code-review` (verify, don't blindly accept).
9. **Track (no commits)** — mark the ticket's TodoWrite item done; flip the `REV-xx` status in `docs/BUG-BACKLOG.md` and check the box in `docs/EXECUTION-PLAN.md`. Re-run `mcp__gitnexus__impact` / `detect_changes` to confirm no unintended ripples, then re-index after a sprint.

**Sprint-level:** open the sprint with a TodoWrite list (one item per ticket + its acceptance test); use `superpowers:executing-plans` to work the sprint against this file; close with `superpowers:verification-before-completion` on the whole sprint's exit gate. Independent verification can be fanned out even solo via `superpowers:dispatching-parallel-agents` / `subagent-driven-development` (e.g., a `test-engineer` agent writing edge-case tests while you implement).

**Intentionally NOT used:** `superpowers:using-git-worktrees` and `superpowers:finishing-a-development-branch` (they imply branches/commits) — isolation is by working-tree discipline + the honest CI gate instead.

**Per-workstream skill/agent/MCP map:**
| Area | Primary skills | Agents | MCP |
|------|----------------|--------|-----|
| CI/harness (Phase 0) | `agent-skills:ci-cd-and-automation` | `test-runner`, `test-engineer` | GitNexus (untested-fn cypher) |
| Crisis/safety (A.1-A.3) | superpowers TDD + systematic-debugging | `code-logic-analyzer`, `bug-analyzer-reproducer` | Context7 (aiokafka, LangGraph) |
| Persistence/RLS (A.4, C.2) | `agent-skills:security-and-hardening` | `code-logic-analyzer` | Context7 (SQLAlchemy, Alembic, asyncpg) |
| AuthZ/PHI/audit (A.5-A.6) | `/security-review`, `agent-skills:security-and-hardening` | `security-vulnerability-scanner`, `security-auditor` | Context7 (PyJWT, cryptography) |
| Deploy/infra (B.1-B.3, C.3-C.4) | `agent-skills:ci-cd-and-automation`, observability | `deployment-engineer`, `devops-troubleshooter`, `cloud-architect`, `kubernetes-architect` | Context7 (Docker, Caddy, Prometheus) |
| Services (B.4-B.5) | superpowers TDD | `bug-analyzer-reproducer` | Context7 |
| API/OAuth/rate-limit (B.6) | `agent-skills:api-and-interface-design`, `spec` | `security-auditor` | Context7 (FastAPI, OAuth), Exa (CVE) |
| Clinical accuracy (C.5) | superpowers TDD, `agent-skills:documentation-and-adrs` | `code-logic-analyzer` | PubMed (thresholds), Context7 |

---

## Phase gate — verification & bug-finding round (run at the END of EVERY phase)

No phase is "done" until this gate passes. It is a **hard stop** before the next phase begins — a fresh set of eyes on everything the phase produced, plus an active hunt for what the phase may have broken or missed.

- **G1 · Full regression.** `make test` entire suite green under honest CI (no masks). `test-runner` agent triages any failure back into `superpowers:systematic-debugging`.
- **G2 · Bug-finding round (the hunt).** Run the repo's `bug-hunt-guide.md` pipeline scoped to the phase's changed surface: dispatch a parallel **triage swarm** (`superpowers:dispatching-parallel-agents`, or a Workflow) of `bug-analyzer-reproducer`, `code-logic-analyzer`, `security-vulnerability-scanner`, `error-detective`, and `performance-engineer` → each reports file:line + severity + repro → dedupe/rank into a mini-ledger → **every P0/P1 found gets a Prove-It failing test, then a fix** (back through the per-ticket loop) → adversarially verify P0 repros by hand. Re-run the swarm until a round surfaces nothing new.
- **G3 · Independent review.** `code-reviewer` agent + `/code-review` on the phase's cumulative change; `/security-review` + `security-auditor` if the phase touched auth/PHI/crypto/IO; `architecture-advisor` if bugs clustered across layers.
- **G4 · Dependency/CVE audit.** `dependency-compatibility-manager` + Exa CVE search on anything added/bumped; Context7 for deprecations.
- **G5 · Ripple check.** GitNexus re-index + `impact`/`detect_changes` to confirm no unintended dependents broke.
- **G6 · Perf smoke** (from B.3, once staging exists). `performance-engineer` / `web-performance-auditor` on hot paths; load-test the rate limiter and the crisis path.
- **G7 · Doc sync.** Flip `REV-` statuses in `docs/BUG-BACKLOG.md`, tick `docs/EXECUTION-PLAN.md`, update `KNOWN-LIMITATIONS.md` / `SYSTEM-REVIEW` if state changed, and record a short phase retrospective (bug classes found + fixes) to memory via `rag-memory-manager`.
- **G8 · Go/no-go.** `superpowers:verification-before-completion` on the phase exit criteria with command evidence. Only then start the next phase.

**Milestone gates** additionally **re-run the full 31-agent review** (the same workflow that produced `docs/SYSTEM-REVIEW-2026-07-21.md`) as the ultimate independent check: the **M1 gate** runs it scoped to safety + data-integrity; the **enterprise gate** runs it whole. Launch only when it returns clean on the gated scope.

---

## Milestone 0 — Harness: make CI mean something (Phase 0)

*Rationale: every downstream "verified" claim is only as good as CI. Today CI is green-but-meaningless — unit/service/integration steps run under `continue-on-error` + `|| true`, and orchestrator/user/notification/analytics/config are not tested at all (`.github/workflows/ci.yml` steps 85-101). Do this first. Skill lead: `agent-skills:ci-cd-and-automation`; GitNexus index runs here.*

### Sprint 0.1 — Un-mask CI & quarantine flakes
- [ ] Remove `continue-on-error` and `|| true` from all test steps in `.github/workflows/ci.yml`; add `orchestrator`, `user-service`, `notification`, `analytics`, `config` to the test job matrix.
- [ ] Quarantine the 5 confirmed flakes with `@pytest.mark.xfail`/skip + tracking ref: safety `test_api.py:95,239`, safety `test_entities.py:187`, therapy `test_main.py:111`, orchestrator `test_memory_node.py:161`.
- [ ] Fix `migrations/env.py:47-53` to honor `DATABASE_URL` (currently only reads hardcoded `alembic.ini:32`) — verify Alembic API via Context7.
- [ ] Run `make test` locally (`test-runner` agent), triage the true failure count among the ~4,800 currently-unenforced tests.
- **Acceptance:** CI fails on a real failure; flake quarantine list documented; baseline true-failure count known (evidence: CI run output).

### Sprint 0.2 — Green the baseline (variable: 1-3 sprints depending on 0.1 triage)
- [ ] Categorize surfaced failures with `error-detective` / `bug-analyzer-reproducer`: stale tests (fix/delete) vs real bugs (tag with `REV-`/`C-`/`H-` id; P0/P1 fixed in their workstream, else quarantine + issue).
- [ ] Un-`--ignore` safety `test_api.py`; fix its 2 crisis flakes via `superpowers:systematic-debugging`.
- **Acceptance:** CI green with an honest, documented quarantine list; safety `test_api.py` runs.
- **Exit gate (Phase 0):** a red CI reliably means a real regression. → **Run the Phase Gate ritual (G1-G8)** before Phase A.

---

## Milestone 1 — MVP: safe clinician demo (Phase A)

*Gate for M1: all P0 closed and verified at unit/integration; a mocked-infra crisis E2E passes; no cross-user data access; PHI encrypted + audit chain durable on all PHI services. Every sprint runs the per-ticket loop above.*

### Sprint A.1 — Crisis delivery, HTTP path + scoring floor (REV-03, REV-02)
- [ ] **REV-03:** reorder routes in `services/user-service/src/api.py` — register static `/users/on-call-clinicians` (:941) before `/users/{user_id}` (:812). RED: request static path → expect 200 not 422.
- [ ] **REV-02:** brainstorm the override design, then add a max-severity clamp in `services/safety_service/src/domain/crisis_detector.py` (~:721-740/:821) so any Layer-1 CRITICAL keyword floors `crisis_level` at HIGH; normalize fusion by active-signal weight. RED: lone critical keyword → HIGH+. Review: `/security-review` (safety-critical).
- **Acceptance:** clinician lookup 200; explicit high-severity utterance always escalates.

### Sprint A.2 — Crisis delivery, event plane (REV-01)
- [ ] Verify aiokafka/outbox semantics via Context7; brainstorm lifespan wiring. Instantiate `OutboxPoller(publisher)` + `await start()/stop()` in service lifespans (reuse `src/solace_events/publisher.py:333`, start/stop :348/:354). MVP priority: safety, notification, orchestrator.
- [ ] Fix safety publisher injection (`services/safety_service/src/main.py:119`); therapy `initialize_event_bridge` arg mismatch (main.py:188 vs `event_bridge.py:112`); diagnosis `EventDispatcher` injection (created main.py:153, never passed to `DiagnosisService` :128). Use `code-logic-analyzer` for the cross-module wiring.
- [ ] Narrow the lifespan `try/except` so wiring fails loud when `ENVIRONMENT != test`.
- RED: mocked-Kafka integration test — safety emits `safety.crisis.detected` → outbox → poller flush → notification consumer receives.
- **Acceptance:** crisis event travels detection → outbox → drain → consumer (mocked broker).

### Sprint A.3 — Escalation persistence + non-blocking (REV-05)
- [ ] Default `EscalationManager` to existing `PostgresEscalationRepository` (`escalation.py:562`) outside tests (currently InMemory :735); persist state transitions; run notification workflow in background with a short per-attempt timeout so `/check` returns promptly.
- RED: escalation round-trips through repo; `/check` returns within target latency.
- **Acceptance:** escalations persist; `/check` not blocked.

### Sprint A.4 — Diagnosis persistence + RLS GUC (REV-04)
- [ ] Context7-verify SQLAlchemy/asyncpg. Populate `encryption_key_id` in `save_record` INSERT (`services/diagnosis_service/src/infrastructure/postgres_repository.py:206`); reconcile `save_session` columns with migration 001; stop swallowing failures (`diagnosis service.py:451-454`).
- [ ] Add per-request middleware issuing `SET LOCAL app.current_user_id` to satisfy RLS policy (`migrations/versions/002_enable_rls_clinical_tables.py:75`) — no code sets this GUC today. `gitnexus impact` before touching the DB session layer.
- RED: diagnosis record round-trips (write → new session → read back); RLS test returns rows only with GUC set.
- **Acceptance:** clinical records persist + survive restart; RLS satisfied, not locking out.

### Sprint A.5 — AuthZ core: IDOR + revocation (REV-06, 07, 08, 09, C-01)
- [ ] Lead with `security-vulnerability-scanner` agent to enumerate every unowned handler. Enforce ownership on orchestrator history (`api.py:196-229`), diagnosis `end_session`/`challenge`, therapy homework, WS session binding.
- [ ] Route auth through the async blacklist path (`auth.py:551-562`) or add a sync Redis check so `RedisTokenBlacklist` (:321) is consulted; wire it into middleware (C-01). Blacklist JTI on logout/deletion; gate `/status` endpoints.
- RED: cross-user access → 403; revoked/logged-out token → 401 next request. Review: `/security-review` + `security-auditor` agent mandatory.
- **Acceptance:** no IDOR; revocation enforced across workers.

### Sprint A.6 — PHI encryption + audit chain (REV-12, 14, 13, H-39)
- [ ] Call `configure_phi_encryption()` in user/notification/analytics/config lifespans (reuse `base_models.py:687`; encrypt/decrypt **already** handles JSONB list/dict — no serialization work).
- [ ] Export + wire `configure_audit_logger` (`audit.py:746`) into PHI services; persist `_last_hash` in Redis/Postgres (currently in-memory :574/:668).
- [ ] Add missing `__phi_fields__` (SafetyPlan warning_signs/coping_strategies/emergency_contacts, RiskFactor.factor_description, MemoryUserProfile) + widen columns to `Text`. Flip H-39 SSL default on for prod/staging.
- RED: PHI fields ciphertext at rest; audit chain verifies across simulated restart. Review: `/security-review` + Context7 (cryptography) check.
- **Acceptance:** PHI encrypted in every PHI service; audit chain durable.

### Sprint A.7 — MVP crisis E2E + milestone gate
- [ ] Wire the crisis E2E integration test (mocked infra) using **production** repositories as the M1 gate: detection → escalation event → consumer → clinician 200 → alert. Fan out a `test-engineer` agent for edge cases.
- [ ] Fix demo-blocking runner-ups on the path: Redis login `TypeError` (`ttl=`→`ex=`, `auth.py:476-490`) and LangGraph concurrent-write crash on the crisis path (`graph_builder.py:449-460`) via `superpowers:systematic-debugging`.
- Deferred to M2: full session-state durability (REV-16) and GDPR-complete deletion (REV-17) — a single-process demo tolerates in-memory session state.
- **M1 GATE:** run the **Phase Gate ritual (G1-G8)** across all of Phase A, **plus the milestone extra** — re-run the 31-agent review scoped to safety + data-integrity. MVP go/no-go green at unit/integration: crisis delivered E2E, persistence durable, no IDOR, PHI+audit wired, CI honest, bug-hunt round clean.

*Milestone 1 estimate: ~9-11 solo weeks (Phase 0 variable), + ~1 week for the M1 gate round.*

---

## Milestone 2 — Enterprise launch (Phases B, C, D)

*Same per-ticket loop; deploy/infra sprints lead with `deployment-engineer`/`devops-troubleshooter`/`cloud-architect` agents + Context7. From B.3, verification is real E2E on staging.*

### Phase B — Deploy & core correctness

| Sprint | Focus | Key work | Acceptance |
|--------|-------|----------|------------|
| B.1 | Container builds (REV-10) | Fix Dockerfiles to COPY repo-root `src/solace_*` + `services/shared` (today own `src/` only); add missing `requirements.txt`; drop dependency-confusion pins. `deployment-engineer` agent + Context7 (Docker) | Every image builds from clean checkout |
| B.2 | Prod compose (REV-11) | Secrets from `.env.prod`; drop mailhog/kafka-ui from prod; align Weaviate 1.22.4↔4.10; fix config port 8010/8008; Prometheus on app network | Prod compose valid + secure |
| B.3 | **Staging + real CI gates** (H-47) | Stand up docker-compose test stack + staging; add integration + docker-build gating jobs; convert crisis E2E to run on real PG/Redis/Kafka/Weaviate. `/run` skill to drive it | Staging live; real E2E green |
| B.4 | Personality (REV-15) | Escape prompt braces (`trait_detector.py:169`); honor update input + preserve `assessment_history`; fix RoBERTa weight. `bug-analyzer-reproducer` | `/detect` works; profiles intact |
| B.5 | Analytics (REV-19) | Window rollup; compute compliance flags from data (stop hardcoding `True`); flatten ingested payload | Reports render real data |
| B.6 | API contract (REV-21,22,23,24) | `agent-skills:api-and-interface-design` + `spec`: reconcile `API-HANDOFF.md` or implement; verify OAuth id_token signature E2E or disable; WS reconnect replay; rate limiting + `Retry-After`. `security-auditor` + Exa (CVE) for OAuth | Contract truthful; rate limits enforced |

**→ Phase B Gate:** run the Phase Gate ritual (G1-G8), now including **G6 perf smoke** on real staging (crisis path + rate limiter load test), before Phase C.

### Phase C — Enterprise & compliance

| Sprint | Focus | Key work | Acceptance |
|--------|-------|----------|------------|
| C.1 | State durability (REV-16, 18) | Back session state with Redis/Postgres (orchestrator Postgres saver correct; therapy/diagnosis/memory stores); unify decay unit | Multi-worker safe |
| C.2 | Full RLS + GDPR (H-56, REV-17) | RLS on all PHI tables (reuse A.4 GUC middleware); deletion across PG+Redis+Weaviate + audit event, fail-loud | RLS complete; erasure verified |
| C.3 | Auth hardening | RS256/JWKS + rotation (replace HS256, `auth.py:43`); mTLS. `cloud-architect`/`kubernetes-architect` — *may need specialist* | Asymmetric JWT + mTLS |
| C.4 | Secrets & platform | Vault/cloud secret manager + rotation; multi-AZ/HA + DR runbook; Loki/Tempo + uptime; prod trace sampling. `aws-*`/observability skills — *specialist* | Enterprise infra posture |
| C.5 | Clinical accuracy (REV-25-30) | Reconcile each claim vs code (0.9/0.85 threshold, Devil's-Advocate 4/6, calibration N, RAG threshold, decay, symptom ordering) + **clinician sign-off**; PubMed MCP for thresholds | Claims match code, signed off |
| C.6 | Compliance | **BAA/DPA execution**, data residency, retention automation — *legal/calendar-bound, start early* | BAA/DPA in place before real PHI |

**→ Phase C Gate:** run the Phase Gate ritual (G1-G8) with a full security-audit pass (`security-auditor` + `/security-review` across all auth/PHI/infra changes) before Phase D.

### Phase D — Polish
- **D.1 — Runner-ups & debt (REV-31):** email-verification deadlock, silent 404 agent fallbacks, dual import roots, missing LLM timeout, un-transactional outbox lock, inconsistent env gates + lint debt. Close with `/simplify` sweep across the wave's changed files. **Acceptance:** backlog clean.
- **→ Phase D Gate:** run the Phase Gate ritual (G1-G8).

*Milestone 2 estimate: ~13-16 solo weeks; C.3/C.4/C.6 may extend on specialist/legal dependencies. Total to enterprise launch: ~6 months solo.*

---

## Enterprise launch go/no-go (final gate)
Run the **Phase Gate ritual (G1-G8)** across Phases B-D cumulatively, **plus the milestone extra — re-run the full 31-agent review whole** (not scoped). Then confirm: every image builds + deploys with real secrets; staging E2E green; personality/analytics correct; API contract truthful + rate-limited; RS256 + mTLS + managed secrets + HA/DR; full RLS; clinical claims signed off; **BAA/DPA executed**. Launch only when the whole-repo review returns clean.

---

## Verification strategy
- **Per ticket (TDD):** `superpowers:test-driven-development` RED→GREEN→REFACTOR; bug fixes use the Prove-It regression test; `superpowers:verification-before-completion` + `/verify` before any "done"; independent `code-reviewer` + `/security-review` on auth/PHI.
- **Milestone 1 (unit/integration):** `make test` full suite green; mocked-infra crisis E2E is the M1 gate. Fixtures mock PG/Redis/Kafka/Weaviate — no live infra needed.
- **Milestone 2 (real E2E from B.3):** crisis pipeline, persistence, deletion, RLS verified on staging via `/run`; `docker compose -f docker-compose.prod.yml build` succeeds for all services.
- **Continuous:** honest CI (from Phase 0) gates every run; GitNexus `impact`/`detect_changes` after each sprint confirms no ripples; re-index after each sprint.
- **No git commits:** completion is proven by tests + verification + review and recorded in TodoWrite, `docs/BUG-BACKLOG.md` (`REV-` status), and `docs/EXECUTION-PLAN.md` checkboxes.

## Critical path (sequential — every → crosses a Phase Gate)
Phase 0 (CI) → ⟦gate⟧ → A.1 (route+scoring) → A.2 (events) → A.3 (escalation) → A.4 (persistence) → A.5 (authz) → A.6 (PHI/audit) → A.7 → **⟦M1 gate + scoped review⟧** → B.1/B.2 (build/deploy) → B.3 (staging) → B.4-B.6 → ⟦gate + perf⟧ → C.1-C.6 → ⟦gate + security audit⟧ → D.1 → **⟦enterprise gate + full review⟧** → launch.

## Traceability
Per-issue detail: `REV-` appendix in `docs/BUG-BACKLOG.md`. Full narrative: `docs/SYSTEM-REVIEW-2026-07-21.md`. Workstreams: `docs/ENTERPRISE-READINESS-REMEDIATION-PLAN.md`.
