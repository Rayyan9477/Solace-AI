# Solace-AI — Enterprise Readiness & Launch Remediation Plan

> **Date compiled**: 2026-07-21
> **Source of truth**: [SYSTEM-REVIEW-2026-07-21.md](SYSTEM-REVIEW-2026-07-21.md) (full review) + the `REV-` appendix in [BUG-BACKLOG.md](BUG-BACKLOG.md) (31 verified-open issues).
> **Purpose**: Turn the review findings into a single, planning-ready roadmap so we can schedule the fixes, close the gaps, and take Solace-AI from "clinician-demo" to **enterprise launch**.
> **How to use**: Each workstream below is independently plannable (owner + sprint). IDs (`REV-xx`, `C-xx`, `H-xx`) map back to the backlog. Do the launch gates in Phase order; within a phase, follow the dependency notes.

---

## 1. Readiness verdict

**Not launch-ready for enterprise / live-patient use.** The MVP remediation landed a large number of real fixes (see the verified-fixed tables in [BUG-BACKLOG.md](BUG-BACKLOG.md)), but the 2026-07-21 review found that several **integration-level and safety-critical** behaviors do not hold in code:

- The **crisis-safety pipeline is not deliverable end-to-end** — the event bus never publishes, and the HTTP fallback is broken by a route-ordering bug. A detected crisis may reach no clinician.
- **Clinical persistence silently no-ops** in parts of the diagnosis path, and a latent RLS migration would lock out all clinical tables if applied.
- **HIPAA-relevant controls are partially wired** — PHI encryption is active in 6 of 10 services, and the audit hash-chain is never configured at runtime.
- **Nothing deploys as written** — no service container builds, and the prod compose ships a dev secret.

None of these are unknown-unknowns anymore; they are enumerated, located, and fixable. This plan sequences them.

**Definition of "enterprise ready" used here** (the launch gate): every P0 closed and verified end-to-end; HIPAA technical safeguards (encryption at rest + in transit, audit trail, access control, RLS) complete on all PHI paths; the fleet builds and deploys reproducibly with real secrets; CI actually gates merges; and the compliance/legal prerequisites (BAA/DPA) are in place before any real PHI is processed.

---

## 2. Workstreams (grouped for ownership)

Each workstream lists its member issues, why it blocks launch, the acceptance criteria that close it, and dependencies. Effort is a rough T-shirt size for planning only.

### WS-1 — Crisis-Safety Pipeline (LAUNCH-BLOCKING, P0)
*The single highest-risk area: a mental-health product must never silently drop a crisis.*

| ID | Issue | Effort |
|----|-------|:---:|
| REV-01 | Kafka event plane filled but never drained; per-service publisher/dispatcher wiring broken | L |
| REV-02 | Crisis scoring can rate an explicit suicidal statement as ELEVATED (no max-severity override) | M |
| REV-03 | Clinician-lookup route shadowed → 422 on every on-call lookup (HTTP fallback path dead) | S |
| REV-05 | Escalation repo in-memory by default; stale single-write rows; notify timeout blocks `/check` | M |

- **Why launch-blocking**: with REV-01 + REV-03 both active, a `safety.crisis.detected` reaches neither the event consumer nor the clinician HTTP path except a hardcoded fallback email. REV-02 means some true-positive crises never trigger escalation at all.
- **Acceptance criteria**: an integration test drives a crisis utterance → escalation event published to Kafka → notification-service consumes it → clinician lookup returns 200 → alert dispatched, **using the production repositories** (Postgres escalation, real consumer), asserted end-to-end. A lone CRITICAL-tier keyword forces at least HIGH.
- **Dependencies**: REV-03 (route fix) unblocks the HTTP path immediately; REV-01 (event drain) unblocks the async path; do both before the E2E test is meaningful.

### WS-2 — Clinical Data Integrity & Persistence (LAUNCH-BLOCKING, P0)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-04 | Diagnosis persistence silently fails (missing `encryption_key_id`); RLS lockout latent; Alembic never run in deploy | L |
| REV-16 | Session/clinical state is process-local across therapy, diagnosis, safety, orchestrator, memory | L |
| REV-17 | GDPR deletion incomplete (skips Redis/Weaviate; partial-fail silent) | M |
| REV-18 | Memory decay unit mismatch can erase long-term memories in days | S |

- **Why launch-blocking**: clinical records that appear saved but are not, sessions lost on restart/scale-out, and right-to-erasure that leaves residue are all disqualifying for enterprise + GDPR.
- **Acceptance criteria**: diagnosis/therapy/memory records round-trip through the DB (write → restart process → read back identical); Alembic `upgrade head` runs in deploy and RLS is satisfied by a per-request GUC; a deletion request provably removes the subject from Postgres + Redis + Weaviate and emits an audit event; a decay test pins half-life in days.
- **Dependencies**: REV-04's RLS GUC work pairs with WS-4 (RLS rollout).

### WS-3 — AuthN / AuthZ / Session Security (LAUNCH-BLOCKING, P0/P1)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-06 | JWT revocation not enforced on the sync request path | M |
| REV-07 | IDOR across orchestrator/diagnosis/therapy history + WS session binding | M |
| REV-08 | Logout/deletion never blacklists the access token (15-min window) | S |
| REV-09 | Unauthenticated `/status` endpoints leak operational stats | S |
| C-01 remaining | Multi-worker JWT revocation (InMemory blacklist in middleware) | S |

- **Why launch-blocking**: REV-07 lets one user read/modify another's clinical data — a reportable breach. Revocation gaps (REV-06/08, C-01) mean a compromised token can't be killed.
- **Acceptance criteria**: every session/history/homework/challenge handler enforces ownership or clinician-role; a revoked/logged-out token is rejected on the next request across workers; `/status` requires auth or returns bare liveness.

### WS-4 — PHI Encryption & Audit Compliance (LAUNCH-BLOCKING, P0/P1 — HIPAA)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-12 | PHI encryption not activated in user/notification/analytics/config services | S |
| REV-13 | PHI field-coverage gaps + encrypted-value column overflow | M |
| REV-14 | Audit hash-chain never configured at runtime; breaks across restarts | M |
| H-56 | RLS on only 3 of ~15 PHI tables | L |
| H-39 | Postgres SSL enforcement off by default in prod | S |

- **Why launch-blocking**: HIPAA technical safeguards (§164.312) require encryption at rest + in transit, an audit trail, and access control on **all** PHI. Currently partial.
- **Acceptance criteria**: `configure_phi_encryption()` + `configure_audit_logger()` active in every PHI-handling service; all documented-PHI fields in `__phi_fields__` with `Text` columns; audit chain persists its last hash in shared storage and verifies across a restart; RLS (or an equivalent documented control) on every PHI table; SSL enforced to the DB in prod/staging.
- **Dependencies**: shares the RLS GUC mechanism with REV-04.

### WS-5 — Deployability & Infrastructure Hardening (LAUNCH-BLOCKING build, P1)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-10 | No service container builds (Dockerfile requirements path + missing shared packages); dependency-confusion pin | M |
| REV-11 | Prod compose: dev JWT secret, open debug ports, Weaviate server/client version mismatch, config port mismatch, Prometheus mis-wired | M |
| H-47 remaining | CI lacks integration + Docker-build jobs | M |

- **Why launch-blocking**: if the images don't build and prod ships a dev secret, there is no trustworthy deploy.
- **Acceptance criteria**: `docker compose -f docker-compose.prod.yml build` succeeds for all services from a clean checkout; prod secrets come from `.env.prod`/secret store; no debug UI or dev secret in the prod profile; Weaviate server/client versions aligned; CI builds every image and runs integration tests.

### WS-6 — Service Correctness (P1)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-15 | Personality service crashes on `/detect` and corrupts profiles on update | M |
| REV-19 | Analytics dashboards/reports read zeros; compliance report is a misleading artifact | M |

- **Why it matters**: REV-15 makes a core feature 500 under defaults and silently degrades personality data; REV-19's hardcoded `data_retention_compliant: True` is a dangerous regulatory artifact.
- **Acceptance criteria**: `/detect` returns valid scores; profile update honors input and preserves history; analytics reports render real aggregates and compute compliance flags from data.

### WS-7 — API Contract & Client Integration (P1)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-21 | Documented endpoint paths / WS auth / response envelope don't match code | S |
| REV-22 | Google OAuth helpers-only; id_token decoded without signature verification | M |
| REV-23 | WebSocket reconnect loses the conversation (fresh thread, no replay) | M |
| REV-24 | Rate limiting entirely unimplemented (Caddyfile comment only) | M |

- **Why it matters**: a frontend built from [API-HANDOFF.md](API-HANDOFF.md) fails on nearly every advanced feature; unverified id_tokens are an auth-bypass risk if OAuth is wired as-is; no rate limiting is a DoS/abuse exposure for a public enterprise endpoint.
- **Acceptance criteria**: the handoff doc matches the running contract (or the contract is implemented to match); OAuth either fully verified end-to-end or explicitly marked unavailable; reconnect replays session tail (or documents new-thread behavior); per-user/session/IP rate limits enforced with `Retry-After`.

### WS-8 — Clinical Accuracy & Model Fidelity (P2)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-25 | Safety CRITICAL threshold 0.9 vs documented 0.85; `detection_time_ms` always 0; no latency SLO | S |
| REV-26 | Devil's Advocate emits 4 of 6 declared bias types | S |
| REV-27 | Diagnosis calibration N=5 not 3; no "Uncertain" label | S |
| REV-28 | Corrective RAG returns SUCCESS with 1-2 docs (no `<3` structured error) | S |
| REV-29 | Memory "diagnoses never decay" false unless explicitly marked permanent | S |
| REV-30 | Bayesian calibration uses prior-turn symptoms, not current step-1 | M |

- **Why it matters**: these are correctness/fidelity gaps between the clinical design and the code — lower launch risk than P0/P1 but each is a claim in [CLINICAL-VALIDATION.md](CLINICAL-VALIDATION.md) that must be either made true or corrected before clinical sign-off.
- **Acceptance criteria**: each item is reconciled — code matches the validated design, or the design doc is updated to the implemented behavior with clinician review.

### WS-9 — Test & CI Trustworthiness (P1, cross-cutting)
| ID | Issue | Effort |
|----|-------|:---:|
| REV-20 | CI masks failures (`continue-on-error`/`\|\| true`); many suites never run; safety `test_api.py` ignored | M |
| DISC-01..04 | Known flakes / stubs (langgraph pin, PyJWT re-verify, root argon2, vignette over-stub) | S |

- **Why it matters**: every other workstream's "verified" claim is only as trustworthy as CI. Right now safety-critical regressions merge green.
- **Acceptance criteria**: no `continue-on-error`/`|| true` on test steps; all service + shared-lib suites in the matrix; safety `test_api.py` un-ignored and its 2 flakes fixed; a required integration job.
- **Do this early** — it is the harness that certifies the rest.

### WS-10 — Enterprise Hardening for Scale & Compliance (post-P0, launch-gating for enterprise tier)
*These are the documented deferrals in [KNOWN-LIMITATIONS.md](KNOWN-LIMITATIONS.md) that an enterprise launch (vs. demo) actually requires.*

- RS256/JWKS instead of HS256 shared secret; mTLS between services (Istio or equivalent); HashiCorp Vault (or cloud secret manager) for secret storage + rotation.
- Full RLS rollout (folds into WS-4); multi-AZ / HA deployment + DR runbook (RTO/RPO targets); centralized log aggregation (Loki/Tempo), synthetic uptime monitoring, trace sampling tuned for prod.
- Data residency + **BAA/DPA execution** before any real PHI; retention/erasure automation verified (folds into WS-2 REV-17).
- **Why it matters**: single-VPS + HS256 + `.env.prod` secrets + 100% trace sampling is a demo posture, not enterprise. These gate the enterprise tier even after all bugs are fixed.

### WS-11 — Runner-ups & Polish (P2/P3)
- REV-31 batch (Redis login `TypeError`, email-verification deadlock, silent 404 agent fallbacks, LangGraph concurrent-write crash on crisis path, dual import roots, missing LLM timeout, un-transactional outbox lock, inconsistent env gates) + accepted lint debt.
- **Note**: the LangGraph concurrent-write crash on the crisis path and the login `TypeError` are effectively P1 in impact even though grouped here — pull them forward if they touch the WS-1/WS-3 test paths.

---

## 3. Phasing & critical path

| Phase | Goal | Workstreams | Exit gate |
|-------|------|-------------|-----------|
| **Phase 0 — Harness** | Make "fixed" mean something | WS-9 | CI gates merges; integration job runs; no masked failures |
| **Phase A — Safety & Integrity (launch-blocking)** | No dropped crisis, no lost/leaked clinical data | WS-1, WS-2, WS-3, WS-4 | All P0 closed with end-to-end verification; HIPAA technical safeguards complete |
| **Phase B — Deploy & Core Correctness** | It builds, deploys, and core services work | WS-5, WS-6, WS-7 | Fleet builds + deploys from clean checkout with real secrets; personality/analytics correct; API contract truthful |
| **Phase C — Enterprise & Compliance** | Enterprise-tier posture | WS-10, WS-8 | RS256/mTLS/Vault/HA/RLS-full; clinical claims reconciled; BAA/DPA in place |
| **Phase D — Polish** | Debt down | WS-11 | Runner-ups + lint debt cleared |

**Critical path**: Phase 0 (CI) → REV-03 + REV-01 (crisis delivery) → REV-04 (persistence) → WS-4 (encryption/audit/RLS) → WS-5 (deployability) → WS-10 (enterprise/compliance). Everything else parallelizes around this spine.

---

## 4. Launch go/no-go checklist

Do not launch to enterprise / live-patient use until every box is checked and independently verified (not self-attested):

- [ ] **Crisis E2E** verified with production repos: detection → escalation event → consumer → clinician alert (WS-1).
- [ ] Crisis scoring cannot rate an explicit high-severity utterance below HIGH (REV-02).
- [ ] Clinical records provably persist and survive restart; deletion is complete across all stores (WS-2).
- [ ] No IDOR; token revocation enforced across workers; no unauthenticated PHI/stats endpoints (WS-3).
- [ ] PHI encrypted at rest in **all** PHI services + in transit to DB; audit chain active and restart-durable; RLS on all PHI tables (WS-4).
- [ ] Every service image builds; prod ships real secrets, no debug surface (WS-5).
- [ ] Personality `/detect` works; analytics reports real data; compliance flags computed not hardcoded (WS-6).
- [ ] API handoff doc matches the running contract; OAuth verified or disabled; rate limiting enforced (WS-7).
- [ ] Clinical-accuracy claims reconciled with clinician sign-off (WS-8).
- [ ] CI gates merges; safety suite runs; no masked failures (WS-9).
- [ ] RS256 + mTLS + managed secrets + HA/DR + **BAA/DPA executed** (WS-10).

---

## 5. Traceability

- **Full narrative + evidence**: [SYSTEM-REVIEW-2026-07-21.md](SYSTEM-REVIEW-2026-07-21.md)
- **Per-issue detail (Location / why / fix, file:line)**: `REV-` appendix in [BUG-BACKLOG.md](BUG-BACKLOG.md)
- **Corrected overclaims**: [KNOWN-LIMITATIONS.md §9](KNOWN-LIMITATIONS.md)
- **Not re-listed**: retracted false positives (NEW-01/02/06), verified-fixed items, and intentional MVP scope cuts — see the backlog's FIXED tables and the review's §4.

> This plan is deliberately scoped to *what the code shows today*. Re-run the review (or a targeted per-workstream verification) after each phase to keep the ground truth current before sign-off.
