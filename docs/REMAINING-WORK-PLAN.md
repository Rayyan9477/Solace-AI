# Solace-AI — Remaining Work: Compiled List + Completion Plan

Date: 2026-07-29. Source of truth for what's left to enterprise launch. Cross-refs:
`docs/EXECUTION-PLAN.md` (phases), `docs/BUG-BACKLOG.md` (REV-xx detail).

**Done to date:** M1 (MVP, GO) · Phase B code + gate (10 bugs) · B.1/B.2 Docker (authored) ·
C.2 (RLS GUC + migration 006 + GDPR erasure orchestrator) · C.3 (RS256/JWKS) · C.3b (OAuth
verify + Redis rate limits) · C.1 REV-18 decay 24× fix. Suites green: security 297, infra 802,
memory 446, therapy 500, personality 360, analytics 182, orchestrator 294.

---

> ## 🟢 2026-08-23 — CORRECTION STAMP (reconciles the Part-1-vs-Part-2 contradiction; sections below preserved as history)
>
> This doc's **Part 1 §A table still lists A1–A7 as "Doable NOW / open", which is stale** — Part 2 Stage 1
> already records A1–A7 **COMPLETE (2026-07-30)**, and Stage 2 (B1/B2) + the Phase C security waves have since
> landed too. Authoritative current state (do not trust the §A "open" framing):
>
> **COMPLETE (code + tests, adversary-verified):**
> - **§A A1–A7** — all done (Stage 1, 2026-07-30). GDPR deleters, GUC/RLS propagation, clinical-accuracy
>   reconciliation (REV-25..30), decay-rate, outbox txn, diagnosis fail-loud, D.1 debt.
> - **§B B1 + B2** — therapy + diagnosis session durability wired (Stage 2). **B3 import-root normalization →
>   Stage 3** (container-build-verified).
> - **Phase C security waves 1+2+3** — 22 of 27 findings resolved, incl. **P1-1 RS256 issuer signing implemented
>   and wired** (config-gated behind HS256). See `docs/PHASE-C-SECURITY-FINDINGS.md` + `docs/PHASE-C-REMEDIATION-PLAN.md`.
>
> **TRUE REMAINING WORK = infra + human/legal residue only (nothing else is code-doable on the dev machine):**
> - **§C Stage 3 staging** — live-build the 10 images, `up`, crisis E2E, real-PG/Redis proof of RLS/erasure/outbox/
>   checkpointer, distributed-limits, B3 import-root, + CI gates. **Plus the one code residual that needs staging to
>   verify: wire the REV-32 `assert_environment_consistency` startup guard (P0-2/P1-8) + integration test.**
> - **§C** P2-13 Postgres verify-full + pinned CA, P2-14 Redis password/TLS, P1-4 cross-user RLS proof.
> - **§D Stage 4 human/legal** — D1 Vault/HA/DR/mTLS + `.env` key rotation (P2-15); D2 clinician sign-off on
>   CLINICAL-VALIDATION §10 deltas; D3 BAA/DPA.
> - **§E gates** — E1 Phase C re-run → E2 Phase D → E3 enterprise launch gate.
>
> **The single unblocking action is unchanged: stand up staging (Stage 3).**

---

## PART 1 — COMPILED LIST OF OPEN ITEMS (grouped by what gates them)

### A. Doable NOW — code + tests, no infra/human (autonomous-safe)
> ⚠️ **SUPERSEDED 2026-08-23 — A1–A7 are all COMPLETE** (Stage 1, 2026-07-30; see the correction stamp at the top
> of this doc and Part 2 Stage 1). The table below is retained as the original scoping record, not as open work.
| # | Item | Ref |
|---|------|-----|
| A1 | **Per-service GDPR store-deleters** — wire each service's `delete_user_data` into the `UserDataErasure` registry; fill gaps: therapy (`treatment_plans`/`therapy_sessions`/`outcome_measures` delete-all-for-user), safety (repos have NO delete), memory Redis purge (not called + pattern misses keys), user-service (prod path is soft-delete only) | REV-17 |
| A2 | **C.2 service-GUC propagation** — `set_current_user_id(request.user_id)` in the service-auth'd safety + personality endpoints, then extend RLS migration 006 to their tables | C.2 |
| A3 | **C.5 clinical-accuracy reconciliation** (code vs claims, fix or correct docs, Prove-It each): thresholds/latency (REV-25), Devil's-Advocate 4/6 bias types (REV-26), calibration N + "Uncertain" label (REV-27), Corrective-RAG <3-docs structured error (REV-28), memory "diagnoses never decay" (REV-29), Bayesian prior-turn vs step-1 symptoms (REV-30) | REV-25..30 |
| A4 | **Decay-rate follow-up** — SQL `apply_batch_decay` uses a flat 0.05/day for all records vs per-category in cache (2.5× divergence); align + `accessed_at` reference | REV-18/29 |
| A5 | **Outbox double-publish** — `PostgresOutboxStore` `FOR UPDATE SKIP LOCKED` runs outside a transaction | REV-39 |
| A6 | **Diagnosis silent session drop** — `_update_session` returns on unknown session_id, dropping symptoms + `safety_flags`; at minimum fail-loud/log | REV-16 |
| A7 | **D.1 debt** — silent-404 agent fallbacks, missing LLM timeout, email-verification deadlock, remaining env-gate consistency, lint/`/simplify` sweep | REV-31 |

### B. Doable now but RISKY — needs your go-ahead (broad/async refactor)
> ⚠️ **UPDATE 2026-08-23 — B1 + B2 COMPLETE** (Stage 2); **B3 deferred to Stage 3** (container-build-verified).
| # | Item | Ref |
|---|------|-----|
| B1 | **REV-16 therapy durability wiring** — async-refactor `SessionManager` (~15 sync call sites) to write/read-through the Redis repo + CAS/version on `safety_flags` (clobber race) | REV-16 |
| B2 | **REV-16 diagnosis durability** — a session store seam (none exists today) | REV-16 |
| B3 | **REV-31 import-root normalization** — unify hyphen/underscore/dual `src` roots; removes the user-service Docker COPY-merge hack (build-destabilizing if done carelessly) | REV-31 |

### C. Infra-gated — unblocked by standing up **staging** (Docker + PG/Redis/Kafka/Weaviate)
| # | Item | Ref |
|---|------|-----|
| C1 | **B.3 staging**: live-build all 10 images (verify REV-10/11), `up` the stack, run crisis E2E on real infra, add CI docker-build + integration gates | B.3, REV-10/11 |
| C2 | **RLS + GDPR erasure PG verification** — RLS returns per-user rows only when GUC set; erasure across PG/Redis/Weaviate | C.2 |
| C3 | **Orchestrator LangGraph checkpointer fix** — construct `AsyncPostgresSaver` from an app pool + `await .setup()` in lifespan (needs live PG DDL) | REV-16 |
| C4 | **OAuth live endpoints** (`/start`/`/callback`, PKCE, link-or-create) + live JWKS fetch; **WS reconnect replay** | REV-22, REV-23 |
| C5 | **Distributed rate-limit + WS per-msg** live verify; **G6 perf smoke** (crisis path + limiter load) | REV-24, Gate G6 |

### D. Human / resource-gated (your action)
| # | Item | Ref |
|---|------|-----|
| D1 | **C.4** — Vault/managed secrets + rotation; multi-AZ HA + DR runbook; mTLS certs; SSL floor → `verify-full` + pinned root cert | C.4, REV-32 note |
| D2 | **C.5 clinician sign-off** on the reconciled clinical claims | C.5 |
| D3 | **C.6 legal** — BAA/DPA execution, data residency, retention automation | C.6 |

### E. Gates (verification milestones)
| # | Item |
|---|------|
| E1 | Phase C gate — full security-audit pass (security-auditor + /security-review across auth/PHI/infra) |
| E2 | Phase D gate |
| E3 | **Enterprise launch gate** — re-run the whole-repo 31-agent review; launch only when clean |

---

## PART 2 — COMPLETION PLAN (sequenced)

**Principle:** finish everything that needs no external resource *first* (autonomous, adversarially verified), collapse the infra bucket with one staging stand-up, and run the human/legal tracks in parallel since they're calendar-bound.

### Stage 1 — Autonomous now (Category A) — ✅ COMPLETE (2026-07-30)
**STATUS: DONE.** All Category-A items (A1–A7) executed via analyze→TDD→adversarial/self-verify→integrate+regress. Suites green: memory 470, safety 461, personality 368, diagnosis 292, orchestrator 298, solace_events 181. Migrations 007 (outbox `claimed_at`) + 008 (personality RLS) added. Docs synced: BUG-BACKLOG (Stage 1 resolution block), CLINICAL-VALIDATION §10 (7 D2 clinician-sign-off deltas). Deferred within Stage 1 to Stage 3 (need real PG/infra): safety-table RLS enable, live RLS/erasure/outbox-concurrency verification, ClickHouse mutation materialization. See memory [[2026-07-30-stage1-progress]].

Order by value/independence, each via analyze→TDD→adversarial-verify→integrate+regress:
1. **A1 per-service GDPR deleters** (completes REV-17 end-to-end at unit level; fills the therapy/safety/memory gaps).
2. **A6 diagnosis fail-loud** + **A5 outbox txn** (safety/correctness, contained).
3. **A2 service-GUC + RLS extension** (unlocks RLS coverage for safety/personality tables).
4. **A3 C.5 clinical-accuracy reconciliation** (REV-25..30) + **A4 decay-rate** — the biggest chunk; produces the claim-vs-code deltas that D2 (clinician) signs off.
5. **A7 D.1 debt** + `/simplify` sweep.
→ Exit: all Category-A green; docs synced. *Est: several autonomous workflow blocks.*

### Stage 2 — Your go-ahead on the risky refactors (Category B)
On your OK (these touch many call sites), I do **B1 therapy async+CAS wiring → B2 diagnosis store → B3 import-root normalization**, each behind its own adversarial gate. B3 also de-risks the Docker builds.

### Stage 3 — Staging stand-up (Category C) — needs you once, then I drive
**You:** start Docker / provision a box + provide `.env.prod`-style secrets for a staging env.
**Me:** C1 build+up+E2E → C2 RLS/erasure PG verify → C3 checkpointer fix (now verifiable) → C4 OAuth/WS → C5 distributed-limits + G6 perf → add the real CI gates. This converts the single largest deferred bucket into finished, verified work.

### Stage 4 — Human/legal tracks (Category D) — parallel, start early
D3 (BAA/DPA) is calendar-bound → kick off ASAP. D2 (clinician sign-off) consumes Stage-1 A3 output. D1 (Vault/HA/DR/mTLS) is you + a platform specialist.

### Stage 5 — Gates → launch (Category E)
E1 Phase C gate after Stages 1-4 auth/PHI/infra land → E2 Phase D gate after B/D.1 → **E3 enterprise launch gate** (whole-repo review); launch only when clean.

### Critical path
`Stage1 (A) ─┬─ Stage2 (B, your OK) ──┐`
`            └─ Stage4 D3/D2 (parallel)─┤`
`Stage3 (C, needs staging) ────────────┴─ E1 → E2 → E3 → launch`

**The one action that unblocks the most:** stand up staging (Stage 3). Until then I can complete all of Stage 1 (and Stage 2 on your OK) autonomously.
