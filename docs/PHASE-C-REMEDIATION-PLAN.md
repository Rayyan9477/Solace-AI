# Phase C Security Remediation + Remaining-Phases Plan

Date: 2026-07-30. Source of truth for closing the launch-blocking security backlog and
finishing the remaining phases. Findings detail: `docs/PHASE-C-SECURITY-FINDINGS.md`
(27 confirmed: 2 P0, 9 P1, 16 P2 — 4 fixed, 23 open).

---

> ## 🟢 2026-08-23 — WAVES COMPLETE (correction stamp; plan below preserved as history)
>
> **Wave 1 — DONE** (six file-disjoint clusters, adversary-verified): P1-2 erase-GUC (therapy/diagnosis/memory),
> P1-5 `_load_session` reconcile, P1-6 outbox FAILED→DLQ, P2-6 GUC on both pool backends, P2-7 personality GUC
> reset, P2-9 crisis-persist fail-loud, P2-10 Redis-backed limiter, P1-9 `_ENV` naming, P2-12 trusted-proxy-hops,
> P2-16 Weaviate key guard, P2-15 `.dockerignore`, P0-2/P1-8 env-gate (partial — see residual below).
>
> **Wave 2 — DONE** (auth, sequential, own gate): **P1-1 RS256 issuer migration** (user-service now signs on the
> shared KeyManager RS256 path, claim schema reconciled, config-gated behind HS256 dual-accept), P2-2 refresh
> rotation + reuse detection, P2-3 service-token async revocation, P1-3 user-service GUC, P1-7 user-service
> rate-limit.
>
> **Wave 3 — DONE** (code residuals, five clusters, verified this session): P2-8 consumer `event_id` idempotency
> (events-idempotency PASS), safety `is_critical`/PCL-5 clinical-drift reconcile (PASS), P2-4/P2-5/P2-11 auth
> timing + health-probe exemption (auth-timing-health PASS), P2-6/P2-12/REV-32-mechanism (rls-env-config — PASS
> with one ISSUE, below), personality config test-hygiene (PASS).
>
> **NET: 22 of 27 findings RESOLVED + verified. 5 remain, all infra- or human/legal-gated:**
> - **P0-2 / P1-8 REV-32 env-gate — launch-blocking follow-up (code residual):** the
>   `assert_environment_consistency` guard exists + is unit-tested but is **never invoked at startup (dead code)**.
>   Wiring a shared startup hook + one integration test is the residual; the compose/`.env.prod` half is Stage 3.
> - **P1-4 cross-user clinician/admin RLS** — design chosen; real-Postgres proof at Stage 3.
> - **P2-13 Postgres verify-full + pinned CA** — Stage 3 / C.4 infra.
> - **P2-14 Redis password + TLS** — Stage 3 / C.4 infra.
> - **P2-15 `.env` key rotation** — human action (D.1); `.dockerignore` already landed.
>
> **NEXT = Stage 3 (staging: live builds + real-PG/Redis RLS/erasure/SSL/TLS/outbox proof + REV-32 startup-wire) +
> Stage 4 (legal/human: D2 clinician sign-off, D3 BAA/DPA, D1 Vault/HA/DR/mTLS) → Stage 5 gates → launch.**
> The "P1-1 RS256 is the biggest open blocker" framing is now OBSOLETE — RS256 issuer signing is implemented and
> wired. The real remaining blocker is live-infra verification (staging) plus the legal gates.

## Principle
Finish everything code-verifiable now (TDD + adversarial verify), in dependency order,
grouped by file-ownership so agents run in parallel without collisions. Defer only what
genuinely needs real infra (TLS floors, mTLS, managed secrets, real-Postgres RLS proof)
or a human/legal action.

---

## WAVE 1 — code-doable, parallel agent team (runs now)
Six file-disjoint clusters, each: analyze → TDD (RED→GREEN) → adversarial verify → integrate.

| # | Cluster (files) | Findings |
|---|-----------------|----------|
| 1 | **therapy** (`therapy_service/src/domain/service.py`) | P1-2 erase-GUC scoping (mirror diagnosis); P1-5 `_load_session` durable-version reconcile-on-hit; P2-9 crisis-persist CAS-exhaustion fail-loud + unconditional crisis event |
| 2 | **diagnosis** (`diagnosis_service/src/domain/service.py`) | P1-5 `_load_session` reconcile-on-hit; P2-9 crisis-persist fail-loud |
| 3 | **memory** (`memory_service/src/domain/service.py`) | P1-2 erase-GUC scoping (mirror diagnosis) |
| 4 | **outbox** (`src/solace_events/{publisher.py,postgres_stores.py}`) | P1-6 FAILED → DLQ/reaper + backoff + raised max_retries; P2-8 conditional `mark_published` (WHERE status='PUBLISHING') + consumer `event_id` idempotency seam |
| 5 | **shared-infra** (`src/solace_infrastructure/postgres.py`, `infrastructure/api_gateway/rate_limiting.py`, `personality_service/src/api.py`) | P2-6 set the RLS GUC on BOTH pool backends (decouple from `use_connection_pool_manager`); P2-10 Redis-back `create_solace_rate_limiter` + fail-loud in prod; P2-7 personality GUC set/reset in `finally` |
| 6 | **config/secrets** (`docker-compose.yml`, `docker-compose.prod.yml`, `.env`, `.dockerignore`) | P1-8 `ENVIRONMENT: ${ENVIRONMENT:?}` on a prod x-app anchor + AUDIT_HMAC_KEY guard; P1-9 fix `_ENV`→`_ENVIRONMENT` naming for memory/notification/analytics + per-service bind test; P2-12 `TRUSTED_PROXY_HOPS=1`; P2-15 `.dockerignore` `.env` + rotation note; P2-16 `${WEAVIATE_API_KEY:?}` guard |

**Not in Wave 1 (why):** the auth cluster (P1-1 RS256 issuer, P2-2 refresh rotation, P2-3
service-token revocation, P2-5/P2-11 timing, P1-3 user-service GUC, P1-7 user-service
rate-limit) ALL touch user-service `api.py`/`jwt_service.py` + `solace_security` — they
collide with each other and are the highest-risk change, so they get their own careful
sequential wave.

## WAVE 2 — auth hardening (careful, sequential, own adversarial gate)
Owns `user-service/src/{api.py,infrastructure/jwt_service.py,domain/service.py}` +
`src/solace_security/{auth.py,middleware.py}`. Order:
1. **P1-1 RS256 issuer migration** (the headline P1): make user-service `JWTService` sign
   user tokens with the shared `KeyManager` RS256 private key (held only by user-service);
   reconcile the claim schema so issuer + validators agree (`sub`/`roles`); flip
   validators to `accept_legacy_hs256=False` once issuance is migrated (keep a dual-accept
   window behind a flag for rollout). Service tokens stay HS256 until mTLS.
2. **P2-2 refresh-token rotation + reuse detection** (blacklist consumed JTI; adopt
   `JWTManager.refresh_token_pair`; family-revoke on reuse).
3. **P2-3 service-token revocation** via the async Redis-aware path in `get_current_service`.
4. **P1-3 user-service GUC** set/reset in the local `get_current_user`; **P1-7** wire
   `RateLimitMiddleware` (fail-loud in prod); **P2-4** dedicated resend policy (IP+email);
   **P2-5/P2-11** move email send off the request path + constant-time login verify.

## DESIGN DECISIONS (recommended; adjust before Wave 2 if you disagree)
- **P1-4 cross-user RLS** (clinician/admin reading another user's RLS rows): the caller-scoped
  GUC only supports self-access. **Recommendation:** for authorized cross-user operations,
  scope `app.current_user_id` to the *data-owner* (`request.user_id`) for the DB work
  (same pattern as the erase fix), guarded by the existing authz check; add a dedicated
  **BYPASSRLS maintenance role** only for erasure/admin batch jobs. Full endpoint sweep +
  real-Postgres proof is Stage 3. (Wave 1 fixes the erase case; the read case is Stage-3-verified.)
- **RS256 rollout:** dual-accept (RS256 + legacy HS256) behind `accept_legacy_hs256` for one
  deploy, then flip strict — so a rolling deploy never rejects in-flight valid tokens.

## INFRA / HUMAN-GATED (Stage 3 / C.4 / legal — NOT code-fixable now)
- P2-13 Postgres SSL `verify-full` + pinned CA; P2-14 Redis password + `rediss://` TLS
  (config + real infra to verify). mTLS (service-to-service). Managed secrets (Vault / AWS
  Secrets Manager / config_service provider) — replaces `.env`. Real-Postgres RLS
  enforcement + cross-user proof (P1-4). Live image builds + **B3 import-root normalization**
  (container-build-verified) + crisis E2E + outbox live-concurrency + LangGraph checkpointer.

## REMAINING PHASES (after security remediation)
- **Stage 3 (infra):** stand up staging (local Docker per `docs/STAGING-BRINGUP.md` or AWS
  IaC) → live builds + B3 + E2E + PG verification of RLS/erasure/outbox/checkpointer + perf + CI gates.
- **Stage 4 (human/legal, parallel):** D2 clinician sign-off on `CLINICAL-VALIDATION §10`
  deltas; D3 BAA/DPA; D1 Vault/HA/DR/mTLS.
- **Stage 5 (gates → launch):** re-run the Phase C security gate clean → Phase D gate →
  enterprise launch gate (whole-repo review). Launch only when clean.

## Critical path
`Wave 1 (parallel) → Wave 2 auth (RS256 first) → Phase C gate re-run` ‖
`Stage 3 staging (RLS/erasure/SSL/TLS/mTLS proof) ‖ Stage 4 D2/D3 (calendar-bound)` → Stage 5 → launch.
