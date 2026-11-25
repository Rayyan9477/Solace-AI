# Pending hand-offs

Tick rule (set 2026-11-24): `[x]` means the work is done or the condition is met, with the evidence on the line. An item that is only routed to a future task stays `[ ]` and names its owner task. Owner tasks carry the item as a "From the W0 review" or "From the W0 bug round" note in their plan text, so `extract-brief` hands it to the implementer.

## Done or met

- [x] WS-03.3 (from WS-03.2): in strict, REDIS_URL is empty unless REDIS_PASSWORD is set -> make REDIS_PASSWORD required in strict and replace both REDIS_URL lines; REDIS_PASSWORD is registered secret / format urlsafe24, so WS-03.3's registry rows must match. (Done in WS-03.3.)
- [x] WS-03.3 (from WS-03.2, controller ruling): app images are unqualified `solace/<name>:dev`; set `pull_policy: build` on every service that has a `build:`. (Done in WS-03.3; audit 2026-11-07 checked the compose lines.)
- [x] WS-01.F4 (from WS-03.2): `.gitattributes` with `*.sh text eol=lf` plus a tests/build case. (Done in WS-01.F4.)
- [x] (in LOG.md 2026-11-03) Barrier note (from WS-03.2): dev postgres_data volumes created before WS-02.1 stay unhealthy -> run `docker compose down -v` once.
- [x] (dispatched) WS-05.4 part 2: live stack self-test + XFAIL manifest after WS-03.5 and WS-03.2.
- [x] W0 barrier (done in WS-01.9 fix3, 25/25; rerun at the barrier anyway): WS-01.9 full 10-image clean-checkout proof + realinfra gate test.
- [x] (resolved by WS-02.6 review: assert_startup enforces consistency; defence-in-depth only) WS-04 follow-up (from WS-02.6): one strictness source.
- [x] (met: WS-14.2, 13.2, 18.2) Phase-B adoptions (from WS-03.2 review SPE-2/OPS-1): user-service, notification-service and analytics-service serve /live via the IF-01 runtime.
- [x] (in phaseB-common.md) Phase-B adoptions (from WS-04.F1): use the canonical lifespan form `async with service_runtime(app, SPEC) as runtime:`.
- [x] WS-14.1 (dispatched) (from WS-05.4, ruling): user-service setting USER_EMAIL_ALLOW_SPECIAL_USE_TLDS.
- [x] (dispatched) WS-05.4 part 2: overlay adds USER_EMAIL_ALLOW_SPECIAL_USE_TLDS=invalid on user-service (+ OVERLAY_ALLOWED_KEYS).
- [x] W0 barrier gate (from WS-17.1 review): WS-17.2 memory PostgresRepository on the default pool. (Met: the 2026-11-07 audit found no private engine or pool in any service src.)
- [x] WS-03.5 (from WS-11.1): strict compose sets CORS_ALLOWED_ORIGINS on orchestrator and user-service. (Met: audit 2026-11-07 checked docker-compose.prod.yml; WS-04.F2 tightened the origin rules.)
- [x] (dispatched) WS-14.3 (from WS-14.2 fix1): revoke-token catches TokenInvalidError locally in api_auth. (Met: audit checked api_auth.py:369.)
- [x] (met by WS-03.F1, commit 0efe9af, 2026-11-24) W0 review W035-SEC-1 / owner question: Caddy routed /ready and /health publicly. Ruling: never routed on api. and auth. (404); /live stays routed; external monitors use /live.
- [x] (met by WS-03.F1) Edge never-routed list matched exact paths only (review SEC-1); prefix entries and a control-character 400 landed, real-Caddy probe green.
- [x] (met by WS-03.F1) A non-strict ENVIRONMENT with the strict overlay (review SEC-2); the overlay mounts strict-environment-${ENVIRONMENT}, defined only for staging and production.
- [x] (met by WS-04.F2, commit e715074) Strict CORS accepted http and localhost origins (review SEC-3); SEC-4 non-audit half: three strict-only guards now treat staging as strict.
- [x] (met by WS-05.F1, commit a30b6e2) CI e2e job without --require-realinfra (review CR-4) and manifest entries that could not pin a cause (CR-3, with the 8 crisis entries pinned to the observed cause).
- [x] (met by WS-13.F1, commit ba7debb) Strict crisis-consumer guard not exercised through the real lifespan (review CR-1); also covers the allow path that WS-13.2 lost (audit section 1).
- [x] (met: WS-15.F1, 36faa8e) W0 bug round C-1: therapy strict accepted THERAPY_KAFKA_USE_MOCK. WS-06.8's strict fail-loud rule subsumes the guard later.
- [x] (ruled 2026-11-24, no task: config-service is not deployed in the MVP, D-19) W0 bug round A-4: config_service/src/main.py copies error['msg'] into 422 bodies and logs error=str(exc). Reopen only if D-19 lifts; risk R-23 in OPEN-ITEMS.md.

## Open, with the owning task

### W0 bug-round findings (restored 2026-11-24)

The hunter and skeptic outputs of 2026-11-03 were not saved. The ledger kept only one-line summaries (2026-11-03T14:24 and T14:28), so the descriptions below are rebuilt from those lines and from each owner task's scope. Each owner re-confirms the failure with a RED test. Each item is also a note in the owner task's plan text.

- [ ] WS-09.13 (bug round A-1, minor): validation input reaches logs through four shapes: a top-level `errors()` dict spread, `value_error` `msg` or `ctx` strings, a model repr holding error dicts, and `str(ValidationError)` of `hide_input_in_errors` models. Add all four to the RED (from WS-04.F1 fix2 review WS04F1F2-PHI-2).
- [ ] WS-09.13 (bug round A-2): platform-level scrub of asyncpg error text (IntegrityError detail, server-side DataError) in the runtime log chain, not per repository. Ruling: W3 is soon enough because only synthetic data exists before W3.
- [ ] WS-15.11 and WS-18.7 (bug round A-3, minor): therapy and analytics API handlers put exception text into error responses or logs. Fixed messages, no `str(exc)`.
- [ ] WS-12.6 (bug round B-1): `check_safety('Just a normal message')` flagged unsafe on the cloud VM (test_events::test_no_events_emitted_without_publisher). The 2026-11-07 audit could not reproduce it locally (safety 508 passed): reproduce on a clean checkout first and close with evidence if it is environment-only.
- [ ] WS-12.8 (bug round B-2): a failed escalation save is swallowed, so the API answers 201 with an id that was never stored.
- [ ] WS-12.8 (bug round B-3): `SafetyEventPublisher` drops its queue on shutdown; `KafkaEventBridge.handle` swallows an outbox save failure. Check that WS-12.9 covers the shutdown drain.
- [ ] WS-06.3 and WS-13.8 (bug round B-4): a crisis event whose handler failed must not be recorded as processed; retry in place, then durable DLQ; RISK_ALERT failures visible.
- [ ] WS-06.4 (bug round B-5): one failing or uncommittable message must not stall the partition or stop the consume loop; RED on the crisis topic with a poison message first.
- [ ] WS-10.4 (bug round B-6, highest impact, schedule first in W1): the orchestrator `SafetyServiceClient` never authenticates to safety (always 401) and the keyword fallback can lower flags; owner told to prioritise.
- [ ] WS-06.8 (bug round C-1 follow-on): the plane's strict fail-loud rule must keep the WS-15.F1 tests green.
- [ ] WS-07.7 (bug round C-2): in strict, user-service's limiter keys on the socket peer (hops 0), so all anonymous users behind Caddy share one bucket; `TRUSTED_PROXY_HOPS` is set in compose but nothing reads it (see also the WS-03.5 entry below).
- [ ] WS-12.11 (bug round C-3, minor): a safety endpoint without a service permission; enumerate routes in the RED.
- [ ] WS-09.11 and WS-04.10 (bug round C-4): the audit logger is wired nowhere and no strict check requires its key.
- [ ] WS-03.6 and WS-13.7 (bug round C-5): notification delivery settings (SMTP TLS mode, sender, fallback mailbox) are not in the env contract and strict does not check them.
- [ ] WS-11.3 (bug round C-6): the orchestrator reports ready while the graph or a required peer is unusable.
- [ ] WS-18.18 (bug round C-7): analytics background work outlives the lifespan, and the retention window is cut short.
- Parked with a ruling and no task: bug round C-8 (static `/` banners and channel booleans; no PHI or security impact).

### Owners assigned to items that had none (2026-11-24, from the 2026-11-07 audit)

- [ ] WS-08.4 (was "WS-14 owner", from WS-14.2 fix1): revoke-token answers a well-formed `Bearer <garbage>` with 503 instead of 401 (pre-existing). WS-08.4 owns the 503 mapping in `api_deps.py`.
- [ ] WS-14.5 (was "WS-14 owner", from the WS-14.1 review SPE-1): `update_preferences` existence check runs as the caller, not under `acting_as_user`; an RLS and authz defect.
- [ ] WS-06.8, with WS-06.9 (was "WS-06 follow-up", from the WS-12.1 and WS-15.1 reviews): `KafkaEventBridge.start()` leaks the producer on partial start, `stop()` is a no-op when `_started` is False, the poller is not a runtime task. The plane is the composition root, so it owns the fix.
- [ ] WS-13.8 (from lane A, CR-2): notification `/ready` says the crisis consumer runs while it polls nothing; readiness must reflect poll progress (last successful poll age) via WS-06.8's `plane.readiness()`. WS-04.7 CrisisConsumerStalled stays the backstop.
- [ ] WS-04.10 (from lane A, SEC-4 audit half): `src/solace_security/audit.py:80-81` uses a plain-hash fallback in production when `AUDIT_HMAC_KEY` is unset; make the key a required strict secret and refuse the fallback in staging too.
- [ ] WS-03.15 (from lane A): `docs/STAGING-BRINGUP.md:116` still curls `/health` on direct ports; external monitors use `/live` because `/ready` and `/health` are 404 at the edge; document the strict `ENVIRONMENT` rule and the https-only CORS rule.
- [ ] WS-05.8 (from lane A, CR-3): when WS-14.9 lands the provisioning CLI, move the 8 crisis entries out of `tests/e2e/expected_failures.yaml` (they pin `No module named scripts\.provision_clinician`).
- [ ] WS-01 lock owner (D-16, from lane A): `tests/build/test_mvp_359` fails 2 cases on new advisories GHSA-vvf4-9x63-8xvv (dnspython 2.9.0) and GHSA-mf5v-w3w3-f6cg (opentelemetry-api 1.45.1) in `requirements.lock` and `requirements-dev.lock`; each needs a bump or an accepted-advisory entry. Feed drift, not caused by W0. Do before the W0 barrier regression is called green.

### Earlier hand-offs, still open

- [ ] W4 backup task (WS-03 backup sidecar, from WS-03.2): the strict hostssl-only pg_hba has no replication rule; the backup sidecar task must add a hostssl replication rule for its role.
- [ ] WS-03.15 (from WS-03.2): staging docs are stale vs the new data plane (migrate job, solace_app, Kafka required in dev).
- [ ] CI (WS-01.8 owner, from WS-05.4): POSIX 0700/0600 e2e secrets dir may be unreadable by non-root container users on Linux CI - choose a group-readable mode or chown in the CI job.
- [ ] WS-02 follow-up (from WS-02.6): scripts/db/migrate.py builds its own URL instead of get_dsn(), so it misses the strict host/password refusals.
- [ ] MVP-273 owners (from WS-02.6): per-service config.py files still interpolate raw passwords into DSNs.
- [ ] WS-02 follow-up (from WS-02.6): get_pool DatabaseError embeds the driver error text, which SolaceError logs on construction - check for value leakage.
- [ ] WS-14.5 (from WS-14.1 review SPE-2): user_saved_postgres debug log passes email=user.email; CREATE_FAILED / user_creation_failed carry raw asyncpg text.
- [ ] W1 decision (D-26; WS-07/WS-14, from WS-11.1 fix1): RateLimitMiddleware fails open on Redis errors for every service incl. user-service login routes - decide whether auth routes fail closed.
- [ ] WS-03.15 (from WS-15.2): deploy/env/env-contract.yaml still lists THERAPY_DB_* / THERAPY_REDIS_* rows; nothing reads them - drop the rows.
- [ ] WS-18.9 (from WS-18.2 review W182-SEC-1): analytics DLQ wiring swallows get_pool failure and DLQ handler failure with no strict check - the replacement must fail closed in strict.
- [ ] WS-17.5/17.6/17.11 (from WS-17.2 real-PG probe W172-REP-1): real-infra tests must cover therapeutic_events too; session_summaries, therapeutic_events and memory_user_profiles have no RLS at head (migration 006 deferral); also the memory repository has only mocked tests, no real-Postgres test is committed (audit 2026-11-07).
- [ ] WS-03.15 (from WS-14.2): USER_DB_* and USER_REDIS_* are dead after WS-14.2 but still set in docker-compose.yml, deploy/env/env-contract.yaml and .github/workflows/ci.yml - drop them.
- [ ] WS-14.7 / WS-04.6 (from WS-14.2 review W142-SPE-1/SPE-2): user-service RateLimitMiddleware sits outside CORS and RequestContextMiddleware; _register_rate_limiting opens a second Redis client with no closer - install_rate_limiting must sit inside CORS and use runtime.redis.
- [ ] WS-07.7 (W1, from WS-03.5 security review): in strict, user-service's limiter keys on the socket peer (hops 0); TRUSTED_PROXY_HOPS is set in compose but nothing reads it.
- [ ] WS-05.8 / W1 barrier (from WS-05.6 review profile ruling): widen tests/e2e/test_crisis_alert_e2e.py to profile('dev','strict') with observed strict hops so the production-like profile gates the crisis path.
