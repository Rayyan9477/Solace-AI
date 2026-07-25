# Solace-AI Known Limitations

> **Date**: 2026-04-25 (end of Sprint 8)
> **Audience**: clinicians, frontend integrators, security reviewers

This document is the canonical list of MVP scope cuts and known gaps.
Every item here is a deliberate decision; nothing is hidden. Each
deferred item has a target window in `docs/POST-MVP-BACKLOG.md`.

---

## 1. Clinical scope

### 1.1 Not a diagnostic device

Solace-AI provides screening-grade differential diagnoses with
calibrated confidence. It is **not** a clinical decision support
system, nor an FDA-cleared device. Every recommendation must be
reviewed by a licensed clinician before any treatment decision.

### 1.2 PCL-5 deviation

The 10-item screener (max 40) replaces the standard 20-item PCL-5
(max 80). Thresholds are halved (16 / 11 / 9 vs 31 / 22 / 17). A
formal probable-PTSD determination still requires the full PCL-5
administered by a clinician. See
[CLINICAL-VALIDATION.md §1.3](CLINICAL-VALIDATION.md#13-pcl-5-ptsd--documented-deviation).

### 1.3 No published RCT

The platform has no published randomised-controlled trial. Outcomes
in this service must not be cited as evidence for AI-assisted therapy
efficacy. Planned post-MVP.

### 1.4 No FDA clearance pathway

Out of scope. Premarket notification (510(k)) requires the RCT
mentioned above plus a substantially equivalent predicate device.
Planned post-MVP.

### 1.5 Cultural adaptation

Hofstede-dimensional adaptation (individualism, power distance,
uncertainty avoidance, masculinity-femininity) was specified in the
design but not implemented. Style adaptation is currently OCEAN-only.

---

## 2. Modality and feature scope

### 2.1 No voice input

Whisper-V3-Turbo ASR, voice cloning, and voice-emotion analysis are
descoped. The README mentions them as future features only. The
multimodal fusion module (`personality_service/src/ml/multimodal.py`)
is wired but the voice modality is fed a fixed mock embedding —
documented in [CLINICAL-VALIDATION.md §7](CLINICAL-VALIDATION.md#7-personality-model).

### 2.2 No mobile / web frontend in this repo

The frontend lives in a separate repo. This backend exposes a stable
REST + WebSocket API documented in
[API-HANDOFF.md](API-HANDOFF.md).

### 2.3 Apple Sign In deferred

Google OAuth ships in MVP. Apple Sign In requires an Apple Developer
Program enrollment ($99/yr) the team chose to defer.

---

## 3. Infrastructure and operations

### 3.1 Single-VPS docker-compose deployment

Production on Hetzner CX32 (cheapest tier per user direction). No
Kubernetes. No multi-AZ failover. RTO is "redeploy from git+ env.prod"
which is on the order of minutes, not seconds. Acceptable for the
demo phase; not a production posture for many concurrent users.

### 3.2 No HashiCorp Vault

Secrets live in `.env.prod` on the VPS, owned by the deployer's
unix account. Rotation is a manual operation. Vault integration is
planned post-MVP.

### 3.3 No mTLS between services

Service-to-service auth uses bearer tokens (`solace_security.service_auth`)
over plain HTTP within the docker network. Caddy-terminated TLS
covers the public boundary. mTLS via Istio is planned post-MVP.

### 3.4 HS256 JWT (not RS256)

JWT signing uses HMAC-SHA256 with a shared secret. RS256 with
asymmetric keys + JWKS rotation is planned post-MVP.

### 3.5 Row-level security on 3 of ~15 tables

Sprint 1 enabled RLS on `diagnosis_sessions`, `therapy_sessions`,
`memory_records`. Other PHI-bearing tables (notification, audit,
safety_events, etc.) rely on application-layer access checks. Full
RLS rollout planned post-MVP.

### 3.6 In-memory token blacklist fallback

The middleware prefers `RedisTokenBlacklist` when `REDIS_URL` is
set, but falls back to `InMemoryTokenBlacklist` if Redis is
unavailable. In multi-worker deployments without Redis, token
revocation does not propagate. Configure a Redis instance for
multi-worker correctness.

---

## 4. Observability

### 4.1 Sampling at 100% (demo only)

Jaeger tracing samples every request to make the clinician demo
visually rich. Production deployment with real users should drop
sampling to 1-5% to control storage cost.

### 4.2 No Loki / Tempo log aggregation

Logs go to Docker JSON files with rotation (10 MB × 3 per service).
No centralised log search. Use `docker logs <service>` or an SSH
tail. Loki integration planned post-MVP.

### 4.3 No synthetic monitoring

No external uptime monitoring (e.g. Pingdom, Uptime Robot).
Operators rely on Caddy access logs and Prometheus alerts only.
Planned post-MVP.

---

## 5. Test coverage gaps

### 5.1 Kafka integration tests require live broker

`tests/solace_infrastructure/kafka/*` is excluded from the standard
regression run because each test bootstraps a real Kafka client
against `localhost:9092`. Run them in a docker-compose-up environment
or with a `KAFKA_BOOTSTRAP_SERVERS` override.

### 5.2 Pre-existing test flakes

Three pre-existing test failures persist from before Sprint 0,
unrelated to Sprints 1-8:

- `services/safety_service/tests/test_api.py::TestSafetyCheckEndpoint::test_safety_check_crisis_content`
- `services/safety_service/tests/test_api.py::TestResourcesEndpoint::test_get_crisis_resources`
- `services/safety_service/tests/test_entities.py::TestSafetyPlan::test_days_until_review`

These are tracked in `docs/BUG-BACKLOG.md` with `DISC-` IDs and will
be closed in a post-MVP polish sprint.

### 5.3 No E2E HTTPS test against the deployed VPS

The Sprint 8 plan includes a manual verification checklist
(`docs/MVP-RETROSPECTIVE.md`) but does not yet have an automated
suite that hits the live HTTPS URL. Planned for the deploy-day
checklist.

---

## 6. Data and privacy

### 6.1 Demo data is synthetic

The seed data generator creates synthetic personas plus clinically-
realistic vignettes derived from published case studies. No real
patient data is in the repo, the database, or the demo VPS.

### 6.2 No data residency commitment

The default VPS is in Hetzner Finland (cheapest EU). The deployment
is GDPR-compliant in posture (encryption at rest, data minimisation,
retention policies) but no formal data-processing agreement is
provided. Production users must execute one before any onboarding.

### 6.3 No HIPAA Business Associate Agreement

Solace-AI is not currently signing BAAs. US patient PHI must not be
processed via this MVP. Planned post-MVP once a covered entity has
contracted for the service.

---

## 7. Anti-patterns left in place (technical debt)

| Pattern | Where | Why deferred |
|---------|-------|--------------|
| Hyphenated project dir name `Solace-AI/` breaks `mypy --explicit-package-bases` | repo root | rename is a multi-tooling change, scoped post-MVP |
| Pre-existing `E701` ruff violations in `severity.py` | `services/diagnosis_service/src/domain/severity.py` | one-line `if x: return y` patterns, cosmetic only |
| Pre-existing `B017 pytest.raises(Exception)` | `services/safety_service/tests/test_value_objects.py` | replace with specific exception in polish sprint |
| Pre-existing `B904` raise-from in `escalation.py:154` | safety_service | minor, single line |

These are tracked in the backlog but do not block the demo.

---

## 8. Things that are NOT limitations

To prevent confusion:

- **Multi-provider LLM** is implemented (Portkey gateway, anthropic +
  openai fallback) — see Sprint 7 tests.
- **PHI encryption at rest** is wired in all 8 services and
  exercised by `tests/integration/test_phi_at_rest.py`.
- **Audit chain integrity** (HMAC-signed) is exercised by
  `tests/integration/test_audit_chain.py`.
- **Crisis detection** runs all 4 layers including Layer 1 regex even
  when ML keyword detection is active (C-12).
- **Per-hypothesis Devil's Advocate** challenges (H-07) are applied
  individually; the bug that applied a single total to all
  hypotheses is closed.

---

## 9. Corrections from the 2026-07-21 review

> Added 2026-07-21. The multi-agent review (see
> [SYSTEM-REVIEW-2026-07-21.md](SYSTEM-REVIEW-2026-07-21.md) and the
> `REV-` appendix in [BUG-BACKLOG.md](BUG-BACKLOG.md)) found that two
> §8 "NOT limitations" claims above no longer hold in code. They are
> corrected here rather than edited in place, to preserve the record.

- **PHI encryption at rest is NOT wired in all 8 services.**
  `configure_phi_encryption()` is called in only 6 services
  (diagnosis, memory, orchestrator, personality, safety, therapy). It
  is absent from user-service, notification-service,
  analytics-service, and config_service. Tracked as **REV-12**.
- **The audit chain is NOT exercised at runtime.** No service calls
  `configure_audit_logger`; `test_audit_chain.py` exercises the
  primitive in isolation, and the per-process `_last_hash` breaks the
  chain across restarts/replicas. Tracked as **REV-14**.
- **Crisis 4-layer detection has an under-escalation gap.** Layer 1
  does run alongside ML keyword detection (C-12 holds), but a lone
  CRITICAL-tier keyword can score below the escalation threshold with
  no max-severity override. Tracked as **REV-02**.

The three §8 claims about multi-provider LLM fallback and the C-12
Layer-1 regex remain accurate; only the encryption/audit/scoring
claims are corrected above.
