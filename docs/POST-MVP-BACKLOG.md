# Solace-AI Post-MVP Backlog

> **Date**: 2026-04-25
> **Source**: every Critical/High/Medium/Low item from the MVP fix
> plan that did not ship in Sprints 1-8, plus newly-discovered
> debt during the sprint cycle.
> **Use**: feeds the post-MVP planning session.

Items are grouped by horizon. Each entry references the original
issue ID so traceability to `docs/MVP-ISSUES.md` is preserved.

---

## Horizon A — required before first paying customer (~4-6 weeks)

| ID | Item | Sprint estimate |
|----|------|----:|
| H-04 (full) | Wire `PostgresEscalationRepository` into `EscalationManager` lifecycle by default; remove `InMemoryEscalationRepository` from production paths. | 2 days |
| H-39 (full) | Pass real SSL context to asyncpg pool when `enforce_database_ssl=True`; require server cert validation | 1 day |
| H-56 | Row-Level Security on the remaining 12 PHI tables (notifications, audit_logs, safety_events, safety_assessments, safety_plans, contraindication_*, treatment_plans, therapy_sessions partial, therapy_interventions, homework_assignments, personality_*, consent_records) | 3-4 days |
| C-01 (full) | Replace InMemoryTokenBlacklist fallback path with required-Redis posture in production | 1 day |
| Apple Sign In | Add Apple Developer Program + Sign-in-with-Apple parallel to Google OAuth (Sprint 8 deferred) | 2 days |
| Loki + Grafana logs | Centralised log aggregation for production debugging | 2 days |
| External uptime monitoring | Pingdom / UptimeRobot probes for Caddy + each service | 0.5 day |
| Synthetic crisis-flow probe | Hourly canary message that exercises detection + escalation + recovery | 1 day |
| BAA / DPA template | Legal artifacts so a covered entity can contract | not engineering |

---

## Horizon B — production hardening (~2-3 months)

| ID | Item |
|----|------|
| RS256 JWT | Replace HMAC HS256 with asymmetric RS256, JWKS rotation |
| mTLS | Istio service mesh with mutual TLS for service-to-service |
| HashiCorp Vault | Secret rotation + auditable issuance for all service credentials |
| Kubernetes | Migration off docker-compose for multi-AZ failover |
| RPO < 1 min | Postgres streaming replica + WAL archiving to S3 |
| RTO < 5 min | Hot standby region, automated failover |
| Voice modality | Whisper-V3 ASR + emotion analysis + multimodal fusion live (today's mock vector replaced) |
| Cultural adaptation | Hofstede-dimensional style adapter |
| Mobile + web frontend | Live integration with separate frontend repo |
| FDA premarket prep | 510(k) pathway scoping (predicate device + clinical study design) |
| Published RCT | Coordinated with academic partner |
| Post-MVP test polish | Remove pre-existing flakes (DISC-01..03), kill all `E701`/`B017`/`B904` lint warnings |

---

## Horizon C — feature backlog (~6+ months)

| Item | Notes |
|------|-------|
| Multi-language support (es, fr, de, ar, zh) | Current models are English-only. Requires per-language LLM provider routing + translated assessment instruments |
| Couples / family therapy modality | New session structure, multiple participants per session |
| Group cohort analytics for clinicians | Aggregated dashboards across a clinician's caseload |
| Long-term memory consolidation V2 | Knowledge-graph based facts beyond the current 5-tier hierarchy |
| Wearable / passive signal ingestion | Sleep, HRV, activity-level integration |
| Voice cloning / celebrity voice | Already in README — descoped from MVP, may stay descoped on ethical grounds |
| Automated personality drift detection | Long-horizon per-user trait evolution alerts |
| Self-serve clinician onboarding | Practice setup wizard, payment, BAA e-sign |

---

## Pre-existing tech debt (non-blocking)

These were left in place during the MVP sprints because they fall
outside any sprint's scope; they should be cleared in a dedicated
polish sprint before public launch:

- `Solace-AI/` (hyphenated repo name) breaks `mypy --explicit-package-bases`
- `services/diagnosis_service/src/domain/severity.py` has 26 `E701`
  one-line-if violations
- `services/safety_service/src/domain/escalation.py:154` `B904`
  `raise ... from err` missing
- `services/safety_service/tests/test_value_objects.py:38,155`
  `B017 pytest.raises(Exception)`
- `services/safety_service/tests/test_crisis_detector_fusion.py:164`
  unused local
- 3 pre-existing flakes (`test_safety_check_crisis_content`,
  `test_get_crisis_resources`, `test_days_until_review`)
- 1 ordering-dependent flake (`test_delete_user_data` passes in
  isolation, fails when run after a sibling test mutates module state)

---

## Discovered-during-sprint items (DISC-NN)

| ID | Description | Sprint discovered |
|----|-------------|------|
| DISC-01 | `requirements.txt` had unsatisfiable `langgraph~=1.0.3` + `langchain-core~=0.3.30` constraint pair. Fixed in Sprint 0 by pinning langgraph to 0.2.76. Track as pending: upgrade both to 1.x in unison post-MVP. | 0 |
| DISC-02 | `requirements.txt` was missing `PyJWT`; `auth.py` imports it directly. Fixed in Sprint 0. Verify pin is still appropriate. | 0 |
| DISC-03 | `argon2-cffi` was missing from `requirements.txt` even though `password_service.py` imports it. Installed ad-hoc in Sprint 8. Add to requirements. | 8 |
| DISC-04 | `tests/integration/test_diagnosis_vignette.py` over-stubbed step 1 and bypassed real `SafetyFlagRaisedEvent` dispatch. Worked around by asserting the result-level `safety_flags`. Track: write a tighter integration test that exercises the real step-1 dispatch path. | 3 |

---

## Definition of "done" for the post-MVP horizon-A pass

The horizon-A items above are required before opening the platform
to a paying clinician. The MVP-RETROSPECTIVE explicitly notes the
demo posture (single VPS, in-memory blacklist fallback, RLS on 3
tables) is acceptable for clinician evaluation but NOT for live
patient onboarding.
