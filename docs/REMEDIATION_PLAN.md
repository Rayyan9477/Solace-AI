# Solace-AI Comprehensive Remediation Plan

**Created:** 2026-02-07
**Updated:** 2026-02-08
**Based on:** 6 code review reports totaling ~496 issues (109 critical, 170 high, 131 medium, 61 low)
**Strategy:** Incremental fixes (no rewrite), core therapy flow first

---

## Progress Summary

| Tier | Description | Tasks | Status |
|------|-------------|-------|--------|
| **Tier 0** | Immediate Crash & Data-Loss Bugs | 20/20 | **COMPLETE** |
| **Tier 1** | Authentication & Authorization | 17/17 | **COMPLETE** |
| **Tier 2** | Core Flow Persistence | 14/14 | **COMPLETE** |
| **Tier 3** | Safety Pipeline & Crisis Flow | 0/11 | Pending |
| **Tier 4** | Event Bus & Orchestrator Integration | 0/14 | Pending |
| **Tier 5** | Configuration, Testing & CI/CD | 0/13 | Pending |
| **Tier 6** | Security Hardening | 0/20 | Pending |
| **Tier 7** | ML, Features & Polish | 0/18 | Pending |

**Cleanup completed:** Deleted `archive/` (118 legacy files), `services/user_service/` (duplicate), 4 dead ML providers (deepseek, minimax, xai, gemini).

---

## Tier 0 — Immediate Crash & Data-Loss Bugs — COMPLETE

All 20 tasks completed. Resolved issues:

| Task | Description | Issues Resolved |
|------|-------------|-----------------|
| T0.1 | Fix therapy/diagnosis infinite recursion | CRITICAL-006, CRITICAL-118, IMPL-CRIT-05 |
| T0.2 | Fix memory service swapped publisher arguments | EVT-CRIT-01 |
| T0.3 | Fix Fernet keys regenerated on startup | CRITICAL-107, SEC-CRIT-05 |
| T0.4 | Replace hardcoded secrets + remove .env from repo | CRITICAL-059, CRITICAL-099, CRITICAL-067, CFG-CRIT-05, CFG-CRIT-06, HIGH-097 |
| T0.5 | Fix SQL injection in analytics LIMIT clause | CRITICAL-115, HIGH-091 |
| T0.6 | Fix port conflicts across all services | CFG-CRIT-01, CFG-HIGH-01, CFG-HIGH-02 |
| T0.7 | Fix `any` vs `Any` type bug (12 occurrences) | CRITICAL-001, INT-HIGH-08 |
| T0.8 | Fix analytics consumer never consuming | CRITICAL-117 |
| T0.9 | Fix memory node calling non-existent method | CRITICAL-062 |
| T0.10 | Fix missing relative import in analytics | CRITICAL-066, IMPL-HIGH-11 |
| T0.11 | Fix EmotionStateDTO string-vs-enum crash | CRITICAL-069, CRITICAL-070 |
| T0.12 | Fix empathy template format KeyError | CRITICAL-074 |
| T0.13 | Fix LIWC evidence TypeError (operator precedence) | CRITICAL-072 |
| T0.14 | Fix ConnectionPoolManager race condition | CRITICAL-004 |
| T0.15 | Fix missing AssessmentType import in safety | CRITICAL-007 |
| T0.16 | Fix JSON deserialization crash in orchestrator | CRITICAL-088 |
| T0.17 | Fix LangGraph conditional edges return type | CRITICAL-081 |
| T0.18 | Fix database password defaults to empty string | CRITICAL-077 |
| T0.19 | Fix memory node wrong AgentType | CRITICAL-063, MEDIUM-105 |
| T0.20 | Fix consolidation crash when pipeline is None | CRITICAL-065 |

---

## Tier 1 — Authentication & Authorization — COMPLETE

All 17 tasks completed. Resolved issues:

| Task | Description | Issues Resolved |
|------|-------------|-----------------|
| T1.1 | Unify JWT issuer/audience | SEC-CRIT-01, INT-HIGH-06 |
| T1.2 | Remove auth fallback stubs | CRITICAL-078, CRITICAL-113, SEC-HIGH-01, HIGH-095 |
| T1.3 | Add auth to 30+ unauthenticated endpoints | CRITICAL-056, CRITICAL-057, CRITICAL-058, CRITICAL-068, CRITICAL-106, CRITICAL-114, CRITICAL-119, SEC-CRIT-02, SEC-CRIT-03, HIGH-075, HIGH-076, HIGH-080, MEDIUM-054 |
| T1.4 | Replace in-memory session/token stores with Redis | CRITICAL-100, CRITICAL-104, CRITICAL-109, SEC-CRIT-04, HIGH-078, HIGH-008, HIGH-009, MEDIUM-099 |
| T1.5 | Remove admin/system role blanket bypass | CRITICAL-060, CRITICAL-101 |
| T1.6 | Fix service token permission escalation | CRITICAL-061 |
| T1.7 | Fix token refresh — validate session status | CRITICAL-108, MEDIUM-102, MEDIUM-104 |
| T1.8 | Fix wildcard CORS with credentials | CRITICAL-103, HIGH-102, HIGH-114 |
| T1.9 | Fix Kong admin API exposure | CRITICAL-102 |
| T1.10 | Fix route regex ReDoS vulnerability | CRITICAL-105 |
| T1.11 | Fix weak email validation | CRITICAL-110 |
| T1.12 | Replace string-based role checks with enum | HIGH-074, HIGH-105, HIGH-079, HIGH-122 |
| T1.13 | Fix password change must invalidate sessions | HIGH-111 |
| T1.14 | Fix clinician-patient relationship verification | HIGH-110, HIGH-077 |
| T1.15 | Fix consent type silent default | HIGH-107 |
| T1.16 | Make token blacklist mandatory | CRITICAL-008 |
| T1.17 | Fix Config service API key validation | SEC-CRIT-07 |

---

## Tier 2 — Core Flow Persistence — COMPLETE

All 14 tasks completed. Resolved issues:

| Task | Description | Issues Resolved |
|------|-------------|-----------------|
| T2.1 | Create 6 missing centralized entity modules | IMPL-CRIT-01, MEDIUM-002 |
| T2.2 | Wire LangGraph postgres checkpointer | IMPL-CRIT-07, EVT-HIGH-07 |
| T2.3 | Fix personality postgres_repository schema mismatch | CRITICAL-076 |
| T2.4 | Wire memory service to persistent storage | IMPL-CRIT-02, CRITICAL-064, EVT-HIGH-06 |
| T2.5 | Wire EncryptedFieldMixin to actual Encryptor | SEC-HIGH-05, MEDIUM-012 |
| T2.6 | Create safety entity mapper layer | INT-CRIT-03 |
| T2.7 | Fix database name mismatch | CFG-CRIT-04 |
| T2.8 | Implement PostgreSQL audit store | CRITICAL-010, CRITICAL-011 |
| T2.9 | Fix in-memory state store + wire postgres | CRITICAL-087, MEDIUM-109 |
| T2.10 | Fix in-memory repos default to postgres | IMPL-CRIT-04, HIGH-007 |
| T2.11 | Fix redundant `id` field redefinitions | HIGH-001 |
| T2.12 | Fix `add_change_record()` KeyError risk | HIGH-003 |
| T2.13 | Fix mutable default in AuditTrailMixin | CRITICAL-002 |
| T2.14 | Fix Schema Registry name collision | CRITICAL-003 |

---

## Tier 3 — Safety Pipeline & Crisis Flow — PENDING

| Task | Description | Issues |
|------|-------------|--------|
| T3.1 | Replace hardcoded notification fallback emails | CRITICAL-111, INT-CRIT-02, HIGH-088, HIGH-090, HIGH-115 |
| T3.2 | Fix SMS truncation losing safety info | CRITICAL-112, MEDIUM-062 |
| T3.3 | Wire crisis handler to supervisor routing | CRITICAL-080 |
| T3.4 | Unify crisis/risk level enums | INT-CRIT-05 |
| T3.5 | Fix safety event bridge to convert all 6 event types | INT-CRIT-04, INT-HIGH-05 |
| T3.6 | Fix safety confidence inversion | HIGH-120 |
| T3.7 | Fix safety content filter false positives | HIGH-123 |
| T3.8 | Populate crisis resources in safety agent | HIGH-119 |
| T3.9 | Fix safety LLM assessor mock fallback | IMPL-CRIT-03 |
| T3.10 | Add crisis notification deduplication | HIGH-114 |
| T3.11 | Fix safety token budget — never zero | HIGH-086, HIGH-087 |

---

## Tier 4 — Event Bus & Orchestrator Integration — PENDING

| Task | Description | Issues |
|------|-------------|--------|
| T4.1 | Create Kafka event bridges for remaining 5 services | EVT-HIGH-03 |
| T4.2 | Move event outbox from in-memory to Postgres | EVT-HIGH-02 |
| T4.3 | Fix async/sync mismatches in LangGraph agent nodes | CRITICAL-082, CRITICAL-083, CRITICAL-084, HIGH-084, HIGH-115 |
| T4.4 | Move DLQ from in-memory to Postgres | EVT-HIGH-01, HIGH-034 |
| T4.5 | Fix orchestrator EventBus sync handler invocation | EVT-CRIT-02 |
| T4.6 | Align dual divergent event schemas | EVT-CRIT-03, EVT-CRIT-04 |
| T4.7 | Remove local stubs shadowing real agent imports | CRITICAL-079 |
| T4.8 | Fix response aggregation silent failure | CRITICAL-085 |
| T4.9 | Wire ServiceAuthenticatedClient for inter-service calls | INT-CRIT-01 |
| T4.10 | Add WebSocket authentication | SEC-CRIT-03, CRITICAL-056, MEDIUM-110 |
| T4.11 | Fix safety_flags merge losing data | HIGH-116 |
| T4.12 | Fix WebSocket zombie connections | HIGH-127 |
| T4.13 | Fix service client retry — use exponential backoff | HIGH-125, HIGH-100 |
| T4.14 | Fix state serialization lossy `default=str` | HIGH-126 |

---

## Tier 5 — Configuration, Testing & CI/CD — PENDING

| Task | Description | Issues |
|------|-------------|--------|
| T5.1 | Enable CI/CD pipeline stages | CFG-CRIT-03, HIGH-096, HIGH-100 |
| T5.2 | Create integration tests for core therapy flow | IMPL-HIGH-07, IMPL-CRIT-06 |
| T5.3 | Replace mock-based tests with real behavior tests | CRITICAL-014 to 018, HIGH-015 to 027 |
| T5.4 | Prune unused dependencies | IMPL-HIGH-06, HIGH-098 |
| T5.5 | Fix percentile calculation in analytics | CRITICAL-116, MEDIUM-066 |
| T5.6 | Create per-service requirements.txt | HIGH-098 |
| T5.7 | Pin dependency versions | HIGH-099, HIGH-101, LOW-011, LOW-012 |
| T5.8 | Fix env prefix collisions | CFG-CRIT-02, CFG-HIGH-03 to 05, CFG-HIGH-13 |
| T5.9 | Fix Docker naming + add missing entries | CFG-CRIT-07, CFG-CRIT-08, MEDIUM-072, MEDIUM-073 |
| T5.10 | Fix Kafka config inconsistencies | CFG-HIGH-07, CFG-HIGH-06 |
| T5.11 | Fix Ruff ignoring hardcoded password detection | MEDIUM-075 |
| T5.12 | Fix Python version inconsistency | Verified remaining |
| T5.13 | Fix pytest-asyncio version pin | CFG-HIGH-08 |

---

## Tier 6 — Security Hardening — PENDING

| Task | Description | Issues |
|------|-------------|--------|
| T6.1 | Fix feature flag decorator async support | CRITICAL-005 |
| T6.2 | Fix Kafka ImportError fallback — fail loud | CRITICAL-021 |
| T6.3 | Change SASL credentials to SecretStr | CRITICAL-022 |
| T6.4 | Fix audit log index immutability | CRITICAL-023 |
| T6.5 | Change API keys to SecretStr in Portkey client | HIGH-028 |
| T6.6 | Fix ownership policy overly permissive matching | HIGH-011 |
| T6.7 | Implement `validate_auth_settings()` | HIGH-012 |
| T6.8 | Add SSL certificate validation | HIGH-013, HIGH-014 |
| T6.9 | Wire PHI sanitizer into logging pipeline | SEC-HIGH-03, MEDIUM-013, MEDIUM-014 |
| T6.10 | Fix email verification user-token binding | HIGH-106 |
| T6.11 | Fix event handler dispatch logic | HIGH-092 |
| T6.12 | Fix unsafe UUID parsing in analytics consumer | HIGH-093 |
| T6.13 | Add missing event category handlers | HIGH-094 |
| T6.14 | Fix Kafka consumer failure swallowed at startup | HIGH-113 |
| T6.15 | Fix Redis PubSub listener restart on error | HIGH-035 |
| T6.16 | Fix DLQ jitter calculated twice | MEDIUM-030 |
| T6.17 | Fix unknown event type silent BaseEvent | HIGH-032 |
| T6.18 | Add connection timeout to ConnectionPoolManager | HIGH-004 |
| T6.19 | Fix metrics reset race condition | HIGH-006 |
| T6.20 | Fix Weaviate anonymous access | MEDIUM-070 |

---

## Tier 7 — ML, Features & Polish — PENDING

| Task | Description | Issues |
|------|-------------|--------|
| T7.1 | Choose ML architecture (Portkey vs generic) | CRITICAL-019, CRITICAL-020, IMPL-HIGH-01, IMPL-HIGH-04 |
| T7.2 | Replace personality ML stubs with real models | CRITICAL-071, HIGH-104 |
| T7.3 | Fix LLM detector timeout enforcement | CRITICAL-073 |
| T7.4 | Wire vector search for memory retrieval | IMPL-HIGH-05 |
| T7.5 | Wire PHI sanitizer into request/response pipeline | SEC-HIGH-03 |
| T7.6 | Implement missing analytics report generators | HIGH-119 (P9-10) |
| T7.7 | Fix bare `except Exception` in all 6 ML providers | HIGH-029 |
| T7.8 | Fix ProfileStore race condition | HIGH-102 (P7-8) |
| T7.9 | Fix StructureAdjuster no-op | HIGH-122 (P7-8) |
| T7.10 | Fix embedding cache — add LRU eviction | HIGH-105 (P7-8) |
| T7.11 | Fix generator misattributes AgentType | HIGH-124 |
| T7.12 | Fix multimodal division by zero | HIGH-107 (P7-8) |
| T7.13 | Fix compassionate strategy missing affective | HIGH-108 (P7-8) |
| T7.14 | Fix orphaned ensemble weight validator | HIGH-112 |
| T7.15 | Export PostgresPersonalityRepository | HIGH-113 |
| T7.16 | Fix user_id defaults to random UUID | HIGH-109 (P7-8) |
| T7.17 | Delete duplicate `services/user_service/` | COMPLETE (cleanup) |
| T7.18 | Delete archive directory | COMPLETE (cleanup) |

---

## All Resolved Issue IDs (Tiers 0-2)

For cross-referencing with review documents:

**CRITICAL resolved (51):** CRITICAL-001, 002, 003, 004, 006, 007, 008, 010, 011, 056, 057, 058, 059, 060, 061, 062, 063, 064, 065, 066, 067, 068, 069, 070, 072, 074, 076, 077, 078, 081, 087, 088, 099, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 113, 114, 115, 117, 118, 119

**HIGH resolved (31):** HIGH-001, 003, 007, 008, 009, 074, 075, 076, 077, 078, 079, 080, 091, 095, 097, 102, 105, 107, 110, 111, 114, 122

**MEDIUM resolved (8):** MEDIUM-002, 012, 054, 099, 102, 104, 105, 109

**Cross-cutting resolved (20):** SEC-CRIT-01, 02, 03, 04, 05, 07; SEC-HIGH-01, 05; INT-CRIT-03, 05; INT-HIGH-06, 08; CFG-CRIT-01, 04, 05, 06; CFG-HIGH-01, 02; IMPL-CRIT-01, 02, 04, 05, 07; EVT-CRIT-01; EVT-HIGH-06, 07

---

## Review Documents

- [Phase 1-2 Code Review](./PHASE_1_2_CODE_REVIEW.md) — 65 issues (13 critical)
- [Phase 3-4 Code Review](./PHASE_3_4_CODE_REVIEW.md) — 80 issues (12 critical)
- [Phase 5-6 Code Review](./PHASE_5_6_CODE_REVIEW.md) — 79 issues (12 critical)
- [Phase 7-8 Code Review](./PHASE_7_8_CODE_REVIEW.md) — 98 issues (21 critical)
- [Phase 9-10 Code Review](./PHASE_9_10_CODE_REVIEW.md) — 79 issues (21 critical)
- [Cross-Cutting Review](./CROSS_CUTTING_REVIEW.md) — ~95 issues (~30 critical)
