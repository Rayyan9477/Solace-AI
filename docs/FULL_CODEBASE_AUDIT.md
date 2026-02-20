# Full Codebase Audit Report

**Date:** February 19, 2026  
**Scope:** Line-by-line review of every `.py` file across `src/`, `services/`, `infrastructure/`, `tests/`, and root config files  
**Total Issues Found:** ~150 across all severity levels

---

## Executive Summary

| Severity | Count | Key Themes |
|----------|-------|------------|
| **CRITICAL** | 16 | Safety pipeline doesn't filter, aggregator picks random response, GDPR deletion failures, blocking sync calls in async code, install-breaking dependency, non-functional rate limiting, hand-rolled JWT |
| **HIGH** | 44 | Unauthenticated endpoints, missing authorization checks, credential exposure, dead code masquerading as functional, race conditions, structlog crash at startup |
| **MEDIUM** | 58 | Duplicate code/settings, swallowed exceptions, in-memory state incompatible with scaling, keyword matching without word boundaries, type mismatches |
| **LOW** | ~32 | Dead code, minor type issues, style, test quality |

---

## CRITICAL Issues (Must Fix)

### C1. Aggregator Picks Random Response — Core Pipeline Broken
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L176-L180)

The `aggregator_node` picks `responses[-1]` — the last response in the list. With parallel fan-out, agent ordering is non-deterministic, so the response shown to the user is **essentially random**. The sophisticated `Aggregator` class in [aggregator.py](services/orchestrator_service/src/langgraph/aggregator.py) (with `ResponseRanker`, `ResponseMerger`, priority strategies, weighted merging) is **dead code** — never used by the graph.

```python
responses = [r["response_content"] for r in agent_results if r.get("response_content")]
final_response = responses[-1] if responses else "I'm here to support you..."
```

### C2. Safety Post-Check Detects But Never Filters Harmful Content
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L188-L195)

`safety_postcheck_node` detects harmful patterns (`"you should"`, `"just do it"`, `"give up"`, `"no point"`) but only sets a metadata flag. **The `final_response` is never modified** — harmful content passes through unchecked to the user. For a mental health platform, this is the most critical safety gap.

### C3. Supervisor Never Gets LLM Client — Intent Refinement Silently Disabled
**Files:** [graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L316), [supervisor.py](services/orchestrator_service/src/langgraph/supervisor.py#L37-L41)

The graph builder constructs `SupervisorAgent(self._supervisor_settings)` without passing `llm_client`. `llm_client` defaults to `None`, so `refine_with_llm()` always early-returns. Low-confidence keyword classifications are never improved — the supervisor is partially non-functional.

### C4. GDPR Deletion Doesn't Delete Weaviate Vector Data
**File:** [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L424-L445)

`delete_user_data()` deletes from Postgres and in-memory caches but **never calls** `self._weaviate_repo.delete_user_data(user_id)`. Any PHI stored as vectors remains in Weaviate — a regulatory violation.

### C5. GDPR Deletion Continues After Postgres Failure
**File:** [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L427-L432)

If the Postgres delete raises an exception, the method catches it, logs a warning, and **continues to clear in-memory caches**. Persistent data remains in Postgres but caches are wiped — partial deletion with inconsistent state.

### C6. Non-Atomic GDPR Deletion in Diagnosis Service
**File:** [services/diagnosis_service/src/infrastructure/postgres_repository.py](services/diagnosis_service/src/infrastructure/postgres_repository.py#L247-L260)

Sessions and records are deleted in **separate database connections/transactions**. If the second `DELETE` fails, sessions are erased but diagnosis records containing PHI remain.

### C7. Non-Atomic GDPR Deletion in Personality Service
**File:** [services/personality_service/src/infrastructure/postgres_repository.py](services/personality_service/src/infrastructure/postgres_repository.py#L350-L400)

Profiles, assessments, and snapshots deleted using **3 separate connections** — not wrapped in a single transaction.

### C8. Weaviate Client Blocks Async Event Loop
**File:** [services/memory_service/src/infrastructure/weaviate_repo.py](services/memory_service/src/infrastructure/weaviate_repo.py#L99-L103)

All Weaviate v4 Python client calls are **synchronous** but called inside `async def` methods without `asyncio.to_thread()`. Blocks the entire FastAPI event loop for every vector operation.

### C9. Vault/AWS Secrets Providers Block Event Loop
**File:** [services/config_service/src/secrets.py](services/config_service/src/secrets.py#L138-L250)

`VaultProvider` uses sync `hvac.Client` calls, `AWSSecretsManagerProvider` uses sync `boto3.client` calls — both inside `async def` methods. Under load, this freezes the entire asyncio event loop.

### C10. Redis Rate Limiting Is Completely Non-Functional
**File:** [infrastructure/api_gateway/rate_limiting.py](infrastructure/api_gateway/rate_limiting.py#L226-L243)

`RedisRateLimitStore.increment()` (sync) is a stub that **always returns `allowed=True`** with full remaining limit. Actual logic is only in `async_increment()` which is never called. When Redis is configured, **all requests bypass rate limiting**.

### C11. Hand-Rolled JWT Implementation in API Gateway
**File:** [infrastructure/api_gateway/auth_plugin.py](infrastructure/api_gateway/auth_plugin.py#L138-L168)

Custom HMAC-based JWT encode/verify instead of using battle-tested libraries. Custom crypto is the #1 cause of authentication bypasses. The project already depends on `python-jose` per requirements.txt.

### C12. In-Memory Token Revocation Lost on Restart
**File:** [infrastructure/api_gateway/auth_plugin.py](infrastructure/api_gateway/auth_plugin.py#L149)

`_revoked_tokens: set[str]`  — all revoked tokens are lost on process restart. In multi-instance deployments, revoking a token on one instance has no effect on others.

### C13. `sqlalchemy~=2.1.0` Doesn't Exist — Blocks `pip install`
**File:** [requirements.txt](requirements.txt#L32)

SQLAlchemy 2.1.0 has never been released (latest is 2.0.x). `pip install -r requirements.txt` fails for every developer and CI pipeline. Should be `sqlalchemy[asyncio]~=2.0.0`.

### C14. `AttributeError` in Memory Service Semantic Filter
**File:** [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L527-L530)

Code references `record.id` and `r.id`, but `MemoryRecord` uses `record_id`. Raises `AttributeError` at runtime whenever the in-memory hybrid search path is taken.

### C15. `AttributeError` in Therapy Service — Missing Method Call
**File:** [services/therapy_service/src/domain/service.py](services/therapy_service/src/domain/service.py#L450-L455)

`get_user_progress()` calls `self._session_manager.get_user_sessions(user_id)` — this method **does not exist** on `SessionManager`. Will crash at runtime.

### C16. Lost `missing_info` Breaks Diagnostic Question Generation
**File:** [services/diagnosis_service/src/domain/service.py](services/diagnosis_service/src/domain/service.py#L128-L133)

Step 4 reads `missing_info` from step 3's result, but step 3 **never returns a `missing_info` key**. The missing info from step 2 is silently discarded. The next-question generator can never use missing-info-based questions, falling back to generic prompts.

---

## HIGH Issues

### Security

| # | File | Issue |
|---|------|-------|
| H1 | [src/solace_security/auth.py](src/solace_security/auth.py#L455-L495) | `decode_token_sync()` skips revocation check. Any sync caller accepts revoked tokens. |
| H2 | [services/orchestrator_service/src/api.py](services/orchestrator_service/src/api.py#L230-L238) | WebSocket endpoint accepts `user_id` as query parameter — no authentication. Any client can connect as any user. |
| H3 | [services/orchestrator_service/src/api.py](services/orchestrator_service/src/api.py#L182-L217) | Session history endpoint doesn't verify the authenticated user owns the session. IDOR vulnerability. |
| H4 | [services/diagnosis_service/src/api.py](services/diagnosis_service/src/api.py#L224-L240) | `/challenge/{session_id}` — no ownership check. Any authenticated user can challenge any session. |
| H5 | [services/diagnosis_service/src/api.py](services/diagnosis_service/src/api.py#L215-L220) | Null `user_id` in session state bypasses authorization entirely. |
| H6 | [infrastructure/api_gateway/cors.py](infrastructure/api_gateway/cors.py#L66-L67) | Default `origins=["*"]` + `credentials=True`. Reflects any Origin header with credentials — CORS bypass. |
| H7 | [infrastructure/api_gateway/cors.py](infrastructure/api_gateway/cors.py#L83-L87) | `re.match()` without end-of-string anchor. `https://example.com` matches `https://example.com.evil.com`. |
| H8 | [infrastructure/api_gateway/auth_plugin.py](infrastructure/api_gateway/auth_plugin.py#L155-L156) | Deterministic `jti` generation via SHA-256 of `subject:timestamp:type`. Same-second tokens get identical JTIs. |
| H9 | [infrastructure/api_gateway/routes.py](infrastructure/api_gateway/routes.py#L32) | `admin_url` defaults to `http://` — sends admin tokens over unencrypted HTTP. |
| H10 | [services/config_service/src/settings.py](services/config_service/src/settings.py#L104-L128) | Hardcoded default passwords: `"changeme"`, `"change-in-production"`. No startup validation. |
| H11 | [services/config_service/src/feature_flags.py](services/config_service/src/feature_flags.py#L79-L80) | `MATCHES_REGEX` operator runs unvalidated user-supplied regex — ReDoS vulnerability. |
| H12 | All services | `DatabaseSettings.password` is `str` not `SecretStr` — passwords leak in logs/repr/tracebacks. Affects therapy, safety, personality, and memory services. |

### Architectural / Dead Code

| # | File | Issue |
|---|------|-------|
| H13 | [services/orchestrator_service/src/response/](services/orchestrator_service/src/response/) | **Entire `response/` module is dead code.** `generator.py`, `style_applicator.py`, `safety_wrapper.py` are never wired into the graph. All empathy enhancement, style adaptation, and safety resource injection is unreachable. |
| H14 | [services/orchestrator_service/src/websocket.py](services/orchestrator_service/src/websocket.py) | Full-featured `ConnectionManager` with heartbeat, auth, cleanup — never used. API has its own inline WebSocket handler. Two diverging implementations. |
| H15 | [services/orchestrator_service/src/infrastructure/clients.py](services/orchestrator_service/src/infrastructure/clients.py) | Production-grade HTTP clients with circuit breakers — never used by agents. Each agent has its own simpler HTTP client without circuit breakers or auth. |
| H16 | [services/config_service/src/api.py](services/config_service/src/api.py#L141-L151) | `HTTPException(404)` swallowed by `except Exception` → re-raised as 500. |
| H17 | [services/config_service/src/api.py](services/config_service/src/api.py#L109-L111) | `SecretsManager` re-created per request — cache, audit buffer, provider state lost. |
| H18 | [services/config_service/src/api.py](services/config_service/src/api.py#L119-L136) | Route path conflict: `/config/{key:path}` shadows `/config/section/{section}`. |

### Bugs

| # | File | Issue |
|---|------|-------|
| H19 | [services/diagnosis_service/src/domain/service.py](services/diagnosis_service/src/domain/service.py#L143-L145) | Symptom-evidence matching uses `str()` coercion + substring. `"sad"` matches inside `"Detected from: I've been feeling sad..."`. False negatives/positives. |
| H20 | [services/diagnosis_service/src/main.py](services/diagnosis_service/src/main.py#L76-L79) | `structlog.stdlib.INFO` doesn't exist — `AttributeError` at startup. Same bug in memory_service and config_service `main.py`. |
| H21 | [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L180) | `metadata` field has no reducer annotation. Parallel agents' metadata updates use last-writer-wins — all but one agent's metadata is silently discarded. |
| H22 | [services/orchestrator_service/src/api.py](services/orchestrator_service/src/api.py#L204-L210) | `checkpointer.get()` is sync — will block event loop or fail with async Postgres checkpointer. |
| H23 | [services/memory_service/src/infrastructure/redis_cache.py](services/memory_service/src/infrastructure/redis_cache.py#L285-L294) | `increment_session_message_count` is a non-atomic read-modify-write. Concurrent messages lose increments. |
| H24 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L240-L245) | Ending one session wipes working memory for ALL sessions of that user. |
| H25 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L216) | Starting a new session unconditionally clears working memory, destroying data from active sessions. |
| H26 | [services/memory_service/src/infrastructure/postgres_repo.py](services/memory_service/src/infrastructure/postgres_repo.py#L186-L198) | Read + access-tracking share one transaction. Update failure causes read to fail too. |
| H27 | [services/memory_service/src/infrastructure/postgres_repo.py](services/memory_service/src/infrastructure/postgres_repo.py#L242-L246) | `**summary_data` can contain `summary_id` → `TypeError: got multiple values for argument`. |
| H28 | [services/memory_service/src/domain/context_assembler.py](services/memory_service/src/domain/context_assembler.py#L203) | `datetime.min` (naive) mixed with `datetime.now(timezone.utc)` (aware) → `TypeError` during sort. |
| H29 | [services/shared/infrastructure/llm_client.py](services/shared/infrastructure/llm_client.py#L108-L115) | When `portkey_ai` not installed: `_client = None` but `_initialized = True`. All `generate()` calls silently return `""`. |
| H30 | [services/config_service/src/main.py](services/config_service/src/main.py#L103-L106) | Readiness returns `JSONResponse` where type annotation says `dict[str, str]` → 500 error on unhealthy path. |

### Concurrency

| # | File | Issue |
|---|------|-------|
| H31 | [src/solace_infrastructure/observability_core.py](src/solace_infrastructure/observability_core.py#L171-L192) | `MetricsRegistry` has a lock but `counter()`/`gauge()`/`histogram()` **never acquire it**. Dict mutations from concurrent threads can corrupt data. |
| H32 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L50-L58) | All tier dicts, `_active_sessions`, `_user_session_counts` are plain dicts with no locking. `await` points between reads and writes create real race windows. |
| H33 | [services/diagnosis_service/src/domain/service.py](services/diagnosis_service/src/domain/service.py#L55-L59) | Same pattern — in-memory session state without any concurrency protection. |

### Safety-Critical Domain Bugs

| # | File | Issue |
|---|------|-------|
| H34 | [services/safety_service/src/domain/crisis_detector.py](services/safety_service/src/domain/crisis_detector.py#L80-L120) | Crisis keyword detection uses `if kw in content` — no word boundary. `"therapist"` matches `"rapist"`, `"skill"` matches `"kill"`. |
| H35 | [services/therapy_service/src/domain/service.py](services/therapy_service/src/domain/service.py#L250-L275) | Same substring matching for crisis keywords. `"pharmacy"` matches `"harm"`. |
| H36 | [services/personality_service/src/domain/service.py](services/personality_service/src/domain/service.py#L50-L80) | Two different `PersonalityProfile` classes with different types. Code importing from different locations gets incompatible objects. |

---

## MEDIUM Issues

### Type Errors & Data Integrity

| # | File | Issue |
|---|------|-------|
| M1 | 6 entity files under [src/solace_infrastructure/database/entities/](src/solace_infrastructure/database/entities/) | **30+ JSONB columns** annotated as `Mapped[dict]` but storing lists (`default=list`). Type checkers report wrong types. |
| M2 | [services/personality_service/src/domain/service.py](services/personality_service/src/domain/service.py#L350-L360) | Confidence weights sum to **1.1** (0.8 + 0.3), systematically inflating over time. |
| M3 | [services/diagnosis_service/src/domain/severity.py](services/diagnosis_service/src/domain/severity.py#L223) | Python banker's rounding (`round()`) in PHQ-9/GAD-7 imputation. `round(2.5) = 2`, not 3. |
| M4 | [services/diagnosis_service/src/domain/differential.py](services/diagnosis_service/src/domain/differential.py#L408-L414) | Adjustment disorder gets inflated confidence (empty `required_symptoms` → 1.0 ratio automatically). |
| M5 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L493-L496) | `_get_tier_records` excludes `tier_1_input` — retrieval from tier 1 always returns `[]`. |

### Caching & Resource Leaks

| # | File | Issue |
|---|------|-------|
| M6 | [src/solace_security/auth.py](src/solace_security/auth.py#L206-L222) | `InMemoryTokenBlacklist` never evicts expired entries — unbounded memory growth. |
| M7 | [src/solace_security/auth.py](src/solace_security/auth.py) `InMemoryLoginAttemptTracker` | Per-user attempt records never cleaned up — DoS vector via distinct user IDs. |
| M8 | [services/personality_service/src/ml/roberta_model.py](services/personality_service/src/ml/roberta_model.py#L185-L195) | Cache key is `processed_text[:256]` — texts sharing 256-char prefix get same cached embedding. Silent collision. |
| M9 | [services/personality_service/src/ml/llm_detector.py](services/personality_service/src/ml/llm_detector.py#L155-L165) | Same 256-char cache key truncation causing collisions. |
| M10 | [services/config_service/src/secrets.py](services/config_service/src/secrets.py#L409) | `_audit_buffer` grows indefinitely — never flushed, persisted, or size-limited. Memory leak. |
| M11 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py#L50-L58) | All tier caches grow without limit — no LRU eviction, no max-size. OOM under sustained load. |

### ML-Specific Issues

| # | File | Issue |
|---|------|-------|
| M12 | [services/personality_service/src/ml/roberta_model.py](services/personality_service/src/ml/roberta_model.py#L205-L220) | `_run_model()` is `async def` but calls tokenizer/model **synchronously** — blocks event loop. Need `asyncio.to_thread()`. |
| M13 | [services/personality_service/src/ml/roberta_model.py](services/personality_service/src/ml/roberta_model.py#L170-L195) | `detect_batch()` processes texts **sequentially** despite the "batch" name — no actual batching. |
| M14 | [services/safety_service/src/ml/llm_assessor.py](services/safety_service/src/ml/llm_assessor.py#L350-L400) | Rule-based fallback matches against **entire instruction template**, not just user content. `"harm"` in instructions triggers false positive. |
| M15 | [services/safety_service/src/ml/sentiment_analyzer.py](services/safety_service/src/ml/sentiment_analyzer.py#L150-L170) | VADER fallback only loaded when CUDA unavailable. GPU servers lose the fallback safety net if transformer fails. |
| M16 | [services/personality_service/src/ml/multimodal.py](services/personality_service/src/ml/multimodal.py#L145-L160) | Below-threshold results can enter final fusion via the fallback path. |

### Architectural Duplication

| # | Scope | Issue |
|---|-------|-------|
| M17 | All services | **Duplicate settings classes** — every service has 2-3 competing settings classes in main.py, config.py, and service.py. |
| M18 | All services | **EventBus swallows handler exceptions** — logged but not retried, queued, or propagated. At-most-once, no-guarantee delivery. |
| M19 | All services | **Hardcoded absolute imports** `from src.solace_events.schemas import ...` — brittle, breaks if restructured. |
| M20 | [safety_service](services/safety_service/src/) | `contraindication_db.py` (675 lines) largely duplicates `infrastructure/database.py` (632 lines). Two independent connection systems for same DB. |
| M21 | [safety_service](services/safety_service/src/) | `observability/telemetry.py` near-identical copy of `infrastructure/telemetry.py`. Deprecated but coexisting. |
| M22 | [personality_service](services/personality_service/src/) | Two incompatible LIWC implementations — `domain/trait_detector.py` (~10 words/category) and `ml/liwc_features.py` (~20-30 words/category). |

### Concurrency

| # | File | Issue |
|---|------|-------|
| M23 | [src/solace_infrastructure/feature_flags.py](src/solace_infrastructure/feature_flags.py#L86) | `_flags` ClassVar mutated by class methods without locking. |
| M24 | [services/orchestrator_service/src/infrastructure/clients.py](services/orchestrator_service/src/infrastructure/clients.py#L75-L100) | `CircuitBreaker` state mutation not async-safe — concurrent requests race on failure count. |
| M25 | [services/orchestrator_service/src/infrastructure/clients.py](services/orchestrator_service/src/infrastructure/clients.py#L120-L129) | `_get_client()` lazy init race — two tasks can create two clients, leaking one. |

### Missing Validation / Auth

| # | File | Issue |
|---|------|-------|
| M26 | [services/diagnosis_service/src/schemas.py](services/diagnosis_service/src/schemas.py#L103-L106) | No `max_length` on message fields — attacker can send megabytes. |
| M27 | [services/memory_service/src/schemas.py](services/memory_service/src/schemas.py#L171) | `role: str` not validated against enum — arbitrary roles accepted. |
| M28 | All services `/status` endpoints | No authentication on status/health endpoints — exposes internal metrics to unauthenticated callers. |
| M29 | [services/config_service/src/feature_flags.py](services/config_service/src/feature_flags.py#L74-L77) | `float()` on attribute values without try/except — non-numeric attributes crash evaluation. |

### Miscellaneous

| # | File | Issue |
|---|------|-------|
| M30 | [services/orchestrator_service/src/agents/*.py](services/orchestrator_service/src/agents/) | All agent node functions create fresh agent instances per invocation. Statistics counters always = 1, settings re-parsed every request. |
| M31 | [services/orchestrator_service/src/events.py](services/orchestrator_service/src/events.py#L248-L259) | `EventBus.publish` error logging loop is dead code (`_safe_invoke` already catches all exceptions). |
| M32 | [services/memory_service/src/domain/consolidation.py](services/memory_service/src/domain/consolidation.py#L299-L307) | `_apply_decay` mutates shared MemoryRecord instances during consolidation — concurrent reads see partially-decayed records. |
| M33 | [services/diagnosis_service/src/domain/severity.py](services/diagnosis_service/src/domain/severity.py#L213-L226) | PHQ-9/GAD-7 scores generated with insufficient data (1-2 items) — clinically misleading without a warning flag. |
| M34 | [services/memory_service/src/main.py](services/memory_service/src/main.py#L257-L264) | Health endpoint always returns "healthy" — doesn't check Postgres, Redis, or Weaviate. |
| M35 | [infrastructure/api_gateway/kong_config.py](infrastructure/api_gateway/kong_config.py#L172-L173) | New `httpx.AsyncClient` created per request — defeats connection pooling. |
| M36 | [infrastructure/api_gateway/kong_config.py](infrastructure/api_gateway/kong_config.py#L174-L192) | All HTTP errors retried including 400/404/409 — only 5xx should retry. |
| M37 | [services/memory_service/src/domain/service.py](services/memory_service/src/domain/service.py) | Three incompatible `KnowledgeTriple` definitions across consolidation.py, knowledge_graph.py, semantic_memory.py. |
| M38 | [services/memory_service/src/config.py](services/memory_service/src/config.py) / [postgres_repo.py](services/memory_service/src/infrastructure/postgres_repo.py) | Two Postgres settings classes reading different env var prefixes (`MEMORY_DB_` vs `POSTGRES_`). Will diverge. |

---

## LOW Issues (Selected)

| # | File | Issue |
|---|------|-------|
| L1 | [src/solace_security/service_auth.py](src/solace_security/service_auth.py#L234) | `__import__("datetime")` instead of standard import. |
| L2 | [src/solace_security/audit.py](src/solace_security/audit.py) | `verify_chain()` loads entire audit log into memory — OOM on long-running systems. |
| L3 | [src/solace_testing/factories.py](src/solace_testing/factories.py) | `random.seed()` pollutes global random state. Should use dedicated `random.Random(seed)`. |
| L4 | [services/orchestrator_service/src/langgraph/supervisor.py](services/orchestrator_service/src/langgraph/supervisor.py#L126-L165) | Substring intent matching: `"help"` matches `"unhelpful"`, `"therapy"` matches `"aromatherapy"`. |
| L5 | [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L117) | `"goodbye"` in `high_risk_keywords` causes false escalation for benign messages. |
| L6 | Tests | `pytest.raises(Exception)` in 14+ test locations — too broad, passes even for wrong exception types. |
| L7 | Tests | `asyncio.sleep(0.15)` in poller tests — time-dependent, flaky. Should use event-based synchronization. |
| L8 | [docker-compose.yml](docker-compose.yml) | Redis has no password set; Weaviate has anonymous access enabled; Postgres uses hardcoded password fallback. |
| L9 | [services/orchestrator_service/src/response/style_applicator.py](services/orchestrator_service/src/response/style_applicator.py#L178-L195) | `\buse\b → utilize`, `\bget\b → obtain` produce unnatural therapeutic text. |
| L10 | [services/personality_service/src/ml/liwc_features.py](services/personality_service/src/ml/liwc_features.py#L165-L175) | Single-quote counting includes all apostrophes, inflating LIWC quotes metric. |

---

## Top 10 Priority Fixes

| Priority | Issue | Impact |
|----------|-------|--------|
| 1 | **C13** — Fix `sqlalchemy~=2.1.0` → `~=2.0.0` | Entire project cannot install |
| 2 | **C2** — Wire safety post-check to actually filter harmful content | User safety — mental health platform |
| 3 | **C1** — Wire the real `Aggregator` into the graph | Core response quality — random response selection |
| 4 | **C3** — Pass LLM client to supervisor during graph construction | Intent classification accuracy |
| 5 | **C4+C5+C6+C7** — Fix all GDPR deletion paths to be atomic and complete | Regulatory compliance |
| 6 | **H1** — Add revocation check to sync JWT decode path | Token revocation completely bypassed |
| 7 | **H2+H3+H4** — Add authentication to WebSocket, authorization checks to all endpoints | User data isolation |
| 8 | **H34+H35** — Use word boundaries (`\b`) in all safety keyword matching | Safety pipeline accuracy |
| 9 | **C8+C9** — Wrap sync I/O (Weaviate, Vault, boto3) in `asyncio.to_thread()` | Server freezes under load |
| 10 | **C10** — Fix Redis rate limiting sync stub | All rate limits bypassed in production |

---

## Architectural Recommendations

1. **Eliminate dead code**: The `response/` module, standalone `websocket.py`, `infrastructure/clients.py`, and `StatePersistenceManager` are all fully implemented but never used. Either wire them in or remove them.

2. **Unify settings**: Each service has 2-3 competing settings classes. Consolidate to one per service.

3. **Add word-boundary matching**: Replace all `kw in text` safety checks with `re.search(rf'\b{re.escape(kw)}\b', text)` — critical for a mental health application.

4. **Move in-memory state to Redis/Postgres**: All services store critical state (sessions, caches, counters) in plain dicts. This is incompatible with horizontal scaling and doesn't survive restarts.

5. **Standardize event handling**: All EventBus implementations swallow exceptions. Add dead-letter queues or at-least-once delivery guarantees for safety-critical events.

6. **Use `SecretStr` everywhere**: All database/Redis passwords should be `pydantic.SecretStr` to prevent credential leakage in logs and tracebacks.

7. **Wrap sync I/O**: Every sync library call inside `async def` (Weaviate, hvac, boto3, HuggingFace inference) must be wrapped in `asyncio.to_thread()`.
