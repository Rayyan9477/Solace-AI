# Core Architecture Audit — Solace-AI

**Date:** 2025-01-XX  
**Scope:** Shared libraries (solace_common, solace_events, solace_security, solace_infrastructure), service domain logic, and cross-service patterns.  
**Excludes:** LangGraph pipeline (see [LANGGRAPH_PIPELINE_AUDIT.md](LANGGRAPH_PIPELINE_AUDIT.md) for 25 additional findings).

---

## Executive Summary

| Severity | Shared Libs | Service Domain | Total |
|----------|:-----------:|:--------------:|:-----:|
| **CRITICAL** | 4 | 3 | **7** |
| **HIGH** | 5 | 6 | **11** |
| **MEDIUM** | 6 | 8 | **14** |
| **LOW** | 3 | 6 | **9** |
| **Total** | **18** | **23** | **41** |

Combined with the LangGraph pipeline audit (3C, 8H, 9M, 5L = 25), the full architecture audit totals **66 findings** across all subsystems.

---

## Part 1 — Shared Library Findings

### CRITICAL

---

#### S-C1: Consumer Never Commits Offsets for Failed/Skipped Messages

**File:** `src/solace_events/consumer.py` — `EventConsumer.consume_loop` (line ~370)  
**Impact:** A single failed or unknown message permanently blocks offset commits for its entire Kafka partition. On consumer restart, ALL messages for that partition are reprocessed from the last committed offset.

**Root cause:** `consume_loop` only calls `mark_processed` when `result.status == ProcessingStatus.SUCCESS`. For `FAILED` or `SKIP` results, the offset stays in `OffsetTracker._pending` forever. Since `mark_processed` can only return a committable value when all lower offsets are cleared from pending, this blocks ALL subsequent commits for that `(topic, partition)` key.

```python
# consumer.py line ~370 — current (broken)
result = await self._process_message(topic, partition, offset, value)
if result.status == ProcessingStatus.SUCCESS:
    committable = await self._offset_tracker.mark_processed(topic, partition, offset)
    if committable:
        await self._maybe_commit({(topic, partition): committable})
# FAILED/SKIP: offset stays in _pending → blocks ALL future commits for partition
```

**Fix:** Call `mark_processed` for ALL statuses. Failed messages should still advance the committed offset (they're already routed to the DLQ).

---

#### S-C2: PostgresClient.connect() Ignores SSL Context

**File:** `src/solace_infrastructure/postgres.py` — `PostgresClient.connect` (line ~174)  
**Impact:** Database connections are NEVER encrypted, even when SSL is configured. The `get_ssl_context()` method exists and returns a proper `ssl.SSLContext`, but `connect()` never passes it to `asyncpg.create_pool()`.

```python
# postgres.py line ~174 — current (broken)
self._pool = await asyncpg.create_pool(
    dsn=self._settings.get_dsn(),
    min_size=self._settings.min_pool_size,
    max_size=self._settings.max_pool_size,
    command_timeout=self._settings.command_timeout,
    statement_cache_size=self._settings.statement_cache_size,
    max_cached_statement_lifetime=self._settings.max_cached_statement_lifetime,
    init=self._init_connection,
    # MISSING: ssl=self._settings.get_ssl_context()
)
```

The `get_effective_ssl_mode()` method even upgrades to "require" in production — but the result is never consumed by `connect()`.

**Fix:** Add `ssl=self._settings.get_ssl_context()` to the `create_pool` call.

---

#### S-C3: ProductionGuards Check Wrong Environment Variable Names

**File:** `src/solace_security/production_guards.py` — `FORBIDDEN_VALUES` / `REQUIRED_IN_PRODUCTION` (lines 63–77)  
**Impact:** Production security validation is completely bypassed for encryption and auth settings. The guards check env vars that don't match the actual settings.

| Guard Checks | Actual Env Var (pydantic prefix + field) | Match? |
|---|---|---|
| `ENCRYPTION_KEY` | `ENCRYPTION_MASTER_KEY` | **NO** |
| `SECRET_KEY` | `AUTH_SECRET_KEY` | **NO** |
| `JWT_SECRET` | `AUTH_SECRET_KEY` | **NO** |

The encryption settings use `env_prefix="ENCRYPTION_"` with field `master_key` → env var `ENCRYPTION_MASTER_KEY`.  
The auth settings use `env_prefix="AUTH_"` with field `secret_key` → env var `AUTH_SECRET_KEY`.

```python
# production_guards.py — current (wrong)
FORBIDDEN_VALUES = {
    "SECRET_KEY": [...],       # Should be AUTH_SECRET_KEY
    "ENCRYPTION_KEY": [...],   # Should be ENCRYPTION_MASTER_KEY
    "JWT_SECRET": [...],       # Should be AUTH_SECRET_KEY
}
REQUIRED_IN_PRODUCTION = [
    "ENCRYPTION_KEY",          # Should be ENCRYPTION_MASTER_KEY
]
```

**Result:** A production deployment with `ENCRYPTION_MASTER_KEY=dev` passes all guards. The encryption key is never validated.

**Fix:** Update env var names to match actual pydantic settings prefixes.

---

#### S-C4: OffsetTracker.mark_processed Has Broken In-Order Commit Logic

**File:** `src/solace_events/consumer.py` — `OffsetTracker.mark_processed` (line ~110)  
**Impact:** Even for successfully processed messages, intermediate offsets are never committed. Only when ALL messages in a batch are processed (pending list empties) is an offset committed.

**Root cause:** The condition `if offset + 1 < min_pending` uses strict `<`. In sequential processing:

```
Receive offsets [0, 1, 2].
Process offset 0: pending = [1, 2], min_pending = 1, offset+1 = 1
  → 1 < 1 = False → returns None (BUG: should commit 1)
Process offset 1: pending = [2], min_pending = 2, offset+1 = 2
  → 2 < 2 = False → returns None (BUG: should commit 2)
Process offset 2: pending = [], committable = 3 → returns 3 ✓
```

Only when ALL messages are processed does a commit happen. If message 2 fails, offsets 0 and 1 are never individually committed despite being successfully processed.

Additionally, for out-of-order processing, when pending empties the code uses `offset + 1` (the last-processed offset + 1), which may be LESS than the highest offset in the original batch — causing already-processed messages to be redelivered.

**Fix:** Change the condition to `<=` and use `max(self._offsets.get(key, 0), min_pending)` for the committable value.

---

### HIGH

---

#### S-H1: Redis PubSub Listener Never Restarts on Failure

**File:** `src/solace_infrastructure/redis.py` — `_pubsub_listener` (line ~290)  
**Impact:** If the Redis connection drops during pub/sub listening, the listener task exits silently. All pub/sub subscriptions stop working permanently — no reconnection, no alerting.

```python
async def _pubsub_listener(self) -> None:
    client = self._ensure_connected()
    pubsub: PubSub = client.pubsub()
    try:
        await pubsub.subscribe(*self._pubsub_handlers.keys())
        async for message in pubsub.listen():
            # ... handler dispatch ...
    except asyncio.CancelledError:
        await pubsub.unsubscribe()
    finally:
        await pubsub.aclose()
    # Task ends here — no restart on ConnectionError, TimeoutError, etc.
```

Any non-`CancelledError` exception (e.g., `ConnectionError`, `TimeoutError`) drops through to `finally` and the task terminates. `self._pubsub_task` retains the dead task reference, so future `subscribe()` calls won't create a new listener.

**Fix:** Wrap the listener in a retry loop with exponential backoff. Reset `self._pubsub_task = None` on failure so new subscriptions trigger a fresh listener.

---

#### S-H2: Topic Map References Non-Existent "solace.notifications" Topic

**File:** `src/solace_events/schemas.py` — `_TOPIC_MAP` (line ~522)  
**Impact:** Notification events are routed to `"solace.notifications"` which doesn't exist in the `SolaceTopic` enum or `TOPIC_CONFIGS`. Any code using `SolaceTopic.from_string("solace.notifications")` will raise `ValueError`.

```python
# schemas.py line ~522
_TOPIC_MAP = {
    "notification.": "solace.notifications",  # ← No SolaceTopic.NOTIFICATIONS exists
    ...
}
# config.py — SolaceTopic enum has: SESSIONS, ASSESSMENTS, THERAPY, SAFETY, MEMORY, ANALYTICS, PERSONALITY
# No NOTIFICATIONS member
```

**Fix:** Add `NOTIFICATIONS = "solace.notifications"` to `SolaceTopic` and add its `TopicConfig` to `TOPIC_CONFIGS`.

---

#### S-H3: Dead Letter Handler Double-Counts Retries

**File:** `src/solace_events/dead_letter.py` — `DeadLetterHandler.handle_failure` (line ~270)  
**Impact:** The initial failure counts as retry attempt 1, effectively reducing `max_retries` by 1. With `max_retries=3`, only 2 actual retry attempts occur.

```python
async def handle_failure(self, event, original_topic, error, retry_count=0):
    record = DeadLetterRecord.from_failed_event(
        event, original_topic, self._consumer_group, error, retry_count  # retry_count=0
    )
    can_retry = record.increment_retry(error, self._retry_policy)  # → retry_count becomes 1
```

`from_failed_event` sets `retry_count=0`, then `increment_retry` immediately bumps it to 1. The initial failure is counted as the first retry.

**Fix:** Either skip `increment_retry` in `handle_failure` (just save the record), or set `retry_count=-1` to compensate.

---

#### S-H4: Audit Chain Has No Tamper Protection by Default

**File:** `src/solace_security/audit.py` — `AuditSettings.hmac_key` (line ~65)  
**Impact:** `hmac_key` defaults to `""` (empty string). When empty, `compute_hash` falls back to plain SHA-256 — anyone with database access can forge audit entries by recomputing hashes.

```python
hmac_key: str = Field(default="", description="HMAC key for audit chain integrity.")

# In compute_hash:
if hmac_key:       # "" is falsy → always False when using default
    return hmac.new(hmac_key.encode(), ...).hexdigest()
return hashlib.new(algorithm, ...).hexdigest()  # Plain hash — trivially forgeable
```

**Fix:** Make `hmac_key` required (no default) or raise on startup in production if unset.

---

#### S-H5: publish_batch Defeats Outbox Transactional Guarantees

**File:** `src/solace_events/publisher.py` — `EventPublisher.publish_batch` (~line 340)  
**Impact:** Events in a batch are saved to the outbox independently via `asyncio.gather`. If the process crashes mid-batch, some events may be persisted and later published while others are lost — violating the exactly-once transactional guarantee the outbox pattern is designed to provide.

**Fix:** Save all batch events in a single database transaction before marking them for relay.

---

### MEDIUM

---

#### S-M1: InMemoryAuditStore Grows Unbounded

**File:** `src/solace_security/audit.py`  
**Impact:** The in-memory audit store appends to `self._events` list with no size limit or eviction policy. In long-running test/dev environments, memory grows linearly with every audited action.

**Fix:** Add a configurable max size with FIFO eviction, or log a warning when size exceeds a threshold.

---

#### S-M2: LRU-Cached JWT Manager Cannot Be Invalidated in Tests

**File:** `src/solace_security/middleware.py` — `_get_jwt_manager()`  
**Impact:** `@lru_cache()` on a zero-argument function creates a process-level singleton. Tests that modify `AuthSettings` env vars get stale JWT managers, causing flaky tests and incorrect auth behavior in test suites.

**Fix:** Use a module-level variable with explicit reset function, or `cache_clear()` in test fixtures.

---

#### S-M3: PHI SSN Pattern Matches Any 9-Digit Number

**File:** `src/solace_security/phi_protection.py`  
**Impact:** The SSN detection pattern `r"\b\d{9}\b"` with confidence 0.80 matches ANY 9-digit number — zip+4 codes, phone numbers without dashes, arbitrary IDs. This causes false positive PHI redaction in normal chat messages, potentially corrupting therapeutic content.

**Fix:** Use the standard SSN format `r"\b\d{3}-\d{2}-\d{4}\b"` or restrict the unformatted pattern to known SSN ranges (not starting with 000, 666, or 900-999).

---

#### S-M4: Weaviate Sync Operations May Exhaust Default Thread Pool

**File:** `src/solace_infrastructure/weaviate.py`  
**Impact:** All Weaviate operations use `loop.run_in_executor(None, ...)` which runs in the default `ThreadPoolExecutor`. Under high load, concurrent vector searches could exhaust the pool (default size: `min(32, os.cpu_count() + 4)`), blocking other async tasks that need the executor.

**Fix:** Create a dedicated executor for Weaviate with a bounded size, or use an async Weaviate client.

---

#### S-M5: ServiceTokenManager._token_cache Is Not Thread-Safe

**File:** `src/solace_security/service_auth.py`  
**Impact:** The `_token_cache` dict is accessed from async contexts without any lock. While Python's GIL protects against corruption for dict operations, concurrent reads and writes during cache population could lead to duplicate token generation or stale reads in edge cases.

**Fix:** Use `asyncio.Lock` around cache access, consistent with `OffsetTracker`'s pattern.

---

#### S-M6: FeatureFlags._flags Is a Mutable Class Variable Without Lock

**File:** `src/solace_infrastructure/feature_flags.py`  
**Impact:** `_flags` is defined at class level and mutated by `register_flag`, `enable_flag`, `disable_flag`. Concurrent async operations could see partially-updated flag state.

**Fix:** Use an instance-level dict with `asyncio.Lock`, or make it immutable with copy-on-write semantics.

---

### LOW

---

#### S-L1: RetryPolicy Imports `random` Inside Function on Every Call

**File:** `src/solace_events/dead_letter.py` — `RetryPolicy.get_delay_ms`  
**Impact:** Minor performance overhead from repeated `import random` on each DECORRELATED/jitter calculation. Python caches module imports, so the overhead is just the lookup — but it's non-idiomatic.

**Fix:** Move `import random` to module level.

---

#### S-L2: EncryptedData.from_compact Doesn't Validate Version

**File:** `src/solace_security/encryption.py`  
**Impact:** The version byte is parsed but never checked. If a future version changes the ciphertext format, old code would attempt decryption with the wrong layout — producing garbage or a confusing error instead of a clean "unsupported version" message.

**Fix:** Add `if version != CURRENT_VERSION: raise ValueError(...)`.

---

#### S-L3: _structlog_sanitizer Global Lazily Initialized Without Thread Safety

**File:** `src/solace_security/phi_protection.py`  
**Impact:** Module-level mutable global is set on first access. In multi-threaded environments, parallel imports could create duplicate instances. Harmless in practice (both instances are equivalent), but non-idiomatic.

**Fix:** Initialize at module import time or use a `Lock`.

---

## Part 2 — Service Domain Logic Findings

### CRITICAL

---

#### D-C1: All Agent HTTP Clients Create New Connection Per Request

**Files:** `services/orchestrator_service/src/agents/therapy_agent.py`, `safety_agent.py`, `diagnosis_agent.py`, `personality_agent.py`  
**Impact:** Every LangGraph invocation creates a new `httpx.AsyncClient` for each agent HTTP call. Connections are never pooled or reused. Under load, this exhausts file descriptors and TCP sockets, causing `OSError: [Errno 24] Too many open files`.

```python
# Example from therapy_agent.py — creates new client per call
async with httpx.AsyncClient() as client:
    response = await client.post(f"{self._base_url}/analyze", json=payload)
```

**Fix:** Create a shared `httpx.AsyncClient` at application startup and inject it into agents. Use FastAPI lifespan events to manage the client lifecycle.

---

#### D-C2: Agent Instances Recreated Per LangGraph Invocation

**Files:** `services/orchestrator_service/src/langgraph/nodes.py` (or equivalent node functions)  
**Impact:** Each LangGraph node function instantiates a new agent object. Combined with D-C1, this means every chat message creates 4+ new HTTP clients (one per agent), all of which are immediately discarded.

**Fix:** Create agent instances once at module/application level and reference them in node functions.

---

#### D-C3: Memory Service Stores All Data in Unbounded In-Memory Dicts

**File:** `services/memory_service/src/domain/service.py`  
**Impact:** Memory tiers (`_tier_3_session`, `_tier_4_episodic`, `_tier_5_semantic`) are plain dicts with no size limit or eviction. With continuous usage, memory grows linearly — eventually causing OOM.

**Fix:** Add LRU eviction or max-size bounds. Use Redis or database-backed tiers for production.

---

### HIGH

---

#### D-H1: Crisis Keyword Detection Has No Negation Handling

**File:** `services/therapy_service/src/domain/service.py`  
**Impact:** Messages like *"I don't want to harm anyone"* or *"I'm not thinking about suicide"* trigger crisis detection because the keywords "harm" and "suicide" match regardless of context. This causes false crisis escalations, disrupting therapeutic sessions.

**Fix:** Add negation-aware analysis — at minimum, skip matches preceded by "not", "don't", "no", "never" within N words.

---

#### D-H2: Safety Assessment Cache TTL Configured But Never Enforced

**File:** `services/safety_service/src/domain/service.py`  
**Impact:** The safety assessment cache has a TTL setting but no expiration mechanism. Stale safety assessments persist indefinitely, meaning a user flagged as high-risk remains flagged even after behavioral change, or a cleared user stays cleared despite new risk signals.

**Fix:** Add a background task or lazy eviction that removes entries older than TTL on access.

---

#### D-H3: Diagnosis Service delete_user_data Only Clears In-Memory State

**File:** `services/diagnosis_service/src/domain/service.py`  
**Impact:** GDPR `delete_user_data` clears in-memory caches but does NOT delete from the PostgreSQL repository. Diagnosis records, session data, and assessment history persist in the database despite the user requesting deletion — a compliance violation.

**Fix:** Add `await self._repository.delete_by_user_id(user_id)` to the deletion flow.

---

#### D-H4: Personality Service Confidence Weights Sum to 1.1

**File:** `services/personality_service/src/domain/service.py` — `_aggregate_scores`  
**Impact:** The aggregation formula uses weights `0.8 + 0.3 = 1.1` instead of `1.0`. This biases all personality confidence scores upward by ~10%, making assessments appear more certain than warranted.

```python
# Current (wrong):
aggregated = current_score * 0.8 + new_score * 0.3  # Sum = 1.1
# Correct:
aggregated = current_score * 0.7 + new_score * 0.3  # Sum = 1.0
```

**Fix:** Change `0.8` to `0.7` (or `0.8` and `0.2` if current data should be weighted more heavily).

---

#### D-H5: Safety Agent Fallback Uses Substring Instead of Word-Boundary Matching

**File:** `services/orchestrator_service/src/agents/safety_agent.py`  
**Impact:** The fallback crisis keyword check uses `kw in message.lower()` (substring match). This means "skill" matches "kill", "penknife" matches "knife", "therapist" matches "the rapist". False positives trigger unnecessary crisis protocols.

Note: The primary `crisis_detector.py` was fixed with word-boundary regex in Round 1 (H34), but this fallback path was missed.

**Fix:** Use `re.search(rf"\b{re.escape(kw)}\b", message, re.IGNORECASE)` for each keyword.

---

#### D-H6: ConfigLoader.to_dict() Exposes Sensitive Values

**File:** `services/config_service/src/domain/service.py`  
**Impact:** `to_dict()` calls `model_dump()` which serializes all fields including `redis_url` and `postgres_url` containing plaintext credentials. If this dict is logged, returned via API, or included in error messages, credentials leak.

**Fix:** Exclude sensitive fields in `model_dump(exclude={"redis_url", "postgres_url", ...})` or use `SecretStr` fields that redact on serialization.

---

### MEDIUM

---

#### D-M1: Memory Service _apply_decay_model Double-Counts Decayed Records

**File:** `services/memory_service/src/domain/service.py`  
**Impact:** The decay function processes both postgres-backed and in-memory records, but some records exist in both. Records present in both tiers get double-decayed — their relevance scores drop faster than intended, causing premature memory eviction.

**Fix:** Deduplicate by record ID before applying decay, or track which store is authoritative.

---

#### D-M2: Therapy Service Session Persist Failure Silently Swallowed

**File:** `services/therapy_service/src/domain/service.py`  
**Impact:** When session persistence fails (database error), the exception is caught and logged but the caller receives a success response. The user's therapy session data is lost without any indication.

**Fix:** Re-raise or return an error status so upstream code can retry or notify the user.

---

#### D-M3: Therapy Service Race Condition in _treatment_plans Dict

**File:** `services/therapy_service/src/domain/service.py`  
**Impact:** TOCTOU (time-of-check-time-of-use) on `if user_id in self._treatment_plans`. Concurrent requests for the same user could both see "not present" and create duplicate plans, with the second overwriting the first.

**Fix:** Use `asyncio.Lock` per user_id or `setdefault` pattern.

---

#### D-M4: Diagnosis Service Endlessly Extends Lists Without Dedup or Bounds

**File:** `services/diagnosis_service/src/domain/service.py` — `_update_session`  
**Impact:** Each diagnosis update appends to `symptoms`, `reasoning`, and `safety_flags` lists without checking for duplicates or enforcing a maximum size. Over a long session, these lists grow unboundedly, increasing payload sizes and LLM context consumption.

**Fix:** Add deduplication (by symptom text) and cap list sizes (e.g., keep most recent 50).

---

#### D-M5: Inconsistent LLM Client Injection Across Agents

**Files:** `services/orchestrator_service/src/agents/chat_agent.py` vs other agents  
**Impact:** `chat_agent.py` uses a global mutable `_llm_client` with setter injection, while other agents instantiate their own clients. This inconsistency makes testing and configuration management fragile — changing the LLM backend requires touching multiple code paths.

**Fix:** Standardize on dependency injection via constructor for all agents.

---

#### D-M6: Safety Service _conversation_history Outer Dict Unbounded

**File:** `services/safety_service/src/domain/service.py`  
**Impact:** The conversation history `dict[user_id, list[messages]]` has per-user list caps but no limit on the number of user_ids tracked. Over time, every user who ever interacted gets an entry — memory grows proportionally to total user count.

**Fix:** Add LRU eviction on the outer dict or use Redis with TTL for conversation history.

---

#### D-M7: Personality Service _compute_stability Divides by Fixed 5

**File:** `services/personality_service/src/domain/service.py`  
**Impact:** The stability computation hardcodes `/ 5` (OCEAN trait count). If `PersonalityTrait` enum is ever extended, this will silently produce incorrect stability scores.

**Fix:** Use `len(PersonalityTrait)` instead of the magic number.

---

#### D-M8: Memory Service end_session Uses `del` Without Concurrency Guard

**File:** `services/memory_service/src/domain/service.py`  
**Impact:** `del self._tier_3_session[session_id]` is called without a lock. Concurrent session operations could raise `KeyError` if another coroutine deletes the same session first.

**Fix:** Use `self._tier_3_session.pop(session_id, None)` under an `asyncio.Lock`.

---

### LOW

---

#### D-L1: Graph Builder safety_precheck_node Is Synchronous

**File:** `services/orchestrator_service/src/langgraph/graph_builder.py`  
**Impact:** `safety_precheck_node` is a sync function. LangGraph handles this, but regex matching blocks the event loop. Negligible for short messages, but inconsistent with other async nodes.

---

#### D-L2: Memory Service _build_basic_context Ignores token_budget

**File:** `services/memory_service/src/domain/service.py`  
**Impact:** The `token_budget` parameter is accepted but never used. All 10 recent working memory records are included regardless, potentially exceeding LLM context limits.

---

#### D-L3: Therapy Service Phase Transition Based on Wall-Clock Time

**File:** `services/therapy_service/src/domain/service.py`  
**Impact:** Phase transition from WORKING to CLOSING triggers after 900s of wall-clock time, even if the user is idle. The user may send their next message into a CLOSING phase unexpectedly.

---

#### D-L4: Diagnosis Hypothesis Confidence Floor Too Low

**File:** `services/diagnosis_service/src/domain/service.py`  
**Impact:** The confidence floor of `Decimal("0.1")` allows valid hypotheses to drop to near-zero after multiple challenges, presenting insufficient certainty even when strong evidence exists.

---

#### D-L5: SQL Schema Names From Config Used in f-strings

**Files:** Multiple repository files across services  
**Impact:** Table names built via `f"DELETE FROM {self._table} ..."` where `self._table` comes from config. Currently safe (config-sourced, not user input), but fragile if config source changes.

---

#### D-L6: Personality Service _extract_evidence Returns Empty Silently

**File:** `services/personality_service/src/domain/service.py`  
**Impact:** When `trait_scores` is empty, `_extract_evidence` returns `[]` without warning. The `include_evidence=True` flag appears broken from the caller's perspective.

---

## Part 3 — Cross-Cutting Concerns

These patterns span multiple components and amplify individual findings:

### 1. No Circuit Breaker Pattern

**Impact:** Infrastructure clients (Redis, Postgres, Weaviate) retry on initial connection but have no circuit breaker for ongoing operations. A repeatedly failing dependency causes cascading request queuing across all services. Each failed call waits for its full timeout before failing, multiplying latency.

**Recommendation:** Add a circuit breaker (e.g., `circuitbreaker` library or custom state machine) to infrastructure clients. After N consecutive failures, fail fast for a cool-down period.

### 2. In-Memory State Without Eviction (Systemic)

**Impact:** Multiple components maintain unbounded in-memory state:
- `InMemoryAuditStore._events` (S-M1)
- `OffsetTracker._pending` blocked by failed messages (S-C1)
- Memory service tiers (D-C3)
- Safety service `_conversation_history` outer dict (D-M6)
- Diagnosis service symptom/reasoning/safety lists (D-M4)
- Dead letter store `_records` (if using in-memory store)
- Feature flags `_flags`

**Recommendation:** Establish a platform-wide policy: every in-memory cache/store MUST have a max size and eviction policy. Add `maxlen` enforcement and TTL tracking to a shared base class.

### 3. Settings/Environment Variable Name Mismatch

**Impact:** Pydantic settings use `env_prefix` + field name (e.g., `ENCRYPTION_MASTER_KEY`), but other code references bare names (e.g., `ENCRYPTION_KEY` in ProductionGuards). This disconnect means production safety validators don't actually validate the settings they're protecting.

**Recommendation:** Derive `REQUIRED_IN_PRODUCTION` and `FORBIDDEN_VALUES` keys programmatically from pydantic settings classes, or create a mapping layer that stays in sync.

### 4. Inconsistent Error Handling Patterns

**Impact:** Some components silently swallow errors (D-M2 therapy persist, consumer SKIP), others fail loudly. There's no standard for what constitutes a retriable vs fatal vs ignorable error. This makes debugging difficult and causes silent data loss.

**Recommendation:** Define an error hierarchy with clear categories (retriable/transient, permanent/fatal, degraded/warn) and enforce consistent handling per category.

### 5. No Graceful Degradation for Event System

**Impact:** If Kafka is down, the publisher's outbox saves events locally — but the consumer has no fallback. Combined with S-C1 (failed messages blocking commits), a partition with even one bad message becomes permanently stuck until manual intervention.

**Recommendation:** Add a mechanism to skip or dead-letter messages that have been pending beyond a configurable threshold. Add alerting when committed offset lag exceeds bounds.

---

## Priority Remediation Order

Based on severity, blast radius, and fix complexity:

| Priority | ID | Finding | Risk | Effort |
|:--------:|:--:|---------|------|--------|
| 1 | S-C1 | Consumer offset commit blocked by failures | Data loss, infinite reprocessing | Small |
| 2 | S-C2 | PostgresClient SSL completely ignored | Unencrypted DB traffic in prod | Small |
| 3 | S-C3 | ProductionGuards wrong env var names | Security bypass in production | Small |
| 4 | S-C4 | OffsetTracker broken commit logic | Reprocessing, data loss | Medium |
| 5 | D-C1+D-C2 | Per-request HTTP clients + agent recreation | Resource exhaustion (FD leak) | Medium |
| 6 | D-C3 | Unbounded memory growth in memory service | OOM in production | Medium |
| 7 | D-H3 | Diagnosis GDPR deletion incomplete | Compliance violation | Small |
| 8 | S-H1 | Redis PubSub listener dies permanently | Silent feature failure | Small |
| 9 | S-H3 | Dead letter double-counts retries | Fewer retries than configured | Small |
| 10 | D-H5 | Safety fallback substring matching | False crisis escalations | Small |
| 11 | S-H4 | Audit HMAC key empty by default | Audit tamper risk | Small |
| 12 | D-H4 | Personality weights sum to 1.1 | Biased confidence scores | Small |
| 13 | S-H2 | Missing NOTIFICATIONS topic | Notification events can't route | Small |
| 14 | D-H1 | No negation handling in crisis detection | False crisis escalations | Medium |
| 15 | D-H2 | Safety cache TTL never enforced | Stale risk assessments | Medium |

---

## Appendix: Related Audit Documents

| Document | Scope | Findings |
|---|---|---|
| [FULL_CODEBASE_AUDIT.md](FULL_CODEBASE_AUDIT.md) | Original line-by-line audit | ~150 findings |
| [REMEDIATION_TRACKING.md](REMEDIATION_TRACKING.md) | Round 1 fixes + remaining issues | 19 fixed, 83 remaining |
| [LANGGRAPH_PIPELINE_AUDIT.md](LANGGRAPH_PIPELINE_AUDIT.md) | LangGraph orchestration pipeline | 25 findings (3C, 8H, 9M, 5L) |
| **This document** | Core architecture (shared libs + services) | 41 findings (7C, 11H, 14M, 9L) |

**Grand total across all audits:** ~166 findings identified, 19 fixed in Round 1, **~147 remaining**.
