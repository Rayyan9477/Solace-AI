# Solace-AI: Phase 5 & 6 Comprehensive Code Review

**Review Date:** 2026-02-07
**Reviewed By:** Senior AI Engineer
**Scope:** Phase 5 (Granular Permissions & Inter-Service Auth) + Phase 6 (Migration & Rollout) + Remaining Services (Orchestrator, Memory, Notification, Analytics) + Deployment Configuration
**Method:** Line-by-line code analysis of all implementation files

---

## Executive Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Phase 5: Permissions & Auth | 6 | 16 | 5 | 2 | 29 |
| Orchestrator Service | 2 | 3 | 5 | 0 | 10 |
| Memory Service | 2 | 3 | 1 | 0 | 6 |
| Notification Service | 0 | 3 | 4 | 0 | 7 |
| Analytics Service | 1 | 5 | 5 | 0 | 11 |
| Phase 6: Deployment & Config | 1 | 6 | 6 | 3 | 16 |
| **TOTAL** | **12** | **36** | **26** | **5** | **79** |

**Verdict:** The codebase has **6 critical authentication bypass vulnerabilities** that allow unauthenticated access to HIPAA-protected endpoints, **2 critical runtime crashes** in the orchestrator/memory pipeline, a **committed .env file with secrets**, and an **entirely disabled CI/CD pipeline**. The permission system exists architecturally but is bypassed or unenforced on the majority of service endpoints.

---

## Cumulative Issue Tracker (All Phases)

| Phase | Critical | High | Medium | Low | Total |
|-------|----------|------|--------|-----|-------|
| Phase 1-2 | 13 | 17 | 21 | 14 | 65 |
| Phase 3-4 | 12 | 24 | 28 | 16 | 80 |
| Phase 5-6 | 12 | 36 | 26 | 5 | 79 |
| **Grand Total** | **37** | **77** | **75** | **35** | **224** |

### Resolution Status (Updated 2026-02-08)

**12 of 12 critical resolved. 12 of 28 high resolved. 1 of 26 medium resolved.**

#### Critical Issues

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| CRITICAL-056 | WebSocket endpoint zero authentication | **RESOLVED** | T1.3 |
| CRITICAL-057 | Escalation endpoint unauthenticated | **RESOLVED** | T1.3 |
| CRITICAL-058 | Orchestrator batch/session endpoints missing auth | **RESOLVED** | T1.3 |
| CRITICAL-059 | Hardcoded JWT secret default | **RESOLVED** | T0.4 |
| CRITICAL-060 | Admin role bypasses all authorization | **RESOLVED** | T1.5 |
| CRITICAL-061 | Service token accepts arbitrary permission escalation | **RESOLVED** | T1.6 |
| CRITICAL-062 | Memory node calls non-existent method | **RESOLVED** | T0.9 |
| CRITICAL-063 | Memory node uses wrong AgentType | **RESOLVED** | T0.19 |
| CRITICAL-064 | Unbounded memory growth in all tiers | **RESOLVED** | T2.4 |
| CRITICAL-065 | Consolidation crashes when pipeline is None | **RESOLVED** | T0.20 |
| CRITICAL-066 | Missing relative import breaks analytics | **RESOLVED** | T0.10 |
| CRITICAL-067 | .env file with secrets committed to repo | **RESOLVED** | T0.4 |

#### High Issues

| Issue ID | Description | Status | Fix Reference |
|----------|-------------|--------|---------------|
| HIGH-074 | String-based role checks throughout all services | **RESOLVED** | T1.12 |
| HIGH-075 | Therapy service endpoints mostly unauthenticated | **RESOLVED** | T1.3 |
| HIGH-076 | Memory service cross-user data access | **RESOLVED** | T1.3 |
| HIGH-077 | Service-to-service auth without user verification | **RESOLVED** | T1.14 |
| HIGH-078 | Token blacklist in-memory & optional | **RESOLVED** | T1.4 |
| HIGH-079 | Invalid role silently ignored | **RESOLVED** | T1.12 |
| HIGH-080 | Detailed health endpoint leaks info without auth | **RESOLVED** | T1.3 |
| HIGH-081 | Failed service identity not logged as security event | OPEN | Backlog |
| HIGH-082 | Safety agent returns None content for non-crisis | OPEN | Backlog |
| HIGH-083 | Hardcoded crisis response bypasses safety service | OPEN | Backlog |
| HIGH-084 | ChatAgent.process() is synchronous in async graph | OPEN | Tier 4 (T4.3) |
| HIGH-085 | Semantic filter is just substring match | OPEN | Tier 7 (T7.4) |
| HIGH-086 | Safety context can get zero tokens | OPEN | Tier 3 (T3.11) |
| HIGH-087 | Token budget can go negative | OPEN | Tier 3 (T3.11) |
| HIGH-088 | Hardcoded fallback email addresses | OPEN | Tier 3 (T3.1) |
| HIGH-089 | Google auth hard failure with no fallback | OPEN | Backlog |
| HIGH-090 | Placeholder clinician email construction | OPEN | Tier 3 (T3.1) |
| HIGH-091 | SQL injection in LIMIT clause | **RESOLVED** | T0.5 |
| HIGH-092 | Event handler dispatch logic inverted | OPEN | Tier 6 (T6.11) |
| HIGH-093 | Unsafe UUID parsing masks data corruption | OPEN | Tier 6 (T6.12) |
| HIGH-094 | Missing event category handlers | OPEN | Tier 6 (T6.13) |
| HIGH-095 | Mutable defaults in auth fallback dataclass | **RESOLVED** | T1.2 |
| HIGH-096 | All CI/CD pipeline steps disabled | OPEN | Tier 5 (T5.1) |
| HIGH-097 | Hardcoded default secrets in docker-compose | **RESOLVED** | T0.4 |
| HIGH-098 | 6 services missing requirements.txt | OPEN | Tier 5 (T5.6) |
| HIGH-099 | Loose version constraints in root requirements | OPEN | Tier 5 (T5.7) |
| HIGH-100 | Pre-commit CI skips type checking and tests | OPEN | Tier 5 (T5.1) |
| HIGH-101 | Uvicorn version inconsistency | OPEN | Tier 5 (T5.7) |

#### Medium/Low Summary
- MEDIUM-054 (Safety /status endpoint without auth): **RESOLVED** (T1.3)
- Remaining 25 medium + 5 low: OPEN (planned for Tiers 3-7 or backlog)

---

## PHASE 5: GRANULAR PERMISSIONS & INTER-SERVICE AUTH

### CRITICAL-056: WebSocket Endpoint Has Zero Authentication
**File:** [api.py](../services/orchestrator_service/src/api.py)
**Lines:** 261-347
**Severity:** CRITICAL (HIPAA Violation)

```python
@router.websocket("/ws/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    session_id: str,
    user_id: str = Query(..., description="User identifier"),  # UNTRUSTED INPUT
) -> None:
    await websocket.accept()  # ACCEPTS WITHOUT ANY AUTH
    # user_id taken directly from query string - no JWT validation
    initial_state = create_initial_state(
        user_id=user_id,  # USER_ID NOT VALIDATED
```

**Attack:** Connect to `/ws/session_id?user_id=victim_id` and receive therapy responses as that user. No token, no auth, no verification.

---

### CRITICAL-057: Escalation Endpoint Completely Unauthenticated
**File:** [api.py](../services/safety_service/src/api.py)
**Lines:** 270-294
**Severity:** CRITICAL (HIPAA Violation)

```python
@router.post("/escalate", response_model=EscalationResponse, status_code=status.HTTP_201_CREATED)
async def trigger_escalation(
    request: EscalationRequest,
    safety_service: SafetyService = Depends(get_safety_service),
    # NO authentication dependency - COMPLETELY OPEN
) -> EscalationResponse:
```

**Attack:** Anyone can POST to `/escalate` to spam clinician escalations. DoS attack on on-call clinical staff. Multiple safety service endpoints also lack auth: `/detect-crisis` (line 247), `/assess` (line 297), `/filter-output` (line 324).

---

### CRITICAL-058: Orchestrator Batch & Session Endpoints Missing Auth
**File:** [api.py](../services/orchestrator_service/src/api.py)
**Lines:** 192-208, 211-244, 349-405
**Severity:** CRITICAL (HIPAA Violation)

```python
@router.post("/sessions", response_model=SessionCreateResponse)
async def create_session(request_data: SessionCreateRequest, request: Request):
    # NO Depends(get_current_user) - ANY request can create sessions

@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    # NO current_user dependency - ANYONE can read ANY user's conversation history

@router.post("/batch")
async def batch_process(request: BatchRequest):
    # NO current_user validation - processes messages without auth
```

---

### CRITICAL-059: Hardcoded JWT Secret Default
**File:** [auth_plugin.py](../infrastructure/api_gateway/auth_plugin.py)
**Line:** 49
**Severity:** CRITICAL

```python
class JWTConfig(BaseSettings):
    secret_key: str = Field(default="your-secret-key-change-in-production")
```

If deployed without overriding, ALL JWTs can be forged using this known default secret. Combined with the admin bypass (line 240), this grants full system access.

---

### CRITICAL-060: Admin Role Bypasses All Authorization Checks
**File:** [auth_plugin.py](../infrastructure/api_gateway/auth_plugin.py)
**Lines:** 240-245
**Severity:** CRITICAL

```python
def authorize(self, claims: TokenClaims, required_roles: list[UserRole] | None = None) -> bool:
    if not required_roles:
        return True
    if UserRole.ADMIN in claims.roles or UserRole.SYSTEM in claims.roles:
        return True  # ADMIN BYPASSES ALL ENDPOINT RESTRICTIONS
    return claims.has_any_role(required_roles)
```

Combined with CRITICAL-059: forge a JWT with `"roles": ["admin"]` using the hardcoded secret → bypass ALL authorization.

---

### CRITICAL-061: Service Token Accepts Arbitrary Permission Escalation
**File:** [service_auth.py](../src/solace_security/service_auth.py)
**Lines:** 190-218
**Severity:** CRITICAL

```python
def create_service_token(
    self,
    service_name: str,
    target_service: str | None = None,
    additional_permissions: list[str] | None = None,  # NO VALIDATION
    expire_minutes: int | None = None,
) -> ServiceCredentials:
    base_permissions = [p.value for p in SERVICE_PERMISSIONS.get(service_identity, [])]
    all_permissions = list(set(base_permissions + (additional_permissions or [])))
    # Accepts ANY string as permission - no validation against allowed set
```

**Attack:** Any service can call `create_service_token("self", additional_permissions=["service:admin:*"])` to escalate privileges beyond its defined scope.

---

### HIGH-074: String-Based Role Checks Throughout All Services
**Files:** Multiple service api.py files
**Severity:** HIGH (8 occurrences)

```python
# diagnosis_service/src/api.py - Lines 95, 134, 165, 195, 224, 252, 284, 329
if "clinician" not in current_user.roles and "admin" not in current_user.roles:
    raise HTTPException(status_code=403, ...)
```

All role checks use raw string comparison instead of Role enum validation. If JWT payload contains arbitrary role strings, they're accepted without validation. Pattern repeats across diagnosis, therapy, and memory service endpoints.

---

### HIGH-075: Therapy Service Endpoints Mostly Unauthenticated
**File:** [api.py](../services/therapy_service/src/api.py)
**Lines:** 268-466
**Severity:** HIGH

Missing `Depends(get_current_user)` on 5 endpoints:
- Line 268: `GET /sessions/{id}/state` - Returns session state
- Line 302: `GET /sessions/{id}/treatment-plan` - Exposes treatment plans
- Line 336: `POST /sessions/{id}/homework` - Creates assignments
- Line 405: `DELETE /sessions/{id}` - Deletes any session
- Line 443: `GET /users/{user_id}/progress` - Returns progress for ANY user_id

---

### HIGH-076: Memory Service Cross-User Data Access
**File:** [api.py](../services/memory_service/src/api.py)
**Lines:** 274-323
**Severity:** HIGH (HIPAA Violation)

```python
# Line 274-299: GET /profile/{user_id}
# Checks "clinician" as string - vulnerable to role tampering

# Line 310-323: DELETE /user/{user_id}
# Checks "admin" not in current_user.roles (string check)
```

Allows accessing or deleting any user's memory profile if token contains role string "clinician" or "admin" without enum validation.

---

### HIGH-077: Service-to-Service Auth Without User Verification
**File:** [api.py](../services/safety_service/src/api.py)
**Lines:** 212-244
**Severity:** HIGH

```python
@router.post("/check")
async def perform_safety_check(
    request: SafetyCheckRequest,
    service: AuthenticatedService = Depends(get_current_service),  # SERVICE auth only
):
    # Doesn't verify user_id in request matches authenticated user
    # Orchestrator can check safety for ANY user without that user's consent
```

---

### HIGH-078: Token Blacklist In-Memory & Optional
**File:** [auth.py](../src/solace_security/auth.py)
**Lines:** 206-223, 374-387
**Severity:** HIGH

```python
async def revoke_token(self, token: str) -> bool:
    if not self._blacklist:
        logger.warning("token_revocation_unavailable")
        return False  # SILENTLY FAILS - caller thinks it worked
```

`InMemoryTokenBlacklist` loses all revoked tokens on server restart. Revoked refresh tokens can be replayed after any restart.

---

### HIGH-079: Invalid Role Silently Ignored
**File:** [middleware.py](../src/solace_security/middleware.py)
**Lines:** 71-81
**Severity:** HIGH

```python
def _get_resolved_permissions(self) -> set[str]:
    for role_name in self.roles:
        try:
            role = Role(role_name)
            role_perms = ROLE_PERMISSIONS.get(role, set())
            all_perms |= {p.value for p in role_perms}
        except ValueError:
            continue  # SILENTLY SKIPS INVALID ROLES - no logging
    return all_perms
```

---

### HIGH-080: Detailed Health Endpoint Leaks Info Without Auth
**File:** [api.py](../services/orchestrator_service/src/api.py)
**Lines:** 247-258
**Severity:** HIGH

```python
@router.get("/health/detailed")
async def detailed_health():
    # NO authentication - exposes active session counts, agent configs
```

---

### HIGH-081: Failed Service Identity Not Logged as Security Event
**File:** [service_auth.py](../src/solace_security/service_auth.py)
**Lines:** 355-373
**Severity:** HIGH

```python
def _get_service_identity(self, service_name: str):
    # Raises ValueError for unknown services
    # But doesn't log the attempted attack or send security alert
```

---

### MEDIUM-050 through MEDIUM-054: Various Permission Issues
- **MEDIUM-050:** Token cache doesn't re-check expiry during use ([service_auth.py](../src/solace_security/service_auth.py):276-292)
- **MEDIUM-051:** Cookie fallback auth source not logged for audit ([auth_plugin.py](../infrastructure/api_gateway/auth_plugin.py):226-232)
- **MEDIUM-052:** In-memory `_revoked_tokens` set not persistent ([auth_plugin.py](../infrastructure/api_gateway/auth_plugin.py):160)
- **MEDIUM-053:** Missing audit logging on all 403 responses across services
- **MEDIUM-054:** Safety `/status` endpoint exposes service statistics without auth ([api.py](../services/safety_service/src/api.py):352)

---

## ORCHESTRATOR SERVICE

### CRITICAL-062: Memory Node Calls Non-Existent Method
**File:** [memory_node.py](../services/orchestrator_service/src/langgraph/memory_node.py)
**Lines:** 108-120
**Severity:** CRITICAL (Runtime Crash)

```python
response = await self._client.post(  # WRONG - MemoryServiceClient has no .post()
    "/context",
    data={...}
)
# Should be:
# response = await self._client.assemble_context(user_id=..., session_id=..., ...)
```

`MemoryServiceClient` (in clients.py) exposes `assemble_context()`, not a raw `post()` method. Every memory retrieval request will crash with `AttributeError`.

---

### CRITICAL-063: Memory Node Uses Wrong AgentType
**File:** [memory_node.py](../services/orchestrator_service/src/langgraph/memory_node.py)
**Lines:** 161, 198
**Severity:** CRITICAL

```python
agent_result = AgentResult(
    agent_type=AgentType.AGGREGATOR,  # WRONG - should be memory-specific type
    success=success,
```

Memory operations tagged as `AGGREGATOR` confuse metrics, routing decisions, and downstream logic.

---

### HIGH-082: Safety Agent Returns None Content for Non-Crisis
**File:** [safety_agent.py](../services/orchestrator_service/src/agents/safety_agent.py)
**Line:** 199
**Severity:** HIGH

```python
response_content = None
if crisis_detected:
    response_content = f"I'm here with you..."
agent_result = AgentResult(..., response_content=response_content, ...)  # None for non-crisis!
```

Aggregator receives `None` for `response_content` on non-crisis paths. Will fail if aggregator tries to concatenate or process response strings.

---

### HIGH-083: Hardcoded Crisis Response Bypasses Safety Service
**File:** [graph_builder.py](../services/orchestrator_service/src/langgraph/graph_builder.py)
**Lines:** 196-198
**Severity:** HIGH

```python
response_content = (
    f"I'm here with you and I hear how difficult things are right now. "
    f"Your safety matters most.{resources_text}\n"
    "Would you like to talk about what you're experiencing?"
)
```

Local stub crisis handler generates a generic response without calling safety service. No context-specific safety assessment, no escalation workflow triggered.

---

### HIGH-084: ChatAgent.process() Is Synchronous in Async Graph
**File:** [chat_agent.py](../services/orchestrator_service/src/agents/chat_agent.py)
**Lines:** 346-377
**Severity:** HIGH

```python
def process(self, state: OrchestratorState) -> dict[str, Any]:  # NOT async
    agent = ChatAgent()
    return agent.process(state)  # Blocks event loop
```

Synchronous function used as LangGraph node. Blocks the async event loop for every message, killing concurrency.

---

### MEDIUM-055: EventBus Creates New List on Every Trim
**File:** [events.py](../services/orchestrator_service/src/events.py)
**Lines:** 158-182
**Severity:** MEDIUM

```python
self._event_history.append(event)
if len(self._event_history) > self._max_history:
    self._event_history = self._event_history[-self._max_history:]  # New list every trim
```

Should use `collections.deque(maxlen=...)` instead.

---

### MEDIUM-056 through MEDIUM-059: Various Orchestrator Issues
- **MEDIUM-056:** Local agent stubs defined but never used when `use_local_agents=True` ([graph_builder.py](../services/orchestrator_service/src/langgraph/graph_builder.py):86-94)
- **MEDIUM-057:** UUID validation raises instead of graceful degradation ([memory_node.py](../services/orchestrator_service/src/langgraph/memory_node.py):101-105)
- **MEDIUM-058:** Async exceptions not properly propagated in error path ([memory_node.py](../services/orchestrator_service/src/langgraph/memory_node.py):86-89)
- **MEDIUM-059:** API path fragility between memory client and memory service ([clients.py](../services/orchestrator_service/src/infrastructure/clients.py):290-306)

---

## MEMORY SERVICE

### CRITICAL-064: Unbounded Memory Growth in All Tiers
**File:** [service.py](../services/memory_service/src/domain/service.py)
**Lines:** 37-41, 313-323
**Severity:** CRITICAL (Production Crash)

```python
self._tier_3_session: dict[UUID, list[MemoryRecord]] = {}  # No limit
self._tier_4_episodic: dict[UUID, list[MemoryRecord]] = {}  # No limit
self._tier_5_semantic: dict[UUID, list[MemoryRecord]] = {}  # No limit

def _store_to_tier(self, record: MemoryRecord, tier: str) -> None:
    storage.setdefault(record.user_id, []).append(record)  # Grows forever
```

Only working memory (tier 2) has a 20-item limit. Tiers 3-5 grow unbounded. With enough sessions, will exhaust server RAM.

---

### CRITICAL-065: Consolidation Crashes When Pipeline Is None
**File:** [service.py](../services/memory_service/src/domain/service.py)
**Lines:** 176-178, 220
**Severity:** CRITICAL

```python
if include_summary and self._consolidation_pipeline:  # Checks None here
    summary_result = await self._consolidation_pipeline.generate_summary(...)

if trigger_consolidation and self._settings.enable_auto_consolidation:
    await self.consolidate(...)  # Does NOT check if pipeline is None → crash
```

---

### HIGH-085: Semantic Filter Is Just Substring Match
**File:** [service.py](../services/memory_service/src/domain/service.py)
**Lines:** 336-338
**Severity:** HIGH

```python
def _semantic_filter(self, records: list[MemoryRecord], query: str) -> list[MemoryRecord]:
    """Basic semantic filtering (placeholder for vector search)."""
    return [r for r in records if query.lower() in r.content.lower()]
```

Named "semantic filter" but does literal substring matching. Will miss synonyms, concepts, and related context entirely. Breaks the RAG pipeline quality.

---

### HIGH-086: Safety Context Can Get Zero Tokens
**File:** [context_assembler.py](../services/memory_service/src/domain/context_assembler.py)
**Lines:** 142-156, 323-325
**Severity:** HIGH

```python
safety = min(self._settings.safety_context_budget, available // 16)  # Can be 0
```

In tight token budgets, critical safety context (risk factors, crisis history) gets allocated 0 tokens. Safety information silently dropped from context assembly.

---

### HIGH-087: Token Budget Can Go Negative
**File:** [context_assembler.py](../services/memory_service/src/domain/context_assembler.py)
**Lines:** 142-156
**Severity:** HIGH

```python
remaining = available - (system + safety + profile + therapeutic + recent + retrieved)
return TokenAllocation(
    remaining=max(0, remaining),  # Clamped but actual assembly may exceed budget
)
```

If individual component allocations sum to more than available, the assembly still proceeds. Context sent to LLM may exceed token limits.

---

### MEDIUM-060: Token Estimation Inaccurate
**File:** [context_assembler.py](../services/memory_service/src/domain/context_assembler.py)
**Lines:** 323-325
**Severity:** MEDIUM

```python
def _estimate_tokens(self, text: str) -> int:
    return max(1, len(text) // self._settings.chars_per_token)
```

`chars/4` approximation is inaccurate for non-English text, code, or special characters. Can be off by 2-3x.

---

## NOTIFICATION SERVICE

### HIGH-088: Hardcoded Fallback Email Addresses
**File:** [consumers.py](../services/notification-service/src/consumers.py)
**Lines:** 316, 354, 365, 401
**Severity:** HIGH

```python
NotificationRecipient(email="oncall@solace-ai.com", name="On-Call Team")
email=f"clinician-{clinician_id}@solace-ai.com"   # Fabricated email
NotificationRecipient(email="escalations@solace-ai.com")
NotificationRecipient(email="monitoring@solace-ai.com")
```

Internal team emails hardcoded in source. The `clinician-{id}@...` pattern fabricates email addresses from user IDs.

---

### HIGH-089: Google Auth Hard Failure With No Fallback
**File:** [channels.py](../services/notification-service/src/domain/channels.py)
**Lines:** 427-432
**Severity:** HIGH

```python
try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
except ImportError:
    raise DeliveryError(...)  # HARD FAILURE - push notifications completely broken
```

If `google-auth` library not installed, push notifications crash instead of degrading to email/SMS.

---

### HIGH-090: Placeholder Clinician Email Construction
**File:** [consumers.py](../services/notification-service/src/consumers.py)
**Line:** 354
**Severity:** HIGH

```python
email=f"clinician-{clinician_id}@solace-ai.com"
```

Crisis escalation emails sent to fabricated addresses. Clinician never receives notification. HIPAA violation for safety-critical alerts.

---

### MEDIUM-061: Variable Shadowing in Channel Status
**File:** [api.py](../services/notification-service/src/api.py)
**Line:** 431
**Severity:** MEDIUM

```python
from fastapi import status  # Imported

# Later in list comprehension:
for channel_type, status in channels  # Shadows the import
```

---

### MEDIUM-062: SMS Truncation Without Warning
**File:** [channels.py](../services/notification-service/src/domain/channels.py)
**Lines:** 356-357
**Severity:** MEDIUM

```python
sms_body = f"{subject}\n\n{body}" if subject else body
sms_body = sms_body[:1600]  # Silent truncation - safety info may be lost
```

---

### MEDIUM-063: Emoji in Crisis Alert Templates
**File:** [templates.py](../services/notification-service/src/domain/templates.py)
**Lines:** 268-271
**Severity:** MEDIUM

```python
subject_template="🚨 CRISIS DETECTED [{{ crisis_level }}] - User {{ user_id }}"
```

Emojis may be stripped by email clients, reducing urgency signaling for safety-critical crisis alerts.

---

### MEDIUM-064: Event Publisher Class Method Issue
**File:** [events.py](../services/notification-service/src/events.py)
**Line:** 268
**Severity:** MEDIUM

Method binding issue in `NotificationEventPublisher.get_recent_events` class.

---

## ANALYTICS SERVICE

### CRITICAL-066: Missing Relative Import Breaks Module Loading
**File:** [repository.py](../services/analytics-service/src/repository.py)
**Line:** 17
**Severity:** CRITICAL

```python
from models import TableName, AnalyticsEvent, MetricRecord, AggregationRecord
# Should be:
from .models import TableName, AnalyticsEvent, MetricRecord, AggregationRecord
```

Will cause `ModuleNotFoundError` when running as a Python package.

---

### HIGH-091: SQL Injection in LIMIT Clause
**File:** [repository.py](../services/analytics-service/src/repository.py)
**Lines:** 179-191, 213-214
**Severity:** HIGH

```python
query = f"""SELECT ... FROM {TableName.EVENTS.value}
    WHERE timestamp >= %(start_time)s AND timestamp < %(end_time)s"""
# Parameterized above, but then:
query += f" ORDER BY timestamp DESC LIMIT {limit}"  # NOT parameterized!
```

`limit` parameter directly interpolated into SQL string. While ClickHouse parameters are used for timestamps, LIMIT bypasses them. Appears in both `query_events` and `query_metrics`.

---

### HIGH-092: Event Handler Dispatch Logic Inverted
**File:** [consumer.py](../services/analytics-service/src/consumer.py)
**Line:** 210
**Severity:** HIGH

```python
handlers = self._handlers.get(event.event_type, []) or self._default_handlers
# Empty list [] is falsy → falls through to default handlers even when specific handlers registered
# Should be:
handlers = self._handlers.get(event.event_type) or self._default_handlers
```

---

### HIGH-093: Unsafe UUID Parsing Masks Data Corruption
**File:** [consumer.py](../services/analytics-service/src/consumer.py)
**Lines:** 65-71
**Severity:** HIGH

```python
event_id=UUID(str(metadata.get("event_id", "00000000-0000-0000-0000-000000000000")))
user_id=UUID(str(data.get("user_id", "00000000-0000-0000-0000-000000000000")))
```

Invalid/missing UUIDs silently become nil UUIDs. Upstream data corruption goes completely undetected.

---

### HIGH-094: Missing Event Category Handlers
**File:** [consumer.py](../services/analytics-service/src/consumer.py)
**Lines:** 228-242
**Severity:** HIGH

```python
async def _route_to_aggregator(self, event: AnalyticsEvent) -> None:
    if event.category == EventCategory.SESSION:
        await self._handle_session_event(event)
    elif event.category == EventCategory.SAFETY:
        await self._handle_safety_event(event)
    elif event.category == EventCategory.THERAPY:
        await self._handle_therapy_event(event)
    elif event.category == EventCategory.DIAGNOSIS:
        await self._handle_diagnosis_event(event)
    # MISSING: MEMORY, PERSONALITY, SYSTEM → silently dropped
```

---

### HIGH-095: Mutable Defaults in Auth Fallback Dataclass
**File:** [api.py](../services/analytics-service/src/api.py)
**Lines:** 62-73
**Severity:** HIGH

```python
@_dataclass
class AuthenticatedUser:
    user_id: str
    token_type: str = "access"
    roles: list = None       # Should be field(default_factory=list)
    permissions: list = None  # Should be field(default_factory=list)
```

---

### MEDIUM-065 through MEDIUM-069: Various Analytics Issues
- **MEDIUM-065:** `period_map` stores method references instead of calling them ([api.py](../services/analytics-service/src/api.py):418-424)
- **MEDIUM-066:** Percentile calculation uses floor instead of proper interpolation ([aggregations.py](../services/analytics-service/src/aggregations.py):189-191)
- **MEDIUM-067:** Batch timing metric divided by batch size inside loop ([consumer.py](../services/analytics-service/src/consumer.py):396-397)
- **MEDIUM-068:** CountAggregator counts None values ([aggregations.py](../services/analytics-service/src/aggregations.py):160-161)
- **MEDIUM-069:** Masked import errors in main.py ([main.py](../services/analytics-service/src/main.py):216-222)

---

## PHASE 6: DEPLOYMENT & CONFIGURATION

### CRITICAL-067: .env File With Secrets Committed to Repository
**File:** [.env](../.env)
**Lines:** 1-9
**Severity:** CRITICAL

```
JWT_SECRET_KEY=test_mvp_secret_key_for_development_only_minimum_32_characters_long
ENVIRONMENT=development
DEBUG=true
```

Even marked as "development only," this file is tracked in git. Anyone with repo access has the JWT secret. Must be added to `.gitignore`.

---

### HIGH-096: All CI/CD Pipeline Steps Disabled
**File:** [ci.yml](../.github/workflows/ci.yml)
**Lines:** 29-138 (all commented out)
**Severity:** HIGH

```yaml
#   typecheck:       # DISABLED - no mypy
#   test:            # DISABLED - no pytest
#   security-scan:   # DISABLED - no bandit SAST
#   docker-build:    # DISABLED - no image verification
```

Only the lint step is active. No type checking, testing, security scanning, or Docker build verification runs in CI.

---

### HIGH-097: Hardcoded Default Secrets in docker-compose.yml
**File:** [docker-compose.yml](../docker-compose.yml)
**Lines:** 25, 170, 195-196, 201
**Severity:** HIGH

```yaml
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-solace_dev_password}
GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
USER_SERVICE_JWT_SECRET_KEY: ${JWT_SECRET_KEY:-dev-jwt-secret-key-minimum-32-chars}
USER_SERVICE_FIELD_ENCRYPTION_KEY: ${FIELD_ENCRYPTION_KEY:-dev-encryption-key-32-characters}
```

All secrets have weak fallback defaults. If env vars not set, deployment uses insecure credentials.

---

### HIGH-098: 6 Services Missing requirements.txt
**Severity:** HIGH

Missing files:
- `services/diagnosis_service/requirements.txt`
- `services/memory_service/requirements.txt`
- `services/orchestrator_service/requirements.txt`
- `services/personality_service/requirements.txt`
- `services/safety_service/requirements.txt`
- `services/therapy_service/requirements.txt`

Dockerfiles use `if [ -f requirements.txt ]` conditional, so builds silently skip dependency installation. Services inherit ALL root dependencies (bloated containers).

---

### HIGH-099: Loose Version Constraints in Root requirements.txt
**File:** [requirements.txt](../requirements.txt)
**Severity:** HIGH

```
fastapi>=0.128.0         # Accepts any future version
uvicorn[standard]>=0.34.0
langgraph>=1.0.3
langchain>=0.3.14
sqlalchemy[asyncio]>=2.1.0
asyncpg>=0.30.0
redis>=5.2.0
```

Using `>=` without upper bounds. Major version bumps (e.g., SQLAlchemy 3.x) could break everything.

---

### HIGH-100: Pre-commit CI Skips Type Checking and Tests
**File:** [.pre-commit-config.yaml](../.pre-commit-config.yaml)
**Lines:** 73-75
**Severity:** HIGH

```yaml
ci:
  skip:
    - mypy
    - pytest-check
```

The two most critical validation steps are explicitly skipped in CI.

---

### HIGH-101: Uvicorn Version Inconsistency
**Severity:** HIGH

```
Root requirements.txt:      uvicorn[standard]>=0.34.0  (loose)
analytics-service:          uvicorn[standard]==0.40.0  (pinned)
notification-service:       uvicorn[standard]==0.40.0  (pinned)
Other services:             (no uvicorn - inherit root)
```

Non-deterministic builds. Root could install 0.34.0 while services expect 0.40.0.

---

### MEDIUM-070: Weaviate Anonymous Access Enabled
**File:** [docker-compose.yml](../docker-compose.yml)
**Lines:** 108-129
**Severity:** MEDIUM

```yaml
AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
```

---

### MEDIUM-071: Kafka Auto-Create Topics Enabled
**File:** [docker-compose.yml](../docker-compose.yml)
**Lines:** 77-106
**Severity:** MEDIUM

```yaml
KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
```

Typos in topic names silently create orphaned topics instead of failing.

---

### MEDIUM-072: Archive Dockerfile Uses Python 3.10
**File:** [archive/Dockerfile](../archive/Dockerfile)
**Lines:** 1-10
**Severity:** MEDIUM

```dockerfile
FROM python:3.10-slim  # Project uses 3.12; 3.10 EOL June 2026
```

---

### MEDIUM-073: Inconsistent Dockerfile Patterns
**Severity:** MEDIUM

- `safety_service`: Copies `pyproject.toml` in builder (only service doing this)
- `notification-service`, `user-service`: Use `requirements.txt` (not `requirements*.txt`)
- Other services: Use `requirements*.txt` glob pattern

---

### MEDIUM-074: Debug Mode in .env
**File:** [.env](../.env)
**Line:** 8
**Severity:** MEDIUM

```
DEBUG=true
```

If .env is loaded in non-dev environments, debug mode exposes stack traces.

---

### MEDIUM-075: Ruff Ignores Hardcoded Password Detection
**File:** [pyproject.toml](../pyproject.toml)
**Lines:** 114-120
**Severity:** MEDIUM

```toml
ignore = [
    "S105",   # hardcoded password (globally disabled)
    "S106",   # hardcoded password in argument default (globally disabled)
]
```

Real hardcoded secrets will not be flagged by linter.

---

### LOW-011 through LOW-013: Minor Config Issues
- **LOW-011:** No dependency lock file (no `requirements.lock` or `poetry.lock`)
- **LOW-012:** PyYAML uses `>=6.0.2` without upper bound
- **LOW-013:** Inconsistent password env var naming (`POSTGRES_PASSWORD` vs `DB_PASSWORD`)

---

## Priority Remediation Plan

### Immediate (Week 1) - HIPAA Blockers
1. Add authentication to ALL unauthenticated endpoints (CRITICAL-056 through 058)
2. Remove .env from version control, add to .gitignore (CRITICAL-067)
3. Remove hardcoded JWT secret default - require env var (CRITICAL-059)
4. Fix `additional_permissions` validation in service tokens (CRITICAL-061)
5. Fix memory node's wrong method call (CRITICAL-062)

### Urgent (Week 2) - Security Hardening
6. Replace all string-based role checks with Role enum validation (HIGH-074)
7. Add auth to therapy service endpoints (HIGH-075)
8. Fix SQL injection in analytics LIMIT clause (HIGH-091)
9. Fix event handler dispatch logic (HIGH-092)
10. Fix missing relative import in analytics repository (CRITICAL-066)

### High Priority (Weeks 3-4) - Stability
11. Re-enable all CI/CD pipeline steps (HIGH-096)
12. Create service-level requirements.txt files (HIGH-098)
13. Pin all dependency versions (HIGH-099)
14. Fix unbounded memory growth in memory service (CRITICAL-064)
15. Implement persistent token blacklist (HIGH-078)

### Medium Priority (Weeks 5-6) - Quality
16. Implement real semantic search for memory service (HIGH-085)
17. Fix safety context zero-token allocation (HIGH-086)
18. Replace hardcoded email addresses with config (HIGH-088)
19. Make ChatAgent async (HIGH-084)
20. Fix remaining medium-severity issues

---

## Cross-Phase Pattern Analysis

### Systemic Issues Identified Across All Phases

| Pattern | Occurrences | Phases |
|---------|-------------|--------|
| Missing authentication on endpoints | 15+ | 5 |
| String-based role checks (not enum) | 10+ | 5 |
| In-memory stores as defaults (HIPAA) | 5 | 1, 2, 5 |
| `any` instead of `Any` type | 12 | 1 |
| Infinite recursion in `_acquire()` | 5 | 1 |
| Silent import failure masking | 4 | 3, 6 |
| Hardcoded credentials | 8 | 5, 6 |
| Unbounded collection growth | 3 | 5 |
| Missing error handling on async ops | 6 | 3, 5 |
| Stub implementations masquerading as real | 8 | 3, 4, 5 |
