# LangGraph Orchestrator Pipeline — Architectural Audit

**Date:** February 20, 2026  
**Scope:** Deep review of the LangGraph graph builder, state schema, supervisor, aggregator, memory node, and all agent implementations  
**Excludes:** All issues already reported in C1–C16, H1–H36, M1–M38 of `FULL_CODEBASE_AUDIT.md`

---

## Summary

| Severity | Count | Key Themes |
|----------|-------|------------|
| **CRITICAL** | 3 | Supervisor sync node in async graph, metadata last-writer-wins silently corrupts supervisor decisions, safety agent URL collision with diagnosis service |
| **HIGH** | 8 | Parallel fan-out state merge conflicts, personality_style overwritten in parallel, memory node leaks client connections, agents retry 4xx errors, missing escalation trigger, crisis handler path skips aggregator messages |
| **MEDIUM** | 9 | Agents create new HTTP clients per request, aggregator includes non-response agent results, safety postcheck replacements can break mid-word, supervisor decision_count always 1, no timeout enforcement on agent nodes, etc. |
| **LOW** | 5 | Dead enum members, inconsistent confidence defaults, unused Decimal import, etc. |

**Total NEW findings: 25**

---

## CRITICAL Issues

### P1. Supervisor `process()` Is Sync-Incompatible With LangGraph Parallel Fan-Out — State Mutation Race
**File:** [services/orchestrator_service/src/langgraph/supervisor.py](services/orchestrator_service/src/langgraph/supervisor.py#L296-L299)  
**Severity:** CRITICAL

`SupervisorAgent.process()` is `async def` and mutates `self._decision_count += 1` (line 297). The graph builder creates one `SupervisorAgent` instance at build time (line 316 of graph_builder.py) and reuses it across all concurrent graph invocations. Since Python `int` increment is not atomic under asyncio, two concurrent requests can read the same value, increment, and write, losing a count. More critically, this means **a single `SupervisorAgent` instance is shared across all concurrent user requests** — any future addition of instance-level state (e.g., caching, session tracking) will create cross-user data leakage.

**Impact:** Cross-request state leakage; race condition on `_decision_count`. Unlike the other agents (which are instantiated per-request), the supervisor is instantiated once.

**Fix:** Either instantiate `SupervisorAgent` per invocation (like all other agents), or make it stateless. Change graph_builder.py line 316 to register a wrapper function:
```python
supervisor = SupervisorAgent(self._supervisor_settings, llm_client=_supervisor_llm_client)
builder.add_node("supervisor", supervisor.process)
# Change to:
async def supervisor_node_fn(state):
    return await SupervisorAgent(self._supervisor_settings, llm_client=_supervisor_llm_client).process(state)
builder.add_node("supervisor", supervisor_node_fn)
```

---

### P2. Safety Agent and Diagnosis Agent Default to Same Port — Silent Routing Collision
**File:** [services/orchestrator_service/src/agents/safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L50) and [services/orchestrator_service/src/agents/diagnosis_agent.py](services/orchestrator_service/src/agents/diagnosis_agent.py#L53)  
**Severity:** CRITICAL

Both `SafetyAgentSettings.service_url` and `DiagnosisAgentSettings.service_url` default to `http://localhost:8002`. The orchestrator config in [config.py](services/orchestrator_service/src/config.py#L33-L35) shows `safety_service_url = 8002` and `diagnosis_service_url = 8004`. But the agents **don't use** the centralized `ServiceEndpoints` — they each have their own settings class reading from different env vars (`ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL` vs `ORCHESTRATOR_DIAGNOSIS_AGENT_SERVICE_URL`). If these env vars are not explicitly set, both agents hit port 8002 (the safety service), meaning **diagnosis requests silently go to the safety service** and either fail with 404 or return incorrect data.

**Impact:** In any environment without explicit env var configuration, diagnosis agent calls the wrong service entirely.

**Fix:** Either wire the agents to use `ServiceEndpoints` from the centralized config, or change the default URL in `DiagnosisAgentSettings` to `http://localhost:8004`.

---

### P3. `metadata` Field Uses Last-Writer-Wins — Supervisor Decisions Silently Overwritten by Parallel Agents
**File:** [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L259)  
**Severity:** CRITICAL

The existing audit (H21) notes that `metadata` has no reducer. The impact is worse than reported: the supervisor writes `supervisor_decision` into `metadata` (supervisor.py line 298), then the aggregator writes `aggregation` into `metadata` (aggregator.py line 249). Because there's no merge reducer, the aggregator's write **completely replaces** the supervisor's metadata — the `supervisor_decision` key is lost. If diagnosis or therapy agents also return `metadata` (they do — diagnosis_agent.py lines 327-331, therapy_agent.py line 302), those parallel writes race and only one survives.

This matters because:
1. The supervisor's routing decision is lost from final state — no audit trail of why agents were chosen
2. Diagnosis metadata (`existing_symptoms`, `differential`) races with therapy metadata (`assigned_homework`)
3. The API returns `result_state.get("metadata", {})` which will be incomplete/wrong

**Impact:** Lost diagnostic data, lost supervisor decisions, lost homework assignments depending on which agent completes last. No deterministic behavior.

**Fix:** Add a proper merge reducer for `metadata`:
```python
def merge_metadata(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged

class OrchestratorState(TypedDict, total=False):
    metadata: Annotated[dict[str, Any], merge_metadata]
```

---

## HIGH Issues

### P4. Parallel Agents Overwrite `personality_style` — Last Writer Wins
**File:** [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L262)  
**Severity:** HIGH

`personality_style: dict[str, Any]` has no reducer annotation. In a parallel fan-out where both `personality_agent` and `chat_agent` run simultaneously, the personality agent writes the style dict. But if `chat_agent` completes after `personality_agent`, LangGraph's default last-writer-wins overwrites it with whatever `chat_agent` returns (which doesn't include `personality_style`, but the merge semantics may zero it out or keep the old value depending on LangGraph version). More importantly, if any agent inadvertently returns a `personality_style` key, it silently replaces the personality agent's carefully computed result.

**Impact:** Personality adaptation may be silently lost, reverting to defaults.

**Fix:** Add a reducer that preserves the most-confident personality style, or use `operator.or_` as a simple merge.

---

### P5. Memory Node Creates New `MemoryServiceClient` Per Invocation — Leaks HTTP Connections
**File:** [services/orchestrator_service/src/langgraph/memory_node.py](services/orchestrator_service/src/langgraph/memory_node.py#L200-L204)  
**Severity:** HIGH

`memory_retrieval_node()` creates a new `MemoryRetrievalNode()` per call, which creates a new `MemoryServiceClient()` (line 55), which creates a new `BaseServiceClient` with `self._client = None` (lazy-initialized `httpx.AsyncClient`). Each invocation creates and never closes the client. The `close()` method is never called.

This is the same pattern as all other agent node functions — but the memory node's `MemoryServiceClient` (from `infrastructure/clients.py`) creates a **persistent** `httpx.AsyncClient` lazily in `_get_client()` that is never closed. Over many requests, this leaks TCP connections.

**Impact:** TCP connection leak. Under sustained load, eventually hits OS file descriptor limits.

**Fix:** Either share a single `MemoryServiceClient` instance (created at startup), or ensure the client is created with `async with` for each request.

---

### P6. All Agent Service Clients Retry on 4xx Errors (Diagnosis, Therapy, Personality, Safety)
**File:** [services/orchestrator_service/src/agents/diagnosis_agent.py](services/orchestrator_service/src/agents/diagnosis_agent.py#L185-L202), [therapy_agent.py](services/orchestrator_service/src/agents/therapy_agent.py#L162-L178), [personality_agent.py](services/orchestrator_service/src/agents/personality_agent.py#L201-L222), [safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L174-L181)  
**Severity:** HIGH

The agent-level HTTP clients (separate from `infrastructure/clients.py`) catch `httpx.HTTPStatusError` for **all** status codes and retry. A 400 Bad Request, 401 Unauthorized, 403 Forbidden, or 404 Not Found will be retried up to `max_retries` times. These are non-transient errors — retrying is wasteful and delays the response by `max_retries * timeout_seconds` (up to 45 seconds for diagnosis).

Note: the `infrastructure/clients.py` `BaseServiceClient._request()` correctly only retries on 5xx, but agents don't use it — they have their own `httpx.AsyncClient` logic.

**Impact:** Non-transient errors cause unnecessary retry delays (up to 3× the timeout). 

**Fix:** Only retry on 5xx and connection errors:
```python
except httpx.HTTPStatusError as e:
    if e.response.status_code < 500:
        raise  # Don't retry client errors
```

---

### P7. Safety Agent Never Triggers Escalation — `trigger_escalation()` Is Dead Code
**File:** [services/orchestrator_service/src/agents/safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L185-L205)  
**Severity:** HIGH

`SafetyServiceClient.trigger_escalation()` is defined (line 195) but **never called** anywhere. When the safety agent detects `requires_escalation=True`, it sets the flag in `SafetyFlags` but never actually calls the escalation endpoint. The `_build_state_update` method (line 207) just records the data. The crisis handler node in `graph_builder.py` generates a message but also never triggers escalation.

For a mental health platform, failing to trigger an actual escalation workflow when Required is a safety-critical gap.

**Impact:** Detected crises that require human intervention are logged but never actually escalated.

**Fix:** Call `self._client.trigger_escalation()` when `result.requires_escalation` is True in `_build_state_update()`.

---

### P8. Crisis Handler Path Exits Without Adding to `agent_results` Properly — Loses Precheck Results
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L140-L171)  
**Severity:** HIGH

When crisis is detected in the precheck, the flow goes `safety_precheck → crisis_handler → END`. The crisis handler returns `agent_results: [agent_result.to_dict()]` with a single SAFETY result. However, the precheck node **also** returns `agent_results`. Because `agent_results` has an append-reducer, the final state has both. This is correct.

But the issue is: the crisis handler edge goes `crisis_handler → END`, which means the response **skips** the `safety_postcheck` node entirely. While the crisis response itself is safe, this means:
1. No harmful content filtering applied (e.g., if crisis resources contain phrases matching `_HARMFUL_REPLACEMENTS`)
2. The `processing_phase` is set to `CRISIS_HANDLING` rather than `COMPLETED`, which the API doesn't check — it just reads `final_response`

More critically, when `route_after_safety_agent` (line 254) routes to `crisis_handler`, the safety agent is running in parallel with other agents (chat, therapy, etc.). The other agents also sent to the aggregator produce results that are **lost** — the `crisis_handler` output becomes the final response, but the system paid the cost of running all parallel agents for nothing.

**Impact:** Wasted compute; inconsistent processing_phase; no postcheck on crisis responses.

**Fix:** For the parallel fan-out case, consider routing all agents to aggregator which then checks safety flags and yields to crisis handler if needed. For the precheck case, the current topology is acceptable but should apply postcheck.

---

### P9.  `selected_agents` Has No Reducer — Parallel Updates Will Race
**File:** [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L256)  
**Severity:** HIGH

`selected_agents: list[str]` has no `Annotated[..., reducer]`. It's set by the supervisor node. Since only the supervisor writes it, this is currently safe. But `retrieved_memories`, `assembled_context`, `memory_sources`, `conversation_context`, `personality_style`, `active_treatment`, `final_response`, `error_message`, and `intent`/`intent_confidence` also lack reducers. Any field without a reducer that is written by more than one node in parallel will silently lose data.

Currently the graph topology routes only one node at a time to memory/supervisor/aggregator, so most of these are written by single nodes. But `final_response` is written by both `aggregator_node` and `crisis_handler_node`, and in the safety-agent-to-crisis-handler case, `aggregator` may also write `final_response` from other parallel agents. Whichever completes last wins.

**Impact:** Potential state corruption if graph topology changes or under race conditions.

**Fix:** Add no-op or merge reducers for all fields that could theoretically be written by multiple paths.

---

### P10. Aggregator Includes Precheck/Supervisor/Memory `agent_results` in Ranking
**File:** [services/orchestrator_service/src/langgraph/aggregator.py](services/orchestrator_service/src/langgraph/aggregator.py#L240-L245)  
**Severity:** HIGH

The aggregator reads `state.get("agent_results", [])` which includes **all** accumulated results from the entire pipeline: safety precheck, memory retrieval, supervisor, AND the actual agent responses. The `ResponseRanker.rank()` filters for results with non-empty `response_content`, so precheck/memory/supervisor results (which have `response_content=None`) are filtered out. 

However, the safety agent's `_build_state_update` (safety_agent.py line 271) **does** set `response_content` on crisis detection:
```python
response_content = f"I'm here with you and I hear how difficult things are..."
```

This means the safety agent's crisis response enters the aggregator's ranking alongside non-crisis chat/therapy responses. If the safety agent runs in parallel with other agents and detects a crisis, its response competes in the aggregator with normal responses. The safety agent has priority 1.0 + 0.3 boost, so it should win, but this is fragile — it relies on numerical tuning rather than an explicit "crisis overrides everything" path.

**Impact:** Crisis responses could theoretically be outranked or merged with non-crisis responses by the aggregator if priority tuning changes.

**Fix:** Add explicit crisis override in `_perform_aggregation`: if any contribution has `agent_type == SAFETY` and the safety flags indicate crisis, return only the safety response.

---

### P11. `httpx.AsyncClient(timeout=self._settings.timeout_seconds)` Passes Float to Timeout
**File:** [services/orchestrator_service/src/agents/diagnosis_agent.py](services/orchestrator_service/src/agents/diagnosis_agent.py#L181), [therapy_agent.py](services/orchestrator_service/src/agents/therapy_agent.py#L158), [personality_agent.py](services/orchestrator_service/src/agents/personality_agent.py#L202), [safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L172)  
**Severity:** HIGH

All four agent service clients pass a bare float to `httpx.AsyncClient(timeout=...)`. Since httpx 0.24+, this creates a `Timeout(timeout)` where all sub-timeouts (connect, read, write, pool) are set to the same value. This means the **pool acquire** timeout is the same as the read timeout. Under load, waiting for a connection from the pool uses up the same timeout budget as the actual HTTP call, leading to premature timeouts.

Worse: each agent creates a **new** `httpx.AsyncClient` per request with `async with httpx.AsyncClient(...)`. This defeats connection pooling entirely and creates + destroys TCP connections for every message. With 3-5 agents per user message and retry logic, a single user request can create 15+ TCP connections.

**Impact:** No connection reuse; excessive TCP connection churn; premature timeouts under load.

**Fix:** Use long-lived shared clients with explicit `httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)`.

---

## MEDIUM Issues

### P12. Safety Postcheck `_HARMFUL_REPLACEMENTS` Can Create Nonsensical Text
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L186-L211)  
**Severity:** MEDIUM

The replacements apply globally with word boundaries: `"give up"` → `"take a break and revisit this later"`, `"no point"` → `"it may not feel like it right now, but there is hope"`. Consider a therapeutic response like:

> "It takes courage not to **give up** when things feel hopeless."

This becomes:

> "It takes courage not to **take a break and revisit this later** when things feel hopeless."

The replacement is grammatically broken and semantically wrong. The filter doesn't understand negation context ("don't give up" is encouraging, not harmful).

**Impact:** Legitimate therapeutic encouragement is corrupted into nonsensical text.

**Fix:** Use a more context-aware approach — at minimum, skip replacement if preceded by negation words like "not", "don't", "never", "shouldn't".

---

### P13. Supervisor IntentClassifier Ignores `conversation_context` Despite Accepting It
**File:** [services/orchestrator_service/src/langgraph/supervisor.py](services/orchestrator_service/src/langgraph/supervisor.py#L140-L145)  
**Severity:** MEDIUM

`classify()` builds `full_context = f"{conversation_context} {message_lower}"` but then **never uses `full_context`** — all subsequent keyword matching operates on `message_lower` only. The conversation context (which now includes rich memory-retrieved context) is completely ignored for intent classification.

**Impact:** Intent classification is based solely on the current message, missing important context like ongoing therapy sessions, previous symptom discussions, etc. A follow-up message like "yes, that's been happening a lot" would be classified as `GENERAL_CHAT` instead of continuing the previous `SYMPTOM_DISCUSSION` intent.

**Fix:** Use `full_context` for keyword matching, or at least check previous intent from context to enable multi-turn intent tracking.

---

### P14. `route_to_agents` Maps Unknown Agent Types to `chat_agent` Silently
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L233-L246)  
**Severity:** MEDIUM

```python
agent_node_map = {
    "safety": "safety_agent",
    "chat": "chat_agent",
    "diagnosis": "diagnosis_agent",
    "therapy": "therapy_agent",
    "personality": "personality_agent",
}
node_name = agent_node_map.get(agent, "chat_agent")
```

If `AgentType` is extended with new values (e.g., `MEMORY`, `AGGREGATOR`, `SUPERVISOR` — which all exist in the enum), the `selected_agents` list could contain `"memory"` or `"aggregator"`. These silently map to `chat_agent`, meaning the wrong agent processes the request with no error indication.

**Impact:** Silent misrouting when the agent type enum is extended.

**Fix:** Raise an error or log a warning for unmapped agent types instead of silently defaulting.

---

### P15. Aggregator Overall Confidence Can Exceed 1.0
**File:** [services/orchestrator_service/src/langgraph/aggregator.py](services/orchestrator_service/src/langgraph/aggregator.py#L282-L290)  
**Severity:** MEDIUM

```python
def _calculate_overall_confidence(self, contributions):
    total_weight = sum(c.priority_score for c in contributions)
    weighted_confidence = sum(c.confidence * c.priority_score for c in contributions)
    if total_weight > 0:
        return weighted_confidence / total_weight
```

The `priority_score` for safety is `base_priority * 0.6 + confidence * 0.4 + safety_priority_boost(0.3)`. With `base_priority=1.0` and `confidence=1.0`: `1.0*0.6 + 1.0*0.4 + 0.3 = 1.3`, then `min(1.3, 1.0) = 1.0`. But `_calculate_priority` returns individual priority, and individual `confidence` values can be 1.0. Since this is a weighted average of confidences already ≤1.0, it's bounded — this is actually fine mathematically.

However, `ResponseRanker._calculate_priority` CAN return >1.0 before the `min()` clamp, but with the clamp it's bounded. The priority_score in contributions is clamped. So overall confidence is correct. Rescinding this — but noting that `priority_score` values are all clamped to 1.0 with different `_calculate_priority` inputs, which means **safety and therapy have the same final priority (1.0)** when confidence is high. The safety priority boost is wasted.

**Impact:** Safety agent priority boost provides no differentiation when confidence is ≥ 0.5. Safety and therapy agents tie at priority 1.0.

**Fix:** Increase the priority cap or restructure the calculation to ensure safety always wins.

---

### P16. `update_safety_flags` Reducer Has Missing ELEVATED in route_after_safety
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L222-L228)  
**Severity:** MEDIUM

`route_after_safety` checks `risk_level in ("HIGH", "CRITICAL")` to route to crisis handler. But `RiskLevel` (aliased from `CrisisLevel`) also includes `"ELEVATED"`. An elevated risk message **skips crisis handling** and goes through normal processing. Depending on the platform's clinical policy, ELEVATED risk may require at least modified handling (e.g., including crisis resources, enhanced monitoring) that the normal path doesn't provide. The `update_safety_flags` reducer correctly handles ELEVATED in the risk ordering, but the routing decision ignores it.

Similarly, `route_after_safety_agent` has the same gap at line 254.

**Impact:** Messages classified as ELEVATED risk receive no special treatment, routing through the same path as NONE/LOW risk.

**Fix:** Decide clinical policy for ELEVATED risk and implement accordingly. If it should include resources but not full crisis handling, add a conditional branch.

---

### P17. `_CrisisResourceManager.get_resources_for_level` Doesn't Handle ELEVATED
**File:** [services/orchestrator_service/src/langgraph/graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L79-L83)  
**Severity:** MEDIUM

```python
if level == "CRITICAL":
    return self._RESOURCES  # All resources including 911
if level == "HIGH":
    return self._RESOURCES[1:]  # Skip 911 for HIGH
return self._RESOURCES[1:3]  # 988 + Crisis Text Line for lower levels
```

For ELEVATED, LOW, and NONE, the same `self._RESOURCES[1:3]` is returned. If ELEVATED users reach the crisis handler (e.g., via `route_after_safety_agent`), they get minimal resources. More importantly, NONE risk gets crisis resources too, which would be confusing if ever triggered.

**Impact:** No differentiation between ELEVATED and NONE/LOW resource levels.

**Fix:** Return empty list for NONE/LOW; return `_RESOURCES[1:3]` for ELEVATED; full list for HIGH/CRITICAL.

---

### P18. Chat Agent LLM Response Bypasses `max_response_length` Enforcement
**File:** [services/orchestrator_service/src/agents/chat_agent.py](services/orchestrator_service/src/agents/chat_agent.py#L467-L480)  
**Severity:** MEDIUM

`_generate_llm_response` passes `max_tokens=self._settings.max_response_length` to the LLM. But `max_response_length` is in **characters** (default 500) while `max_tokens` is in **tokens**. 500 tokens ≈ 375 words ≈ 2000+ characters. The LLM may produce responses far exceeding the character limit, which the template path respects but the LLM path doesn't.

**Impact:** LLM-generated responses can be 3-4× longer than intended.

**Fix:** Either convert characters to approximate token count (`max_tokens = max_response_length // 4`), or truncate the LLM response to `max_response_length`.

---

### P19. Diagnosis Agent Passes Full Metadata to `_perform_assessment` — Potential Data Leakage
**File:** [services/orchestrator_service/src/agents/diagnosis_agent.py](services/orchestrator_service/src/agents/diagnosis_agent.py#L274-L277)  
**Severity:** MEDIUM

The diagnosis agent reads `metadata = state.get("metadata", {})` (line 270) and passes it to `_perform_assessment`. Due to the metadata last-writer-wins issue (P3), this metadata can contain data from other agents/users in a race condition. More concretely, `existing_symptoms` and `diagnosis_phase` are read from this shared metadata dict, which could contain stale or incorrect values from a previous request's parallel execution.

**Impact:** Diagnosis could operate on stale/incorrect symptom data from another agent's context.

**Fix:** Store diagnosis-specific state in a dedicated field (e.g., `diagnosis_context`) with its own reducer.

---

### P20. Therapy Agent Logs But Doesn't Use `assembled_context` 
**File:** [services/orchestrator_service/src/agents/therapy_agent.py](services/orchestrator_service/src/agents/therapy_agent.py#L243-L255)  
**Severity:** MEDIUM

The therapy agent accepts `assembled_context` from memory retrieval but just logs it:
```python
if assembled_context:
    logger.debug("therapy_request_with_memory_context", context_length=len(assembled_context))
```
The memory service assembled rich context (safety history, therapeutic progress, previous sessions) specifically so downstream agents could use it. The therapy agent ignores it, sending only the last 10 raw messages to the therapy service.

**Impact:** Therapy responses lack personalization from past session memory, defeating the purpose of the memory retrieval node.

**Fix:** Include `assembled_context` in the request payload to the therapy service, either as a dedicated field or prepended to conversation history.

---

## LOW Issues

### P21. `AgentType` Enum Has Members Never Used in Graph Routing
**File:** [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L40-L48)  
**Severity:** LOW

`AgentType.SUPERVISOR`, `AgentType.MEMORY`, and `AgentType.AGGREGATOR` exist in the enum. They're used as metadata labels in `AgentResult` but are never valid values in `selected_agents` or `route_to_agents`. Adding them to `selected_agents` would cause silent fallback to chat_agent (see P14).

---

### P22. `ProcessingMetadata` Dataclass Is Defined But Never Used
**File:** [services/orchestrator_service/src/langgraph/state_schema.py](services/orchestrator_service/src/langgraph/state_schema.py#L164-L195)  
**Severity:** LOW

`ProcessingMetadata` with `request_id`, `session_id`, `active_agents`, `completed_agents`, `retry_count`, `is_streaming` is defined, has full serialization, but is never instantiated anywhere in the codebase. The `metadata: dict[str, Any]` field on `OrchestratorState` serves the same purpose in an unstructured way.

---

### P23. Unused `Decimal` Import in Safety Agent
**File:** [services/orchestrator_service/src/agents/safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L9)  
**Severity:** LOW

`from decimal import Decimal` is imported and used in `SafetyCheckResult.risk_score` but the score is never read or compared against anything in the agent logic — it's immediately discarded after deserialization.

---

### P24. `AggregationStrategy.CONSENSUS` Is Never Used
**File:** [services/orchestrator_service/src/langgraph/aggregator.py](services/orchestrator_service/src/langgraph/aggregator.py#L28)  
**Severity:** LOW

The `CONSENSUS` strategy enum member exists but `_select_strategy` never returns it, and `ResponseMerger.merge()` doesn't handle it (would fall through to `_priority_based_merge`). Dead enum value.

---

### P25. Duplicate Crisis Keyword Lists Across Precheck, Safety Agent, and Supervisor
**File:** [graph_builder.py](services/orchestrator_service/src/langgraph/graph_builder.py#L112), [supervisor.py](services/orchestrator_service/src/langgraph/supervisor.py#L79-L83), [safety_agent.py](services/orchestrator_service/src/agents/safety_agent.py#L255-L265)  
**Severity:** LOW

Three separate hardcoded lists of crisis keywords exist in the orchestrator pipeline:
- `safety_precheck_node`: 10 crisis keywords + 6 high-risk keywords
- `SupervisorSettings.crisis_keywords`: 9 keywords (missing "end it all")
- `SafetyAgent._build_fallback_response`: 10 keywords (different set — includes "cutting", "overdose")

These lists diverge in content and there's no single source of truth. Adding a keyword to one list won't appear in the others.

**Fix:** Extract to a shared constant or configuration value used by all three components.

---

## Top Priority Fixes

| Priority | Issue | Impact |
|----------|-------|--------|
| 1 | **P2** — Fix safety/diagnosis agent port collision | Diagnosis is broken in default config |
| 2 | **P3** — Add merge reducer for `metadata` field | Supervisor decisions, diagnosis data lost |
| 3 | **P1** — Don't share SupervisorAgent across requests | Cross-request state leakage |
| 4 | **P7** — Wire escalation trigger when `requires_escalation=True` | Crisis escalation never happens |
| 5 | **P6** — Don't retry 4xx errors in agent HTTP clients | 3× delay on client errors |
| 6 | **P5/P11** — Share HTTP clients, fix connection lifecycle | Connection leaks, no pooling |
| 7 | **P10** — Add explicit crisis override in aggregator | Crisis response could be outranked |
| 8 | **P4/P9** — Add reducers for all dict/list fields | Parallel state merge corruption |
| 9 | **P13** — Use conversation context in intent classification | Poor multi-turn intent tracking |
| 10 | **P12** — Context-aware harmful content replacement | Therapeutic text corruption |
