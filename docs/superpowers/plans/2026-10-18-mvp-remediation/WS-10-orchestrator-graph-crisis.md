# WS-10: Orchestrator graph, crisis ingress and turn runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development as adapted in the master plan (no git commits; checkpoints through D:\tmp\mvp-exec\tools). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every chat message is screened at ingress by the shared lexicon and the safety service with a max-only, fail-safe fusion, so ELEVATED and above always returns in-band crisis resources and HIGH/CRITICAL publishes exactly one durable crisis event per turn, through a turn runner whose graph state is turn-scoped, serialized per thread and (from W2) checkpointed encrypted in Postgres.

**Architecture:** A new `langgraph/crisis_screen.py` wraps `solace_clinical.screen_text` (IF-07) and a pooled safety-service `/check` client (IF-06) behind `screen_message()`, fused with `fuse_risk`. `OrchestratorGraphBuilder.run_turn()` (IF-10) screens and publishes the crisis event before the graph, runs the graph under a per-thread lock and the turn deadline, and turns any graph, agent or LLM failure into a crisis-safe `TurnResult`; every graph input resets the turn-scoped channels with LangGraph `Overwrite`, and a bounded D-06 crisis watch replaces the old sticky OR-merge. From W2 the checkpointer is `AsyncPostgresSaver` over the slot-017 tables with LangGraph's `EncryptedSerializer` whose cipher is the IF-13 PHI codec (AAD bound to the conversation owner), plus TTL/prune and `delete_threads` for erasure.

**Covers:** 22 register entries (blocker 7, high 8, medium 5, low 2). **Waves:** W1 (phase B: WS-10.1 to WS-10.8); W2 (phase A: WS-10.9; phase B: WS-10.10 to WS-10.16). **Depends on:** WS-12 (IF-07 `solace_clinical` lexicon, `fuse_risk`, `crisis_resources`; IF-06 `/check` contract and the phrase set), WS-06 (IF-05 `ServiceEventPlane.publish`, `EventPublishError`), WS-07 (IF-04 `service_auth_headers(..., on_behalf_of=)`, `acting_as_user`), WS-09 (IF-15 `LLMUnavailableError`; IF-13 `PHICodec` for checkpoint payloads), WS-04 (IF-01 `ServiceRuntime`, `StartupDependencyError`, `is_strict_environment`, IF-18 metric objects), WS-02 (slot 017 scaffold and `solace_app` default privileges), WS-05 (IF-19 `pg_migrated` fixture; IF-20 contract helpers), WS-01 (lock with langgraph 1.0.x, langgraph-checkpoint-postgres, psycopg, psycopg-pool, PyYAML), WS-11 (slot 016 precedes slot 017 in the W2 schema lane; WS-11 owns `main.py` and makes the `from_runtime` call). **Owned paths:** `services/orchestrator_service/src/langgraph/{graph_builder,supervisor,state_schema,aggregator,__init__,crisis_screen,checkpointer}.py`, `services/orchestrator_service/src/agents/{safety_agent,chat_agent,emotion_agent}.py`, `services/orchestrator_service/src/events.py`, `services/orchestrator_service/src/response/`, `migrations/versions/017_orchestrator_checkpoints.py`; new tests under `services/orchestrator_service/tests/test_mvp_<id>_<slug>.py`, `tests/realinfra/orchestrator-graph-crisis/`, plus edits to existing test files that test WS-10 code (shared-tests protocol); the IF-20 consumer contract `tests/contracts/orchestrator__safety/check.json` (the consumer writes the contract file; the `/check` consumer is WS-10's `crisis_screen.py`). **Spec:** docs/MVP-ISSUE-REGISTER.md (entries per task) and the master plan docs/superpowers/plans/2026-10-18-mvp-remediation.md.

All commands run from `D:\Repo\Contextual-Chatbot` in PowerShell 5.1. Test data is synthetic; never print `.env` or key material.

## File map

| File | Create/Modify | Responsibility | Tasks |
|---|---|---|---|
| `services/orchestrator_service/src/langgraph/state_schema.py` | Modify | turn-scoped channels and `build_turn_input`, `SafetyFlags` risk fields, `degraded`/`crisis_watch` channels, the turn's `locale` key (D-29), bounded history; removes `CRISIS_KEYWORDS` | 10.1, 10.2, 10.5, 10.10 |
| `services/orchestrator_service/src/langgraph/crisis_screen.py` | Create | local lexicon screen, `SafetyCheckClient` (IF-06), `screen_message` fusion, `screen_reply` harmful-output rules | 10.2, 10.4, 10.8 |
| `services/orchestrator_service/src/langgraph/graph_builder.py` | Modify | precheck node, per-thread lock, `run_turn`, `TurnResult`, `CrisisInfo`, `TurnSettings`, `from_runtime`, crisis watch, postcheck, finalize node, `delete_threads`, `get_thread_messages` | 10.1, 10.2, 10.4, 10.5, 10.6, 10.7, 10.8, 10.10, 10.15 |
| `services/orchestrator_service/src/langgraph/supervisor.py` | Modify | lexicon crisis check, LLM intent on every non-crisis turn, crisis routing without personality, progress routing | 10.2, 10.7, 10.14 |
| `services/orchestrator_service/src/langgraph/aggregator.py` | Modify | stops writing assistant messages (finalize node owns history) | 10.15 |
| `services/orchestrator_service/src/langgraph/checkpointer.py` | Create | Postgres saver factory, owner-bound encrypted serializer, janitor (TTL + prune) | 10.10 |
| `services/orchestrator_service/src/langgraph/__init__.py` | Modify | export `TurnResult`, `CrisisInfo`, `TurnSettings` | 10.5 |
| `services/orchestrator_service/src/agents/safety_agent.py` | Modify | IF-06 client, `SERVICE_SAFETY_SERVICE_URL`, obo headers, shared-screen fallback, max-only merge | 10.4 |
| `services/orchestrator_service/src/agents/chat_agent.py` | Modify | thread history to the LLM, `LLMUnavailableError` handling, word-bounded topics | 10.11, 10.13 |
| `services/orchestrator_service/src/agents/emotion_agent.py` | Modify | negation-aware, count-based emotion scoring | 10.12 |
| `services/orchestrator_service/src/events.py` | Modify | `session_uuid`, `build_crisis_detected_event`, `publish_crisis_event`, `to_kafka_event` without fabricated fields, bare `solace_events` import | 10.3 |
| `services/orchestrator_service/src/response/safety_copy.py` | Create | crisis reply, safety footer, safe fallback and harmful-reply replacement copy (D-05 sign-off) | 10.5, 10.6, 10.8 |
| `migrations/versions/017_orchestrator_checkpoints.py` | Modify | LangGraph checkpoint DDL in `upgrade()`/`downgrade()` | 10.9 |
| `tests/contracts/orchestrator__safety/check.json` | Create | IF-11 orchestrator-to-safety contract (consumer file per IF-20) | 10.16 |
| `services/orchestrator_service/tests/test_mvp_*.py` | Create | per-task unit tests | all |
| `tests/realinfra/orchestrator-graph-crisis/test_mvp_066_*.py` | Create | real-infra slot and checkpointer tests | 10.9, 10.10 |
| `services/orchestrator_service/tests/{test_graph_builder,test_supervisor,test_state_schema,test_agents,test_response}.py` | Modify | existing tests of WS-10 code updated to new semantics, never weakened on crisis inputs | 10.2, 10.4, 10.5, 10.15 |
| `tests/integration/{test_crisis_flow,test_therapy_flow,test_orchestrator_crisis_emit}.py` | Modify | existing integration tests of WS-10 code | 10.2, 10.3, 10.5 |

## Tasks

### Task WS-10.1: Turn-scoped graph state and per-thread serialization
**Covers:** MVP-018, MVP-019, MVP-161
**Severity:** blocker
**Wave:** 1
**Depends on:** none
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: safety flags are per turn; the bounded crisis watch is added in WS-10.6)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/state_schema.py` (new channels `turn_id`, `degraded`, `crisis_watch`; reducer `add_degraded`; `TURN_SCOPED_CHANNELS`; `build_turn_input()`; `create_initial_state(..., turn_id=None)`)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`OrchestratorGraphBuilder.invoke`: per-thread lock, graph input from `build_turn_input`)
- Test: `services/orchestrator_service/tests/test_mvp_018_turn_scoped_state.py`
**Interfaces:**
- Consumes: `langgraph.types.Overwrite` (LangGraph 1.0.4, D-16).
- Produces: `TURN_SCOPED_CHANNELS: tuple[str, ...]`; `build_turn_input(state: OrchestratorState) -> dict[str, Any]`; `add_degraded(left: list[str], right: list[str]) -> list[str]` (ordered, de-duplicated union); state keys `turn_id: str`, `degraded: list[str]`, `crisis_watch: dict[str, Any] | None` (cross-turn, last write wins); `OrchestratorGraphBuilder.invoke(state, thread_id=None)` serialized per `thread_id`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_018_turn_scoped_state.py`
```python
"""MVP-018 / MVP-019 / MVP-161: graph state is turn-scoped and turns on one thread are serialized."""
from __future__ import annotations
import asyncio
import time
import pytest
from langgraph.types import Overwrite
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder
from services.orchestrator_service.src.langgraph.state_schema import (
    AgentResult, AgentType, TURN_SCOPED_CHANNELS, build_turn_input, create_initial_state,
)

def _install(monkeypatch, delay: float = 0.0):
    counter = [0]
    async def chat(state):
        counter[0] += 1
        n = counter[0]
        if delay:
            await asyncio.sleep(delay)
        return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True,
                                              response_content=f"REPLY #{n}", confidence=0.7).to_dict()]}
    async def noop(state):
        return {}
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    nodes = {k: noop for k in ("diagnosis", "therapy", "personality", "safety", "assessment", "emotion")}
    nodes["chat"] = chat
    monkeypatch.setattr(gb, "_import_agent_nodes", lambda: nodes)
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: None)
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True))
    b.compile()
    return b

async def _turn(b, tid, text):
    return await b.invoke(create_initial_state("user-1", "session-1", text, thread_id=tid), thread_id=tid)

@pytest.mark.asyncio
async def test_mvp_018_reply_is_fresh_each_turn(monkeypatch):
    b = _install(monkeypatch)
    for _ in range(3):
        result = await _turn(b, "thread-018", "hello there friend")
    assert result["final_response"] == "REPLY #3"
    assert all(r.get("response_content") != "REPLY #1" for r in result["agent_results"])

@pytest.mark.asyncio
async def test_mvp_019_crisis_flags_do_not_stick_to_later_turns(monkeypatch):
    b = _install(monkeypatch)
    first = await _turn(b, "thread-019", "I want to die")
    assert first["safety_flags"]["crisis_detected"] is True
    second = await _turn(b, "thread-019", "can we talk about my job?")
    assert second["safety_flags"]["crisis_detected"] is False
    assert second["processing_phase"] != "crisis_handling"
    assert second["final_response"] == "REPLY #1"

def test_mvp_018_turn_input_overwrites_turn_scoped_channels():
    inp = build_turn_input(create_initial_state("user-1", "session-1", "hi", thread_id="t"))
    for ch in ("agent_results", "safety_flags", "final_response", "degraded"):
        assert isinstance(inp[ch], Overwrite), ch
    for kept in ("active_treatment", "crisis_watch", "personality_style"):
        assert kept not in inp
    assert {"agent_results", "safety_flags", "final_response", "degraded", "selected_agents", "intent"} <= set(TURN_SCOPED_CHANNELS)

@pytest.mark.asyncio
async def test_mvp_161_concurrent_turns_on_one_thread_are_serialized(monkeypatch):
    b = _install(monkeypatch, delay=0.05)
    await asyncio.gather(_turn(b, "thread-161", "turn A apples"), _turn(b, "thread-161", "turn B bananas"))
    snap = await b.get_compiled_graph().aget_state({"configurable": {"thread_id": "thread-161"}})
    users = sorted(m["content"] for m in snap.values["messages"] if m["role"] == "user")
    assert users == ["turn A apples", "turn B bananas"]
    assert len([m for m in snap.values["messages"] if m["role"] == "assistant"]) == 2

@pytest.mark.asyncio
async def test_mvp_161_guard_different_threads_still_run_concurrently(monkeypatch):
    # Guard (passes before and after): the lock must be per thread, never global.
    b = _install(monkeypatch, delay=0.2)
    t0 = time.perf_counter()
    await asyncio.gather(_turn(b, "thread-x", "hello there friend"), _turn(b, "thread-y", "hello there friend"))
    assert time.perf_counter() - t0 < 0.35
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_018_turn_scoped_state.py -o "addopts=" -p no:logging -q`
Expected: FAIL at collection with `ImportError: cannot import name 'TURN_SCOPED_CHANNELS'`; after adding only that name, `test_mvp_018_reply_is_fresh_each_turn` fails with `'REPLY #1' != 'REPLY #3'` (the append reducer keeps every turn's `agent_results`), `test_mvp_019...` fails on `crisis_detected is True` (OR-merge across checkpointed turns), and `test_mvp_161...` fails with one user message missing (last writer wins).
- [ ] **Step 3: Implement** `build_turn_input(state: OrchestratorState) -> dict[str, Any]`, `TURN_SCOPED_CHANNELS`, `add_degraded` and the new keys in `state_schema.py`; make `OrchestratorGraphBuilder.invoke(state, thread_id=None)` acquire `self._thread_lock(thread_id)` and call `graph.ainvoke(build_turn_input(state), config)`. `build_turn_input` returns `user_id, session_id, thread_id, turn_id, current_message, messages, metadata, processing_phase` as plain values, `Overwrite(...)` for the reducer channels `agent_results, safety_flags, final_response, degraded`, plain resets for the last-write turn channels (`intent, intent_confidence, selected_agents, error_message, routing_reason, styled_response, aggregated_response, conversation_context, assembled_context, memory_context, memory_sources, retrieved_memories, diagnosis_output, therapy_output, personality_output, emotion_output, active_agents, crisis_level, safety_override_active`) and omits `active_treatment`, `crisis_watch` and `personality_style` so they persist. Reducers stay as they are within a super-step (existing reducer tests stay green). The lock map is `dict[str, asyncio.Lock]`, pruned of unlocked entries above 10,000 keys. If the pinned LangGraph ignores `Overwrite` in graph input, return the same `Overwrite` writes from the first node (`safety_precheck_node`) instead; the tests decide.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (5 passed)
- [ ] **Step 5: Regression:** run the owning suites, one process each. Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q` then `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/integration -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.1:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.2: Shared-lexicon screen in the precheck and the supervisor
**Covers:** MVP-025, MVP-026
**Supports:** MVP-017 (local half of ingress screening; closed by WS-10.5)
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-10.1, WS-12.2, WS-12.3
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional lexicon v1 from `solace_clinical`; the orchestrator never adds its own keyword list and never lowers a lexicon level)
**Real infra:** no
**Files:**
- Create: `services/orchestrator_service/src/langgraph/crisis_screen.py` (`screen_local`, `risk_rank`, `max_level`)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`safety_precheck_node` uses the screen and the run-turn pre-fused flags; delete `_word_boundary_match` and the `CRISIS_KEYWORDS` import)
- Modify: `services/orchestrator_service/src/langgraph/state_schema.py` (delete `CRISIS_KEYWORDS`/`HIGH_RISK_KEYWORDS`; `SafetyFlags` gains `categories`, `detection_layer`, `confidence`, `remote_check`, `watch_active`, `signal_source`, `lexicon_version`; `triggered_keywords` holds rule ids only; `update_safety_flags` unions `categories`)
- Modify: `services/orchestrator_service/src/langgraph/supervisor.py` (`IntentClassifier._check_crisis_indicators` uses `screen_local`; remove the `crisis_keywords` setting)
- Test: `services/orchestrator_service/tests/test_mvp_025_precheck_phrase_set.py` (new); `services/orchestrator_service/tests/test_graph_builder.py` (TestSafetyPrecheckNode: assert categories and rule ids, never weaker levels on crisis text), `services/orchestrator_service/tests/test_supervisor.py` (replace `test_crisis_keywords_exist`; crisis classify tests assert rule ids), `services/orchestrator_service/tests/test_state_schema.py` (SafetyFlags round trip of the new fields), `tests/integration/test_crisis_flow.py` (replace `test_crisis_keywords_superset_of_keywords_json_critical` with: every `keywords.json` CRITICAL phrase screens HIGH or above through `safety_precheck_node`), `tests/integration/test_therapy_flow.py` (`test_mild_mood_produces_low_risk` asserts `risk_level in ("NONE", "LOW")` and no crisis)
**Interfaces:**
- Consumes: IF-07 `solace_clinical.screen_text(text) -> ScreenResult`, `ScreenResult(level, categories, rule_ids, negated_rule_ids, lexicon_version)`, `CATEGORIES`, `LEXICON_VERSION`; phrase set `tests/clinical/crisis_phrases_v1.yaml`.
- Produces: `crisis_screen.screen_local(text: str) -> ScreenResult` (never raises; an internal error yields level ELEVATED, `rule_ids=("screen_error",)` and a CRITICAL log); `risk_rank(level: CrisisLevel | str) -> int` (NONE 0 .. CRITICAL 4); `max_level(*levels) -> CrisisLevel`; `SafetyFlags` dict keys `categories: list[str]`, `detection_layer: int | None`, `confidence: float | None`, `remote_check: str` (`ok|timeout|error|unauthorized|not_run`), `watch_active: bool`, `signal_source: str | None`, `lexicon_version: str | None`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_025_precheck_phrase_set.py`
```python
"""MVP-025 / MVP-026 (supports MVP-017): the orchestrator precheck passes the shared phrase set (IF-07)."""
from __future__ import annotations
import pathlib
import pytest
import yaml
from services.orchestrator_service.src.langgraph.crisis_screen import risk_rank
from services.orchestrator_service.src.langgraph.graph_builder import route_after_safety, safety_precheck_node
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

_REPO = pathlib.Path(__file__).resolve().parents[3]
PHRASES = yaml.safe_load((_REPO / "tests/clinical/crisis_phrases_v1.yaml").read_text(encoding="utf-8"))

def _flags(text: str) -> dict:
    return safety_precheck_node(create_initial_state("user-1", "session-1", text))["safety_flags"]

@pytest.mark.clinical
@pytest.mark.parametrize("case", PHRASES["positive"], ids=lambda c: c["text"][:40])
def test_mvp_025_positive_phrases_reach_min_level(case):
    flags = _flags(case["text"])
    assert risk_rank(flags["risk_level"]) >= risk_rank(case["min_level"])
    assert set(case.get("categories", [])) <= set(flags["categories"])

@pytest.mark.clinical
@pytest.mark.parametrize("case", PHRASES["negative"], ids=lambda c: c["text"][:40])
def test_mvp_025_negative_phrases_stay_at_or_below_max_level(case):
    assert risk_rank(_flags(case["text"])["risk_level"]) <= risk_rank(case["max_level"])

@pytest.mark.parametrize("text,category", [
    ("I can\u2019t go on", "suicidal_ideation"),
    ("I feel suicidal", "suicidal_ideation"),
    ("I'm going to kill him", "homicidal_ideation"),
    ("my dad hits me and I'm scared to go home", "abuse_disclosure"),
])
def test_mvp_026_register_phrases_route_to_crisis(text, category):
    state = create_initial_state("user-1", "session-1", text)
    out = safety_precheck_node(state)
    assert risk_rank(out["safety_flags"]["risk_level"]) >= risk_rank("HIGH")
    assert out["safety_flags"]["crisis_detected"] is True
    assert category in out["safety_flags"]["categories"]
    assert route_after_safety({**state, **out}) == "crisis_handler"

def test_mvp_025_flags_hold_rule_ids_and_lexicon_version_never_text():
    flags = _flags("I want to kill myself")
    assert flags["triggered_keywords"] and all("kill myself" not in r for r in flags["triggered_keywords"])
    assert flags["lexicon_version"]

def test_mvp_025_supervisor_crisis_check_uses_the_lexicon():
    from services.orchestrator_service.src.langgraph.state_schema import IntentType
    from services.orchestrator_service.src.langgraph.supervisor import IntentClassifier, SupervisorSettings
    intent, conf, _ = IntentClassifier(SupervisorSettings()).classify("I can\u2019t go on")
    assert intent == IntentType.CRISIS_DISCLOSURE and conf >= 0.9
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_025_precheck_phrase_set.py -o "addopts=" -p no:logging -q`
Expected: FAIL at import (`crisis_screen` does not exist); once it exists, `KeyError: 'categories'` and level failures: the 14-phrase keyword precheck rates "I can\u2019t go on", "I feel suicidal", "I'm going to kill him" and the abuse disclosure NONE, and rates the negative "cutting vegetables for dinner tonight" CRITICAL.
- [ ] **Step 3: Implement** `screen_local(text: str) -> ScreenResult` in `crisis_screen.py` and rewrite `safety_precheck_node(state)` in `graph_builder.py`: `level = max_level(state.safety_flags.risk_level (the run-turn pre-fused value, NONE when absent), screen_local(message).level)`, `categories = sorted(pre.categories | screen.categories)`, `crisis_detected = risk_rank(level) >= risk_rank("HIGH")`, `requires_escalation = crisis_detected`, `triggered_keywords = list(screen.rule_ids)`, `lexicon_version = screen.lexicon_version`, `detection_layer = 1` when the local screen set the level. `update_safety_flags` takes `detection_layer`, `confidence` and `signal_source` from the side with the higher `risk_level` (left on ties), so a lower-risk writer in the same super-step can never relabel the detection. The supervisor's `_check_crisis_indicators(message)` returns `risk_rank(screen_local(message).level) >= risk_rank("HIGH")` and `classify` returns the rule ids as matched keywords. Update the listed existing tests in place; crisis inputs keep asserting `crisis_detected is True` and HIGH or CRITICAL.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`, then with `tests/clinical`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.2:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.3: Crisis event content and a publisher that reports its outcome and retries a failed publish
**Covers:** MVP-167
**Supports:** MVP-067, MVP-069 (closed by WS-10.5)
**Severity:** medium
**Wave:** 1
**Depends on:** WS-06.8, WS-06.10 (`record_publish_failure`, crisis-review F2), WS-05.7, WS-01.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-28 (default assumed: the orchestrator publishes `CrisisDetectedEvent` with the turn correlation id and notification coalesces; a publish that fails, for example while Postgres or the outbox is down, is retried in the background until it lands, crisis-review F2); D-05 (default assumed: an uncategorized HIGH/CRITICAL detection carries `["imminent_danger"]`)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/events.py` (new `session_uuid`, `build_crisis_detected_event`, `publish_crisis_event`, `CrisisPublishOutcome`, `CrisisPublishRetrier`; `EventFactory.crisis_detected` gains keyword-only `categories`, `detection_layer`, `confidence`, `turn_id`; `to_kafka_event` CRISIS branch never fabricates; import `solace_events.schemas`, not `src.solace_events.schemas`)
- Create: `services/orchestrator_service/tests/test_mvp_167_crisis_event_contract.py`; `services/orchestrator_service/tests/test_mvp_357_event_contracts.py` (producer-side `verify_event_producer` test against the orchestrator's notification event contract, requested by WS-05.10, request 9 of WS-05)
- Test: `services/orchestrator_service/tests/test_mvp_167_crisis_event_contract.py`, `services/orchestrator_service/tests/test_mvp_357_event_contracts.py`; `tests/integration/test_orchestrator_crisis_emit.py` (`test_to_kafka_event_maps_crisis_detected_to_safety_topic` passes the full keyword payload)
**Interfaces:**
- Consumes: IF-05 `ServiceEventPlane.publish(event, *, conn=None) -> UUID`, `solace_events.errors.EventPublishError` (WS-06.8); `solace_events.service_plane.record_publish_failure(service_name)` (WS-06.10); `solace_events.schemas.CrisisDetectedEvent`, `EventMetadata`; IF-07 `CATEGORIES`; IF-20 `solace_testing.contracts.load_event_contracts(contracts_dir) -> list[EventContract]` and `verify_event_producer(contract, produced) -> list[str]` plus the event contract `tests/contracts/events/orchestrator__notification/safety.crisis.detected.json` (all WS-05.7, W1 phase A); WS-01.7's strict-xfail WS-10 case in the import-root guard, the W0-barrier allow-list of this module that WS-10 requested (this task's import rewrite turns it into a strict XPASS, and the controller deletes the marker by ledger Ruling).
- Produces: `session_uuid(raw: str) -> UUID` (identity for UUID strings, else `uuid5(NAMESPACE_URL, "solace:session:" + raw)`); `build_crisis_detected_event(*, user_id: str | UUID, session_id: str, turn_id: UUID, level: CrisisLevel | str, categories: Iterable[str], detection_layer: int, confidence: float) -> CrisisDetectedEvent` (raises `ValueError` for a non-UUID user id; keeps only codes in `CATEGORIES`, sorted, `["imminent_danger"]` when none remain; `escalation_action="page_on_call"`, `requires_human_review=True`, `metadata.correlation_id=turn_id`, `metadata.source_service="orchestrator-service"`); `@dataclass(frozen=True) CrisisPublishOutcome(status: Literal["published", "failed"], event_id: UUID | None, error_code: str | None)`; `async publish_crisis_event(plane, event, *, retrier: CrisisPublishRetrier | None = None) -> CrisisPublishOutcome` (logs `orchestrator_crisis_event_emitted` with event_id, correlation_id, user_id and session_id only after `publish` returns; on `EventPublishError` or any exception logs CRITICAL `orchestrator_crisis_event_publish_failed` with `event_id`, `correlation_id`, `user_id` and `error_type` (so operators can act on the turn), hands the event to `retrier` when one is given, and returns `failed`; never raises). `class CrisisPublishRetrier(plane, *, max_pending: int = 10_000, base_backoff_seconds: float = 1.0, max_backoff_seconds: float = 60.0, attempt_timeout_seconds: float = 10.0)` (crisis-review F2): `submit(event) -> bool` (keyed by `metadata.correlation_id`, so one pending entry per turn; a submit for a turn already pending returns `True` and adds nothing; when full it keeps the queue, logs CRITICAL `orchestrator_crisis_publish_retry_full` with `correlation_id` and `user_id`, and returns `False`), `pending() -> int`, `async run_once() -> int` (attempts every due entry once through `plane.publish(event)` bounded by `attempt_timeout_seconds`; a new entry is due at once, and each failed attempt doubles that entry's backoff starting at `base_backoff_seconds`; the same `event_id` every time, so the consumer's ledger and notification's correlation coalescing make a late duplicate harmless; a failure doubles that entry's backoff up to the cap; a success logs `orchestrator_crisis_event_emitted` with `retried=True`; returns the number published) and `async run(*, interval_seconds: float = 1.0) -> None` (loops `run_once`; never raises except `CancelledError`). Each attempt that ends in `EventPublishError` is counted by the plane's `solace_event_publish_failures_total` (WS-06.10), and a timed-out attempt is counted through `record_publish_failure("orchestrator-service")`, so `CrisisEventPublishFailures` fires for as long as the outage lasts. Also the orchestrator's producer verification against `tests/contracts/events/orchestrator__notification/` (WS-05 request #9, needed by WS-05.10's contract-coverage guard).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_167_crisis_event_contract.py`
```python
"""MVP-167 (supports MVP-067/MVP-069): crisis events carry real fields and report the publish outcome."""
from __future__ import annotations
from decimal import Decimal
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5
import pytest
from structlog.testing import capture_logs
from solace_events.errors import EventPublishError
from services.orchestrator_service.src.events import (
    CrisisPublishRetrier, EventFactory, build_crisis_detected_event, publish_crisis_event, session_uuid, to_kafka_event,
)

USER = UUID("8a3f0d4e-0000-4000-8000-000000000001")
TURN = UUID("4b6f1c2a-0000-4000-8000-000000000002")

class FakePlane:
    def __init__(self, exc=None):
        self.events, self.exc = [], exc
    async def publish(self, event, *, conn=None):
        if self.exc:
            raise self.exc
        self.events.append(event)
        return event.metadata.event_id

def _event(**kw):
    base = dict(user_id=str(USER), session_id="s1", turn_id=TURN, level="HIGH",
                categories=["self_harm"], detection_layer=1, confidence=1.0)
    return build_crisis_detected_event(**{**base, **kw})

def test_mvp_167_incomplete_crisis_payload_is_not_fabricated():
    assert to_kafka_event(EventFactory.crisis_detected(uuid4(), USER, "HIGH", "self_harm")) is None

def test_mvp_167_kafka_event_carries_categories_layer_confidence_and_turn():
    ev = EventFactory.crisis_detected(uuid4(), USER, "HIGH", None, categories=["self_harm"],
                                      detection_layer=3, confidence=0.62, turn_id=TURN)
    k = to_kafka_event(ev)
    assert k.event_type == "safety.crisis.detected"
    assert k.trigger_indicators == ["self_harm"]
    assert (k.detection_layer, k.confidence) == (3, Decimal("0.62"))
    assert k.metadata.correlation_id == TURN and k.metadata.source_service == "orchestrator-service"
    assert k.escalation_action == "page_on_call" and k.requires_human_review is True

def test_mvp_067_session_uuid_maps_client_ids():
    assert session_uuid("session-abc") == uuid5(NAMESPACE_URL, "solace:session:session-abc")
    assert session_uuid(str(USER)) == USER

def test_mvp_067_user_id_is_never_fabricated():
    with pytest.raises(ValueError):
        _event(user_id="user-1")

def test_mvp_167_categories_are_lexicon_codes_only():
    assert _event(categories=["suicide_plan", "free text from the user"]).trigger_indicators == ["suicide_plan"]
    assert _event(categories=[]).trigger_indicators == ["imminent_danger"]

@pytest.mark.asyncio
async def test_mvp_069_failed_publish_is_reported_and_never_logged_as_emitted():
    with capture_logs() as logs:
        out = await publish_crisis_event(FakePlane(exc=EventPublishError("outbox down")), _event())
    assert out.status == "failed" and out.event_id is None
    assert not [l for l in logs if l["event"] == "orchestrator_crisis_event_emitted"]
    assert [l for l in logs if l["event"] == "orchestrator_crisis_event_publish_failed" and l["log_level"] == "critical"]

@pytest.mark.asyncio
async def test_mvp_069_successful_publish_logs_ids():
    plane = FakePlane()
    ev = _event()
    with capture_logs() as logs:
        out = await publish_crisis_event(plane, ev)
    assert out.status == "published" and out.event_id == ev.metadata.event_id and plane.events == [ev]
    emitted = [l for l in logs if l["event"] == "orchestrator_crisis_event_emitted"][0]
    assert emitted["event_id"] == str(ev.metadata.event_id) and emitted["correlation_id"] == str(TURN)

class FlakyPlane(FakePlane):
    def __init__(self, failures):
        super().__init__(); self.failures = failures
    async def publish(self, event, *, conn=None):
        if self.failures:
            self.failures -= 1
            raise EventPublishError("outbox down")
        return await super().publish(event, conn=conn)

@pytest.mark.asyncio
async def test_crisis_review_f2_failed_publish_is_retried_until_the_outbox_accepts_it():
    plane = FlakyPlane(failures=2)
    retrier = CrisisPublishRetrier(plane, base_backoff_seconds=0.0)
    ev = _event()
    assert (await publish_crisis_event(plane, ev, retrier=retrier)).status == "failed"
    assert retrier.pending() == 1
    assert [await retrier.run_once() for _ in range(2)] == [0, 1]
    assert plane.events == [ev] and retrier.pending() == 0

@pytest.mark.asyncio
async def test_crisis_review_f2_failure_log_names_the_turn_and_the_user():
    with capture_logs() as logs:
        await publish_crisis_event(FakePlane(exc=EventPublishError("outbox down")), _event())
    failed = [l for l in logs if l["event"] == "orchestrator_crisis_event_publish_failed"][0]
    assert (failed["correlation_id"], failed["user_id"], failed["error_type"]) == (str(TURN), str(USER), "EventPublishError")

def test_crisis_review_f2_one_pending_entry_per_turn_and_a_bounded_queue():
    retrier = CrisisPublishRetrier(FakePlane(), max_pending=1)
    assert retrier.submit(_event()) is True and retrier.submit(_event()) is True and retrier.pending() == 1
    with capture_logs() as logs:
        assert retrier.submit(_event(turn_id=UUID("4b6f1c2a-0000-4000-8000-0000000000ff"))) is False
    assert [l for l in logs if l["event"] == "orchestrator_crisis_publish_retry_full" and l["log_level"] == "critical"]
```
and in `services/orchestrator_service/tests/test_mvp_357_event_contracts.py` (the file name and the `orchestrator__` directory string are what WS-05.10's coverage guard looks for)
```python
"""Supports MVP-357 (WS-05): the orchestrator's real crisis event satisfies its notification event contract and carries no message text."""
from __future__ import annotations
import json
import pathlib
from uuid import UUID
import pytest
from solace_testing.contracts import load_event_contracts, verify_event_producer
from services.orchestrator_service.src.events import build_crisis_detected_event

REPO = pathlib.Path(__file__).resolve().parents[3]
PAIR_DIR = REPO / "tests/contracts/events/orchestrator__notification"
TEXT = "I can't go on and I want to end it all tonight"

def _event():
    return build_crisis_detected_event(user_id="8a3f0d4e-0000-4000-8000-000000000001", session_id="client-session-7",
                                       turn_id=UUID("4b6f1c2a-0000-4000-8000-000000000002"), level="HIGH",
                                       categories=["suicidal_ideation", TEXT], detection_layer=1, confidence=1.0)

@pytest.mark.contract
def test_mvp_357_orchestrator_crisis_event_matches_the_notification_contract():
    contracts = [c for c in load_event_contracts(REPO / "tests/contracts/events")
                 if c.producer.removesuffix("-service") == "orchestrator"]
    assert PAIR_DIR.is_dir() and [c.event_type for c in contracts] == ["safety.crisis.detected"]
    assert verify_event_producer(contracts[0], _event()) == []

@pytest.mark.contract
def test_mvp_357_crisis_event_payload_carries_no_message_text_or_raw_session_id():
    payload = json.dumps(_event().to_dict())
    assert "end it all" not in payload and "client-session-7" not in payload
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_167_crisis_event_contract.py services/orchestrator_service/tests/test_mvp_357_event_contracts.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ImportError: cannot import name 'build_crisis_detected_event'` in both files; with the imports stubbed, `test_mvp_167_incomplete_crisis_payload_is_not_fabricated` fails because today's `to_kafka_event` returns an event with the hard-coded `detection_layer=1`, `confidence=Decimal("0.9")` and `escalation_action="notify_on_call_clinician"`, and an event built the way today's `to_kafka_event` builds it would break the contract's IF-06 rule `escalation_action` = `page_on_call` (no orchestrator test checks the event against its consumer's contract today, MVP-357). The three `crisis_review_f2` tests fail on the missing `CrisisPublishRetrier`: today a failed crisis publish is logged and dropped, so a database outage during a crisis turn loses the page for good (crisis-review F2).
- [ ] **Step 3: Implement** `session_uuid`, `build_crisis_detected_event`, `CrisisPublishOutcome`, `CrisisPublishRetrier` and `publish_crisis_event(plane, event, *, retrier=None)` in `events.py` with the signatures above (the retrier keeps an insertion-ordered dict of `correlation_id -> (event, next_attempt_at, backoff)` on the monotonic clock); the CRISIS branch of `to_kafka_event` returns `None` (and logs ERROR `crisis_event_incomplete`) unless the payload carries `categories`, `detection_layer`, `confidence` and `turn_id`, and otherwise delegates to `build_crisis_detected_event`. Switch the lazy import in `to_kafka_event` to `from solace_events.schemas import ...` (MVP-270 import-root rule). Update `test_to_kafka_event_maps_crisis_detected_to_safety_topic` to the keyword payload. A `verify_event_producer` mismatch is fixed in `events.py`, never by editing WS-05's contract file; a contract that contradicts IF-06 goes to WS-05 as a ledger Ruling.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (12 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass. Then the same command with `"tests/build/test_import_roots.py::test_mvp_270_no_src_root_imports[WS-10]"`. Expected: XPASS(strict) reported as a failure, the planned signal for the controller's Ruling that removes the WS-01.7 marker; record it in the ledger.
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.3:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.4: Safety-service check client, max-only fusion and the safety agent
**Covers:** MVP-166
**Supports:** MVP-017 (remote half), MVP-020 and MVP-021 (safety agent URL and credentials; closed by WS-11.2)
**Severity:** medium
**Wave:** 1
**Depends on:** WS-10.2, WS-10.3, WS-07.3, WS-12.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-22 (default assumed: /check on every message, 2 s timeout, can only raise; on failure the local result stands and the turn is degraded); D-03 (default assumed: obo service token for the user)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/crisis_screen.py` (`SafetyCheckClient`, `RemoteCheck`, `IngressScreen`, `screen_message`)
- Modify: `services/orchestrator_service/src/agents/safety_agent.py` (uses `SafetyCheckClient`; `service_url` from `SERVICE_SAFETY_SERVICE_URL`; fallback is `screen_local`; never lowers existing flags; reuses this turn's remote result; every name re-exported by `agents/__init__.py` stays importable)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`GraphBuilderSettings.use_safety_service_precheck` removed; `build()` adds `SafetyAgent(client=self._safety_client).process` as the `safety_agent` node instead of `_import_agent_nodes()["safety"]`)
- Test: `services/orchestrator_service/tests/test_mvp_017_safety_check_fusion.py`, `services/orchestrator_service/tests/test_mvp_166_safety_agent_fallback.py` (new); `services/orchestrator_service/tests/test_agents.py` (safety agent tests: rule ids instead of keyword text; IF-06 shapes)
**Interfaces:**
- Consumes: IF-04 `service_auth_headers(target_service, *, on_behalf_of=None) -> dict[str, str]`; IF-06 request `{user_id, session_id, content, turn_id, context}` and response `{level, risk_score, categories, requires_escalation, resources, detection_layer}`; IF-07 `RiskSignal`, `fuse_risk`; WS-11 `config.ServiceEndpoints.safety_service_url` (read only); `events.session_uuid` (WS-10.3).
- Produces: `class SafetyCheckClient(base_url: str, timeout_seconds: float = 2.0, transport: httpx.AsyncBaseTransport | None = None)` with `async check(*, user_id: str, session_id: str, turn_id: UUID, content: str, channel: str) -> RemoteCheck` and `async aclose()`; `@dataclass(frozen=True) RemoteCheck(status: Literal["ok", "timeout", "error", "unauthorized"], level: CrisisLevel | None, categories: frozenset[str], risk_score: float | None, detection_layer: int | None, requires_escalation: bool)`; `@dataclass(frozen=True) IngressScreen(level: CrisisLevel, categories: frozenset[str], detection_layer: int, confidence: float, signals: tuple[RiskSignal, ...], degraded: tuple[str, ...], local: ScreenResult, remote: RemoteCheck | None)`; `async screen_message(*, message: str, user_id: str, session_id: str, turn_id: UUID, channel: str, client: SafetyCheckClient | None) -> IngressScreen`; `SafetyAgent(settings: SafetyAgentSettings | None = None, *, client: SafetyCheckClient | None = None)`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_017_safety_check_fusion.py`
```python
"""MVP-017 (remote half) / MVP-020: /check per IF-06 with an obo token, fused max-only, fail-safe."""
from __future__ import annotations
import asyncio
import json
from uuid import UUID
import httpx
import pytest
from services.orchestrator_service.src.events import session_uuid
from services.orchestrator_service.src.langgraph import crisis_screen as cs

USER = "8a3f0d4e-0000-4000-8000-000000000001"
TURN = UUID("4b6f1c2a-0000-4000-8000-000000000002")

def _ok(level, categories=()):
    return httpx.Response(200, json={"level": level, "risk_score": 0.5, "categories": list(categories),
                                     "requires_escalation": False, "resources": [], "detection_layer": 2})

def _client(handler, timeout=2.0):
    return cs.SafetyCheckClient(base_url="http://safety-service:8002", timeout_seconds=timeout,
                                transport=httpx.MockTransport(handler))

@pytest.fixture(autouse=True)
def obo(monkeypatch):
    calls = []
    async def fake(target, *, on_behalf_of=None):
        calls.append((target, on_behalf_of))
        return {"Authorization": "Bearer obo-test", "X-Turn-ID": str(TURN)}
    monkeypatch.setattr(cs, "service_auth_headers", fake)
    return calls

@pytest.mark.asyncio
async def test_mvp_017_check_sends_if06_request_with_obo_token(obo):
    seen = {}
    def handler(req):
        seen.update(path=req.url.path, body=json.loads(req.content), auth=req.headers["Authorization"])
        return _ok("LOW")
    res = await _client(handler).check(user_id=USER, session_id="s1", turn_id=TURN, content="hello", channel="rest")
    assert seen["path"] == "/api/v1/safety/check"
    assert seen["body"] == {"user_id": USER, "session_id": str(session_uuid("s1")), "content": "hello",
                            "turn_id": str(TURN), "context": {"channel": "rest"}}
    assert seen["auth"] == "Bearer obo-test" and obo == [("safety-service", USER)]
    assert res.status == "ok" and res.level == "LOW"

@pytest.mark.asyncio
async def test_mvp_017_timeout_keeps_the_local_result_and_degrades():
    async def handler(req):
        await asyncio.sleep(3)
        return _ok("NONE")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    s = await cs.screen_message(message="I want to kill myself", user_id=USER, session_id="s1", turn_id=TURN,
                                channel="rest", client=_client(handler, timeout=0.3))
    assert loop.time() - t0 < 1.0
    assert cs.risk_rank(s.level) >= cs.risk_rank("HIGH")
    assert s.degraded == ("safety_service_timeout",)

@pytest.mark.asyncio
@pytest.mark.parametrize("text,remote,floor", [
    ("I want to kill myself", "LOW", "HIGH"),
    ("hello there", "HIGH", "HIGH"),
    ("hello there", "CRITICAL", "CRITICAL"),
])
async def test_mvp_017_fusion_is_max_only(text, remote, floor):
    s = await cs.screen_message(message=text, user_id=USER, session_id="s1", turn_id=TURN, channel="ws",
                                client=_client(lambda r: _ok(remote, ["suicidal_ideation"])))
    assert cs.risk_rank(s.level) >= cs.risk_rank(floor)
    assert s.degraded == ()

@pytest.mark.asyncio
async def test_mvp_017_unauthorized_check_degrades_and_local_stands():
    s = await cs.screen_message(message="I want to kill myself", user_id=USER, session_id="s1", turn_id=TURN,
                                channel="rest", client=_client(lambda r: httpx.Response(401, json={"detail": "x"})))
    assert s.remote.status == "unauthorized" and s.degraded == ("safety_service_unavailable",)
    assert cs.risk_rank(s.level) >= cs.risk_rank("HIGH")

def test_mvp_020_safety_agent_reads_the_compose_safety_url(monkeypatch):
    monkeypatch.setenv("SERVICE_SAFETY_SERVICE_URL", "http://safety-service:8002")
    monkeypatch.delenv("ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL", raising=False)
    from services.orchestrator_service.src.agents.safety_agent import SafetyAgentSettings
    assert SafetyAgentSettings().service_url == "http://safety-service:8002"
```
and in `services/orchestrator_service/tests/test_mvp_166_safety_agent_fallback.py`
```python
"""MVP-166: the safety agent's fallback is the shared screen, never a narrower list, and never lowers."""
from __future__ import annotations
import pytest
from services.orchestrator_service.src.agents.safety_agent import SafetyAgent
from services.orchestrator_service.src.langgraph import crisis_screen as cs
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

USER = "8a3f0d4e-0000-4000-8000-000000000001"

@pytest.fixture
def unreachable(monkeypatch):
    # Discard port: connection refused today (legacy env name) and after the fix (compose env name).
    monkeypatch.setenv("SERVICE_SAFETY_SERVICE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL", "http://127.0.0.1:9")

@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["I'd be better off dead", "I just want to end it all", "I can't go on",
                                  "I'm going to take my life", "thinking of hanging myself"])
async def test_mvp_166_fallback_uses_the_shared_screen(unreachable, text):
    flags = (await SafetyAgent().process(create_initial_state(USER, "s1", text)))["safety_flags"]
    assert flags["crisis_detected"] is True
    assert flags["risk_level"] == cs.screen_local(text).level

@pytest.mark.asyncio
async def test_mvp_166_fallback_does_not_flag_cutting_vegetables(unreachable):
    flags = (await SafetyAgent().process(create_initial_state(USER, "s1", "I hate cutting vegetables")))["safety_flags"]
    assert flags["crisis_detected"] is False

@pytest.mark.asyncio
async def test_mvp_166_agent_never_lowers_this_turns_level(unreachable):
    state = create_initial_state(USER, "s1", "hmm")
    state["safety_flags"] = {"risk_level": "HIGH", "crisis_detected": True, "categories": ["suicidal_ideation"],
                             "detection_layer": 3, "remote_check": "ok"}
    flags = (await SafetyAgent().process(state))["safety_flags"]
    assert flags["risk_level"] == "HIGH" and "suicidal_ideation" in flags["categories"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_166_safety_agent_fallback.py -o "addopts=" -p no:logging -q` and the same command with `test_mvp_017_safety_check_fusion.py`
Expected: FAIL. `test_mvp_166_fallback_uses_the_shared_screen` fails with `crisis_detected False` for "I'd be better off dead" (today's 10-substring fallback list); `test_mvp_020...` fails because `SafetyAgentSettings` reads `ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL` and returns `http://localhost:8002`; the fusion file fails with `AttributeError: module ... has no attribute 'SafetyCheckClient'`.
- [ ] **Step 3: Implement** `SafetyCheckClient.check(...)`, `RemoteCheck`, `IngressScreen` and `screen_message(...)` in `crisis_screen.py`: one pooled `httpx.AsyncClient`, overall `asyncio.wait_for(timeout_seconds)`, no retries (the 2 s budget allows none); 401/403 give `unauthorized`, other failures (including an exception while minting headers) give `error`, a timeout gives `timeout`; `screen_message` builds `RiskSignal(source="orchestrator.local", level, confidence=1.0 if a rule matched else 0.0, reliable=True)` and, when status is `ok`, `RiskSignal(source="safety-service", level, confidence=risk_score, reliable=True)`, fuses with `fuse_risk`, unions categories filtered to `CATEGORIES`, sets `detection_layer` to the signal that set the level and maps statuses to degraded codes (`safety_service_timeout`, `safety_service_unavailable`); with `client=None` (tests and development only; `from_runtime` always passes a client) the remote check is skipped and `remote` is `None`. `SafetyAgent.process`: when `state.safety_flags.remote_check == "ok"` reuse it, else call `check`; on failure use `screen_local`; merge max-only with the existing flags; `response_content=None` (the crisis handler writes the reply). `SafetyAgentSettings.service_url` uses `Field(validation_alias="SERVICE_SAFETY_SERVICE_URL")` with `populate_by_name=True`; remove the substring `include_resources` heuristic and the 10-phrase list.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.4:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.5: run_turn: screen and page before the graph, one ingress deadline, crisis-safe fallback, refused-message screen, from_runtime
**Covers:** MVP-017, MVP-067, MVP-069, MVP-079
**Supports:** MVP-162 (turn deadline; closed by WS-11.8), MVP-356 (turn id bound for every hop; closed by WS-11.8)
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-10.1, WS-10.2, WS-10.3, WS-10.4, WS-06.8, WS-07.1, WS-07.3, WS-04.3, WS-04.6, WS-12.2
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-22 (default assumed: screen and /check before the graph; in-band resources never depend on the graph, LLM, downstream services, Kafka or the database; for HIGH/CRITICAL the whole in-band answer is bounded by one deadline measured from `run_turn` entry, crisis-review F7); D-28 (default assumed: one orchestrator event per turn when the fused or graph-raised level is HIGH or above, coalesced downstream by turn id; a failed publish is retried in the background, crisis-review F2); D-29 (decided 2026-10-19: resources come from the per-locale registry, `solace_clinical.crisis_resource_set(resolve_locale(locale))`, where `locale` is the user's access-token claim `loc` that the edge passes in without any lookup; a missing claim means en-US, so en-US output is unchanged; an unsupported or unknown locale gets the generic fallback, which is static data, so the in-band answer still never depends on the LLM, a downstream service, Kafka or the database, and fusion and paging are unchanged)
**Real infra:** no
**Files:**
- Create: `services/orchestrator_service/src/response/safety_copy.py` (`crisis_reply_text(level, resources) -> str`, `safety_footer(resources) -> str`, `SAFE_FALLBACK_TEXT: str`)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`TurnSettings`, `CrisisInfo`, `TurnResult`, `run_turn`, `crisis_for_refused_message`, `from_runtime`, constructor keyword arguments; `crisis_handler_node` uses `crisis_resources(state["locale"])`; delete `_CrisisResourceManager`, `_maybe_emit_crisis_event` and `invoke_sync`; `invoke()` no longer publishes)
- Modify: `services/orchestrator_service/src/langgraph/state_schema.py` (D-29: state key `locale: str`, listed in `TURN_SCOPED_CHANNELS`; `create_initial_state(..., locale: str = DEFAULT_LOCALE)`)
- Modify: `services/orchestrator_service/src/langgraph/__init__.py` (export `TurnResult`, `CrisisInfo`, `TurnSettings`)
- Test: `services/orchestrator_service/tests/test_mvp_079_run_turn.py` (new); `services/orchestrator_service/tests/test_graph_builder.py` (drop `test_invoke_sync_crisis`; crisis handler tests assert `crisis_resources()` contacts); `tests/integration/test_orchestrator_crisis_emit.py` (`test_invoke_emits_crisis_event...` becomes a `run_turn` test with a fake plane)
**Interfaces:**
- Consumes: IF-01 `ServiceRuntime` (`spec.environment`, `register_shutdown`; WS-04.3) with `event_plane` populated by the bootstrap (WS-04.6), `is_strict_environment`, `StartupDependencyError`; IF-05 `ServiceEventPlane.publish` (WS-06.8); IF-04 `acting_as_user(user_id)` (WS-07.1); IF-07 `crisis_resources`, `RESOURCES_VERSION` and the D-29 names `crisis_resource_set`, `resolve_locale`, `DEFAULT_LOCALE`, `FALLBACK_RESOURCES_VERSION` (WS-12.2); IF-18 counter object `CRISIS_DETECTED` (`solace_crisis_detected_total{source,level}`) from `solace_infrastructure.runtime.metrics` (WS-04.3); `screen_message`, `screen_local`, `SafetyCheckClient` (WS-10.4, WS-10.2); `build_crisis_detected_event`, `publish_crisis_event`, `CrisisPublishRetrier` (WS-10.3); `build_turn_input` (WS-10.1); IF-01 `runtime.register_background_task(name, task)`.
- Produces (IF-10): `class TurnSettings(BaseSettings)` with env prefix `ORCHESTRATOR_`: `turn_timeout_seconds: float = 20.0`, `safety_check_timeout_seconds: float = 2.0`, `crisis_response_budget_seconds: float = 2.8`; `@dataclass(frozen=True) CrisisInfo(level: str, categories: list[str], resources: list[dict[str, str]], resources_version: str)` with `to_dict() -> dict`; `@dataclass TurnResult(response_text: str, crisis: CrisisInfo | None, degraded: list[str], agents: list[str], turn_id: UUID, safety_flags: dict[str, Any], crisis_watch_until: datetime | None)`; `async OrchestratorGraphBuilder.run_turn(self, *, user_id: str, thread_id: str, session_id: str, message: str, turn_id: UUID, channel: Literal["rest", "ws"], locale: str | None = None) -> TurnResult` (D-29: `locale` is the raw access-token claim `loc` as WS-11 receives it, `None` when absent; `run_turn` resolves it once with `resolve_locale(locale)`, writes the result to the graph input's `locale` key and builds every `CrisisInfo` from one `crisis_resource_set(resolved)` call: `resources` and `resources_version = set.version`, so en-US keeps `RESOURCES_VERSION` and the fallback carries `FALLBACK_RESOURCES_VERSION`); `@classmethod async from_runtime(cls, runtime: ServiceRuntime, settings: GraphBuilderSettings | None = None) -> OrchestratorGraphBuilder`; `OrchestratorGraphBuilder.__init__(self, settings=None, *, event_plane=None, safety_client=None, turn_settings=None, checkpointer=None, clock=None, publish_retrier=None)` (when `publish_retrier` is `None` and a plane is given, the builder makes its own unstarted `CrisisPublishRetrier(event_plane)`) and the read-only property `publish_retrier`; degraded codes `safety_service_timeout`, `safety_service_unavailable`, `crisis_alert_publish_failed`, `crisis_alert_publish_pending`, `graph_error`, `turn_timeout`.
- Produces (crisis-review F7, a precision of IF-10's budget): for a turn whose fused level is HIGH or CRITICAL, one deadline `entry + crisis_response_budget_seconds` (`entry` is the monotonic time at `run_turn` entry, which the edge calls as its first await after authentication) bounds the screen, the graph wait and the publish wait together; at the deadline `run_turn` returns the crisis reply from the fused screen at once. A slow or hung `/check` therefore shortens the graph's share instead of adding to it.
- Produces (crisis-review F12): any HIGH/CRITICAL level the graph raises (LLM intent, a diagnosis or therapy provider, whatever `signal_source`) and the screen did not already publish is published exactly once with `detection_layer = safety_flags.detection_layer or 3` (IF-06: orchestrator layer 3 is a non-local detection) and `confidence = safety_flags.confidence or 0.6`; building or publishing that event can never raise out of `run_turn` or drop the crisis object.
- Produces (crisis-review F3, consumed by WS-11.16, WS-11.17 and WS-11.20): `async OrchestratorGraphBuilder.crisis_for_refused_message(self, *, user_id: str, session_id: str, message: str, turn_id: UUID, channel: Literal["rest", "ws"], locale: str | None = None) -> CrisisInfo | None` (D-29: `locale` resolved exactly as in `run_turn`) for a message the edge refuses (consent, expired token, rate limit): `screen_local(message)` only (no graph, `/check`, LLM, thread lock or checkpoint write); at HIGH/CRITICAL one `build_crisis_detected_event(..., detection_layer=1, confidence=1.0)` through `publish_crisis_event(..., retrier=self.publish_retrier)` (awaited at most `crisis_response_budget_seconds`, then left running), `CRISIS_DETECTED{source="orchestrator"}` incremented and INFO `orchestrator_refused_message_crisis` logged with `channel`, `level` and `turn_id` (never text); returns the `CrisisInfo` (categories, and resources and version from `crisis_resource_set(resolve_locale(locale))`) at ELEVATED or above, else `None`; never raises (an internal error returns the ELEVATED crisis object, as `screen_local`'s own fail-safe does).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_079_run_turn.py`
```python
"""MVP-079 / MVP-017 / MVP-067 / MVP-069: run_turn screens and pages before the graph and never fails unsafe."""
from __future__ import annotations
import asyncio
import time
from uuid import UUID
import pytest
import structlog
from structlog.testing import capture_logs
from solace_clinical import FALLBACK_RESOURCES_VERSION, RESOURCES_VERSION, crisis_resources
from solace_common.request_context import get_current_user_id
from solace_events.errors import EventPublishError
from services.orchestrator_service.src.events import session_uuid
from services.orchestrator_service.src.langgraph import crisis_screen as cs
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings
from services.orchestrator_service.src.langgraph.state_schema import AgentResult, AgentType, create_initial_state
from services.orchestrator_service.src.response.safety_copy import SAFE_FALLBACK_TEXT

USER = "8a3f0d4e-0000-4000-8000-000000000001"
TURN = UUID("4b6f1c2a-0000-4000-8000-000000000002")
SEEN: dict = {}

class FakePlane:
    def __init__(self, exc=None, delay=0.0, failures=None):
        self.events, self.exc, self.delay, self.failures = [], exc, delay, failures
    async def publish(self, event, *, conn=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failures:                                   # fail this many times, then succeed
            self.failures -= 1
            raise EventPublishError("outbox down")
        if self.exc:
            raise self.exc
        self.events.append(event)
        return event.metadata.event_id

class FakeSafety:
    def __init__(self, level="NONE", status="ok", delay=0.0):
        self.level, self.status, self.delay, self.calls = level, status, delay, []
    async def check(self, *, user_id, session_id, turn_id, content, channel):
        self.calls.append(turn_id)
        if self.delay:
            await asyncio.sleep(self.delay)                 # a hung safety-service until the client's own timeout
        return cs.RemoteCheck(status=self.status, level=self.level if self.status == "ok" else None,
                              categories=frozenset(), risk_score=0.1, detection_layer=2, requires_escalation=False)
    async def aclose(self):
        pass

def _builder(monkeypatch, plane=None, safety=None, fallback_chat=False, **turn):
    async def chat(state):
        SEEN["user"] = get_current_user_id()
        SEEN["turn"] = structlog.contextvars.get_contextvars().get("turn_id")
        meta = {"fallback_mode": True} if fallback_chat else {}
        return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True,
                                              response_content="Tell me more.", confidence=0.7, metadata=meta).to_dict()]}
    async def noop(state):
        return {}
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    nodes = {k: noop for k in ("diagnosis", "therapy", "personality", "safety", "assessment", "emotion")}
    nodes["chat"] = chat
    monkeypatch.setattr(gb, "_import_agent_nodes", lambda: nodes)
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: None)
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True), event_plane=plane or FakePlane(),
                                 safety_client=safety or FakeSafety(),
                                 turn_settings=TurnSettings(turn_timeout_seconds=turn.get("deadline", 2.0),
                                                            safety_check_timeout_seconds=turn.get("check_timeout", 0.5),
                                                            crisis_response_budget_seconds=turn.get("budget", 1.5)))
    b.compile()
    return b

def _hang(monkeypatch, b, seconds):
    async def slow(*a, **k):
        await asyncio.sleep(seconds)
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", slow)

async def _run(b, text, session="s1", thread="t-079"):
    return await b.run_turn(user_id=USER, thread_id=thread, session_id=session, message=text, turn_id=TURN, channel="rest")

@pytest.mark.asyncio
async def test_mvp_079_graph_exception_still_returns_resources_and_one_page(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    async def boom(*a, **k):
        raise RuntimeError("checkpoint write failed")
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", boom)
    r = await _run(b, "I want to kill myself")
    assert r.crisis is not None and cs.risk_rank(r.crisis.level) >= cs.risk_rank("HIGH")
    assert r.crisis.resources and "988" in r.response_text
    assert "graph_error" in r.degraded
    assert len(plane.events) == 1 and plane.events[0].metadata.correlation_id == TURN

@pytest.mark.asyncio
async def test_mvp_079_non_crisis_graph_exception_gives_safe_reply(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    async def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", boom)
    r = await _run(b, "hello there friend")
    assert (r.crisis, r.degraded, r.response_text, plane.events) == (None, ["graph_error"], SAFE_FALLBACK_TEXT, [])

@pytest.mark.asyncio
async def test_mvp_079_turn_deadline_bounds_a_hung_graph(monkeypatch):
    b = _builder(monkeypatch, deadline=0.5)
    _hang(monkeypatch, b, 10)
    t0 = time.perf_counter()
    r = await _run(b, "hello there friend")
    assert time.perf_counter() - t0 < 1.0 and "turn_timeout" in r.degraded

@pytest.mark.asyncio
async def test_mvp_079_crisis_reply_within_budget_when_graph_hangs(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane, deadline=20.0, budget=1.0)
    _hang(monkeypatch, b, 10)
    t0 = time.perf_counter()
    r = await _run(b, "I want to kill myself")
    assert time.perf_counter() - t0 < 1.4
    assert r.crisis is not None and len(plane.events) == 1

@pytest.mark.asyncio
async def test_mvp_017_every_message_gets_the_safety_check_and_it_can_raise(monkeypatch):
    safety, plane = FakeSafety(level="HIGH"), FakePlane()
    b = _builder(monkeypatch, plane=plane, safety=safety)
    r = await _run(b, "hello there friend")
    assert safety.calls == [TURN]
    assert r.crisis is not None and r.crisis.level == "HIGH" and len(plane.events) == 1

@pytest.mark.asyncio
async def test_mvp_017_safety_timeout_is_degraded_not_silent(monkeypatch):
    b = _builder(monkeypatch, safety=FakeSafety(status="timeout"))
    r = await _run(b, "hello there friend")
    assert "safety_service_timeout" in r.degraded and r.crisis is None

@pytest.mark.asyncio
async def test_mvp_067_non_uuid_session_still_pages(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    await _run(b, "I want to kill myself", session="session-abc", thread="t-067")
    ev = plane.events[0]
    assert ev.session_id == session_uuid("session-abc") and ev.user_id == UUID(USER)
    assert set(ev.trigger_indicators) <= set(cs.CATEGORIES)

@pytest.mark.asyncio
async def test_mvp_069_publish_failure_is_reported(monkeypatch):
    b = _builder(monkeypatch, plane=FakePlane(exc=EventPublishError("outbox down")))
    with capture_logs() as logs:
        r = await _run(b, "I want to kill myself")
    assert "crisis_alert_publish_failed" in r.degraded and r.crisis is not None
    assert not [l for l in logs if l["event"] == "orchestrator_crisis_event_emitted"]

@pytest.mark.asyncio
async def test_mvp_069_slow_publish_does_not_delay_resources(monkeypatch):
    plane = FakePlane(delay=1.5)
    b = _builder(monkeypatch, plane=plane, budget=1.0)
    t0 = time.perf_counter()
    r = await _run(b, "I want to kill myself")
    assert time.perf_counter() - t0 < 1.4 and "crisis_alert_publish_pending" in r.degraded
    await asyncio.sleep(0.6)
    assert len(plane.events) == 1

@pytest.mark.asyncio
async def test_mvp_069_invoke_alone_never_claims_an_emit():
    # Today's API only: a UUID session lets today's _maybe_emit_crisis_event reach its success log.
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=False))
    b.compile()
    with capture_logs() as logs:
        await b.invoke(create_initial_state(USER, "0b6f1c2a-0000-4000-8000-0000000000aa", "I want to kill myself"))
    assert not [l for l in logs if l["event"] == "orchestrator_crisis_event_emitted"]

@pytest.mark.asyncio
async def test_mvp_079_one_event_per_turn_and_identity_bound_for_hops(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane, safety=FakeSafety(level="CRITICAL"))
    r = await _run(b, "I want to kill myself")
    assert len(plane.events) == 1 and plane.events[0].crisis_level == "CRITICAL" and r.crisis.level == "CRITICAL"
    await _run(b, "hello there friend", thread="t-bind")
    assert SEEN == {"user": USER, "turn": str(TURN)}

@pytest.mark.asyncio
async def test_mvp_079_agents_lists_only_real_contributions(monkeypatch):
    assert (await _run(_builder(monkeypatch), "hello there friend", thread="t-a1")).agents == ["chat"]
    assert (await _run(_builder(monkeypatch, fallback_chat=True), "hello there friend", thread="t-a2")).agents == []

@pytest.mark.asyncio
async def test_crisis_review_f2_outbox_outage_is_retried_after_the_turn(monkeypatch):
    plane = FakePlane(failures=1)
    b = _builder(monkeypatch, plane=plane)
    r = await _run(b, "I want to kill myself")
    assert r.crisis is not None and "crisis_alert_publish_failed" in r.degraded and plane.events == []
    assert b.publish_retrier.pending() == 1
    assert await b.publish_retrier.run_once() == 1
    assert len(plane.events) == 1 and plane.events[0].metadata.correlation_id == TURN

@pytest.mark.asyncio
async def test_crisis_review_f7_slow_check_and_hung_graph_stay_inside_one_deadline(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane, safety=FakeSafety(status="timeout", delay=2.0),
                 deadline=20.0, budget=2.8, check_timeout=2.0)
    _hang(monkeypatch, b, 10)
    t0 = time.perf_counter()
    r = await _run(b, "I want to kill myself")
    assert time.perf_counter() - t0 < 3.0                      # 2.0 s /check plus the graph never exceed entry + 2.8 s
    assert r.crisis is not None and "safety_service_timeout" in r.degraded and len(plane.events) == 1

@pytest.mark.asyncio
async def test_crisis_review_f12_provider_raised_high_without_layer_pages_once(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    async def provider_raised(state, *a, **k):                 # a diagnosis-only HIGH: no layer or confidence in the flags
        flags = {**state["safety_flags"], "risk_level": "HIGH", "categories": ["self_harm"], "signal_source": "diagnosis",
                 "detection_layer": None, "confidence": None}
        return {**state, "safety_flags": flags, "final_response": "Tell me more."}
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", provider_raised)
    r = await _run(b, "hello there friend")
    assert r.crisis is not None and r.crisis.level == "HIGH" and "graph_error" not in r.degraded
    (ev,) = plane.events
    assert (ev.detection_layer, float(ev.confidence), ev.metadata.correlation_id) == (3, 0.6, TURN)

@pytest.mark.asyncio
async def test_crisis_review_f3_refused_message_gets_resources_and_one_page_without_the_graph(monkeypatch):
    plane, safety = FakePlane(), FakeSafety()
    b = _builder(monkeypatch, plane=plane, safety=safety)
    async def never(*a, **k):
        raise AssertionError("the graph must not run for a refused message")
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", never)
    info = await b.crisis_for_refused_message(user_id=USER, session_id="s1", message="I want to kill myself",
                                              turn_id=TURN, channel="rest")
    assert info is not None and cs.risk_rank(info.level) >= cs.risk_rank("HIGH") and info.resources
    assert len(plane.events) == 1 and plane.events[0].metadata.correlation_id == TURN and plane.events[0].detection_layer == 1
    assert safety.calls == []                                  # refusal paths use the local screen only

@pytest.mark.asyncio
async def test_crisis_review_f3_benign_refused_message_gets_nothing(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    assert await b.crisis_for_refused_message(user_id=USER, session_id="s1", message="hello there friend",
                                              turn_id=TURN, channel="ws") is None
    assert plane.events == []

@pytest.mark.asyncio
@pytest.mark.parametrize("locale", [None, "en-US", "en_us"])
async def test_d29_en_us_or_a_missing_locale_keeps_todays_resources(monkeypatch, locale):
    b = _builder(monkeypatch)
    r = await b.run_turn(user_id=USER, thread_id="t-d29-us", session_id="s1", message="I want to kill myself",
                         turn_id=TURN, channel="rest", locale=locale)
    assert r.crisis.resources == crisis_resources("en-US") and r.crisis.resources_version == RESOURCES_VERSION

@pytest.mark.asyncio
async def test_d29_unsupported_locale_gets_the_fallback_and_still_pages_when_the_graph_fails(monkeypatch):
    plane = FakePlane()
    b = _builder(monkeypatch, plane=plane)
    async def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(b.get_compiled_graph(), "ainvoke", boom)
    r = await b.run_turn(user_id=USER, thread_id="t-d29-fr", session_id="s1", message="I want to kill myself",
                         turn_id=TURN, channel="rest", locale="fr-FR")
    assert cs.risk_rank(r.crisis.level) >= cs.risk_rank("HIGH") and len(plane.events) == 1   # fusion and paging unchanged
    assert r.crisis.resources == crisis_resources("fr-FR") and r.crisis.resources_version == FALLBACK_RESOURCES_VERSION
    assert "988" not in r.response_text and "911" not in r.response_text
    assert all(res["contact"] in r.response_text for res in r.crisis.resources)

@pytest.mark.asyncio
async def test_d29_refused_message_uses_the_same_locale(monkeypatch):
    b = _builder(monkeypatch)
    info = await b.crisis_for_refused_message(user_id=USER, session_id="s1", message="I want to kill myself",
                                              turn_id=TURN, channel="ws", locale="de-DE")
    assert info.resources == crisis_resources("de-DE") and info.resources_version == FALLBACK_RESOURCES_VERSION
    default = await b.crisis_for_refused_message(user_id=USER, session_id="s1", message="I want to kill myself",
                                                 turn_id=TURN, channel="ws")
    assert default.resources == crisis_resources("en-US") and default.resources_version == RESOURCES_VERSION
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_079_run_turn.py -o "addopts=" -p no:logging -q`
Expected: FAIL at import (`TurnSettings`, `safety_copy` missing). `test_mvp_069_invoke_alone_never_claims_an_emit` uses only today's API and fails because `_maybe_emit_crisis_event` logs `orchestrator_crisis_event_emitted` although the EventBus has no subscriber (MVP-069); the run-turn tests then fail because no runner exists, so a graph exception reaches api.py's bare 500 with no resources and no event (MVP-079). The five `crisis_review` tests fail at the same import; they pin that a failed publish is retried after the turn (F2), the crisis budget counts from `run_turn` entry so a 2 s `/check` cannot push the answer past 3 s (F7), a provider-raised HIGH without a layer still pages and keeps the crisis object (F12), and a refused message is screened and paged without the graph (F3). Today no edge refusal screens anything and `_maybe_emit_crisis_event` drops a failed publish. The three `test_d29_*` tests (five cases) fail at the same import. They pin D-29: en-US and a missing locale keep today's resources and version; `fr-FR` gets the generic fallback, with no 988 or 911, even when the graph fails, and still pages once; a refused message uses the same locale. Today `_CrisisResourceManager` returns one hard-coded US list whatever the user's locale.
- [ ] **Step 3: Implement** `run_turn(...)`, `crisis_for_refused_message(...)` and `from_runtime(...)` with the signatures above. Order inside `run_turn`, all under `acting_as_user(user_id)` and `structlog.contextvars.bound_contextvars(turn_id=str(turn_id))`: (0) `entry = loop.time()`; (1) `screen_message(...)` (never inside the thread lock); (2) when the fused level is HIGH or above, set `deadline = entry + crisis_response_budget_seconds` (else `entry + turn_timeout_seconds`), build the event and start `publish_crisis_event(plane, event, retrier=self.publish_retrier)` as a task; (3) `asyncio.wait_for(self.invoke(state, thread_id), max(0.0, deadline - loop.time()))` where `state` is `create_initial_state(..., turn_id=turn_id)` with `safety_flags` set to the fused screen; when the deadline has already passed the graph is skipped and the turn takes the timeout path; (4) the final level is the max of the fused level and the graph's `safety_flags.risk_level`; a HIGH/CRITICAL the graph raised that was not yet published is published now, at most once per turn, with the F12 layer and confidence defaults, inside a `try` that turns any exception into a CRITICAL `orchestrator_crisis_event_build_failed` log plus `crisis_alert_publish_failed` and never drops the crisis object; (5) await the publish task with `asyncio.wait_for(asyncio.shield(task), max(0.0, min(deadline, publish_started + crisis_response_budget_seconds) - loop.time()))`, so a publish started in step 4 never holds the reply longer than the crisis budget either: failure gives `crisis_alert_publish_failed` (the event is already with the retrier), still running gives `crisis_alert_publish_pending` while a done-callback logs the outcome and hands a failed outcome's event to the retrier; (6) `TurnResult`: `crisis` set when the final level is ELEVATED or above (`categories`, and `resources` and `resources_version` from one `crisis_resource_set(resolved_locale)` call, made only when a crisis object or crisis reply is built, so a fallback is logged and counted only when it is served; `resolved_locale = resolve_locale(locale)` is computed at step (0) and never raises; the same resources feed `crisis_reply_text`, and `create_initial_state(..., locale=resolved_locale)` hands the locale to the graph so `crisis_handler_node` and the WS-10.6 and WS-10.8 copy use the same set), `agents` = sorted distinct `agent_type` of this turn's successful results, excluding `supervisor`, `aggregator` and `memory`, the precheck/postcheck bookkeeping results (`metadata.phase` in `{"precheck", "postcheck"}`) and any result with `metadata.fallback_mode`; `degraded` = screen codes + graph `degraded` + runner codes, de-duplicated. On any exception or timeout from the graph: HIGH/CRITICAL or ELEVATED gives `crisis_reply_text(level, resources)`, anything else gives `SAFE_FALLBACK_TEXT`, plus `graph_error` or `turn_timeout`. Increment `solace_crisis_detected_total{source="orchestrator", level}` once per HIGH/CRITICAL turn. `crisis_for_refused_message` follows Produces and shares steps (2) and (5) with `run_turn`, bounded by its own `entry + crisis_response_budget_seconds`, and resolves `locale` the same way. `from_runtime` builds `TurnSettings()`, a pooled `SafetyCheckClient` on `ServiceEndpoints().safety_service_url` (registered with `runtime.register_shutdown("safety_check_client", ...)`), binds `runtime.event_plane`, creates `CrisisPublishRetrier(runtime.event_plane)` and registers `runtime.register_background_task("crisis_publish_retry", asyncio.create_task(retrier.run()))` (its death fails `/live`), uses `InMemorySaver` in W1 and compiles; it is `async` so W2 can open the Postgres pool without an interface change. Copy in `safety_copy.py` is provisional under D-05 and lists each resource's name and contact.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (22 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.5:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.6: Bounded crisis watch instead of a permanent crisis lock (D-06)
**Covers:** MVP-019
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-10.5
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-06 (default assumed: after HIGH/CRITICAL the thread watches the next 3 user turns and 30 minutes; it ends only when both have passed (the more protective reading); during the watch every reply carries the safety footer, a LOW fused level is shown as ELEVATED, and a failed safety-service check is treated as ELEVATED; a new HIGH/CRITICAL pages again and restarts the watch)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`run_turn` reads and writes `crisis_watch` inside the thread lock; `safety_postcheck_node` appends `safety_footer()` when `safety_flags.watch_active` or the level is ELEVATED, except on crisis-handler replies)
- Modify: `services/orchestrator_service/src/response/safety_copy.py` (`safety_footer` wording)
- Test: `services/orchestrator_service/tests/test_mvp_019_crisis_watch.py`
**Interfaces:**
- Consumes: `crisis_watch` channel (WS-10.1); `TurnResult` (WS-10.5).
- Produces: `crisis_watch` value `{"started_at": iso, "until": iso, "turns_remaining": int, "level": str}`; module constants `CRISIS_WATCH_TURNS = 3`, `CRISIS_WATCH_MINUTES = 30` (clinical values, not environment settings); `TurnResult.crisis_watch_until` populated while active.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_019_crisis_watch.py`
```python
"""MVP-019 (D-06): a crisis starts a bounded watch: footer and lowered threshold, no lock, no re-paging."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import pytest
from solace_clinical import RESOURCES_VERSION, ScreenResult
from solace_common.enums import CrisisLevel
from services.orchestrator_service.src.langgraph import crisis_screen as cs
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings
from services.orchestrator_service.src.langgraph.state_schema import AgentResult, AgentType

USER = "8a3f0d4e-0000-4000-8000-000000000001"

class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
    def __call__(self):
        return self.now

class FakePlane:
    def __init__(self):
        self.events = []
    async def publish(self, event, *, conn=None):
        self.events.append(event)
        return event.metadata.event_id

class FakeSafety:
    def __init__(self):
        self.status = "ok"
    async def check(self, **kw):
        return cs.RemoteCheck(status=self.status, level="NONE" if self.status == "ok" else None, categories=frozenset(),
                              risk_score=0.0, detection_layer=1, requires_escalation=False)
    async def aclose(self):
        pass

@pytest.fixture
def env(monkeypatch):
    async def chat(state):
        return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True,
                                              response_content="Tell me more.", confidence=0.7).to_dict()]}
    async def noop(state):
        return {}
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    nodes = {k: noop for k in ("diagnosis", "therapy", "personality", "safety", "assessment", "emotion")}
    nodes["chat"] = chat
    monkeypatch.setattr(gb, "_import_agent_nodes", lambda: nodes)
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: None)
    clock, plane, safety = Clock(), FakePlane(), FakeSafety()
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True), event_plane=plane,
                                 safety_client=safety, turn_settings=TurnSettings(), clock=clock)
    b.compile()
    async def turn(text):
        return await b.run_turn(user_id=USER, thread_id="t-watch", session_id="s1", message=text,
                                turn_id=uuid4(), channel="ws")
    return turn, clock, plane, safety, monkeypatch

@pytest.mark.asyncio
async def test_mvp_019_watch_footer_for_three_turns_without_repaging(env):
    turn, clock, plane, _, _ = env
    first = await turn("I want to kill myself")
    assert first.crisis is not None and len(plane.events) == 1
    for _ in range(3):
        clock.now += timedelta(minutes=11)
        r = await turn("can we talk about my job?")
        assert "Tell me more." in r.response_text and "988" in r.response_text
        assert r.safety_flags["crisis_detected"] is False and r.crisis is None and r.crisis_watch_until is not None
    clock.now += timedelta(minutes=1)
    after = await turn("can we talk about my job?")
    assert "988" not in after.response_text and after.crisis_watch_until is None
    assert len(plane.events) == 1

@pytest.mark.asyncio
async def test_mvp_019_watch_needs_both_turns_and_minutes(env):
    turn, clock, _, _, _ = env
    await turn("I want to kill myself")
    for _ in range(3):
        await turn("can we talk about my job?")
    clock.now += timedelta(minutes=10)
    assert "988" in (await turn("can we talk about my job?")).response_text
    clock.now += timedelta(minutes=31)
    assert "988" not in (await turn("can we talk about my job?")).response_text

@pytest.mark.asyncio
async def test_mvp_019_watch_lowers_the_elevated_threshold(env):
    turn, _, _, _, monkeypatch = env
    await turn("I want to kill myself")
    low = ScreenResult(level=CrisisLevel.LOW, categories=frozenset(), rule_ids=("t",), negated_rule_ids=(),
                       lexicon_version="test")
    monkeypatch.setattr(cs, "screen_local", lambda text: low)
    r = await turn("a bit flat today")
    assert r.crisis is not None and r.crisis.level == "ELEVATED" and r.crisis.resources_version == RESOURCES_VERSION

@pytest.mark.asyncio
async def test_mvp_019_failed_check_during_watch_shows_resources(env):
    turn, _, _, safety, _ = env
    await turn("I want to kill myself")
    safety.status = "timeout"
    r = await turn("can we talk about my job?")
    assert r.crisis is not None and r.crisis.level == "ELEVATED" and "safety_service_timeout" in r.degraded

@pytest.mark.asyncio
async def test_mvp_019_new_crisis_during_watch_pages_again(env):
    turn, _, plane, _, _ = env
    await turn("I want to kill myself")
    await turn("can we talk about my job?")
    await turn("I want to kill myself")
    assert len(plane.events) == 2
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_019_crisis_watch.py -o "addopts=" -p no:logging -q`
Expected: FAIL: after WS-10.5 the post-crisis turns are fresh but carry no footer (`assert "988" in r.response_text` fails), `crisis_watch_until` is `None`, and the LOW-during-watch turn has `crisis is None`; the constructor rejects `clock=` until it is wired to the watch.
- [ ] **Step 3: Implement** the watch in `run_turn`: inside the thread lock read the previous `crisis_watch` with `await graph.aget_state(config)` (a read failure means no watch and degraded `checkpoint_unavailable`); a HIGH/CRITICAL turn writes a new watch (`until = now + 30 min`, `turns_remaining = 3`); at the start of any other turn an existing watch is expired (and cleared) only when `turns_remaining == 0 and now >= until`, otherwise it is active: it raises LOW to ELEVATED, raises the level to ELEVATED when the safety check failed, sets `safety_flags.watch_active=True` in the graph input and decrements `turns_remaining` (not below 0); the new value goes into the graph input (last-write channel). `safety_postcheck_node` appends `safety_footer(crisis_resources(state["locale"]))` when `watch_active` or the level is ELEVATED (D-29: the turn's resolved locale from WS-10.5, en-US for these tests), so the checkpointed reply equals the shown reply. The clock comes from the constructor `clock` (default `datetime.now(timezone.utc)`).
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (5 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.6:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.7: Supervisor: LLM intent on every non-crisis turn, LLM crisis routed to the crisis path, no personality on crisis
**Covers:** MVP-023, MVP-024, MVP-173, MVP-295
**Severity:** blocker
**Wave:** 1
**Depends on:** WS-10.5, WS-09.3, WS-04.3
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-22 (default assumed: the LLM classifier runs for every non-crisis message when available and its `crisis_disclosure` routes to the crisis path at HIGH); D-05 (default assumed: no LLM confidence floor for crisis; an LLM category outside `CATEGORIES` becomes `imminent_danger`)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/supervisor.py` (`make_decision` always calls `refine_with_llm` for non-crisis input; LLM `crisis_disclosure` wins at any confidence; the prompt asks for a `category` from `CATEGORIES`; `process()` writes crisis `safety_flags` (`risk_level="HIGH"`, `detection_layer=3`, `signal_source="llm_intent"`, `categories`, `confidence`) and `degraded=["llm_unavailable"]` on `LLMUnavailableError`; `AgentRouter.select_agents` never adds PERSONALITY when the intent is CRISIS_DISCLOSURE or the flags are HIGH/CRITICAL; word-bounded progress/assessment/coping/education matchers (absorbed MVP-077); `solace_llm_fallback_total{service="orchestrator-service", reason}` incremented on LLM failure; the empty-reply test `if not response:` in `IntentClassifier.refine_with_llm` (supervisor.py:230) is removed, because IF-15 `generate()` raises instead of returning `""` (CR-7 of WS-09))
- Test: `services/orchestrator_service/tests/test_mvp_024_supervisor_llm_routing.py` (new); `services/orchestrator_service/tests/test_supervisor.py` (crisis-disclosure routing now asserts `agents == [AgentType.SAFETY]`)
**Interfaces:**
- Consumes: IF-15 `LLMUnavailableError` (raised by `UnifiedLLMClient.generate`; WS-09.3); IF-18 counter object `LLM_FALLBACK` (`solace_llm_fallback_total{service,reason}`) from `solace_infrastructure.runtime.metrics` (WS-04.3); `screen_local`, `RemoteCheck` (WS-10.2/10.4); `run_turn` (WS-10.5).
- Produces: supervisor `safety_flags` update for LLM crisis; LLM prompt JSON `{"intent": str, "confidence": float, "category": str | null}`; the supervisor half of CR-7 (call sites catch `LLMUnavailableError`, never test for `""`), requested by WS-09 for WS-09.4.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_024_supervisor_llm_routing.py`
```python
"""MVP-024 / MVP-023 / MVP-173 / MVP-295: the LLM classifier always runs and its crisis call is honoured."""
from __future__ import annotations
import json
from uuid import UUID
import pytest
from solace_clinical import ScreenResult
from solace_common.enums import CrisisLevel
from services.shared.infrastructure.llm_client import LLMUnavailableError
from services.orchestrator_service.src.langgraph import crisis_screen as cs
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings
from services.orchestrator_service.src.langgraph.state_schema import AgentType, IntentType, create_initial_state
from services.orchestrator_service.src.langgraph.supervisor import AgentRouter, IntentClassifier, SupervisorAgent, SupervisorSettings

USER = "8a3f0d4e-0000-4000-8000-000000000001"
TURN = UUID("4b6f1c2a-0000-4000-8000-000000000002")

class FakeLLM:
    is_available = True
    def __init__(self, reply=None, exc=None):
        self.calls, self.reply, self.exc = [], reply, exc
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return json.dumps(self.reply)

CRISIS = {"intent": "crisis_disclosure", "confidence": 0.4, "category": "suicidal_ideation"}

@pytest.mark.asyncio
async def test_mvp_024_llm_consulted_even_when_keywords_are_confident():
    llm = FakeLLM({"intent": "treatment_inquiry", "confidence": 0.7, "category": None})
    await SupervisorAgent(llm_client=llm).make_decision("Things are getting worse and I need help")
    assert len(llm.calls) == 1

@pytest.mark.asyncio
@pytest.mark.parametrize("text", [
    "I need help, nothing feels worth it",
    "honestly might just kms tonight",                           # slang
    "i wanna dye, im so tired of evrything",                     # misspelling
    "ya no quiero vivir",                                        # non-English
    "\U0001F52A\U0001FA78 tonight, done with all of it",         # emoji
    "my friend said 'I want to die' and honestly same",          # quoted speech that is also a disclosure
])
async def test_mvp_024_llm_crisis_wins_at_any_confidence(text):
    # Review focus 1: whatever the lexicon misses, the LLM is asked and its crisis call wins (over-detect, never miss).
    d = await SupervisorAgent(llm_client=FakeLLM(CRISIS)).make_decision(text)
    assert d.intent == IntentType.CRISIS_DISCLOSURE and d.requires_safety_override is True
    assert d.selected_agents == [AgentType.SAFETY]

@pytest.mark.asyncio
async def test_mvp_024_llm_outage_is_degraded_and_keywords_stand():
    out = await SupervisorAgent(llm_client=FakeLLM(exc=LLMUnavailableError("down"))).process(
        create_initial_state(USER, "s1", "Things are getting worse and I need help"))
    assert out["degraded"] == ["llm_unavailable"] and out["intent"] != IntentType.CRISIS_DISCLOSURE.value

def test_mvp_077_heuristics_are_word_bounded():
    intent, _, _ = IntentClassifier(SupervisorSettings()).classify("I keep forgetting my keys")
    assert intent != IntentType.PROGRESS_UPDATE

def test_mvp_173_no_personality_on_crisis_routes():
    r = AgentRouter(SupervisorSettings())
    assert r.select_agents(IntentType.CRISIS_DISCLOSURE, {})[0] == [AgentType.SAFETY]
    assert r.select_agents(IntentType.GENERAL_CHAT, {"risk_level": "HIGH"})[0] == [AgentType.SAFETY]

class FakePlane:
    def __init__(self):
        self.events = []
    async def publish(self, event, *, conn=None):
        self.events.append(event)
        return event.metadata.event_id

class DownSafety:
    async def check(self, **kw):
        return cs.RemoteCheck(status="error", level=None, categories=frozenset(), risk_score=None,
                              detection_layer=None, requires_escalation=False)
    async def aclose(self):
        pass

@pytest.mark.asyncio
async def test_mvp_023_llm_crisis_reaches_crisis_handler_and_pages_once(monkeypatch):
    none = ScreenResult(level=CrisisLevel.NONE, categories=frozenset(), rule_ids=(), negated_rule_ids=(),
                        lexicon_version="test")
    monkeypatch.setattr(cs, "screen_local", lambda text: none)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: FakeLLM({**CRISIS, "confidence": 0.95}))
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    plane = FakePlane()
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True), event_plane=plane,
                                 safety_client=DownSafety(), turn_settings=TurnSettings())
    b.compile()
    r = await b.run_turn(user_id=USER, thread_id="t-023", session_id="s1", message="I'm thinking about ending it",
                         turn_id=TURN, channel="rest")
    assert r.crisis is not None and r.crisis.level == "HIGH" and "988" in r.response_text
    assert "temporary difficulties" not in r.response_text
    assert len(plane.events) == 1 and plane.events[0].detection_layer == 3
    assert plane.events[0].trigger_indicators == ["suicidal_ideation"]
    snap = await b.get_compiled_graph().aget_state({"configurable": {"thread_id": "t-023"}})
    assistant = [m for m in snap.values["messages"] if m["role"] == "assistant"]
    postchecks = [x for x in snap.values["agent_results"] if x.get("metadata", {}).get("phase") == "postcheck"]
    assert len(assistant) == 1 and len(postchecks) == 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_024_supervisor_llm_routing.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `len(llm.calls) == 0` (LLM only below 0.6 keyword confidence, MVP-024); the low-confidence crisis is overridden by the keyword intent; `select_agents` returns `[SAFETY, PERSONALITY]` (MVP-173); "forgetting" matches the "getting" substring; the graph test answers "temporary difficulties" with `crisis is None` and no event (MVP-023) and records two postcheck results and two assistant messages (MVP-295).
- [ ] **Step 3: Implement** in `supervisor.py`: `make_decision` calls `refine_with_llm` whenever the keyword intent is not CRISIS_DISCLOSURE and the client is available; `refine_with_llm` returns `(CRISIS_DISCLOSURE, llm_confidence, category)` whenever the LLM says crisis, never lowers a keyword crisis, and lets `LLMUnavailableError` and parse failures fall back to the keyword result with degraded `llm_unavailable`; `process()` returns the crisis `safety_flags` update so `route_after_safety_agent` sends the turn to `crisis_handler`; `AgentRouter.select_agents` returns `[SAFETY]` alone for crisis intent or HIGH/CRITICAL flags; replace the `in` substring tests in `_is_progress_related`, `_is_assessment_request`, `_is_coping_request` and `_is_psychoeducation_request` with `\b`-bounded regexes. The out-of-lexicon cases in `test_mvp_024_llm_crisis_wins_at_any_confidence` are routing tests, not lexicon claims: a case the lexicon already rates HIGH passes through the keyword path, every other case must reach CRISIS_DISCLOSURE through the LLM; never remove a case to make the test pass.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (11 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.7:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.8: Safety postcheck never inverts protective text and replaces harmful replies whole
**Covers:** MVP-068
**Severity:** high
**Wave:** 1
**Depends on:** WS-10.6
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** D-05 (default assumed: provisional harmful-output rules (an instruction cue near a self-harm method term) and replacement copy, both submitted for clinical sign-off); D-29 (decided 2026-10-19: the replacement lists the turn's locale resources)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/crisis_screen.py` (`screen_reply`, `ReplyScreen`, `HARMFUL_REPLY_RULES`)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`safety_postcheck_node`: delete `_HARMFUL_REPLACEMENTS` substitution; harmful reply replaced as a whole; crisis-handler bypass and the WS-10.6 footer kept)
- Modify: `services/orchestrator_service/src/response/safety_copy.py` (`harmful_reply_replacement(resources) -> str` and `HARMFUL_REPLY_REPLACEMENT = harmful_reply_replacement(crisis_resources("en-US"))`, D-29)
- Test: `services/orchestrator_service/tests/test_mvp_068_postcheck.py` (new); `services/orchestrator_service/tests/test_graph_builder.py` and `tests/integration/test_e2e_graph.py` / `tests/integration/test_therapy_flow.py` (the "You should just give up" tests now assert the reply is unchanged and `filtered is False`)
**Interfaces:**
- Produces: `screen_reply(text: str) -> ReplyScreen`; `@dataclass(frozen=True) ReplyScreen(harmful: bool, rule_ids: tuple[str, ...])`; postcheck metadata `{"phase": "postcheck", "filtered": bool, "rule_ids": [...]}`; `safety_copy.harmful_reply_replacement(resources) -> str` (D-29: the replacement lists the turn's locale resources, `state["locale"]` from WS-10.5).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_068_postcheck.py`
```python
"""MVP-068: postcheck leaves protective text alone and blocks method instructions as a whole."""
from __future__ import annotations
import pytest
from services.orchestrator_service.src.langgraph.graph_builder import safety_postcheck_node
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state
from solace_clinical import crisis_resources
from services.orchestrator_service.src.response.safety_copy import HARMFUL_REPLY_REPLACEMENT, harmful_reply_replacement

def _post(reply: str, locale: str = "en-US", **flags) -> dict:
    state = create_initial_state("user-1", "s1", "hello", locale=locale)
    state["final_response"] = reply
    state["safety_flags"] = {**state["safety_flags"], **flags}
    return safety_postcheck_node(state)

@pytest.mark.parametrize("reply", [
    "There is no point in hurting yourself, please reach out to 988.",
    "Please do not give up on yourself.",
    "You should talk to someone you trust.",
])
def test_mvp_068_protective_statements_are_untouched(reply):
    out = _post(reply)
    assert out["final_response"] == reply and out["agent_results"][0]["metadata"]["filtered"] is False

def test_mvp_068_method_instructions_are_replaced_whole():
    out = _post("Here is how to tie a noose: first take a rope and ...")
    assert out["final_response"] == HARMFUL_REPLY_REPLACEMENT and "988" in out["final_response"]
    meta = out["agent_results"][0]["metadata"]
    assert meta["filtered"] is True and meta["rule_ids"]

def test_mvp_068_crisis_handler_reply_is_preserved():
    text = "I'm really concerned. Please call or text 988."
    assert _post(text, crisis_detected=True)["final_response"] == text

def test_d29_harmful_reply_replacement_follows_the_turn_locale():
    out = _post("Here is how to tie a noose: first take a rope and ...", locale="fr-FR")
    assert out["final_response"] == harmful_reply_replacement(crisis_resources("fr-FR"))
    assert "988" not in out["final_response"] and HARMFUL_REPLY_REPLACEMENT == harmful_reply_replacement(crisis_resources("en-US"))
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_068_postcheck.py -o "addopts=" -p no:logging -q`
Expected: FAIL: "There is no point in hurting yourself..." becomes "There is it may not feel like it right now, but there is hope in hurting yourself..." and the other two are rewritten (`filtered True`); the noose reply passes through unchanged. Collection first fails with `ImportError: cannot import name 'harmful_reply_replacement'`; the D-29 test pins that the replacement lists the turn's locale resources, never 988 for a `fr-FR` user.
- [ ] **Step 3: Implement** `screen_reply(text)` over `normalize_text(text)` with rules that need an instruction cue (`how to`, `steps to`, `here is how`, `first,`, `you can`) within 8 tokens of a method term (`noose`, `hang yourself`, `lethal dose`, `overdose on`, `slit`, `cut deeper`, `jump from`); `safety_postcheck_node` replaces a harmful reply with `harmful_reply_replacement(crisis_resources(state["locale"]))` (copy plus the turn's locale resources; for en-US this equals `HARMFUL_REPLY_REPLACEMENT`) and otherwise returns the text unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (6 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.8:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.9: Slot 017 applies the LangGraph checkpoint schema
**Covers:** MVP-066
**Severity:** high
**Wave:** 2
**Depends on:** WS-11.6, WS-02.3, WS-01.2, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-01 (default assumed: Alembic is the only schema of record; `AsyncPostgresSaver.setup()` never runs at runtime)
**Real infra:** yes
**Files:**
- Modify: `migrations/versions/017_orchestrator_checkpoints.py` (`upgrade()`/`downgrade()` only; revision ids untouched)
- Test: `tests/realinfra/orchestrator-graph-crisis/test_mvp_066_checkpoint_slot.py`, `services/orchestrator_service/tests/test_mvp_066_slot017_ddl.py` (new)
**Interfaces:**
- Consumes: IF-03 slot `017_orchestrator_checkpoints` (after `016_orchestrator_threads`), roles `solace_migrator`/`solace_app`; IF-19 `pg_migrated -> PgHandle(dsn_app, dsn_migrator, dsn_superuser, app_pool)`; `langgraph.checkpoint.postgres.base.BasePostgresSaver.MIGRATIONS` of the locked version.
- Produces: tables `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` with every library migration recorded; module constant `CHECKPOINT_MIGRATIONS: list[str]` in the slot file.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/orchestrator-graph-crisis/test_mvp_066_checkpoint_slot.py`
```python
"""MVP-066: slot 017 creates the LangGraph checkpoint tables so no runtime DDL is needed."""
from __future__ import annotations
import asyncpg
import pytest
from langgraph.checkpoint.postgres.base import BasePostgresSaver

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
TABLES = {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}

@pytest.mark.asyncio
async def test_mvp_066_checkpoint_tables_exist_owned_by_migrator(pg_migrated):
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        rows = await conn.fetch("SELECT tablename, tableowner FROM pg_tables WHERE schemaname='public' AND tablename = ANY($1)", list(TABLES))
        assert {r["tablename"]: r["tableowner"] for r in rows} == {t: "solace_migrator" for t in TABLES}
        assert await conn.fetchval("SELECT max(v) FROM checkpoint_migrations") == len(BasePostgresSaver.MIGRATIONS) - 1
    finally:
        await conn.close()

@pytest.mark.asyncio
async def test_mvp_066_app_role_has_dml_but_no_ddl(pg_migrated):
    async with pg_migrated.app_pool.acquire() as conn:
        await conn.execute("INSERT INTO checkpoint_migrations (v) VALUES (9999)")
        await conn.execute("DELETE FROM checkpoint_migrations WHERE v = 9999")
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await conn.execute("DROP TABLE checkpoint_writes")
```
and in `services/orchestrator_service/tests/test_mvp_066_slot017_ddl.py`
```python
"""MVP-066: the slot's embedded DDL equals the locked library's migrations (drift guard)."""
import importlib.util
import pathlib
from langgraph.checkpoint.postgres.base import BasePostgresSaver

def test_mvp_066_slot017_embeds_the_library_migrations():
    path = pathlib.Path(__file__).resolve().parents[3] / "migrations/versions/017_orchestrator_checkpoints.py"
    spec = importlib.util.spec_from_file_location("slot017", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CHECKPOINT_MIGRATIONS == list(BasePostgresSaver.MIGRATIONS)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/orchestrator-graph-crisis/test_mvp_066_checkpoint_slot.py -o "addopts=" -p no:logging -q` and `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_066_slot017_ddl.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the scaffold slot is a no-op, so the table query returns `{}` and `checkpoint_migrations` does not exist (`UndefinedTableError`); the drift test fails with `AttributeError: module 'slot017' has no attribute 'CHECKPOINT_MIGRATIONS'`.
- [ ] **Step 3: Implement** `upgrade()` in the slot: `CHECKPOINT_MIGRATIONS` copied verbatim from the locked `BasePostgresSaver.MIGRATIONS`; execute them in order, running every statement containing `CONCURRENTLY` inside `with op.get_context().autocommit_block():`; then `INSERT INTO checkpoint_migrations (v) SELECT generate_series(0, <len-1>) ON CONFLICT DO NOTHING`. `downgrade()` drops `checkpoint_writes`, `checkpoint_blobs`, `checkpoints`, `checkpoint_migrations`. The docstring records that these tables have no `user_id` column, so isolation comes from `orchestrator_threads` ownership (WS-11.7) and owner-bound encryption (WS-10.10), not RLS.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/migrations -o "addopts=" -p no:logging -q` (the chain test: upgrade head, downgrade base, upgrade head), then `services/orchestrator_service/tests` with the unit command. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.9:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.10: Durable, encrypted Postgres checkpointer with TTL, prune and erasure
**Covers:** MVP-066
**Severity:** high
**Wave:** 2
**Depends on:** WS-10.9, WS-10.5, WS-10.6, WS-09.7, WS-04.4, WS-04.8, WS-05.2
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-24 (default assumed: idle threads are deleted after 30 days; erasure deletes immediately through `delete_threads`); D-04 (default assumed: checkpoint payloads are PHI and always encrypted)
**Real infra:** yes
**Files:**
- Create: `services/orchestrator_service/src/langgraph/checkpointer.py` (`PHICodecCipher`, `OwnerBoundEncryptedSerializer`, `create_checkpointer`, `CheckpointerHandle`, `build_conninfo`, `CheckpointJanitor`, `JanitorReport`, `CheckpointAccessError`; a module-level `from solace_security.phi_codec import PHICodec` import, the adopter import that WS-09.15's ADOPTERS lock checks for this path)
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (`from_runtime` builds the Postgres saver (strict: fatal on failure); `invoke` passes `durability="exit"`; new `delete_threads`, `get_thread_messages`; `TurnSettings` gains `checkpoint_ttl_days: int = 30`, `checkpoint_janitor_interval_seconds: int = 3600`)
- Modify: `services/orchestrator_service/src/langgraph/state_schema.py` (`MAX_THREAD_MESSAGES = 200`; `add_messages` keeps the newest 200)
- Test: `tests/realinfra/orchestrator-graph-crisis/test_mvp_066_postgres_checkpointer.py`, `services/orchestrator_service/tests/test_mvp_066_checkpointer_unit.py` (new)
**Interfaces:**
- Consumes: IF-13 `PHICodec.encrypt(value, *, field, owner_id)`/`decrypt(...)` and `get_phi_codec()` (WS-09.7), `ServiceRuntime.phi_codec` (populated by WS-04.8); `langgraph.checkpoint.serde.encrypted.EncryptedSerializer(cipher, serde)`, `CipherProtocol`; `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver`; `psycopg_pool.AsyncConnectionPool`; `solace_infrastructure.postgres.PostgresSettings` (read only); `solace_common.request_context.get_current_user_id`; IF-01 `StartupDependencyError`, `register_background_task`, `register_shutdown`.
- Produces (IF-10): `async create_checkpointer(*, environment: str, conninfo: str | None, codec: PHICodec | None, max_pool_size: int = 4, open_timeout_seconds: float = 10.0) -> CheckpointerHandle` (`.saver`, `.pool`, `.durable: bool`, `async aclose()`); `build_conninfo(settings: PostgresSettings) -> str` (libpq keywords incl. `sslmode`/`sslrootcert` via `psycopg.conninfo.make_conninfo`); `CheckpointJanitor(pool, *, ttl_days: int, keep_last: int = 1).run_once() -> JanitorReport(threads_deleted: int, checkpoints_pruned: int)`; `async OrchestratorGraphBuilder.delete_threads(self, thread_ids: Sequence[str]) -> int`; `async OrchestratorGraphBuilder.get_thread_messages(self, thread_id: str, *, limit: int = 50) -> list[dict[str, Any]]` (raises `CheckpointAccessError` when the caller is not the encryption owner). WS-10's part of CR-10, requested by WS-09 for WS-09.15: `checkpointer.py` imports `solace_security.phi_codec` and calls neither `get_phi_encryptor` nor `configure_phi_encryption`; the slot-017 tables have no `encryption_key_id` column because every IF-13 envelope (`enc:v1:<key_id>:...`) already names its key, so rotation (WS-09.8) decrypts old checkpoints by key id.

- [ ] **Step 1: Write the failing test** in `tests/realinfra/orchestrator-graph-crisis/test_mvp_066_postgres_checkpointer.py`
```python
"""MVP-066: checkpoints are durable in Postgres, ciphertext at rest, owner-bound, erasable and pruned."""
from __future__ import annotations
from uuid import uuid4
import asyncpg
import pytest
from solace_common.request_context import acting_as_user
from solace_security.phi_codec import get_phi_codec
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.checkpointer import CheckpointAccessError, CheckpointJanitor, create_checkpointer
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder, TurnSettings
from services.orchestrator_service.src.langgraph.state_schema import AgentResult, AgentType

pytestmark = [pytest.mark.realinfra, pytest.mark.integration]
MARK = "XYZZY-PHI-1234"

@pytest.fixture
def codec_env(monkeypatch):
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", "11" * 32)  # synthetic test key
    monkeypatch.setenv("ENCRYPTION_KEY_ID", "test-k1")
    monkeypatch.setenv("ENCRYPTION_SEARCH_HASH_SALT", "22" * 16)
    return get_phi_codec()

async def _builder(monkeypatch, pg, codec):
    async def chat(state):
        return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True,
                                              response_content=f"echo {MARK}", confidence=0.7).to_dict()]}
    async def noop(state):
        return {}
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    nodes = {k: noop for k in ("diagnosis", "therapy", "personality", "safety", "assessment", "emotion")}
    nodes["chat"] = chat
    monkeypatch.setattr(gb, "_import_agent_nodes", lambda: nodes)
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: None)
    handle = await create_checkpointer(environment="production", conninfo=pg.dsn_app, codec=codec)
    b = OrchestratorGraphBuilder(GraphBuilderSettings(), checkpointer=handle.saver, turn_settings=TurnSettings())
    b.compile()
    return b, handle

@pytest.mark.asyncio
async def test_mvp_066_ciphertext_at_rest_and_resume_after_restart(monkeypatch, pg_migrated, codec_env):
    user, tid = str(uuid4()), f"t-{uuid4()}"
    b, h = await _builder(monkeypatch, pg_migrated, codec_env)
    await b.run_turn(user_id=user, thread_id=tid, session_id="s1", message=f"my secret {MARK}", turn_id=uuid4(), channel="rest")
    await h.aclose()
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        dumps = [await conn.fetchval("SELECT coalesce(string_agg(checkpoint::text || metadata::text, ''), '') FROM checkpoints WHERE thread_id=$1", tid),
                 await conn.fetchval("SELECT coalesce(string_agg(encode(blob, 'escape'), ''), '') FROM checkpoint_blobs WHERE thread_id=$1", tid),
                 await conn.fetchval("SELECT coalesce(string_agg(encode(blob, 'escape'), ''), '') FROM checkpoint_writes WHERE thread_id=$1", tid)]
    finally:
        await conn.close()
    assert all(MARK not in d for d in dumps) and any(dumps)
    b2, h2 = await _builder(monkeypatch, pg_migrated, codec_env)  # simulated restart
    with acting_as_user(user):
        msgs = await b2.get_thread_messages(tid)
    assert [m["content"] for m in msgs if m["role"] == "user"] == [f"my secret {MARK}"]
    with acting_as_user(str(uuid4())), pytest.raises(CheckpointAccessError):
        await b2.get_thread_messages(tid)
    await h2.aclose()

@pytest.mark.asyncio
async def test_mvp_066_one_checkpoint_per_turn_prune_and_delete(monkeypatch, pg_migrated, codec_env):
    user, tid = str(uuid4()), f"t-{uuid4()}"
    b, h = await _builder(monkeypatch, pg_migrated, codec_env)
    for _ in range(2):
        await b.run_turn(user_id=user, thread_id=tid, session_id="s1", message="hello there friend", turn_id=uuid4(), channel="ws")
    conn = await asyncpg.connect(pg_migrated.dsn_superuser)
    try:
        assert await conn.fetchval("SELECT count(*) FROM checkpoints WHERE thread_id=$1", tid) == 2
        report = await CheckpointJanitor(h.pool, ttl_days=30, keep_last=1).run_once()
        assert report.checkpoints_pruned >= 1
        assert await conn.fetchval("SELECT count(*) FROM checkpoints WHERE thread_id=$1", tid) == 1
        with acting_as_user(user):
            assert len(await b.get_thread_messages(tid)) == 4
        assert await b.delete_threads([tid]) == 1
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
            assert await conn.fetchval(f"SELECT count(*) FROM {table} WHERE thread_id=$1", tid) == 0
    finally:
        await conn.close()
        await h.aclose()
```
and in `services/orchestrator_service/tests/test_mvp_066_checkpointer_unit.py`
```python
"""MVP-066: strict environments never fall back to memory; dev does with a warning; history is bounded."""
import pytest
from structlog.testing import capture_logs
from langgraph.checkpoint.memory import InMemorySaver
from solace_infrastructure.runtime import StartupDependencyError
from services.orchestrator_service.src.langgraph.checkpointer import create_checkpointer
from services.orchestrator_service.src.langgraph.state_schema import MAX_THREAD_MESSAGES, add_messages

@pytest.mark.asyncio
@pytest.mark.parametrize("env", ["production", "staging"])
async def test_mvp_066_strict_without_postgres_is_fatal(env):
    with pytest.raises(StartupDependencyError):
        await create_checkpointer(environment=env, conninfo=None, codec=None)

@pytest.mark.asyncio
async def test_mvp_066_strict_unreachable_postgres_is_fatal():
    with pytest.raises(StartupDependencyError):
        await create_checkpointer(environment="production", conninfo="host=127.0.0.1 port=9 dbname=x user=x connect_timeout=1",
                                  codec=None, open_timeout_seconds=2.0)

@pytest.mark.asyncio
async def test_mvp_066_development_falls_back_with_warning():
    with capture_logs() as logs:
        h = await create_checkpointer(environment="development", conninfo=None, codec=None)
    assert isinstance(h.saver, InMemorySaver) and h.durable is False
    assert [l for l in logs if l["log_level"] == "warning"]

def test_mvp_066_history_is_bounded():
    big = [{"role": "user", "content": str(i)} for i in range(MAX_THREAD_MESSAGES)]
    out = add_messages(big, [{"role": "user", "content": "new"}])
    assert len(out) == MAX_THREAD_MESSAGES == 200 and out[-1]["content"] == "new"
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; $env:SOLACE_REQUIRE_REALINFRA="1"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest tests/realinfra/orchestrator-graph-crisis/test_mvp_066_postgres_checkpointer.py -o "addopts=" -p no:logging -q` and `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_066_checkpointer_unit.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `ModuleNotFoundError: ... langgraph.checkpointer` (today the checkpointer is always `InMemorySaver`, the Postgres branch is unreachable, and nothing bounds or erases threads); `add_messages` returns 201 entries.
- [ ] **Step 3: Implement** `checkpointer.py`: `PHICodecCipher.encrypt(plaintext)` returns `("solace-phi-v1", codec.encrypt(base64(plaintext), field="orchestrator_checkpoint", owner_id=checkpoint_owner()).encode())` and `decrypt` reverses it, turning any codec failure into `CheckpointAccessError`; `checkpoint_owner()` is `get_current_user_id()` (strict: `CheckpointAccessError` when unset; development/test: `"unbound"`); `OwnerBoundEncryptedSerializer(EncryptedSerializer)` refuses unencrypted payloads in strict environments. `create_checkpointer` opens `AsyncConnectionPool(conninfo, min_size=1, max_size=max_pool_size, open=False, kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})` with `await pool.open(wait=True, timeout=open_timeout_seconds)`, builds `AsyncPostgresSaver(pool, serde=OwnerBoundEncryptedSerializer(PHICodecCipher(codec)))`, verifies `max(v)` in `checkpoint_migrations` equals `len(MIGRATIONS) - 1` and never calls `setup()`; in strict, any failure raises `StartupDependencyError`, in development/test it falls back to `InMemorySaver` with a WARNING. Verify against the locked langgraph-checkpoint-postgres source: if it stores primitive channel values inline in `checkpoints.checkpoint` or copies node writes into `checkpoints.metadata`, subclass the saver so every channel value passes through the serializer; the raw-SQL assertions are the acceptance criterion. `CheckpointJanitor.run_once()` deletes threads whose newest `(checkpoint->>'ts')::timestamptz` is older than `ttl_days` (via `adelete_thread`), keeps only the newest checkpoint per `(thread_id, checkpoint_ns)`, and deletes writes and blobs no longer referenced by the kept checkpoint's `channel_versions`. `from_runtime` builds the conninfo with `build_conninfo(PostgresSettings())`, uses `runtime.phi_codec`, registers the pool closer and a `checkpoint_janitor` background task; `invoke` passes `durability="exit"`; `delete_threads` calls `adelete_thread` per id and drops its lock; `get_thread_messages` reads `aget_state` under the caller's context.
- [ ] **Step 4: Run it and confirm it passes**
Run: same commands. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`, then `tests/realinfra/orchestrator-graph-crisis` with `$env:SOLACE_REQUIRE_REALINFRA="1"`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.10:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.11: Chat agent sends the thread history to the LLM and reports an LLM outage
**Covers:** MVP-073
**Severity:** high
**Wave:** 2
**Depends on:** WS-10.5, WS-09.3, WS-04.3
**Implementer model:** opus
**Review:** security-critical
**Decision gate:** D-04 (decided 2026-10-19: history goes only to the shared LLM client's configured provider, which in strict environments is Anthropic on AWS Bedrock (`LLM_PROVIDER=bedrock`, under the AWS BAA; Portkey only in development and test with synthetic data), with caching off; roles are user and assistant only; no metadata leaves the orchestrator; until the BAA is recorded the LLM is off for real users and the template path is normal)
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/chat_agent.py` (`_generate_llm_response` passes `conversation_history`; catches `LLMUnavailableError` and returns degraded `llm_unavailable` with the template reply; increments `solace_llm_fallback_total{service="orchestrator-service", reason="unavailable"}`; the broad `except Exception: ... return ""` at chat_agent.py:479-481 is replaced, so no caller tests for an empty reply any more (CR-7 of WS-09))
- Test: `services/orchestrator_service/tests/test_mvp_073_chat_history.py` (new)
**Interfaces:**
- Consumes: IF-15 `UnifiedLLMClient.generate(system_prompt, user_message, conversation_history=None, context=None, ...)`, `LLMUnavailableError` (WS-09.3); IF-18 counter object `LLM_FALLBACK` from `solace_infrastructure.runtime.metrics` (WS-04.3); `degraded` channel (WS-10.1).
- Produces: chat-agent state update may contain `degraded: ["llm_unavailable"]`; the chat-agent half of CR-7 (call sites catch `LLMUnavailableError`, never test for `""`), requested by WS-09 for WS-09.4. The `/ready` half of CR-7 (`llm_client.readiness`) is main.py's and is planned in WS-11.3.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_073_chat_history.py`
```python
"""MVP-073: the general-chat LLM sees the earlier turns of the same thread."""
from __future__ import annotations
import pytest
from services.shared.infrastructure.llm_client import LLMUnavailableError
from services.orchestrator_service.src.agents.chat_agent import ChatAgent
from services.orchestrator_service.src.langgraph.state_schema import MessageEntry, create_initial_state

class RecordingLLM:
    is_available = True
    def __init__(self, exc=None):
        self.calls, self.exc = [], exc
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return "I'm so sorry about Max."

def _state():
    s = create_initial_state("user-1", "s1", "I keep thinking about him all the time lately honestly")
    s["messages"] = [MessageEntry.user_message("my dog Max died yesterday and I am really sad about it").to_dict(),
                     MessageEntry.assistant_message("I'm so sorry about Max.", metadata={"x": 1}).to_dict(),
                     MessageEntry.user_message("I keep thinking about him all the time lately honestly").to_dict()]
    return s

@pytest.mark.asyncio
async def test_mvp_073_chat_llm_receives_prior_turns():
    llm = RecordingLLM()
    await ChatAgent(llm_client=llm).process(_state())
    assert llm.calls[0]["conversation_history"] == [
        {"role": "user", "content": "my dog Max died yesterday and I am really sad about it"},
        {"role": "assistant", "content": "I'm so sorry about Max."},
    ]

@pytest.mark.asyncio
async def test_mvp_073_llm_outage_is_degraded_not_silent():
    out = await ChatAgent(llm_client=RecordingLLM(exc=LLMUnavailableError("down"))).process(_state())
    assert out["degraded"] == ["llm_unavailable"] and out["agent_results"][0]["response_content"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_073_chat_history.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `KeyError: 'conversation_history'` (generate is called without history) and `KeyError: 'degraded'`.
- [ ] **Step 3: Implement** `_history_for_llm(state) -> list[dict[str, str]]`: the thread's `messages` except the current user message, roles `user`/`assistant` only, `{"role", "content"}` only, newest 10; pass it as `conversation_history=`. Keep memory context as `context=`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.11:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.12: Negation-aware emotion scoring
**Covers:** MVP-074
**Severity:** high
**Wave:** 2
**Depends on:** none
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/emotion_agent.py` (`_detect_emotions`: clause-scoped negation, count-based scores, negative emotions win ties; the joy template is never chosen when a negative emotion is present)
- Test: `services/orchestrator_service/tests/test_mvp_074_emotion_negation.py` (new)
**Interfaces:**
- Produces: `_detect_emotions(text: str) -> list[tuple[str, float]]` (same signature, new semantics).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_074_emotion_negation.py`
```python
"""MVP-074: negated joy words never outscore sadness, and hopeless users never hear 'positive feelings'."""
import pytest
from services.orchestrator_service.src.agents.emotion_agent import _detect_emotions, emotion_agent_node
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

HOPELESS = "I am hopeless and worthless, nothing makes me happy or proud anymore"

def test_mvp_074_negated_joy_is_dropped():
    found = dict(_detect_emotions(HOPELESS))
    assert "joy" not in found and max(found, key=found.get) == "sadness"

def test_mvp_074_single_negated_joy_word():
    found = dict(_detect_emotions("I'm not happy and I feel lonely"))
    assert "joy" not in found and max(found, key=found.get) == "sadness"

def test_mvp_074_guard_genuine_joy_is_still_detected():
    assert _detect_emotions("I'm so happy and proud of myself today")[0][0] == "joy"

@pytest.mark.asyncio
async def test_mvp_074_hopeless_user_never_gets_the_positive_template():
    out = await emotion_agent_node(create_initial_state("user-1", "s1", HOPELESS))
    assert "wonderful to hear some positive feelings" not in out["agent_results"][0]["response_content"]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_074_emotion_negation.py -o "addopts=" -p no:logging -q`
Expected: FAIL: `joy` 0.5 > `sadness` 0.469 because the score is the share of a smaller keyword set; the node returns the joy template.
- [ ] **Step 3: Implement** negation scope from a negator (`not`, `no`, `never`, `nothing`, `nobody`, `none`, `without`, `n't`, `don't`, `can't`, `isn't`) to the end of the clause (`. , ; ! ?`, `but`, `however`, `though`) and a trailing `anymore` that negates its clause; score `min(0.95, 0.4 + 0.1 * matches)`; on equal scores order sadness, anxiety, shame, anger, confusion, joy; drop `joy` when any negative emotion is present.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (4 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.12:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.13: Word-bounded chat topics; templates only for short formulaic messages
**Covers:** MVP-075
**Severity:** high
**Wave:** 2
**Depends on:** WS-10.11
**Implementer model:** sonnet
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/agents/chat_agent.py` (`TopicClassifier`: `\b`-bounded phrase regexes; GREETING, FAREWELL and GRATITUDE only for messages of at most 4 words)
- Test: `services/orchestrator_service/tests/test_mvp_075_topic_word_boundary.py` (new)
**Interfaces:**
- Produces: `TopicClassifier.classify(message) -> TopicClassification` (same signature).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_075_topic_word_boundary.py`
```python
"""MVP-075: distress messages are never answered with greeting, farewell or gratitude templates."""
import pytest
from services.orchestrator_service.src.agents.chat_agent import ChatAgent, TopicCategory, TopicClassifier
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state

DISTRESS = ["they said I am a burden and everyone would be better without me",
            "they'd all be better without me around",
            "I can't take care of myself anymore",
            "I feel awful. thanks for nothing"]

@pytest.mark.parametrize("text", DISTRESS)
def test_mvp_075_distress_is_not_formulaic(text):
    assert TopicClassifier().classify(text).category == TopicCategory.GENERAL

@pytest.mark.parametrize("text,cat", [("hey", TopicCategory.GREETING), ("hi there", TopicCategory.GREETING),
                                      ("thanks so much", TopicCategory.GRATITUDE), ("bye, take care", TopicCategory.FAREWELL)])
def test_mvp_075_guard_short_formulaic_messages(text, cat):
    assert TopicClassifier().classify(text).category == cat

class RecordingLLM:
    is_available = True
    def __init__(self):
        self.calls = []
    async def generate(self, **kw):
        self.calls.append(kw)
        return "That sounds really painful."

@pytest.mark.asyncio
async def test_mvp_075_distress_reaches_the_llm():
    llm = RecordingLLM()
    await ChatAgent(llm_client=llm).process(create_initial_state("user-1", "s1", DISTRESS[0]))
    assert len(llm.calls) == 1
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_075_topic_word_boundary.py -o "addopts=" -p no:logging -q`
Expected: FAIL: "they said..." is GREETING ("hey" inside "they"), "I can't take care..." is FAREWELL, "thanks for nothing" is GRATITUDE, and the LLM is never called.
- [ ] **Step 3: Implement** one compiled `\b`-bounded regex per pattern list; a formulaic category applies only when the message has at most 4 words, otherwise the classifier falls through to CHECK_IN, CLARIFICATION, SMALL_TALK or GENERAL. Existing `TestTopicClassifier` cases stay green.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.13:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.14: Progress updates reach therapy; active treatment is honoured
**Covers:** MVP-165
**Severity:** medium
**Wave:** 2
**Depends on:** WS-10.7
**Implementer model:** opus
**Review:** safety-critical
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/supervisor.py` (`AgentRouter.select_agents`: PROGRESS_UPDATE without a plan gives `[THERAPY, CHAT]`, with `active_treatment` gives `[THERAPY, DIAGNOSIS]`; a deteriorating progress message (`worse`, `hopeless`, `can't cope`, word-bounded) adds EMOTION)
- Test: `services/orchestrator_service/tests/test_mvp_165_progress_routing.py` (new)
**Interfaces:**
- Consumes: state `active_treatment: {"therapy_session_id": str, "treatment_plan_id": str} | None`, persisted across turns (WS-10.1) and written by the therapy agent (WS-11.9).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_165_progress_routing.py`
```python
"""MVP-165: progress updates reach therapy, deteriorating ones also reach emotion."""
import pytest
from services.orchestrator_service.src.langgraph.state_schema import AgentType, IntentType
from services.orchestrator_service.src.langgraph.supervisor import SupervisorAgent

@pytest.mark.asyncio
async def test_mvp_165_progress_without_plan_reaches_therapy():
    d = await SupervisorAgent(llm_client=None).make_decision("I've been getting better since last week")
    assert d.intent == IntentType.PROGRESS_UPDATE and AgentType.THERAPY in d.selected_agents

@pytest.mark.asyncio
async def test_mvp_165_deteriorating_progress_adds_emotion():
    d = await SupervisorAgent(llm_client=None).make_decision("it's getting worse and I feel hopeless")
    assert {AgentType.THERAPY, AgentType.EMOTION} <= set(d.selected_agents)

@pytest.mark.asyncio
async def test_mvp_165_active_treatment_reaches_therapy_and_diagnosis():
    d = await SupervisorAgent(llm_client=None).make_decision(
        "I've been getting better since last week", active_treatment={"therapy_session_id": "x", "treatment_plan_id": "y"})
    assert {AgentType.THERAPY, AgentType.DIAGNOSIS} <= set(d.selected_agents)
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_165_progress_routing.py -o "addopts=" -p no:logging -q`
Expected: FAIL: both progress messages route to `[CHAT, PERSONALITY]` because `active_treatment` is never populated.
- [ ] **Step 3: Implement** the routing table change and `_is_deteriorating(message) -> bool` in `supervisor.py`; `max_parallel_agents` still caps the list and SAFETY routing for crisis is unchanged.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS (3 passed)
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.14:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.15: History stores exactly the reply the user was shown
**Covers:** MVP-298
**Severity:** low
**Wave:** 2
**Depends on:** WS-10.10
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Modify: `services/orchestrator_service/src/langgraph/graph_builder.py` (new `finalize_turn_node` after `safety_postcheck` appends the single assistant `MessageEntry` with the final text; `crisis_handler_node` stops appending messages)
- Modify: `services/orchestrator_service/src/langgraph/aggregator.py` (`Aggregator.aggregate` no longer returns `messages`)
- Test: `services/orchestrator_service/tests/test_mvp_298_history_matches_reply.py` (new); `services/orchestrator_service/tests/test_graph_builder.py` (`test_crisis_adds_message` moves to the finalize node); `services/orchestrator_service/tests/test_response.py` (aggregator returns no `messages`)
**Interfaces:**
- Produces: graph node `finalize_turn` (`safety_postcheck -> finalize_turn -> END`); assistant message metadata `{"turn_id", "is_crisis_response"}`.

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_298_history_matches_reply.py`
```python
"""MVP-298: the checkpointed assistant message equals the styled, postchecked reply."""
import pytest
from services.orchestrator_service.src.langgraph import graph_builder as gb
from services.orchestrator_service.src.langgraph.graph_builder import GraphBuilderSettings, OrchestratorGraphBuilder
from services.orchestrator_service.src.langgraph.state_schema import AgentResult, AgentType, create_initial_state

@pytest.mark.asyncio
async def test_mvp_298_history_matches_shown_reply(monkeypatch):
    async def chat(state):
        return {"agent_results": [AgentResult(agent_type=AgentType.CHAT, success=True,
                                              response_content="Tell me more about that.", confidence=0.7).to_dict()]}
    async def personality(state):
        return {"personality_style": {"warmth": 0.9},
                "agent_results": [AgentResult(agent_type=AgentType.PERSONALITY, success=True, confidence=0.7).to_dict()]}
    async def noop(state):
        return {}
    async def memory(state):
        return {"conversation_context": "", "assembled_context": "", "memory_context": {}}
    nodes = {k: noop for k in ("diagnosis", "therapy", "safety", "assessment", "emotion")}
    nodes.update(chat=chat, personality=personality)
    monkeypatch.setattr(gb, "_import_agent_nodes", lambda: nodes)
    monkeypatch.setattr(gb, "memory_retrieval_node", memory)
    monkeypatch.setattr(gb, "get_supervisor_llm_client", lambda: None)
    b = OrchestratorGraphBuilder(GraphBuilderSettings(enable_checkpointing=True))
    b.compile()
    result = await b.invoke(create_initial_state("user-1", "s1", "hello there friend", thread_id="t-298"), thread_id="t-298")
    snap = await b.get_compiled_graph().aget_state({"configurable": {"thread_id": "t-298"}})
    assistant = [m for m in snap.values["messages"] if m["role"] == "assistant"]
    assert result["final_response"] == "I appreciate you sharing that. Tell me more about that."
    assert [m["content"] for m in assistant] == [result["final_response"]]
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_298_history_matches_reply.py -o "addopts=" -p no:logging -q`
Expected: FAIL: the stored assistant message is "Tell me more about that." (the aggregator's pre-style text).
- [ ] **Step 3: Implement** `finalize_turn_node(state)` returning `{"messages": [MessageEntry.assistant_message(state["final_response"], metadata=...).to_dict()]}`, wire it after `safety_postcheck`, and remove the message appends from the aggregator and the crisis handler.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/integration`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.15:done"` and record the printed tree id in the ledger. No git commit.

### Task WS-10.16: Orchestrator-to-safety /check contract (IF-11)
**Covers:** MVP-017
**Severity:** blocker
**Wave:** 2
**Depends on:** WS-10.4, WS-05.7
**Implementer model:** sonnet
**Review:** standard
**Decision gate:** none
**Real infra:** no
**Files:**
- Create: `tests/contracts/orchestrator__safety/check.json` (consumer contract file, IF-20 format)
- Test: `services/orchestrator_service/tests/test_mvp_017_safety_contract.py` (new)
**Interfaces:**
- Consumes: IF-20 `solace_testing.contracts.ContractFile` and `assert_request_matches(contract: ContractFile, request)` (WS-05.7); IF-06 `/check` shapes.
- Produces: `tests/contracts/orchestrator__safety/check.json` `{consumer: "orchestrator", provider: "safety", interaction: "check", request: {method: "POST", path: "/api/v1/safety/check", headers_required: ["Authorization", "X-Request-ID", "X-Correlation-ID", "X-Turn-ID"], body_example}, response: {status: 200, body_schema}}`. It answers WS-12's request R-12-10 (addressed to WS-11, but the `/check` consumer `crisis_screen.SafetyCheckClient` is WS-10 code, so WS-10 writes it), which WS-12.10's provider test runs, and it is the `orchestrator__safety` consumer file of WS-05's request #10 (needed by WS-05.10).

- [ ] **Step 1: Write the failing test** in `services/orchestrator_service/tests/test_mvp_017_safety_contract.py`
```python
"""MVP-017: the orchestrator's /check request and parser match the shared contract."""
import json
import pathlib
from uuid import UUID
import httpx
import pytest
from solace_testing.contracts import ContractFile, assert_request_matches
from services.orchestrator_service.src.langgraph import crisis_screen as cs

CONTRACT_PATH = pathlib.Path(__file__).resolve().parents[3] / "tests/contracts/orchestrator__safety/check.json"

@pytest.mark.contract
@pytest.mark.asyncio
async def test_mvp_017_check_request_matches_contract(monkeypatch):
    contract = ContractFile.model_validate(json.loads(CONTRACT_PATH.read_text(encoding="utf-8")))
    async def headers(target, *, on_behalf_of=None):
        return {"Authorization": "Bearer obo", "X-Request-ID": "r", "X-Correlation-ID": "c", "X-Turn-ID": "t"}
    monkeypatch.setattr(cs, "service_auth_headers", headers)
    seen = {}
    def handler(req):
        seen["req"] = req
        return httpx.Response(200, json={"level": "HIGH", "risk_score": 0.8, "categories": ["suicidal_ideation"],
                                         "requires_escalation": True, "resources": [], "detection_layer": 1})
    client = cs.SafetyCheckClient(base_url="http://safety-service:8002", transport=httpx.MockTransport(handler))
    res = await client.check(user_id="8a3f0d4e-0000-4000-8000-000000000001", session_id="s1",
                             turn_id=UUID("4b6f1c2a-0000-4000-8000-000000000002"), content="hello", channel="rest")
    assert_request_matches(contract, seen["req"])
    assert res.level == "HIGH" and res.categories == frozenset({"suicidal_ideation"})
```
- [ ] **Step 2: Run it and confirm it fails**
Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests/test_mvp_017_safety_contract.py -o "addopts=" -p no:logging -q`
Expected: FAIL with `FileNotFoundError` for `tests/contracts/orchestrator__safety/check.json` (no producer/consumer contract exists for the crisis check, MVP-357 pattern).
- [ ] **Step 3: Implement** the contract JSON from the IF-06 request and response, with a synthetic `body_example` and a `body_schema` requiring `level` (enum NONE..CRITICAL), `risk_score`, `categories`, `requires_escalation`, `resources`, `detection_layer`.
- [ ] **Step 4: Run it and confirm it passes**
Run: same command. Expected: PASS
- [ ] **Step 5: Regression:** Run: `$env:PYTHONPATH="D:\Repo\Contextual-Chatbot;D:\Repo\Contextual-Chatbot\src"; $env:ENVIRONMENT="test"; & "D:\tmp\mvp-exec\venv\Scripts\python.exe" -m pytest services/orchestrator_service/tests -o "addopts=" -p no:logging -q`, then the same command with `tests/contracts`. Expected: all pass
- [ ] **Step 6: Checkpoint:** `& D:\tmp\mvp-exec\tools\checkpoint.ps1 -Label "WS-10.16:done"` and record the printed tree id in the ledger. No git commit.

## Cross-workstream requests

| Owner WS | Change | Files | Needed by |
|---|---|---|---|
| WS-01 | `tests/build/test_import_roots.py` allow-lists `services/orchestrator_service/src/events.py` for the W0 barrier only (WS-10 has no W0 wave); WS-10.3 rewrites its `src.solace_events` import in W1 phase B and the entry is removed by Ruling when WS-10.3 completes. Planned in WS-01.7 (creates the guard; the WS-10 case is `xfail(strict=True)`); WS-10.3 depends on WS-01.7. | `tests/build/test_import_roots.py` | WS-10.3 |
| WS-01 | The lock keeps `langgraph` 1.0.x (needs `langgraph.types.Overwrite` and `ainvoke(durability=)`), `langgraph-checkpoint-postgres` whose `MIGRATIONS` slot 017 embeds, `psycopg[binary]`, `psycopg-pool`, `PyYAML`; a version bump of langgraph-checkpoint-postgres must ship with a slot-017 follow-up revision. | `requirements.in`, `requirements.lock` | WS-10.1, WS-10.9, WS-10.10 |
| WS-03 | Env registry entries: `ORCHESTRATOR_CRISIS_RESPONSE_BUDGET_SECONDS` (orchestrator, not secret, default 2.8, W1); `ORCHESTRATOR_CHECKPOINT_TTL_DAYS` (default 30) and `ORCHESTRATOR_CHECKPOINT_JANITOR_INTERVAL_SECONDS` (default 3600) (W2); mark retired after W1: `ORCHESTRATOR_GRAPH_USE_SAFETY_SERVICE_PRECHECK`, `ORCHESTRATOR_SAFETY_AGENT_SERVICE_URL`. The orchestrator also needs `POSTGRES_*` as `solace_app` with verify-full TLS for the psycopg pool (W2). | `deploy/env/env-contract.yaml`, `docker-compose.yml` | WS-10.5, WS-10.10 |
| WS-04 | Export the IF-18 counters as module objects in `solace_infrastructure.runtime.metrics` so service code increments them without re-registering. Planned by WS-04.3 as `CRISIS_DETECTED` (`solace_crisis_detected_total{source,level}`) and `LLM_FALLBACK` (`solace_llm_fallback_total{service,reason}`); WS-10 uses those names. | `src/solace_infrastructure/runtime/metrics.py` | WS-10.5, WS-10.7, WS-10.11 |
| WS-07 | Confirm in IF-04 that `service_auth_headers` reads `request_id`, `correlation_id` and `turn_id` from `structlog.contextvars` (the keys `run_turn` and the IF-01 middleware bind), and that a header-minting failure raises (the safety client maps it to `error`). | `src/solace_security/service_auth.py` | WS-10.4, WS-10.5 |
| WS-12 | The phrase set `tests/clinical/crisis_phrases_v1.yaml` contains, with `min_level` HIGH or above, the register phrases WS-10 tests: "I feel suicidal", "I can\u2019t go on", "I'm going to kill him", "my dad hits me and I'm scared to go home", "I'd be better off dead", "I just want to end it all", "I can't go on", "I'm going to take my life", "thinking of hanging myself"; and as negatives "cutting vegetables for dinner tonight", "I hate cutting vegetables". Record for D-05 sign-off in `docs/CLINICAL-VALIDATION.md`: the LLM-only crisis fallback category `imminent_danger`, the harmful-reply rules, the crisis reply, footer and fallback copy in `response/safety_copy.py`, and the D-06 watch values. | `tests/clinical/crisis_phrases_v1.yaml`, `docs/CLINICAL-VALIDATION.md` | WS-10.2, WS-10.4, WS-10.6, WS-10.7, WS-10.8 |
| WS-12 | Provider verification of `tests/contracts/orchestrator__safety/check.json` with `verify_provider` in the safety suite (W2); planned by WS-12.10 (`test_orchestrator_contracts_verify_against_the_real_router`). WS-12's R-12-10 names WS-11 as the writer of this file; the writer is WS-10.16 (the `/check` consumer is WS-10's `crisis_screen.SafetyCheckClient`), so WS-12.10's skip reason should read "written by WS-10.16 in W2". | `services/safety_service/tests/test_if06_check_contract.py` | WS-10.16 |
| WS-05 | `tests/contracts/README.md` protocol line: `orchestrator__safety/` is written by WS-10 (WS-10.16), the other `orchestrator__*` directories by WS-11; `tests/contracts/expected_failures.yaml` names WS-10 as the owner of the `orchestrator__safety` consumer entry. | `tests/contracts/README.md`, `tests/contracts/expected_failures.yaml` | WS-10.16 |
| WS-11 | The `/ready` half of WS-09's CR-7: main.py registers `runtime.register_readiness_check("llm", llm_client.readiness, critical=False)` (WS-11.3); WS-10 does the call-site half in WS-10.7 and WS-10.11. | `services/orchestrator_service/src/main.py` | WS-09.4 (CR-7) |

## New interfaces

- IF-10 precision: `OrchestratorGraphBuilder.from_runtime` is an `async` classmethod (awaited by `main.py`), so W2 can open the psycopg pool without changing the call site.
- `OrchestratorGraphBuilder.__init__(settings=None, *, event_plane=None, safety_client=None, turn_settings=None, checkpointer=None, clock=None)` for tests and for `from_runtime`.
- `TurnSettings` (env prefix `ORCHESTRATOR_`): `turn_timeout_seconds` (IF-02 `ORCHESTRATOR_TURN_TIMEOUT_SECONDS`), `safety_check_timeout_seconds` (IF-02 `ORCHESTRATOR_SAFETY_CHECK_TIMEOUT_SECONDS`), new `crisis_response_budget_seconds`, `checkpoint_ttl_days`, `checkpoint_janitor_interval_seconds`.
- `CrisisInfo(level: str, categories: list[str], resources: list[dict[str, str]], resources_version: str).to_dict()`; `TurnResult` field types as in WS-10.5.
- Degraded vocabulary produced by WS-10: `safety_service_timeout`, `safety_service_unavailable`, `crisis_alert_publish_failed`, `crisis_alert_publish_pending`, `graph_error`, `turn_timeout`, `llm_unavailable`, `checkpoint_unavailable`. WS-11 agents add `<agent>_unavailable` and `memory_write_failed`.
- State contract for agents (consumed by WS-11): agents may return `degraded: [code]`; a provider-reported risk may be returned as a partial `safety_flags` update `{"risk_level": <the provider's level when it reports one, never below "ELEVATED">, "categories": [...], "signal_source": "<agent>"}` (the reducer max-merges; `run_turn` shows resources for ELEVATED and above; the provider pages for its own HIGH/CRITICAL under IF-06, and notification coalesces by turn id); the therapy agent may return `active_treatment`.
- `events.session_uuid(raw: str) -> UUID`, `build_crisis_detected_event(...)`, `publish_crisis_event(plane, event) -> CrisisPublishOutcome` (WS-10.3).
- `crisis_screen`: `screen_local`, `risk_rank`, `max_level`, `SafetyCheckClient`, `RemoteCheck`, `IngressScreen`, `screen_message`, `screen_reply`, `ReplyScreen`.
- `state_schema`: `TURN_SCOPED_CHANNELS`, `build_turn_input`, `add_degraded`, `MAX_THREAD_MESSAGES = 200`, `SafetyFlags` risk fields.
- `OrchestratorGraphBuilder.get_thread_messages(thread_id, *, limit=50)` and `checkpointer.CheckpointAccessError` (consumed by WS-11.7 history).
- `response/safety_copy.py`: `crisis_reply_text`, `safety_footer`, `SAFE_FALLBACK_TEXT`, `HARMFUL_REPLY_REPLACEMENT`, `harmful_reply_replacement(resources)` (D-29).
- IF-10 addition (D-29, 2026-10-19): `run_turn(..., locale: str | None = None)` and `crisis_for_refused_message(..., locale: str | None = None)`. `locale` is the raw access-token claim `loc` (WS-08.21) that WS-11 reads from `AuthenticatedUser.locale` with no lookup; `None` means en-US. Both resolve it with `solace_clinical.resolve_locale`, and every crisis object and crisis copy comes from `crisis_resource_set(resolved)`. Graph state key `locale: str` is turn-scoped (`create_initial_state(..., locale=DEFAULT_LOCALE)`); WS-11.21's therapy agent relays it to therapy.

## Self-review

**Coverage**

| Register id | Severity | Tasks |
|---|---|---|
| MVP-017 | blocker | WS-10.5 (closes; supported by WS-10.2, WS-10.4), WS-10.16 (contract) |
| MVP-018 | blocker | WS-10.1 |
| MVP-019 | blocker | WS-10.1, WS-10.6 (closes) |
| MVP-023 | blocker | WS-10.7 |
| MVP-024 (absorbs MVP-077) | blocker | WS-10.7 |
| MVP-025 | blocker | WS-10.2 |
| MVP-026 | blocker | WS-10.2 |
| MVP-066 | high | WS-10.9, WS-10.10 (closes) |
| MVP-067 | high | WS-10.5 (supported by WS-10.3) |
| MVP-068 | high | WS-10.8 |
| MVP-069 | high | WS-10.5 (supported by WS-10.3) |
| MVP-073 | high | WS-10.11 |
| MVP-074 | high | WS-10.12 |
| MVP-075 | high | WS-10.13 |
| MVP-079 | high | WS-10.5 |
| MVP-161 | medium | WS-10.1 |
| MVP-165 | medium | WS-10.14 |
| MVP-166 | medium | WS-10.4 |
| MVP-167 | medium | WS-10.3 |
| MVP-173 | medium | WS-10.7 |
| MVP-295 | low | WS-10.7 |
| MVP-298 | low | WS-10.15 |

All 22 ids are covered; none are left uncovered.

**Review focus** (the skeleton's five review_focus failure modes, numbered 1-5 in skeleton order)

REVIEW_FOCUS: 1=WS-10.2,WS-10.6,WS-10.7; 2=WS-10.1,WS-10.3,WS-10.5,WS-10.6; 3=WS-10.10; 4=WS-10.2,WS-10.3,WS-10.10; 5=WS-10.10
- 1: WS-10.2 runs the whole signed-off phrase set (its variants are WS-12's) through `safety_precheck_node`; WS-10.7 `test_mvp_024_llm_crisis_wins_at_any_confidence` pins slang, misspelling, non-English, emoji and quoted-speech disclosures reaching CRISIS_DISCLOSURE through the LLM whatever the lexicon says; WS-10.6 pins the post-crisis watch across turns.
- 2: WS-10.1 serializes concurrent turns on one thread; WS-10.3 never logs a failed publish as emitted; WS-10.5 pins one event per turn, a reported publish failure, resources that do not wait for a slow publish, and a graph crash that still pages once; WS-10.6 pins no re-page during the watch and a re-page for a new crisis.
- 3: WS-10.10 reads a thread under another user's context and expects `CheckpointAccessError`.
- 4: WS-10.2 (rule ids, never text, in flags), WS-10.3 (category codes only and no message text or raw session id in the event payload), WS-10.10 (raw SQL over all three checkpoint tables finds no plaintext).
- 5: WS-10.10 pins strict boots with no or an unreachable Postgres as fatal and the development fallback as a logged warning.

**Dependencies**
- Every cross-workstream dependency names the producing task: WS-12.2/WS-12.3 (IF-07), WS-06.8 (IF-05), WS-07.1/WS-07.3 (IF-04), WS-04.3/WS-04.4/WS-04.6/WS-04.8 (IF-01, IF-18 objects, `event_plane`, `phi_codec`), WS-09.3 (IF-15), WS-09.7 (IF-13), WS-05.2 (IF-19), WS-05.7 (IF-20), WS-02.3 (slot-017 scaffold), WS-01.2 (lock). All are earlier-wave or same-wave phase-A tasks; the only phase-A task here, WS-10.9, depends on phase-A or earlier-wave work only (WS-11.6 is W2 phase A).

**Interface consistency**
- IF-10 names and signatures match the skeleton (`from_runtime`, `run_turn(*, user_id, thread_id, session_id, message, turn_id, channel)`, `TurnResult` fields, `delete_threads(thread_ids) -> int`); the only additions are listed above (including D-29's optional `locale` keyword, whose default keeps every existing call valid), and WS-11's plan consumes exactly these (`run_turn`, `TurnResult`, `CrisisInfo.to_dict`, `get_thread_messages`, `delete_threads`, `session_uuid`, `screen_local`, `degraded` and `safety_flags` state contract, `active_treatment`).
- IF-06 request/response field names are used verbatim; turn ids are minted by WS-11 and passed in, never taken from client headers; X-Turn-ID reaches the safety service via `service_auth_headers` and the bound contextvar.
- IF-05 `publish(event, *, conn=None)` and `EventPublishError` are the only event-plane calls; no code path publishes crisis events through the in-process EventBus any more.
- IF-07 is consumed, never re-implemented: no orchestrator keyword list remains; fusion is `fuse_risk`; resources are `crisis_resource_set(resolve_locale(locale))` (D-29), which for en-US or a missing claim is exactly `crisis_resources("en-US")` with `RESOURCES_VERSION`.
- IF-13 is consumed through LangGraph's `EncryptedSerializer` with `field="orchestrator_checkpoint"` and owner = current user id.
- Names re-exported by WS-11's `agents/__init__.py` from `safety_agent.py` and `chat_agent.py` stay importable.
- Shared test files: WS-10 edits only test files whose code under test is WS-10's; new files follow `test_mvp_<id>_<slug>.py`.

**Open risks**
- Clinical sign-off (D-05/D-06) may change the lexicon, the harmful-reply rules, the LLM crisis floor, the fallback category or the watch values; the phrase-set test and WS-10.6 tests then change in the same wave under a ledger Ruling, never weakening a positive.
- The checkpoint tables have no RLS (no `user_id` column); isolation rests on WS-11.7 thread ownership plus owner-bound AES-GCM AAD. A reviewer should confirm that no path reads checkpoints without the owner's context.
- The locked langgraph-checkpoint-postgres may inline primitive channel values or node writes in JSONB; WS-10.10's raw-SQL assertions catch it, and the fix is a saver subclass that adds maintenance cost at every library upgrade.
- `Overwrite` in graph input is assumed to be honoured by LangGraph 1.0.4; WS-10.1 names the first-node fallback if it is not.
- The orchestrator publishes on every fused HIGH/CRITICAL turn, including turns where only the safety service detected risk; notification coalescing by turn id (IF-09) keeps that to one page, and an E2E (WS-05) must prove it.
- `services/orchestrator_service/src/events.py` still has a `src.solace_events` import at the W0 barrier; WS-01.7 accepted the allow-list request as a strict-xfail case, so the W0 barrier stays green and WS-10.3 removes the import in W1.
- WS-10 and WS-11 share one service test suite in parallel lanes; a half-finished edit in one lane can redden the other's suite run, so the controller should not run WS-10 and WS-11 orchestrator tasks concurrently.
- Review focus 1, sub-case "risk that builds across several turns" with no single HIGH turn: the supervisor's LLM intent call sees only the current message, so WS-10 pins only the post-crisis watch (WS-10.6). Giving the intent call the bounded thread history (as WS-10.11 does for the chat LLM) is a D-05 clinical decision; raise it at sign-off.

## Review responses

- [Error] WS-05 request #9 (needed by WS-05.10): orchestrator `verify_event_producer` test against `tests/contracts/events/orchestrator__notification/`: fixed. Joined WS-10.3 (W1): new `services/orchestrator_service/tests/test_mvp_357_event_contracts.py` checks the real `build_crisis_detected_event` output with `verify_event_producer` and asserts that no message text or raw session id reaches the payload. WS-10.3 now depends on WS-05.7, and its Produces line names WS-05.10.
- [Error] WS-09 CR-7 (needed by WS-09.4): `llm` readiness check plus `LLMUnavailableError` at the call sites: fixed. The two orchestrator LLM call sites are WS-10 code (`supervisor.py:223` and `chat_agent.py:464`). WS-10.7 removes the `if not response:` test at supervisor.py:230, and WS-10.11 replaces the `except Exception: return ""` at chat_agent.py:479-481. Both Produces lines name WS-09.4. The readiness half lives in `main.py`, which WS-11 owns, so it is planned in WS-11.3 (cross-workstream row added).
- [Error] WS-09 CR-10 (needed by WS-09.15): adopt IF-13 in the ADOPTERS path `langgraph/checkpointer.py`: fixed. Joined WS-10.10: the module imports `solace_security.phi_codec` and no legacy encryptor, and its Produces line names WS-09.15. The "`encryption_key_id` column" part does not apply. The slot-017 tables are LangGraph's own, and the IF-13 envelope `enc:v1:<key_id>:...` (skeleton IF-13) already carries the key id.
- [Warning] WS-10.2 depends on an interface (WS-12:IF-07): fixed, now WS-12.2 and WS-12.3.
- [Warning] WS-10.3 (WS-06:IF-05): fixed, now WS-06.8 (plus WS-05.7 for the joined contract test).
- [Warning] WS-10.4 (WS-07:IF-04; WS-12:IF-07): fixed, now WS-07.3 and WS-12.2.
- [Warning] WS-10.5 (WS-06:IF-05; WS-07:IF-04; WS-04:IF-01; WS-04:IF-18-metrics; WS-12:IF-07): fixed, now WS-06.8, WS-07.1, WS-07.3, WS-04.3, WS-04.6 and WS-12.2.
- [Warning] WS-10.7 (WS-09:IF-15): fixed, now WS-09.3 (plus WS-04.3 for the `LLM_FALLBACK` object).
- [Warning] WS-10.9 (WS-02:slot-017-scaffold; WS-01:IF-21-lock; WS-05:IF-19): fixed, now WS-02.3, WS-01.2 and WS-05.2.
- [Warning] WS-10.10 (WS-09:IF-13; WS-04:IF-01; WS-05:IF-19): fixed, now WS-09.7, WS-04.4, WS-04.8 (`ServiceRuntime.phi_codec`) and WS-05.2.
- [Warning] WS-10.11 (WS-09:IF-15): fixed, now WS-09.3 (plus WS-04.3).
- [Warning] WS-10.16 (WS-05:IF-20): fixed, now WS-05.7.
- [Self-check] The IF-18 counter names in the WS-04 request (`CRISIS_DETECTED_TOTAL`, `LLM_FALLBACK_TOTAL`) did not match WS-04.3's planned objects `CRISIS_DETECTED` and `LLM_FALLBACK`: fixed in the request row and in the WS-10.5, WS-10.7 and WS-10.11 Consumes lines.
- [Self-check] WS-05.7 declares `assert_request_matches(contract: ContractFile, request)`, but the WS-10.16 test passed a raw dict: fixed. The test now validates the file with `ContractFile.model_validate` and reads it inside the test, so RED is a clean `FileNotFoundError`.
- [Self-check] WS-12's R-12-10 asks WS-11 to write `orchestrator__safety/check.json`, which WS-10.16 already writes: resolved with WS-10.16 as the writer, because the consumer code is WS-10's. WS-10.16's Produces line names WS-12.10, and request rows went to WS-12 (skip-reason text) and WS-05 (README ownership line).
- [Reconciliation 2] WS-10's request to WS-01 (W0 allow-list of the orchestrator's `events.py` in `tests/build/test_import_roots.py`): planned by WS-01.7, which now lists the guard under Create and carries the WS-10 case as `xfail(strict=True)`. WS-10.3 depends on WS-01.7, consumes the case, and its Step 5 records the strict XPASS that triggers the controller's marker-removal Ruling. The request row and the open risk now say so.
- [Reconciliation 10] WS-05 request 9 (producer `verify_event_producer` tests, needed by WS-05.10): WS-10.3 already had the test body, but listed `services/orchestrator_service/tests/test_mvp_357_event_contracts.py` (and `test_mvp_167_crisis_event_contract.py`) only on its `Test:` line as "(new)", which the checker does not count as a creation. Both files are now on a Create line. WS-12.8, WS-15.5 and WS-16.5 already create their own `test_mvp_357_event_contracts.py` with the same check; WS-12.8's Produces line now also names WS-05.10.
- crisis-review F2: fixed. Verified at WS-10.3's Produces (`publish_crisis_event` logs CRITICAL and returns `failed`) and WS-10.5's step 5 (a pending publish only got a logging done-callback): an outbox or Postgres failure during a crisis turn lost the orchestrator's page for good, and the failure log carried no user reference. WS-10.3 adds `CrisisPublishRetrier` (one entry per turn, bounded at 10,000, backoff 1 s doubling to 60 s, the same `event_id` on every attempt so a late duplicate coalesces) and logs `correlation_id`, `user_id` and `error_type`; WS-10.5 hands failed and pending publishes to it, and `from_runtime` runs it as the registered background task `crisis_publish_retry`. Not adopted: the direct Kafka produce as a fallback. During a database outage the orchestrator stays live (IF-18 ties `/live`, not `/ready`, to restarts), so the in-memory retry survives the outage; notification needs Postgres to page anyway, so a direct produce would not page sooner, and it would add a second, non-transactional publish path to IF-05. The safety half is WS-12.9; the E2E is WS-05.8.
- crisis-review F3: fixed (WS-10 half). WS-10.5 adds `crisis_for_refused_message(...)`, which the edge calls before any refusal (consent, expired token, rate limit): local screen only, no graph, `/check`, LLM or checkpoint; the crisis object at ELEVATED or above; one published event at HIGH/CRITICAL; never raises. Two unit tests pin it. The edge half is in WS-11.16, WS-11.17 and WS-11.20.
- crisis-review F7: fixed. Verified: step 1 awaited the `/check` (up to 2.0 s) and step 3 then gave the graph the whole 2.8 s crisis budget, so a hung safety-service plus a slow graph answered after about 4.8 s, and the budget test used an instant `FakeSafety`. WS-10.5 now has one deadline, `run_turn` entry plus 2.8 s, for HIGH/CRITICAL turns, shared by the graph wait and the publish wait. `FakeSafety` gains a delay, and `test_crisis_review_f7_slow_check_and_hung_graph_stay_inside_one_deadline` asserts under 3 s with a 2 s check and a hung graph; the E2E with a paused safety-service is in WS-05.8.
- crisis-review F12: fixed. Verified: WS-10.2's `update_safety_flags` takes `detection_layer` and `confidence` from the higher-risk side, and the diagnosis and therapy updates (WS-11.9, WS-11.10) carry neither, so a diagnosis-only HIGH reached `build_crisis_detected_event` with `None` and raised after the graph; step 4 also named only an LLM-raised HIGH. Step 4 now publishes any graph-raised HIGH/CRITICAL once, with `detection_layer` defaulting to 3 and `confidence` to 0.6, inside a `try` that never drops the crisis object; `test_crisis_review_f12_provider_raised_high_without_layer_pages_once` pins it. WS-11.9 now maps therapy's provider level.
- decision 2026-10-19 D-29: WS-10.5 takes the user's locale with no DB or network call. `run_turn` and `crisis_for_refused_message` gain `locale: str | None = None` (the raw `loc` access-token claim that WS-11.21 passes from `AuthenticatedUser.locale`). The locale is resolved once with `solace_clinical.resolve_locale`, and every crisis object and crisis reply is built from one `crisis_resource_set(resolved)` (made only when one is needed). The graph state gets the turn-scoped `locale` key (`state_schema.py` added to WS-10.5's Files). The old default "resources from `crisis_resources("en-US")`" is replaced. New RED tests: en-US, `en_us` and a missing claim keep today's resources and `RESOURCES_VERSION`; `fr-FR` gets the fallback with `FALLBACK_RESOURCES_VERSION`, no 988 or 911, and still pages once when the graph fails; a refused message uses the same locale. Step 4 now expects 22 passed. Max-only fusion, paging and the deadline are unchanged.
- decision 2026-10-19 D-29: WS-10.6's footer and WS-10.8's harmful-reply replacement use `crisis_resources(state["locale"])`. WS-10.8 adds `harmful_reply_replacement(resources)` (`HARMFUL_REPLY_REPLACEMENT` stays the en-US text) and `test_d29_harmful_reply_replacement_follows_the_turn_locale`, and Step 4 now expects 6 passed. The file map, New interfaces and Interface consistency (IF-07, IF-10) are updated. No new cross-workstream request: the claim's producer and carrier are WS-08.21, WS-07 (requested in WS-11's plan) and WS-11.21.
- decision 2026-10-19 D-04: WS-10.11's decision gate no longer sends history to "the LLM gateway" (the Portkey wording). The strict-environment provider is now Anthropic on AWS Bedrock (`LLM_PROVIDER=bedrock`, under the AWS BAA); Portkey is allowed only in development and test with synthetic data. The LLM-optional behaviour and the template fallback are unchanged.
