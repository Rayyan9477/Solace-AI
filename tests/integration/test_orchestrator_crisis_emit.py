"""M1 gate round-2 (P0): the orchestrator must ALERT A CLINICIAN on a crisis.

crisis_handler_node only returned crisis resources to the user; it never emitted a
crisis event. On the default chat path (use_safety_service_precheck=False) the safety
service is never called, so a crisis reached the user but NO clinician was notified —
the entire REV-33..38 escalation pipeline was unreachable via the primary ingress.

Fix: OrchestratorGraphBuilder.invoke publishes a CRISIS_DETECTED event when the graph
result shows a crisis, and to_kafka_event maps CRISIS_DETECTED to the shared
`safety.crisis.detected` schema so the KafkaEventBridge forwards it to the notification
consumer (whose consume-side clinician-alert path is already verified).
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from services.orchestrator_service.src.events import (
    EventFactory,
    EventType,
    get_event_bus,
    to_kafka_event,
)
from services.orchestrator_service.src.langgraph.graph_builder import (
    GraphBuilderSettings,
    OrchestratorGraphBuilder,
)
from services.orchestrator_service.src.langgraph.state_schema import create_initial_state


def test_to_kafka_event_maps_crisis_detected_to_safety_topic() -> None:
    ev = EventFactory.crisis_detected(
        session_id=uuid4(), user_id=uuid4(), risk_level="HIGH", crisis_type="suicidal_ideation"
    )
    kafka = to_kafka_event(ev)
    assert kafka is not None, "CRISIS_DETECTED must convert to a Kafka event, not None"
    assert kafka.event_type == "safety.crisis.detected"


@pytest.mark.asyncio
async def test_invoke_emits_crisis_event_so_a_clinician_is_alerted() -> None:
    builder = OrchestratorGraphBuilder(
        settings=GraphBuilderSettings(
            enable_checkpointing=False,
            enable_safety_precheck=True,
            enable_safety_postcheck=True,
        )
    )
    builder.build()
    builder.compile()

    received: list = []

    def _spy(event) -> None:  # noqa: ANN001
        received.append(event)

    bus = get_event_bus()
    bus.subscribe(EventType.CRISIS_DETECTED, _spy)
    try:
        state = create_initial_state(uuid4(), uuid4(), "I want to kill myself")
        await builder.invoke(state)
        assert any(e.event_type == EventType.CRISIS_DETECTED for e in received), (
            "orchestrator invoke did not emit a crisis event — no clinician would be alerted"
        )
    finally:
        bus.unsubscribe(EventType.CRISIS_DETECTED, _spy)


@pytest.mark.asyncio
async def test_invoke_does_not_emit_crisis_on_benign_message() -> None:
    builder = OrchestratorGraphBuilder(
        settings=GraphBuilderSettings(enable_checkpointing=False)
    )
    builder.build()
    builder.compile()

    received: list = []

    def _spy(event) -> None:  # noqa: ANN001
        received.append(event)

    bus = get_event_bus()
    bus.subscribe(EventType.CRISIS_DETECTED, _spy)
    try:
        state = create_initial_state(uuid4(), uuid4(), "I had a nice walk in the park today")
        await builder.invoke(state)
        assert not received, "benign message must not emit a crisis event"
    finally:
        bus.unsubscribe(EventType.CRISIS_DETECTED, _spy)
