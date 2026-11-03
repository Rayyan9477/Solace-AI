"""REV-01 (Sprint A.2): the Kafka event bridge must DRAIN its outbox.

The bridge publishes via an outbox-backed EventPublisher. Without a running
OutboxPoller, events are saved to the outbox but never sent to the producer —
which is the core reason no ``safety.crisis.detected`` event ever reached the
bus. These tests pin the drain behaviour with a mock producer.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest

from services.safety_service.src.events import CrisisDetectedEvent
from services.safety_service.src.infrastructure.event_bridge import KafkaEventBridge


@pytest.mark.asyncio
async def test_bridge_drains_outbox_to_producer() -> None:
    """A published crisis event must reach the (mock) producer after start()."""
    bridge = KafkaEventBridge(use_mock=True)
    assert bridge._publisher is not None, "solace_events must be importable in tests"

    await bridge.start()
    try:
        event = CrisisDetectedEvent(
            user_id=uuid4(),
            crisis_level="CRITICAL",
            risk_score=Decimal("0.95"),
            requires_escalation=True,
        )
        await bridge.handle(event)

        # Wait for the outbox poller to drain the record to the producer.
        messages: list = []
        for _ in range(40):
            messages = bridge._publisher._producer.get_messages()
            if messages:
                break
            await asyncio.sleep(0.05)

        assert len(messages) >= 1, (
            "crisis event was queued in the outbox but never drained to Kafka "
            "(no OutboxPoller running)"
        )
        topic, _key, value = messages[0]
        assert topic
        assert value  # non-empty serialized payload
    finally:
        await bridge.stop()
