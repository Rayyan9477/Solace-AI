"""REV-01 (Sprint A.2): end-to-end crisis event flow over a mocked broker.

Detection -> local SafetyEventPublisher -> KafkaEventBridge -> outbox -> poller
drain -> mock Kafka producer. This is the A.2 acceptance path: it guards that
SafetyService, once given an event publisher, actually delivers
``safety.crisis.detected`` to the bus (the wiring main.py must perform).
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from services.safety_service.src.domain.crisis_detector import (
    CrisisDetector,
    CrisisDetectorSettings,
)
from services.safety_service.src.domain.escalation import (
    EscalationManager,
    EscalationSettings,
)
from services.safety_service.src.domain.service import (
    SafetyService,
    SafetyServiceSettings,
)
from services.safety_service.src.events import EventType, SafetyEventPublisher
from services.safety_service.src.infrastructure.event_bridge import KafkaEventBridge


@pytest.mark.asyncio
async def test_crisis_detection_publishes_to_bus() -> None:
    publisher = SafetyEventPublisher()
    await publisher.start()

    bridge = KafkaEventBridge(use_mock=True)
    await bridge.start()
    publisher.register_handler(EventType.CRISIS_DETECTED, bridge)

    service = SafetyService(
        settings=SafetyServiceSettings(),
        crisis_detector=CrisisDetector(CrisisDetectorSettings()),
        escalation_manager=EscalationManager(EscalationSettings()),
        event_publisher=publisher,
    )
    await service.initialize()

    try:
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I want to kill myself",
            check_type="pre_check",
        )
        # REV-02 guarantees an explicit high-severity utterance escalates.
        assert result.is_safe is False

        messages: list = []
        for _ in range(60):
            messages = bridge._publisher._producer.get_messages()
            if messages:
                break
            await asyncio.sleep(0.05)

        assert len(messages) >= 1, (
            "crisis detected but no event reached the (mock) Kafka producer — "
            "the event plane is not delivering"
        )
        topics = [t for t, _k, _v in messages]
        assert any("crisis" in t.lower() or "safety" in t.lower() for t in topics), topics
    finally:
        await bridge.stop()
        await publisher.stop()
