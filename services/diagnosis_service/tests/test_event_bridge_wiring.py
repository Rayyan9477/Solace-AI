"""REV-01 (Sprint A.2): diagnosis Kafka event bridge must drain its outbox.

The bridge published via an outbox-backed EventPublisher but never ran an
OutboxPoller, so diagnosis events were saved to the outbox and never sent.
(Separately, main.py now injects the EventDispatcher into DiagnosisService so
events are actually dispatched to this bridge.)
"""
from __future__ import annotations

import pytest

from services.diagnosis_service.src.infrastructure.event_bridge import (
    initialize_event_bridge,
)


@pytest.mark.asyncio
async def test_diagnosis_bridge_runs_outbox_poller() -> None:
    bridge = await initialize_event_bridge(use_mock=True, postgres_pool=None)
    try:
        assert bridge._publisher is not None, "solace_events must be importable"
        assert bridge._poller is not None, "outbox drain poller must be running"
    finally:
        await bridge.stop()
