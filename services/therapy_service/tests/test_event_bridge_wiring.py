"""REV-01 (Sprint A.2): therapy Kafka event bridge wiring.

main.py calls ``initialize_event_bridge(..., postgres_pool=pool)``, but the
function did not accept that argument — raising a TypeError that was swallowed
by the lifespan try/except, so the therapy bridge never started. It also lacked
an OutboxPoller, so published events were never drained. These tests pin both.
"""
from __future__ import annotations

import pytest

from services.therapy_service.src.events import EventBus
from services.therapy_service.src.infrastructure.event_bridge import (
    initialize_event_bridge,
    shutdown_event_bridge,
)


@pytest.mark.asyncio
async def test_bridge_accepts_postgres_pool_and_runs_poller() -> None:
    bus = EventBus()
    # Before the fix this raised TypeError (unexpected keyword 'postgres_pool').
    bridge = await initialize_event_bridge(
        event_bus=bus, use_mock=True, postgres_pool=None
    )
    try:
        assert bridge._publisher is not None, "solace_events must be importable"
        assert bridge._poller is not None, "outbox drain poller must be running"
    finally:
        await shutdown_event_bridge()
