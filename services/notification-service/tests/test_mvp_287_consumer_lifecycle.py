"""MVP-287: a half-initialised safety consumer stops cleanly and never lingers in the module global."""
from __future__ import annotations

import pytest

from src import consumers


async def test_mvp_287_half_initialised_consumer_stops_cleanly(monkeypatch, mock_notification_service) -> None:
    monkeypatch.setattr(consumers, "_KAFKA_AVAILABLE", False)
    consumer = consumers.SafetyEventConsumer(mock_notification_service, use_mock=True)
    await consumer.stop()
    assert consumer.is_running is False and consumer.consume_task is None


async def test_mvp_287_failed_start_leaves_no_global(monkeypatch, mock_notification_service) -> None:
    monkeypatch.setattr(consumers, "_KAFKA_AVAILABLE", False)
    monkeypatch.setattr(consumers, "_safety_consumer", None)
    with pytest.raises(RuntimeError):
        await consumers.initialize_safety_consumer(mock_notification_service, use_mock=True)
    assert consumers.get_safety_consumer() is None
    await consumers.shutdown_safety_consumer()


async def test_mvp_287_stop_is_idempotent(mock_notification_service) -> None:
    consumer = consumers.SafetyEventConsumer(mock_notification_service, use_mock=True)
    await consumer.start()
    inner_stops = 0
    real_stop = consumer._consumer.stop

    async def counting_stop() -> None:
        nonlocal inner_stops
        inner_stops += 1
        await real_stop()

    consumer._consumer.stop = counting_stop
    await consumer.stop()
    await consumer.stop()
    assert inner_stops == 1 and consumer.is_running is False and consumer.consume_task is None


async def test_mvp_287_failed_start_releases_the_partial_consumer(monkeypatch, mock_notification_service) -> None:
    monkeypatch.setattr(consumers, "_safety_consumer", None)
    stopped: list[bool] = []

    async def failing_start(self) -> None:
        raise ConnectionError("broker unreachable")

    async def recording_stop(self) -> None:
        stopped.append(True)

    monkeypatch.setattr(consumers.SafetyEventConsumer, "start", failing_start)
    monkeypatch.setattr(consumers.SafetyEventConsumer, "stop", recording_stop)
    with pytest.raises(ConnectionError):
        await consumers.initialize_safety_consumer(mock_notification_service, use_mock=True)
    assert stopped == [True] and consumers.get_safety_consumer() is None
