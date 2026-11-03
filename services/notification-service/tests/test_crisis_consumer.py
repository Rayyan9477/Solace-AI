"""A.7 crisis E2E — the consumer-receipt half of the crisis path.

Sprint A.2 proved the *publish* side end to end (safety detects → local event →
outbox → poller drain → producer emits on ``safety.crisis.detected``). This test
closes the loop: given that same ``safety.crisis.detected`` Kafka event, the
notification service's ``SafetyEventConsumer`` must *receive* it and dispatch an
urgent clinician alert on the right channels. Broker + delivery are mocked; the
consumer, its handler, priority mapping, and the shared event schema are real.
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from solace_events.schemas import CrisisDetectedEvent, CrisisLevel

from src.consumers import SafetyEventConsumer
from src.domain import NotificationPriority


def _crisis_event(level: CrisisLevel) -> CrisisDetectedEvent:
    return CrisisDetectedEvent(
        user_id=uuid4(),
        session_id=uuid4(),
        crisis_level=level,
        trigger_indicators=["kill myself", "tonight"],
        detection_layer=1,
        confidence=Decimal("0.95"),
        escalation_action="notify_on_call_clinician",
        requires_human_review=True,
    )


class TestCrisisConsumerReceipt:
    """The notification consumer must turn a crisis event into an urgent alert."""

    def test_high_crisis_event_dispatches_urgent_multichannel_alert(
        self, mock_notification_service
    ) -> None:
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        # Spy the dispatch step; we assert the handler routes to it correctly.
        consumer._send_crisis_notification = AsyncMock()

        event = _crisis_event(CrisisLevel.HIGH)
        asyncio.run(consumer._handle_crisis_detected(event))

        consumer._send_crisis_notification.assert_awaited_once()
        kwargs = consumer._send_crisis_notification.await_args.kwargs
        assert kwargs["priority"] == NotificationPriority.HIGH
        assert kwargs["send_sms"] is True   # HIGH/CRITICAL escalate on every channel
        assert kwargs["send_push"] is True
        assert kwargs["user_id"] == event.user_id

    def test_critical_crisis_event_dispatches_critical_priority(
        self, mock_notification_service
    ) -> None:
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        consumer._send_crisis_notification = AsyncMock()

        event = _crisis_event(CrisisLevel.CRITICAL)
        asyncio.run(consumer._handle_crisis_detected(event))

        kwargs = consumer._send_crisis_notification.await_args.kwargs
        assert kwargs["priority"] == NotificationPriority.CRITICAL
        assert kwargs["send_sms"] is True
        assert kwargs["send_push"] is True

    def test_consumer_registers_crisis_handler(self, mock_notification_service) -> None:
        """The consumer must actually subscribe its handler to the crisis topic."""
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        # The mock consumer records registered handlers; the crisis type must be wired.
        assert consumer._consumer is not None
        handlers = consumer._consumer._handlers.get("safety.crisis.detected", [])
        assert consumer._handle_crisis_detected in handlers


class TestCrisisDeliveryFailureSurfaced:
    """M1 gate P0: a crisis alert that fails on ALL channels must NOT be silently
    committed. The handler must raise so the Kafka consumer marks it FAILED and
    routes it to the DLQ/retry path instead of dropping it."""

    def test_total_delivery_failure_raises_for_dlq(
        self, mock_notification_service
    ) -> None:
        mock_notification_service.send_notification = AsyncMock(
            return_value=SimpleNamespace(successful_deliveries=0, failed_deliveries=3)
        )
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        consumer._get_oncall_clinicians = AsyncMock(return_value=[])  # -> fallback recipient

        event = _crisis_event(CrisisLevel.CRITICAL)
        with pytest.raises(Exception):
            asyncio.run(consumer._handle_crisis_detected(event))

    def test_partial_delivery_success_does_not_raise(
        self, mock_notification_service
    ) -> None:
        """At least one successful channel is acceptable — don't re-DLQ a delivered alert."""
        mock_notification_service.send_notification = AsyncMock(
            return_value=SimpleNamespace(successful_deliveries=1, failed_deliveries=2)
        )
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        consumer._get_oncall_clinicians = AsyncMock(return_value=[])

        event = _crisis_event(CrisisLevel.CRITICAL)
        asyncio.run(consumer._handle_crisis_detected(event))  # must not raise


class TestFailedCrisisAlertDeadLettered:
    """M1 gate R2 (P0): a crisis alert that fails delivery must NOT be silently dropped.

    REV-35 made the handler raise on 0 deliveries, but the consumer marks a raised
    handler FAILED and COMMITS the offset — with no dead-letter handler the event was
    dropped. A DLQ handler must capture it for recovery."""

    def test_dead_letter_handler_is_registered(self, mock_notification_service) -> None:
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        assert consumer._consumer._dead_letter_handler == consumer._handle_dead_letter

    def test_failed_crisis_alert_is_captured_not_lost(
        self, mock_notification_service
    ) -> None:
        from solace_events.consumer import ProcessingStatus

        mock_notification_service.send_notification = AsyncMock(
            return_value=SimpleNamespace(successful_deliveries=0, failed_deliveries=3)
        )
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        consumer._get_oncall_clinicians = AsyncMock(return_value=[])

        event = _crisis_event(CrisisLevel.CRITICAL)
        # Drive the event through the real consumer path (deserialize → handler → DLQ).
        result = asyncio.run(
            consumer._consumer._process_message("solace.safety", 0, 0, event.to_dict())
        )
        # Marked FAILED (offset would commit) ...
        assert result.status == ProcessingStatus.FAILED
        # ... but the dead-letter handler captured it, so the crisis alert isn't lost.
        assert len(consumer._failed_events) == 1
        assert consumer._failed_events[0]["event_type"] == "safety.crisis.detected"


class _FakeDurableDLQStore:
    """A stand-in for the durable PostgresDLQStore: records survive a restart.

    Mirrors the ``save(DeadLetterRecord)`` surface of
    ``solace_events.postgres_stores.PostgresDLQStore`` so it can be injected
    without any consumer code change.
    """

    def __init__(self) -> None:
        self.saved: list = []

    async def save(self, record) -> None:  # noqa: ANN001 - DeadLetterRecord
        self.saved.append(record)


class TestFailedCrisisAlertDurablyPersisted:
    """A crisis alert that failed delivery must land in a DURABLE store, not only
    the in-memory ``_failed_events`` list which evaporates on worker restart.

    RED before the seam: injecting a durable store has no effect and nothing is
    persisted to it. GREEN: ``_handle_dead_letter`` writes the failed record
    through the injected store."""

    def test_failed_crisis_alert_is_persisted_to_injected_durable_store(
        self, mock_notification_service
    ) -> None:
        from solace_events.consumer import ProcessingStatus

        durable = _FakeDurableDLQStore()
        mock_notification_service.send_notification = AsyncMock(
            return_value=SimpleNamespace(successful_deliveries=0, failed_deliveries=3)
        )
        consumer = SafetyEventConsumer(
            mock_notification_service, use_mock=True, dlq_store=durable
        )
        consumer._get_oncall_clinicians = AsyncMock(return_value=[])

        event = _crisis_event(CrisisLevel.CRITICAL)
        result = asyncio.run(
            consumer._consumer._process_message("solace.safety", 0, 0, event.to_dict())
        )

        assert result.status == ProcessingStatus.FAILED
        # The record must be durably persisted, not only appended in memory.
        assert len(durable.saved) == 1
        persisted = durable.saved[0]
        assert persisted.event_type == "safety.crisis.detected"
        assert persisted.event_id == event.metadata.event_id
        assert persisted.user_id == event.user_id
        # Existing in-memory behavior is preserved alongside the durable seam.
        assert len(consumer._failed_events) == 1

    def test_default_dlq_store_is_in_memory_and_backward_compatible(
        self, mock_notification_service
    ) -> None:
        """No injected store → in-memory default, existing tests/dev unaffected."""
        consumer = SafetyEventConsumer(mock_notification_service, use_mock=True)
        assert consumer._dlq_store is not None  # a default seam exists
        # still captures in the legacy list
        event = _crisis_event(CrisisLevel.CRITICAL)
        asyncio.run(consumer._handle_dead_letter(event, "boom"))
        assert len(consumer._failed_events) == 1
