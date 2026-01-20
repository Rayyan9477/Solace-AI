"""
Solace-AI Analytics Service - Event Consumer.

Consumes events from all Solace topics and feeds them to the analytics aggregator.
Implements event routing, filtering, and batch processing.

Architecture Layer: Infrastructure
Principles: Event-Driven Architecture, Consumer Groups, At-Least-Once Delivery
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict
import structlog

try:
    from .aggregations import AnalyticsAggregator
except ImportError:
    from aggregations import AnalyticsAggregator

logger = structlog.get_logger(__name__)

EventHandler = Callable[["AnalyticsEvent"], Awaitable[None]]


class EventCategory(str, Enum):
    """Categories of events consumed."""
    SESSION = "session"
    SAFETY = "safety"
    DIAGNOSIS = "diagnosis"
    THERAPY = "therapy"
    MEMORY = "memory"
    PERSONALITY = "personality"
    SYSTEM = "system"


@dataclass(frozen=True)
class AnalyticsEvent:
    """Normalized event for analytics processing."""
    event_id: UUID
    event_type: str
    category: EventCategory
    user_id: UUID
    session_id: UUID | None
    timestamp: datetime
    correlation_id: UUID
    source_service: str
    payload: dict[str, Any]

    @classmethod
    def from_raw(cls, data: dict[str, Any]) -> AnalyticsEvent:
        """Create AnalyticsEvent from raw event data."""
        metadata = data.get("metadata", {})
        event_type = data.get("event_type", "unknown")
        category = cls._determine_category(event_type)

        return cls(
            event_id=UUID(str(metadata.get("event_id", "00000000-0000-0000-0000-000000000000"))),
            event_type=event_type,
            category=category,
            user_id=UUID(str(data.get("user_id", "00000000-0000-0000-0000-000000000000"))),
            session_id=UUID(str(data["session_id"])) if data.get("session_id") else None,
            timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now(timezone.utc).isoformat())),
            correlation_id=UUID(str(metadata.get("correlation_id", "00000000-0000-0000-0000-000000000000"))),
            source_service=metadata.get("source_service", "unknown"),
            payload=data,
        )

    @staticmethod
    def _determine_category(event_type: str) -> EventCategory:
        """Determine event category from event type."""
        prefix_map = {
            "session.": EventCategory.SESSION,
            "safety.": EventCategory.SAFETY,
            "diagnosis.": EventCategory.DIAGNOSIS,
            "therapy.": EventCategory.THERAPY,
            "memory.": EventCategory.MEMORY,
            "personality.": EventCategory.PERSONALITY,
            "system.": EventCategory.SYSTEM,
        }
        for prefix, category in prefix_map.items():
            if event_type.startswith(prefix):
                return category
        return EventCategory.SYSTEM


class ConsumerConfig(BaseModel):
    """Configuration for analytics event consumer."""
    group_id: str = Field(default="analytics-service")
    topics: list[str] = Field(default_factory=lambda: [
        "solace.sessions",
        "solace.safety",
        "solace.assessments",
        "solace.therapy",
        "solace.memory",
        "solace.personality",
        "solace.analytics",
    ])
    batch_size: int = Field(default=100, ge=1, le=1000)
    batch_timeout_ms: int = Field(default=5000, ge=100, le=60000)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_ms: int = Field(default=1000, ge=100, le=30000)
    model_config = ConfigDict(frozen=True)


@dataclass
class ConsumerMetrics:
    """Metrics for consumer monitoring."""
    events_received: int = 0
    events_processed: int = 0
    events_failed: int = 0
    events_skipped: int = 0
    batches_processed: int = 0
    last_event_at: datetime | None = None
    processing_time_ms_total: int = 0
    events_by_category: dict[str, int] = field(default_factory=dict)

    def record_event(self, category: EventCategory, processing_time_ms: int) -> None:
        """Record processed event."""
        self.events_received += 1
        self.events_processed += 1
        self.processing_time_ms_total += processing_time_ms
        self.last_event_at = datetime.now(timezone.utc)
        cat_key = category.value
        self.events_by_category[cat_key] = self.events_by_category.get(cat_key, 0) + 1

    def record_failure(self) -> None:
        """Record failed event."""
        self.events_received += 1
        self.events_failed += 1
        self.last_event_at = datetime.now(timezone.utc)

    def record_skip(self) -> None:
        """Record skipped event."""
        self.events_received += 1
        self.events_skipped += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "events_received": self.events_received,
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "events_skipped": self.events_skipped,
            "batches_processed": self.batches_processed,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at else None,
            "avg_processing_time_ms": (
                self.processing_time_ms_total / max(1, self.events_processed)
            ),
            "events_by_category": self.events_by_category,
        }


class EventFilter:
    """Filters events based on configurable criteria."""

    def __init__(
        self,
        include_categories: list[EventCategory] | None = None,
        exclude_event_types: list[str] | None = None,
        sample_rate: float = 1.0,
    ) -> None:
        self._include_categories = set(include_categories) if include_categories else None
        self._exclude_event_types = set(exclude_event_types) if exclude_event_types else set()
        self._sample_rate = sample_rate
        self._sample_counter = 0

    def should_process(self, event: AnalyticsEvent) -> bool:
        """Determine if event should be processed."""
        if self._include_categories and event.category not in self._include_categories:
            return False
        if event.event_type in self._exclude_event_types:
            return False
        if self._sample_rate < 1.0:
            self._sample_counter += 1
            if (self._sample_counter % int(1 / self._sample_rate)) != 0:
                return False
        return True


class AnalyticsEventProcessor:
    """Processes events and feeds them to the aggregator."""

    def __init__(self, aggregator: AnalyticsAggregator) -> None:
        self._aggregator = aggregator
        self._handlers: dict[str, list[EventHandler]] = {}
        self._default_handlers: list[EventHandler] = []
        self._lock = asyncio.Lock()

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register handler for specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug("handler_registered", event_type=event_type)

    def register_default_handler(self, handler: EventHandler) -> None:
        """Register default handler for unhandled events."""
        self._default_handlers.append(handler)

    async def process(self, event: AnalyticsEvent) -> None:
        """Process a single analytics event."""
        handlers = self._handlers.get(event.event_type, []) or self._default_handlers

        for handler in handlers:
            await handler(event)

        await self._route_to_aggregator(event)

    async def _route_to_aggregator(self, event: AnalyticsEvent) -> None:
        """Route event to appropriate aggregator method."""
        if event.category == EventCategory.SESSION:
            await self._handle_session_event(event)
        elif event.category == EventCategory.SAFETY:
            await self._handle_safety_event(event)
        elif event.category == EventCategory.THERAPY:
            await self._handle_therapy_event(event)
        elif event.category == EventCategory.DIAGNOSIS:
            await self._handle_diagnosis_event(event)

    async def _handle_session_event(self, event: AnalyticsEvent) -> None:
        """Handle session-related events."""
        metadata = {
            "duration_seconds": event.payload.get("duration_seconds"),
            "generation_time_ms": event.payload.get("generation_time_ms"),
            "model_used": event.payload.get("model_used"),
            "tokens_used": event.payload.get("tokens_used"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        await self._aggregator.track_session_event(
            event_type=event.event_type,
            user_id=event.user_id,
            session_id=event.session_id,
            metadata=metadata,
        )

    async def _handle_safety_event(self, event: AnalyticsEvent) -> None:
        """Handle safety-related events."""
        risk_level = event.payload.get("risk_level") or event.payload.get("crisis_level", "NONE")
        detection_layer = event.payload.get("detection_layer", 1)
        metadata = {
            "risk_score": event.payload.get("risk_score"),
            "recommended_action": event.payload.get("recommended_action"),
        }
        await self._aggregator.track_safety_event(
            risk_level=str(risk_level),
            detection_layer=int(detection_layer),
            metadata=metadata,
        )

    async def _handle_therapy_event(self, event: AnalyticsEvent) -> None:
        """Handle therapy-related events."""
        modality = event.payload.get("modality", "unknown")
        technique = event.payload.get("technique", "unknown")
        engagement = event.payload.get("user_engagement_score")
        engagement_decimal = Decimal(str(engagement)) if engagement is not None else None
        await self._aggregator.track_therapy_event(
            modality=str(modality),
            technique=str(technique),
            engagement_score=engagement_decimal,
        )

    async def _handle_diagnosis_event(self, event: AnalyticsEvent) -> None:
        """Handle diagnosis-related events."""
        primary = event.payload.get("primary_hypothesis", {})
        severity = primary.get("severity", "unknown") if isinstance(primary, dict) else "unknown"
        stepped_care = event.payload.get("stepped_care_level", 0)
        await self._aggregator.track_diagnosis_event(
            assessment_type=event.event_type,
            severity=str(severity),
            stepped_care_level=int(stepped_care),
        )


class AnalyticsConsumer:
    """Main analytics event consumer."""

    def __init__(
        self,
        aggregator: AnalyticsAggregator,
        config: ConsumerConfig | None = None,
        event_filter: EventFilter | None = None,
    ) -> None:
        self._config = config or ConsumerConfig()
        self._aggregator = aggregator
        self._processor = AnalyticsEventProcessor(aggregator)
        self._filter = event_filter or EventFilter()
        self._metrics = ConsumerMetrics()
        self._running = False
        self._event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._batch: list[AnalyticsEvent] = []
        self._last_batch_time = datetime.now(timezone.utc)
        logger.info("analytics_consumer_initialized", group_id=self._config.group_id)

    @property
    def metrics(self) -> ConsumerMetrics:
        """Get consumer metrics."""
        return self._metrics

    @property
    def processor(self) -> AnalyticsEventProcessor:
        """Get event processor for handler registration."""
        return self._processor

    async def start(self) -> None:
        """Start the consumer."""
        self._running = True
        logger.info("analytics_consumer_started", topics=self._config.topics)

    async def stop(self) -> None:
        """Stop the consumer."""
        self._running = False
        if self._batch:
            await self._process_batch()
        logger.info("analytics_consumer_stopped", metrics=self._metrics.to_dict())

    async def enqueue_event(self, event_data: dict[str, Any]) -> None:
        """Enqueue an event for processing."""
        await self._event_queue.put(event_data)

    async def process_event(self, event_data: dict[str, Any]) -> bool:
        """Process a single event directly."""
        start_time = datetime.now(timezone.utc)
        try:
            event = AnalyticsEvent.from_raw(event_data)

            if not self._filter.should_process(event):
                self._metrics.record_skip()
                return True

            await self._processor.process(event)
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._metrics.record_event(event.category, processing_time)

            logger.debug(
                "event_processed",
                event_type=event.event_type,
                category=event.category.value,
                processing_time_ms=processing_time,
            )
            return True

        except Exception as e:
            self._metrics.record_failure()
            logger.error("event_processing_failed", error=str(e), event_data=event_data)
            return False

    async def consume_loop(self) -> None:
        """Main consumption loop for batch processing."""
        while self._running:
            try:
                event_data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=self._config.batch_timeout_ms / 1000,
                )
                event = AnalyticsEvent.from_raw(event_data)

                if self._filter.should_process(event):
                    self._batch.append(event)

                if len(self._batch) >= self._config.batch_size:
                    await self._process_batch()

            except asyncio.TimeoutError:
                if self._batch:
                    await self._process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consume_loop_error", error=str(e))
                await asyncio.sleep(1)

    async def _process_batch(self) -> None:
        """Process accumulated batch of events."""
        if not self._batch:
            return

        batch = self._batch
        self._batch = []
        self._metrics.batches_processed += 1

        start_time = datetime.now(timezone.utc)
        processed = 0
        failed = 0

        for event in batch:
            try:
                await self._processor.process(event)
                processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
                self._metrics.record_event(event.category, processing_time // max(1, len(batch)))
                processed += 1
            except Exception as e:
                self._metrics.record_failure()
                failed += 1
                logger.error("batch_event_failed", event_type=event.event_type, error=str(e))

        total_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        logger.info(
            "batch_processed",
            batch_size=len(batch),
            processed=processed,
            failed=failed,
            total_time_ms=total_time,
        )

    async def get_statistics(self) -> dict[str, Any]:
        """Get consumer statistics."""
        return {
            "config": {
                "group_id": self._config.group_id,
                "topics": self._config.topics,
                "batch_size": self._config.batch_size,
            },
            "metrics": self._metrics.to_dict(),
            "running": self._running,
            "queue_size": self._event_queue.qsize(),
            "current_batch_size": len(self._batch),
        }


def create_analytics_consumer(
    aggregator: AnalyticsAggregator,
    config: ConsumerConfig | None = None,
) -> AnalyticsConsumer:
    """Factory function to create analytics consumer."""
    return AnalyticsConsumer(aggregator=aggregator, config=config)
