"""
Solace-AI Memory Service - Event Definitions and Handlers.

Defines memory-specific events and provides publishing/consuming utilities.
Integrates with the solace_events infrastructure for Kafka messaging.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Callable, Awaitable
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
import structlog

from src.solace_events.schemas import BaseEvent, EventMetadata
from src.solace_events.publisher import EventPublisher
from src.solace_events.consumer import EventConsumer

logger = structlog.get_logger(__name__)

MEMORY_TOPIC = "solace.memory"
MEMORY_DLQ_TOPIC = "solace.memory.dlq"


class MemoryStoredEvent(BaseEvent):
    """Emitted when a memory is stored."""
    event_type: Literal["memory.stored"] = "memory.stored"
    record_id: UUID
    tier: str
    content_type: str
    importance_score: Decimal = Field(ge=Decimal("0"), le=Decimal("1"))
    storage_backend: str
    storage_time_ms: int = Field(ge=0)


class MemoryRetrievedEvent(BaseEvent):
    """Emitted when memories are retrieved."""
    event_type: Literal["memory.retrieved"] = "memory.retrieved"
    query_text: str
    records_returned: int = Field(ge=0)
    tiers_searched: list[str]
    retrieval_time_ms: int = Field(ge=0)
    cache_hit: bool = False


class MemoryConsolidatedEvent(BaseEvent):
    """Emitted when session memory is consolidated."""
    event_type: Literal["memory.consolidated"] = "memory.consolidated"
    consolidation_id: UUID
    summary_generated: bool
    facts_extracted: int = Field(ge=0)
    triples_created: int = Field(ge=0)
    memories_archived: int = Field(ge=0)
    consolidation_time_ms: int = Field(ge=0)


class MemoryDecayedEvent(BaseEvent):
    """Emitted when memory decay is applied."""
    event_type: Literal["memory.decayed"] = "memory.decayed"
    records_processed: int = Field(ge=0)
    records_archived: int = Field(ge=0)
    records_deleted: int = Field(ge=0)
    decay_run_id: UUID = Field(default_factory=uuid4)


class ContextAssembledEvent(BaseEvent):
    """Emitted when context is assembled for LLM."""
    event_type: Literal["memory.context.assembled"] = "memory.context.assembled"
    context_id: UUID
    total_tokens: int = Field(ge=0)
    sources_used: list[str]
    retrieval_count: int = Field(ge=0)
    assembly_time_ms: int = Field(ge=0)


class UserProfileUpdatedEvent(BaseEvent):
    """Emitted when user profile is updated."""
    event_type: Literal["memory.profile.updated"] = "memory.profile.updated"
    update_type: str
    fields_updated: list[str]
    profile_version: int = Field(ge=1)


class KnowledgeGraphUpdatedEvent(BaseEvent):
    """Emitted when knowledge graph is modified."""
    event_type: Literal["memory.knowledge_graph.updated"] = "memory.knowledge_graph.updated"
    triples_added: int = Field(ge=0)
    triples_removed: int = Field(ge=0)
    entities_affected: list[str]


class SafetyMemoryCreatedEvent(BaseEvent):
    """Emitted when safety-critical memory is created."""
    event_type: Literal["memory.safety.created"] = "memory.safety.created"
    record_id: UUID
    safety_type: str
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class MemoryEventFactory:
    """Factory for creating memory events with proper metadata."""

    @staticmethod
    def memory_stored(user_id: UUID, session_id: UUID | None, record_id: UUID, tier: str, content_type: str,
                      importance_score: Decimal, storage_backend: str, storage_time_ms: int,
                      correlation_id: UUID | None = None) -> MemoryStoredEvent:
        metadata = EventMetadata(source_service="memory-service", correlation_id=correlation_id or uuid4())
        return MemoryStoredEvent(user_id=user_id, session_id=session_id, metadata=metadata, record_id=record_id,
                                 tier=tier, content_type=content_type, importance_score=importance_score,
                                 storage_backend=storage_backend, storage_time_ms=storage_time_ms)

    @staticmethod
    def memory_retrieved(user_id: UUID, session_id: UUID | None, query_text: str, records_returned: int,
                         tiers_searched: list[str], retrieval_time_ms: int, cache_hit: bool = False,
                         correlation_id: UUID | None = None) -> MemoryRetrievedEvent:
        metadata = EventMetadata(source_service="memory-service", correlation_id=correlation_id or uuid4())
        return MemoryRetrievedEvent(user_id=user_id, session_id=session_id, metadata=metadata, query_text=query_text,
                                    records_returned=records_returned, tiers_searched=tiers_searched,
                                    retrieval_time_ms=retrieval_time_ms, cache_hit=cache_hit)

    @staticmethod
    def memory_consolidated(user_id: UUID, session_id: UUID, consolidation_id: UUID, summary_generated: bool,
                            facts_extracted: int, triples_created: int, memories_archived: int,
                            consolidation_time_ms: int, correlation_id: UUID | None = None) -> MemoryConsolidatedEvent:
        metadata = EventMetadata(source_service="memory-service", correlation_id=correlation_id or uuid4())
        return MemoryConsolidatedEvent(user_id=user_id, session_id=session_id, metadata=metadata,
                                       consolidation_id=consolidation_id, summary_generated=summary_generated,
                                       facts_extracted=facts_extracted, triples_created=triples_created,
                                       memories_archived=memories_archived, consolidation_time_ms=consolidation_time_ms)

    @staticmethod
    def context_assembled(user_id: UUID, session_id: UUID, context_id: UUID, total_tokens: int,
                          sources_used: list[str], retrieval_count: int, assembly_time_ms: int,
                          correlation_id: UUID | None = None) -> ContextAssembledEvent:
        metadata = EventMetadata(source_service="memory-service", correlation_id=correlation_id or uuid4())
        return ContextAssembledEvent(user_id=user_id, session_id=session_id, metadata=metadata, context_id=context_id,
                                     total_tokens=total_tokens, sources_used=sources_used,
                                     retrieval_count=retrieval_count, assembly_time_ms=assembly_time_ms)

    @staticmethod
    def safety_memory_created(user_id: UUID, session_id: UUID | None, record_id: UUID,
                              safety_type: str, priority: str, correlation_id: UUID | None = None) -> SafetyMemoryCreatedEvent:
        metadata = EventMetadata(source_service="memory-service", correlation_id=correlation_id or uuid4())
        return SafetyMemoryCreatedEvent(user_id=user_id, session_id=session_id, metadata=metadata,
                                        record_id=record_id, safety_type=safety_type, priority=priority)


class MemoryEventPublisher:
    """Publisher for memory service events."""

    def __init__(self, publisher: EventPublisher | None = None) -> None:
        self._publisher = publisher
        self._enabled = publisher is not None
        self._stats = {"published": 0, "failed": 0}

    async def publish(self, event: BaseEvent) -> bool:
        """Publish a memory event."""
        if not self._enabled or not self._publisher:
            logger.debug("event_publishing_disabled", event_type=event.event_type)
            return False
        try:
            await self._publisher.publish(event, MEMORY_TOPIC)
            self._stats["published"] += 1
            logger.debug("event_published", event_type=event.event_type, event_id=str(event.metadata.event_id))
            return True
        except Exception as e:
            self._stats["failed"] += 1
            logger.error("event_publish_failed", event_type=event.event_type, error=str(e))
            return False

    async def publish_memory_stored(self, user_id: UUID, session_id: UUID | None, record_id: UUID, tier: str,
                                    content_type: str, importance_score: Decimal, storage_backend: str,
                                    storage_time_ms: int) -> bool:
        event = MemoryEventFactory.memory_stored(user_id, session_id, record_id, tier, content_type,
                                                 importance_score, storage_backend, storage_time_ms)
        return await self.publish(event)

    async def publish_memory_consolidated(self, user_id: UUID, session_id: UUID, consolidation_id: UUID,
                                          summary_generated: bool, facts_extracted: int, triples_created: int,
                                          memories_archived: int, consolidation_time_ms: int) -> bool:
        event = MemoryEventFactory.memory_consolidated(user_id, session_id, consolidation_id, summary_generated,
                                                       facts_extracted, triples_created, memories_archived, consolidation_time_ms)
        return await self.publish(event)

    async def publish_safety_memory(self, user_id: UUID, session_id: UUID | None, record_id: UUID,
                                    safety_type: str, priority: str = "HIGH") -> bool:
        event = MemoryEventFactory.safety_memory_created(user_id, session_id, record_id, safety_type, priority)
        return await self.publish(event)

    def get_stats(self) -> dict[str, Any]:
        return {**self._stats, "enabled": self._enabled, "topic": MEMORY_TOPIC}


EventHandler = Callable[[BaseEvent], Awaitable[None]]


class MemoryEventConsumer:
    """Consumer for memory service events."""

    def __init__(self, consumer: EventConsumer | None = None) -> None:
        self._consumer = consumer
        self._handlers: dict[str, list[EventHandler]] = {}
        self._enabled = consumer is not None
        self._stats = {"processed": 0, "failed": 0}

    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.info("handler_registered", event_type=event_type)

    async def process_event(self, event_data: dict[str, Any]) -> bool:
        """Process a received event."""
        event_type = event_data.get("event_type", "unknown")
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug("no_handler_for_event", event_type=event_type)
            return True
        try:
            event = self._deserialize_event(event_data)
            for handler in handlers:
                await handler(event)
            self._stats["processed"] += 1
            return True
        except Exception as e:
            self._stats["failed"] += 1
            logger.error("event_processing_failed", event_type=event_type, error=str(e))
            return False

    def _deserialize_event(self, data: dict[str, Any]) -> BaseEvent:
        """Deserialize event data to appropriate event class."""
        event_type = data.get("event_type", "")
        event_classes = {
            "memory.stored": MemoryStoredEvent,
            "memory.retrieved": MemoryRetrievedEvent,
            "memory.consolidated": MemoryConsolidatedEvent,
            "memory.decayed": MemoryDecayedEvent,
            "memory.context.assembled": ContextAssembledEvent,
            "memory.profile.updated": UserProfileUpdatedEvent,
            "memory.knowledge_graph.updated": KnowledgeGraphUpdatedEvent,
            "memory.safety.created": SafetyMemoryCreatedEvent,
        }
        event_class = event_classes.get(event_type, BaseEvent)
        return event_class.model_validate(data)

    async def start(self) -> None:
        """Start consuming events."""
        if not self._enabled or not self._consumer:
            logger.warning("event_consumer_not_enabled")
            return
        logger.info("memory_event_consumer_starting", topic=MEMORY_TOPIC)
        await self._consumer.start()

    async def stop(self) -> None:
        """Stop consuming events."""
        if self._consumer:
            await self._consumer.stop()
            logger.info("memory_event_consumer_stopped")

    def get_stats(self) -> dict[str, Any]:
        return {**self._stats, "enabled": self._enabled, "handlers": list(self._handlers.keys())}
