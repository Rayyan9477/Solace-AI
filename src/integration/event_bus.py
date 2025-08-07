"""
Event-Driven Communication System for Solace-AI

This module provides a comprehensive event bus implementation with:
- Asynchronous pub/sub messaging
- Event sourcing for complete audit trails
- Dead letter queues for failed events
- Event replay capabilities
- Subscription management
- Comprehensive metrics and monitoring
- Circuit breaker patterns for resilience
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import logging
from contextlib import asynccontextmanager
import weakref

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Standard event types for the Solace-AI ecosystem."""
    # Agent Communication
    AGENT_REQUEST = "agent.request"
    AGENT_RESPONSE = "agent.response" 
    AGENT_ERROR = "agent.error"
    AGENT_STATUS = "agent.status"
    
    # Clinical Events
    CLINICAL_ASSESSMENT = "clinical.assessment"
    RISK_ALERT = "clinical.risk_alert"
    INTERVENTION_REQUIRED = "clinical.intervention_required"
    THERAPY_SESSION_START = "therapy.session.start"
    THERAPY_SESSION_END = "therapy.session.end"
    
    # Supervision Events
    VALIDATION_REQUEST = "supervision.validation_request"
    VALIDATION_RESULT = "supervision.validation_result"
    QUALITY_GATE = "supervision.quality_gate"
    CONSENSUS_REQUEST = "supervision.consensus_request"
    
    # Friction Events
    FRICTION_ASSESSMENT = "friction.assessment"
    FRICTION_APPLICATION = "friction.application"
    READINESS_UPDATE = "friction.readiness_update"
    BREAKTHROUGH_DETECTED = "friction.breakthrough_detected"
    
    # System Events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "system.health_check"
    METRICS_UPDATE = "system.metrics_update"


class EventPriority(Enum):
    """Event priority levels for processing order."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Event:
    """Core event structure for the Solace-AI event bus."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Union[EventType, str] = field(default=EventType.AGENT_REQUEST)
    source_agent: Optional[str] = None
    target_agent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = field(default=EventPriority.NORMAL)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.event_type, str):
            try:
                self.event_type = EventType(self.event_type)
            except ValueError:
                # Allow custom event types as strings
                pass
        
        # Set default TTL based on priority
        if self.ttl is None:
            self.ttl = {
                EventPriority.CRITICAL: 300,  # 5 minutes
                EventPriority.HIGH: 600,      # 10 minutes
                EventPriority.NORMAL: 1800,   # 30 minutes
                EventPriority.LOW: 3600       # 1 hour
            }.get(self.priority, 1800)
    
    def is_expired(self) -> bool:
        """Check if the event has expired based on TTL."""
        if self.ttl is None:
            return False
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl
    
    def can_retry(self) -> bool:
        """Check if the event can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value if isinstance(self.event_type, EventType) else self.event_type
        result['priority'] = self.priority.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['priority'] = EventPriority(data['priority'])
        return cls(**data)


EventHandler = Callable[[Event], Any]


@dataclass 
class EventSubscription:
    """Represents a subscription to events."""
    
    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_types: Set[Union[EventType, str]] = field(default_factory=set)
    handler: Optional[EventHandler] = None
    agent_id: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_processed: Optional[datetime] = None
    processed_count: int = 0
    error_count: int = 0
    max_errors: int = 10
    
    def matches(self, event: Event) -> bool:
        """Check if this subscription matches the given event."""
        if not self.is_active:
            return False
        
        # Check event type
        event_type = event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
        type_matches = any(
            (et.value if isinstance(et, EventType) else et) == event_type
            for et in self.event_types
        )
        
        if not type_matches:
            return False
        
        # Apply filters
        for key, value in self.filters.items():
            if hasattr(event, key):
                if getattr(event, key) != value:
                    return False
            elif key in event.data:
                if event.data[key] != value:
                    return False
            elif key in event.metadata:
                if event.metadata[key] != value:
                    return False
            else:
                return False
        
        return True
    
    def should_disable(self) -> bool:
        """Check if subscription should be disabled due to errors."""
        return self.error_count >= self.max_errors


@dataclass
class DeadLetter:
    """Represents a failed event in the dead letter queue."""
    
    event: Event
    subscription_id: str
    error_message: str
    failed_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'event': self.event.to_dict(),
            'subscription_id': self.subscription_id,
            'error_message': self.error_message,
            'failed_at': self.failed_at.isoformat(),
            'retry_count': self.retry_count
        }


@dataclass
class EventMetrics:
    """Event processing metrics."""
    
    events_published: int = 0
    events_processed: int = 0
    events_failed: int = 0
    dead_letters: int = 0
    active_subscriptions: int = 0
    average_processing_time: float = 0.0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_processing_time(self, duration: float) -> None:
        """Record event processing time."""
        self.processing_times.append(duration)
        self.average_processing_time = sum(self.processing_times) / len(self.processing_times)


class CircuitBreaker:
    """Circuit breaker for event processing resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0
        self.state = 'closed'
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'


class EventBus:
    """
    Comprehensive event bus for Solace-AI with pub/sub messaging,
    event sourcing, dead letter queues, and replay capabilities.
    """
    
    def __init__(self, max_event_store_size: int = 10000):
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_store: deque = deque(maxlen=max_event_store_size)
        self.dead_letter_queue: List[DeadLetter] = []
        self.metrics = EventMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = defaultdict(CircuitBreaker)
        self._running = False
        self._processing_tasks: Set[asyncio.Task] = set()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        
        # Weak references to prevent memory leaks
        self._subscription_refs: Dict[str, weakref.ref] = {}
        
        logger.info("EventBus initialized")
    
    async def start(self) -> None:
        """Start the event bus processing."""
        if self._running:
            return
        
        self._running = True
        
        # Start event processing tasks
        for i in range(3):  # Multiple workers for parallel processing
            task = asyncio.create_task(self._process_events())
            self._processing_tasks.add(task)
        
        # Start maintenance task
        maintenance_task = asyncio.create_task(self._maintenance_loop())
        self._processing_tasks.add(maintenance_task)
        
        logger.info("EventBus started with event processing workers")
    
    async def stop(self) -> None:
        """Stop the event bus processing."""
        self._running = False
        
        # Cancel all processing tasks
        for task in self._processing_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        self._processing_tasks.clear()
        logger.info("EventBus stopped")
    
    async def publish(self, event: Event) -> str:
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
            
        Returns:
            Event ID
        """
        if event.is_expired():
            logger.warning(f"Event {event.event_id} is expired, not publishing")
            return event.event_id
        
        # Store event for replay capabilities
        self.event_store.append(event)
        
        # Add to processing queue
        await self._event_queue.put(event)
        
        self.metrics.events_published += 1
        
        logger.debug(f"Published event {event.event_id} of type {event.event_type}")
        return event.event_id
    
    def subscribe(
        self,
        event_types: Union[EventType, List[Union[EventType, str]], str],
        handler: EventHandler,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Subscribe to events.
        
        Args:
            event_types: Event type(s) to subscribe to
            handler: Handler function for events
            agent_id: Optional agent identifier
            filters: Optional filters for events
            
        Returns:
            Subscription ID
        """
        if isinstance(event_types, (EventType, str)):
            event_types = [event_types]
        
        subscription = EventSubscription(
            event_types=set(event_types),
            handler=handler,
            agent_id=agent_id,
            filters=filters or {}
        )
        
        self.subscriptions[subscription.subscription_id] = subscription
        self.metrics.active_subscriptions += 1
        
        logger.info(f"Created subscription {subscription.subscription_id} for types {event_types}")
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: Subscription to remove
            
        Returns:
            True if successfully unsubscribed
        """
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            self.metrics.active_subscriptions -= 1
            logger.info(f"Removed subscription {subscription_id}")
            return True
        return False
    
    async def replay_events(
        self,
        from_timestamp: datetime,
        to_timestamp: Optional[datetime] = None,
        event_types: Optional[List[Union[EventType, str]]] = None,
        target_subscription: Optional[str] = None
    ) -> int:
        """
        Replay events from the event store.
        
        Args:
            from_timestamp: Start timestamp for replay
            to_timestamp: End timestamp for replay (default: now)
            event_types: Optional event types to filter
            target_subscription: Optional specific subscription to replay to
            
        Returns:
            Number of events replayed
        """
        to_timestamp = to_timestamp or datetime.now()
        replayed_count = 0
        
        # Filter events by timestamp and type
        events_to_replay = []
        for event in self.event_store:
            if from_timestamp <= event.timestamp <= to_timestamp:
                if event_types is None or any(
                    (et.value if isinstance(et, EventType) else et) == 
                    (event.event_type.value if isinstance(event.event_type, EventType) else event.event_type)
                    for et in event_types
                ):
                    events_to_replay.append(event)
        
        # Replay events
        for event in events_to_replay:
            # Create a copy with new ID to avoid conflicts
            replay_event = Event(
                event_type=event.event_type,
                source_agent=event.source_agent,
                target_agent=event.target_agent,
                user_id=event.user_id,
                session_id=event.session_id,
                timestamp=event.timestamp,
                priority=event.priority,
                data=event.data.copy(),
                metadata={**event.metadata, 'replayed': True, 'original_id': event.event_id},
                correlation_id=event.correlation_id
            )
            
            if target_subscription:
                # Replay to specific subscription
                if target_subscription in self.subscriptions:
                    subscription = self.subscriptions[target_subscription]
                    if subscription.matches(replay_event):
                        await self._process_event_for_subscription(replay_event, subscription)
                        replayed_count += 1
            else:
                # Replay to all matching subscriptions
                await self._event_queue.put(replay_event)
                replayed_count += 1
        
        logger.info(f"Replayed {replayed_count} events")
        return replayed_count
    
    def get_dead_letters(self) -> List[DeadLetter]:
        """Get all dead letter events."""
        return self.dead_letter_queue.copy()
    
    async def reprocess_dead_letter(self, dead_letter: DeadLetter) -> bool:
        """
        Reprocess a dead letter event.
        
        Args:
            dead_letter: Dead letter to reprocess
            
        Returns:
            True if successfully reprocessed
        """
        dead_letter.retry_count += 1
        
        try:
            if dead_letter.subscription_id in self.subscriptions:
                subscription = self.subscriptions[dead_letter.subscription_id]
                await self._process_event_for_subscription(dead_letter.event, subscription)
                
                # Remove from dead letter queue
                self.dead_letter_queue.remove(dead_letter)
                logger.info(f"Successfully reprocessed dead letter {dead_letter.event.event_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to reprocess dead letter {dead_letter.event.event_id}: {e}")
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            'events_published': self.metrics.events_published,
            'events_processed': self.metrics.events_processed,
            'events_failed': self.metrics.events_failed,
            'dead_letters': len(self.dead_letter_queue),
            'active_subscriptions': len(self.subscriptions),
            'average_processing_time': self.metrics.average_processing_time,
            'event_store_size': len(self.event_store),
            'queue_size': self._event_queue.qsize() if hasattr(self._event_queue, 'qsize') else 0
        }
    
    async def _process_events(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                
                if event.is_expired():
                    logger.warning(f"Event {event.event_id} expired before processing")
                    continue
                
                start_time = time.time()
                
                # Process event for all matching subscriptions
                processed = False
                for subscription in list(self.subscriptions.values()):
                    if subscription.matches(event):
                        try:
                            await self._process_event_for_subscription(event, subscription)
                            processed = True
                        except Exception as e:
                            logger.error(f"Error processing event {event.event_id} for subscription {subscription.subscription_id}: {e}")
                
                if processed:
                    processing_time = time.time() - start_time
                    self.metrics.record_processing_time(processing_time)
                    self.metrics.events_processed += 1
                
            except asyncio.TimeoutError:
                # Normal timeout, continue processing
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_event_for_subscription(self, event: Event, subscription: EventSubscription) -> None:
        """
        Process an event for a specific subscription.
        
        Args:
            event: Event to process
            subscription: Subscription to process for
        """
        if subscription.should_disable():
            logger.warning(f"Subscription {subscription.subscription_id} disabled due to too many errors")
            subscription.is_active = False
            return
        
        circuit_breaker = self.circuit_breakers[subscription.subscription_id]
        
        try:
            # Execute handler with circuit breaker protection
            if asyncio.iscoroutinefunction(subscription.handler):
                await circuit_breaker.call(subscription.handler, event)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, circuit_breaker.call, subscription.handler, event
                )
            
            subscription.processed_count += 1
            subscription.last_processed = datetime.now()
            
        except Exception as e:
            subscription.error_count += 1
            self.metrics.events_failed += 1
            
            # Add to dead letter queue
            dead_letter = DeadLetter(
                event=event,
                subscription_id=subscription.subscription_id,
                error_message=str(e)
            )
            self.dead_letter_queue.append(dead_letter)
            
            logger.error(f"Failed to process event {event.event_id} for subscription {subscription.subscription_id}: {e}")
    
    async def _maintenance_loop(self) -> None:
        """Maintenance loop for cleanup and monitoring."""
        while self._running:
            try:
                # Clean up dead letters older than 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.dead_letter_queue = [
                    dl for dl in self.dead_letter_queue
                    if dl.failed_at > cutoff_time
                ]
                
                # Clean up inactive subscriptions
                inactive_subscriptions = [
                    sub_id for sub_id, sub in self.subscriptions.items()
                    if not sub.is_active and sub.should_disable()
                ]
                
                for sub_id in inactive_subscriptions:
                    self.unsubscribe(sub_id)
                
                # Log metrics periodically
                metrics = self.get_metrics()
                logger.debug(f"EventBus metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
            
            await asyncio.sleep(300)  # Run every 5 minutes
    
    @asynccontextmanager
    async def managed_subscription(self, *args, **kwargs):
        """Context manager for automatic subscription cleanup."""
        subscription_id = self.subscribe(*args, **kwargs)
        try:
            yield subscription_id
        finally:
            self.unsubscribe(subscription_id)


# Singleton instance for global access
_event_bus_instance = None

def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = EventBus()
    return _event_bus_instance