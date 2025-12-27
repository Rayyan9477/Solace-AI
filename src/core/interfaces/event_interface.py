"""
Abstract interface for Event handling.

This interface provides a contract for event-driven architecture,
enabling loose coupling between components through events.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import uuid


class EventPriority(Enum):
    """Enum for event priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Represents an event in the system."""
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EventSubscription:
    """Represents an event subscription."""
    event_type: str
    handler: Callable[[Event], Awaitable[None]]
    subscriber_id: str
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Optional[Callable[[Event], bool]] = None


class EventInterface(ABC):
    """
    Abstract base class for event handling.
    
    This interface defines the contract for event publishers and subscribers,
    enabling event-driven communication between components.
    """
    
    @abstractmethod
    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the system.
        
        Args:
            event: Event to publish
            
        Returns:
            bool: True if event was published successfully
        """
        pass
    
    @abstractmethod
    async def subscribe(self, subscription: EventSubscription) -> bool:
        """
        Subscribe to events of a specific type.
        
        Args:
            subscription: Event subscription details
            
        Returns:
            bool: True if subscription was successful
        """
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscriber_id: str, event_type: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscriber_id: ID of the subscriber
            event_type: Type of events to unsubscribe from
            
        Returns:
            bool: True if unsubscription was successful
        """
        pass
    
    @abstractmethod
    async def get_subscriptions(self, subscriber_id: str) -> List[EventSubscription]:
        """
        Get all subscriptions for a subscriber.
        
        Args:
            subscriber_id: ID of the subscriber
            
        Returns:
            List of event subscriptions
        """
        pass


class EventBus(EventInterface):
    """
    Default implementation of EventInterface.
    
    Provides in-memory event bus functionality with support for
    async event handling, priority queues, and filtering.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscriptions: Dict[str, List[EventSubscription]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'active_subscriptions': 0,
            'processing_errors': 0
        }
    
    async def start(self) -> None:
        """Start the event bus processing."""
        if not self._running:
            self._running = True
            self._processing_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop the event bus processing."""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    async def publish(self, event: Event) -> bool:
        """Publish an event to the bus."""
        try:
            await self._event_queue.put(event)
            self._stats['events_published'] += 1
            return True
        except (asyncio.QueueFull, RuntimeError, TypeError):
            return False

    async def subscribe(self, subscription: EventSubscription) -> bool:
        """Subscribe to events of a specific type."""
        try:
            event_type = subscription.event_type
            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []

            # Remove existing subscription if it exists
            self._subscriptions[event_type] = [
                sub for sub in self._subscriptions[event_type]
                if sub.subscriber_id != subscription.subscriber_id
            ]

            self._subscriptions[event_type].append(subscription)
            self._stats['active_subscriptions'] = sum(
                len(subs) for subs in self._subscriptions.values()
            )
            return True
        except (KeyError, TypeError, AttributeError):
            return False

    async def unsubscribe(self, subscriber_id: str, event_type: str) -> bool:
        """Unsubscribe from events."""
        try:
            if event_type in self._subscriptions:
                self._subscriptions[event_type] = [
                    sub for sub in self._subscriptions[event_type]
                    if sub.subscriber_id != subscriber_id
                ]

                if not self._subscriptions[event_type]:
                    del self._subscriptions[event_type]

                self._stats['active_subscriptions'] = sum(
                    len(subs) for subs in self._subscriptions.values()
                )
            return True
        except (KeyError, TypeError, AttributeError):
            return False
    
    async def get_subscriptions(self, subscriber_id: str) -> List[EventSubscription]:
        """Get all subscriptions for a subscriber."""
        subscriptions = []
        for event_type, subs in self._subscriptions.items():
            for sub in subs:
                if sub.subscriber_id == subscriber_id:
                    subscriptions.append(sub)
        return subscriptions
    
    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                # Wait for events with timeout to allow for shutdown
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self._handle_event(event)
                self._stats['events_processed'] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._stats['processing_errors'] += 1
                # Log error (would use proper logger in real implementation)
                print(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Event) -> None:
        """Handle a single event by notifying subscribers."""
        event_type = event.event_type
        if event_type in self._subscriptions:
            # Sort subscriptions by priority
            subscriptions = sorted(
                self._subscriptions[event_type],
                key=lambda sub: sub.priority.value,
                reverse=True
            )
            
            for subscription in subscriptions:
                try:
                    # Apply filter if present
                    if subscription.filter_func and not subscription.filter_func(event):
                        continue
                    
                    # Call handler
                    await subscription.handler(event)
                except Exception as e:
                    self._stats['processing_errors'] += 1
                    # Log error (would use proper logger in real implementation)
                    print(f"Error in event handler: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return self._stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the event bus."""
        return {
            "status": "healthy" if self._running else "stopped",
            "running": self._running,
            "queue_size": self._event_queue.qsize(),
            "active_subscriptions": self._stats['active_subscriptions'],
            "stats": self.get_stats()
        }


# Event type constants for common system events
class SystemEvents:
    """Constants for common system events."""
    
    # Application lifecycle events
    APP_STARTED = "app.started"
    APP_STOPPING = "app.stopping"
    APP_STOPPED = "app.stopped"
    
    # Agent events
    AGENT_INITIALIZED = "agent.initialized"
    AGENT_REQUEST_RECEIVED = "agent.request_received"
    AGENT_RESPONSE_SENT = "agent.response_sent"
    AGENT_ERROR = "agent.error"
    
    # LLM events
    LLM_REQUEST_STARTED = "llm.request_started"
    LLM_REQUEST_COMPLETED = "llm.request_completed"
    LLM_REQUEST_FAILED = "llm.request_failed"
    
    # Storage events
    STORAGE_READ = "storage.read"
    STORAGE_WRITE = "storage.write"
    STORAGE_DELETE = "storage.delete"
    STORAGE_ERROR = "storage.error"
    
    # Security events
    SECURITY_AUTH_SUCCESS = "security.auth_success"
    SECURITY_AUTH_FAILURE = "security.auth_failure"
    SECURITY_SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    
    # Performance events
    PERFORMANCE_SLOW_OPERATION = "performance.slow_operation"
    PERFORMANCE_HIGH_MEMORY = "performance.high_memory"
    PERFORMANCE_HIGH_CPU = "performance.high_cpu"


# Utility functions for creating common events
def create_agent_event(
    event_type: str,
    agent_id: str,
    **payload
) -> Event:
    """Create an agent-related event."""
    return Event(
        event_type=event_type,
        payload={
            'agent_id': agent_id,
            **payload
        },
        source=f'agent.{agent_id}'
    )


def create_llm_event(
    event_type: str,
    provider: str,
    **payload
) -> Event:
    """Create an LLM-related event."""
    return Event(
        event_type=event_type,
        payload={
            'provider': provider,
            **payload
        },
        source=f'llm.{provider}'
    )


def create_performance_event(
    operation: str,
    duration_ms: float,
    **payload
) -> Event:
    """Create a performance-related event."""
    priority = EventPriority.HIGH if duration_ms > 5000 else EventPriority.NORMAL
    return Event(
        event_type=SystemEvents.PERFORMANCE_SLOW_OPERATION,
        payload={
            'operation': operation,
            'duration_ms': duration_ms,
            **payload
        },
        priority=priority,
        source='performance_monitor'
    )