"""
Solace-AI Memory Service - Episodic Memory (Tier 4).
Manages past session summaries, events, and therapeutic timeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class EpisodicMemorySettings(BaseSettings):
    """Configuration for episodic memory behavior."""
    max_summaries_per_user: int = Field(default=100, description="Max session summaries")
    max_events_per_user: int = Field(default=500, description="Max events per user")
    summary_retention_days: int = Field(default=365, description="Summary retention period")
    event_retention_days: int = Field(default=180, description="Event retention period")
    enable_bi_temporal: bool = Field(default=True, description="Enable bi-temporal tracking")
    model_config = SettingsConfigDict(env_prefix="EPISODIC_MEMORY_", env_file=".env", extra="ignore")


class EventType(str, Enum):
    """Types of therapeutic events."""
    SESSION = "session"
    ASSESSMENT = "assessment"
    TREATMENT = "treatment"
    CRISIS = "crisis"
    MILESTONE = "milestone"
    HOMEWORK = "homework"
    PROGRESS = "progress"


class EventSeverity(str, Enum):
    """Severity/importance of events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SessionSummary:
    """Summary of a completed session (Tier 4 primary storage)."""
    summary_id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    summary_text: str = ""
    key_topics: list[str] = field(default_factory=list)
    emotional_arc: list[str] = field(default_factory=list)
    techniques_used: list[str] = field(default_factory=list)
    key_insights: list[str] = field(default_factory=list)
    homework_assigned: list[str] = field(default_factory=list)
    message_count: int = 0
    duration_minutes: int = 0
    session_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_strength: Decimal = Decimal("1.0")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TherapeuticEvent:
    """An event in the therapeutic timeline."""
    event_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    event_type: EventType = EventType.SESSION
    severity: EventSeverity = EventSeverity.MEDIUM
    title: str = ""
    description: str = ""
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    related_events: list[UUID] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    retention_strength: Decimal = Decimal("1.0")


class TimelineQuery(BaseModel):
    """Query parameters for timeline retrieval."""
    user_id: UUID
    event_types: list[EventType] | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    severity_min: EventSeverity | None = None
    limit: int = Field(default=50, le=200)


class EpisodicMemoryManager:
    """Manages Tier 4 Episodic Memory - past sessions and events."""

    def __init__(self, settings: EpisodicMemorySettings | None = None) -> None:
        self._settings = settings or EpisodicMemorySettings()
        self._summaries: dict[UUID, list[SessionSummary]] = {}
        self._events: dict[UUID, list[TherapeuticEvent]] = {}
        self._stats = {"summaries_stored": 0, "events_stored": 0, "queries": 0, "cleanups": 0}

    def store_session_summary(self, user_id: UUID, session_id: UUID, session_number: int,
                              summary_text: str, key_topics: list[str], emotional_arc: list[str],
                              techniques_used: list[str], key_insights: list[str],
                              homework_assigned: list[str], message_count: int,
                              duration_minutes: int, session_date: datetime | None = None,
                              metadata: dict[str, Any] | None = None) -> SessionSummary:
        """Store a session summary."""
        self._stats["summaries_stored"] += 1
        summary = SessionSummary(
            session_id=session_id, user_id=user_id, session_number=session_number,
            summary_text=summary_text, key_topics=key_topics, emotional_arc=emotional_arc,
            techniques_used=techniques_used, key_insights=key_insights,
            homework_assigned=homework_assigned, message_count=message_count,
            duration_minutes=duration_minutes, session_date=session_date or datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        summaries = self._summaries.setdefault(user_id, [])
        summaries.append(summary)
        self._enforce_summary_limit(user_id)
        logger.info("session_summary_stored", user_id=str(user_id), session_id=str(session_id),
                    session_number=session_number)
        return summary

    def get_session_summary(self, session_id: UUID) -> SessionSummary | None:
        """Get summary for a specific session."""
        for summaries in self._summaries.values():
            for summary in summaries:
                if summary.session_id == session_id:
                    return summary
        return None

    def get_recent_summaries(self, user_id: UUID, limit: int = 5) -> list[SessionSummary]:
        """Get most recent session summaries for user."""
        self._stats["queries"] += 1
        summaries = self._summaries.get(user_id, [])
        return sorted(summaries, key=lambda s: s.session_date, reverse=True)[:limit]

    def get_summaries_by_date_range(self, user_id: UUID, start_date: datetime,
                                    end_date: datetime) -> list[SessionSummary]:
        """Get summaries within date range."""
        self._stats["queries"] += 1
        summaries = self._summaries.get(user_id, [])
        return [s for s in summaries if start_date <= s.session_date <= end_date]

    def get_summaries_by_topic(self, user_id: UUID, topic: str, limit: int = 10) -> list[SessionSummary]:
        """Get summaries that discussed a specific topic."""
        self._stats["queries"] += 1
        summaries = self._summaries.get(user_id, [])
        topic_lower = topic.lower()
        matching = [s for s in summaries if any(topic_lower in t.lower() for t in s.key_topics)]
        return sorted(matching, key=lambda s: s.session_date, reverse=True)[:limit]

    def store_event(self, user_id: UUID, event_type: EventType, title: str, description: str,
                    severity: EventSeverity = EventSeverity.MEDIUM,
                    session_id: UUID | None = None, occurred_at: datetime | None = None,
                    related_events: list[UUID] | None = None,
                    payload: dict[str, Any] | None = None) -> TherapeuticEvent:
        """Store a therapeutic event."""
        self._stats["events_stored"] += 1
        now = datetime.now(timezone.utc)
        event = TherapeuticEvent(
            user_id=user_id, session_id=session_id, event_type=event_type,
            severity=severity, title=title, description=description,
            occurred_at=occurred_at or now, ingested_at=now,
            valid_from=occurred_at or now if self._settings.enable_bi_temporal else None,
            related_events=related_events or [], payload=payload or {},
        )
        events = self._events.setdefault(user_id, [])
        events.append(event)
        self._enforce_event_limit(user_id)
        logger.info("therapeutic_event_stored", user_id=str(user_id), event_type=event_type.value,
                    severity=severity.value)
        return event

    def get_event(self, event_id: UUID) -> TherapeuticEvent | None:
        """Get a specific event by ID."""
        for events in self._events.values():
            for event in events:
                if event.event_id == event_id:
                    return event
        return None

    def get_timeline(self, query: TimelineQuery) -> list[TherapeuticEvent]:
        """Get events matching query criteria."""
        self._stats["queries"] += 1
        events = self._events.get(query.user_id, [])
        if query.event_types:
            events = [e for e in events if e.event_type in query.event_types]
        if query.start_date:
            events = [e for e in events if e.occurred_at >= query.start_date]
        if query.end_date:
            events = [e for e in events if e.occurred_at <= query.end_date]
        if query.severity_min:
            severity_order = {EventSeverity.LOW: 0, EventSeverity.MEDIUM: 1,
                             EventSeverity.HIGH: 2, EventSeverity.CRITICAL: 3}
            min_level = severity_order.get(query.severity_min, 0)
            events = [e for e in events if severity_order.get(e.severity, 0) >= min_level]
        events = sorted(events, key=lambda e: e.occurred_at, reverse=True)
        return events[:query.limit]

    def get_crisis_history(self, user_id: UUID) -> list[TherapeuticEvent]:
        """Get all crisis events for user (never decay)."""
        events = self._events.get(user_id, [])
        crisis_events = [e for e in events if e.event_type == EventType.CRISIS]
        return sorted(crisis_events, key=lambda e: e.occurred_at, reverse=True)

    def get_milestone_history(self, user_id: UUID) -> list[TherapeuticEvent]:
        """Get milestone events showing progress."""
        events = self._events.get(user_id, [])
        milestones = [e for e in events if e.event_type == EventType.MILESTONE]
        return sorted(milestones, key=lambda e: e.occurred_at, reverse=True)

    def get_treatment_history(self, user_id: UUID, limit: int = 20) -> list[TherapeuticEvent]:
        """Get treatment-related events."""
        events = self._events.get(user_id, [])
        treatment_types = [EventType.TREATMENT, EventType.HOMEWORK, EventType.ASSESSMENT]
        treatment_events = [e for e in events if e.event_type in treatment_types]
        return sorted(treatment_events, key=lambda e: e.occurred_at, reverse=True)[:limit]

    def link_events(self, event_id: UUID, related_event_id: UUID) -> bool:
        """Link two related events."""
        event = self.get_event(event_id)
        related = self.get_event(related_event_id)
        if event and related:
            if related_event_id not in event.related_events:
                event.related_events.append(related_event_id)
            if event_id not in related.related_events:
                related.related_events.append(event_id)
            return True
        return False

    def search_summaries(self, user_id: UUID, query: str, limit: int = 10) -> list[SessionSummary]:
        """Search summaries by text content."""
        self._stats["queries"] += 1
        summaries = self._summaries.get(user_id, [])
        query_lower = query.lower()
        matching = [s for s in summaries if query_lower in s.summary_text.lower() or
                    any(query_lower in topic.lower() for topic in s.key_topics)]
        return sorted(matching, key=lambda s: s.session_date, reverse=True)[:limit]

    def get_therapeutic_context(self, user_id: UUID) -> dict[str, Any]:
        """Get consolidated therapeutic context for LLM."""
        recent_summaries = self.get_recent_summaries(user_id, limit=3)
        crisis_history = self.get_crisis_history(user_id)
        milestones = self.get_milestone_history(user_id)[:5]
        all_topics: dict[str, int] = {}
        all_techniques: dict[str, int] = {}
        for summary in self._summaries.get(user_id, []):
            for topic in summary.key_topics:
                all_topics[topic] = all_topics.get(topic, 0) + 1
            for technique in summary.techniques_used:
                all_techniques[technique] = all_techniques.get(technique, 0) + 1
        return {
            "recent_sessions": [{"number": s.session_number, "date": s.session_date.isoformat(),
                                 "topics": s.key_topics, "insights": s.key_insights}
                                for s in recent_summaries],
            "total_sessions": len(self._summaries.get(user_id, [])),
            "crisis_count": len(crisis_history),
            "has_recent_crisis": any((datetime.now(timezone.utc) - e.occurred_at).days < 30
                                     for e in crisis_history),
            "milestones": [{"title": m.title, "date": m.occurred_at.isoformat()} for m in milestones],
            "common_topics": sorted(all_topics.keys(), key=lambda t: all_topics[t], reverse=True)[:5],
            "effective_techniques": sorted(all_techniques.keys(),
                                           key=lambda t: all_techniques[t], reverse=True)[:5],
        }

    def apply_decay(self, user_id: UUID, decay_rates: dict[str, Decimal]) -> tuple[int, int]:
        """Apply decay to summaries and events."""
        decayed = 0
        archived = 0
        base_rate = decay_rates.get("episodic", Decimal("0.02"))
        for summary in self._summaries.get(user_id, []):
            age_days = (datetime.now(timezone.utc) - summary.session_date).days
            decay = base_rate * Decimal(str(age_days)) / Decimal("30")
            summary.retention_strength = max(Decimal("0.1"), summary.retention_strength - decay)
            decayed += 1
            if summary.retention_strength < Decimal("0.3"):
                archived += 1
        for event in self._events.get(user_id, []):
            if event.event_type == EventType.CRISIS:
                continue
            age_days = (datetime.now(timezone.utc) - event.occurred_at).days
            decay = base_rate * Decimal(str(age_days)) / Decimal("30")
            event.retention_strength = max(Decimal("0.1"), event.retention_strength - decay)
            decayed += 1
            if event.retention_strength < Decimal("0.3"):
                archived += 1
        return decayed, archived

    def cleanup_expired(self) -> tuple[int, int]:
        """Remove expired summaries and events."""
        self._stats["cleanups"] += 1
        removed_summaries = 0
        removed_events = 0
        summary_cutoff = datetime.now(timezone.utc) - timedelta(days=self._settings.summary_retention_days)
        event_cutoff = datetime.now(timezone.utc) - timedelta(days=self._settings.event_retention_days)
        for user_id in list(self._summaries.keys()):
            original = len(self._summaries[user_id])
            self._summaries[user_id] = [s for s in self._summaries[user_id]
                                        if s.session_date >= summary_cutoff]
            removed_summaries += original - len(self._summaries[user_id])
        for user_id in list(self._events.keys()):
            original = len(self._events[user_id])
            self._events[user_id] = [e for e in self._events[user_id]
                                     if e.event_type == EventType.CRISIS or e.occurred_at >= event_cutoff]
            removed_events += original - len(self._events[user_id])
        if removed_summaries or removed_events:
            logger.info("episodic_cleanup", removed_summaries=removed_summaries, removed_events=removed_events)
        return removed_summaries, removed_events

    def get_statistics(self) -> dict[str, Any]:
        """Get episodic memory statistics."""
        total_summaries = sum(len(s) for s in self._summaries.values())
        total_events = sum(len(e) for e in self._events.values())
        return {**self._stats, "total_summaries": total_summaries, "total_events": total_events,
                "users_with_summaries": len(self._summaries), "users_with_events": len(self._events)}

    def _enforce_summary_limit(self, user_id: UUID) -> None:
        """Enforce max summaries per user."""
        summaries = self._summaries.get(user_id, [])
        if len(summaries) > self._settings.max_summaries_per_user:
            summaries.sort(key=lambda s: (s.retention_strength, s.session_date))
            self._summaries[user_id] = summaries[-self._settings.max_summaries_per_user:]

    def _enforce_event_limit(self, user_id: UUID) -> None:
        """Enforce max events per user."""
        events = self._events.get(user_id, [])
        if len(events) > self._settings.max_events_per_user:
            critical = [e for e in events if e.event_type == EventType.CRISIS]
            others = [e for e in events if e.event_type != EventType.CRISIS]
            others.sort(key=lambda e: (e.retention_strength, e.occurred_at))
            keep_others = self._settings.max_events_per_user - len(critical)
            self._events[user_id] = critical + others[-keep_others:] if keep_others > 0 else critical
