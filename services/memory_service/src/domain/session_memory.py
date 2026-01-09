"""
Solace-AI Memory Service - Session Memory (Tier 3).
Manages full current session transcript with persistence support.
"""
from __future__ import annotations
import time
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


class SessionMemorySettings(BaseSettings):
    """Configuration for session memory behavior."""
    ttl_hours: int = Field(default=24, description="Session memory TTL after end")
    max_messages_per_session: int = Field(default=500, description="Max messages per session")
    auto_archive_after_hours: int = Field(default=48, description="Auto-archive threshold")
    enable_emotion_tracking: bool = Field(default=True, description="Track emotions")
    enable_topic_detection: bool = Field(default=True, description="Detect topics")
    model_config = SettingsConfigDict(env_prefix="SESSION_MEMORY_", env_file=".env", extra="ignore")


class SessionStatus(str, Enum):
    """Status of a session."""
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"
    ARCHIVED = "archived"


@dataclass
class SessionMessage:
    """A message within a session."""
    message_id: UUID = field(default_factory=uuid4)
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    role: str = "user"
    content: str = ""
    emotion_detected: str | None = None
    importance: Decimal = Decimal("0.5")
    token_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionTranscript:
    """Full session transcript (Tier 3 storage)."""
    session_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_number: int = 1
    session_type: str = "therapeutic"
    status: SessionStatus = SessionStatus.ACTIVE
    messages: list[SessionMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    topics_detected: list[str] = field(default_factory=list)
    emotional_progression: list[str] = field(default_factory=list)
    total_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStatistics(BaseModel):
    """Statistics for a session."""
    session_id: UUID
    message_count: int = Field(default=0)
    user_message_count: int = Field(default=0)
    assistant_message_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    duration_minutes: int = Field(default=0)
    unique_emotions: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class SessionMemoryManager:
    """Manages Tier 3 Session Memory - full session transcripts."""

    def __init__(self, settings: SessionMemorySettings | None = None) -> None:
        self._settings = settings or SessionMemorySettings()
        self._sessions: dict[UUID, SessionTranscript] = {}
        self._user_active_sessions: dict[UUID, UUID] = {}
        self._user_session_history: dict[UUID, list[UUID]] = {}
        self._stats = {"sessions_created": 0, "messages_stored": 0, "sessions_ended": 0, "sessions_archived": 0}

    def create_session(self, user_id: UUID, session_type: str = "therapeutic",
                       metadata: dict[str, Any] | None = None) -> SessionTranscript:
        """Create a new session transcript."""
        self._stats["sessions_created"] += 1
        history = self._user_session_history.setdefault(user_id, [])
        session_number = len(history) + 1
        transcript = SessionTranscript(
            user_id=user_id, session_number=session_number,
            session_type=session_type, metadata=metadata or {},
        )
        self._sessions[transcript.session_id] = transcript
        self._user_active_sessions[user_id] = transcript.session_id
        history.append(transcript.session_id)
        logger.info("session_created", user_id=str(user_id), session_id=str(transcript.session_id),
                    session_number=session_number)
        return transcript

    def get_session(self, session_id: UUID) -> SessionTranscript | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_active_session(self, user_id: UUID) -> SessionTranscript | None:
        """Get the active session for a user."""
        session_id = self._user_active_sessions.get(user_id)
        if session_id:
            session = self._sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                return session
        return None

    def add_message(self, session_id: UUID, role: str, content: str,
                    emotion: str | None = None, importance: Decimal | None = None,
                    metadata: dict[str, Any] | None = None) -> SessionMessage | None:
        """Add a message to a session."""
        session = self._sessions.get(session_id)
        if not session or session.status not in (SessionStatus.ACTIVE, SessionStatus.PAUSED):
            logger.warning("add_message_failed", session_id=str(session_id), reason="session_not_active")
            return None
        if len(session.messages) >= self._settings.max_messages_per_session:
            logger.warning("session_message_limit", session_id=str(session_id))
            return None
        self._stats["messages_stored"] += 1
        token_count = len(content) // 4
        message = SessionMessage(
            session_id=session_id, user_id=session.user_id, role=role, content=content,
            emotion_detected=emotion, importance=importance or Decimal("0.5"),
            token_count=token_count, metadata=metadata or {},
        )
        session.messages.append(message)
        session.total_tokens += token_count
        session.last_activity_at = datetime.now(timezone.utc)
        if emotion and self._settings.enable_emotion_tracking:
            session.emotional_progression.append(emotion)
        if self._settings.enable_topic_detection and role == "user":
            self._update_topics(session, content)
        logger.debug("message_added", session_id=str(session_id), role=role,
                     message_count=len(session.messages))
        return message

    def get_messages(self, session_id: UUID, limit: int | None = None,
                     role_filter: str | None = None, since: datetime | None = None) -> list[SessionMessage]:
        """Get messages from a session with optional filters."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        messages = session.messages
        if role_filter:
            messages = [m for m in messages if m.role == role_filter]
        if since:
            messages = [m for m in messages if m.created_at >= since]
        if limit:
            messages = messages[-limit:]
        return messages

    def get_recent_context(self, session_id: UUID, token_budget: int) -> list[SessionMessage]:
        """Get recent messages within token budget."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        result = []
        current_tokens = 0
        for message in reversed(session.messages):
            if current_tokens + message.token_count <= token_budget:
                result.insert(0, message)
                current_tokens += message.token_count
            else:
                break
        return result

    def end_session(self, session_id: UUID) -> SessionTranscript | None:
        """End a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        self._stats["sessions_ended"] += 1
        session.status = SessionStatus.ENDED
        session.ended_at = datetime.now(timezone.utc)
        if session.user_id in self._user_active_sessions:
            if self._user_active_sessions[session.user_id] == session_id:
                del self._user_active_sessions[session.user_id]
        logger.info("session_ended", session_id=str(session_id),
                    message_count=len(session.messages),
                    duration_minutes=self._calculate_duration(session))
        return session

    def pause_session(self, session_id: UUID) -> bool:
        """Pause an active session."""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.ACTIVE:
            session.status = SessionStatus.PAUSED
            return True
        return False

    def resume_session(self, session_id: UUID) -> bool:
        """Resume a paused session."""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.PAUSED:
            session.status = SessionStatus.ACTIVE
            session.last_activity_at = datetime.now(timezone.utc)
            self._user_active_sessions[session.user_id] = session_id
            return True
        return False

    def archive_session(self, session_id: UUID) -> bool:
        """Archive an ended session."""
        session = self._sessions.get(session_id)
        if session and session.status == SessionStatus.ENDED:
            self._stats["sessions_archived"] += 1
            session.status = SessionStatus.ARCHIVED
            logger.info("session_archived", session_id=str(session_id))
            return True
        return False

    def get_session_statistics(self, session_id: UUID) -> SessionStatistics | None:
        """Get statistics for a session."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        user_msgs = [m for m in session.messages if m.role == "user"]
        assistant_msgs = [m for m in session.messages if m.role == "assistant"]
        emotions = list(set(m.emotion_detected for m in session.messages if m.emotion_detected))
        return SessionStatistics(
            session_id=session_id, message_count=len(session.messages),
            user_message_count=len(user_msgs), assistant_message_count=len(assistant_msgs),
            total_tokens=session.total_tokens, duration_minutes=self._calculate_duration(session),
            unique_emotions=emotions, topics=session.topics_detected,
        )

    def get_user_session_history(self, user_id: UUID, limit: int = 10,
                                 include_active: bool = False) -> list[SessionTranscript]:
        """Get user's session history."""
        session_ids = self._user_session_history.get(user_id, [])
        sessions = []
        for sid in reversed(session_ids):
            session = self._sessions.get(sid)
            if session:
                if include_active or session.status != SessionStatus.ACTIVE:
                    sessions.append(session)
            if len(sessions) >= limit:
                break
        return sessions

    def cleanup_expired(self) -> tuple[int, int]:
        """Cleanup expired sessions based on TTL."""
        archived = 0
        deleted = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._settings.ttl_hours)
        archive_cutoff = datetime.now(timezone.utc) - timedelta(hours=self._settings.auto_archive_after_hours)
        for session_id, session in list(self._sessions.items()):
            if session.status == SessionStatus.ENDED:
                if session.ended_at and session.ended_at < archive_cutoff:
                    session.status = SessionStatus.ARCHIVED
                    archived += 1
            if session.status == SessionStatus.ARCHIVED:
                if session.ended_at and session.ended_at < cutoff:
                    del self._sessions[session_id]
                    deleted += 1
        if archived or deleted:
            logger.info("session_cleanup", archived=archived, deleted=deleted)
        return archived, deleted

    def search_messages(self, user_id: UUID, query: str, limit: int = 20) -> list[SessionMessage]:
        """Search messages across user's sessions."""
        results = []
        query_lower = query.lower()
        session_ids = self._user_session_history.get(user_id, [])
        for sid in reversed(session_ids):
            session = self._sessions.get(sid)
            if session:
                for message in session.messages:
                    if query_lower in message.content.lower():
                        results.append(message)
                        if len(results) >= limit:
                            return results
        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get overall session memory statistics."""
        active = sum(1 for s in self._sessions.values() if s.status == SessionStatus.ACTIVE)
        ended = sum(1 for s in self._sessions.values() if s.status == SessionStatus.ENDED)
        archived = sum(1 for s in self._sessions.values() if s.status == SessionStatus.ARCHIVED)
        return {**self._stats, "active_sessions": active, "ended_sessions": ended,
                "archived_sessions": archived, "total_sessions": len(self._sessions)}

    def _calculate_duration(self, session: SessionTranscript) -> int:
        """Calculate session duration in minutes."""
        end = session.ended_at or datetime.now(timezone.utc)
        return int((end - session.started_at).total_seconds() / 60)

    def _update_topics(self, session: SessionTranscript, content: str) -> None:
        """Update detected topics from message content."""
        topic_keywords = {
            "anxiety": ["anxious", "worry", "nervous", "panic"],
            "depression": ["sad", "hopeless", "depressed", "empty"],
            "relationships": ["relationship", "partner", "family", "friend"],
            "work": ["work", "job", "boss", "career"],
            "sleep": ["sleep", "insomnia", "tired"],
            "stress": ["stress", "overwhelmed", "pressure"],
        }
        content_lower = content.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in content_lower for kw in keywords):
                if topic not in session.topics_detected:
                    session.topics_detected.append(topic)
