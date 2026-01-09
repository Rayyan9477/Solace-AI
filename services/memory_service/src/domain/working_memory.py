"""
Solace-AI Memory Service - Working Memory (Tier 1-2).
Manages input buffer (current message) and working memory (context window for LLM).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class WorkingMemorySettings(BaseSettings):
    """Configuration for working memory behavior."""
    max_tokens: int = Field(default=8000, description="Maximum tokens in working memory")
    max_messages: int = Field(default=50, description="Maximum messages to retain")
    input_buffer_ttl_seconds: int = Field(default=30, description="Input buffer TTL")
    summarize_threshold: float = Field(default=0.8, description="Summarize at 80% capacity")
    recent_verbatim_count: int = Field(default=10, description="Recent messages kept verbatim")
    priority_boost_safety: float = Field(default=2.0, description="Safety message priority boost")
    priority_boost_therapeutic: float = Field(default=1.5, description="Therapeutic boost")
    token_estimate_chars: int = Field(default=4, description="Chars per token estimate")
    model_config = SettingsConfigDict(env_prefix="WORKING_MEMORY_", env_file=".env", extra="ignore")


@dataclass
class InputBufferItem:
    """Tier 1: Current message being processed."""
    item_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    content: str = ""
    role: str = "user"
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_started: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkingMemoryItem:
    """Tier 2: Item in working memory context window."""
    item_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    content: str = ""
    role: str = "user"
    token_count: int = 0
    priority_score: float = 1.0
    importance: Decimal = Decimal("0.5")
    is_summarized: bool = False
    original_count: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextWindowState(BaseModel):
    """Current state of the working memory context window."""
    user_id: UUID
    session_id: UUID | None = None
    total_tokens: int = Field(default=0)
    message_count: int = Field(default=0)
    capacity_used: float = Field(default=0.0)
    needs_summarization: bool = Field(default=False)
    oldest_message_age_seconds: int = Field(default=0)


class WorkingMemoryManager:
    """Manages Tier 1 (Input Buffer) and Tier 2 (Working Memory)."""

    def __init__(self, settings: WorkingMemorySettings | None = None) -> None:
        self._settings = settings or WorkingMemorySettings()
        self._input_buffers: dict[UUID, InputBufferItem] = {}
        self._working_memory: dict[UUID, list[WorkingMemoryItem]] = {}
        self._user_sessions: dict[UUID, UUID] = {}
        self._stats = {"inputs_processed": 0, "items_added": 0, "summarizations": 0, "evictions": 0}

    def set_input(self, user_id: UUID, session_id: UUID | None, content: str,
                  role: str, metadata: dict[str, Any] | None = None) -> InputBufferItem:
        """Set the current input buffer (Tier 1) for processing."""
        self._stats["inputs_processed"] += 1
        item = InputBufferItem(
            user_id=user_id, session_id=session_id, content=content,
            role=role, metadata=metadata or {},
        )
        self._input_buffers[user_id] = item
        if session_id:
            self._user_sessions[user_id] = session_id
        logger.debug("input_buffer_set", user_id=str(user_id), role=role, content_len=len(content))
        return item

    def get_input(self, user_id: UUID) -> InputBufferItem | None:
        """Get the current input buffer for user."""
        return self._input_buffers.get(user_id)

    def clear_input(self, user_id: UUID) -> bool:
        """Clear the input buffer after processing."""
        if user_id in self._input_buffers:
            del self._input_buffers[user_id]
            return True
        return False

    def start_processing(self, user_id: UUID) -> InputBufferItem | None:
        """Mark input as being processed."""
        item = self._input_buffers.get(user_id)
        if item:
            item.processing_started = datetime.now(timezone.utc)
        return item

    def add_to_working_memory(self, user_id: UUID, session_id: UUID | None, content: str,
                              role: str, importance: Decimal | None = None,
                              metadata: dict[str, Any] | None = None) -> WorkingMemoryItem:
        """Add item to working memory (Tier 2)."""
        self._stats["items_added"] += 1
        token_count = self._estimate_tokens(content)
        priority = self._calculate_priority(content, role, importance)
        item = WorkingMemoryItem(
            user_id=user_id, session_id=session_id, content=content, role=role,
            token_count=token_count, priority_score=priority,
            importance=importance or Decimal("0.5"), metadata=metadata or {},
        )
        memory = self._working_memory.setdefault(user_id, [])
        memory.append(item)
        self._enforce_limits(user_id)
        logger.debug("working_memory_added", user_id=str(user_id), tokens=token_count,
                     total_items=len(memory))
        return item

    def get_working_memory(self, user_id: UUID, max_tokens: int | None = None,
                           include_metadata: bool = False) -> list[WorkingMemoryItem]:
        """Get working memory items within token budget."""
        items = self._working_memory.get(user_id, [])
        if not max_tokens:
            return items
        result = []
        current_tokens = 0
        for item in reversed(items):
            if current_tokens + item.token_count <= max_tokens:
                result.insert(0, item)
                current_tokens += item.token_count
            else:
                break
        return result

    def get_context_window_state(self, user_id: UUID) -> ContextWindowState:
        """Get current state of user's context window."""
        items = self._working_memory.get(user_id, [])
        total_tokens = sum(item.token_count for item in items)
        capacity = total_tokens / self._settings.max_tokens if self._settings.max_tokens > 0 else 0
        oldest_age = 0
        if items:
            oldest = min(items, key=lambda x: x.created_at)
            oldest_age = int((datetime.now(timezone.utc) - oldest.created_at).total_seconds())
        return ContextWindowState(
            user_id=user_id, session_id=self._user_sessions.get(user_id),
            total_tokens=total_tokens, message_count=len(items),
            capacity_used=min(capacity, 1.0),
            needs_summarization=capacity >= self._settings.summarize_threshold,
            oldest_message_age_seconds=oldest_age,
        )

    def clear_working_memory(self, user_id: UUID) -> int:
        """Clear all working memory for user."""
        items = self._working_memory.pop(user_id, [])
        self._user_sessions.pop(user_id, None)
        logger.info("working_memory_cleared", user_id=str(user_id), items_cleared=len(items))
        return len(items)

    def summarize_old_messages(self, user_id: UUID) -> tuple[int, int]:
        """Summarize older messages to free up token budget."""
        items = self._working_memory.get(user_id, [])
        if len(items) <= self._settings.recent_verbatim_count:
            return 0, 0
        keep_verbatim = items[-self._settings.recent_verbatim_count:]
        to_summarize = items[:-self._settings.recent_verbatim_count]
        if not to_summarize:
            return 0, 0
        self._stats["summarizations"] += 1
        user_msgs = [m.content for m in to_summarize if m.role == "user"]
        assistant_msgs = [m.content for m in to_summarize if m.role == "assistant"]
        original_tokens = sum(m.token_count for m in to_summarize)
        summary_parts = []
        if user_msgs:
            summary_parts.append(f"User discussed: {'; '.join(user_msgs[:3])[:200]}")
        if assistant_msgs:
            summary_parts.append(f"Assistant covered: {assistant_msgs[0][:100]}")
        summary_content = " | ".join(summary_parts) if summary_parts else "Previous exchanges summarized."
        summary_item = WorkingMemoryItem(
            user_id=user_id, session_id=to_summarize[0].session_id if to_summarize else None,
            content=summary_content, role="system", token_count=self._estimate_tokens(summary_content),
            priority_score=0.5, is_summarized=True, original_count=len(to_summarize),
        )
        self._working_memory[user_id] = [summary_item] + keep_verbatim
        tokens_saved = original_tokens - summary_item.token_count
        logger.info("working_memory_summarized", user_id=str(user_id),
                    messages_summarized=len(to_summarize), tokens_saved=tokens_saved)
        return len(to_summarize), tokens_saved

    def get_for_llm_context(self, user_id: UUID, token_budget: int,
                            include_system: bool = True) -> tuple[str, dict[str, int]]:
        """Get formatted context for LLM within token budget."""
        items = self.get_working_memory(user_id, token_budget)
        parts = []
        breakdown = {"system": 0, "user": 0, "assistant": 0}
        for item in items:
            if not include_system and item.role == "system":
                continue
            prefix = {"user": "User", "assistant": "Assistant", "system": "Context"}.get(item.role, "Unknown")
            formatted = f"{prefix}: {item.content}"
            parts.append(formatted)
            breakdown[item.role] = breakdown.get(item.role, 0) + item.token_count
        return "\n\n".join(parts), breakdown

    def update_item_importance(self, user_id: UUID, item_id: UUID, new_importance: Decimal) -> bool:
        """Update importance score for a specific item."""
        items = self._working_memory.get(user_id, [])
        for item in items:
            if item.item_id == item_id:
                item.importance = new_importance
                item.priority_score = self._calculate_priority(item.content, item.role, new_importance)
                return True
        return False

    def get_statistics(self) -> dict[str, Any]:
        """Get working memory statistics."""
        total_items = sum(len(items) for items in self._working_memory.values())
        total_tokens = sum(sum(i.token_count for i in items) for items in self._working_memory.values())
        return {**self._stats, "active_users": len(self._working_memory),
                "total_items": total_items, "total_tokens": total_tokens}

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count from content."""
        return max(1, len(content) // self._settings.token_estimate_chars)

    def _calculate_priority(self, content: str, role: str, importance: Decimal | None) -> float:
        """Calculate priority score for ordering."""
        base = float(importance) if importance else 0.5
        if role == "user":
            base *= 1.2
        safety_keywords = ["crisis", "emergency", "suicide", "suicidal", "harm", "danger", "help"]
        if any(kw in content.lower() for kw in safety_keywords):
            base *= self._settings.priority_boost_safety
        therapeutic_keywords = ["therapy", "treatment", "medication", "diagnosis", "symptoms"]
        if any(kw in content.lower() for kw in therapeutic_keywords):
            base *= self._settings.priority_boost_therapeutic
        return min(base, 10.0)

    def _enforce_limits(self, user_id: UUID) -> None:
        """Enforce token and message limits."""
        items = self._working_memory.get(user_id, [])
        while len(items) > self._settings.max_messages:
            self._stats["evictions"] += 1
            non_priority = [i for i in items if i.priority_score < 2.0]
            if non_priority:
                items.remove(min(non_priority, key=lambda x: x.priority_score))
            else:
                items.pop(0)
        total_tokens = sum(i.token_count for i in items)
        while total_tokens > self._settings.max_tokens and items:
            non_priority = [i for i in items if i.priority_score < 2.0]
            if non_priority:
                removed = min(non_priority, key=lambda x: x.priority_score)
                items.remove(removed)
                total_tokens -= removed.token_count
                self._stats["evictions"] += 1
            else:
                break
        self._working_memory[user_id] = items
