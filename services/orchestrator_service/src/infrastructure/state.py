"""
Solace-AI Orchestrator Service - State Persistence.
LangGraph state checkpointing with multiple backend support.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID
import json
import structlog
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _POSTGRES_CHECKPOINT_AVAILABLE = True
except ImportError:
    _POSTGRES_CHECKPOINT_AVAILABLE = False

from ..config import PersistenceSettings, get_config
from ..langgraph.state_schema import OrchestratorState

logger = structlog.get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a state checkpoint."""
    checkpoint_id: str
    thread_id: str
    user_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    version: int = 1
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "version": self.version,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            thread_id=data["thread_id"],
            user_id=data["user_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            version=data.get("version", 1),
            size_bytes=data.get("size_bytes", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Checkpoint:
    """Complete checkpoint with state and metadata."""
    metadata: CheckpointMetadata
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"metadata": self.metadata.to_dict(), "state": self.state}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create from dictionary."""
        return cls(metadata=CheckpointMetadata.from_dict(data["metadata"]), state=data["state"])


class StateStore(ABC):
    """Abstract base for state storage backends."""

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint to store."""
        pass

    @abstractmethod
    async def load(self, thread_id: str) -> Checkpoint | None:
        """Load checkpoint from store."""
        pass

    @abstractmethod
    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from store."""
        pass

    @abstractmethod
    async def list_checkpoints(self, user_id: str | None = None) -> list[CheckpointMetadata]:
        """List checkpoints, optionally filtered by user."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired checkpoints."""
        pass


class MemoryStateStore(StateStore):
    """In-memory state store for development and testing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save(self, checkpoint: Checkpoint) -> bool:
        """Save checkpoint to memory."""
        self._checkpoints[checkpoint.metadata.thread_id] = checkpoint
        logger.debug("checkpoint_saved_memory", thread_id=checkpoint.metadata.thread_id)
        return True

    async def load(self, thread_id: str) -> Checkpoint | None:
        """Load checkpoint from memory."""
        checkpoint = self._checkpoints.get(thread_id)
        if checkpoint and checkpoint.metadata.expires_at < datetime.now(timezone.utc):
            del self._checkpoints[thread_id]
            return None
        return checkpoint

    async def delete(self, thread_id: str) -> bool:
        """Delete checkpoint from memory."""
        if thread_id in self._checkpoints:
            del self._checkpoints[thread_id]
            return True
        return False

    async def list_checkpoints(self, user_id: str | None = None) -> list[CheckpointMetadata]:
        """List all checkpoints in memory."""
        now = datetime.now(timezone.utc)
        checkpoints = [cp.metadata for cp in self._checkpoints.values() if cp.metadata.expires_at >= now]
        if user_id:
            checkpoints = [cp for cp in checkpoints if cp.user_id == user_id]
        return checkpoints

    async def cleanup_expired(self) -> int:
        """Remove expired checkpoints."""
        now = datetime.now(timezone.utc)
        expired = [tid for tid, cp in self._checkpoints.items() if cp.metadata.expires_at < now]
        for tid in expired:
            del self._checkpoints[tid]
        return len(expired)


class StatePersistenceManager:
    """Manages state persistence for LangGraph orchestration."""

    def __init__(self, settings: PersistenceSettings | None = None) -> None:
        self._settings = settings or get_config().persistence()
        self._store = self._create_store()
        self._langgraph_saver = self._create_langgraph_checkpointer()
        self._save_count = 0
        self._load_count = 0

    def _create_store(self) -> StateStore:
        """Create appropriate state store based on configuration."""
        backend = self._settings.checkpoint_backend
        if backend == "memory":
            return MemoryStateStore()
        if backend == "postgres":
            logger.info("using_memory_state_store_with_postgres_checkpointer")
            return MemoryStateStore()
        logger.warning("unknown_backend_using_memory", backend=backend)
        return MemoryStateStore()

    def _create_langgraph_checkpointer(self) -> Any:
        """Create LangGraph checkpointer based on configuration."""
        if not self._settings.enable_checkpointing:
            return None
        backend = self._settings.checkpoint_backend
        if backend == "postgres" and _POSTGRES_CHECKPOINT_AVAILABLE:
            postgres_url = self._settings.postgres_url
            logger.info("creating_postgres_langgraph_checkpointer", url=postgres_url.split("@")[-1] if "@" in postgres_url else "***")
            return AsyncPostgresSaver.from_conn_string(postgres_url)
        if backend == "postgres" and not _POSTGRES_CHECKPOINT_AVAILABLE:
            logger.warning("postgres_checkpointer_unavailable", msg="Install langgraph-checkpoint-postgres")
        return MemorySaver()

    def get_langgraph_checkpointer(self) -> Any:
        """Get LangGraph checkpointer for graph compilation."""
        return self._langgraph_saver

    async def save_state(self, state: OrchestratorState, thread_id: str | None = None) -> CheckpointMetadata:
        """Save orchestrator state as checkpoint."""
        self._save_count += 1
        tid = thread_id or state.get("thread_id", "")
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        now = datetime.now(timezone.utc)
        ttl_hours = self._settings.checkpoint_ttl_hours
        state_dict = dict(state)
        state_json = json.dumps(state_dict, default=str)
        size_bytes = len(state_json.encode("utf-8"))
        metadata = CheckpointMetadata(
            checkpoint_id=f"{tid}_{int(now.timestamp())}",
            thread_id=tid,
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            expires_at=now + timedelta(hours=ttl_hours),
            version=1,
            size_bytes=size_bytes,
        )
        checkpoint = Checkpoint(metadata=metadata, state=state_dict)
        await self._store.save(checkpoint)
        logger.info("state_saved", thread_id=tid, size_bytes=size_bytes)
        return metadata

    async def load_state(self, thread_id: str) -> OrchestratorState | None:
        """Load orchestrator state from checkpoint."""
        self._load_count += 1
        checkpoint = await self._store.load(thread_id)
        if checkpoint is None:
            logger.debug("state_not_found", thread_id=thread_id)
            return None
        logger.info("state_loaded", thread_id=thread_id)
        return OrchestratorState(**checkpoint.state)

    async def delete_state(self, thread_id: str) -> bool:
        """Delete checkpoint for thread."""
        success = await self._store.delete(thread_id)
        if success:
            logger.info("state_deleted", thread_id=thread_id)
        return success

    async def list_user_checkpoints(self, user_id: str) -> list[CheckpointMetadata]:
        """List all checkpoints for a user."""
        return await self._store.list_checkpoints(user_id)

    async def cleanup_expired(self) -> int:
        """Remove expired checkpoints."""
        count = await self._store.cleanup_expired()
        if count > 0:
            logger.info("expired_checkpoints_cleaned", count=count)
        return count

    def get_statistics(self) -> dict[str, Any]:
        """Get persistence statistics."""
        return {
            "save_count": self._save_count,
            "load_count": self._load_count,
            "backend": self._settings.checkpoint_backend,
            "checkpointing_enabled": self._settings.enable_checkpointing,
            "ttl_hours": self._settings.checkpoint_ttl_hours,
        }


_persistence_manager: StatePersistenceManager | None = None


def get_persistence_manager() -> StatePersistenceManager:
    """Get singleton persistence manager instance."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = StatePersistenceManager()
    return _persistence_manager
