"""
Solace-AI Orchestrator Service - WebSocket Management.
WebSocket connection lifecycle, message routing, and session management.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
import asyncio
import structlog
from fastapi import WebSocket, WebSocketDisconnect

from .config import WebSocketSettings, get_config
from .events import EventFactory, get_event_bus

logger = structlog.get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


class MessageType(str, Enum):
    """WebSocket message types."""
    CHAT = "chat"
    PING = "ping"
    PONG = "pong"
    SYSTEM = "system"
    ERROR = "error"
    TYPING = "typing"
    STREAM_START = "stream_start"
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"


@dataclass
class WebSocketConnection:
    """Represents an active WebSocket connection."""
    connection_id: UUID
    websocket: WebSocket
    user_id: UUID
    session_id: UUID
    state: ConnectionState
    connected_at: datetime
    last_activity: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connection_id": str(self.connection_id),
            "user_id": str(self.user_id),
            "session_id": str(self.session_id),
            "state": self.state.value,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    message_id: UUID
    message_type: MessageType
    payload: dict[str, Any]
    timestamp: datetime
    connection_id: UUID | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "message_id": str(self.message_id),
            "type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], connection_id: UUID | None = None) -> WebSocketMessage:
        """Create from dictionary."""
        return cls(
            message_id=UUID(data.get("message_id", str(uuid4()))),
            message_type=MessageType(data.get("type", "chat")),
            payload=data.get("payload", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
            connection_id=connection_id,
        )

    @classmethod
    def chat(cls, content: str, connection_id: UUID | None = None) -> WebSocketMessage:
        """Create chat message."""
        return cls(uuid4(), MessageType.CHAT, {"content": content}, datetime.now(timezone.utc), connection_id)

    @classmethod
    def system(cls, message: str) -> WebSocketMessage:
        """Create system message."""
        return cls(uuid4(), MessageType.SYSTEM, {"message": message}, datetime.now(timezone.utc))

    @classmethod
    def error(cls, error: str, code: str = "UNKNOWN") -> WebSocketMessage:
        """Create error message."""
        return cls(uuid4(), MessageType.ERROR, {"error": error, "code": code}, datetime.now(timezone.utc))

    @classmethod
    def pong(cls) -> WebSocketMessage:
        """Create pong response."""
        return cls(uuid4(), MessageType.PONG, {}, datetime.now(timezone.utc))


class ConnectionManager:
    """Manages WebSocket connections for the orchestrator."""

    def __init__(self, settings: WebSocketSettings | None = None) -> None:
        self._settings = settings or get_config().websocket()
        self._connections: dict[UUID, WebSocketConnection] = {}
        self._user_connections: dict[UUID, set[UUID]] = {}
        self._event_bus = get_event_bus()

    async def connect(self, websocket: WebSocket, user_id: UUID, session_id: UUID) -> WebSocketConnection:
        """Accept and register a new WebSocket connection."""
        user_conns = self._user_connections.get(user_id, set())
        if len(user_conns) >= self._settings.max_connections_per_user:
            await websocket.close(code=4003, reason="Too many connections")
            raise ConnectionError(f"User {user_id} has too many connections")
        await websocket.accept()
        connection_id = uuid4()
        now = datetime.now(timezone.utc)
        connection = WebSocketConnection(
            connection_id=connection_id, websocket=websocket, user_id=user_id, session_id=session_id,
            state=ConnectionState.CONNECTED, connected_at=now, last_activity=now,
        )
        self._connections[connection_id] = connection
        if user_id not in self._user_connections:
            self._user_connections[user_id] = set()
        self._user_connections[user_id].add(connection_id)
        self._event_bus.publish(EventFactory.session_started(session_id, user_id))
        logger.info("websocket_connected", connection_id=str(connection_id), user_id=str(user_id))
        return connection

    async def disconnect(self, connection_id: UUID, reason: str = "normal") -> None:
        """Disconnect and cleanup a WebSocket connection."""
        connection = self._connections.get(connection_id)
        if not connection:
            return
        connection.state = ConnectionState.DISCONNECTING
        try:
            await connection.websocket.close(code=1000, reason=reason)
        except Exception:
            pass
        self._connections.pop(connection_id, None)
        user_conns = self._user_connections.get(connection.user_id)
        if user_conns:
            user_conns.discard(connection_id)
            if not user_conns:
                self._user_connections.pop(connection.user_id, None)
        self._event_bus.publish(EventFactory.session_ended(connection.session_id, connection.user_id, reason))
        logger.info("websocket_disconnected", connection_id=str(connection_id), reason=reason)

    async def send(self, connection_id: UUID, message: WebSocketMessage) -> bool:
        """Send message to specific connection."""
        connection = self._connections.get(connection_id)
        if not connection or connection.state not in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
            return False
        try:
            await connection.websocket.send_json(message.to_dict())
            connection.last_activity = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error("websocket_send_error", connection_id=str(connection_id), error=str(e))
            return False

    async def broadcast_to_user(self, user_id: UUID, message: WebSocketMessage) -> int:
        """Broadcast message to all user's connections."""
        user_conns = self._user_connections.get(user_id, set())
        sent_count = 0
        for conn_id in list(user_conns):
            if await self.send(conn_id, message):
                sent_count += 1
        return sent_count

    async def receive(self, connection_id: UUID) -> WebSocketMessage | None:
        """Receive message from connection."""
        connection = self._connections.get(connection_id)
        if not connection:
            return None
        try:
            data = await asyncio.wait_for(
                connection.websocket.receive_json(),
                timeout=self._settings.connection_timeout_seconds,
            )
            connection.last_activity = datetime.now(timezone.utc)
            return WebSocketMessage.from_dict(data, connection_id)
        except asyncio.TimeoutError:
            logger.warning("websocket_timeout", connection_id=str(connection_id))
            return None
        except WebSocketDisconnect:
            return None

    def get_connection(self, connection_id: UUID) -> WebSocketConnection | None:
        """Get connection by ID."""
        return self._connections.get(connection_id)

    def get_user_connections(self, user_id: UUID) -> list[WebSocketConnection]:
        """Get all connections for a user."""
        conn_ids = self._user_connections.get(user_id, set())
        return [self._connections[cid] for cid in conn_ids if cid in self._connections]

    def get_connection_count(self) -> int:
        """Get total connection count."""
        return len(self._connections)

    def get_statistics(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self._connections),
            "unique_users": len(self._user_connections),
            "connections_by_state": {
                state.value: sum(1 for c in self._connections.values() if c.state == state)
                for state in ConnectionState
            },
        }


_connection_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Get singleton connection manager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager
