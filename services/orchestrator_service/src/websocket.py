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

    def __init__(self, settings: WebSocketSettings | None = None, jwt_manager: Any = None) -> None:
        self._settings = settings or get_config().websocket()
        self._connections: dict[UUID, WebSocketConnection] = {}
        self._user_connections: dict[UUID, set[UUID]] = {}
        self._event_bus = get_event_bus()
        self._jwt_manager = jwt_manager
        self._heartbeat_task: asyncio.Task[None] | None = None

    async def authenticate_token(self, token: str | None) -> dict[str, Any] | None:
        """Validate JWT token. Returns decoded payload or None on failure."""
        if not token:
            return None
        if self._jwt_manager is None:
            logger.warning("websocket_jwt_manager_not_configured")
            return None
        try:
            result = self._jwt_manager.decode_token(token)
            if result.success and result.payload:
                return {"sub": result.payload.sub, "user_id": result.payload.sub}
            logger.warning("websocket_auth_failed", error=result.error_message)
            return None
        except Exception as e:
            logger.warning("websocket_auth_error", error=str(e))
            return None

    async def connect(
        self,
        websocket: WebSocket,
        user_id: UUID,
        session_id: UUID,
        token: str | None = None,
    ) -> WebSocketConnection:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance.
            user_id: User identifier.
            session_id: Session identifier.
            token: JWT token for authentication (from query param).
                   When jwt_manager is configured, token is required.
        """
        # Authenticate if jwt_manager is configured
        if self._jwt_manager is not None:
            auth_payload = await self.authenticate_token(token)
            if auth_payload is None:
                await websocket.close(code=1008, reason="Authentication required")
                raise ConnectionError("WebSocket authentication failed")

        user_conns = self._user_connections.get(user_id, set())
        if len(user_conns) >= self._settings.max_connections_per_user:
            await websocket.close(code=4003, reason="Too many connections")
            raise ConnectionError(f"User {user_id} has too many connections")
        await websocket.accept()
        connection_id = uuid4()
        now = datetime.now(timezone.utc)
        initial_state = ConnectionState.AUTHENTICATED if self._jwt_manager else ConnectionState.CONNECTED
        connection = WebSocketConnection(
            connection_id=connection_id, websocket=websocket, user_id=user_id, session_id=session_id,
            state=initial_state, connected_at=now, last_activity=now,
        )
        self._connections[connection_id] = connection
        if user_id not in self._user_connections:
            self._user_connections[user_id] = set()
        self._user_connections[user_id].add(connection_id)
        await self._event_bus.publish(EventFactory.session_started(session_id, user_id))
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
        except Exception as e:
            logger.warning("websocket_close_failed", connection_id=str(connection_id), error=str(e))
        self._connections.pop(connection_id, None)
        user_conns = self._user_connections.get(connection.user_id)
        if user_conns:
            user_conns.discard(connection_id)
            if not user_conns:
                self._user_connections.pop(connection.user_id, None)
        await self._event_bus.publish(EventFactory.session_ended(connection.session_id, connection.user_id, reason))
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

    async def start_heartbeat(self) -> None:
        """Start background heartbeat loop that pings connections every interval."""
        if self._heartbeat_task is not None:
            return
        interval = self._settings.heartbeat_interval_seconds

        async def _heartbeat_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self._ping_all()
                    await self.cleanup_stale()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("heartbeat_error", error=str(e))

        self._heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info("websocket_heartbeat_started", interval_seconds=interval)

    async def stop_heartbeat(self) -> None:
        """Stop the heartbeat loop."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _ping_all(self) -> None:
        """Send ping to all active connections."""
        pong_msg = WebSocketMessage.pong()
        for conn_id in list(self._connections.keys()):
            conn = self._connections.get(conn_id)
            if conn and conn.state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                try:
                    await conn.websocket.send_json(
                        {"type": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}
                    )
                except Exception:
                    logger.debug("ping_failed_marking_stale", connection_id=str(conn_id))

    async def cleanup_stale(self) -> int:
        """Disconnect connections that have been idle beyond the timeout."""
        now = datetime.now(timezone.utc)
        timeout = self._settings.connection_timeout_seconds
        stale_ids: list[UUID] = []
        for conn_id, conn in self._connections.items():
            idle_seconds = (now - conn.last_activity).total_seconds()
            if idle_seconds > timeout:
                stale_ids.append(conn_id)
        for conn_id in stale_ids:
            logger.info("disconnecting_stale_connection", connection_id=str(conn_id))
            await self.disconnect(conn_id, reason="idle_timeout")
        if stale_ids:
            logger.info("stale_connections_cleaned", count=len(stale_ids))
        return len(stale_ids)

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
