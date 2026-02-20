"""
Tests for Orchestrator Service Infrastructure - Batch 8.4.
"""
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4


class TestConfigModule:
    """Tests for config.py module."""

    def test_environment_enum_values(self):
        from services.orchestrator_service.src.config import Environment
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.PRODUCTION.value == "production"
        assert Environment.STAGING.value == "staging"
        assert Environment.TEST.value == "test"

    def test_service_endpoints_defaults(self):
        from services.orchestrator_service.src.config import ServiceEndpoints
        endpoints = ServiceEndpoints()
        assert endpoints.personality_service_url == "http://localhost:8007"
        assert endpoints.diagnosis_service_url == "http://localhost:8004"
        assert endpoints.treatment_service_url == "http://localhost:8006"
        assert endpoints.memory_service_url == "http://localhost:8005"

    def test_llm_settings_defaults(self):
        from services.orchestrator_service.src.config import LLMSettings
        llm = LLMSettings()
        assert llm.provider == "anthropic"
        assert llm.max_tokens == 2048
        assert llm.temperature == 0.7
        assert llm.timeout_seconds == 60

    def test_safety_settings_defaults(self):
        from services.orchestrator_service.src.config import SafetySettings
        safety = SafetySettings()
        assert safety.enable_crisis_detection is True
        assert safety.crisis_confidence_threshold == 0.8
        assert safety.enable_content_filtering is True

    def test_websocket_settings_defaults(self):
        from services.orchestrator_service.src.config import WebSocketSettings
        ws = WebSocketSettings()
        assert ws.heartbeat_interval_seconds == 30
        assert ws.max_connections_per_user == 5
        assert ws.enable_compression is True

    def test_persistence_settings_defaults(self):
        from services.orchestrator_service.src.config import PersistenceSettings
        persistence = PersistenceSettings()
        assert persistence.enable_checkpointing is True
        assert persistence.checkpoint_backend == "memory"
        assert persistence.checkpoint_ttl_hours == 24

    def test_orchestrator_config_defaults(self):
        from services.orchestrator_service.src.config import OrchestratorConfig, Environment
        config = OrchestratorConfig()
        assert config.environment == Environment.DEVELOPMENT
        assert config.service_name == "orchestrator-service"
        assert config.port == 8000

    def test_orchestrator_config_cors_origins_list(self):
        from services.orchestrator_service.src.config import OrchestratorConfig
        config = OrchestratorConfig(cors_origins="*")
        assert config.cors_origins_list == ["*"]

    def test_orchestrator_config_is_production(self):
        from services.orchestrator_service.src.config import OrchestratorConfig, Environment
        config = OrchestratorConfig(environment=Environment.PRODUCTION)
        assert config.is_production is True
        config_dev = OrchestratorConfig(environment=Environment.DEVELOPMENT)
        assert config_dev.is_production is False

    def test_config_loader_caching(self):
        from services.orchestrator_service.src.config import ConfigLoader
        loader = ConfigLoader()
        config1 = loader.load()
        config2 = loader.load()
        assert config1 is config2

    def test_config_loader_to_dict(self):
        from services.orchestrator_service.src.config import ConfigLoader
        loader = ConfigLoader()
        config_dict = loader.to_dict()
        assert "main" in config_dict
        assert "endpoints" in config_dict
        assert "llm" in config_dict
        assert "safety" in config_dict

    def test_get_config_singleton(self):
        from services.orchestrator_service.src.config import get_config
        loader1 = get_config()
        loader2 = get_config()
        assert loader1 is loader2


class TestEventsModule:
    """Tests for events.py module."""

    def test_event_type_values(self):
        from services.orchestrator_service.src.events import EventType
        assert EventType.SESSION_STARTED.value == "session_started"
        assert EventType.CRISIS_DETECTED.value == "crisis_detected"
        assert EventType.AGENT_COMPLETED.value == "agent_completed"

    def test_orchestrator_event_creation(self):
        from services.orchestrator_service.src.events import OrchestratorEvent, EventType
        event = OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.SESSION_STARTED, timestamp=datetime.now(timezone.utc),
            session_id=uuid4(), user_id=uuid4(), payload={"action": "start"},
        )
        assert event.event_type == EventType.SESSION_STARTED
        assert event.payload == {"action": "start"}

    def test_orchestrator_event_to_dict(self):
        from services.orchestrator_service.src.events import OrchestratorEvent, EventType
        session_id = uuid4()
        user_id = uuid4()
        event = OrchestratorEvent(
            event_id=uuid4(), event_type=EventType.MESSAGE_RECEIVED, timestamp=datetime.now(timezone.utc),
            session_id=session_id, user_id=user_id, payload={"message_length": 100},
        )
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "message_received"
        assert event_dict["session_id"] == str(session_id)
        assert event_dict["payload"]["message_length"] == 100

    def test_orchestrator_event_from_dict(self):
        from services.orchestrator_service.src.events import OrchestratorEvent, EventType
        data = {
            "event_id": str(uuid4()),
            "event_type": "agent_started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": str(uuid4()),
            "user_id": str(uuid4()),
            "payload": {"agent_type": "safety"},
        }
        event = OrchestratorEvent.from_dict(data)
        assert event.event_type == EventType.AGENT_STARTED
        assert event.payload["agent_type"] == "safety"

    def test_event_factory_session_started(self):
        from services.orchestrator_service.src.events import EventFactory, EventType
        session_id = uuid4()
        user_id = uuid4()
        event = EventFactory.session_started(session_id, user_id)
        assert event.event_type == EventType.SESSION_STARTED
        assert event.session_id == session_id
        assert event.user_id == user_id

    def test_event_factory_crisis_detected(self):
        from services.orchestrator_service.src.events import EventFactory, EventType
        event = EventFactory.crisis_detected(uuid4(), uuid4(), "HIGH", "self_harm")
        assert event.event_type == EventType.CRISIS_DETECTED
        assert event.payload["risk_level"] == "HIGH"
        assert event.payload["crisis_type"] == "self_harm"

    def test_event_factory_agent_completed(self):
        from services.orchestrator_service.src.events import EventFactory, EventType
        event = EventFactory.agent_completed(uuid4(), uuid4(), "therapy", 0.85, 150.0)
        assert event.event_type == EventType.AGENT_COMPLETED
        assert event.payload["agent_type"] == "therapy"
        assert event.payload["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_event_bus_subscribe_and_publish(self):
        from services.orchestrator_service.src.events import EventBus, EventFactory, EventType
        bus = EventBus()
        received_events = []
        def handler(event):
            received_events.append(event)
        bus.subscribe(EventType.SESSION_STARTED, handler)
        event = EventFactory.session_started(uuid4(), uuid4())
        await bus.publish(event)
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.SESSION_STARTED

    @pytest.mark.asyncio
    async def test_event_bus_subscribe_all(self):
        from services.orchestrator_service.src.events import EventBus, EventFactory
        bus = EventBus()
        all_events = []
        bus.subscribe_all(lambda e: all_events.append(e))
        await bus.publish(EventFactory.session_started(uuid4(), uuid4()))
        await bus.publish(EventFactory.session_ended(uuid4(), uuid4()))
        assert len(all_events) == 2

    @pytest.mark.asyncio
    async def test_event_bus_get_history(self):
        from services.orchestrator_service.src.events import EventBus, EventFactory, EventType
        bus = EventBus()
        await bus.publish(EventFactory.session_started(uuid4(), uuid4()))
        await bus.publish(EventFactory.session_ended(uuid4(), uuid4()))
        history = bus.get_history()
        assert len(history) == 2
        filtered = bus.get_history(EventType.SESSION_STARTED)
        assert len(filtered) == 1

    def test_get_event_bus_singleton(self):
        from services.orchestrator_service.src.events import get_event_bus
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2


class TestWebSocketModule:
    """Tests for websocket.py module."""

    def test_connection_state_values(self):
        from services.orchestrator_service.src.websocket import ConnectionState
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.DISCONNECTED.value == "disconnected"

    def test_message_type_values(self):
        from services.orchestrator_service.src.websocket import MessageType
        assert MessageType.CHAT.value == "chat"
        assert MessageType.PING.value == "ping"
        assert MessageType.STREAM_CHUNK.value == "stream_chunk"

    def test_websocket_message_chat_factory(self):
        from services.orchestrator_service.src.websocket import WebSocketMessage, MessageType
        msg = WebSocketMessage.chat("Hello world")
        assert msg.message_type == MessageType.CHAT
        assert msg.payload["content"] == "Hello world"

    def test_websocket_message_system_factory(self):
        from services.orchestrator_service.src.websocket import WebSocketMessage, MessageType
        msg = WebSocketMessage.system("Connection established")
        assert msg.message_type == MessageType.SYSTEM
        assert msg.payload["message"] == "Connection established"

    def test_websocket_message_error_factory(self):
        from services.orchestrator_service.src.websocket import WebSocketMessage, MessageType
        msg = WebSocketMessage.error("Invalid request", "VALIDATION")
        assert msg.message_type == MessageType.ERROR
        assert msg.payload["error"] == "Invalid request"
        assert msg.payload["code"] == "VALIDATION"

    def test_websocket_message_to_dict(self):
        from services.orchestrator_service.src.websocket import WebSocketMessage
        msg = WebSocketMessage.chat("Test message")
        msg_dict = msg.to_dict()
        assert msg_dict["type"] == "chat"
        assert msg_dict["payload"]["content"] == "Test message"
        assert "message_id" in msg_dict
        assert "timestamp" in msg_dict

    def test_websocket_message_from_dict(self):
        from services.orchestrator_service.src.websocket import WebSocketMessage, MessageType
        data = {"type": "chat", "payload": {"content": "Hello"}, "timestamp": datetime.now(timezone.utc).isoformat()}
        msg = WebSocketMessage.from_dict(data)
        assert msg.message_type == MessageType.CHAT
        assert msg.payload["content"] == "Hello"

    def test_connection_manager_statistics(self):
        from services.orchestrator_service.src.websocket import ConnectionManager
        manager = ConnectionManager()
        stats = manager.get_statistics()
        assert stats["total_connections"] == 0
        assert stats["unique_users"] == 0

    def test_get_connection_manager_singleton(self):
        from services.orchestrator_service.src.websocket import get_connection_manager
        mgr1 = get_connection_manager()
        mgr2 = get_connection_manager()
        assert mgr1 is mgr2


class TestServiceClients:
    """Tests for infrastructure/clients.py module."""

    def test_circuit_state_values(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitState
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_client_config_defaults(self):
        from services.orchestrator_service.src.infrastructure.clients import ClientConfig
        config = ClientConfig(base_url="http://localhost:8000")
        assert config.base_url == "http://localhost:8000"
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3

    def test_service_response_success(self):
        from services.orchestrator_service.src.infrastructure.clients import ServiceResponse
        response = ServiceResponse(success=True, data={"id": "123"}, status_code=200, response_time_ms=50.0)
        assert response.success is True
        assert response.data == {"id": "123"}
        assert response.status_code == 200

    def test_service_response_failure(self):
        from services.orchestrator_service.src.infrastructure.clients import ServiceResponse
        response = ServiceResponse(success=False, error="Not found", status_code=404)
        assert response.success is False
        assert response.error == "Not found"

    def test_service_response_to_dict(self):
        from services.orchestrator_service.src.infrastructure.clients import ServiceResponse
        response = ServiceResponse(success=True, data={"key": "value"}, status_code=200, response_time_ms=25.5)
        resp_dict = response.to_dict()
        assert resp_dict["success"] is True
        assert resp_dict["data"] == {"key": "value"}
        assert resp_dict["response_time_ms"] == 25.5

    def test_circuit_breaker_initial_state(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True

    def test_circuit_breaker_opens_on_failures(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_circuit_breaker_resets_on_success(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_service_client_factory_creates_clients(self):
        from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory, PersonalityServiceClient, DiagnosisServiceClient
        factory = ServiceClientFactory()
        personality = factory.personality()
        diagnosis = factory.diagnosis()
        assert isinstance(personality, PersonalityServiceClient)
        assert isinstance(diagnosis, DiagnosisServiceClient)

    def test_service_client_factory_caches_clients(self):
        from services.orchestrator_service.src.infrastructure.clients import ServiceClientFactory
        factory = ServiceClientFactory()
        client1 = factory.personality()
        client2 = factory.personality()
        assert client1 is client2

    def test_service_client_health(self):
        from services.orchestrator_service.src.infrastructure.clients import PersonalityServiceClient
        client = PersonalityServiceClient()
        health = client.get_health()
        assert "circuit_state" in health
        assert "base_url" in health


class TestStatePersistence:
    """Tests for infrastructure/state.py module."""

    def test_checkpoint_metadata_creation(self):
        from services.orchestrator_service.src.infrastructure.state import CheckpointMetadata
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_123", thread_id="thread_1", user_id="user_1", session_id="session_1",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        assert metadata.checkpoint_id == "cp_123"
        assert metadata.thread_id == "thread_1"

    def test_checkpoint_metadata_to_dict(self):
        from services.orchestrator_service.src.infrastructure.state import CheckpointMetadata
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_456", thread_id="thread_2", user_id="user_2", session_id="session_2",
            created_at=now, expires_at=now + timedelta(hours=12), version=2, size_bytes=1024,
        )
        md_dict = metadata.to_dict()
        assert md_dict["checkpoint_id"] == "cp_456"
        assert md_dict["version"] == 2
        assert md_dict["size_bytes"] == 1024

    def test_checkpoint_metadata_from_dict(self):
        from services.orchestrator_service.src.infrastructure.state import CheckpointMetadata
        now = datetime.now(timezone.utc)
        data = {
            "checkpoint_id": "cp_789", "thread_id": "thread_3", "user_id": "user_3", "session_id": "session_3",
            "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=24)).isoformat(),
        }
        metadata = CheckpointMetadata.from_dict(data)
        assert metadata.checkpoint_id == "cp_789"
        assert metadata.thread_id == "thread_3"

    def test_checkpoint_creation(self):
        from services.orchestrator_service.src.infrastructure.state import Checkpoint, CheckpointMetadata
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_1", thread_id="t_1", user_id="u_1", session_id="s_1",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        checkpoint = Checkpoint(metadata=metadata, state={"current_message": "Hello"})
        assert checkpoint.state["current_message"] == "Hello"

    def test_checkpoint_to_dict(self):
        from services.orchestrator_service.src.infrastructure.state import Checkpoint, CheckpointMetadata
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_2", thread_id="t_2", user_id="u_2", session_id="s_2",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        checkpoint = Checkpoint(metadata=metadata, state={"phase": "processing"})
        cp_dict = checkpoint.to_dict()
        assert cp_dict["metadata"]["checkpoint_id"] == "cp_2"
        assert cp_dict["state"]["phase"] == "processing"

    @pytest.mark.asyncio
    async def test_memory_state_store_save_and_load(self):
        from services.orchestrator_service.src.infrastructure.state import MemoryStateStore, Checkpoint, CheckpointMetadata
        store = MemoryStateStore()
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_test", thread_id="thread_test", user_id="user_test", session_id="session_test",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        checkpoint = Checkpoint(metadata=metadata, state={"message": "test"})
        result = await store.save(checkpoint)
        assert result is True
        loaded = await store.load("thread_test")
        assert loaded is not None
        assert loaded.state["message"] == "test"

    @pytest.mark.asyncio
    async def test_memory_state_store_delete(self):
        from services.orchestrator_service.src.infrastructure.state import MemoryStateStore, Checkpoint, CheckpointMetadata
        store = MemoryStateStore()
        now = datetime.now(timezone.utc)
        metadata = CheckpointMetadata(
            checkpoint_id="cp_del", thread_id="thread_del", user_id="user_del", session_id="session_del",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        checkpoint = Checkpoint(metadata=metadata, state={})
        await store.save(checkpoint)
        deleted = await store.delete("thread_del")
        assert deleted is True
        loaded = await store.load("thread_del")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_memory_state_store_list_checkpoints(self):
        from services.orchestrator_service.src.infrastructure.state import MemoryStateStore, Checkpoint, CheckpointMetadata
        store = MemoryStateStore()
        now = datetime.now(timezone.utc)
        for i in range(3):
            metadata = CheckpointMetadata(
                checkpoint_id=f"cp_{i}", thread_id=f"thread_{i}", user_id="user_1", session_id=f"session_{i}",
                created_at=now, expires_at=now + timedelta(hours=24),
            )
            await store.save(Checkpoint(metadata=metadata, state={}))
        checkpoints = await store.list_checkpoints("user_1")
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_memory_state_store_cleanup_expired(self):
        from services.orchestrator_service.src.infrastructure.state import MemoryStateStore, Checkpoint, CheckpointMetadata
        store = MemoryStateStore()
        now = datetime.now(timezone.utc)
        expired_metadata = CheckpointMetadata(
            checkpoint_id="cp_expired", thread_id="thread_expired", user_id="user_exp", session_id="session_exp",
            created_at=now - timedelta(hours=48), expires_at=now - timedelta(hours=24),
        )
        await store.save(Checkpoint(metadata=expired_metadata, state={}))
        valid_metadata = CheckpointMetadata(
            checkpoint_id="cp_valid", thread_id="thread_valid", user_id="user_val", session_id="session_val",
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        await store.save(Checkpoint(metadata=valid_metadata, state={}))
        cleaned = await store.cleanup_expired()
        assert cleaned == 1

    @pytest.mark.asyncio
    async def test_state_persistence_manager_save_and_load(self):
        from services.orchestrator_service.src.infrastructure.state import StatePersistenceManager
        from services.orchestrator_service.src.langgraph.state_schema import create_initial_state
        manager = StatePersistenceManager()
        state = create_initial_state(user_id=uuid4(), session_id=uuid4(), message="Hello test")
        metadata = await manager.save_state(state)
        assert metadata.thread_id == state["thread_id"]
        loaded = await manager.load_state(state["thread_id"])
        assert loaded is not None
        assert loaded["current_message"] == "Hello test"

    def test_state_persistence_manager_statistics(self):
        from services.orchestrator_service.src.infrastructure.state import StatePersistenceManager
        manager = StatePersistenceManager()
        stats = manager.get_statistics()
        assert "save_count" in stats
        assert "load_count" in stats
        assert "backend" in stats

    def test_get_persistence_manager_singleton(self):
        from services.orchestrator_service.src.infrastructure.state import get_persistence_manager
        mgr1 = get_persistence_manager()
        mgr2 = get_persistence_manager()
        assert mgr1 is mgr2
