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

    @pytest.mark.asyncio
    async def test_circuit_breaker_initial_state(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert breaker.state == CircuitState.CLOSED
        assert await breaker.allow_request() is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert await breaker.allow_request() is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        from services.orchestrator_service.src.infrastructure.clients import CircuitBreaker, CircuitState
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_success()
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
