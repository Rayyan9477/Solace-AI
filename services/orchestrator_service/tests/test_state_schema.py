"""
Unit tests for Orchestrator Service State Schema.
Tests state types, reducers, and validation.
"""
from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4
import pytest

from services.orchestrator_service.src.langgraph.state_schema import (
    RiskLevel,
    IntentType,
    AgentType,
    ProcessingPhase,
    MessageEntry,
    SafetyFlags,
    AgentResult,
    ProcessingMetadata,
    OrchestratorState,
    create_initial_state,
    StateValidator,
    add_messages,
    add_agent_results,
    update_safety_flags,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self) -> None:
        """Test all risk level values exist."""
        assert RiskLevel.NONE.value == "NONE"
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.ELEVATED.value == "ELEVATED"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_risk_level_from_string(self) -> None:
        """Test creating risk level from string."""
        assert RiskLevel("NONE") == RiskLevel.NONE
        assert RiskLevel("CRITICAL") == RiskLevel.CRITICAL


class TestIntentType:
    """Tests for IntentType enum."""

    def test_all_intent_types_defined(self) -> None:
        """Test all intent types are defined."""
        intent_types = [
            "general_chat", "emotional_support", "crisis_disclosure",
            "symptom_discussion", "treatment_inquiry", "progress_update",
            "assessment_request", "coping_strategy", "psychoeducation",
            "session_management",
        ]
        for intent in intent_types:
            assert IntentType(intent) is not None


class TestAgentType:
    """Tests for AgentType enum."""

    def test_all_agent_types_defined(self) -> None:
        """Test all agent types are defined."""
        agent_types = ["safety", "supervisor", "diagnosis", "therapy", "personality", "chat", "aggregator"]
        for agent in agent_types:
            assert AgentType(agent) is not None


class TestMessageEntry:
    """Tests for MessageEntry dataclass."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = MessageEntry.user_message("Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"
        assert msg.message_id is not None
        assert msg.timestamp is not None

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = MessageEntry.assistant_message("I'm here to help.")
        assert msg.role == "assistant"
        assert msg.content == "I'm here to help."

    def test_message_to_dict(self) -> None:
        """Test message serialization."""
        msg = MessageEntry.user_message("Test", metadata={"key": "value"})
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert data["metadata"] == {"key": "value"}
        assert "message_id" in data
        assert "timestamp" in data

    def test_message_from_dict(self) -> None:
        """Test message deserialization."""
        original = MessageEntry.user_message("Test message")
        data = original.to_dict()
        restored = MessageEntry.from_dict(data)
        assert restored.role == original.role
        assert restored.content == original.content

    def test_message_immutability(self) -> None:
        """Test that MessageEntry is immutable."""
        msg = MessageEntry.user_message("Test")
        with pytest.raises(AttributeError):
            msg.content = "Modified"


class TestSafetyFlags:
    """Tests for SafetyFlags dataclass."""

    def test_create_safe_flags(self) -> None:
        """Test creating safe default flags."""
        flags = SafetyFlags.safe()
        assert flags.risk_level == RiskLevel.NONE
        assert flags.crisis_detected is False
        assert flags.is_safe() is True

    def test_create_crisis_flags(self) -> None:
        """Test creating crisis flags."""
        flags = SafetyFlags(
            risk_level=RiskLevel.CRITICAL,
            crisis_detected=True,
            crisis_type="suicidal_ideation",
            requires_escalation=True,
        )
        assert flags.is_safe() is False
        assert flags.crisis_detected is True

    def test_safety_flags_to_dict(self) -> None:
        """Test safety flags serialization."""
        flags = SafetyFlags(
            risk_level=RiskLevel.HIGH,
            crisis_detected=True,
            triggered_keywords=["suicide"],
        )
        data = flags.to_dict()
        assert data["risk_level"] == "HIGH"
        assert data["crisis_detected"] is True
        assert "suicide" in data["triggered_keywords"]

    def test_safety_flags_from_dict(self) -> None:
        """Test safety flags deserialization."""
        data = {
            "risk_level": "ELEVATED",
            "crisis_detected": False,
            "monitoring_level": "enhanced",
            "contraindications": ["exposure_therapy"],
        }
        flags = SafetyFlags.from_dict(data)
        assert flags.risk_level == RiskLevel.ELEVATED
        assert flags.monitoring_level == "enhanced"
        assert "exposure_therapy" in flags.contraindications


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful agent result."""
        result = AgentResult(
            agent_type=AgentType.THERAPY,
            success=True,
            response_content="Here's a coping technique...",
            confidence=0.85,
            processing_time_ms=150.5,
        )
        assert result.success is True
        assert result.agent_type == AgentType.THERAPY
        assert result.confidence == 0.85

    def test_create_failed_result(self) -> None:
        """Test creating a failed agent result."""
        result = AgentResult(
            agent_type=AgentType.DIAGNOSIS,
            success=False,
            error="Service unavailable",
        )
        assert result.success is False
        assert result.error == "Service unavailable"

    def test_agent_result_serialization(self) -> None:
        """Test agent result round-trip serialization."""
        original = AgentResult(
            agent_type=AgentType.PERSONALITY,
            success=True,
            confidence=0.75,
            metadata={"style": "warm"},
        )
        data = original.to_dict()
        restored = AgentResult.from_dict(data)
        assert restored.agent_type == original.agent_type
        assert restored.confidence == original.confidence


class TestProcessingMetadata:
    """Tests for ProcessingMetadata dataclass."""

    def test_create_processing_metadata(self) -> None:
        """Test creating processing metadata."""
        user_id = uuid4()
        session_id = uuid4()
        meta = ProcessingMetadata(
            user_id=user_id,
            session_id=session_id,
            is_streaming=True,
        )
        assert meta.user_id == user_id
        assert meta.session_id == session_id
        assert meta.is_streaming is True
        assert meta.retry_count == 0

    def test_metadata_serialization(self) -> None:
        """Test metadata round-trip serialization."""
        original = ProcessingMetadata(
            active_agents=[AgentType.THERAPY, AgentType.PERSONALITY],
            completed_agents=[AgentType.SAFETY],
        )
        data = original.to_dict()
        restored = ProcessingMetadata.from_dict(data)
        assert AgentType.THERAPY in restored.active_agents
        assert AgentType.SAFETY in restored.completed_agents


class TestReducers:
    """Tests for state reducers."""

    def test_add_messages_reducer(self) -> None:
        """Test message aggregation reducer."""
        left = [{"role": "user", "content": "Hello"}]
        right = [{"role": "assistant", "content": "Hi there"}]
        result = add_messages(left, right)
        assert len(result) == 2
        assert result[0]["content"] == "Hello"
        assert result[1]["content"] == "Hi there"

    def test_add_agent_results_reducer(self) -> None:
        """Test agent results aggregation reducer."""
        left = [{"agent_type": "safety", "success": True}]
        right = [{"agent_type": "therapy", "success": True}]
        result = add_agent_results(left, right)
        assert len(result) == 2

    def test_update_safety_flags_reducer_escalates_risk(self) -> None:
        """Test safety flags reducer escalates to higher risk."""
        left = {"risk_level": "LOW", "crisis_detected": False}
        right = {"risk_level": "HIGH", "crisis_detected": True}
        result = update_safety_flags(left, right)
        assert result["risk_level"] == "HIGH"
        assert result["crisis_detected"] is True

    def test_update_safety_flags_reducer_preserves_higher_risk(self) -> None:
        """Test safety flags reducer preserves existing higher risk."""
        left = {"risk_level": "CRITICAL", "crisis_detected": True}
        right = {"risk_level": "ELEVATED", "crisis_detected": False}
        result = update_safety_flags(left, right)
        assert result["risk_level"] == "CRITICAL"
        assert result["crisis_detected"] is True

    def test_update_safety_flags_merges_contraindications(self) -> None:
        """Test safety flags reducer merges contraindications."""
        left = {"risk_level": "LOW", "contraindications": ["a", "b"]}
        right = {"risk_level": "LOW", "contraindications": ["b", "c"]}
        result = update_safety_flags(left, right)
        assert set(result["contraindications"]) == {"a", "b", "c"}


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_create_initial_state_minimal(self) -> None:
        """Test creating initial state with minimal parameters."""
        user_id = uuid4()
        session_id = uuid4()
        state = create_initial_state(user_id, session_id, "Hello")
        assert state["user_id"] == str(user_id)
        assert state["session_id"] == str(session_id)
        assert state["current_message"] == "Hello"
        assert state["processing_phase"] == "initialized"
        assert len(state["messages"]) == 1

    def test_create_initial_state_with_context(self) -> None:
        """Test creating initial state with conversation context."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="How are you?",
            thread_id="thread-789",
            conversation_context="Previous conversation about anxiety",
        )
        assert state["thread_id"] == "thread-789"
        assert state["conversation_context"] == "Previous conversation about anxiety"

    def test_create_initial_state_with_metadata(self) -> None:
        """Test creating initial state with metadata."""
        state = create_initial_state(
            user_id="user-123",
            session_id="session-456",
            message="Test",
            metadata={"source": "mobile_app"},
        )
        assert state["metadata"]["source"] == "mobile_app"


class TestStateValidator:
    """Tests for StateValidator class."""

    def test_validate_valid_state(self) -> None:
        """Test validating a valid state."""
        state = create_initial_state("user-1", "session-1", "Hello")
        is_valid, errors = StateValidator.validate_state(state)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_user_id(self) -> None:
        """Test validation fails for missing user_id."""
        state = OrchestratorState(
            user_id="",
            session_id="session-1",
            current_message="Hello",
        )
        is_valid, errors = StateValidator.validate_state(state)
        assert is_valid is False
        assert "Missing user_id" in errors

    def test_validate_invalid_processing_phase(self) -> None:
        """Test validation fails for invalid processing phase."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["processing_phase"] = "invalid_phase"
        is_valid, errors = StateValidator.validate_state(state)
        assert is_valid is False
        assert any("processing_phase" in e for e in errors)

    def test_is_terminal_phase_completed(self) -> None:
        """Test terminal phase detection for completed."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["processing_phase"] = "completed"
        assert StateValidator.is_terminal_phase(state) is True

    def test_is_terminal_phase_error(self) -> None:
        """Test terminal phase detection for error."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["processing_phase"] = "error"
        assert StateValidator.is_terminal_phase(state) is True

    def test_is_terminal_phase_processing(self) -> None:
        """Test non-terminal phase detection."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["processing_phase"] = "parallel_processing"
        assert StateValidator.is_terminal_phase(state) is False

    def test_requires_safety_intervention_crisis(self) -> None:
        """Test safety intervention required for crisis."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["safety_flags"] = {"crisis_detected": True}
        assert StateValidator.requires_safety_intervention(state) is True

    def test_requires_safety_intervention_escalation(self) -> None:
        """Test safety intervention required for escalation."""
        state = create_initial_state("user-1", "session-1", "Hello")
        state["safety_flags"] = {"requires_escalation": True}
        assert StateValidator.requires_safety_intervention(state) is True

    def test_no_safety_intervention_required(self) -> None:
        """Test no safety intervention for safe state."""
        state = create_initial_state("user-1", "session-1", "Hello")
        assert StateValidator.requires_safety_intervention(state) is False
