"""
Unit tests for Therapy Service Orchestrator.
Tests hybrid rules+LLM therapy orchestration with CBT/DBT/ACT/MI modalities.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.schemas import SessionPhase, TherapyModality, SeverityLevel, RiskLevel
from services.therapy_service.src.domain.service import (
    TherapyOrchestrator, TherapyOrchestratorSettings
)
from services.therapy_service.src.domain.technique_selector import TechniqueSelector
from services.therapy_service.src.domain.session_manager import SessionManager


class TestTherapyOrchestratorSettings:
    """Tests for TherapyOrchestratorSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings are properly initialized."""
        settings = TherapyOrchestratorSettings()
        assert settings.enable_safety_checks is True
        assert len(settings.crisis_keywords) > 0
        assert settings.max_processing_time_ms == 15000
        assert settings.enable_homework_assignment is True
        assert settings.session_timeout_minutes == 60

    def test_custom_settings(self) -> None:
        """Test custom settings override defaults."""
        settings = TherapyOrchestratorSettings(
            enable_safety_checks=False,
            max_processing_time_ms=5000,
            session_timeout_minutes=30,
        )
        assert settings.enable_safety_checks is False
        assert settings.max_processing_time_ms == 5000
        assert settings.session_timeout_minutes == 30


class TestTherapyOrchestratorLifecycle:
    """Tests for TherapyOrchestrator lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize_orchestrator(self) -> None:
        """Test orchestrator initialization."""
        orchestrator = TherapyOrchestrator()
        await orchestrator.initialize()
        assert orchestrator._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown_orchestrator(self) -> None:
        """Test orchestrator shutdown."""
        orchestrator = TherapyOrchestrator()
        await orchestrator.initialize()
        await orchestrator.shutdown()
        assert orchestrator._initialized is False

    @pytest.mark.asyncio
    async def test_orchestrator_with_components(self) -> None:
        """Test orchestrator with technique selector and session manager."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        assert orchestrator._technique_selector is selector
        assert orchestrator._session_manager is manager
        await orchestrator.shutdown()


class TestSessionLifecycle:
    """Tests for session lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_session(self) -> None:
        """Test starting a new session."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        plan_id = uuid4()
        result = await orchestrator.start_session(user_id, plan_id, {"diagnosis": "Depression"})
        assert result.session_id is not None
        assert result.session_number == 1
        assert result.initial_message is not None
        assert len(result.suggested_agenda) > 0
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_start_session_creates_treatment_plan(self) -> None:
        """Test starting a session creates treatment plan if not exists."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        plan_id = uuid4()
        await orchestrator.start_session(user_id, plan_id, {"diagnosis": "Anxiety"})
        assert plan_id in orchestrator._treatment_plans
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_end_session(self) -> None:
        """Test ending a session."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        plan_id = uuid4()
        start_result = await orchestrator.start_session(user_id, plan_id, {})
        end_result = await orchestrator.end_session(start_result.session_id, user_id)
        assert end_result.summary is not None
        assert end_result.duration_minutes >= 0
        assert len(end_result.recommendations) > 0
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_end_session_user_mismatch(self) -> None:
        """Test ending session with wrong user raises error."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        start_result = await orchestrator.start_session(uuid4(), uuid4(), {})
        with pytest.raises(ValueError, match="User ID mismatch"):
            await orchestrator.end_session(start_result.session_id, uuid4())
        await orchestrator.shutdown()


class TestMessageProcessing:
    """Tests for message processing."""

    @pytest.mark.asyncio
    async def test_process_message_basic(self) -> None:
        """Test basic message processing."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {"diagnosis": "Depression"})
        result = await orchestrator.process_message(
            session_id=start_result.session_id,
            user_id=user_id,
            message="I've been feeling down lately.",
            conversation_history=[],
        )
        assert result.response_text is not None
        assert result.current_phase is not None
        assert result.processing_time_ms >= 0
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_process_message_updates_state(self) -> None:
        """Test message processing updates session state."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        await orchestrator.process_message(
            session_id=start_result.session_id,
            user_id=user_id,
            message="Test message",
            conversation_history=[],
        )
        state = await orchestrator.get_session_state(start_result.session_id)
        assert state is not None
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_process_message_user_mismatch(self) -> None:
        """Test processing message with wrong user raises error."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        start_result = await orchestrator.start_session(uuid4(), uuid4(), {})
        with pytest.raises(ValueError, match="User ID mismatch"):
            await orchestrator.process_message(
                session_id=start_result.session_id,
                user_id=uuid4(),
                message="Test",
                conversation_history=[],
            )
        await orchestrator.shutdown()


class TestSafetyChecks:
    """Tests for safety check functionality."""

    @pytest.mark.asyncio
    async def test_crisis_keyword_detection(self) -> None:
        """Test crisis keyword detection."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        result = await orchestrator.process_message(
            session_id=start_result.session_id,
            user_id=user_id,
            message="I want to hurt myself",
            conversation_history=[],
        )
        assert len(result.safety_alerts) > 0
        assert "988" in result.response_text
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_crisis_triggers_closing_phase(self) -> None:
        """Test crisis detection transitions to closing phase."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        result = await orchestrator.process_message(
            session_id=start_result.session_id,
            user_id=user_id,
            message="I want to end it all",
            conversation_history=[],
        )
        assert result.current_phase == SessionPhase.CLOSING
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_safety_checks_can_be_disabled(self) -> None:
        """Test safety checks can be disabled."""
        settings = TherapyOrchestratorSettings(enable_safety_checks=False)
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            settings=settings,
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        result = await orchestrator.process_message(
            session_id=start_result.session_id,
            user_id=user_id,
            message="I want to hurt myself",
            conversation_history=[],
        )
        assert len(result.safety_alerts) == 0
        await orchestrator.shutdown()


class TestTreatmentPlanManagement:
    """Tests for treatment plan management."""

    @pytest.mark.asyncio
    async def test_get_treatment_plan(self) -> None:
        """Test getting treatment plan for session."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        plan_id = uuid4()
        start_result = await orchestrator.start_session(user_id, plan_id, {"diagnosis": "Anxiety"})
        plan = await orchestrator.get_treatment_plan(start_result.session_id)
        assert plan is not None
        assert plan.plan_id == plan_id
        assert plan.primary_diagnosis == "Anxiety"
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_treatment_plan_has_modality(self) -> None:
        """Test treatment plan has primary modality."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        start_result = await orchestrator.start_session(uuid4(), uuid4(), {})
        plan = await orchestrator.get_treatment_plan(start_result.session_id)
        assert plan.primary_modality is not None
        await orchestrator.shutdown()


class TestTechniqueManagement:
    """Tests for technique management."""

    @pytest.mark.asyncio
    async def test_list_techniques(self) -> None:
        """Test listing available techniques."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        techniques = await orchestrator.list_techniques()
        assert len(techniques) > 0
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_list_techniques_by_modality(self) -> None:
        """Test listing techniques filtered by modality."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        techniques = await orchestrator.list_techniques(TherapyModality.CBT)
        assert all(t.modality == TherapyModality.CBT for t in techniques)
        await orchestrator.shutdown()


class TestStatistics:
    """Tests for service statistics."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        """Test getting service status."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        status = await orchestrator.get_status()
        assert status["status"] == "operational"
        assert status["initialized"] is True
        assert "statistics" in status
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_statistics_track_sessions(self) -> None:
        """Test statistics track session counts."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        await orchestrator.end_session(start_result.session_id, user_id)
        status = await orchestrator.get_status()
        assert status["statistics"]["sessions_started"] == 1
        assert status["statistics"]["sessions_ended"] == 1
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_statistics_track_messages(self) -> None:
        """Test statistics track message counts."""
        selector = TechniqueSelector()
        manager = SessionManager()
        orchestrator = TherapyOrchestrator(
            technique_selector=selector,
            session_manager=manager,
        )
        await orchestrator.initialize()
        user_id = uuid4()
        start_result = await orchestrator.start_session(user_id, uuid4(), {})
        await orchestrator.process_message(
            start_result.session_id, user_id, "Test", []
        )
        status = await orchestrator.get_status()
        assert status["statistics"]["messages_processed"] == 1
        await orchestrator.shutdown()
