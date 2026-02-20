"""
Unit tests for Solace-AI Escalation Manager.
Tests escalation workflows, clinician assignment, and notifications.
"""
from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import AsyncMock
import pytest
from uuid import UUID, uuid4
from services.safety_service.src.domain.escalation import (
    EscalationManager, EscalationSettings, EscalationResult,
    EscalationPriority, EscalationStatus, NotificationType,
    EscalationRecord, NotificationService, ClinicianAssigner,
    CrisisResourceManager, EscalationWorkflow,
)


@dataclass
class _MockClinicianContact:
    """Mock clinician contact for testing."""
    clinician_id: UUID
    email: str
    name: str
    phone: str | None = None
    is_on_call: bool = True


def _make_mock_registry(pool_size: int = 3) -> AsyncMock:
    """Create a mock clinician registry that returns pool_size clinicians."""
    contacts = [
        _MockClinicianContact(
            clinician_id=uuid4(),
            email=f"clinician{i}@test.com",
            name=f"Dr. Test {i}",
        )
        for i in range(pool_size)
    ]
    registry = AsyncMock()
    registry.get_oncall_clinicians = AsyncMock(return_value=contacts)
    return registry


class TestEscalationPriority:
    """Tests for EscalationPriority enum."""

    def test_from_crisis_level_critical(self) -> None:
        """Test CRITICAL crisis maps to CRITICAL priority."""
        assert EscalationPriority.from_crisis_level("CRITICAL") == EscalationPriority.CRITICAL

    def test_from_crisis_level_high(self) -> None:
        """Test HIGH crisis maps to HIGH priority."""
        assert EscalationPriority.from_crisis_level("HIGH") == EscalationPriority.HIGH

    def test_from_crisis_level_elevated(self) -> None:
        """Test ELEVATED crisis maps to MEDIUM priority."""
        assert EscalationPriority.from_crisis_level("ELEVATED") == EscalationPriority.MEDIUM

    def test_from_crisis_level_low(self) -> None:
        """Test LOW crisis maps to LOW priority."""
        assert EscalationPriority.from_crisis_level("LOW") == EscalationPriority.LOW

    def test_from_crisis_level_none(self) -> None:
        """Test NONE crisis maps to LOW priority."""
        assert EscalationPriority.from_crisis_level("NONE") == EscalationPriority.LOW


class TestNotificationService:
    """Tests for NotificationService."""

    @pytest.fixture
    def service(self) -> NotificationService:
        """Create notification service."""
        return NotificationService(EscalationSettings())

    @pytest.mark.asyncio
    async def test_send_notification(self, service: NotificationService) -> None:
        """Test sending single notification."""
        clinician_id = uuid4()
        escalation = EscalationRecord(user_id=uuid4(), crisis_level="HIGH")
        result = await service.send_notification(clinician_id, escalation, NotificationType.SMS)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_multi_channel(self, service: NotificationService) -> None:
        """Test sending notifications across multiple channels."""
        clinician_id = uuid4()
        escalation = EscalationRecord(user_id=uuid4(), crisis_level="CRITICAL")
        channels = [NotificationType.SMS, NotificationType.EMAIL]
        results = await service.send_multi_channel(clinician_id, escalation, channels)
        assert all(sent for sent in results.values())


class TestClinicianAssigner:
    """Tests for ClinicianAssigner."""

    @pytest.fixture
    def assigner(self) -> ClinicianAssigner:
        """Create clinician assigner with mock registry."""
        registry = _make_mock_registry(pool_size=3)
        return ClinicianAssigner(
            EscalationSettings(on_call_clinician_pool_size=3),
            clinician_registry=registry,
        )

    @pytest.mark.asyncio
    async def test_assign_clinician(self, assigner: ClinicianAssigner) -> None:
        """Test assigning clinician to escalation."""
        escalation = EscalationRecord(user_id=uuid4(), crisis_level="HIGH")
        clinician = await assigner.assign_clinician(escalation)
        assert clinician is not None

    @pytest.mark.asyncio
    async def test_workload_balancing(self, assigner: ClinicianAssigner) -> None:
        """Test workload balancing across clinicians."""
        escalations = [EscalationRecord(user_id=uuid4()) for _ in range(3)]
        clinicians = []
        for esc in escalations:
            clinician = await assigner.assign_clinician(esc)
            clinicians.append(clinician)
        assert len(set(clinicians)) == 3

    @pytest.mark.asyncio
    async def test_release_clinician(self, assigner: ClinicianAssigner) -> None:
        """Test releasing clinician."""
        # First assign a clinician so workload dict gets populated
        escalation = EscalationRecord(user_id=uuid4(), crisis_level="HIGH")
        clinician_id = await assigner.assign_clinician(escalation)
        assert clinician_id is not None
        # Bump workload to 2 then release
        assigner._clinician_workload[clinician_id] = 2
        assigner.release_clinician(clinician_id)
        assert assigner._clinician_workload[clinician_id] == 1


class TestCrisisResourceManager:
    """Tests for CrisisResourceManager."""

    @pytest.fixture
    def manager(self) -> CrisisResourceManager:
        """Create crisis resource manager."""
        return CrisisResourceManager()

    def test_get_resources_critical(self, manager: CrisisResourceManager) -> None:
        """Test getting resources for CRITICAL level."""
        resources = manager.get_resources_for_level("CRITICAL")
        assert len(resources) > 0
        assert any("911" in r["contact"] for r in resources)

    def test_get_resources_high(self, manager: CrisisResourceManager) -> None:
        """Test getting resources for HIGH level."""
        resources = manager.get_resources_for_level("HIGH")
        assert len(resources) > 0
        assert any("988" in r["contact"] for r in resources)

    def test_get_resources_low(self, manager: CrisisResourceManager) -> None:
        """Test getting resources for LOW level."""
        resources = manager.get_resources_for_level("LOW")
        assert len(resources) > 0


class TestEscalationWorkflow:
    """Tests for EscalationWorkflow."""

    @pytest.fixture
    def workflow(self) -> EscalationWorkflow:
        """Create escalation workflow with mock registry."""
        settings = EscalationSettings()
        notification_service = NotificationService(settings)
        registry = _make_mock_registry(pool_size=3)
        clinician_assigner = ClinicianAssigner(settings, clinician_registry=registry)
        return EscalationWorkflow(settings, notification_service, clinician_assigner)

    @pytest.mark.asyncio
    async def test_critical_workflow(self, workflow: EscalationWorkflow) -> None:
        """Test CRITICAL priority workflow execution."""
        escalation = EscalationRecord(
            user_id=uuid4(),
            crisis_level="CRITICAL",
            priority=EscalationPriority.CRITICAL,
        )
        result = await workflow.execute_critical_workflow(escalation)
        assert "CRITICAL workflow initiated" in result.actions_taken
        assert result.status == EscalationStatus.IN_PROGRESS
        assert result.assigned_clinician_id is not None

    @pytest.mark.asyncio
    async def test_high_workflow(self, workflow: EscalationWorkflow) -> None:
        """Test HIGH priority workflow execution."""
        escalation = EscalationRecord(
            user_id=uuid4(),
            crisis_level="HIGH",
            priority=EscalationPriority.HIGH,
        )
        result = await workflow.execute_high_workflow(escalation)
        assert "HIGH workflow initiated" in result.actions_taken
        assert result.status == EscalationStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_medium_workflow(self, workflow: EscalationWorkflow) -> None:
        """Test MEDIUM priority workflow execution."""
        escalation = EscalationRecord(
            user_id=uuid4(),
            crisis_level="ELEVATED",
            priority=EscalationPriority.MEDIUM,
        )
        result = await workflow.execute_medium_workflow(escalation)
        assert "MEDIUM workflow initiated" in result.actions_taken
        assert "Enhanced monitoring enabled" in result.actions_taken

    @pytest.mark.asyncio
    async def test_low_workflow(self, workflow: EscalationWorkflow) -> None:
        """Test LOW priority workflow execution."""
        escalation = EscalationRecord(
            user_id=uuid4(),
            crisis_level="LOW",
            priority=EscalationPriority.LOW,
        )
        result = await workflow.execute_low_workflow(escalation)
        assert "LOW workflow initiated" in result.actions_taken


class TestEscalationManager:
    """Tests for main EscalationManager class."""

    @pytest.fixture
    def manager(self) -> EscalationManager:
        """Create escalation manager with mock clinician registry."""
        registry = _make_mock_registry(pool_size=3)
        return EscalationManager(EscalationSettings(), clinician_registry=registry)

    @pytest.mark.asyncio
    async def test_escalate_critical(self, manager: EscalationManager) -> None:
        """Test escalating CRITICAL crisis."""
        user_id = uuid4()
        result = await manager.escalate(
            user_id=user_id,
            session_id=uuid4(),
            crisis_level="CRITICAL",
            reason="Suicidal ideation detected",
        )
        assert result.priority == "CRITICAL"
        assert result.assigned_clinician_id is not None
        assert result.notification_sent is True
        assert result.estimated_response_minutes == 5

    @pytest.mark.asyncio
    async def test_escalate_high(self, manager: EscalationManager) -> None:
        """Test escalating HIGH crisis."""
        result = await manager.escalate(
            user_id=uuid4(),
            session_id=None,
            crisis_level="HIGH",
            reason="Self-harm indicators",
        )
        assert result.priority == "HIGH"
        assert result.estimated_response_minutes == 15

    @pytest.mark.asyncio
    async def test_escalate_with_priority_override(self, manager: EscalationManager) -> None:
        """Test escalation with priority override."""
        result = await manager.escalate(
            user_id=uuid4(),
            session_id=None,
            crisis_level="LOW",
            reason="Test reason",
            priority_override="CRITICAL",
        )
        assert result.priority == "CRITICAL"

    def test_get_crisis_resources(self, manager: EscalationManager) -> None:
        """Test getting crisis resources."""
        resources = manager.get_crisis_resources("CRITICAL")
        assert len(resources) > 0
        assert any("988" in r["contact"] for r in resources)

    @pytest.mark.asyncio
    async def test_acknowledge_escalation(self, manager: EscalationManager) -> None:
        """Test acknowledging escalation."""
        result = await manager.escalate(
            user_id=uuid4(),
            session_id=None,
            crisis_level="HIGH",
            reason="Test",
        )
        success = await manager.acknowledge_escalation(result.escalation_id, uuid4())
        assert success is True

    @pytest.mark.asyncio
    async def test_resolve_escalation(self, manager: EscalationManager) -> None:
        """Test resolving escalation."""
        result = await manager.escalate(
            user_id=uuid4(),
            session_id=None,
            crisis_level="HIGH",
            reason="Test",
        )
        success = await manager.resolve_escalation(result.escalation_id, "Crisis resolved")
        assert success is True

    @pytest.mark.asyncio
    async def test_get_active_escalations(self, manager: EscalationManager) -> None:
        """Test getting active escalations."""
        user_id = uuid4()
        await manager.escalate(user_id=user_id, session_id=None, crisis_level="HIGH", reason="Test")
        active = manager.get_active_escalations(user_id)
        assert len(active) >= 1

    def test_get_statistics(self, manager: EscalationManager) -> None:
        """Test getting escalation statistics."""
        stats = manager.get_statistics()
        assert "total_active" in stats
        assert "by_priority" in stats
        assert "by_status" in stats


class TestEscalationSettings:
    """Tests for EscalationSettings."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = EscalationSettings()
        assert settings.auto_escalate_critical is True
        assert settings.critical_response_sla_minutes == 5
        assert settings.high_response_sla_minutes == 15

    def test_custom_sla(self) -> None:
        """Test custom SLA settings."""
        settings = EscalationSettings(
            critical_response_sla_minutes=3,
            high_response_sla_minutes=10,
        )
        assert settings.critical_response_sla_minutes == 3
        assert settings.high_response_sla_minutes == 10


class TestEscalationResult:
    """Tests for EscalationResult model."""

    def test_create_result(self) -> None:
        """Test creating escalation result."""
        result = EscalationResult(
            status="IN_PROGRESS",
            priority="HIGH",
            notification_sent=True,
            actions_taken=["Action 1", "Action 2"],
        )
        assert result.status == "IN_PROGRESS"
        assert result.priority == "HIGH"
        assert len(result.actions_taken) == 2

    def test_default_values(self) -> None:
        """Test default values."""
        result = EscalationResult(priority="LOW")
        assert result.status == "PENDING"
        assert result.notification_sent is False
        assert result.resources_provided is False
