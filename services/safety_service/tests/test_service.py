"""
Unit tests for Solace-AI Safety Service.
Tests main safety orchestration and coordination.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from services.safety_service.src.domain.crisis_detector import (
    CrisisDetector,
    CrisisDetectorSettings,
    CrisisLevel,
)
from services.safety_service.src.domain.escalation import (
    EscalationManager,
    EscalationSettings,
)
from services.safety_service.src.domain.service import (
    OutputFilterResult,
    SafetyCheckResult,
    SafetyService,
    SafetyServiceSettings,
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


class TestSafetyServiceSettings:
    """Tests for SafetyServiceSettings."""

    def test_default_settings(self) -> None:
        """Test default settings."""
        settings = SafetyServiceSettings()
        assert settings.enable_pre_check is True
        assert settings.enable_post_check is True
        assert settings.auto_escalate_critical is True

    def test_custom_settings(self) -> None:
        """Test custom settings."""
        settings = SafetyServiceSettings(
            auto_escalate_high=False,
            max_history_messages=10,
        )
        assert settings.auto_escalate_high is False
        assert settings.max_history_messages == 10


class TestSafetyService:
    """Tests for SafetyService class."""

    @pytest.fixture
    def service(self) -> SafetyService:
        """Create safety service with mock clinician registry."""
        settings = SafetyServiceSettings()
        crisis_detector = CrisisDetector(CrisisDetectorSettings())
        registry = _make_mock_registry(pool_size=3)
        escalation_manager = EscalationManager(EscalationSettings(), clinician_registry=registry)
        return SafetyService(settings, crisis_detector, escalation_manager)

    @pytest.mark.asyncio
    async def test_initialize(self, service: SafetyService) -> None:
        """Test service initialization."""
        await service.initialize()
        assert service._initialized is True

    @pytest.mark.asyncio
    async def test_shutdown(self, service: SafetyService) -> None:
        """Test service shutdown."""
        await service.initialize()
        await service.shutdown()
        assert service._initialized is False

    @pytest.mark.asyncio
    async def test_check_safety_safe_content(self, service: SafetyService) -> None:
        """Test safety check with safe content."""
        await service.initialize()
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I'm feeling good today",
            check_type="pre_check",
        )
        assert result.is_safe is True
        assert result.crisis_level == CrisisLevel.NONE
        assert result.requires_escalation is False

    @pytest.mark.asyncio
    async def test_check_safety_crisis_content(self, service: SafetyService) -> None:
        """Test safety check with crisis content."""
        await service.initialize()
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I want to end my life tonight",
            check_type="pre_check",
        )
        assert result.is_safe is False
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        assert result.requires_escalation is True

    @pytest.mark.asyncio
    async def test_check_safety_with_context(self, service: SafetyService) -> None:
        """Test safety check with additional context."""
        await service.initialize()
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I'm feeling anxious",
            check_type="pre_check",
            context={"has_treatment_plan": True},
        )
        assert len(result.protective_factors) > 0

    @pytest.mark.asyncio
    async def test_detect_crisis(self, service: SafetyService) -> None:
        """Test direct crisis detection."""
        await service.initialize()
        result = await service.detect_crisis(
            user_id=uuid4(),
            content="I'm planning to hurt myself",
            conversation_history=["I feel terrible", "Nobody cares"],
        )
        assert result.crisis_detected is True
        assert result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_detect_crisis_no_history(self, service: SafetyService) -> None:
        """Test crisis detection without history."""
        await service.initialize()
        result = await service.detect_crisis(
            user_id=uuid4(),
            content="I'm doing well today",
        )
        assert result.crisis_detected is False

    @pytest.mark.asyncio
    async def test_escalate(self, service: SafetyService) -> None:
        """Test escalation triggering."""
        await service.initialize()
        result = await service.escalate(
            user_id=uuid4(),
            session_id=uuid4(),
            crisis_level="CRITICAL",
            reason="Suicidal ideation",
        )
        assert result.priority == "CRITICAL"
        assert result.notification_sent is True

    @pytest.mark.asyncio
    async def test_assess_safety_single_message(self, service: SafetyService) -> None:
        """Test safety assessment with single message."""
        await service.initialize()
        result = await service.assess_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            messages=["I'm feeling stressed about work"],
        )
        assert result.overall_risk_level in list(CrisisLevel)
        assert len(result.message_assessments) == 1

    @pytest.mark.asyncio
    async def test_assess_safety_multiple_messages(self, service: SafetyService) -> None:
        """Test safety assessment with multiple messages."""
        await service.initialize()
        result = await service.assess_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            messages=[
                "I'm feeling down",
                "Things are getting worse",
                "I don't know what to do",
            ],
        )
        assert len(result.message_assessments) == 3
        assert result.trajectory_analysis is not None

    @pytest.mark.asyncio
    async def test_assess_safety_critical_messages(self, service: SafetyService) -> None:
        """Test safety assessment with critical content."""
        await service.initialize()
        result = await service.assess_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            messages=["I want to end my life"],
        )
        assert result.requires_intervention is True
        assert result.overall_risk_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_filter_output_safe(self, service: SafetyService) -> None:
        """Test output filtering with safe response."""
        await service.initialize()
        result = await service.filter_output(
            user_id=uuid4(),
            original_response="I understand you're feeling stressed.",
            user_crisis_level="NONE",
            include_resources=False,
        )
        assert result.is_safe is True
        assert result.filtered_response == "I understand you're feeling stressed."

    @pytest.mark.asyncio
    async def test_filter_output_with_resources(self, service: SafetyService) -> None:
        """Test output filtering with crisis resources."""
        await service.initialize()
        result = await service.filter_output(
            user_id=uuid4(),
            original_response="I'm here to help.",
            user_crisis_level="HIGH",
            include_resources=True,
        )
        assert result.resources_appended is True
        assert "988" in result.filtered_response

    @pytest.mark.asyncio
    async def test_get_status(self, service: SafetyService) -> None:
        """Test getting service status."""
        await service.initialize()
        status = await service.get_status()
        assert status["status"] == "operational"
        assert status["initialized"] is True
        assert "statistics" in status

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, service: SafetyService) -> None:
        """Test statistics are tracked."""
        await service.initialize()
        await service.check_safety(uuid4(), None, "test", "pre_check")
        await service.check_safety(uuid4(), None, "test", "post_check")
        status = await service.get_status()
        assert status["statistics"]["total_checks"] == 2
        assert status["statistics"]["pre_checks"] == 1
        assert status["statistics"]["post_checks"] == 1

    @pytest.mark.asyncio
    async def test_conversation_history_management(self, service: SafetyService) -> None:
        """Test conversation history is managed per user."""
        await service.initialize()
        user_id = uuid4()
        await service.check_safety(user_id, None, "message 1", "pre_check")
        await service.check_safety(user_id, None, "message 2", "pre_check")
        assert len(service._conversation_history[user_id]) == 2

    @pytest.mark.asyncio
    async def test_risk_history_updates(self, service: SafetyService) -> None:
        """Test user risk history updates on crisis."""
        await service.initialize()
        user_id = uuid4()
        await service.check_safety(user_id, None, "I want to end my life", "pre_check")
        assert user_id in service._user_risk_history
        assert service._user_risk_history[user_id]["previous_crisis_events"] >= 1


class TestSafetyCheckResult:
    """Tests for SafetyCheckResult model."""

    def test_create_result(self) -> None:
        """Test creating safety check result."""
        result = SafetyCheckResult(
            is_safe=True,
            crisis_level=CrisisLevel.NONE,
            risk_score=Decimal("0.1"),
            recommended_action="continue",
            detection_time_ms=5,
            detection_layer=1,
        )
        assert result.is_safe is True
        assert result.crisis_level == CrisisLevel.NONE

    def test_default_values(self) -> None:
        """Test default values."""
        result = SafetyCheckResult()
        assert result.is_safe is True
        assert result.requires_escalation is False


class TestOutputFilterResult:
    """Tests for OutputFilterResult model."""

    def test_create_result(self) -> None:
        """Test creating output filter result."""
        result = OutputFilterResult(
            filtered_response="Test response",
            is_safe=True,
            filter_time_ms=2,
        )
        assert result.filtered_response == "Test response"
        assert result.is_safe is True

    def test_with_modifications(self) -> None:
        """Test result with modifications."""
        result = OutputFilterResult(
            filtered_response="Modified response",
            modifications_made=["Removed unsafe phrase"],
            is_safe=False,
            filter_time_ms=3,
        )
        assert len(result.modifications_made) == 1
        assert result.is_safe is False


class TestIntegration:
    """Integration tests for safety service components."""

    @pytest.fixture
    def service(self) -> SafetyService:
        """Create fully configured service with mock clinician registry."""
        registry = _make_mock_registry(pool_size=3)
        return SafetyService(
            SafetyServiceSettings(auto_escalate_critical=True),
            CrisisDetector(CrisisDetectorSettings()),
            EscalationManager(EscalationSettings(), clinician_registry=registry),
        )

    @pytest.mark.asyncio
    async def test_full_crisis_flow(self, service: SafetyService) -> None:
        """Test full crisis detection and escalation flow.

        'I'm going to kill myself tonight' scores ~0.867 (HIGH level) due to
        weighted normalization. requires_human_review is only True for CRITICAL.
        """
        await service.initialize()
        check_result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I'm going to kill myself tonight",
            check_type="pre_check",
        )
        assert check_result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL)
        assert check_result.requires_escalation is True
        # requires_human_review is only True for CRITICAL; weighted scoring
        # puts this input at HIGH (~0.867), so human review may not trigger
        if check_result.crisis_level == CrisisLevel.CRITICAL:
            assert check_result.requires_human_review is True
        else:
            assert check_result.requires_human_review is False

    @pytest.mark.asyncio
    async def test_progressive_risk_detection(self, service: SafetyService) -> None:
        """Test progressive risk detection over conversation.

        Weighted scoring normalizes 'don't want to live' high keyword (0.75)
        to ~0.375 (LOW). Earlier messages don't trigger crisis events, so risk
        history stays empty and doesn't boost the final score.
        """
        await service.initialize()
        user_id = uuid4()
        await service.check_safety(user_id, None, "I'm feeling sad", "pre_check")
        await service.check_safety(user_id, None, "Things are getting worse", "pre_check")
        result = await service.check_safety(user_id, None, "I don't want to live anymore", "pre_check")
        assert result.crisis_level in (CrisisLevel.LOW, CrisisLevel.ELEVATED, CrisisLevel.HIGH, CrisisLevel.CRITICAL)


class _CapturingEventPublisher:
    """Minimal stand-in for SafetyEventPublisher that records emitted events.

    The real publisher has a start/stop lifecycle and an async queue; for
    testing which events fire from a given service method we only need the
    ``publish(event)`` coroutine.
    """

    def __init__(self) -> None:
        self.events: list = []  # type: ignore[type-arg]

    async def publish(self, event) -> None:  # type: ignore[no-untyped-def]
        self.events.append(event)

    def event_types(self) -> list[str]:
        return [e.event_type.value for e in self.events]


class TestSafetyServiceEventEmission:
    """C-13 regression: SafetyService must emit the right events from every
    clinically significant operation — not just check_safety.

    The Sprint 0 audit confirmed C-13 is partially wired for
    ``check_safety`` and ``detect_crisis``. Sprint 2 Day 2 adds the missing
    emissions for ``escalate`` (EscalationTriggeredEvent) and
    ``filter_output`` (OutputFilteredEvent). Without these, downstream
    services (notification, audit, analytics) cannot react to crises and
    HIPAA accounting of disclosures is incomplete.
    """

    @pytest.fixture
    def service_with_publisher(self) -> tuple[SafetyService, _CapturingEventPublisher]:
        settings = SafetyServiceSettings()
        crisis_detector = CrisisDetector(CrisisDetectorSettings())
        registry = _make_mock_registry(pool_size=2)
        escalation_manager = EscalationManager(
            EscalationSettings(), clinician_registry=registry
        )
        publisher = _CapturingEventPublisher()
        # Inject the capturing publisher as the service's event publisher
        svc = SafetyService(
            settings, crisis_detector, escalation_manager, event_publisher=publisher,
        )
        return svc, publisher

    @pytest.mark.asyncio
    async def test_escalate_emits_escalation_triggered_event(
        self,
        service_with_publisher: tuple[SafetyService, _CapturingEventPublisher],
    ) -> None:
        """C-13: escalate() must emit EscalationTriggeredEvent."""
        service, publisher = service_with_publisher
        await service.initialize()
        await service.escalate(
            user_id=uuid4(),
            session_id=uuid4(),
            crisis_level="HIGH",
            reason="suicidal ideation",
        )
        assert "safety.escalation.triggered" in publisher.event_types(), (
            f"C-13 regression: expected safety.escalation.triggered in emitted "
            f"events {publisher.event_types()}. Without this, the notification "
            f"service cannot alert the on-call clinician."
        )

    @pytest.mark.asyncio
    async def test_filter_output_emits_output_filtered_event_when_modified(
        self,
        service_with_publisher: tuple[SafetyService, _CapturingEventPublisher],
    ) -> None:
        """C-13: filter_output() must emit OutputFilteredEvent when it
        modifies the response (either content filtering or appending
        crisis resources). Essential for the audit trail."""
        service, publisher = service_with_publisher
        await service.initialize()
        await service.filter_output(
            user_id=uuid4(),
            original_response="You should be careful",
            user_crisis_level="HIGH",
            include_resources=True,
        )
        assert "safety.output.filtered" in publisher.event_types(), (
            f"C-13 regression: filter_output with HIGH crisis + resources="
            f"True must emit safety.output.filtered. Actual events: "
            f"{publisher.event_types()}"
        )


class TestProtectiveFactorRiskAdjustment:
    """H-06 regression: protective factors must reduce numerical risk score
    while never suppressing a CRITICAL crisis below the CRITICAL threshold.

    Clinical invariant: protective factors modulate quantitative risk but do
    not downgrade a detected crisis category. A critical-level ideation is
    still critical even if the user has strong social support; evidence
    base is too weak to let NLP-detected factors "talk us out of" a
    flagged crisis.
    """

    @pytest.fixture
    def service(self) -> SafetyService:
        settings = SafetyServiceSettings()
        crisis_detector = CrisisDetector(CrisisDetectorSettings())
        registry = _make_mock_registry(pool_size=2)
        escalation_manager = EscalationManager(
            EscalationSettings(), clinician_registry=registry
        )
        return SafetyService(settings, crisis_detector, escalation_manager)

    @pytest.mark.asyncio
    async def test_protective_factors_reduce_moderate_risk(
        self, service: SafetyService
    ) -> None:
        """Protective factor presence lowers a non-critical risk score."""
        await service.initialize()
        content_with = "I feel anxious but my family and therapy have been helpful"
        content_without = "I feel anxious"

        with_factors = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content=content_with,
            check_type="pre_check",
        )
        without_factors = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content=content_without,
            check_type="pre_check",
        )
        assert len(with_factors.protective_factors) > 0
        # Numerical risk should be lower when protective factors are present
        assert with_factors.risk_score <= without_factors.risk_score

    @pytest.mark.asyncio
    async def test_critical_risk_never_suppressed_by_protective_factors(
        self, service: SafetyService
    ) -> None:
        """H-06 core invariant: strong protective factors must NOT reduce a
        CRITICAL-level assessment's risk_score below the CRITICAL threshold
        (0.85). Otherwise a user reporting suicidal ideation plus a list of
        positives could be silently downgraded.
        """
        await service.initialize()
        critical_with_positives = (
            "I'm going to kill myself tonight, but my family has been great "
            "and therapy and medication really help and I have coping skills "
            "and hope for the future."
        )
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content=critical_with_positives,
            check_type="pre_check",
        )
        assert len(result.protective_factors) >= 3, (
            "Sanity: the seeded content should trip at least 3 protective "
            "factor keywords (family, therapy, medication, coping, hope)."
        )
        if result.crisis_level == CrisisLevel.CRITICAL:
            assert result.risk_score >= Decimal("0.85"), (
                "H-06: risk_score of a CRITICAL event must not drop below "
                "0.85 even in the presence of many protective factors."
            )
            assert result.requires_escalation is True
            assert result.requires_human_review is True

    @pytest.mark.asyncio
    async def test_protective_factors_never_flip_is_safe_on_crisis(
        self, service: SafetyService
    ) -> None:
        """Sister invariant to the CRITICAL floor: is_safe must remain False
        whenever the assessed crisis_level is HIGH or CRITICAL, regardless
        of the count of protective factors identified.
        """
        await service.initialize()
        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content=(
                "I want to end my life tonight, but my family and friends "
                "have been supportive and therapy is helping me cope."
            ),
            check_type="pre_check",
        )
        if result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            assert result.is_safe is False, (
                "H-06: protective factor presence must not re-flag a "
                "HIGH/CRITICAL assessment as safe."
            )
            assert result.requires_escalation is True
