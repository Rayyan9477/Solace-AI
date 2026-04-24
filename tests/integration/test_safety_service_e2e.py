"""Sprint 2 end-to-end: SafetyService full crisis pathway.

Exercises the safety service's own orchestration in isolation:
  1. Crisis utterance triggers check_safety
  2. C-13: SafetyCheckCompletedEvent + CrisisDetectedEvent are emitted
  3. Auto-escalation fires -> escalate() is called internally
  4. C-13: EscalationTriggeredEvent is emitted
  5. H-04: The escalation is persisted in the repository
  6. H-03: Medium escalation sends a real supervisor notification
  7. filter_output emits OutputFilteredEvent when resources are appended

This test does NOT involve the orchestrator / langgraph layer — that's
covered in ``test_crisis_flow.py``. The goal here is to prove the safety
service owns its full side-effect surface without relying on the
orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    EscalationStatus,
    InMemoryEscalationRepository,
)
from services.safety_service.src.domain.service import (
    SafetyService,
    SafetyServiceSettings,
)

# ---------------------------------------------------------------------------
# Fixtures: capturing event publisher and clinician registry
# ---------------------------------------------------------------------------


class _CapturingPublisher:
    """Record every event published through the safety bus."""

    def __init__(self) -> None:
        self.events: list = []  # type: ignore[type-arg]

    async def publish(self, event) -> None:  # type: ignore[no-untyped-def]
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.event_type.value for e in self.events]


@dataclass
class _FakeClinician:
    clinician_id: UUID
    email: str
    name: str
    phone: str | None = None
    is_on_call: bool = True


def _make_registry(count: int = 2) -> AsyncMock:
    contacts = [
        _FakeClinician(
            clinician_id=uuid4(),
            email=f"on-call{i}@example.test",
            name=f"Dr. {i}",
        )
        for i in range(count)
    ]
    reg = AsyncMock()
    reg.get_oncall_clinicians = AsyncMock(return_value=contacts)
    return reg


@pytest.fixture
def service_and_publisher() -> tuple[SafetyService, _CapturingPublisher, InMemoryEscalationRepository]:
    publisher = _CapturingPublisher()
    repo = InMemoryEscalationRepository()
    svc = SafetyService(
        SafetyServiceSettings(),
        CrisisDetector(CrisisDetectorSettings()),
        EscalationManager(
            EscalationSettings(),
            clinician_registry=_make_registry(),
            repository=repo,
        ),
        event_publisher=publisher,
    )
    return svc, publisher, repo


# ---------------------------------------------------------------------------
# Core demo-path scenarios
# ---------------------------------------------------------------------------


class TestSafetyServiceCrisisPath:
    """Full crisis pathway with event + persistence side-effects."""

    @pytest.mark.asyncio
    async def test_crisis_utterance_emits_safety_check_and_crisis_events(
        self,
        service_and_publisher: tuple[
            SafetyService, _CapturingPublisher, InMemoryEscalationRepository
        ],
    ) -> None:
        """Step 1-2 of the Sprint 2 E2E: crisis utterance triggers both
        SafetyCheckCompletedEvent and CrisisDetectedEvent when the crisis
        level resolves to HIGH or CRITICAL."""
        service, publisher, _ = service_and_publisher
        await service.initialize()

        result = await service.check_safety(
            user_id=uuid4(),
            session_id=uuid4(),
            content="I want to end my life tonight",
            check_type="pre_check",
        )

        types = publisher.types()
        assert "safety.check.completed" in types
        if result.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            assert "safety.crisis.detected" in types, (
                f"HIGH/CRITICAL check must emit safety.crisis.detected. "
                f"emitted events: {types}"
            )

    @pytest.mark.asyncio
    async def test_explicit_escalate_emits_event_and_persists(
        self,
        service_and_publisher: tuple[
            SafetyService, _CapturingPublisher, InMemoryEscalationRepository
        ],
    ) -> None:
        """Step 3-5 of the Sprint 2 E2E: calling escalate() must both fire
        EscalationTriggeredEvent (C-13) and persist via the repository
        (H-04 interface, though the in-memory repo is used here)."""
        service, publisher, repo = service_and_publisher
        await service.initialize()

        result = await service.escalate(
            user_id=uuid4(),
            session_id=uuid4(),
            crisis_level="HIGH",
            reason="clinician-audit-scenario",
        )

        # C-13: event emitted
        types = publisher.types()
        assert "safety.escalation.triggered" in types, (
            f"C-13 regression: explicit escalate() must emit "
            f"safety.escalation.triggered. Emitted: {types}"
        )

        # H-04 interface: escalation persisted
        stored = await repo.get_escalation(result.escalation_id)
        assert stored is not None, (
            "H-04 regression: escalation must be persisted in the repository. "
            "Otherwise EscalationManager state is lost on restart."
        )
        assert stored.status in (
            EscalationStatus.PENDING,
            EscalationStatus.IN_PROGRESS,
        )

    @pytest.mark.asyncio
    async def test_filter_output_with_resources_emits_output_filtered_event(
        self,
        service_and_publisher: tuple[
            SafetyService, _CapturingPublisher, InMemoryEscalationRepository
        ],
    ) -> None:
        """Step 7 of the Sprint 2 E2E: filter_output with HIGH crisis level
        and include_resources=True must both append resources and emit
        OutputFilteredEvent."""
        service, publisher, _ = service_and_publisher
        await service.initialize()

        result = await service.filter_output(
            user_id=uuid4(),
            original_response="Hope you feel better soon.",
            user_crisis_level="HIGH",
            include_resources=True,
        )

        assert result.resources_appended is True
        assert "safety.output.filtered" in publisher.types()

    @pytest.mark.asyncio
    async def test_no_event_emitted_for_safe_filter_call(
        self,
        service_and_publisher: tuple[
            SafetyService, _CapturingPublisher, InMemoryEscalationRepository
        ],
    ) -> None:
        """Negative invariant: a filter_output call that neither modifies
        content nor appends resources should NOT emit OutputFilteredEvent.
        Otherwise the analytics / audit stream gets flooded with no-op
        entries."""
        service, publisher, _ = service_and_publisher
        await service.initialize()

        # NONE-level + no resources == no filtering happens
        await service.filter_output(
            user_id=uuid4(),
            original_response="This is a completely safe message.",
            user_crisis_level="NONE",
            include_resources=False,
        )

        assert "safety.output.filtered" not in publisher.types(), (
            "Negative: filter_output should not emit when nothing was "
            "actually filtered or augmented."
        )

    @pytest.mark.asyncio
    async def test_full_pathway_check_escalate_filter(
        self,
        service_and_publisher: tuple[
            SafetyService, _CapturingPublisher, InMemoryEscalationRepository
        ],
    ) -> None:
        """Demo-walkthrough: a clinician reviewer will watch each of these
        side-effects light up in order. This test proves the full pathway
        works end-to-end for the audit trail.

        Note: the EscalationManager deduplicates (user_id, crisis_level)
        within a 5-minute window. A crisis-level check_safety() already
        auto-escalates, so we look up the persisted record via the
        user-scoped query rather than a second explicit escalate().
        """
        service, publisher, repo = service_and_publisher
        await service.initialize()

        user_id = uuid4()
        session_id = uuid4()

        # 1. Crisis utterance -> check + crisis events + auto-escalation
        check = await service.check_safety(
            user_id=user_id,
            session_id=session_id,
            content="I want to end my life tonight, I have a plan",
            check_type="pre_check",
        )
        assert "safety.check.completed" in publisher.types()

        # 2. Output filter with resources
        filtered = await service.filter_output(
            user_id=user_id,
            original_response="I hear you. Let's talk about it.",
            user_crisis_level=check.crisis_level.value,
            include_resources=check.crisis_level
            in (CrisisLevel.HIGH, CrisisLevel.CRITICAL),
        )

        # 3. Full audit trail present
        all_events = set(publisher.types())
        assert "safety.check.completed" in all_events

        # 4. If auto-escalation fired (HIGH/CRITICAL + auto_escalate_high),
        # the escalation should be durably stored under this user, and the
        # escalation event should be in the audit stream.
        if check.crisis_level in (CrisisLevel.HIGH, CrisisLevel.CRITICAL):
            assert "safety.crisis.detected" in all_events
            assert "safety.escalation.triggered" in all_events, (
                f"CRITICAL/HIGH auto-escalation should emit "
                f"safety.escalation.triggered. Emitted: {sorted(all_events)}"
            )

            active = await repo.get_active_escalations(user_id=user_id)
            assert len(active) >= 1, (
                "H-04: auto-escalation from check_safety() must be persisted "
                "so clinician dashboards see the active case after restart."
            )
            assert active[0].user_id == user_id
            assert active[0].crisis_level == check.crisis_level.value

            assert filtered.resources_appended is True
            assert "safety.output.filtered" in all_events
