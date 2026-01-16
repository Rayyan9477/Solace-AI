"""
Unit tests for Repository Infrastructure.
Tests TreatmentPlanRepository, TherapySessionRepository, and related repositories.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
import pytest

from services.therapy_service.src.infrastructure.repository import (
    TreatmentPlanRepository, TherapySessionRepository,
    TechniqueRepository, OutcomeMeasureRepository, UnitOfWork,
)
from services.therapy_service.src.domain.entities import (
    TreatmentPlanEntity, TherapySessionEntity,
)
from services.therapy_service.src.domain.value_objects import (
    Technique, OutcomeMeasure,
)
from services.therapy_service.src.schemas import (
    TherapyModality, TechniqueCategory, SeverityLevel,
    OutcomeInstrument, TreatmentPhase,
)


@pytest.fixture
def treatment_plan_repo() -> TreatmentPlanRepository:
    """Create treatment plan repository fixture."""
    return TreatmentPlanRepository()


@pytest.fixture
def session_repo() -> TherapySessionRepository:
    """Create therapy session repository fixture."""
    return TherapySessionRepository()


@pytest.fixture
def technique_repo() -> TechniqueRepository:
    """Create technique repository fixture."""
    return TechniqueRepository()


@pytest.fixture
def outcome_repo() -> OutcomeMeasureRepository:
    """Create outcome measure repository fixture."""
    return OutcomeMeasureRepository()


class TestTreatmentPlanRepository:
    """Tests for TreatmentPlanRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test saving and retrieving plan."""
        plan = TreatmentPlanEntity(
            primary_diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            primary_modality=TherapyModality.CBT,
        )
        saved = await treatment_plan_repo.save(plan)
        assert saved.plan_id == plan.plan_id
        retrieved = await treatment_plan_repo.get(plan.plan_id)
        assert retrieved is not None
        assert retrieved.primary_diagnosis == "Depression"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test getting non-existent plan."""
        result = await treatment_plan_repo.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test deleting plan."""
        plan = TreatmentPlanEntity()
        await treatment_plan_repo.save(plan)
        deleted = await treatment_plan_repo.delete(plan.plan_id)
        assert deleted is True
        result = await treatment_plan_repo.get(plan.plan_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test deleting non-existent plan."""
        deleted = await treatment_plan_repo.delete(uuid4())
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_by_user(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test getting plans by user."""
        user_id = uuid4()
        plan1 = TreatmentPlanEntity(user_id=user_id)
        plan2 = TreatmentPlanEntity(user_id=user_id)
        plan3 = TreatmentPlanEntity(user_id=uuid4())
        await treatment_plan_repo.save(plan1)
        await treatment_plan_repo.save(plan2)
        await treatment_plan_repo.save(plan3)
        user_plans = await treatment_plan_repo.get_by_user(user_id)
        assert len(user_plans) == 2

    @pytest.mark.asyncio
    async def test_get_active_by_user(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test getting active plans by user."""
        user_id = uuid4()
        plan1 = TreatmentPlanEntity(user_id=user_id, is_active=True)
        plan2 = TreatmentPlanEntity(user_id=user_id, is_active=False)
        await treatment_plan_repo.save(plan1)
        await treatment_plan_repo.save(plan2)
        active_plans = await treatment_plan_repo.get_active_by_user(user_id)
        assert len(active_plans) == 1

    @pytest.mark.asyncio
    async def test_find_by_criteria(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test finding plans by criteria."""
        user_id = uuid4()
        plan1 = TreatmentPlanEntity(
            user_id=user_id,
            primary_modality=TherapyModality.CBT,
            is_active=True,
            total_sessions_completed=5,
        )
        plan2 = TreatmentPlanEntity(
            user_id=user_id,
            primary_modality=TherapyModality.DBT,
            is_active=True,
        )
        await treatment_plan_repo.save(plan1)
        await treatment_plan_repo.save(plan2)
        cbt_plans = await treatment_plan_repo.find_by_criteria(modality="cbt")
        assert len(cbt_plans) == 1
        session_plans = await treatment_plan_repo.find_by_criteria(min_sessions=3)
        assert len(session_plans) == 1

    @pytest.mark.asyncio
    async def test_count(self, treatment_plan_repo: TreatmentPlanRepository) -> None:
        """Test counting plans."""
        await treatment_plan_repo.save(TreatmentPlanEntity())
        await treatment_plan_repo.save(TreatmentPlanEntity())
        count = await treatment_plan_repo.count()
        assert count == 2


class TestTherapySessionRepository:
    """Tests for TherapySessionRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, session_repo: TherapySessionRepository) -> None:
        """Test saving and retrieving session."""
        session = TherapySessionEntity(session_number=1)
        saved = await session_repo.save(session)
        assert saved.session_id == session.session_id
        retrieved = await session_repo.get(session.session_id)
        assert retrieved is not None
        assert retrieved.session_number == 1

    @pytest.mark.asyncio
    async def test_get_by_user(self, session_repo: TherapySessionRepository) -> None:
        """Test getting sessions by user."""
        user_id = uuid4()
        s1 = TherapySessionEntity(user_id=user_id, session_number=1)
        s2 = TherapySessionEntity(user_id=user_id, session_number=2)
        s3 = TherapySessionEntity(user_id=uuid4(), session_number=1)
        await session_repo.save(s1)
        await session_repo.save(s2)
        await session_repo.save(s3)
        user_sessions = await session_repo.get_by_user(user_id)
        assert len(user_sessions) == 2

    @pytest.mark.asyncio
    async def test_get_by_plan(self, session_repo: TherapySessionRepository) -> None:
        """Test getting sessions by plan."""
        plan_id = uuid4()
        s1 = TherapySessionEntity(treatment_plan_id=plan_id, session_number=1)
        s2 = TherapySessionEntity(treatment_plan_id=plan_id, session_number=2)
        await session_repo.save(s1)
        await session_repo.save(s2)
        plan_sessions = await session_repo.get_by_plan(plan_id)
        assert len(plan_sessions) == 2
        assert plan_sessions[0].session_number == 1

    @pytest.mark.asyncio
    async def test_get_active_by_user(self, session_repo: TherapySessionRepository) -> None:
        """Test getting active session."""
        user_id = uuid4()
        active_session = TherapySessionEntity(user_id=user_id)
        ended_session = TherapySessionEntity(user_id=user_id)
        ended_session.end_session()
        await session_repo.save(active_session)
        await session_repo.save(ended_session)
        active = await session_repo.get_active_by_user(user_id)
        assert active is not None
        assert active.session_id == active_session.session_id

    @pytest.mark.asyncio
    async def test_get_recent_sessions(self, session_repo: TherapySessionRepository) -> None:
        """Test getting recent sessions."""
        user_id = uuid4()
        recent = TherapySessionEntity(user_id=user_id)
        old = TherapySessionEntity(user_id=user_id)
        old.started_at = datetime.now(timezone.utc) - timedelta(days=60)
        await session_repo.save(recent)
        await session_repo.save(old)
        recent_sessions = await session_repo.get_recent_sessions(user_id, days=30)
        assert len(recent_sessions) == 1

    @pytest.mark.asyncio
    async def test_delete(self, session_repo: TherapySessionRepository) -> None:
        """Test deleting session."""
        session = TherapySessionEntity()
        await session_repo.save(session)
        deleted = await session_repo.delete(session.session_id)
        assert deleted is True
        result = await session_repo.get(session.session_id)
        assert result is None


class TestTechniqueRepository:
    """Tests for TechniqueRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, technique_repo: TechniqueRepository) -> None:
        """Test saving and retrieving technique."""
        technique = Technique(
            technique_id=uuid4(),
            name="Thought Record",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description="Test",
        )
        saved = await technique_repo.save(technique)
        assert saved.technique_id == technique.technique_id
        retrieved = await technique_repo.get(technique.technique_id)
        assert retrieved is not None
        assert retrieved.name == "Thought Record"

    @pytest.mark.asyncio
    async def test_get_by_name(self, technique_repo: TechniqueRepository) -> None:
        """Test getting technique by name."""
        technique = Technique(
            technique_id=uuid4(),
            name="Grounding Exercise",
            modality=TherapyModality.DBT,
            category=TechniqueCategory.DISTRESS_TOLERANCE,
            description="Test",
        )
        await technique_repo.save(technique)
        retrieved = await technique_repo.get_by_name("grounding exercise")
        assert retrieved is not None
        assert retrieved.technique_id == technique.technique_id

    @pytest.mark.asyncio
    async def test_get_by_modality(self, technique_repo: TechniqueRepository) -> None:
        """Test getting techniques by modality."""
        t1 = Technique(
            technique_id=uuid4(),
            name="CBT Technique",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.COGNITIVE_RESTRUCTURING,
            description="Test",
        )
        t2 = Technique(
            technique_id=uuid4(),
            name="DBT Technique",
            modality=TherapyModality.DBT,
            category=TechniqueCategory.DISTRESS_TOLERANCE,
            description="Test",
        )
        await technique_repo.save(t1)
        await technique_repo.save(t2)
        cbt_techniques = await technique_repo.get_by_modality("cbt")
        assert len(cbt_techniques) == 1
        assert cbt_techniques[0].name == "CBT Technique"

    @pytest.mark.asyncio
    async def test_search(self, technique_repo: TechniqueRepository) -> None:
        """Test searching techniques."""
        t1 = Technique(
            technique_id=uuid4(),
            name="Test1",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.RELAXATION,
            description="Test",
            duration_minutes=10,
        )
        t2 = Technique(
            technique_id=uuid4(),
            name="Test2",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.RELAXATION,
            description="Test",
            duration_minutes=30,
        )
        await technique_repo.save(t1)
        await technique_repo.save(t2)
        short_techniques = await technique_repo.search(max_duration=15)
        assert len(short_techniques) == 1

    @pytest.mark.asyncio
    async def test_delete(self, technique_repo: TechniqueRepository) -> None:
        """Test deleting technique."""
        technique = Technique(
            technique_id=uuid4(),
            name="To Delete",
            modality=TherapyModality.CBT,
            category=TechniqueCategory.RELAXATION,
            description="Test",
        )
        await technique_repo.save(technique)
        deleted = await technique_repo.delete(technique.technique_id)
        assert deleted is True
        result = await technique_repo.get(technique.technique_id)
        assert result is None


class TestOutcomeMeasureRepository:
    """Tests for OutcomeMeasureRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get(self, outcome_repo: OutcomeMeasureRepository) -> None:
        """Test saving and retrieving measure."""
        user_id = uuid4()
        measure = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=14,
        )
        saved = await outcome_repo.save(measure, user_id)
        assert saved.measure_id == measure.measure_id
        retrieved = await outcome_repo.get(measure.measure_id)
        assert retrieved is not None
        assert retrieved.raw_score == 14

    @pytest.mark.asyncio
    async def test_get_by_session(self, outcome_repo: OutcomeMeasureRepository) -> None:
        """Test getting measures by session."""
        user_id = uuid4()
        session_id = uuid4()
        m1 = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.PHQ9,
            raw_score=10,
            session_id=session_id,
        )
        m2 = OutcomeMeasure(
            measure_id=uuid4(),
            instrument=OutcomeInstrument.GAD7,
            raw_score=8,
            session_id=session_id,
        )
        await outcome_repo.save(m1, user_id)
        await outcome_repo.save(m2, user_id)
        session_measures = await outcome_repo.get_by_session(session_id)
        assert len(session_measures) == 2

    @pytest.mark.asyncio
    async def test_get_history(self, outcome_repo: OutcomeMeasureRepository) -> None:
        """Test getting measure history."""
        user_id = uuid4()
        m1 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=16)
        m2 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=12)
        m3 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.GAD7, raw_score=10)
        await outcome_repo.save(m1, user_id)
        await outcome_repo.save(m2, user_id)
        await outcome_repo.save(m3, user_id)
        phq9_history = await outcome_repo.get_history(user_id, "phq9")
        assert len(phq9_history) == 2

    @pytest.mark.asyncio
    async def test_get_latest(self, outcome_repo: OutcomeMeasureRepository) -> None:
        """Test getting latest measure."""
        user_id = uuid4()
        older_time = datetime.now(timezone.utc) - timedelta(hours=1)
        newer_time = datetime.now(timezone.utc)
        m1 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=16, recorded_at=older_time)
        m2 = OutcomeMeasure(measure_id=uuid4(), instrument=OutcomeInstrument.PHQ9, raw_score=12, recorded_at=newer_time)
        await outcome_repo.save(m1, user_id)
        await outcome_repo.save(m2, user_id)
        latest = await outcome_repo.get_latest(user_id, "phq9")
        assert latest is not None
        assert latest.raw_score == 12


class TestUnitOfWork:
    """Tests for UnitOfWork."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test unit of work context manager."""
        plan_repo = TreatmentPlanRepository()
        session_repo = TherapySessionRepository()
        technique_repo = TechniqueRepository()
        outcome_repo = OutcomeMeasureRepository()
        async with UnitOfWork(plan_repo, session_repo, technique_repo, outcome_repo) as uow:
            plan = TreatmentPlanEntity()
            await uow.treatment_plans.save(plan)
        retrieved = await plan_repo.get(plan.plan_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_rollback_on_exception(self) -> None:
        """Test rollback on exception."""
        plan_repo = TreatmentPlanRepository()
        session_repo = TherapySessionRepository()
        technique_repo = TechniqueRepository()
        outcome_repo = OutcomeMeasureRepository()
        try:
            async with UnitOfWork(plan_repo, session_repo, technique_repo, outcome_repo):
                raise ValueError("Test error")
        except ValueError:
            pass
