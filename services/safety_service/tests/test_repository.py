"""
Unit tests for Solace-AI Safety Service Repository Layer.
Tests repository abstractions and in-memory implementations.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from uuid import uuid4
from services.safety_service.src.infrastructure.repository import (
    RepositoryError, EntityNotFoundError, DuplicateEntityError,
    SafetyRepositoryFactory, get_repository_factory, reset_repositories,
)
from services.safety_service.tests.fixtures import (
    InMemorySafetyAssessmentRepository, InMemorySafetyPlanRepository,
    InMemorySafetyIncidentRepository, InMemoryUserRiskProfileRepository,
)
from services.safety_service.src.domain.entities import (
    SafetyAssessment, AssessmentType, SafetyPlan, SafetyPlanStatus,
    WarningSign, CopingStrategy, EmergencyContact,
    SafetyIncident, IncidentSeverity, IncidentStatus, UserRiskProfile,
)


class TestEntityNotFoundError:
    """Tests for EntityNotFoundError."""

    def test_error_message(self) -> None:
        """Test error message format."""
        entity_id = uuid4()
        error = EntityNotFoundError("SafetyPlan", entity_id)
        assert "SafetyPlan" in str(error)
        assert str(entity_id) in str(error)


class TestInMemorySafetyAssessmentRepository:
    """Tests for InMemorySafetyAssessmentRepository."""

    @pytest.fixture
    def repo(self) -> InMemorySafetyAssessmentRepository:
        """Create test repository."""
        return InMemorySafetyAssessmentRepository()

    @pytest.fixture
    def assessment(self) -> SafetyAssessment:
        """Create test assessment."""
        return SafetyAssessment(
            user_id=uuid4(),
            session_id=uuid4(),
            content_assessed="Test content",
            risk_score=Decimal("0.3"),
            crisis_level="LOW",
        )

    @pytest.mark.asyncio
    async def test_save(self, repo: InMemorySafetyAssessmentRepository,
                        assessment: SafetyAssessment) -> None:
        """Test saving assessment."""
        saved = await repo.save(assessment)
        assert saved.assessment_id == assessment.assessment_id

    @pytest.mark.asyncio
    async def test_get_by_id(self, repo: InMemorySafetyAssessmentRepository,
                             assessment: SafetyAssessment) -> None:
        """Test getting assessment by ID."""
        await repo.save(assessment)
        retrieved = await repo.get_by_id(assessment.assessment_id)
        assert retrieved is not None
        assert retrieved.assessment_id == assessment.assessment_id

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test getting non-existent assessment."""
        result = await repo.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_user(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test getting assessments by user."""
        user_id = uuid4()
        for i in range(3):
            assessment = SafetyAssessment(
                user_id=user_id,
                content_assessed=f"Test {i}",
            )
            await repo.save(assessment)
        results = await repo.get_by_user(user_id)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_by_user_with_limit(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test getting assessments with limit."""
        user_id = uuid4()
        for i in range(5):
            await repo.save(SafetyAssessment(user_id=user_id, content_assessed=f"Test {i}"))
        results = await repo.get_by_user(user_id, limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_by_session(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test getting assessments by session."""
        session_id = uuid4()
        await repo.save(SafetyAssessment(
            user_id=uuid4(), session_id=session_id, content_assessed="Test"
        ))
        results = await repo.get_by_session(session_id)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_recent(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test getting recent assessments."""
        user_id = uuid4()
        await repo.save(SafetyAssessment(user_id=user_id, content_assessed="Recent"))
        results = await repo.get_recent(user_id, hours=24)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_count_by_crisis_level(self, repo: InMemorySafetyAssessmentRepository) -> None:
        """Test counting assessments by crisis level."""
        user_id = uuid4()
        await repo.save(SafetyAssessment(
            user_id=user_id, content_assessed="Test", crisis_level="HIGH"
        ))
        await repo.save(SafetyAssessment(
            user_id=user_id, content_assessed="Test", crisis_level="HIGH"
        ))
        await repo.save(SafetyAssessment(
            user_id=user_id, content_assessed="Test", crisis_level="LOW"
        ))
        count = await repo.count_by_crisis_level(user_id, "HIGH")
        assert count == 2


class TestInMemorySafetyPlanRepository:
    """Tests for InMemorySafetyPlanRepository."""

    @pytest.fixture
    def repo(self) -> InMemorySafetyPlanRepository:
        """Create test repository."""
        return InMemorySafetyPlanRepository()

    @pytest.fixture
    def plan(self) -> SafetyPlan:
        """Create test plan."""
        return SafetyPlan(user_id=uuid4())

    @pytest.mark.asyncio
    async def test_save(self, repo: InMemorySafetyPlanRepository, plan: SafetyPlan) -> None:
        """Test saving plan."""
        saved = await repo.save(plan)
        assert saved.plan_id == plan.plan_id

    @pytest.mark.asyncio
    async def test_get_by_id(self, repo: InMemorySafetyPlanRepository, plan: SafetyPlan) -> None:
        """Test getting plan by ID."""
        await repo.save(plan)
        retrieved = await repo.get_by_id(plan.plan_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_by_user(self, repo: InMemorySafetyPlanRepository) -> None:
        """Test getting plans by user."""
        user_id = uuid4()
        await repo.save(SafetyPlan(user_id=user_id))
        await repo.save(SafetyPlan(user_id=user_id))
        results = await repo.get_by_user(user_id)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_active_by_user(self, repo: InMemorySafetyPlanRepository) -> None:
        """Test getting active plan for user."""
        user_id = uuid4()
        plan = SafetyPlan(user_id=user_id, status=SafetyPlanStatus.ACTIVE)
        await repo.save(plan)
        result = await repo.get_active_by_user(user_id)
        assert result is not None
        assert result.status == SafetyPlanStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_active_by_user_none(self, repo: InMemorySafetyPlanRepository) -> None:
        """Test getting active plan when none exists."""
        result = await repo.get_active_by_user(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_update(self, repo: InMemorySafetyPlanRepository, plan: SafetyPlan) -> None:
        """Test updating plan."""
        await repo.save(plan)
        plan.status = SafetyPlanStatus.ACTIVE
        updated = await repo.update(plan)
        assert updated.status == SafetyPlanStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_not_found(self, repo: InMemorySafetyPlanRepository) -> None:
        """Test updating non-existent plan."""
        plan = SafetyPlan(user_id=uuid4())
        with pytest.raises(EntityNotFoundError):
            await repo.update(plan)

    @pytest.mark.asyncio
    async def test_delete(self, repo: InMemorySafetyPlanRepository, plan: SafetyPlan) -> None:
        """Test deleting plan."""
        await repo.save(plan)
        result = await repo.delete(plan.plan_id)
        assert result is True
        assert await repo.get_by_id(plan.plan_id) is None

    @pytest.mark.asyncio
    async def test_delete_not_found(self, repo: InMemorySafetyPlanRepository) -> None:
        """Test deleting non-existent plan."""
        result = await repo.delete(uuid4())
        assert result is False


class TestInMemorySafetyIncidentRepository:
    """Tests for InMemorySafetyIncidentRepository."""

    @pytest.fixture
    def repo(self) -> InMemorySafetyIncidentRepository:
        """Create test repository."""
        return InMemorySafetyIncidentRepository()

    @pytest.fixture
    def incident(self) -> SafetyIncident:
        """Create test incident."""
        return SafetyIncident(
            user_id=uuid4(),
            severity=IncidentSeverity.HIGH,
            crisis_level="HIGH",
            description="Test incident",
        )

    @pytest.mark.asyncio
    async def test_save(self, repo: InMemorySafetyIncidentRepository,
                        incident: SafetyIncident) -> None:
        """Test saving incident."""
        saved = await repo.save(incident)
        assert saved.incident_id == incident.incident_id

    @pytest.mark.asyncio
    async def test_get_by_id(self, repo: InMemorySafetyIncidentRepository,
                             incident: SafetyIncident) -> None:
        """Test getting incident by ID."""
        await repo.save(incident)
        retrieved = await repo.get_by_id(incident.incident_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_by_user(self, repo: InMemorySafetyIncidentRepository) -> None:
        """Test getting incidents by user."""
        user_id = uuid4()
        for _ in range(3):
            await repo.save(SafetyIncident(
                user_id=user_id, severity=IncidentSeverity.HIGH,
                crisis_level="HIGH", description="Test"
            ))
        results = await repo.get_by_user(user_id)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_open(self, repo: InMemorySafetyIncidentRepository) -> None:
        """Test getting open incidents."""
        await repo.save(SafetyIncident(
            user_id=uuid4(), severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test", status=IncidentStatus.OPEN
        ))
        await repo.save(SafetyIncident(
            user_id=uuid4(), severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test", status=IncidentStatus.RESOLVED
        ))
        results = await repo.get_open()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_open_by_user(self, repo: InMemorySafetyIncidentRepository) -> None:
        """Test getting open incidents for specific user."""
        user_id = uuid4()
        await repo.save(SafetyIncident(
            user_id=user_id, severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test", status=IncidentStatus.OPEN
        ))
        await repo.save(SafetyIncident(
            user_id=uuid4(), severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test", status=IncidentStatus.OPEN
        ))
        results = await repo.get_open(user_id)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_by_status(self, repo: InMemorySafetyIncidentRepository) -> None:
        """Test getting incidents by status."""
        await repo.save(SafetyIncident(
            user_id=uuid4(), severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test", status=IncidentStatus.ACKNOWLEDGED
        ))
        results = await repo.get_by_status(IncidentStatus.ACKNOWLEDGED)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_update(self, repo: InMemorySafetyIncidentRepository,
                          incident: SafetyIncident) -> None:
        """Test updating incident."""
        await repo.save(incident)
        incident.status = IncidentStatus.RESOLVED
        updated = await repo.update(incident)
        assert updated.status == IncidentStatus.RESOLVED

    @pytest.mark.asyncio
    async def test_update_not_found(self, repo: InMemorySafetyIncidentRepository) -> None:
        """Test updating non-existent incident."""
        incident = SafetyIncident(
            user_id=uuid4(), severity=IncidentSeverity.HIGH,
            crisis_level="HIGH", description="Test"
        )
        with pytest.raises(EntityNotFoundError):
            await repo.update(incident)


class TestInMemoryUserRiskProfileRepository:
    """Tests for InMemoryUserRiskProfileRepository."""

    @pytest.fixture
    def repo(self) -> InMemoryUserRiskProfileRepository:
        """Create test repository."""
        return InMemoryUserRiskProfileRepository()

    @pytest.mark.asyncio
    async def test_save(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test saving profile."""
        profile = UserRiskProfile(user_id=uuid4())
        saved = await repo.save(profile)
        assert saved.user_id == profile.user_id

    @pytest.mark.asyncio
    async def test_get_by_user(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test getting profile by user."""
        user_id = uuid4()
        await repo.save(UserRiskProfile(user_id=user_id))
        retrieved = await repo.get_by_user(user_id)
        assert retrieved is not None
        assert retrieved.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test get_or_create with existing profile."""
        user_id = uuid4()
        await repo.save(UserRiskProfile(user_id=user_id, baseline_risk_level="LOW"))
        profile = await repo.get_or_create(user_id)
        assert profile.baseline_risk_level == "LOW"

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test get_or_create creates new profile."""
        user_id = uuid4()
        profile = await repo.get_or_create(user_id)
        assert profile.user_id == user_id
        assert profile.baseline_risk_level == "NONE"

    @pytest.mark.asyncio
    async def test_update(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test updating profile."""
        user_id = uuid4()
        await repo.save(UserRiskProfile(user_id=user_id))
        profile = await repo.get_by_user(user_id)
        profile.high_risk_flag = True
        updated = await repo.update(profile)
        assert updated.high_risk_flag is True

    @pytest.mark.asyncio
    async def test_get_high_risk_users(self, repo: InMemoryUserRiskProfileRepository) -> None:
        """Test getting high risk users."""
        await repo.save(UserRiskProfile(user_id=uuid4(), high_risk_flag=True))
        await repo.save(UserRiskProfile(user_id=uuid4(), high_risk_flag=False))
        await repo.save(UserRiskProfile(user_id=uuid4(), high_risk_flag=True))
        results = await repo.get_high_risk_users()
        assert len(results) == 2


class TestSafetyRepositoryFactory:
    """Tests for SafetyRepositoryFactory."""

    def test_get_assessment_repository_raises_without_postgres(self) -> None:
        """Test getting assessment repository raises without PostgreSQL."""
        factory = SafetyRepositoryFactory()
        with pytest.raises(RepositoryError, match="PostgreSQL is required"):
            factory.get_assessment_repository()

    def test_get_plan_repository_raises_without_postgres(self) -> None:
        """Test getting plan repository raises without PostgreSQL."""
        factory = SafetyRepositoryFactory()
        with pytest.raises(RepositoryError, match="PostgreSQL is required"):
            factory.get_plan_repository()

    def test_get_incident_repository_raises_without_postgres(self) -> None:
        """Test getting incident repository raises without PostgreSQL."""
        factory = SafetyRepositoryFactory()
        with pytest.raises(RepositoryError, match="PostgreSQL is required"):
            factory.get_incident_repository()

    def test_get_profile_repository_raises_without_postgres(self) -> None:
        """Test getting profile repository raises without PostgreSQL."""
        factory = SafetyRepositoryFactory()
        with pytest.raises(RepositoryError, match="PostgreSQL is required"):
            factory.get_profile_repository()


class TestRepositoryFactorySingleton:
    """Tests for repository factory singleton."""

    def setup_method(self) -> None:
        """Reset repositories before each test."""
        reset_repositories()

    def teardown_method(self) -> None:
        """Reset repositories after each test."""
        reset_repositories()

    def test_get_repository_factory_raises_without_postgres(self) -> None:
        """Test singleton raises RepositoryError when no PostgreSQL is configured.

        Must disable the use_connection_pool_manager feature flag so the code
        reaches the 'no database configured' branch instead of creating a
        PostgresSafetyRepositoryFactory via ConnectionPoolManager.
        """
        from solace_infrastructure.feature_flags import FeatureFlags
        FeatureFlags.disable_flag("use_connection_pool_manager")
        try:
            with pytest.raises(RepositoryError, match="PostgreSQL is required in production"):
                get_repository_factory()
        finally:
            FeatureFlags.enable_flag("use_connection_pool_manager")

    def test_reset_repositories(self) -> None:
        """Test resetting repositories clears state."""
        reset_repositories()
        # After reset, calling get_repository_factory should raise again
        # (with feature flag disabled so no ConnectionPoolManager path)
        from solace_infrastructure.feature_flags import FeatureFlags
        FeatureFlags.disable_flag("use_connection_pool_manager")
        try:
            with pytest.raises(RepositoryError, match="PostgreSQL is required in production"):
                get_repository_factory()
        finally:
            FeatureFlags.enable_flag("use_connection_pool_manager")
