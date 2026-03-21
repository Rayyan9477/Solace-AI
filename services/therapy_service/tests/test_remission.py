"""
Solace-AI Therapy Service - Remission Classification Unit Tests.
Verify that treatment response evaluation correctly identifies remission,
deterioration, and edge cases around baseline=0.
"""
from __future__ import annotations

import pytest
from uuid import uuid4

from services.therapy_service.src.domain.treatment_planner import (
    TreatmentPlanner,
    TreatmentPlannerSettings,
    TreatmentPlan,
)
from services.therapy_service.src.schemas import (
    TherapyModality,
    SeverityLevel,
    ResponseStatus,
)


class TestRemissionClassification:
    """Verify remission is detected when current PHQ-9 <= 4, regardless of percentage drop."""

    def setup_method(self) -> None:
        self.planner = TreatmentPlanner(TreatmentPlannerSettings())

    def _create_plan_with_baseline(self, baseline: int) -> TreatmentPlan:
        """Helper to create a plan and establish baseline PHQ-9."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=baseline,
        )
        return plan

    def test_phq9_score_4_is_remission(self) -> None:
        """current<=4 should return REMISSION regardless of percentage."""
        plan = self._create_plan_with_baseline(16)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=4)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_phq9_score_3_is_remission(self) -> None:
        """current<=4 (score=3) should return REMISSION."""
        plan = self._create_plan_with_baseline(14)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=3)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_phq9_score_0_is_remission(self) -> None:
        """current=0 should return REMISSION."""
        plan = self._create_plan_with_baseline(10)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=0)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_phq9_score_1_is_remission(self) -> None:
        """current=1 should return REMISSION."""
        plan = self._create_plan_with_baseline(12)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=1)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_phq9_score_4_from_low_baseline_is_remission(self) -> None:
        """Even with a baseline of 5, current=4 should be REMISSION (only 20% drop)."""
        plan = self._create_plan_with_baseline(5)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=4)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_phq9_score_5_is_not_remission(self) -> None:
        """current=5 should NOT be REMISSION (above threshold)."""
        plan = self._create_plan_with_baseline(16)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=5)
        assert result["response_status"] != ResponseStatus.REMISSION.value

    def test_remission_recommendations_include_maintenance(self) -> None:
        """REMISSION status should recommend transition to maintenance."""
        plan = self._create_plan_with_baseline(15)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=3)
        assert any("maintenance" in r.lower() for r in result["recommendations"])

    def test_remission_recommendations_include_relapse_prevention(self) -> None:
        """REMISSION status should recommend relapse prevention."""
        plan = self._create_plan_with_baseline(15)
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=2)
        assert any("relapse" in r.lower() for r in result["recommendations"])


class TestBaselineZeroDeteriorationEdgeCase:
    """Verify that baseline=0 with current>0 correctly detects DETERIORATING."""

    def setup_method(self) -> None:
        self.planner = TreatmentPlanner(TreatmentPlannerSettings())

    def test_baseline_zero_current_above_zero_is_deteriorating(self) -> None:
        """baseline=0 and current>0 (but >4) should be DETERIORATING."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MINIMAL,
            modality=TherapyModality.CBT,
            phq9_score=0,
        )
        # First update establishes previous=0
        # Second update tests baseline=0, previous=0, current=8
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=8)
        assert result["response_status"] == ResponseStatus.DETERIORATING.value

    def test_baseline_zero_current_5_is_deteriorating(self) -> None:
        """baseline=0 and current=5 should be DETERIORATING (score > 4, so not remission)."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MINIMAL,
            modality=TherapyModality.CBT,
            phq9_score=0,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=5)
        assert result["response_status"] == ResponseStatus.DETERIORATING.value

    def test_baseline_zero_current_4_is_remission(self) -> None:
        """baseline=0 and current=4 should be REMISSION (remission check precedes baseline=0 check)."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MINIMAL,
            modality=TherapyModality.CBT,
            phq9_score=0,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=4)
        assert result["response_status"] == ResponseStatus.REMISSION.value

    def test_deterioration_recommends_safety_assessment(self) -> None:
        """DETERIORATING status should recommend safety assessment."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MINIMAL,
            modality=TherapyModality.CBT,
            phq9_score=0,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=10)
        assert any("safety" in r.lower() for r in result["recommendations"])

    def test_deterioration_recommends_clinician_consultation(self) -> None:
        """DETERIORATING status should recommend human clinician consultation."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MINIMAL,
            modality=TherapyModality.CBT,
            phq9_score=0,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=12)
        assert any("clinician" in r.lower() for r in result["recommendations"])


class TestTreatmentResponseEvaluation:
    """Additional tests for treatment response categories beyond remission."""

    def setup_method(self) -> None:
        self.planner = TreatmentPlanner(TreatmentPlannerSettings())

    def test_responding_status_50_percent_reduction(self) -> None:
        """>=50% reduction from baseline should be RESPONDING."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=20,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=9)
        assert result["response_status"] == ResponseStatus.RESPONDING.value

    def test_partial_response_25_to_50_percent(self) -> None:
        """25-50% reduction from baseline should be PARTIAL_RESPONSE."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            phq9_score=20,
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=14)
        assert result["response_status"] == ResponseStatus.PARTIAL_RESPONSE.value

    def test_no_baseline_returns_not_started(self) -> None:
        """No baseline PHQ-9 should return NOT_STARTED."""
        plan = self.planner.create_plan(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            # No phq9_score -> baseline_phq9 is None
        )
        result = self.planner.update_outcome_score(plan.plan_id, phq9_score=10)
        assert result["response_status"] == ResponseStatus.NOT_STARTED.value
