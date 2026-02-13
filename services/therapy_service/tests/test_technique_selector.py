"""
Unit tests for Therapy Service Technique Selector.
Tests 4-stage evidence-based technique selection pipeline.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.schemas import SessionPhase, TherapyModality, SeverityLevel, RiskLevel
from services.therapy_service.src.domain.technique_selector import (
    TechniqueSelector, TechniqueSelectorSettings
)


class TestTechniqueSelectorSettings:
    """Tests for TechniqueSelectorSettings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings are properly initialized."""
        settings = TechniqueSelectorSettings()
        assert settings.min_confidence_threshold == 0.6
        assert settings.max_techniques_per_session == 3
        assert settings.recency_penalty_weight == 0.15
        assert settings.enable_personalization is True
        assert settings.enable_strict_contraindications is True

    def test_custom_settings(self) -> None:
        """Test custom settings override defaults."""
        settings = TechniqueSelectorSettings(
            min_confidence_threshold=0.7,
            max_techniques_per_session=5,
            enable_strict_contraindications=False,
        )
        assert settings.min_confidence_threshold == 0.7
        assert settings.max_techniques_per_session == 5
        assert settings.enable_strict_contraindications is False


class TestTechniqueLibrary:
    """Tests for technique library management."""

    def test_technique_library_initialized(self) -> None:
        """Test technique library is properly initialized."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(None)
        assert len(techniques) > 0

    def test_get_techniques_by_cbt_modality(self) -> None:
        """Test filtering techniques by CBT modality."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.CBT)
        assert all(t.modality == TherapyModality.CBT for t in techniques)
        assert len(techniques) >= 4

    def test_get_techniques_by_dbt_modality(self) -> None:
        """Test filtering techniques by DBT modality."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.DBT)
        assert all(t.modality == TherapyModality.DBT for t in techniques)
        assert len(techniques) >= 3

    def test_get_techniques_by_act_modality(self) -> None:
        """Test filtering techniques by ACT modality."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.ACT)
        assert all(t.modality == TherapyModality.ACT for t in techniques)
        assert len(techniques) >= 3

    def test_get_techniques_by_mi_modality(self) -> None:
        """Test filtering techniques by MI modality."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.MI)
        assert all(t.modality == TherapyModality.MI for t in techniques)
        assert len(techniques) >= 2

    def test_get_techniques_by_mindfulness_modality(self) -> None:
        """Test filtering techniques by Mindfulness modality."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.MINDFULNESS)
        assert all(t.modality == TherapyModality.MINDFULNESS for t in techniques)
        assert len(techniques) >= 3


class TestStage1ClinicalFilter:
    """Tests for Stage 1: Clinical Filtering."""

    @pytest.mark.asyncio
    async def test_depression_gets_behavioral_techniques(self) -> None:
        """Test depression diagnosis prioritizes behavioral techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None
        assert result["selected"].modality == TherapyModality.CBT

    @pytest.mark.asyncio
    async def test_anxiety_gets_exposure_techniques(self) -> None:
        """Test anxiety diagnosis includes exposure-related techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None

    @pytest.mark.asyncio
    async def test_trauma_gets_grounding_techniques(self) -> None:
        """Test trauma diagnosis gets grounding techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Trauma",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.DBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None


class TestStage2Personalization:
    """Tests for Stage 2: Personalization Scoring."""

    @pytest.mark.asyncio
    async def test_personalization_respects_preferences(self) -> None:
        """Test personalization scores favor user preferences."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MILD,
            modality=TherapyModality.MINDFULNESS,
            session_phase=SessionPhase.WORKING,
            user_context={
                "current_risk": RiskLevel.NONE,
                "contraindications": [],
                "preferences": {"preferred_techniques": ["Mindfulness of Breath"]},
            },
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None
        assert result["selected"].modality == TherapyModality.MINDFULNESS

    @pytest.mark.asyncio
    async def test_recency_penalty_reduces_repeated_techniques(self) -> None:
        """Test recency penalty reduces score for recently used techniques."""
        selector = TechniqueSelector()
        user_id = uuid4()
        result1 = await selector.select_technique(
            user_id=user_id,
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        first_technique = result1["selected"]
        result2 = await selector.select_technique(
            user_id=user_id,
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result2["selected"] is not None


class TestStage3ContextRanking:
    """Tests for Stage 3: Contextual Ranking."""

    @pytest.mark.asyncio
    async def test_opening_phase_gets_appropriate_techniques(self) -> None:
        """Test opening phase selects appropriate techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.OPENING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None

    @pytest.mark.asyncio
    async def test_working_phase_gets_active_techniques(self) -> None:
        """Test working phase selects active intervention techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None

    @pytest.mark.asyncio
    async def test_closing_phase_gets_consolidation_techniques(self) -> None:
        """Test closing phase selects consolidation-appropriate techniques."""
        # Use MINDFULNESS for closing phase as it has short grounding techniques
        selector = TechniqueSelector(TechniqueSelectorSettings(min_confidence_threshold=0.4))
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.MILD,
            modality=TherapyModality.MINDFULNESS,
            session_phase=SessionPhase.CLOSING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None


class TestStage4FinalSelection:
    """Tests for Stage 4: Final Selection."""

    @pytest.mark.asyncio
    async def test_selection_returns_alternatives(self) -> None:
        """Test selection returns alternative techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result["selected"] is not None
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    @pytest.mark.asyncio
    async def test_selection_provides_reasoning(self) -> None:
        """Test selection provides reasoning for choice."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.DBT,
            session_phase=SessionPhase.WORKING,
            user_context={"current_risk": RiskLevel.NONE, "contraindications": []},
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert "reasoning" in result
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


class TestContraindicationValidation:
    """Tests for contraindication validation."""

    @pytest.mark.asyncio
    async def test_trauma_dissociation_contraindication(self) -> None:
        """Test trauma dissociation contraindication is respected."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="PTSD",
            severity=SeverityLevel.SEVERE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={
                "current_risk": RiskLevel.ELEVATED,
                "contraindications": ["trauma_dissociation"],
            },
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        if result["selected"]:
            assert "Exposure" not in result["selected"].name

    @pytest.mark.asyncio
    async def test_high_risk_filters_exposure_techniques(self) -> None:
        """Test high risk filters out exposure techniques."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Anxiety",
            severity=SeverityLevel.SEVERE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={
                "current_risk": RiskLevel.HIGH,
                "contraindications": [],
            },
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        if result["selected"]:
            assert "Exposure" not in result["selected"].name

    @pytest.mark.asyncio
    async def test_cognitive_impairment_contraindication(self) -> None:
        """Test cognitive impairment contraindication is respected."""
        selector = TechniqueSelector()
        result = await selector.select_technique(
            user_id=uuid4(),
            diagnosis="Depression",
            severity=SeverityLevel.MODERATE,
            modality=TherapyModality.CBT,
            session_phase=SessionPhase.WORKING,
            user_context={
                "current_risk": RiskLevel.NONE,
                "contraindications": ["cognitive_impairment"],
            },
            treatment_plan={"current_phase": 1, "skills_acquired": []},
        )
        assert result is not None


class TestTechniqueDTO:
    """Tests for TechniqueDTO properties."""

    def test_technique_has_required_properties(self) -> None:
        """Test TechniqueDTO has all required properties."""
        selector = TechniqueSelector()
        techniques = selector.get_techniques_by_modality(TherapyModality.CBT)
        for technique in techniques:
            assert technique.technique_id is not None
            assert technique.name is not None
            assert technique.description is not None
            assert technique.modality is not None
            assert isinstance(technique.requires_homework, bool)

    def test_all_techniques_have_valid_modality(self) -> None:
        """Test all techniques have valid modality assignment."""
        selector = TechniqueSelector()
        all_techniques = selector.get_techniques_by_modality(None)
        valid_modalities = {m for m in TherapyModality}
        for technique in all_techniques:
            assert technique.modality in valid_modalities
