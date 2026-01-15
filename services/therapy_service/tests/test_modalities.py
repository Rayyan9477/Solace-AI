"""
Unit tests for Therapy Modalities.
Tests CBT, DBT, ACT, MI implementations and modality registry.
"""
from __future__ import annotations
import pytest
from uuid import uuid4

from services.therapy_service.src.domain.modalities import (
    ModalityRegistry,
    ModalityProtocol,
    TechniqueProtocol,
    InterventionContext,
    InterventionResult,
    CBTProvider,
    DBTProvider,
    ACTProvider,
    MIProvider,
    ModalityPhase,
)
from services.therapy_service.src.schemas import TherapyModality, SessionPhase, SeverityLevel


class TestCBTProvider:
    """Tests for CBT modality provider."""

    def test_cbt_initialization(self) -> None:
        """Test CBT provider initializes correctly."""
        cbt = CBTProvider()
        assert cbt.modality == TherapyModality.CBT
        assert cbt.protocol.name == "Cognitive Behavioral Therapy"

    def test_cbt_has_techniques(self) -> None:
        """Test CBT provider has techniques."""
        cbt = CBTProvider()
        techniques = cbt.get_techniques()
        assert len(techniques) > 0
        assert any(t.name == "Thought Record" for t in techniques)

    def test_cbt_protocol_structure(self) -> None:
        """Test CBT protocol has proper structure."""
        cbt = CBTProvider()
        protocol = cbt.protocol
        assert len(protocol.core_principles) > 0
        assert len(protocol.key_techniques) > 0
        assert "opening" in protocol.session_structure

    def test_cbt_technique_selection_cognitive(self) -> None:
        """Test CBT selects cognitive technique for thought-related concerns."""
        cbt = CBTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I keep thinking I'm a failure",
        )
        technique = cbt.select_intervention(context)
        assert technique is not None
        assert technique.name == "Thought Record"

    def test_cbt_technique_selection_behavioral(self) -> None:
        """Test CBT selects behavioral technique for activity concerns."""
        cbt = CBTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I can't find motivation to do anything",
        )
        technique = cbt.select_intervention(context)
        assert technique is not None
        assert technique.name == "Behavioral Activation"

    def test_cbt_generate_response(self) -> None:
        """Test CBT generates therapeutic response."""
        cbt = CBTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="negative thoughts",
        )
        technique = cbt.select_intervention(context)
        result = cbt.generate_response(technique, "I feel worthless", context)
        assert result.success is True
        assert len(result.response_generated) > 0
        assert len(result.follow_up_prompts) > 0


class TestDBTProvider:
    """Tests for DBT modality provider."""

    def test_dbt_initialization(self) -> None:
        """Test DBT provider initializes correctly."""
        dbt = DBTProvider()
        assert dbt.modality == TherapyModality.DBT
        assert "Dialectical" in dbt.protocol.name

    def test_dbt_has_techniques(self) -> None:
        """Test DBT provider has techniques."""
        dbt = DBTProvider()
        techniques = dbt.get_techniques()
        assert any(t.name == "STOP Skill" for t in techniques)
        assert any(t.name == "DEAR MAN" for t in techniques)

    def test_dbt_technique_selection_severe(self) -> None:
        """Test DBT selects grounding for severe cases."""
        dbt = DBTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.SEVERE,
            current_concern="I'm overwhelmed",
        )
        technique = dbt.select_intervention(context)
        assert technique is not None
        assert technique.name == "STOP Skill"

    def test_dbt_technique_selection_interpersonal(self) -> None:
        """Test DBT selects interpersonal technique."""
        dbt = DBTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I need to assert myself more in relationships",
        )
        technique = dbt.select_intervention(context)
        assert technique is not None
        assert technique.name == "DEAR MAN"

    def test_dbt_generate_response(self) -> None:
        """Test DBT generates therapeutic response."""
        dbt = DBTProvider()
        techniques = dbt.get_techniques()
        stop_skill = next(t for t in techniques if t.name == "STOP Skill")
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="distress",
        )
        result = dbt.generate_response(stop_skill, "I'm so angry", context)
        assert result.success is True
        assert "STOP" in result.response_generated or "pause" in result.response_generated.lower()


class TestACTProvider:
    """Tests for ACT modality provider."""

    def test_act_initialization(self) -> None:
        """Test ACT provider initializes correctly."""
        act = ACTProvider()
        assert act.modality == TherapyModality.ACT
        assert "Acceptance and Commitment" in act.protocol.name

    def test_act_has_techniques(self) -> None:
        """Test ACT provider has techniques."""
        act = ACTProvider()
        techniques = act.get_techniques()
        assert any(t.name == "Values Clarification" for t in techniques)
        assert any(t.name == "Cognitive Defusion" for t in techniques)

    def test_act_technique_selection_values(self) -> None:
        """Test ACT selects values technique."""
        act = ACTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I don't know what I want in life, I've lost my sense of purpose",
        )
        technique = act.select_intervention(context)
        assert technique is not None
        assert technique.name == "Values Clarification"

    def test_act_technique_selection_defusion(self) -> None:
        """Test ACT selects defusion for stuck thoughts."""
        act = ACTProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I can't stop worrying about everything",
        )
        technique = act.select_intervention(context)
        assert technique is not None
        assert technique.name == "Cognitive Defusion"

    def test_act_generate_response(self) -> None:
        """Test ACT generates therapeutic response."""
        act = ACTProvider()
        techniques = act.get_techniques()
        values_tech = next(t for t in techniques if t.name == "Values Clarification")
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="meaning",
        )
        result = act.generate_response(values_tech, "I feel empty", context)
        assert result.success is True
        assert "value" in result.response_generated.lower() or "matter" in result.response_generated.lower()


class TestMIProvider:
    """Tests for MI modality provider."""

    def test_mi_initialization(self) -> None:
        """Test MI provider initializes correctly."""
        mi = MIProvider()
        assert mi.modality == TherapyModality.MI
        assert "Motivational Interviewing" in mi.protocol.name

    def test_mi_has_techniques(self) -> None:
        """Test MI provider has techniques."""
        mi = MIProvider()
        techniques = mi.get_techniques()
        assert any(t.name == "Reflective Listening" for t in techniques)
        assert any(t.name == "Change Talk Elicitation" for t in techniques)

    def test_mi_technique_selection_ambivalence(self) -> None:
        """Test MI selects change talk for ambivalence."""
        mi = MIProvider()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I want to change but I'm not sure",
        )
        technique = mi.select_intervention(context)
        assert technique is not None
        assert technique.name == "Change Talk Elicitation"

    def test_mi_generate_response(self) -> None:
        """Test MI generates therapeutic response."""
        mi = MIProvider()
        techniques = mi.get_techniques()
        change_talk = next(t for t in techniques if t.name == "Change Talk Elicitation")
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="change",
        )
        result = mi.generate_response(change_talk, "I want to quit but it's hard", context)
        assert result.success is True
        assert len(result.follow_up_prompts) > 0


class TestModalityRegistry:
    """Tests for ModalityRegistry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes with all modalities."""
        registry = ModalityRegistry()
        modalities = registry.list_modalities()
        assert TherapyModality.CBT in modalities
        assert TherapyModality.DBT in modalities
        assert TherapyModality.ACT in modalities
        assert TherapyModality.MI in modalities

    def test_get_provider(self) -> None:
        """Test getting a specific provider."""
        registry = ModalityRegistry()
        cbt = registry.get_provider(TherapyModality.CBT)
        assert cbt is not None
        assert cbt.modality == TherapyModality.CBT

    def test_get_all_techniques(self) -> None:
        """Test getting all techniques from all modalities."""
        registry = ModalityRegistry()
        techniques = registry.get_all_techniques()
        assert len(techniques) > 10  # Multiple techniques across modalities

    def test_select_intervention_preferred_modality(self) -> None:
        """Test intervention selection with preferred modality."""
        registry = ModalityRegistry()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I need help with my thoughts",
        )
        result = registry.select_intervention_for_context(context, preferred_modality=TherapyModality.CBT)
        assert result is not None
        provider, technique = result
        assert provider.modality == TherapyModality.CBT

    def test_select_intervention_no_preference(self) -> None:
        """Test intervention selection without preference."""
        registry = ModalityRegistry()
        context = InterventionContext(
            user_id=uuid4(),
            session_phase=SessionPhase.WORKING,
            severity=SeverityLevel.MODERATE,
            current_concern="I'm struggling",
        )
        result = registry.select_intervention_for_context(context)
        assert result is not None


class TestTechniqueProtocol:
    """Tests for TechniqueProtocol structure."""

    def test_technique_has_steps(self) -> None:
        """Test technique protocols have steps."""
        cbt = CBTProvider()
        thought_record = next(t for t in cbt.get_techniques() if t.name == "Thought Record")
        assert len(thought_record.steps) > 0
        assert thought_record.steps[0].step_number == 1

    def test_technique_has_duration(self) -> None:
        """Test technique has duration estimate."""
        dbt = DBTProvider()
        stop_skill = next(t for t in dbt.get_techniques() if t.name == "STOP Skill")
        assert stop_skill.duration_minutes > 0

    def test_technique_has_rationale(self) -> None:
        """Test technique has therapeutic rationale."""
        act = ACTProvider()
        values = next(t for t in act.get_techniques() if t.name == "Values Clarification")
        assert len(values.rationale) > 0


class TestModalityProtocol:
    """Tests for ModalityProtocol structure."""

    def test_protocol_has_principles(self) -> None:
        """Test protocol has core principles."""
        cbt = CBTProvider()
        assert len(cbt.protocol.core_principles) > 0

    def test_protocol_has_contraindications(self) -> None:
        """Test protocol lists contraindications."""
        cbt = CBTProvider()
        assert len(cbt.protocol.contraindications) > 0

    def test_protocol_has_evidence(self) -> None:
        """Test protocol includes efficacy evidence."""
        cbt = CBTProvider()
        assert len(cbt.protocol.efficacy_evidence) > 0
