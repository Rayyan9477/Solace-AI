"""Sprint 6 orchestrator + personality regression lock-ins.

Sprint 0 verification confirmed each of these was already fixed in
source. These tests prevent silent regressions in a future refactor:

  - H-20 Assessment + Emotion agent files present in orchestrator.
  - H-21 / H-22 Personality TraitDetector instantiates RoBERTa and
         uses the spec ensemble weights (RoBERTa=0.5, LLM=0.3, LIWC=0.2).
  - H-25 Personality agent calls the ``/api/v1/personality/...`` prefix.
  - H-43 Orchestrator safety check payload uses the lowercase
         ``full_assessment`` value so the safety service's API
         validator accepts it.
"""
from __future__ import annotations

import inspect


class TestH20AgentModulesPresent:
    """H-20: assessment and emotion agents exist as first-class modules,
    not stubs or missing files.
    """

    def test_assessment_agent_module_exists(self) -> None:
        from services.orchestrator_service.src.agents import assessment_agent

        assert assessment_agent is not None
        # Must define an actual agent, not just a comment
        module_src = inspect.getsource(assessment_agent)
        assert "class " in module_src or "def " in module_src

    def test_emotion_agent_module_exists(self) -> None:
        from services.orchestrator_service.src.agents import emotion_agent

        assert emotion_agent is not None
        module_src = inspect.getsource(emotion_agent)
        assert "class " in module_src or "def " in module_src


class TestH21H22PersonalityEnsembleWeights:
    """H-21 / H-22: TraitDetector wires RoBERTa and uses spec weights."""

    def test_roberta_detector_imported_and_instantiated(self) -> None:
        from services.personality_service.src.domain import trait_detector

        src = inspect.getsource(trait_detector)
        assert "RoBERTaPersonalityDetector" in src, (
            "H-21: TraitDetector module must import RoBERTaPersonalityDetector"
        )
        assert "self._roberta = RoBERTaPersonalityDetector(" in src, (
            "H-21: TraitDetector must instantiate RoBERTa, not leave it unused"
        )

    def test_ensemble_weights_match_spec(self) -> None:
        """Spec: RoBERTa=0.5, LLM=0.3, LIWC=0.2."""
        from services.personality_service.src.domain import trait_detector

        src = inspect.getsource(trait_detector)
        assert "ensemble_weights_roberta: float = Field(default=0.5" in src, (
            "H-22: RoBERTa ensemble weight must default to 0.5 per spec"
        )
        assert "ensemble_weights_llm: float = Field(default=0.3" in src, (
            "H-22: LLM ensemble weight must default to 0.3 per spec"
        )
        assert "ensemble_weights_liwc: float = Field(default=0.2" in src, (
            "H-22: LIWC ensemble weight must default to 0.2 per spec"
        )


class TestH25PersonalityAgentUrlPrefix:
    """H-25: orchestrator personality agent targets the ``/api/v1/personality``
    prefix. Missing the prefix would return 404 for every personality call.
    """

    def test_personality_agent_uses_api_v1_prefix(self) -> None:
        from services.orchestrator_service.src.agents import personality_agent

        src = inspect.getsource(personality_agent)
        assert "/api/v1/personality/detect" in src, (
            "H-25: personality agent must POST to /api/v1/personality/detect"
        )
        assert "/api/v1/personality/style" in src, (
            "H-25: personality agent must POST to /api/v1/personality/style"
        )


class TestH43SafetyCheckCasing:
    """H-43: safety check payload uses lowercase ``full_assessment`` so the
    safety service's request validator accepts it. The old uppercase
    ``FULL_ASSESSMENT`` was producing 422 on every orchestrator safety call.
    """

    def test_safety_client_uses_lowercase_full_assessment(self) -> None:
        from services.orchestrator_service.src.infrastructure import clients

        src = inspect.getsource(clients)
        # The default check_type parameter must be lowercase
        assert 'check_type: str = "full_assessment"' in src, (
            "H-43: safety client default check_type must be lowercase "
            "'full_assessment' to match the safety service validator"
        )
        # And the uppercase legacy value must not be the active default
        assert 'check_type: str = "FULL_ASSESSMENT"' not in src, (
            "H-43: uppercase FULL_ASSESSMENT default must be gone"
        )
