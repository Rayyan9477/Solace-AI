"""
Solace-AI Orchestrator Service - Personality Agent.
Coordinates with Personality Service for Big Five trait detection and style adaptation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..langgraph.state_schema import (
    OrchestratorState,
    AgentType,
    AgentResult,
)

logger = structlog.get_logger(__name__)


class PersonalityTrait(str, Enum):
    """Big Five personality traits (OCEAN model)."""
    OPENNESS = "OPENNESS"
    CONSCIENTIOUSNESS = "CONSCIENTIOUSNESS"
    EXTRAVERSION = "EXTRAVERSION"
    AGREEABLENESS = "AGREEABLENESS"
    NEUROTICISM = "NEUROTICISM"


class CommunicationStyleType(str, Enum):
    """Communication style types based on personality."""
    ANALYTICAL = "ANALYTICAL"
    EXPRESSIVE = "EXPRESSIVE"
    DRIVER = "DRIVER"
    AMIABLE = "AMIABLE"
    BALANCED = "BALANCED"


class AssessmentSource(str, Enum):
    """Sources for personality assessment."""
    TEXT_ANALYSIS = "TEXT_ANALYSIS"
    LLM_ZERO_SHOT = "LLM_ZERO_SHOT"
    LIWC_FEATURES = "LIWC_FEATURES"
    ENSEMBLE = "ENSEMBLE"
    CACHED_PROFILE = "CACHED_PROFILE"


class PersonalityAgentSettings(BaseSettings):
    """Configuration for the personality agent."""
    service_url: str = Field(default="http://localhost:8004")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    fallback_on_service_error: bool = Field(default=True)
    enable_style_adaptation: bool = Field(default=True)
    enable_empathy_components: bool = Field(default=True)
    min_text_length_for_detection: int = Field(default=50)
    cache_profile_ttl_seconds: int = Field(default=3600)
    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRATOR_PERSONALITY_AGENT_",
        env_file=".env",
        extra="ignore"
    )


@dataclass
class TraitScoreDTO:
    """Individual trait score with confidence."""
    trait: PersonalityTrait
    value: float
    confidence_lower: float
    confidence_upper: float
    evidence_markers: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraitScoreDTO:
        return cls(trait=PersonalityTrait(data.get("trait", "OPENNESS")), value=float(data.get("value", 0.5)), confidence_lower=float(data.get("confidence_lower", 0.3)), confidence_upper=float(data.get("confidence_upper", 0.7)), evidence_markers=data.get("evidence_markers", []))

    def to_dict(self) -> dict[str, Any]:
        return {"trait": self.trait.value, "value": self.value, "confidence_lower": self.confidence_lower, "confidence_upper": self.confidence_upper, "evidence_markers": self.evidence_markers}


@dataclass
class OceanScoresDTO:
    """Big Five OCEAN personality scores."""
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    overall_confidence: float
    assessed_at: datetime
    trait_scores: list[TraitScoreDTO] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OceanScoresDTO:
        assessed_at = data.get("assessed_at")
        if isinstance(assessed_at, str):
            assessed_at = datetime.fromisoformat(assessed_at)
        return cls(openness=float(data.get("openness", 0.5)), conscientiousness=float(data.get("conscientiousness", 0.5)), extraversion=float(data.get("extraversion", 0.5)), agreeableness=float(data.get("agreeableness", 0.5)), neuroticism=float(data.get("neuroticism", 0.5)), overall_confidence=float(data.get("overall_confidence", 0.5)), assessed_at=assessed_at or datetime.now(timezone.utc), trait_scores=[TraitScoreDTO.from_dict(t) for t in data.get("trait_scores", [])])

    def to_dict(self) -> dict[str, Any]:
        return {"openness": self.openness, "conscientiousness": self.conscientiousness, "extraversion": self.extraversion, "agreeableness": self.agreeableness, "neuroticism": self.neuroticism, "overall_confidence": self.overall_confidence, "assessed_at": self.assessed_at.isoformat(), "trait_scores": [t.to_dict() for t in self.trait_scores]}

    def dominant_traits(self, threshold: float = 0.6) -> list[PersonalityTrait]:
        traits = []
        if self.openness >= threshold: traits.append(PersonalityTrait.OPENNESS)
        if self.conscientiousness >= threshold: traits.append(PersonalityTrait.CONSCIENTIOUSNESS)
        if self.extraversion >= threshold: traits.append(PersonalityTrait.EXTRAVERSION)
        if self.agreeableness >= threshold: traits.append(PersonalityTrait.AGREEABLENESS)
        if self.neuroticism >= threshold: traits.append(PersonalityTrait.NEUROTICISM)
        return traits


@dataclass
class StyleParametersDTO:
    """Communication style parameters for response adaptation."""
    warmth: float
    structure: float
    complexity: float
    directness: float
    energy: float
    validation_level: float
    style_type: CommunicationStyleType
    custom_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleParametersDTO:
        return cls(warmth=float(data.get("warmth", 0.6)), structure=float(data.get("structure", 0.5)), complexity=float(data.get("complexity", 0.5)), directness=float(data.get("directness", 0.5)), energy=float(data.get("energy", 0.5)), validation_level=float(data.get("validation_level", 0.6)), style_type=CommunicationStyleType(data.get("style_type", "BALANCED")), custom_params=data.get("custom_params", {}))

    def to_dict(self) -> dict[str, Any]:
        return {"warmth": self.warmth, "structure": self.structure, "complexity": self.complexity, "directness": self.directness, "energy": self.energy, "validation_level": self.validation_level, "style_type": self.style_type.value, "custom_params": self.custom_params}

    @classmethod
    def default(cls) -> StyleParametersDTO:
        return cls(warmth=0.6, structure=0.5, complexity=0.5, directness=0.5, energy=0.5, validation_level=0.6, style_type=CommunicationStyleType.BALANCED)


@dataclass
class PersonalityDetectionResult:
    """Result from personality detection."""
    user_id: str
    ocean_scores: OceanScoresDTO
    assessment_source: AssessmentSource
    confidence: float
    evidence: list[str]
    processing_time_ms: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalityDetectionResult:
        return cls(user_id=data.get("user_id", ""), ocean_scores=OceanScoresDTO.from_dict(data.get("ocean_scores", {})), assessment_source=AssessmentSource(data.get("assessment_source", "TEXT_ANALYSIS")), confidence=float(data.get("confidence", 0.5)), evidence=data.get("evidence", []), processing_time_ms=float(data.get("processing_time_ms", 0.0)))


@dataclass
class StyleResponse:
    """Response from style adaptation request."""
    style_parameters: StyleParametersDTO
    recommendations: list[str]
    profile_confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleResponse:
        return cls(style_parameters=StyleParametersDTO.from_dict(data.get("style_parameters", {})), recommendations=data.get("recommendations", []), profile_confidence=float(data.get("profile_confidence", 0.5)))


class PersonalityServiceClient:
    """HTTP client for Personality Service communication."""

    def __init__(self, settings: PersonalityAgentSettings) -> None:
        self._settings = settings
        self._base_url = settings.service_url.rstrip("/")

    async def detect_personality(
        self,
        user_id: str,
        text: str,
        session_id: str | None = None,
    ) -> PersonalityDetectionResult:
        """Detect personality traits from text."""
        url = f"{self._base_url}/detect"
        payload = {
            "user_id": user_id,
            "text": text,
            "session_id": session_id,
            "include_evidence": True,
            "sources": ["TEXT_ANALYSIS", "LLM_ZERO_SHOT"],
        }
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return PersonalityDetectionResult.from_dict(response.json())
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "personality_service_http_error",
                        status_code=e.response.status_code,
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
                except httpx.RequestError as e:
                    logger.warning(
                        "personality_service_request_error",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == self._settings.max_retries:
                        raise
        raise RuntimeError("Personality service request failed after retries")

    async def get_style(self, user_id: str) -> StyleResponse:
        """Get communication style for user."""
        url = f"{self._base_url}/style"
        payload = {"user_id": user_id}
        async with httpx.AsyncClient(timeout=self._settings.timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return StyleResponse.from_dict(response.json())


class PersonalityAgent:
    """
    Personality adaptation agent for the orchestrator.
    Coordinates with Personality Service for trait detection and style adaptation.
    """

    def __init__(self, settings: PersonalityAgentSettings | None = None) -> None:
        self._settings = settings or PersonalityAgentSettings()
        self._client = PersonalityServiceClient(self._settings)
        self._detection_count = 0

    async def process(self, state: OrchestratorState) -> dict[str, Any]:
        """
        Process state and adapt personality style.
        This is the main LangGraph node function.

        Args:
            state: Current orchestrator state

        Returns:
            State updates dictionary
        """
        self._detection_count += 1
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        message = state.get("current_message", "")
        existing_style = state.get("personality_style", {})
        logger.info(
            "personality_agent_processing",
            user_id=user_id,
            message_length=len(message),
            has_existing_style=bool(existing_style),
        )
        try:
            style_params = await self._get_personality_style(
                user_id=user_id,
                session_id=session_id,
                message=message,
                existing_style=existing_style,
            )
            return self._build_state_update(style_params)
        except Exception as e:
            logger.error("personality_agent_error", error=str(e))
            if self._settings.fallback_on_service_error:
                return self._build_fallback_response(existing_style)
            raise

    async def _get_personality_style(
        self,
        user_id: str,
        session_id: str,
        message: str,
        existing_style: dict[str, Any],
    ) -> StyleParametersDTO:
        """Get personality style via service or detection."""
        if existing_style and existing_style.get("confidence", 0) > 0.7:
            return StyleParametersDTO.from_dict(existing_style)
        if len(message) >= self._settings.min_text_length_for_detection:
            detection = await self._client.detect_personality(
                user_id=user_id,
                text=message,
                session_id=session_id,
            )
            return self._scores_to_style(detection.ocean_scores)
        try:
            style_response = await self._client.get_style(user_id)
            return style_response.style_parameters
        except Exception:
            return StyleParametersDTO.default()

    def _scores_to_style(self, scores: OceanScoresDTO) -> StyleParametersDTO:
        """Convert OCEAN scores to style parameters."""
        warmth = (scores.agreeableness + (1 - scores.neuroticism)) / 2
        structure = scores.conscientiousness
        complexity = scores.openness
        directness = scores.extraversion * 0.7 + (1 - scores.agreeableness) * 0.3
        energy = scores.extraversion
        validation_level = scores.agreeableness * 0.6 + (1 - scores.neuroticism) * 0.4
        style_type = self._determine_style_type(scores)
        return StyleParametersDTO(
            warmth=round(warmth, 2),
            structure=round(structure, 2),
            complexity=round(complexity, 2),
            directness=round(directness, 2),
            energy=round(energy, 2),
            validation_level=round(validation_level, 2),
            style_type=style_type,
            custom_params={"ocean_based": True, "confidence": scores.overall_confidence},
        )

    def _determine_style_type(self, scores: OceanScoresDTO) -> CommunicationStyleType:
        """Determine communication style type from OCEAN scores."""
        if scores.extraversion > 0.6 and scores.openness > 0.6:
            return CommunicationStyleType.EXPRESSIVE
        if scores.conscientiousness > 0.6 and scores.openness > 0.5:
            return CommunicationStyleType.ANALYTICAL
        if scores.extraversion > 0.6 and scores.conscientiousness > 0.6:
            return CommunicationStyleType.DRIVER
        if scores.agreeableness > 0.6:
            return CommunicationStyleType.AMIABLE
        return CommunicationStyleType.BALANCED

    def _build_state_update(self, style: StyleParametersDTO) -> dict[str, Any]:
        """Build state update from style parameters."""
        style_dict = style.to_dict()
        agent_result = AgentResult(
            agent_type=AgentType.PERSONALITY,
            success=True,
            confidence=style.custom_params.get("confidence", 0.7),
            metadata={
                "style_type": style.style_type.value,
                "warmth": style.warmth,
                "validation_level": style.validation_level,
            },
        )
        logger.info(
            "personality_agent_complete",
            style_type=style.style_type.value,
            warmth=style.warmth,
        )
        return {
            "personality_style": style_dict,
            "agent_results": [agent_result.to_dict()],
        }

    def _build_fallback_response(self, existing_style: dict[str, Any]) -> dict[str, Any]:
        """Build fallback response using existing or default style."""
        if existing_style:
            style = StyleParametersDTO.from_dict(existing_style)
        else:
            style = StyleParametersDTO.default()
        agent_result = AgentResult(
            agent_type=AgentType.PERSONALITY,
            success=True,
            confidence=0.5,
            metadata={"fallback_mode": True, "style_type": style.style_type.value},
        )
        logger.warning("personality_agent_fallback_used")
        return {
            "personality_style": style.to_dict(),
            "agent_results": [agent_result.to_dict()],
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "total_detections": self._detection_count,
            "service_url": self._settings.service_url,
            "style_adaptation_enabled": self._settings.enable_style_adaptation,
        }


async def personality_agent_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node function for personality agent processing.

    Args:
        state: Current orchestrator state

    Returns:
        State updates dictionary
    """
    agent = PersonalityAgent()
    return await agent.process(state)
