"""
Solace-AI Personality Service - API Request/Response Schemas.
Pydantic models for Big Five (OCEAN) personality detection and style adaptation.
"""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class PersonalityTrait(str, Enum):
    """Big Five (OCEAN) personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class AssessmentSource(str, Enum):
    """Source of personality assessment."""
    TEXT_ANALYSIS = "text_analysis"
    LLM_ZERO_SHOT = "llm_zero_shot"
    LIWC_FEATURES = "liwc_features"
    VOICE_ANALYSIS = "voice_analysis"
    BEHAVIORAL = "behavioral"
    ENSEMBLE = "ensemble"
    SELF_REPORT = "self_report"


class CommunicationStyleType(str, Enum):
    """Communication style archetypes."""
    ANALYTICAL = "analytical"
    EXPRESSIVE = "expressive"
    DRIVER = "driver"
    AMIABLE = "amiable"
    BALANCED = "balanced"


class EmotionCategory(str, Enum):
    """Primary emotion categories for empathy detection."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class EmpathyComponent(str, Enum):
    """Three-component empathy model."""
    COGNITIVE = "cognitive"
    AFFECTIVE = "affective"
    COMPASSIONATE = "compassionate"


class TraitScoreDTO(BaseModel):
    """Individual trait score with confidence interval."""
    trait: PersonalityTrait
    value: float = Field(ge=0.0, le=1.0)
    confidence_lower: float = Field(ge=0.0, le=1.0)
    confidence_upper: float = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0, default=1)
    evidence_markers: list[str] = Field(default_factory=list)

    @field_validator("confidence_upper")
    @classmethod
    def validate_confidence_range(cls, v: float, info) -> float:
        """Ensure upper bound >= lower bound."""
        lower = info.data.get("confidence_lower", 0.0)
        if v < lower:
            return lower
        return v


class OceanScoresDTO(BaseModel):
    """Complete OCEAN personality scores."""
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    trait_scores: list[TraitScoreDTO] = Field(default_factory=list)

    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get score for a specific trait."""
        return getattr(self, trait.value)

    def dominant_traits(self, threshold: float = 0.7) -> list[PersonalityTrait]:
        """Get traits above threshold."""
        result = []
        for trait in PersonalityTrait:
            if self.get_trait(trait) >= threshold:
                result.append(trait)
        return result

    def low_traits(self, threshold: float = 0.3) -> list[PersonalityTrait]:
        """Get traits below threshold."""
        result = []
        for trait in PersonalityTrait:
            if self.get_trait(trait) <= threshold:
                result.append(trait)
        return result


class StyleParametersDTO(BaseModel):
    """Communication style parameters for response adaptation."""
    warmth: float = Field(ge=0.0, le=1.0, default=0.5)
    structure: float = Field(ge=0.0, le=1.0, default=0.5)
    complexity: float = Field(ge=0.0, le=1.0, default=0.5)
    directness: float = Field(ge=0.0, le=1.0, default=0.5)
    energy: float = Field(ge=0.0, le=1.0, default=0.5)
    validation_level: float = Field(ge=0.0, le=1.0, default=0.5)
    style_type: CommunicationStyleType = CommunicationStyleType.BALANCED
    custom_params: dict[str, Any] = Field(default_factory=dict)


class EmotionStateDTO(BaseModel):
    """Current emotional state with intensity."""
    primary_emotion: EmotionCategory
    secondary_emotion: EmotionCategory | None = None
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class EmpathyComponentsDTO(BaseModel):
    """Empathy response components for generation."""
    cognitive_content: str = Field(default="")
    affective_content: str = Field(default="")
    compassionate_content: str = Field(default="")
    selected_strategy: str = Field(default="balanced")
    emotion_state: EmotionStateDTO | None = None


class DetectPersonalityRequest(BaseModel):
    """Request to detect personality from text."""
    user_id: UUID
    text: str = Field(min_length=10, max_length=10000)
    session_id: UUID | None = None
    include_evidence: bool = Field(default=False)
    sources: list[AssessmentSource] = Field(default_factory=lambda: [AssessmentSource.ENSEMBLE])


class DetectPersonalityResponse(BaseModel):
    """Response with detected personality traits."""
    user_id: UUID
    ocean_scores: OceanScoresDTO
    assessment_source: AssessmentSource
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0)


class GetStyleRequest(BaseModel):
    """Request for style parameters."""
    user_id: UUID
    context: str | None = None
    emotion_context: EmotionStateDTO | None = None


class GetStyleResponse(BaseModel):
    """Response with style parameters."""
    user_id: UUID
    style_parameters: StyleParametersDTO
    recommendations: list[str] = Field(default_factory=list)
    profile_confidence: float = Field(ge=0.0, le=1.0)


class AdaptResponseRequest(BaseModel):
    """Request to adapt a response to personality."""
    user_id: UUID
    base_response: str = Field(min_length=1, max_length=10000)
    style_parameters: StyleParametersDTO | None = None
    include_empathy: bool = Field(default=True)


class AdaptResponseResponse(BaseModel):
    """Response with personality-adapted content."""
    user_id: UUID
    adapted_content: str
    applied_style: StyleParametersDTO
    empathy_components: EmpathyComponentsDTO | None = None
    adaptation_confidence: float = Field(ge=0.0, le=1.0)


class ProfileSummaryDTO(BaseModel):
    """Summary of personality profile for API response."""
    user_id: UUID
    ocean_scores: OceanScoresDTO
    style_parameters: StyleParametersDTO
    dominant_traits: list[PersonalityTrait]
    assessment_count: int = Field(ge=0)
    stability_score: float = Field(ge=0.0, le=1.0, default=0.0)
    last_updated: datetime
    version: int = Field(ge=1)


class GetProfileResponse(BaseModel):
    """Response with user profile summary."""
    profile: ProfileSummaryDTO
    exists: bool = True


class UpdateProfileRequest(BaseModel):
    """Request to update personality profile with new assessment."""
    user_id: UUID
    ocean_scores: OceanScoresDTO
    source: AssessmentSource = AssessmentSource.TEXT_ANALYSIS


class UpdateProfileResponse(BaseModel):
    """Response after profile update."""
    user_id: UUID
    previous_version: int
    new_version: int
    changed_traits: list[PersonalityTrait] = Field(default_factory=list)
    update_reason: str = Field(default="new_assessment")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    detection_models: list[str]
    style_adapter: str


class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: ErrorDetail
