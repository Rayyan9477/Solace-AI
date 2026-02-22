"""
Solace-AI Personality Service - Domain Service Orchestration.
Orchestrates personality detection, profile management, and style adaptation.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID, uuid4
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from decimal import Decimal
from services.shared import ServiceBase

if TYPE_CHECKING:
    from services.shared.infrastructure import UnifiedLLMClient
    from ..infrastructure.repository import PersonalityRepositoryPort

from ..schemas import (
    PersonalityTrait, AssessmentSource, EmotionCategory, OceanScoresDTO, StyleParametersDTO,
    EmotionStateDTO, EmpathyComponentsDTO, ProfileSummaryDTO,
    DetectPersonalityRequest, DetectPersonalityResponse,
    GetStyleRequest, GetStyleResponse,
    AdaptResponseRequest, AdaptResponseResponse,
)
from .trait_detector import TraitDetector, TraitDetectorSettings
from .style_adapter import StyleAdapter, StyleAdapterSettings

logger = structlog.get_logger(__name__)


class PersonalityServiceSettings(BaseSettings):
    """Personality service configuration."""
    enable_profile_caching: bool = Field(default=True)
    profile_cache_ttl_seconds: int = Field(default=600)
    min_assessment_interval_seconds: int = Field(default=60)
    profile_update_threshold: float = Field(default=0.15)
    max_profile_history: int = Field(default=100)
    enable_llm_detection: bool = Field(default=True)
    model_config = SettingsConfigDict(env_prefix="PERSONALITY_SERVICE_", env_file=".env", extra="ignore")


@dataclass
class PersonalityProfile:
    """In-memory personality profile entity."""
    user_id: UUID
    profile_id: UUID = field(default_factory=uuid4)
    ocean_scores: OceanScoresDTO | None = None
    style_parameters: StyleParametersDTO | None = None
    assessment_count: int = 0
    stability_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    assessment_history: list[OceanScoresDTO] = field(default_factory=list)


class ProfileStore:
    """In-memory profile storage with caching. Thread-safe via asyncio.Lock."""

    def __init__(self) -> None:
        self._profiles: dict[UUID, PersonalityProfile] = {}
        self._cache_timestamps: dict[UUID, float] = {}
        self._locks: dict[UUID, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def _get_lock(self, user_id: UUID) -> asyncio.Lock:
        """Get per-user lock, creating if needed."""
        if user_id not in self._locks:
            async with self._global_lock:
                if user_id not in self._locks:
                    self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def get(self, user_id: UUID) -> PersonalityProfile | None:
        """Get profile by user ID."""
        return self._profiles.get(user_id)

    async def save(self, profile: PersonalityProfile) -> PersonalityProfile:
        """Save profile (per-user lock prevents concurrent overwrites)."""
        lock = await self._get_lock(profile.user_id)
        async with lock:
            profile.updated_at = datetime.now(timezone.utc)
            self._profiles[profile.user_id] = profile
            self._cache_timestamps[profile.user_id] = time.time()
            return profile

    def exists(self, user_id: UUID) -> bool:
        """Check if profile exists."""
        return user_id in self._profiles

    async def delete(self, user_id: UUID) -> bool:
        """Delete profile."""
        lock = await self._get_lock(user_id)
        async with lock:
            if user_id in self._profiles:
                del self._profiles[user_id]
                self._cache_timestamps.pop(user_id, None)
                return True
            return False

    def count(self) -> int:
        """Count total profiles."""
        return len(self._profiles)


class PersonalityOrchestrator(ServiceBase):
    """Main personality service orchestrator."""

    def __init__(
        self,
        settings: PersonalityServiceSettings | None = None,
        trait_detector: TraitDetector | None = None,
        style_adapter: StyleAdapter | None = None,
        llm_client: UnifiedLLMClient | None = None,
        repository: PersonalityRepositoryPort | None = None,
    ) -> None:
        self._settings = settings or PersonalityServiceSettings()
        self._trait_detector = trait_detector or TraitDetector(
            TraitDetectorSettings(enable_llm_detection=self._settings.enable_llm_detection),
            llm_client,
        )
        self._style_adapter = style_adapter or StyleAdapter()
        self._profile_store = ProfileStore()
        self._repository = repository
        self._initialized = False
        self._request_count = 0
        self._detection_count = 0

    async def initialize(self) -> None:
        """Initialize the personality orchestrator."""
        await self._trait_detector.initialize()
        await self._style_adapter.initialize()
        self._initialized = True
        logger.info(
            "personality_orchestrator_initialized",
            enable_llm=self._settings.enable_llm_detection,
            persistence=("postgres" if self._repository else "in_memory"),
        )

    async def _get_cached_profile(self, user_id: UUID) -> PersonalityProfile | None:
        """Get profile from cache, falling back to repository if available."""
        profile = self._profile_store.get(user_id)
        if profile is not None:
            return profile
        if self._repository is None:
            return None
        try:
            entity = await self._repository.get_profile_by_user(user_id)
            if entity is None:
                return None
            profile = self._entity_to_profile(entity)
            await self._profile_store.save(profile)
            logger.info("profile_loaded_from_repo", user_id=str(user_id))
            return profile
        except Exception as e:
            logger.warning("profile_repo_load_failed", user_id=str(user_id), error=str(e))
            return None

    async def _persist_profile(self, profile: PersonalityProfile) -> None:
        """Persist profile to repository if available."""
        if self._repository is None:
            return
        try:
            entity = self._profile_to_entity(profile)
            await self._repository.save_profile(entity)
        except Exception as e:
            logger.warning("profile_persist_failed", user_id=str(profile.user_id), error=str(e))

    def _profile_to_entity(self, profile: PersonalityProfile) -> Any:
        """Convert in-memory profile to domain entity for persistence."""
        from .entities import PersonalityProfile as EntityProfile
        from .value_objects import OceanScores
        ocean_scores = None
        if profile.ocean_scores:
            ocean_scores = OceanScores.from_dict(profile.ocean_scores.model_dump())
        comm_style = None
        if ocean_scores:
            from .value_objects import CommunicationStyle
            comm_style = CommunicationStyle.from_ocean(ocean_scores)
        return EntityProfile(
            user_id=profile.user_id,
            profile_id=profile.profile_id,
            ocean_scores=ocean_scores,
            communication_style=comm_style,
            assessment_count=profile.assessment_count,
            stability_score=Decimal(str(profile.stability_score)),
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            version=profile.version,
        )

    def _entity_to_profile(self, entity: Any) -> PersonalityProfile:
        """Convert domain entity to in-memory profile."""
        ocean_scores = None
        if entity.ocean_scores:
            scores_dict = entity.ocean_scores.to_dict()
            ocean_scores = OceanScoresDTO(**{
                k: v for k, v in scores_dict.items()
                if k in OceanScoresDTO.model_fields
            })
        style_params = None
        if ocean_scores:
            style_params = self._style_adapter.get_style_parameters(ocean_scores)
        return PersonalityProfile(
            user_id=entity.user_id,
            profile_id=entity.profile_id,
            ocean_scores=ocean_scores,
            style_parameters=style_params,
            assessment_count=entity.assessment_count,
            stability_score=float(entity.stability_score),
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            version=entity.version,
        )

    async def detect_personality(self, request: DetectPersonalityRequest) -> DetectPersonalityResponse:
        """Detect personality traits from text."""
        start_time = time.perf_counter()
        self._request_count += 1
        self._detection_count += 1
        ocean_scores = await self._trait_detector.detect(request.text, request.sources)
        evidence = []
        if request.include_evidence:
            evidence = self._extract_evidence(ocean_scores)
        profile = await self._get_or_create_profile(request.user_id)
        profile = self._update_profile_with_assessment(profile, ocean_scores)
        await self._profile_store.save(profile)
        await self._persist_profile(profile)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info("personality_detected", user_id=str(request.user_id), confidence=ocean_scores.overall_confidence, processing_time_ms=round(processing_time_ms, 2))
        return DetectPersonalityResponse(
            user_id=request.user_id,
            ocean_scores=ocean_scores,
            assessment_source=AssessmentSource.ENSEMBLE if len(request.sources) > 1 else request.sources[0],
            confidence=ocean_scores.overall_confidence,
            evidence=evidence,
            processing_time_ms=processing_time_ms,
        )

    async def get_style(self, request: GetStyleRequest) -> GetStyleResponse:
        """Get communication style parameters for user."""
        self._request_count += 1
        profile = await self._get_cached_profile(request.user_id)
        if not profile or not profile.ocean_scores:
            return GetStyleResponse(
                user_id=request.user_id,
                style_parameters=StyleParametersDTO(),
                recommendations=["Insufficient personality data - using default style"],
                profile_confidence=0.0,
            )
        style = self._style_adapter.get_style_parameters(profile.ocean_scores)
        recommendations = self._style_adapter.get_recommendations(profile.ocean_scores)
        if profile.style_parameters is None:
            profile.style_parameters = style
            await self._profile_store.save(profile)
        logger.info("style_retrieved", user_id=str(request.user_id), style_type=style.style_type.value)
        return GetStyleResponse(
            user_id=request.user_id,
            style_parameters=style,
            recommendations=recommendations,
            profile_confidence=profile.ocean_scores.overall_confidence,
        )

    async def adapt_response(self, request: AdaptResponseRequest) -> AdaptResponseResponse:
        """Adapt response content to user personality."""
        self._request_count += 1
        profile = await self._get_cached_profile(request.user_id)
        style = request.style_parameters
        if style is None:
            if profile and profile.style_parameters:
                style = profile.style_parameters
            elif profile and profile.ocean_scores:
                style = self._style_adapter.get_style_parameters(profile.ocean_scores)
            else:
                style = StyleParametersDTO()
        adapted_content = self._style_adapter.adapt_response(request.base_response, style)
        empathy_components = None
        if request.include_empathy:
            emotion = EmotionStateDTO(primary_emotion=EmotionCategory.NEUTRAL, intensity=0.3, valence=0.0)
            empathy_components = self._style_adapter.get_empathy_components(emotion, style)
        confidence = profile.ocean_scores.overall_confidence if profile and profile.ocean_scores else 0.3
        logger.info("response_adapted", user_id=str(request.user_id), content_length=len(adapted_content))
        return AdaptResponseResponse(
            user_id=request.user_id,
            adapted_content=adapted_content,
            applied_style=style,
            empathy_components=empathy_components,
            adaptation_confidence=confidence,
        )

    async def get_profile(self, user_id: UUID) -> ProfileSummaryDTO | None:
        """Get user profile summary."""
        profile = await self._get_cached_profile(user_id)
        if not profile or not profile.ocean_scores:
            return None
        return ProfileSummaryDTO(
            user_id=profile.user_id,
            ocean_scores=profile.ocean_scores,
            style_parameters=profile.style_parameters or StyleParametersDTO(),
            dominant_traits=profile.ocean_scores.dominant_traits(),
            assessment_count=profile.assessment_count,
            stability_score=profile.stability_score,
            last_updated=profile.updated_at,
            version=profile.version,
        )

    async def _get_or_create_profile(self, user_id: UUID) -> PersonalityProfile:
        """Get existing profile or create new one."""
        profile = await self._get_cached_profile(user_id)
        if profile is None:
            profile = PersonalityProfile(user_id=user_id)
            logger.info("profile_created", user_id=str(user_id))
        return profile

    def _update_profile_with_assessment(self, profile: PersonalityProfile, scores: OceanScoresDTO) -> PersonalityProfile:
        """Update profile with new assessment."""
        if profile.ocean_scores is None:
            profile.ocean_scores = scores
            profile.assessment_count = 1
        else:
            profile.ocean_scores = self._aggregate_scores(profile.ocean_scores, scores)
            profile.assessment_count += 1
            profile.assessment_history.append(scores)
            if len(profile.assessment_history) > self._settings.max_profile_history:
                profile.assessment_history = profile.assessment_history[-self._settings.max_profile_history:]
        profile.stability_score = self._compute_stability(profile)
        profile.style_parameters = self._style_adapter.get_style_parameters(profile.ocean_scores)
        profile.version += 1
        return profile

    def _aggregate_scores(self, current: OceanScoresDTO, new: OceanScoresDTO) -> OceanScoresDTO:
        """Aggregate scores using exponential moving average."""
        alpha = 0.3
        return OceanScoresDTO(
            openness=current.openness * (1 - alpha) + new.openness * alpha,
            conscientiousness=current.conscientiousness * (1 - alpha) + new.conscientiousness * alpha,
            extraversion=current.extraversion * (1 - alpha) + new.extraversion * alpha,
            agreeableness=current.agreeableness * (1 - alpha) + new.agreeableness * alpha,
            neuroticism=current.neuroticism * (1 - alpha) + new.neuroticism * alpha,
            overall_confidence=min(0.9, current.overall_confidence * 0.7 + new.overall_confidence * 0.3),
        )

    def _compute_stability(self, profile: PersonalityProfile) -> float:
        """Compute profile stability score based on assessment variance."""
        if profile.assessment_count < 3:
            return 0.3
        if len(profile.assessment_history) < 2:
            return 0.5
        variance_sum = 0.0
        for trait in PersonalityTrait:
            values = [s.get_trait(trait) for s in profile.assessment_history[-5:]]
            if len(values) > 1:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                variance_sum += variance
        avg_variance = variance_sum / 5
        return max(0.0, min(1.0, 1.0 - avg_variance * 10))

    def _extract_evidence(self, scores: OceanScoresDTO) -> list[str]:
        """Extract evidence markers from trait scores."""
        evidence = []
        for trait_score in scores.trait_scores:
            evidence.extend(trait_score.evidence_markers)
        return list(set(evidence))[:10]

    async def get_status(self) -> dict[str, Any]:
        """Get service status and statistics."""
        return {
            "status": "operational" if self._initialized else "initializing",
            "initialized": self._initialized,
            "statistics": {
                "total_requests": self._request_count,
                "total_detections": self._detection_count,
            },
            "profiles_count": self._profile_store.count(),
            "enable_llm": self._settings.enable_llm_detection,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Get service statistics counters."""
        return {
            "total_requests": self._request_count,
            "total_detections": self._detection_count,
        }

    async def shutdown(self) -> None:
        """Shutdown the personality orchestrator."""
        await self._trait_detector.shutdown()
        await self._style_adapter.shutdown()
        self._initialized = False
        logger.info("personality_orchestrator_shutdown", total_requests=self._request_count, total_detections=self._detection_count)

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized
