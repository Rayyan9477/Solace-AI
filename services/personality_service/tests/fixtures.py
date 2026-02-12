"""
Test fixtures for Personality Service.
In-memory repository implementations for use in tests only.
"""
from __future__ import annotations
import os
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID
import structlog

from services.personality_service.src.domain.entities import (
    PersonalityProfile, TraitAssessment, ProfileSnapshot,
)
from services.personality_service.src.infrastructure.repository import PersonalityRepositoryPort
from services.personality_service.src.schemas import (
    PersonalityTrait, AssessmentSource, CommunicationStyleType,
)

logger = structlog.get_logger(__name__)


class InMemoryPersonalityRepository(PersonalityRepositoryPort):
    """In-memory implementation of personality repository."""
    def __init__(self) -> None:
        if os.getenv("ENVIRONMENT") == "production":
            raise RuntimeError("In-memory repositories are not allowed in production.")
        self._profiles: dict[UUID, PersonalityProfile] = {}
        self._user_profiles: dict[UUID, UUID] = {}
        self._assessments: dict[UUID, TraitAssessment] = {}
        self._user_assessments: dict[UUID, list[UUID]] = {}
        self._snapshots: dict[UUID, ProfileSnapshot] = {}
        self._profile_snapshots: dict[UUID, list[UUID]] = {}
        self._stats = {"profiles_saved": 0, "assessments_saved": 0, "snapshots_saved": 0, "queries": 0, "deletes": 0}

    async def save_profile(self, profile: PersonalityProfile) -> None:
        profile.touch()
        self._profiles[profile.profile_id] = profile
        self._user_profiles[profile.user_id] = profile.profile_id
        self._stats["profiles_saved"] += 1
        logger.debug("profile_saved", profile_id=str(profile.profile_id), user_id=str(profile.user_id))

    async def get_profile(self, profile_id: UUID) -> PersonalityProfile | None:
        self._stats["queries"] += 1
        return self._profiles.get(profile_id)

    async def get_profile_by_user(self, user_id: UUID) -> PersonalityProfile | None:
        self._stats["queries"] += 1
        profile_id = self._user_profiles.get(user_id)
        return self._profiles.get(profile_id) if profile_id else None

    async def list_profiles(self, limit: int = 100, offset: int = 0) -> list[PersonalityProfile]:
        self._stats["queries"] += 1
        profiles = sorted(self._profiles.values(), key=lambda p: p.created_at, reverse=True)
        return profiles[offset:offset + limit]

    async def delete_profile(self, profile_id: UUID) -> bool:
        profile = self._profiles.pop(profile_id, None)
        if profile:
            self._user_profiles.pop(profile.user_id, None)
            self._stats["deletes"] += 1
            logger.debug("profile_deleted", profile_id=str(profile_id))
            return True
        return False

    async def save_assessment(self, assessment: TraitAssessment) -> None:
        self._assessments[assessment.assessment_id] = assessment
        self._user_assessments.setdefault(assessment.user_id, [])
        if assessment.assessment_id not in self._user_assessments[assessment.user_id]:
            self._user_assessments[assessment.user_id].append(assessment.assessment_id)
        self._stats["assessments_saved"] += 1
        logger.debug("assessment_saved", assessment_id=str(assessment.assessment_id), user_id=str(assessment.user_id))

    async def get_assessment(self, assessment_id: UUID) -> TraitAssessment | None:
        self._stats["queries"] += 1
        return self._assessments.get(assessment_id)

    async def list_user_assessments(self, user_id: UUID, limit: int = 10) -> list[TraitAssessment]:
        self._stats["queries"] += 1
        assessment_ids = self._user_assessments.get(user_id, [])
        assessments = sorted([self._assessments[aid] for aid in assessment_ids if aid in self._assessments], key=lambda a: a.created_at, reverse=True)
        return assessments[:limit]

    async def save_snapshot(self, snapshot: ProfileSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._profile_snapshots.setdefault(snapshot.profile_id, [])
        if snapshot.snapshot_id not in self._profile_snapshots[snapshot.profile_id]:
            self._profile_snapshots[snapshot.profile_id].append(snapshot.snapshot_id)
        self._stats["snapshots_saved"] += 1
        logger.debug("snapshot_saved", snapshot_id=str(snapshot.snapshot_id), profile_id=str(snapshot.profile_id))

    async def get_snapshot(self, snapshot_id: UUID) -> ProfileSnapshot | None:
        self._stats["queries"] += 1
        return self._snapshots.get(snapshot_id)

    async def list_snapshots(self, profile_id: UUID, limit: int = 10) -> list[ProfileSnapshot]:
        self._stats["queries"] += 1
        snapshot_ids = self._profile_snapshots.get(profile_id, [])
        snapshots = sorted([self._snapshots[sid] for sid in snapshot_ids if sid in self._snapshots], key=lambda s: s.captured_at, reverse=True)
        return snapshots[:limit]

    async def delete_user_data(self, user_id: UUID) -> int:
        deleted_count = 0
        profile_id = self._user_profiles.pop(user_id, None)
        if profile_id:
            if profile_id in self._profiles:
                del self._profiles[profile_id]
                deleted_count += 1
            for sid in self._profile_snapshots.pop(profile_id, []):
                if sid in self._snapshots:
                    del self._snapshots[sid]
                    deleted_count += 1
        for aid in self._user_assessments.pop(user_id, []):
            if aid in self._assessments:
                del self._assessments[aid]
                deleted_count += 1
        self._stats["deletes"] += deleted_count
        logger.info("user_data_deleted", user_id=str(user_id), deleted_count=deleted_count)
        return deleted_count

    async def get_statistics(self) -> dict[str, Any]:
        return {**self._stats, "total_profiles": len(self._profiles), "total_assessments": len(self._assessments), "total_snapshots": len(self._snapshots), "total_users": len(self._user_profiles)}


class ProfileQueryBuilder:
    """Query builder for profile searches."""
    def __init__(self, repository: InMemoryPersonalityRepository) -> None:
        self._repository = repository
        self._style_type: CommunicationStyleType | None = None
        self._min_stability: Decimal | None = None
        self._min_assessments: int | None = None
        self._has_dominant_trait: PersonalityTrait | None = None
        self._date_from: datetime | None = None
        self._date_to: datetime | None = None
        self._limit: int = 100

    def with_style_type(self, style_type: CommunicationStyleType) -> ProfileQueryBuilder:
        self._style_type = style_type
        return self

    def with_min_stability(self, stability: Decimal) -> ProfileQueryBuilder:
        self._min_stability = stability
        return self

    def with_min_assessments(self, count: int) -> ProfileQueryBuilder:
        self._min_assessments = count
        return self

    def with_dominant_trait(self, trait: PersonalityTrait) -> ProfileQueryBuilder:
        self._has_dominant_trait = trait
        return self

    def created_since(self, date: datetime) -> ProfileQueryBuilder:
        self._date_from = date
        return self

    def created_until(self, date: datetime) -> ProfileQueryBuilder:
        self._date_to = date
        return self

    def limit(self, count: int) -> ProfileQueryBuilder:
        self._limit = count
        return self

    async def execute(self) -> list[PersonalityProfile]:
        results: list[PersonalityProfile] = []
        for profile in self._repository._profiles.values():
            if self._style_type and profile.communication_style and profile.communication_style.style_type != self._style_type:
                continue
            if self._min_stability is not None and profile.stability_score < self._min_stability:
                continue
            if self._min_assessments is not None and profile.assessment_count < self._min_assessments:
                continue
            if self._has_dominant_trait is not None and self._has_dominant_trait not in profile.dominant_traits:
                continue
            if self._date_from and profile.created_at < self._date_from:
                continue
            if self._date_to and profile.created_at > self._date_to:
                continue
            results.append(profile)
        return sorted(results, key=lambda p: p.created_at, reverse=True)[:self._limit]


class AssessmentQueryBuilder:
    """Query builder for assessment searches."""
    def __init__(self, repository: InMemoryPersonalityRepository) -> None:
        self._repository = repository
        self._user_id: UUID | None = None
        self._source: AssessmentSource | None = None
        self._min_confidence: Decimal | None = None
        self._date_from: datetime | None = None
        self._date_to: datetime | None = None
        self._limit: int = 100

    def for_user(self, user_id: UUID) -> AssessmentQueryBuilder:
        self._user_id = user_id
        return self

    def from_source(self, source: AssessmentSource) -> AssessmentQueryBuilder:
        self._source = source
        return self

    def with_min_confidence(self, confidence: Decimal) -> AssessmentQueryBuilder:
        self._min_confidence = confidence
        return self

    def since(self, date: datetime) -> AssessmentQueryBuilder:
        self._date_from = date
        return self

    def until(self, date: datetime) -> AssessmentQueryBuilder:
        self._date_to = date
        return self

    def limit(self, count: int) -> AssessmentQueryBuilder:
        self._limit = count
        return self

    async def execute(self) -> list[TraitAssessment]:
        results: list[TraitAssessment] = []
        for assessment in self._repository._assessments.values():
            if self._user_id and assessment.user_id != self._user_id:
                continue
            if self._source and assessment.source != self._source:
                continue
            if self._min_confidence is not None and assessment.confidence < self._min_confidence:
                continue
            if self._date_from and assessment.created_at < self._date_from:
                continue
            if self._date_to and assessment.created_at > self._date_to:
                continue
            results.append(assessment)
        return sorted(results, key=lambda a: a.created_at, reverse=True)[:self._limit]
