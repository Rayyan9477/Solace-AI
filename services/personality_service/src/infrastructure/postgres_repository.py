"""
Solace-AI Personality Service - PostgreSQL Repository Implementation.

PostgreSQL-backed repository for personality profile persistence.
Uses asyncpg for async database operations with connection pooling.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from ..domain.entities import PersonalityProfile, TraitAssessment, ProfileSnapshot
from ..domain.value_objects import OceanScores, CommunicationStyle, AssessmentMetadata
from ..schemas import AssessmentSource, CommunicationStyleType
from .repository import PersonalityRepositoryPort

try:
    from solace_infrastructure.database import ConnectionPoolManager
    from solace_infrastructure.feature_flags import FeatureFlags
except ImportError:
    ConnectionPoolManager = None
    FeatureFlags = None

if TYPE_CHECKING:
    from solace_infrastructure.postgres import PostgresClient

logger = structlog.get_logger(__name__)


def _decimal_encoder(obj: Any) -> Any:
    """JSON encoder for Decimal types."""
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class PostgresPersonalityRepository(PersonalityRepositoryPort):
    """PostgreSQL implementation of personality repository."""

    POOL_NAME = "personality_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._profiles_table = f"{schema}.personality_profiles"
        self._assessments_table = f"{schema}.trait_assessments"
        self._snapshots_table = f"{schema}.profile_snapshots"
        self._stats = {"profiles_saved": 0, "assessments_saved": 0, "snapshots_saved": 0, "queries": 0, "deletes": 0}

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._client.acquire()
        raise Exception("No database connection available.")

    async def save_profile(self, profile: PersonalityProfile) -> None:
        """Save a personality profile to PostgreSQL."""
        profile.touch()

        # Serialize complex fields to JSON
        ocean_scores_json = json.dumps(
            profile.ocean_scores.to_dict(),
            default=_decimal_encoder,
        ) if profile.ocean_scores else None

        communication_style_json = json.dumps(
            profile.communication_style.to_dict(),
            default=_decimal_encoder,
        ) if profile.communication_style else None

        assessment_history_json = json.dumps(
            [a.to_dict() for a in profile.assessment_history],
            default=_decimal_encoder,
        )

        query = f"""
            INSERT INTO {self._profiles_table} (
                profile_id, user_id, ocean_scores, communication_style,
                assessment_count, stability_score, assessment_history,
                created_at, updated_at, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
            )
            ON CONFLICT (profile_id) DO UPDATE SET
                ocean_scores = EXCLUDED.ocean_scores,
                communication_style = EXCLUDED.communication_style,
                assessment_count = EXCLUDED.assessment_count,
                stability_score = EXCLUDED.stability_score,
                assessment_history = EXCLUDED.assessment_history,
                updated_at = EXCLUDED.updated_at,
                version = EXCLUDED.version
        """
        async with self._acquire() as conn:
            await conn.execute(
                query,
                profile.profile_id,
                profile.user_id,
                ocean_scores_json,
                communication_style_json,
                profile.assessment_count,
                float(profile.stability_score),
                assessment_history_json,
                profile.created_at,
                profile.updated_at,
                profile.version,
            )
            self._stats["profiles_saved"] += 1
            logger.debug("profile_saved_postgres", profile_id=str(profile.profile_id))

    async def get_profile(self, profile_id: UUID) -> PersonalityProfile | None:
        """Get a profile by ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._profiles_table} WHERE profile_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, profile_id)
            if row is None:
                return None
            return self._row_to_profile(dict(row))

    async def get_profile_by_user(self, user_id: UUID) -> PersonalityProfile | None:
        """Get profile by user ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._profiles_table} WHERE user_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_profile(dict(row))

    async def list_profiles(self, limit: int = 100, offset: int = 0) -> list[PersonalityProfile]:
        """List profiles with pagination from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._profiles_table}
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, limit, offset)
            return [self._row_to_profile(dict(row)) for row in rows]

    async def delete_profile(self, profile_id: UUID) -> bool:
        """Delete a profile from PostgreSQL."""
        query = f"DELETE FROM {self._profiles_table} WHERE profile_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, profile_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                self._stats["deletes"] += 1
                logger.debug("profile_deleted_postgres", profile_id=str(profile_id))
            return deleted

    async def save_assessment(self, assessment: TraitAssessment) -> None:
        """Save a trait assessment to PostgreSQL."""
        ocean_scores_json = json.dumps(
            assessment.ocean_scores.to_dict(),
            default=_decimal_encoder,
        )

        metadata_json = json.dumps(
            assessment.metadata.to_dict(),
            default=_decimal_encoder,
        ) if assessment.metadata else None

        query = f"""
            INSERT INTO {self._assessments_table} (
                assessment_id, user_id, ocean_scores, source,
                metadata, evidence, created_at, version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            ON CONFLICT (assessment_id) DO UPDATE SET
                ocean_scores = EXCLUDED.ocean_scores,
                metadata = EXCLUDED.metadata,
                evidence = EXCLUDED.evidence,
                version = EXCLUDED.version
        """
        async with self._acquire() as conn:
            await conn.execute(
                query,
                assessment.assessment_id,
                assessment.user_id,
                ocean_scores_json,
                assessment.source.value,
                metadata_json,
                assessment.evidence,
                assessment.created_at,
                assessment.version,
            )
            self._stats["assessments_saved"] += 1
            logger.debug("assessment_saved_postgres", assessment_id=str(assessment.assessment_id))

    async def get_assessment(self, assessment_id: UUID) -> TraitAssessment | None:
        """Get assessment by ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._assessments_table} WHERE assessment_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, assessment_id)
            if row is None:
                return None
            return self._row_to_assessment(dict(row))

    async def list_user_assessments(self, user_id: UUID, limit: int = 10) -> list[TraitAssessment]:
        """List assessments for a user from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._assessments_table}
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id, limit)
            return [self._row_to_assessment(dict(row)) for row in rows]

    async def save_snapshot(self, snapshot: ProfileSnapshot) -> None:
        """Save a profile snapshot to PostgreSQL."""
        traits_json = json.dumps(
            {k.value: str(v) for k, v in snapshot.traits.items()},
            default=_decimal_encoder,
        ) if snapshot.traits else None

        communication_style_json = json.dumps(
            snapshot.communication_style.to_dict(),
            default=_decimal_encoder,
        ) if snapshot.communication_style else None

        dominant_traits_json = json.dumps(
            [t.value for t in snapshot.dominant_traits],
        ) if snapshot.dominant_traits else None

        query = f"""
            INSERT INTO {self._snapshots_table} (
                snapshot_id, profile_id, traits, communication_style,
                stability_score, assessment_count, dominant_traits,
                captured_at, trigger_reason
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            )
            ON CONFLICT (snapshot_id) DO NOTHING
        """
        async with self._acquire() as conn:
            await conn.execute(
                query,
                snapshot.snapshot_id,
                snapshot.profile_id,
                traits_json,
                communication_style_json,
                float(snapshot.stability_score),
                snapshot.assessment_count,
                dominant_traits_json,
                snapshot.captured_at,
                snapshot.trigger_reason,
            )
            self._stats["snapshots_saved"] += 1
            logger.debug("snapshot_saved_postgres", snapshot_id=str(snapshot.snapshot_id))

    async def get_snapshot(self, snapshot_id: UUID) -> ProfileSnapshot | None:
        """Get snapshot by ID from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"SELECT * FROM {self._snapshots_table} WHERE snapshot_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, snapshot_id)
            if row is None:
                return None
            return self._row_to_snapshot(dict(row))

    async def list_snapshots(self, profile_id: UUID, limit: int = 10) -> list[ProfileSnapshot]:
        """List snapshots for a profile from PostgreSQL."""
        self._stats["queries"] += 1
        query = f"""
            SELECT * FROM {self._snapshots_table}
            WHERE profile_id = $1
            ORDER BY captured_at DESC
            LIMIT $2
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, profile_id, limit)
            return [self._row_to_snapshot(dict(row)) for row in rows]

    async def delete_user_data(self, user_id: UUID) -> int:
        """Delete all data for a user (GDPR) from PostgreSQL."""
        deleted_count = 0

        # Get profile ID first
        query = f"SELECT profile_id FROM {self._profiles_table} WHERE user_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            profile_id = row["profile_id"] if row else None

        # Delete snapshots
        if profile_id:
            snapshots_query = f"DELETE FROM {self._snapshots_table} WHERE profile_id = $1"
            async with self._acquire() as conn:
                result = await conn.execute(snapshots_query, profile_id)
                deleted_count += int(result.split()[-1])

        # Delete assessments
        assessments_query = f"DELETE FROM {self._assessments_table} WHERE user_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(assessments_query, user_id)
            deleted_count += int(result.split()[-1])

        # Delete profile
        profiles_query = f"DELETE FROM {self._profiles_table} WHERE user_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(profiles_query, user_id)
            deleted_count += int(result.split()[-1])

        self._stats["deletes"] += deleted_count
        logger.info("user_data_deleted_postgres", user_id=str(user_id), deleted_count=deleted_count)
        return deleted_count

    async def get_statistics(self) -> dict[str, Any]:
        """Get repository statistics from PostgreSQL."""
        profiles_query = f"SELECT COUNT(*) FROM {self._profiles_table}"
        assessments_query = f"SELECT COUNT(*) FROM {self._assessments_table}"
        snapshots_query = f"SELECT COUNT(*) FROM {self._snapshots_table}"
        users_query = f"SELECT COUNT(DISTINCT user_id) FROM {self._profiles_table}"

        async with self._acquire() as conn:
            profiles_count = await conn.fetchval(profiles_query)
            assessments_count = await conn.fetchval(assessments_query)
            snapshots_count = await conn.fetchval(snapshots_query)
            users_count = await conn.fetchval(users_query)

        return {
            **self._stats,
            "total_profiles": profiles_count or 0,
            "total_assessments": assessments_count or 0,
            "total_snapshots": snapshots_count or 0,
            "total_users": users_count or 0,
        }

    def _row_to_profile(self, row: dict[str, Any]) -> PersonalityProfile:
        """Convert a database row to a PersonalityProfile entity."""
        ocean_scores_data = row.get("ocean_scores")
        if isinstance(ocean_scores_data, str):
            ocean_scores_data = json.loads(ocean_scores_data)
        ocean_scores = OceanScores.from_dict(ocean_scores_data) if ocean_scores_data else None

        comm_style_data = row.get("communication_style")
        if isinstance(comm_style_data, str):
            comm_style_data = json.loads(comm_style_data)
        communication_style = CommunicationStyle.from_dict(comm_style_data) if comm_style_data else None

        history_data = row.get("assessment_history", [])
        if isinstance(history_data, str):
            history_data = json.loads(history_data)
        assessment_history = [TraitAssessment.from_dict(a) for a in history_data]

        return PersonalityProfile(
            profile_id=row["profile_id"],
            user_id=row["user_id"],
            ocean_scores=ocean_scores,
            communication_style=communication_style,
            assessment_count=row.get("assessment_count", 0),
            stability_score=Decimal(str(row.get("stability_score", "0.0"))),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
            version=row.get("version", 1),
            assessment_history=assessment_history,
        )

    def _row_to_assessment(self, row: dict[str, Any]) -> TraitAssessment:
        """Convert a database row to a TraitAssessment entity."""
        ocean_scores_data = row.get("ocean_scores")
        if isinstance(ocean_scores_data, str):
            ocean_scores_data = json.loads(ocean_scores_data)
        ocean_scores = OceanScores.from_dict(ocean_scores_data)

        metadata_data = row.get("metadata")
        if isinstance(metadata_data, str):
            metadata_data = json.loads(metadata_data)
        metadata = AssessmentMetadata.from_dict(metadata_data) if metadata_data else None

        return TraitAssessment(
            assessment_id=row["assessment_id"],
            user_id=row["user_id"],
            ocean_scores=ocean_scores,
            source=AssessmentSource(row.get("source", "text_analysis")),
            metadata=metadata,
            evidence=list(row.get("evidence", [])),
            created_at=row.get("created_at", datetime.now(timezone.utc)),
            version=row.get("version", 1),
        )

    def _row_to_snapshot(self, row: dict[str, Any]) -> ProfileSnapshot:
        """Convert a database row to a ProfileSnapshot entity."""
        from ..schemas import PersonalityTrait

        traits_data = row.get("traits")
        if isinstance(traits_data, str):
            traits_data = json.loads(traits_data)
        traits = {}
        if traits_data:
            for k, v in traits_data.items():
                traits[PersonalityTrait(k)] = Decimal(v)

        comm_style_data = row.get("communication_style")
        if isinstance(comm_style_data, str):
            comm_style_data = json.loads(comm_style_data)
        communication_style = CommunicationStyle.from_dict(comm_style_data) if comm_style_data else None

        dominant_traits_data = row.get("dominant_traits", [])
        if isinstance(dominant_traits_data, str):
            dominant_traits_data = json.loads(dominant_traits_data)
        dominant_traits = [PersonalityTrait(t) for t in dominant_traits_data] if dominant_traits_data else []

        return ProfileSnapshot(
            snapshot_id=row["snapshot_id"],
            profile_id=row["profile_id"],
            traits=traits,
            communication_style=communication_style,
            stability_score=Decimal(str(row.get("stability_score", "0.0"))),
            assessment_count=row.get("assessment_count", 0),
            dominant_traits=dominant_traits,
            captured_at=row.get("captured_at", datetime.now(timezone.utc)),
            trigger_reason=row.get("trigger_reason"),
        )
