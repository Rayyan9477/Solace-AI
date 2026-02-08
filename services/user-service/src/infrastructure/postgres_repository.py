"""
Solace-AI User Service - PostgreSQL Repository Implementations.

PostgreSQL-backed repositories for user data persistence.
Uses asyncpg for async database operations with connection pooling.

Architecture Layer: Infrastructure
Principles: Repository Pattern, Dependency Inversion
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

try:
    from solace_infrastructure.database import ConnectionPoolManager
    from solace_infrastructure.feature_flags import FeatureFlags
except ImportError:
    ConnectionPoolManager = None
    FeatureFlags = None

from ..domain.entities import User, UserPreferences
from ..domain.value_objects import AccountStatus, ConsentRecord, ConsentType, UserRole
from .repository import (
    ConsentRepository,
    DuplicateEntityError,
    EntityNotFoundError,
    UserPreferencesRepository,
    UserRepository,
)

if TYPE_CHECKING:
    from solace_infrastructure.postgres import PostgresClient

logger = structlog.get_logger(__name__)


class PostgresUserRepository(UserRepository):
    """PostgreSQL implementation of user repository."""

    POOL_NAME = "user_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.users"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._client.acquire()
        raise Exception("No database connection available.")

    async def save(self, user: User) -> User:
        """Save a new user to PostgreSQL."""
        query = f"""
            INSERT INTO {self._table} (
                user_id, email, password_hash, display_name, role, status,
                is_on_call, phone_number, timezone, locale, avatar_url, bio,
                email_verified, email_verification_token, login_attempts,
                locked_until, created_at, updated_at, last_login, deleted_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19, $20
            )
            RETURNING *
        """
        try:
            async with self._acquire() as conn:
                await conn.fetchrow(
                    query,
                    user.user_id,
                    user.email,
                    user.password_hash,
                    user.display_name,
                    user.role.value,
                    user.status.value,
                    user.is_on_call,
                    user.phone_number,
                    user.timezone,
                    user.locale,
                    user.avatar_url,
                    user.bio,
                    user.email_verified,
                    user.email_verification_token,
                    user.login_attempts,
                    user.locked_until,
                    user.created_at,
                    user.updated_at,
                    user.last_login,
                    user.deleted_at,
                )
                logger.debug("user_saved_postgres", user_id=str(user.user_id), email=user.email)
                return user
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise DuplicateEntityError("User", user.email)
            raise

    async def get_by_id(self, user_id: UUID) -> User | None:
        """Get user by ID from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1 AND deleted_at IS NULL
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE LOWER(email) = LOWER($1) AND deleted_at IS NULL
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, email)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def update(self, user: User) -> User:
        """Update an existing user in PostgreSQL."""
        query = f"""
            UPDATE {self._table} SET
                email = $2,
                password_hash = $3,
                display_name = $4,
                role = $5,
                status = $6,
                is_on_call = $7,
                phone_number = $8,
                timezone = $9,
                locale = $10,
                avatar_url = $11,
                bio = $12,
                email_verified = $13,
                email_verification_token = $14,
                login_attempts = $15,
                locked_until = $16,
                updated_at = $17,
                last_login = $18,
                deleted_at = $19
            WHERE user_id = $1
            RETURNING *
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                query,
                user.user_id,
                user.email,
                user.password_hash,
                user.display_name,
                user.role.value,
                user.status.value,
                user.is_on_call,
                user.phone_number,
                user.timezone,
                user.locale,
                user.avatar_url,
                user.bio,
                user.email_verified,
                user.email_verification_token,
                user.login_attempts,
                user.locked_until,
                user.updated_at,
                user.last_login,
                user.deleted_at,
            )
            if row is None:
                raise EntityNotFoundError("User", user.user_id)
            logger.debug("user_updated_postgres", user_id=str(user.user_id))
            return self._row_to_entity(dict(row))

    async def delete(self, user_id: UUID) -> bool:
        """Delete a user by ID (hard delete) from PostgreSQL."""
        query = f"DELETE FROM {self._table} WHERE user_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, user_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("user_deleted_postgres", user_id=str(user_id))
            return deleted

    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[User]:
        """List users with pagination from PostgreSQL."""
        if include_deleted:
            query = f"""
                SELECT * FROM {self._table}
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
        else:
            query = f"""
                SELECT * FROM {self._table}
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, limit, offset)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def count(self, include_deleted: bool = False) -> int:
        """Count total users in PostgreSQL."""
        if include_deleted:
            query = f"SELECT COUNT(*) FROM {self._table}"
        else:
            query = f"SELECT COUNT(*) FROM {self._table} WHERE deleted_at IS NULL"
        async with self._acquire() as conn:
            count = await conn.fetchval(query)
            return count or 0

    async def find_on_call_clinicians(self) -> list[User]:
        """Find all clinicians currently on-call for crisis notifications."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE deleted_at IS NULL
              AND status = $1
              AND role = $2
              AND is_on_call = true
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(
                query,
                AccountStatus.ACTIVE.value,
                UserRole.CLINICIAN.value,
            )
            return [self._row_to_entity(dict(row)) for row in rows]

    async def find_by_role(self, role: str, active_only: bool = True) -> list[User]:
        """Find users by role from PostgreSQL."""
        if active_only:
            query = f"""
                SELECT * FROM {self._table}
                WHERE role = $1 AND deleted_at IS NULL AND status = $2
            """
            async with self._acquire() as conn:
                rows = await conn.fetch(query, role, AccountStatus.ACTIVE.value)
        else:
            query = f"""
                SELECT * FROM {self._table}
                WHERE role = $1 AND deleted_at IS NULL
            """
            async with self._acquire() as conn:
                rows = await conn.fetch(query, role)
        return [self._row_to_entity(dict(row)) for row in rows]

    async def assign_patient_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Assign a patient to a clinician in PostgreSQL."""
        query = f"""
            INSERT INTO {self._schema}.clinician_patient_assignments
                (clinician_id, patient_id, assigned_at)
            VALUES ($1, $2, NOW())
            ON CONFLICT (clinician_id, patient_id) DO NOTHING
        """
        async with self._acquire() as conn:
            await conn.execute(query, clinician_id, patient_id)
            return True

    async def unassign_patient_from_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Remove a patient assignment from a clinician."""
        query = f"""
            DELETE FROM {self._schema}.clinician_patient_assignments
            WHERE clinician_id = $1 AND patient_id = $2
        """
        async with self._acquire() as conn:
            result = await conn.execute(query, clinician_id, patient_id)
            return result == "DELETE 1"

    async def is_patient_assigned_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Check if a specific patient is assigned to a specific clinician."""
        query = f"""
            SELECT 1 FROM {self._schema}.clinician_patient_assignments
            WHERE clinician_id = $1 AND patient_id = $2
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, clinician_id, patient_id)
            return row is not None

    async def get_assigned_patients(self, clinician_id: UUID) -> list[UUID]:
        """Get all patient IDs assigned to a clinician."""
        query = f"""
            SELECT patient_id FROM {self._schema}.clinician_patient_assignments
            WHERE clinician_id = $1
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, clinician_id)
            return [row["patient_id"] for row in rows]

    async def get_assigned_clinician(self, patient_id: UUID) -> UUID | None:
        """Get the clinician assigned to a patient, if any."""
        query = f"""
            SELECT clinician_id FROM {self._schema}.clinician_patient_assignments
            WHERE patient_id = $1
            LIMIT 1
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, patient_id)
            return row["clinician_id"] if row else None

    def _row_to_entity(self, row: dict[str, Any]) -> User:
        """Convert a database row to a User entity."""
        return User(
            user_id=row["user_id"],
            email=row["email"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            role=UserRole(row["role"]),
            status=AccountStatus(row["status"]),
            is_on_call=row["is_on_call"],
            phone_number=row.get("phone_number"),
            timezone=row["timezone"],
            locale=row["locale"],
            avatar_url=row.get("avatar_url"),
            bio=row.get("bio"),
            email_verified=row["email_verified"],
            email_verification_token=row.get("email_verification_token"),
            login_attempts=row["login_attempts"],
            locked_until=row.get("locked_until"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login=row.get("last_login"),
            deleted_at=row.get("deleted_at"),
        )


class PostgresUserPreferencesRepository(UserPreferencesRepository):
    """PostgreSQL implementation of user preferences repository."""

    POOL_NAME = "user_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.user_preferences"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._client.acquire()
        raise Exception("No database connection available.")

    async def save(self, preferences: UserPreferences) -> UserPreferences:
        """Save user preferences to PostgreSQL."""
        query = f"""
            INSERT INTO {self._table} (
                user_id, notification_email, notification_sms, notification_push,
                notification_channels, session_reminders, progress_updates,
                marketing_emails, data_sharing_research, data_sharing_improvement,
                theme, language, accessibility_high_contrast, accessibility_large_text,
                accessibility_screen_reader, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (user_id) DO UPDATE SET
                notification_email = EXCLUDED.notification_email,
                notification_sms = EXCLUDED.notification_sms,
                notification_push = EXCLUDED.notification_push,
                notification_channels = EXCLUDED.notification_channels,
                session_reminders = EXCLUDED.session_reminders,
                progress_updates = EXCLUDED.progress_updates,
                marketing_emails = EXCLUDED.marketing_emails,
                data_sharing_research = EXCLUDED.data_sharing_research,
                data_sharing_improvement = EXCLUDED.data_sharing_improvement,
                theme = EXCLUDED.theme,
                language = EXCLUDED.language,
                accessibility_high_contrast = EXCLUDED.accessibility_high_contrast,
                accessibility_large_text = EXCLUDED.accessibility_large_text,
                accessibility_screen_reader = EXCLUDED.accessibility_screen_reader,
                updated_at = EXCLUDED.updated_at
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                preferences.user_id,
                preferences.notification_email,
                preferences.notification_sms,
                preferences.notification_push,
                json.dumps(preferences.notification_channels),
                preferences.session_reminders,
                preferences.progress_updates,
                preferences.marketing_emails,
                preferences.data_sharing_research,
                preferences.data_sharing_improvement,
                preferences.theme,
                preferences.language,
                preferences.accessibility_high_contrast,
                preferences.accessibility_large_text,
                preferences.accessibility_screen_reader,
                preferences.created_at,
                preferences.updated_at,
            )
            logger.debug("preferences_saved_postgres", user_id=str(preferences.user_id))
            return preferences

    async def get_by_user_id(self, user_id: UUID) -> UserPreferences | None:
        """Get preferences for a user from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE user_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    async def update(self, preferences: UserPreferences) -> UserPreferences:
        """Update existing preferences in PostgreSQL."""
        query = f"""
            UPDATE {self._table} SET
                notification_email = $2,
                notification_sms = $3,
                notification_push = $4,
                notification_channels = $5,
                session_reminders = $6,
                progress_updates = $7,
                marketing_emails = $8,
                data_sharing_research = $9,
                data_sharing_improvement = $10,
                theme = $11,
                language = $12,
                accessibility_high_contrast = $13,
                accessibility_large_text = $14,
                accessibility_screen_reader = $15,
                updated_at = $16
            WHERE user_id = $1
            RETURNING *
        """
        async with self._acquire() as conn:
            row = await conn.fetchrow(
                query,
                preferences.user_id,
                preferences.notification_email,
                preferences.notification_sms,
                preferences.notification_push,
                json.dumps(preferences.notification_channels),
                preferences.session_reminders,
                preferences.progress_updates,
                preferences.marketing_emails,
                preferences.data_sharing_research,
                preferences.data_sharing_improvement,
                preferences.theme,
                preferences.language,
                preferences.accessibility_high_contrast,
                preferences.accessibility_large_text,
                preferences.accessibility_screen_reader,
                preferences.updated_at,
            )
            if row is None:
                raise EntityNotFoundError("UserPreferences", preferences.user_id)
            logger.debug("preferences_updated_postgres", user_id=str(preferences.user_id))
            return self._row_to_entity(dict(row))

    async def delete(self, user_id: UUID) -> bool:
        """Delete preferences for a user from PostgreSQL."""
        query = f"DELETE FROM {self._table} WHERE user_id = $1"
        async with self._acquire() as conn:
            result = await conn.execute(query, user_id)
            deleted = result.split()[-1] != "0"
            if deleted:
                logger.debug("preferences_deleted_postgres", user_id=str(user_id))
            return deleted

    def _row_to_entity(self, row: dict[str, Any]) -> UserPreferences:
        """Convert a database row to a UserPreferences entity."""
        channels = row.get("notification_channels", [])
        if isinstance(channels, str):
            channels = json.loads(channels)

        return UserPreferences(
            user_id=row["user_id"],
            notification_email=row["notification_email"],
            notification_sms=row["notification_sms"],
            notification_push=row["notification_push"],
            notification_channels=channels,
            session_reminders=row["session_reminders"],
            progress_updates=row["progress_updates"],
            marketing_emails=row["marketing_emails"],
            data_sharing_research=row["data_sharing_research"],
            data_sharing_improvement=row["data_sharing_improvement"],
            theme=row["theme"],
            language=row["language"],
            accessibility_high_contrast=row["accessibility_high_contrast"],
            accessibility_large_text=row["accessibility_large_text"],
            accessibility_screen_reader=row["accessibility_screen_reader"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresConsentRepository(ConsentRepository):
    """PostgreSQL implementation of consent repository."""

    POOL_NAME = "user_db"

    def __init__(self, client: PostgresClient, schema: str = "public") -> None:
        self._client = client
        self._schema = schema
        self._table = f"{schema}.consent_records"

    def _acquire(self):
        """Get connection from ConnectionPoolManager or legacy client."""
        if ConnectionPoolManager is not None and FeatureFlags is not None and FeatureFlags.is_enabled("use_connection_pool_manager"):
            return ConnectionPoolManager.acquire(self.POOL_NAME)
        if self._client is not None:
            return self._client.acquire()
        raise Exception("No database connection available.")

    async def save(self, consent: ConsentRecord) -> ConsentRecord:
        """Save a consent record to PostgreSQL."""
        query = f"""
            INSERT INTO {self._table} (
                consent_id, user_id, consent_type, granted, granted_at,
                ip_address, user_agent, consent_version, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            )
            RETURNING *
        """
        async with self._acquire() as conn:
            await conn.fetchrow(
                query,
                consent.consent_id,
                consent.user_id,
                consent.consent_type.value,
                consent.granted,
                consent.granted_at,
                consent.ip_address,
                consent.user_agent,
                consent.consent_version,
                json.dumps(consent.metadata) if consent.metadata else None,
            )
            logger.debug(
                "consent_saved_postgres",
                user_id=str(consent.user_id),
                consent_type=consent.consent_type.value,
            )
            return consent

    async def get_by_user_id(self, user_id: UUID) -> list[ConsentRecord]:
        """Get all consent records for a user from PostgreSQL."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE user_id = $1
            ORDER BY granted_at ASC
        """
        async with self._acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_entity(dict(row)) for row in rows]

    async def get_by_id(self, consent_id: UUID) -> ConsentRecord | None:
        """Get consent record by ID from PostgreSQL."""
        query = f"SELECT * FROM {self._table} WHERE consent_id = $1"
        async with self._acquire() as conn:
            row = await conn.fetchrow(query, consent_id)
            if row is None:
                return None
            return self._row_to_entity(dict(row))

    def _row_to_entity(self, row: dict[str, Any]) -> ConsentRecord:
        """Convert a database row to a ConsentRecord entity."""
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return ConsentRecord(
            consent_id=row["consent_id"],
            user_id=row["user_id"],
            consent_type=ConsentType(row["consent_type"]),
            granted=row["granted"],
            granted_at=row["granted_at"],
            ip_address=row.get("ip_address"),
            user_agent=row.get("user_agent"),
            consent_version=row.get("consent_version"),
            metadata=metadata or {},
        )
