"""
Test fixtures for User Service repository layer.

Contains in-memory repository implementations for use in tests only.
These are NOT suitable for production - data is lost on restart.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import structlog

from src.domain.entities import User, UserPreferences
from src.domain.value_objects import (
    UserRole,
    AccountStatus,
    ConsentRecord,
)
from src.infrastructure.repository import (
    UserRepository,
    UserPreferencesRepository,
    ConsentRepository,
    DuplicateEntityError,
    EntityNotFoundError,
)

logger = structlog.get_logger(__name__)


class InMemoryUserRepository(UserRepository):
    """
    In-memory user repository for testing.

    NOT suitable for production - data is lost on restart.
    Thread-safe using asyncio locks.
    """

    def __init__(self) -> None:
        self._users: dict[UUID, User] = {}
        self._users_by_email: dict[str, UUID] = {}
        self._assignments: dict[UUID, set[UUID]] = {}  # clinician_id -> set of patient_ids
        self._lock = asyncio.Lock()

    async def save(self, user: User) -> User:
        """Save a new user."""
        async with self._lock:
            email_lower = user.email.lower()
            if email_lower in self._users_by_email:
                raise DuplicateEntityError("User", user.email)

            self._users[user.user_id] = user
            self._users_by_email[email_lower] = user.user_id

            logger.debug(
                "user_saved",
                user_id=str(user.user_id),
                email=user.email,
            )
            return user

    async def get_by_id(self, user_id: UUID) -> User | None:
        """Get user by ID."""
        user = self._users.get(user_id)
        if user and user.deleted_at is None:
            return user
        return None

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        user_id = self._users_by_email.get(email.lower())
        if user_id:
            return await self.get_by_id(user_id)
        return None

    async def update(self, user: User) -> User:
        """Update an existing user."""
        async with self._lock:
            if user.user_id not in self._users:
                raise EntityNotFoundError("User", user.user_id)

            old_user = self._users[user.user_id]

            # Handle email change
            if old_user.email.lower() != user.email.lower():
                del self._users_by_email[old_user.email.lower()]
                self._users_by_email[user.email.lower()] = user.user_id

            self._users[user.user_id] = user

            logger.debug("user_updated", user_id=str(user.user_id))
            return user

    async def delete(self, user_id: UUID) -> bool:
        """Delete a user by ID (hard delete)."""
        async with self._lock:
            if user_id not in self._users:
                return False

            user = self._users[user_id]
            del self._users_by_email[user.email.lower()]
            del self._users[user_id]

            logger.debug("user_deleted", user_id=str(user_id))
            return True

    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[User]:
        """List users with pagination."""
        if include_deleted:
            users = list(self._users.values())
        else:
            users = [u for u in self._users.values() if u.deleted_at is None]

        users.sort(key=lambda u: u.created_at, reverse=True)
        return users[offset : offset + limit]

    async def count(self, include_deleted: bool = False) -> int:
        """Count total users."""
        if include_deleted:
            return len(self._users)
        return len([u for u in self._users.values() if u.deleted_at is None])

    async def find_on_call_clinicians(self) -> list[User]:
        """Find all clinicians currently on-call for crisis notifications."""
        return [
            user for user in self._users.values()
            if user.deleted_at is None
            and user.status == AccountStatus.ACTIVE
            and user.role == UserRole.CLINICIAN
            and user.is_on_call
        ]

    async def find_by_role(self, role: str, active_only: bool = True) -> list[User]:
        """Find users by role."""
        try:
            role_enum = UserRole(role)
        except ValueError:
            return []

        users = [
            user for user in self._users.values()
            if user.role == role_enum
            and user.deleted_at is None
        ]

        if active_only:
            users = [u for u in users if u.status == AccountStatus.ACTIVE]

        return users

    async def assign_patient_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Assign a patient to a clinician."""
        async with self._lock:
            if clinician_id not in self._assignments:
                self._assignments[clinician_id] = set()
            self._assignments[clinician_id].add(patient_id)
            return True

    async def unassign_patient_from_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Remove a patient assignment from a clinician."""
        async with self._lock:
            patients = self._assignments.get(clinician_id)
            if patients and patient_id in patients:
                patients.discard(patient_id)
                return True
            return False

    async def is_patient_assigned_to_clinician(
        self, clinician_id: UUID, patient_id: UUID,
    ) -> bool:
        """Check if a specific patient is assigned to a specific clinician."""
        patients = self._assignments.get(clinician_id, set())
        return patient_id in patients

    async def get_assigned_patients(self, clinician_id: UUID) -> list[UUID]:
        """Get all patient IDs assigned to a clinician."""
        return list(self._assignments.get(clinician_id, set()))

    async def get_assigned_clinician(self, patient_id: UUID) -> UUID | None:
        """Get the clinician assigned to a patient, if any."""
        for clinician_id, patients in self._assignments.items():
            if patient_id in patients:
                return clinician_id
        return None

    # --- Testing Utilities ---

    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._users.clear()
        self._users_by_email.clear()
        self._assignments.clear()

    async def search_users(self, query: str, limit: int = 20) -> list[User]:
        """
        Search users by email or display name.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching users
        """
        query_lower = query.lower()
        matches = [
            u
            for u in self._users.values()
            if u.deleted_at is None
            and (query_lower in u.email.lower() or query_lower in u.display_name.lower())
        ]
        return matches[:limit]


class InMemoryUserPreferencesRepository(UserPreferencesRepository):
    """
    In-memory preferences repository for testing.

    NOT suitable for production - data is lost on restart.
    """

    def __init__(self) -> None:
        self._preferences: dict[UUID, UserPreferences] = {}
        self._lock = asyncio.Lock()

    async def save(self, preferences: UserPreferences) -> UserPreferences:
        """Save user preferences."""
        async with self._lock:
            self._preferences[preferences.user_id] = preferences
            logger.debug("preferences_saved", user_id=str(preferences.user_id))
            return preferences

    async def get_by_user_id(self, user_id: UUID) -> UserPreferences | None:
        """Get preferences for a user."""
        return self._preferences.get(user_id)

    async def update(self, preferences: UserPreferences) -> UserPreferences:
        """Update existing preferences."""
        async with self._lock:
            self._preferences[preferences.user_id] = preferences
            logger.debug("preferences_updated", user_id=str(preferences.user_id))
            return preferences

    async def delete(self, user_id: UUID) -> bool:
        """Delete preferences for a user."""
        async with self._lock:
            if user_id in self._preferences:
                del self._preferences[user_id]
                return True
            return False

    # --- Testing Utilities ---

    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._preferences.clear()

    async def get_users_with_notification_enabled(
        self, channel: str
    ) -> list[UUID]:
        """Get users with specific notification channel enabled."""
        users = []
        for user_id, prefs in self._preferences.items():
            if channel == "email" and prefs.notification_email:
                users.append(user_id)
            elif channel == "sms" and prefs.notification_sms:
                users.append(user_id)
            elif channel == "push" and prefs.notification_push:
                users.append(user_id)
        return users


class InMemoryConsentRepository(ConsentRepository):
    """
    In-memory consent repository for testing.

    NOT suitable for production - data is lost on restart.
    """

    def __init__(self) -> None:
        self._consents: dict[UUID, list[ConsentRecord]] = {}
        self._consents_by_id: dict[UUID, ConsentRecord] = {}
        self._lock = asyncio.Lock()

    async def save(self, consent: ConsentRecord) -> ConsentRecord:
        """Save a consent record."""
        async with self._lock:
            if consent.user_id not in self._consents:
                self._consents[consent.user_id] = []

            self._consents[consent.user_id].append(consent)
            self._consents_by_id[consent.consent_id] = consent

            logger.debug(
                "consent_saved",
                user_id=str(consent.user_id),
                consent_type=consent.consent_type.value,
            )
            return consent

    async def get_by_user_id(self, user_id: UUID) -> list[ConsentRecord]:
        """Get all consent records for a user."""
        records = self._consents.get(user_id, [])
        # Return sorted by timestamp
        return sorted(records, key=lambda c: c.granted_at)

    async def get_by_id(self, consent_id: UUID) -> ConsentRecord | None:
        """Get consent record by ID."""
        return self._consents_by_id.get(consent_id)

    # --- Testing Utilities ---

    def clear(self) -> None:
        """Clear all data (for testing)."""
        self._consents.clear()
        self._consents_by_id.clear()
