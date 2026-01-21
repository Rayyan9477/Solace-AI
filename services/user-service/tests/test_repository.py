"""
Tests for User Service repository layer.

Tests data persistence operations for users, preferences, and consent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from src.domain.entities import User, UserPreferences
from src.domain.value_objects import UserRole, AccountStatus, ConsentType, ConsentRecord
from src.infrastructure.repository import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    InMemoryConsentRepository,
    RepositoryFactory,
    EntityNotFoundError,
    DuplicateEntityError,
)


class TestInMemoryUserRepository:
    """Tests for InMemoryUserRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryUserRepository:
        """Create repository instance."""
        return InMemoryUserRepository()

    @pytest.fixture
    def sample_user(self) -> User:
        """Create a sample user."""
        return User(
            email="test@example.com",
            password_hash="hashed_password",
            display_name="Test User",
            role=UserRole.USER,
            status=AccountStatus.ACTIVE,
        )

    @pytest.mark.asyncio
    async def test_save_user(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test saving a user."""
        saved_user = await repository.save(sample_user)

        assert saved_user.user_id == sample_user.user_id
        assert saved_user.email == sample_user.email

    @pytest.mark.asyncio
    async def test_save_duplicate_email_raises_error(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test saving user with duplicate email raises error."""
        await repository.save(sample_user)

        duplicate_user = User(
            email="test@example.com",  # Same email
            password_hash="different_hash",
            display_name="Another User",
        )

        with pytest.raises(DuplicateEntityError):
            await repository.save(duplicate_user)

    @pytest.mark.asyncio
    async def test_get_by_id(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test getting user by ID."""
        await repository.save(sample_user)

        found_user = await repository.get_by_id(sample_user.user_id)

        assert found_user is not None
        assert found_user.user_id == sample_user.user_id

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test getting non-existent user returns None."""
        found_user = await repository.get_by_id(uuid4())

        assert found_user is None

    @pytest.mark.asyncio
    async def test_get_by_email(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test getting user by email."""
        await repository.save(sample_user)

        found_user = await repository.get_by_email("test@example.com")

        assert found_user is not None
        assert found_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_by_email_case_insensitive(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test email lookup is case-insensitive."""
        await repository.save(sample_user)

        found_user = await repository.get_by_email("TEST@EXAMPLE.COM")

        assert found_user is not None
        assert found_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_update_user(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test updating a user."""
        await repository.save(sample_user)

        sample_user.display_name = "Updated Name"
        updated_user = await repository.update(sample_user)

        assert updated_user.display_name == "Updated Name"

        # Verify persistence
        found_user = await repository.get_by_id(sample_user.user_id)
        assert found_user.display_name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_user_raises_error(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test updating non-existent user raises error."""
        with pytest.raises(EntityNotFoundError):
            await repository.update(sample_user)

    @pytest.mark.asyncio
    async def test_delete_user(
        self, repository: InMemoryUserRepository, sample_user: User
    ) -> None:
        """Test deleting a user."""
        await repository.save(sample_user)

        result = await repository.delete(sample_user.user_id)

        assert result is True

        # Verify user is gone
        found_user = await repository.get_by_id(sample_user.user_id)
        assert found_user is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_user(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test deleting non-existent user returns False."""
        result = await repository.delete(uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_list_users(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test listing users with pagination."""
        # Create multiple users
        for i in range(5):
            user = User(
                email=f"user{i}@example.com",
                password_hash="hash",
                display_name=f"User {i}",
            )
            await repository.save(user)

        # List with limit
        users = await repository.list_users(limit=3)
        assert len(users) == 3

        # List with offset
        users = await repository.list_users(limit=3, offset=3)
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_list_users_excludes_deleted(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test listing excludes soft-deleted users."""
        # Create users
        active_user = User(
            email="active@example.com",
            password_hash="hash",
            display_name="Active User",
        )
        deleted_user = User(
            email="deleted@example.com",
            password_hash="hash",
            display_name="Deleted User",
            deleted_at=datetime.now(timezone.utc),
        )

        await repository.save(active_user)
        await repository.save(deleted_user)

        users = await repository.list_users()

        assert len(users) == 1
        assert users[0].email == "active@example.com"

    @pytest.mark.asyncio
    async def test_count_users(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test counting users."""
        for i in range(3):
            user = User(
                email=f"count{i}@example.com",
                password_hash="hash",
                display_name=f"User {i}",
            )
            await repository.save(user)

        count = await repository.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_search_users(
        self, repository: InMemoryUserRepository
    ) -> None:
        """Test searching users."""
        user1 = User(
            email="john@example.com",
            password_hash="hash",
            display_name="John Doe",
        )
        user2 = User(
            email="jane@example.com",
            password_hash="hash",
            display_name="Jane Doe",
        )
        user3 = User(
            email="bob@example.com",
            password_hash="hash",
            display_name="Bob Smith",
        )

        await repository.save(user1)
        await repository.save(user2)
        await repository.save(user3)

        # Search by name
        results = await repository.search_users("doe")
        assert len(results) == 2

        # Search by email
        results = await repository.search_users("bob")
        assert len(results) == 1


class TestInMemoryUserPreferencesRepository:
    """Tests for InMemoryUserPreferencesRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryUserPreferencesRepository:
        """Create repository instance."""
        return InMemoryUserPreferencesRepository()

    @pytest.fixture
    def user_id(self) -> UUID:
        """Create a sample user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_save_preferences(
        self, repository: InMemoryUserPreferencesRepository, user_id: UUID
    ) -> None:
        """Test saving preferences."""
        preferences = UserPreferences(user_id=user_id)

        saved = await repository.save(preferences)

        assert saved.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_by_user_id(
        self, repository: InMemoryUserPreferencesRepository, user_id: UUID
    ) -> None:
        """Test getting preferences by user ID."""
        preferences = UserPreferences(user_id=user_id)
        await repository.save(preferences)

        found = await repository.get_by_user_id(user_id)

        assert found is not None
        assert found.user_id == user_id

    @pytest.mark.asyncio
    async def test_update_preferences(
        self, repository: InMemoryUserPreferencesRepository, user_id: UUID
    ) -> None:
        """Test updating preferences."""
        preferences = UserPreferences(user_id=user_id)
        await repository.save(preferences)

        preferences.theme = "dark"
        updated = await repository.update(preferences)

        assert updated.theme == "dark"

    @pytest.mark.asyncio
    async def test_delete_preferences(
        self, repository: InMemoryUserPreferencesRepository, user_id: UUID
    ) -> None:
        """Test deleting preferences."""
        preferences = UserPreferences(user_id=user_id)
        await repository.save(preferences)

        result = await repository.delete(user_id)

        assert result is True

        found = await repository.get_by_user_id(user_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_get_users_with_notification_enabled(
        self, repository: InMemoryUserPreferencesRepository
    ) -> None:
        """Test getting users with notifications enabled."""
        user1_id = uuid4()
        user2_id = uuid4()
        user3_id = uuid4()

        # User 1: email enabled
        prefs1 = UserPreferences(user_id=user1_id, notification_email=True)
        # User 2: email disabled
        prefs2 = UserPreferences(user_id=user2_id, notification_email=False)
        # User 3: email enabled
        prefs3 = UserPreferences(user_id=user3_id, notification_email=True)

        await repository.save(prefs1)
        await repository.save(prefs2)
        await repository.save(prefs3)

        email_users = await repository.get_users_with_notification_enabled("email")

        assert len(email_users) == 2
        assert user1_id in email_users
        assert user3_id in email_users


class TestInMemoryConsentRepository:
    """Tests for InMemoryConsentRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryConsentRepository:
        """Create repository instance."""
        return InMemoryConsentRepository()

    @pytest.fixture
    def user_id(self) -> UUID:
        """Create a sample user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_save_consent(
        self, repository: InMemoryConsentRepository, user_id: UUID
    ) -> None:
        """Test saving consent record."""
        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        saved = await repository.save(consent)

        assert saved.consent_id == consent.consent_id
        assert saved.granted is True

    @pytest.mark.asyncio
    async def test_get_by_user_id(
        self, repository: InMemoryConsentRepository, user_id: UUID
    ) -> None:
        """Test getting consent records by user ID."""
        # Create multiple consent records
        consent1 = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )
        consent2 = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            granted=True,
            version="1.0",
        )

        await repository.save(consent1)
        await repository.save(consent2)

        records = await repository.get_by_user_id(user_id)

        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_get_by_id(
        self, repository: InMemoryConsentRepository, user_id: UUID
    ) -> None:
        """Test getting consent record by ID."""
        consent_id = uuid4()
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            granted=True,
            version="1.0",
        )

        await repository.save(consent)

        found = await repository.get_by_id(consent_id)

        assert found is not None
        assert found.consent_id == consent_id


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""

    def test_creates_repositories(self) -> None:
        """Test factory creates repositories."""
        factory = RepositoryFactory()

        user_repo = factory.get_user_repository()
        prefs_repo = factory.get_preferences_repository()
        consent_repo = factory.get_consent_repository()

        assert user_repo is not None
        assert prefs_repo is not None
        assert consent_repo is not None

    def test_returns_same_instance(self) -> None:
        """Test factory returns singleton instances."""
        factory = RepositoryFactory()

        repo1 = factory.get_user_repository()
        repo2 = factory.get_user_repository()

        assert repo1 is repo2

    def test_reset_clears_instances(self) -> None:
        """Test reset clears repository instances."""
        factory = RepositoryFactory()

        repo1 = factory.get_user_repository()
        factory.reset()
        repo2 = factory.get_user_repository()

        assert repo1 is not repo2
