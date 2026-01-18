"""
Tests for User Service repository layer.
"""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from uuid import UUID, uuid4

from services.user_service.src.domain.service import User, UserPreferences, ConsentRecord, UserRole, AccountStatus
from services.user_service.src.infrastructure.repository import (
    InMemoryUserRepository,
    InMemoryUserPreferencesRepository,
    UserRepositoryFactory,
    UserRepositoryConfig,
    EntityNotFoundError,
    DuplicateEntityError,
)


@pytest.fixture
def user_repo() -> InMemoryUserRepository:
    """Create a fresh user repository for each test."""
    return InMemoryUserRepository()


@pytest.fixture
def prefs_repo() -> InMemoryUserPreferencesRepository:
    """Create a fresh preferences repository for each test."""
    return InMemoryUserPreferencesRepository()


def create_test_user(email: str = "test@example.com", display_name: str = "Test User") -> User:
    """Create a test user."""
    return User(
        email=email,
        password_hash="test:hash",
        display_name=display_name,
    )


class TestInMemoryUserRepository:
    """Tests for InMemoryUserRepository."""

    @pytest.mark.asyncio
    async def test_save_user(self, user_repo: InMemoryUserRepository) -> None:
        """Test saving a user."""
        user = create_test_user()
        saved = await user_repo.save(user)
        assert saved.user_id == user.user_id
        assert saved.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_save_duplicate_email_raises_error(self, user_repo: InMemoryUserRepository) -> None:
        """Test that saving duplicate email raises error."""
        user1 = create_test_user(email="test@example.com")
        await user_repo.save(user1)
        user2 = create_test_user(email="test@example.com")
        with pytest.raises(DuplicateEntityError) as exc_info:
            await user_repo.save(user2)
        assert "test@example.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_by_id(self, user_repo: InMemoryUserRepository) -> None:
        """Test getting user by ID."""
        user = create_test_user()
        await user_repo.save(user)
        found = await user_repo.get_by_id(user.user_id)
        assert found is not None
        assert found.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, user_repo: InMemoryUserRepository) -> None:
        """Test getting non-existent user returns None."""
        found = await user_repo.get_by_id(uuid4())
        assert found is None

    @pytest.mark.asyncio
    async def test_get_by_id_deleted_user(self, user_repo: InMemoryUserRepository) -> None:
        """Test getting deleted user returns None."""
        user = create_test_user()
        user.deleted_at = datetime.now(timezone.utc)
        await user_repo.save(user)
        found = await user_repo.get_by_id(user.user_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_get_by_email(self, user_repo: InMemoryUserRepository) -> None:
        """Test getting user by email."""
        user = create_test_user(email="test@example.com")
        await user_repo.save(user)
        found = await user_repo.get_by_email("test@example.com")
        assert found is not None
        assert found.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_get_by_email_case_insensitive(self, user_repo: InMemoryUserRepository) -> None:
        """Test email lookup is case insensitive."""
        user = create_test_user(email="Test@Example.com")
        await user_repo.save(user)
        found = await user_repo.get_by_email("test@example.com")
        assert found is not None

    @pytest.mark.asyncio
    async def test_update_user(self, user_repo: InMemoryUserRepository) -> None:
        """Test updating a user."""
        user = create_test_user()
        await user_repo.save(user)
        user.display_name = "Updated Name"
        updated = await user_repo.update(user)
        assert updated.display_name == "Updated Name"

    @pytest.mark.asyncio
    async def test_update_nonexistent_user_raises_error(self, user_repo: InMemoryUserRepository) -> None:
        """Test updating non-existent user raises error."""
        user = create_test_user()
        with pytest.raises(EntityNotFoundError):
            await user_repo.update(user)

    @pytest.mark.asyncio
    async def test_delete_user(self, user_repo: InMemoryUserRepository) -> None:
        """Test deleting a user."""
        user = create_test_user()
        await user_repo.save(user)
        result = await user_repo.delete(user.user_id)
        assert result is True
        found = await user_repo.get_by_id(user.user_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_user(self, user_repo: InMemoryUserRepository) -> None:
        """Test deleting non-existent user returns False."""
        result = await user_repo.delete(uuid4())
        assert result is False

    @pytest.mark.asyncio
    async def test_list_users(self, user_repo: InMemoryUserRepository) -> None:
        """Test listing users."""
        for i in range(5):
            user = create_test_user(email=f"user{i}@example.com")
            await user_repo.save(user)
        users = await user_repo.list_users(limit=3)
        assert len(users) == 3

    @pytest.mark.asyncio
    async def test_list_users_pagination(self, user_repo: InMemoryUserRepository) -> None:
        """Test listing users with pagination."""
        for i in range(5):
            user = create_test_user(email=f"user{i}@example.com")
            await user_repo.save(user)
        users = await user_repo.list_users(limit=2, offset=2)
        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_count_users(self, user_repo: InMemoryUserRepository) -> None:
        """Test counting users."""
        for i in range(3):
            user = create_test_user(email=f"user{i}@example.com")
            await user_repo.save(user)
        count = await user_repo.count_users()
        assert count == 3

    @pytest.mark.asyncio
    async def test_search_users(self, user_repo: InMemoryUserRepository) -> None:
        """Test searching users."""
        user1 = create_test_user(email="john@example.com", display_name="John Doe")
        user2 = create_test_user(email="jane@example.com", display_name="Jane Smith")
        await user_repo.save(user1)
        await user_repo.save(user2)
        results = await user_repo.search_users("john")
        assert len(results) == 1
        assert results[0].email == "john@example.com"

    @pytest.mark.asyncio
    async def test_consent_records(self, user_repo: InMemoryUserRepository) -> None:
        """Test consent record management."""
        user = create_test_user()
        await user_repo.save(user)
        consent = ConsentRecord(
            user_id=user.user_id,
            consent_type="terms",
            granted=True,
            version="1.0",
        )
        saved = await user_repo.save_consent(consent)
        assert saved.consent_id == consent.consent_id
        records = await user_repo.get_consent_records(user.user_id)
        assert len(records) == 1
        assert records[0].consent_type == "terms"


class TestInMemoryUserPreferencesRepository:
    """Tests for InMemoryUserPreferencesRepository."""

    @pytest.mark.asyncio
    async def test_save_preferences(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test saving preferences."""
        user_id = uuid4()
        prefs = UserPreferences(user_id=user_id)
        saved = await prefs_repo.save(prefs)
        assert saved.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_by_user(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test getting preferences by user ID."""
        user_id = uuid4()
        prefs = UserPreferences(user_id=user_id, theme="dark")
        await prefs_repo.save(prefs)
        found = await prefs_repo.get_by_user(user_id)
        assert found is not None
        assert found.theme == "dark"

    @pytest.mark.asyncio
    async def test_get_by_user_not_found(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test getting non-existent preferences returns None."""
        found = await prefs_repo.get_by_user(uuid4())
        assert found is None

    @pytest.mark.asyncio
    async def test_update_preferences(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test updating preferences."""
        user_id = uuid4()
        prefs = UserPreferences(user_id=user_id)
        await prefs_repo.save(prefs)
        prefs.theme = "dark"
        prefs.notification_email = False
        updated = await prefs_repo.update(prefs)
        assert updated.theme == "dark"
        assert updated.notification_email is False

    @pytest.mark.asyncio
    async def test_delete_preferences(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test deleting preferences."""
        user_id = uuid4()
        prefs = UserPreferences(user_id=user_id)
        await prefs_repo.save(prefs)
        result = await prefs_repo.delete(user_id)
        assert result is True
        found = await prefs_repo.get_by_user(user_id)
        assert found is None

    @pytest.mark.asyncio
    async def test_get_users_with_notification_enabled(self, prefs_repo: InMemoryUserPreferencesRepository) -> None:
        """Test getting users with specific notification enabled."""
        user1_id = uuid4()
        user2_id = uuid4()
        prefs1 = UserPreferences(user_id=user1_id, notification_email=True)
        prefs2 = UserPreferences(user_id=user2_id, notification_email=False)
        await prefs_repo.save(prefs1)
        await prefs_repo.save(prefs2)
        users = await prefs_repo.get_users_with_notification_enabled("email")
        assert len(users) == 1
        assert user1_id in users


class TestUserRepositoryFactory:
    """Tests for UserRepositoryFactory."""

    def test_get_user_repository(self) -> None:
        """Test getting user repository."""
        factory = UserRepositoryFactory()
        repo = factory.get_user_repository()
        assert repo is not None
        assert isinstance(repo, InMemoryUserRepository)

    def test_get_preferences_repository(self) -> None:
        """Test getting preferences repository."""
        factory = UserRepositoryFactory()
        repo = factory.get_preferences_repository()
        assert repo is not None
        assert isinstance(repo, InMemoryUserPreferencesRepository)

    def test_get_repository_returns_same_instance(self) -> None:
        """Test that factory returns same repository instance."""
        factory = UserRepositoryFactory()
        repo1 = factory.get_user_repository()
        repo2 = factory.get_user_repository()
        assert repo1 is repo2

    def test_reset_clears_repositories(self) -> None:
        """Test that reset clears repositories."""
        factory = UserRepositoryFactory()
        repo1 = factory.get_user_repository()
        factory.reset()
        repo2 = factory.get_user_repository()
        assert repo1 is not repo2
