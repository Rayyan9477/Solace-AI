"""
Unit tests for Consent Management Domain Service.

Tests cover consent granting, revocation, verification, and HIPAA compliance.
"""
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from typing import List

from src.domain.consent import (
    ConsentService,
    ConsentRepository,
    EventPublisher,
    ConsentGrantResult,
    ConsentRevokeResult,
    ConsentVerificationResult,
)
from src.domain.value_objects import ConsentType, ConsentRecord
from src.domain.entities import User
from src.events import ConsentGrantedEvent, ConsentRevokedEvent


class MockConsentRepository:
    """Mock consent repository for testing."""

    def __init__(self):
        self.consents: List[ConsentRecord] = []

    async def save(self, consent: ConsentRecord) -> ConsentRecord:
        """Save consent record."""
        self.consents.append(consent)
        return consent

    async def get_by_id(self, consent_id) -> ConsentRecord | None:
        """Get consent by ID."""
        for consent in self.consents:
            if consent.consent_id == consent_id:
                return consent
        return None

    async def get_by_user(self, user_id) -> List[ConsentRecord]:
        """Get all consents for user."""
        return [c for c in self.consents if c.user_id == user_id]

    async def get_by_user_and_type(self, user_id, consent_type) -> List[ConsentRecord]:
        """Get consents by user and type."""
        return [
            c for c in self.consents
            if c.user_id == user_id and c.consent_type == consent_type
        ]

    async def get_active_consent(self, user_id, consent_type) -> ConsentRecord | None:
        """Get active consent for user and type."""
        consents = await self.get_by_user_and_type(user_id, consent_type)
        for consent in reversed(consents):
            if consent.is_active():
                return consent
        return None


class MockEventPublisher:
    """Mock event publisher for testing."""

    def __init__(self):
        self.published_events = []

    async def publish(self, event) -> None:
        """Publish event."""
        self.published_events.append(event)


@pytest.fixture
def consent_repository():
    """Create mock consent repository."""
    return MockConsentRepository()


@pytest.fixture
def event_publisher():
    """Create mock event publisher."""
    return MockEventPublisher()


@pytest.fixture
def consent_service(consent_repository, event_publisher):
    """Create consent service with mocks."""
    return ConsentService(
        repository=consent_repository,
        event_publisher=event_publisher,
    )


class TestConsentService:
    """Test cases for ConsentService."""

    @pytest.mark.asyncio
    async def test_grant_consent_success(self, consent_service):
        """Test successfully granting consent."""
        user_id = uuid4()

        result = await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert result.consent is not None
        assert result.error is None
        assert result.consent.user_id == user_id
        assert result.consent.consent_type == ConsentType.TERMS_OF_SERVICE
        assert result.consent.granted is True

    @pytest.mark.asyncio
    async def test_grant_consent_publishes_event(self, consent_service, event_publisher):
        """Test that granting consent publishes domain event."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            version="1.0",
        )

        assert len(event_publisher.published_events) == 1
        event = event_publisher.published_events[0]
        assert isinstance(event, ConsentGrantedEvent)
        assert event.aggregate_id == user_id

    @pytest.mark.asyncio
    async def test_grant_consent_with_expiry(self, consent_service):
        """Test granting consent with expiry."""
        user_id = uuid4()

        result = await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
            expires_in_days=30,
        )

        assert result.consent is not None
        assert result.consent.expires_at is not None

    @pytest.mark.asyncio
    async def test_revoke_consent_success(self, consent_service):
        """Test successfully revoking consent."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
        )

        result = await consent_service.revoke_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            reason="User opted out",
        )

        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_revoke_consent_publishes_event(self, consent_service, event_publisher):
        """Test that revoking consent publishes domain event."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
        )

        event_publisher.published_events.clear()

        await consent_service.revoke_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        assert len(event_publisher.published_events) == 1
        event = event_publisher.published_events[0]
        assert isinstance(event, ConsentRevokedEvent)

    @pytest.mark.asyncio
    async def test_revoke_consent_without_active_consent_fails(self, consent_service):
        """Test that revoking non-existent consent fails."""
        user_id = uuid4()

        result = await consent_service.revoke_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        assert result.success is False
        assert result.error is not None
        assert "No active consent" in result.error

    @pytest.mark.asyncio
    async def test_verify_consent_valid(self, consent_service):
        """Test verifying valid consent."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
        )

        result = await consent_service.verify_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
        )

        assert result.is_valid is True
        assert result.consent is not None
        assert result.reason is None

    @pytest.mark.asyncio
    async def test_verify_consent_no_active_consent(self, consent_service):
        """Test verifying consent when none exists."""
        user_id = uuid4()

        result = await consent_service.verify_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
        )

        assert result.is_valid is False
        assert result.reason is not None
        assert "No consent" in result.reason

    @pytest.mark.asyncio
    async def test_verify_consent_revoked(self, consent_service):
        """Test verifying revoked consent."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
        )

        # Small delay to ensure different timestamps
        import asyncio
        await asyncio.sleep(0.001)

        await consent_service.revoke_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        result = await consent_service.verify_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        assert result.is_valid is False
        assert "revoked" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_verify_consent_expired(self, consent_service):
        """Test verifying expired consent."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
            expires_in_days=-1,
        )

        result = await consent_service.verify_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        assert result.is_valid is False
        assert "expired" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_get_consent_records(self, consent_service):
        """Test getting all consent records for user."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            version="1.0",
        )

        records = await consent_service.get_consent_records(user_id)

        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_get_active_consents(self, consent_service):
        """Test getting all active consents for user."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            version="1.0",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
        )

        await consent_service.revoke_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
        )

        active_consents = await consent_service.get_active_consents(user_id)

        assert len(active_consents) == 2
        assert ConsentType.TERMS_OF_SERVICE in active_consents
        assert ConsentType.PRIVACY_POLICY in active_consents
        assert ConsentType.MARKETING_COMMUNICATIONS not in active_consents

    @pytest.mark.asyncio
    async def test_check_required_consents_all_present(self, consent_service):
        """Test checking required consents when all are present."""
        user_id = uuid4()
        user = User(
            user_id=user_id,
            email="test@example.com",
            password_hash="hashed",
            display_name="Test User",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.PRIVACY_POLICY,
            version="1.0",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.DATA_PROCESSING,
            version="1.0",
        )

        has_all, missing = await consent_service.check_required_consents(user)

        assert has_all is True
        assert len(missing) == 0

    @pytest.mark.asyncio
    async def test_check_required_consents_some_missing(self, consent_service):
        """Test checking required consents when some are missing."""
        user_id = uuid4()
        user = User(
            user_id=user_id,
            email="test@example.com",
            password_hash="hashed",
            display_name="Test User",
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.TERMS_OF_SERVICE,
            version="1.0",
        )

        has_all, missing = await consent_service.check_required_consents(user)

        assert has_all is False
        assert len(missing) == 2
        assert ConsentType.PRIVACY_POLICY in missing
        assert ConsentType.DATA_PROCESSING in missing

    @pytest.mark.asyncio
    async def test_expire_old_consents(self, consent_service):
        """Test expiring old consents."""
        user_id = uuid4()

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.MARKETING_COMMUNICATIONS,
            version="1.0",
            expires_in_days=-1,
        )

        await consent_service.grant_consent(
            user_id=user_id,
            consent_type=ConsentType.ANALYTICS_TRACKING,
            version="1.0",
            expires_in_days=-5,
        )

        expired_count = await consent_service.expire_old_consents(user_id)

        assert expired_count == 2
