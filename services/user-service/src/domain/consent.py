"""
Solace-AI User Service - Consent Management Domain Service.

Manages user consent for HIPAA/GDPR compliance.
Implements audit trail, versioning, and expiry logic for consent records.

Architecture Layer: Domain
Principles: Clean Architecture, HIPAA Compliance, GDPR Compliance
Compliance: Consent must be freely given, specific, informed, unambiguous
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from .entities import User
from .value_objects import ConsentType, ConsentRecord
from ..events import ConsentGrantedEvent, ConsentRevokedEvent

import structlog

logger = structlog.get_logger(__name__)


class ConsentRepository(Protocol):
    """Port for consent persistence (Hexagonal Architecture adapter)."""
    async def save(self, consent: ConsentRecord) -> ConsentRecord: ...
    async def get_by_id(self, consent_id: UUID) -> ConsentRecord | None: ...
    async def get_by_user(self, user_id: UUID) -> list[ConsentRecord]: ...
    async def get_by_user_and_type(self, user_id: UUID, consent_type: ConsentType) -> list[ConsentRecord]: ...
    async def get_active_consent(self, user_id: UUID, consent_type: ConsentType) -> ConsentRecord | None: ...


class EventPublisher(Protocol):
    """Port for event publishing (Hexagonal Architecture adapter)."""
    async def publish(self, event: ConsentGrantedEvent | ConsentRevokedEvent) -> None: ...


@dataclass
class ConsentGrantResult:
    """Result of consent granting operation."""
    consent: ConsentRecord | None = None
    error: str | None = None


@dataclass
class ConsentRevokeResult:
    """Result of consent revocation operation."""
    success: bool = False
    error: str | None = None


@dataclass
class ConsentVerificationResult:
    """Result of consent verification."""
    is_valid: bool = False
    consent: ConsentRecord | None = None
    reason: str | None = None


class ConsentService:
    """
    Domain service for consent management (HIPAA/GDPR compliance).

    Business Rules: Consent must be explicit, documented, and revocable.
    HIPAA consents require audit metadata. Records are immutable.
    """

    def __init__(
        self,
        repository: ConsentRepository | None = None,
        event_publisher: EventPublisher | None = None,
    ) -> None:
        self._repository = repository
        self._event_publisher = event_publisher

    async def grant_consent(
        self,
        user_id: UUID,
        consent_type: ConsentType,
        version: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        expires_in_days: int | None = None,
    ) -> ConsentGrantResult:
        """Grant user consent. HIPAA consents require audit metadata."""
        if not self._repository:
            return ConsentGrantResult(error="Repository not configured")

        if consent_type.is_hipaa_protected():
            if not ip_address:
                logger.warning(
                    "hipaa_consent_without_ip",
                    user_id=str(user_id),
                    consent_type=consent_type.value
                )

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        consent = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=consent_type,
            granted=True,
            version=version,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
        )

        try:
            saved_consent = await self._repository.save(consent)

            if self._event_publisher:
                event = ConsentGrantedEvent(
                    aggregate_id=user_id,
                    consent_type=consent_type.value,
                    version=version,
                    ip_address=ip_address,
                )
                await self._event_publisher.publish(event)

            logger.info(
                "consent_granted",
                user_id=str(user_id),
                consent_type=consent_type.value,
                version=version,
                expires_at=expires_at.isoformat() if expires_at else None,
            )

            return ConsentGrantResult(consent=saved_consent)

        except Exception as e:
            logger.error("consent_grant_failed", user_id=str(user_id), error=str(e))
            return ConsentGrantResult(error=f"Failed to grant consent: {str(e)}")

    async def revoke_consent(
        self,
        user_id: UUID,
        consent_type: ConsentType,
        reason: str | None = None,
    ) -> ConsentRevokeResult:
        """Revoke user consent. Creates revocation record and publishes event."""
        if not self._repository:
            return ConsentRevokeResult(error="Repository not configured")

        if consent_type.is_required():
            logger.warning(
                "attempted_revoke_required_consent",
                user_id=str(user_id),
                consent_type=consent_type.value,
            )
            return ConsentRevokeResult(error=f"Cannot revoke required consent: {consent_type.value}")

        active_consent = await self._repository.get_active_consent(user_id, consent_type)
        if not active_consent:
            return ConsentRevokeResult(error="No active consent to revoke")

        revocation = ConsentRecord(
            consent_id=uuid4(),
            user_id=user_id,
            consent_type=consent_type,
            granted=False,
            version=active_consent.version,
        )

        try:
            await self._repository.save(revocation)

            if self._event_publisher:
                event = ConsentRevokedEvent(
                    aggregate_id=user_id,
                    consent_type=consent_type.value,
                    reason=reason,
                )
                await self._event_publisher.publish(event)

            logger.info(
                "consent_revoked",
                user_id=str(user_id),
                consent_type=consent_type.value,
                reason=reason,
            )

            return ConsentRevokeResult(success=True)

        except Exception as e:
            logger.error("consent_revoke_failed", user_id=str(user_id), error=str(e))
            return ConsentRevokeResult(error=f"Failed to revoke consent: {str(e)}")

    async def verify_consent(
        self,
        user_id: UUID,
        consent_type: ConsentType,
    ) -> ConsentVerificationResult:
        """Verify if user has valid active consent (granted and not expired)."""
        if not self._repository:
            return ConsentVerificationResult(
                is_valid=False,
                reason="Repository not configured"
            )

        try:
            # Get all consent records for this type and find the most recent one
            all_consents = await self._repository.get_by_user_and_type(user_id, consent_type)

            if not all_consents:
                return ConsentVerificationResult(
                    is_valid=False,
                    reason=f"No consent records for {consent_type.value}"
                )

            # Get the most recent consent record (latest granted_at)
            latest_consent = max(all_consents, key=lambda c: c.granted_at)

            if not latest_consent.granted:
                return ConsentVerificationResult(
                    is_valid=False,
                    consent=latest_consent,
                    reason="Consent has been revoked"
                )

            if latest_consent.is_expired():
                return ConsentVerificationResult(
                    is_valid=False,
                    consent=latest_consent,
                    reason="Consent has expired"
                )

            return ConsentVerificationResult(
                is_valid=True,
                consent=latest_consent
            )

        except Exception as e:
            logger.error("consent_verification_failed", user_id=str(user_id), error=str(e))
            return ConsentVerificationResult(
                is_valid=False,
                reason=f"Verification failed: {str(e)}"
            )

    async def get_consent_records(self, user_id: UUID) -> list[ConsentRecord]:
        """Get all consent records for user (audit trail, sorted by granted_at desc)."""
        if not self._repository:
            return []

        records = await self._repository.get_by_user(user_id)
        return sorted(records, key=lambda r: r.granted_at, reverse=True)

    async def get_active_consents(self, user_id: UUID) -> dict[ConsentType, ConsentRecord]:
        """Get all active consents for user (dict mapping type to record)."""
        if not self._repository:
            return {}

        all_records = await self._repository.get_by_user(user_id)

        # Group by consent type and get the latest record
        # If timestamps are equal, prefer revocation (granted=False)
        latest_records: dict[ConsentType, ConsentRecord] = {}
        for record in all_records:
            if record.consent_type not in latest_records:
                latest_records[record.consent_type] = record
            else:
                current_latest = latest_records[record.consent_type]
                # Replace if this record is newer, or if same time but this is a revocation
                if (record.granted_at > current_latest.granted_at or
                    (record.granted_at == current_latest.granted_at and not record.granted)):
                    latest_records[record.consent_type] = record

        # Filter to only include active (granted and not expired) consents
        active_consents: dict[ConsentType, ConsentRecord] = {}
        for consent_type, record in latest_records.items():
            if record.is_active():
                active_consents[consent_type] = record

        return active_consents

    async def check_required_consents(self, user: User) -> tuple[bool, list[ConsentType]]:
        """Check if user has all required consents. Returns (has_all, missing_types)."""
        required_consents = [
            ConsentType.TERMS_OF_SERVICE,
            ConsentType.PRIVACY_POLICY,
            ConsentType.DATA_PROCESSING,
        ]

        missing_consents: list[ConsentType] = []

        for consent_type in required_consents:
            result = await self.verify_consent(user.user_id, consent_type)
            if not result.is_valid:
                missing_consents.append(consent_type)

        has_all = len(missing_consents) == 0

        if not has_all:
            logger.warning(
                "missing_required_consents",
                user_id=str(user.user_id),
                missing=[ ct.value for ct in missing_consents]
            )

        return has_all, missing_consents

    async def expire_old_consents(self, user_id: UUID) -> int:
        """Utility to count expired consents (called by background jobs). Returns count."""
        if not self._repository:
            return 0

        all_records = await self._repository.get_by_user(user_id)
        expired_count = 0

        for record in all_records:
            if record.granted and record.is_expired():
                expired_count += 1

        if expired_count > 0:
            logger.info("consents_expired", user_id=str(user_id), count=expired_count)

        return expired_count
