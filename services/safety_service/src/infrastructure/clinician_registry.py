"""
Solace-AI Safety Service - Clinician Registry.
HTTP lookup to User Service for on-call clinician contact information with caching.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID

import httpx
import structlog
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = structlog.get_logger(__name__)


class ClinicianRegistrySettings(BaseSettings):
    """Configuration for the clinician registry."""

    user_service_url: str = Field(
        default="http://localhost:8006",
        description="URL of the User Service for clinician lookup",
    )
    request_timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0)
    cache_ttl_minutes: int = Field(default=5, ge=1, le=60)
    fallback_oncall_email: str = Field(
        ...,
        description="Fallback email for on-call team when user service is unavailable",
    )
    max_retries: int = Field(default=2, ge=0, le=5)

    model_config = SettingsConfigDict(
        env_prefix="CLINICIAN_REGISTRY_",
        env_file=".env",
        extra="ignore",
    )


@dataclass
class ClinicianContact:
    """Contact information for a clinician."""

    clinician_id: UUID
    email: str
    name: str
    phone: str | None = None
    is_on_call: bool = False


@dataclass
class _CacheEntry:
    """Cached clinician lookup result."""

    contacts: list[ClinicianContact]
    fetched_at: datetime
    ttl_minutes: int

    @property
    def is_expired(self) -> bool:
        elapsed = datetime.now(timezone.utc) - self.fetched_at
        return elapsed > timedelta(minutes=self.ttl_minutes)


class ClinicianRegistry:
    """
    Registry for clinician contact information.

    Fetches clinician data from the User Service with caching.
    Falls back to configurable fallback email when the service is unavailable.
    """

    def __init__(self, settings: ClinicianRegistrySettings) -> None:
        self._settings = settings
        self._oncall_cache: _CacheEntry | None = None
        self._clinician_cache: dict[UUID, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get_oncall_clinicians(self) -> list[ClinicianContact]:
        """Get currently on-call clinicians.

        Returns cached result if available and not expired.
        Falls back to fallback_oncall_email if user service is unavailable.
        """
        async with self._lock:
            if self._oncall_cache and not self._oncall_cache.is_expired:
                return self._oncall_cache.contacts

        try:
            contacts = await self._fetch_oncall_clinicians()
            async with self._lock:
                self._oncall_cache = _CacheEntry(
                    contacts=contacts,
                    fetched_at=datetime.now(timezone.utc),
                    ttl_minutes=self._settings.cache_ttl_minutes,
                )
            return contacts
        except Exception as e:
            logger.warning(
                "clinician_registry_fetch_failed",
                error=str(e),
                fallback_email=self._settings.fallback_oncall_email,
            )
            return [
                ClinicianContact(
                    clinician_id=UUID("00000000-0000-0000-0000-000000000000"),
                    email=self._settings.fallback_oncall_email,
                    name="On-Call Team (fallback)",
                    is_on_call=True,
                )
            ]

    async def get_clinician_contact(self, clinician_id: UUID) -> ClinicianContact | None:
        """Get contact information for a specific clinician.

        Returns cached result if available and not expired.
        """
        async with self._lock:
            cached = self._clinician_cache.get(clinician_id)
            if cached and not cached.is_expired and cached.contacts:
                return cached.contacts[0]

        try:
            contact = await self._fetch_clinician(clinician_id)
            if contact:
                async with self._lock:
                    self._clinician_cache[clinician_id] = _CacheEntry(
                        contacts=[contact],
                        fetched_at=datetime.now(timezone.utc),
                        ttl_minutes=self._settings.cache_ttl_minutes,
                    )
            return contact
        except Exception as e:
            logger.warning(
                "clinician_contact_fetch_failed",
                clinician_id=str(clinician_id),
                error=str(e),
            )
            return None

    async def _fetch_oncall_clinicians(self) -> list[ClinicianContact]:
        """Fetch on-call clinicians from User Service."""
        url = f"{self._settings.user_service_url}/api/v1/users/on-call-clinicians"
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(self._settings.max_retries + 1):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                    clinicians = data.get("clinicians", [])
                    contacts = [
                        ClinicianContact(
                            clinician_id=UUID(c["user_id"]),
                            email=c.get("email", ""),
                            name=c.get("display_name", "Clinician"),
                            phone=c.get("phone_number"),
                            is_on_call=True,
                        )
                        for c in clinicians
                        if c.get("user_id") and c.get("email")
                    ]
                    logger.info("oncall_clinicians_fetched", count=len(contacts))
                    return contacts
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    logger.warning(
                        "oncall_fetch_attempt_failed",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    if attempt == self._settings.max_retries:
                        raise
        return []

    async def _fetch_clinician(self, clinician_id: UUID) -> ClinicianContact | None:
        """Fetch a specific clinician's contact info from User Service."""
        url = f"{self._settings.user_service_url}/api/v1/users/{clinician_id}"
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                if not data.get("email"):
                    return None
                return ClinicianContact(
                    clinician_id=clinician_id,
                    email=data["email"],
                    name=data.get("display_name", "Clinician"),
                    phone=data.get("phone_number"),
                )
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.warning(
                    "clinician_fetch_failed",
                    clinician_id=str(clinician_id),
                    error=str(e),
                )
                return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._oncall_cache = None
        self._clinician_cache.clear()
