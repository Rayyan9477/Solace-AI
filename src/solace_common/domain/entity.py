"""
Solace-AI Base Entity Implementation.

Provides identity-based domain entities with:
- Unique identity (UUID-based)
- Timestamps (created_at, updated_at)
- Optimistic locking via version numbers
- Equality based on identity
- Immutable identity after creation
"""

from __future__ import annotations

import uuid
from abc import ABC
from datetime import datetime, timezone
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import structlog

logger = structlog.get_logger(__name__)


class EntityId(BaseModel):
    """
    Strongly-typed entity identifier.

    Provides type-safe identity management with UUID validation.
    Immutable after creation.
    """

    value: str = Field(..., min_length=1, max_length=64)

    model_config = ConfigDict(frozen=True)

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: Any) -> str:
        """Validate and normalize entity ID."""
        if isinstance(v, uuid.UUID):
            return str(v)
        if isinstance(v, EntityId):
            return v.value
        if not isinstance(v, str):
            raise ValueError(f"Entity ID must be string or UUID, got {type(v).__name__}")
        return v.strip()

    @classmethod
    def generate(cls) -> EntityId:
        """Generate a new unique entity ID using UUID4."""
        return cls(value=str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> EntityId:
        """Create entity ID from string value."""
        return cls(value=value)

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EntityId):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False


# Type variable for entity ID
TId = TypeVar("TId", bound=EntityId)


class EntityMetadata(BaseModel):
    """
    Metadata for entity lifecycle tracking.

    Tracks creation time, last update time, and version for optimistic locking.
    """

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when entity was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last update",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Version number for optimistic locking",
    )
    created_by: str | None = Field(
        default=None,
        description="ID of user who created this entity",
    )
    updated_by: str | None = Field(
        default=None,
        description="ID of user who last updated this entity",
    )

    model_config = ConfigDict(frozen=True)

    def increment_version(self, updated_by: str | None = None) -> EntityMetadata:
        """Create new metadata with incremented version."""
        return EntityMetadata(
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
            version=self.version + 1,
            created_by=self.created_by,
            updated_by=updated_by or self.updated_by,
        )

    @model_validator(mode="after")
    def validate_timestamps(self) -> EntityMetadata:
        """Ensure updated_at is not before created_at."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        return self


class Entity(BaseModel, ABC, Generic[TId]):
    """
    Base class for all domain entities.

    Entities are domain objects with a distinct identity that persists
    across state changes. Two entities are considered equal if they
    have the same identity, regardless of their attribute values.

    Features:
    - Unique identity (immutable after creation)
    - Automatic timestamp tracking
    - Version-based optimistic locking
    - Equality based solely on identity
    - Type-safe generic ID support
    """

    id: TId = Field(..., description="Unique entity identifier")
    metadata: EntityMetadata = Field(
        default_factory=EntityMetadata,
        description="Entity lifecycle metadata",
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True,
    )

    # Class-level type name for logging and error messages
    _entity_type: ClassVar[str] = "Entity"

    @classmethod
    def get_entity_type(cls) -> str:
        """Return the entity type name."""
        return getattr(cls, "_entity_type", cls.__name__)

    @property
    def created_at(self) -> datetime:
        """Convenience accessor for creation timestamp."""
        return self.metadata.created_at

    @property
    def updated_at(self) -> datetime:
        """Convenience accessor for update timestamp."""
        return self.metadata.updated_at

    @property
    def version(self) -> int:
        """Convenience accessor for version number."""
        return self.metadata.version

    def __eq__(self, other: object) -> bool:
        """
        Entities are equal if they have the same ID and type.

        This follows DDD principles where entity identity determines equality,
        not attribute values.
        """
        if not isinstance(other, Entity):
            return False
        if not isinstance(other, type(self)):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on identity for use in sets and dicts."""
        return hash((type(self).__name__, str(self.id)))

    def __repr__(self) -> str:
        return f"{self.get_entity_type()}(id={self.id}, version={self.version})"


class MutableEntity(Entity[TId], ABC):
    """
    Base class for entities that support modification.

    Provides methods for updating entity state while maintaining
    version tracking for optimistic locking.
    """

    def _prepare_update(self, updated_by: str | None = None) -> EntityMetadata:
        """
        Prepare updated metadata for entity modification.

        Returns new metadata with incremented version and updated timestamp.
        """
        return self.metadata.increment_version(updated_by=updated_by)

    def touch(self, updated_by: str | None = None) -> None:
        """
        Update the entity's metadata without changing other attributes.

        Increments version and updates timestamp.
        """
        object.__setattr__(
            self, "metadata", self.metadata.increment_version(updated_by=updated_by)
        )

    def check_version(self, expected_version: int) -> bool:
        """
        Check if entity version matches expected version.

        Used for optimistic locking validation.
        """
        return self.version == expected_version


class TimestampedMixin(BaseModel):
    """
    Mixin providing timestamp fields for non-entity models.

    Use this for value objects or other models that need
    timestamp tracking but are not entities.
    """

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )

    model_config = ConfigDict(frozen=True)


class AuditableMixin(BaseModel):
    """
    Mixin providing full audit trail fields.

    Tracks who created/updated and when.
    """

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    created_by: str | None = Field(default=None, description="Creator user ID")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")
    updated_by: str | None = Field(default=None, description="Last updater user ID")

    model_config = ConfigDict(frozen=True)

    def with_update(self, updated_by: str) -> dict[str, Any]:
        """Return dict of audit fields for an update operation."""
        return {
            "updated_at": datetime.now(timezone.utc),
            "updated_by": updated_by,
        }
