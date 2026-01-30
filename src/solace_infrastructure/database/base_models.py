"""Solace-AI Base SQLAlchemy Models - Foundation for all database entities.

Provides enterprise-grade base models with:
- UUID primary keys
- Automatic timestamp tracking
- Optimistic locking via version numbers
- Soft delete support
- Audit trail capabilities
- JSON field support for flexible data
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, TypeVar

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    event,
    inspect,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound="Base")


class ModelState(str, Enum):
    """Entity lifecycle states for soft delete support."""
    ACTIVE = "active"
    DELETED = "deleted"
    ARCHIVED = "archived"


class Base(AsyncAttrs, DeclarativeBase):
    """SQLAlchemy declarative base with async support."""

    type_annotation_map: ClassVar[dict[type, Any]] = {
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
    }


class TimestampMixin:
    """Mixin providing automatic timestamp tracking.

    Automatically sets created_at on insert and updated_at on update.
    Uses UTC timezone for consistency across distributed systems.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class SoftDeleteMixin:
    """Mixin providing soft delete capability.

    Instead of physical deletion, marks records with deleted_at timestamp.
    Enables data recovery and maintains referential integrity.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        index=True,
    )
    state: Mapped[str] = mapped_column(
        String(20),
        default=ModelState.ACTIVE.value,
        nullable=False,
        index=True,
    )

    @property
    def is_deleted(self) -> bool:
        """Check if record is soft-deleted."""
        return self.deleted_at is not None or self.state == ModelState.DELETED.value

    def soft_delete(self) -> None:
        """Mark record as deleted without physical removal."""
        self.deleted_at = datetime.now(timezone.utc)
        self.state = ModelState.DELETED.value

    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.deleted_at = None
        self.state = ModelState.ACTIVE.value


class VersionMixin:
    """Mixin providing optimistic locking via version numbers.

    Prevents lost updates in concurrent modification scenarios.
    Increment version on each update for conflict detection.
    """

    version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
    )

    def increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self.version += 1


class AuditMixin:
    """Mixin providing audit trail fields.

    Tracks who created and last modified each record.
    Essential for compliance and debugging.
    """

    created_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )


class TenantMixin:
    """Mixin for multi-tenancy support.

    Enables data isolation between tenants in shared database.
    All queries should filter by tenant_id for security.
    """

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )


class BaseModel(Base, TimestampMixin, VersionMixin):
    """Abstract base model with UUID primary key and core features.

    All domain entities should inherit from this class.
    Provides: UUID PK, timestamps, version tracking.
    """

    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    @declared_attr.directive
    @classmethod
    def __tablename__(cls) -> str:
        """Generate table name from class name (snake_case)."""
        name = cls.__name__
        return "".join(
            f"_{c.lower()}" if c.isupper() else c for c in name
        ).lstrip("_")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        result: dict[str, Any] = {}
        for column in inspect(self).mapper.column_attrs:
            value = getattr(self, column.key)
            if isinstance(value, uuid.UUID):
                result[column.key] = str(value)
            elif isinstance(value, datetime):
                result[column.key] = value.isoformat()
            elif isinstance(value, Enum):
                result[column.key] = value.value
            else:
                result[column.key] = value
        return result

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.id})>"


class AuditableModel(BaseModel, AuditMixin, SoftDeleteMixin):
    """Base model with full audit capabilities and soft delete.

    Use for entities requiring compliance tracking and data retention.
    Provides: UUID PK, timestamps, version, audit trail, soft delete.
    """

    __abstract__ = True


class TenantModel(AuditableModel, TenantMixin):
    """Base model for multi-tenant entities.

    Use for entities that must be isolated per tenant.
    Provides all auditable features plus tenant isolation.
    """

    __abstract__ = True


# Domain-specific base models for Solace-AI

class UserProfileBase(AuditableModel):
    """Base for user profile related entities."""

    __abstract__ = True

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


class SessionBase(AuditableModel):
    """Base for therapy session related entities."""

    __abstract__ = True

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


class ClinicalBase(AuditableModel):
    """Base for clinical data entities with PHI handling.

    Entities inheriting this require encryption and access controls.
    """

    __abstract__ = True

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_phi: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    encryption_key_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )


class SafetyEventBase(AuditableModel):
    """Base for safety-critical event entities.

    These records are immutable and never deleted for compliance.
    """

    __abstract__ = True

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    severity_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )


class ConfigurationBase(BaseModel):
    """Base for system configuration entities."""

    __abstract__ = True

    key: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )
    value: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )


class AuditLog(Base, TimestampMixin):
    """Immutable audit log for HIPAA compliance.

    Records all security-relevant events: authentication, authorization,
    data access, and modifications. Never deleted — retained per policy.
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    actor_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )
    actor_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="user",
    )
    resource_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    action: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    outcome: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="success",
    )
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
    )


# SQLAlchemy event listeners for automatic behavior

@event.listens_for(BaseModel, "before_update", propagate=True)
def receive_before_update(mapper: Any, connection: Any, target: BaseModel) -> None:
    """Automatically increment version on update for optimistic locking."""
    if hasattr(target, "version"):
        target.increment_version()
    logger.debug(
        "model_before_update",
        model=target.__class__.__name__,
        id=str(target.id),
        version=getattr(target, "version", None),
    )


@event.listens_for(AuditableModel, "before_insert", propagate=True)
def receive_before_insert_audit(
    mapper: Any, connection: Any, target: AuditableModel
) -> None:
    """Set initial audit fields on insert."""
    if target.state is None:
        target.state = ModelState.ACTIVE.value
    logger.debug(
        "model_before_insert",
        model=target.__class__.__name__,
        id=str(target.id),
    )


def get_model_table_name(model_class: type[BaseModel]) -> str:
    """Get the table name for a model class."""
    if hasattr(model_class, "__tablename__"):
        tablename = model_class.__tablename__
        if callable(tablename):
            return tablename()
        return tablename
    return model_class.__name__.lower()


def create_all_tables(engine: Any) -> None:
    """Create all tables defined in the Base metadata."""
    Base.metadata.create_all(bind=engine)
    logger.info("database_tables_created")


async def create_all_tables_async(engine: Any) -> None:
    """Create all tables asynchronously."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_tables_created_async")
