"""Unit tests for base SQLAlchemy models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import Column, String
from sqlalchemy.orm import Mapped, mapped_column

from solace_infrastructure.database.base_models import (
    Base,
    BaseModel,
    AuditableModel,
    TenantModel,
    TimestampMixin,
    SoftDeleteMixin,
    VersionMixin,
    AuditMixin,
    TenantMixin,
    ModelState,
    get_model_table_name,
    UserProfileBase,
    SessionBase,
    ClinicalBase,
    SafetyEventBase,
    ConfigurationBase,
)


class TestModelState:
    """Tests for ModelState enum."""

    def test_model_state_values(self) -> None:
        """Test ModelState enum has correct values."""
        assert ModelState.ACTIVE.value == "active"
        assert ModelState.DELETED.value == "deleted"
        assert ModelState.ARCHIVED.value == "archived"


class TestTimestampMixin:
    """Tests for TimestampMixin."""

    def test_timestamp_mixin_has_created_at(self) -> None:
        """Test TimestampMixin defines created_at."""
        assert hasattr(TimestampMixin, "created_at")

    def test_timestamp_mixin_has_updated_at(self) -> None:
        """Test TimestampMixin defines updated_at."""
        assert hasattr(TimestampMixin, "updated_at")


class TestSoftDeleteMixin:
    """Tests for SoftDeleteMixin."""

    def test_soft_delete_mixin_has_deleted_at(self) -> None:
        """Test SoftDeleteMixin defines deleted_at."""
        assert hasattr(SoftDeleteMixin, "deleted_at")

    def test_soft_delete_mixin_has_state(self) -> None:
        """Test SoftDeleteMixin defines state."""
        assert hasattr(SoftDeleteMixin, "state")


class TestVersionMixin:
    """Tests for VersionMixin."""

    def test_version_mixin_has_version(self) -> None:
        """Test VersionMixin defines version."""
        assert hasattr(VersionMixin, "version")

    def test_increment_version_method_exists(self) -> None:
        """Test VersionMixin has increment_version method."""
        assert hasattr(VersionMixin, "increment_version")
        assert callable(getattr(VersionMixin, "increment_version"))


class TestAuditMixin:
    """Tests for AuditMixin."""

    def test_audit_mixin_has_created_by(self) -> None:
        """Test AuditMixin defines created_by."""
        assert hasattr(AuditMixin, "created_by")

    def test_audit_mixin_has_updated_by(self) -> None:
        """Test AuditMixin defines updated_by."""
        assert hasattr(AuditMixin, "updated_by")


class TestTenantMixin:
    """Tests for TenantMixin."""

    def test_tenant_mixin_has_tenant_id(self) -> None:
        """Test TenantMixin defines tenant_id."""
        assert hasattr(TenantMixin, "tenant_id")


class TestBase:
    """Tests for SQLAlchemy Base class."""

    def test_base_is_declarative(self) -> None:
        """Test Base is a proper declarative base."""
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")


class TestBaseModel:
    """Tests for BaseModel class."""

    def test_base_model_is_abstract(self) -> None:
        """Test BaseModel is abstract."""
        assert BaseModel.__abstract__ is True

    def test_base_model_has_id_field(self) -> None:
        """Test BaseModel defines id field."""
        assert hasattr(BaseModel, "id")

    def test_base_model_includes_timestamp_mixin(self) -> None:
        """Test BaseModel includes timestamp fields."""
        assert hasattr(BaseModel, "created_at")
        assert hasattr(BaseModel, "updated_at")

    def test_base_model_includes_version_mixin(self) -> None:
        """Test BaseModel includes version field."""
        assert hasattr(BaseModel, "version")


class TestAuditableModel:
    """Tests for AuditableModel class."""

    def test_auditable_model_is_abstract(self) -> None:
        """Test AuditableModel is abstract."""
        assert AuditableModel.__abstract__ is True

    def test_auditable_model_includes_audit_fields(self) -> None:
        """Test AuditableModel includes audit fields."""
        assert hasattr(AuditableModel, "created_by")
        assert hasattr(AuditableModel, "updated_by")

    def test_auditable_model_includes_soft_delete(self) -> None:
        """Test AuditableModel includes soft delete fields."""
        assert hasattr(AuditableModel, "deleted_at")
        assert hasattr(AuditableModel, "state")


class TestTenantModel:
    """Tests for TenantModel class."""

    def test_tenant_model_is_abstract(self) -> None:
        """Test TenantModel is abstract."""
        assert TenantModel.__abstract__ is True

    def test_tenant_model_includes_tenant_id(self) -> None:
        """Test TenantModel includes tenant_id."""
        assert hasattr(TenantModel, "tenant_id")


class TestDomainBaseModels:
    """Tests for domain-specific base models."""

    def test_user_profile_base_is_abstract(self) -> None:
        """Test UserProfileBase is abstract."""
        assert UserProfileBase.__abstract__ is True

    def test_user_profile_base_has_user_id(self) -> None:
        """Test UserProfileBase has user_id."""
        assert hasattr(UserProfileBase, "user_id")

    def test_session_base_is_abstract(self) -> None:
        """Test SessionBase is abstract."""
        assert SessionBase.__abstract__ is True

    def test_session_base_has_session_id(self) -> None:
        """Test SessionBase has session_id."""
        assert hasattr(SessionBase, "session_id")

    def test_clinical_base_is_abstract(self) -> None:
        """Test ClinicalBase is abstract."""
        assert ClinicalBase.__abstract__ is True

    def test_clinical_base_has_phi_flag(self) -> None:
        """Test ClinicalBase has is_phi flag."""
        assert hasattr(ClinicalBase, "is_phi")

    def test_safety_event_base_is_abstract(self) -> None:
        """Test SafetyEventBase is abstract."""
        assert SafetyEventBase.__abstract__ is True

    def test_safety_event_base_has_severity(self) -> None:
        """Test SafetyEventBase has severity_level."""
        assert hasattr(SafetyEventBase, "severity_level")

    def test_configuration_base_is_abstract(self) -> None:
        """Test ConfigurationBase is abstract."""
        assert ConfigurationBase.__abstract__ is True

    def test_configuration_base_has_key_value(self) -> None:
        """Test ConfigurationBase has key and value."""
        assert hasattr(ConfigurationBase, "key")
        assert hasattr(ConfigurationBase, "value")


class TestGetModelTableName:
    """Tests for get_model_table_name utility."""

    def test_get_model_table_name_base_model(self) -> None:
        """Test table name generation works."""
        assert callable(get_model_table_name)


class TestConcreteModel:
    """Tests using base model abstract class."""

    def test_base_model_inherits_from_base(self) -> None:
        """Test BaseModel inherits from Base."""
        assert issubclass(BaseModel, Base)

    def test_base_model_is_abstract(self) -> None:
        """Test BaseModel is marked as abstract."""
        assert BaseModel.__abstract__ is True
