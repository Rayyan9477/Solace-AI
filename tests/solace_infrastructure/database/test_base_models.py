"""Unit tests for base SQLAlchemy models."""
from __future__ import annotations

from solace_infrastructure.database.base_models import (
    AuditableModel,
    AuditMixin,
    Base,
    BaseModel,
    ClinicalBase,
    ConfigurationBase,
    ModelState,
    SafetyEventBase,
    SessionBase,
    SoftDeleteMixin,
    TenantMixin,
    TenantModel,
    TimestampMixin,
    UserProfileBase,
    VersionMixin,
    get_model_table_name,
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
        assert callable(VersionMixin.increment_version)


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


# ---------------------------------------------------------------------------
# PHI Encryption Tests — NEW-03/NEW-04: JSONB list/dict PHI support
# ---------------------------------------------------------------------------

class _FakeEntity:
    """Minimal stand-in for a ClinicalBase-like entity.

    We reuse only the encrypt_phi_fields / decrypt_phi_fields methods by
    binding them unbound — this avoids the SQLAlchemy declarative ceremony
    of spinning up a full mapped class. ClinicalBase's PHI helpers only
    touch attributes named in __phi_fields__, so any plain object works.
    """

    __phi_fields__: list[str]

    def __init__(self, phi_fields: list[str], **attrs: object) -> None:
        self.__phi_fields__ = phi_fields
        for k, v in attrs.items():
            setattr(self, k, v)


def _make_field_encryptor():
    """Build a FieldEncryptor suitable for unit tests (dev key + test salt)."""
    from pydantic import SecretStr

    from solace_security.encryption import (
        EncryptionSettings,
        Encryptor,
        FieldEncryptor,
    )

    settings = EncryptionSettings.for_development()
    settings = settings.model_copy(
        update={"search_hash_salt": SecretStr("phi-jsonb-roundtrip-salt-32-bytes!!")}
    )
    return FieldEncryptor(Encryptor(settings), settings=settings)


class TestPhiEncryptionJSONB:
    """NEW-03/NEW-04: encrypt_phi_fields must also protect JSONB list/dict fields.

    DiagnosisSession.messages and Hypothesis.supporting_evidence are JSONB
    columns that hold clinical conversation content and rationale — PHI by
    any reasonable reading of HIPAA. The legacy implementation only handled
    str values and silently ignored list/dict, leaving conversation history
    in plaintext in the database.

    After the fix, list and dict values are JSON-serialized, encrypted, and
    stored as the "v1$"-prefixed ciphertext string. On load, the string is
    detected, decrypted, and JSON-deserialized back to the original type.
    """

    # ---- str path (regression guard) ----

    def test_str_value_roundtrips(self) -> None:
        fe = _make_field_encryptor()
        entity = _FakeEntity(["summary"], summary="depressed for 3 weeks")
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert isinstance(entity.summary, str)
        assert entity.summary.startswith("v1$")
        ClinicalBase.decrypt_phi_fields(entity, fe)
        assert entity.summary == "depressed for 3 weeks"

    # ---- list path (NEW-03: DiagnosisSession.messages) ----

    def test_list_of_dicts_roundtrips(self) -> None:
        fe = _make_field_encryptor()
        original = [
            {"role": "user", "content": "I can't sleep"},
            {"role": "assistant", "content": "tell me more"},
        ]
        entity = _FakeEntity(["messages"], messages=original)
        ClinicalBase.encrypt_phi_fields(entity, fe)
        # After encryption, stored value is a ciphertext string, not a list
        assert isinstance(entity.messages, str)
        assert entity.messages.startswith("v1$")
        ClinicalBase.decrypt_phi_fields(entity, fe)
        assert entity.messages == original

    def test_empty_list_roundtrips(self) -> None:
        fe = _make_field_encryptor()
        entity = _FakeEntity(["messages"], messages=[])
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert isinstance(entity.messages, str)
        assert entity.messages.startswith("v1$")
        ClinicalBase.decrypt_phi_fields(entity, fe)
        assert entity.messages == []

    # ---- dict path (NEW-04: Hypothesis evidence as dict) ----

    def test_dict_roundtrips(self) -> None:
        fe = _make_field_encryptor()
        original = {"dsm5_code": "F41.1", "criteria_met": 5, "notes": "GAD likely"}
        entity = _FakeEntity(["evidence"], evidence=original)
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert isinstance(entity.evidence, str)
        assert entity.evidence.startswith("v1$")
        ClinicalBase.decrypt_phi_fields(entity, fe)
        assert entity.evidence == original

    # ---- idempotency guards ----

    def test_already_encrypted_str_is_not_double_encrypted(self) -> None:
        fe = _make_field_encryptor()
        entity = _FakeEntity(["summary"], summary="hello")
        ClinicalBase.encrypt_phi_fields(entity, fe)
        before = entity.summary
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert entity.summary == before

    def test_none_value_is_skipped(self) -> None:
        fe = _make_field_encryptor()
        entity = _FakeEntity(["summary", "messages"], summary=None, messages=None)
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert entity.summary is None
        assert entity.messages is None

    def test_non_phi_field_untouched(self) -> None:
        """Fields not declared in __phi_fields__ must remain unchanged."""
        fe = _make_field_encryptor()
        entity = _FakeEntity(
            ["messages"], messages=[{"x": 1}], extra_notes="not PHI"
        )
        ClinicalBase.encrypt_phi_fields(entity, fe)
        assert entity.extra_notes == "not PHI"
        ClinicalBase.decrypt_phi_fields(entity, fe)
        assert entity.extra_notes == "not PHI"


# ---------------------------------------------------------------------------
# NEW-03 / NEW-04 wire-up tests — declared __phi_fields__ on diagnosis entities
# ---------------------------------------------------------------------------

class TestDiagnosisEntityPhiFields:
    """NEW-03, NEW-04: verify diagnosis_entities declare the right PHI fields."""

    def test_diagnosis_session_messages_is_phi(self) -> None:
        """NEW-03: DiagnosisSession.messages holds conversation history (PHI)."""
        from solace_infrastructure.database.entities.diagnosis_entities import (
            DiagnosisSession,
        )

        assert "messages" in DiagnosisSession.__phi_fields__, (
            "DiagnosisSession.messages stores plaintext conversation history "
            "and must be listed in __phi_fields__ for auto-encryption at rest."
        )

    def test_hypothesis_declares_phi_fields(self) -> None:
        """NEW-04: Hypothesis supporting evidence fields must be encrypted."""
        from solace_infrastructure.database.entities.diagnosis_entities import (
            Hypothesis,
        )

        required_phi = {
            "supporting_evidence",
            "contra_evidence",
            "challenge_results",
            "criteria_met",
            "criteria_missing",
        }
        declared = set(Hypothesis.__phi_fields__)
        missing = required_phi - declared
        assert not missing, (
            f"Hypothesis is missing PHI field declarations for: {sorted(missing)}. "
            f"All JSONB clinical-evidence fields must be encrypted at rest."
        )
