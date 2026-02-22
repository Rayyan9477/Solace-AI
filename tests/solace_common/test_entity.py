"""
Unit tests for Solace-AI Entity Module.
"""

import pytest
import uuid
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from solace_common.domain.entity import (
    AuditableMixin,
    Entity,
    EntityId,
    EntityMetadata,
    MutableEntity,
    TimestampedMixin,
)


class TestEntityId:
    """Tests for EntityId."""

    def test_generate_unique_id(self) -> None:
        """Test generating unique IDs."""
        id1 = EntityId.generate()
        id2 = EntityId.generate()

        assert id1 != id2
        assert len(id1.value) == 36  # UUID format

    def test_from_string(self) -> None:
        """Test creating ID from string."""
        value = "test-entity-123"
        entity_id = EntityId.from_string(value)

        assert entity_id.value == value
        assert str(entity_id) == value

    def test_from_uuid(self) -> None:
        """Test creating ID from UUID."""
        uid = uuid.uuid4()
        entity_id = EntityId(value=uid)

        assert entity_id.value == str(uid)

    def test_equality(self) -> None:
        """Test EntityId equality."""
        id1 = EntityId.from_string("test-123")
        id2 = EntityId.from_string("test-123")
        id3 = EntityId.from_string("test-456")

        assert id1 == id2
        assert id1 != id3
        assert id1 == "test-123"  # Compare with string

    def test_hashable(self) -> None:
        """Test EntityId can be used in sets and dicts."""
        id1 = EntityId.from_string("test-123")
        id2 = EntityId.from_string("test-123")

        id_set = {id1, id2}
        assert len(id_set) == 1

        id_dict = {id1: "value"}
        assert id_dict[id2] == "value"

    def test_immutability(self) -> None:
        """Test EntityId is immutable."""
        entity_id = EntityId.generate()
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            entity_id.value = "new-value"  # type: ignore[misc]

    def test_validation_empty_string(self) -> None:
        """Test validation rejects empty string."""
        with pytest.raises(ValueError):
            EntityId(value="")

    def test_strip_whitespace(self) -> None:
        """Test whitespace is stripped."""
        entity_id = EntityId(value="  test-123  ")
        assert entity_id.value == "test-123"


class TestEntityMetadata:
    """Tests for EntityMetadata."""

    def test_default_values(self) -> None:
        """Test default metadata values."""
        metadata = EntityMetadata()

        assert metadata.version == 1
        assert metadata.created_at is not None
        assert metadata.updated_at is not None
        assert metadata.created_at <= metadata.updated_at

    def test_increment_version(self) -> None:
        """Test version increment."""
        metadata = EntityMetadata()
        new_metadata = metadata.increment_version(updated_by="user-123")

        assert new_metadata.version == 2
        assert new_metadata.updated_by == "user-123"
        assert new_metadata.created_at == metadata.created_at
        # updated_at should be >= original (may be equal if operation is instant)
        assert new_metadata.updated_at >= metadata.updated_at

    def test_timestamp_validation(self) -> None:
        """Test updated_at cannot be before created_at."""
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(hours=1)

        with pytest.raises(ValueError):
            EntityMetadata(created_at=now, updated_at=earlier)

    def test_immutability(self) -> None:
        """Test metadata is immutable."""
        metadata = EntityMetadata()
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            metadata.version = 5  # type: ignore[misc]


class ConcreteEntityId(EntityId):
    """Concrete EntityId for testing."""
    pass


class ConcreteEntity(Entity[ConcreteEntityId]):
    """Concrete Entity for testing."""

    _entity_type = "ConcreteEntity"
    name: str


class ConcreteMutableEntity(MutableEntity[ConcreteEntityId]):
    """Concrete MutableEntity for testing."""

    _entity_type = "ConcreteMutableEntity"
    name: str
    status: str = "active"


class TestEntity:
    """Tests for Entity base class."""

    def test_entity_creation(self) -> None:
        """Test basic entity creation."""
        entity_id = ConcreteEntityId.generate()
        entity = ConcreteEntity(id=entity_id, name="Test Entity")

        assert entity.id == entity_id
        assert entity.name == "Test Entity"
        assert entity.version == 1

    def test_entity_equality_by_id(self) -> None:
        """Test entities are equal by ID."""
        entity_id = ConcreteEntityId.generate()

        entity1 = ConcreteEntity(id=entity_id, name="Name 1")
        entity2 = ConcreteEntity(id=entity_id, name="Name 2")

        assert entity1 == entity2  # Same ID, different attributes

    def test_entity_inequality(self) -> None:
        """Test entities with different IDs are not equal."""
        entity1 = ConcreteEntity(id=ConcreteEntityId.generate(), name="Same Name")
        entity2 = ConcreteEntity(id=ConcreteEntityId.generate(), name="Same Name")

        assert entity1 != entity2  # Different IDs

    def test_entity_hashable(self) -> None:
        """Test entity can be used in sets."""
        entity_id = ConcreteEntityId.generate()
        entity1 = ConcreteEntity(id=entity_id, name="Name 1")
        entity2 = ConcreteEntity(id=entity_id, name="Name 2")

        entity_set = {entity1, entity2}
        assert len(entity_set) == 1  # Same ID, so same entity

    def test_convenience_properties(self) -> None:
        """Test convenience property accessors."""
        entity = ConcreteEntity(id=ConcreteEntityId.generate(), name="Test")

        assert entity.created_at == entity.metadata.created_at
        assert entity.updated_at == entity.metadata.updated_at
        assert entity.version == entity.metadata.version

    def test_get_entity_type(self) -> None:
        """Test entity type accessor."""
        entity = ConcreteEntity(id=ConcreteEntityId.generate(), name="Test")

        assert entity.get_entity_type() == "ConcreteEntity"

    def test_repr(self) -> None:
        """Test entity string representation."""
        entity_id = ConcreteEntityId.from_string("test-123")
        entity = ConcreteEntity(id=entity_id, name="Test")

        repr_str = repr(entity)
        assert "ConcreteEntity" in repr_str
        assert "test-123" in repr_str


class TestMutableEntity:
    """Tests for MutableEntity."""

    def test_touch_updates_metadata(self) -> None:
        """Test touch() updates version and timestamp."""
        entity = ConcreteMutableEntity(
            id=ConcreteEntityId.generate(),
            name="Test",
        )
        original_version = entity.version
        original_updated = entity.updated_at

        entity.touch(updated_by="user-123")

        assert entity.version == original_version + 1
        # updated_at should be >= original (may be equal if operation is instant)
        assert entity.updated_at >= original_updated

    def test_check_version(self) -> None:
        """Test version checking for optimistic locking."""
        entity = ConcreteMutableEntity(
            id=ConcreteEntityId.generate(),
            name="Test",
        )

        assert entity.check_version(1) is True
        assert entity.check_version(2) is False

        entity.touch()
        assert entity.check_version(2) is True
        assert entity.check_version(1) is False


class TestTimestampedMixin:
    """Tests for TimestampedMixin."""

    def test_default_timestamp(self) -> None:
        """Test default timestamp is set."""

        class TimestampedModel(TimestampedMixin):
            value: str

        model = TimestampedModel(value="test")

        assert model.created_at is not None
        assert model.created_at.tzinfo == timezone.utc

    def test_immutability(self) -> None:
        """Test mixin is immutable."""

        class TimestampedModel(TimestampedMixin):
            value: str

        model = TimestampedModel(value="test")
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            model.created_at = datetime.now(timezone.utc)  # type: ignore[misc]


class TestAuditableMixin:
    """Tests for AuditableMixin."""

    def test_audit_fields(self) -> None:
        """Test audit fields are set."""

        class AuditableModel(AuditableMixin):
            name: str

        model = AuditableModel(name="test", created_by="user-123")

        assert model.created_at is not None
        assert model.created_by == "user-123"
        assert model.updated_at is None
        assert model.updated_by is None

    def test_with_update(self) -> None:
        """Test with_update returns update fields."""

        class AuditableModel(AuditableMixin):
            name: str

        model = AuditableModel(name="test")
        update_fields = model.with_update("user-456")

        assert "updated_at" in update_fields
        assert update_fields["updated_by"] == "user-456"
