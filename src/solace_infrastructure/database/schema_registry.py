"""
Central registry for all database schemas in the Solace-AI platform.

This module provides a centralized SchemaRegistry that all services import from,
eliminating schema fragmentation across 13+ files. All domain entities should
register with this registry to ensure consistency and proper inheritance from
base models.

Usage:
    from solace_infrastructure.database.schema_registry import SchemaRegistry
    from solace_infrastructure.database.base_models import ClinicalBase

    @SchemaRegistry.register
    class SafetyAssessment(ClinicalBase):
        __tablename__ = "safety_assessments"
        # ... field definitions
"""

from __future__ import annotations

import warnings
from typing import Dict, Type, TypeVar

from sqlalchemy.orm import DeclarativeBase

from .base_models import BaseModel

T = TypeVar("T", bound=BaseModel)


class SchemaRegistry:
    """Central registry for all database schemas across services.

    This registry enforces centralized schema management and prevents schema
    fragmentation. All entity classes should be registered using the @register
    decorator to ensure:
    1. Proper inheritance from BaseModel hierarchy
    2. Centralized schema discovery
    3. Consistent naming conventions
    4. Automatic encryption and audit trail support
    """

    _entities: Dict[str, Type[BaseModel]] = {}
    _registered_tables: set[str] = set()

    @classmethod
    def register(cls, entity_class: Type[T]) -> Type[T]:
        """Register an entity class in the schema registry.

        Args:
            entity_class: The SQLAlchemy ORM entity class to register.
                Must have __tablename__ attribute defined.

        Returns:
            The same entity class (allows use as decorator)

        Raises:
            ValueError: If entity_class doesn't have __tablename__ or
                if table is already registered with a different class

        Example:
            @SchemaRegistry.register
            class User(BaseModel):
                __tablename__ = "users"
                # ... field definitions
        """
        # Validation
        if not hasattr(entity_class, "__tablename__"):
            raise ValueError(
                f"{entity_class.__name__} must define __tablename__ attribute"
            )

        table_name = entity_class.__tablename__
        class_name = entity_class.__name__

        # Check for duplicate registration
        if table_name in cls._registered_tables:
            existing_class = cls._entities.get(table_name)
            if existing_class and existing_class != entity_class:
                raise ValueError(
                    f"Table '{table_name}' is already registered with "
                    f"{existing_class.__name__}. Cannot register {class_name}."
                )

        # Register entity
        cls._entities[table_name] = entity_class
        cls._registered_tables.add(table_name)

        return entity_class

    @classmethod
    def get(cls, table_name: str) -> Type[BaseModel]:
        """Get registered entity by table name.

        Args:
            table_name: The name of the table to retrieve

        Returns:
            The registered entity class

        Raises:
            KeyError: If table_name is not registered
        """
        if table_name not in cls._entities:
            raise KeyError(
                f"Table '{table_name}' not found in schema registry. "
                f"Available tables: {', '.join(sorted(cls._registered_tables))}"
            )
        return cls._entities[table_name]

    @classmethod
    def get_by_class_name(cls, class_name: str) -> Type[BaseModel] | None:
        """Get registered entity by class name.

        Args:
            class_name: The name of the entity class

        Returns:
            The registered entity class if found, None otherwise
        """
        for entity in cls._entities.values():
            if entity.__name__ == class_name:
                return entity
        return None

    @classmethod
    def all_entities(cls) -> Dict[str, Type[BaseModel]]:
        """Get all registered entities.

        Returns:
            Dictionary mapping table names to entity classes
        """
        return cls._entities.copy()

    @classmethod
    def all_table_names(cls) -> set[str]:
        """Get all registered table names.

        Returns:
            Set of all registered table names
        """
        return cls._registered_tables.copy()

    @classmethod
    def is_registered(cls, table_name: str) -> bool:
        """Check if a table is registered.

        Args:
            table_name: The name of the table to check

        Returns:
            True if table is registered, False otherwise
        """
        return table_name in cls._registered_tables

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered entities.

        WARNING: This method is intended for testing only.
        Do not use in production code.
        """
        warnings.warn(
            "SchemaRegistry.clear_registry() called. "
            "This should only be used in tests.",
            UserWarning,
            stacklevel=2,
        )
        cls._entities.clear()
        cls._registered_tables.clear()

    @classmethod
    def get_statistics(cls) -> dict[str, int]:
        """Get registry statistics.

        Returns:
            Dictionary with counts of registered entities
        """
        return {
            "total_entities": len(cls._entities),
            "unique_tables": len(cls._registered_tables),
        }


# Export for convenience
__all__ = ["SchemaRegistry"]
