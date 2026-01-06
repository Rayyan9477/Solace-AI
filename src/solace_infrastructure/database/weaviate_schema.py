"""Solace-AI Weaviate Schema Manager - Vector database collections setup."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

from solace_infrastructure.weaviate import (
    WeaviateClient, CollectionConfig, PropertyConfig, PropertyDataType, VectorDistanceMetric,
)
from solace_common.exceptions import InfrastructureError

logger = structlog.get_logger(__name__)


class CollectionType(str, Enum):
    """Types of Weaviate collections in Solace-AI."""
    CONVERSATION = "conversation"
    SESSION = "session"
    THERAPEUTIC = "therapeutic"
    SAFETY = "safety"
    USER_FACT = "user_fact"


class SchemaVersion(str, Enum):
    """Schema versions for migration tracking."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


CURRENT_SCHEMA_VERSION = SchemaVersion.V2_0


class WeaviateSchemaSettings(BaseSettings):
    """Weaviate schema configuration from environment."""
    replication_factor: int = Field(default=1, ge=1, le=5)
    multi_tenancy_enabled: bool = Field(default=True)
    auto_create_collections: bool = Field(default=True)
    delete_existing_on_init: bool = Field(default=False)
    embedding_dimensions: int = Field(default=1536)
    model_config = SettingsConfigDict(env_prefix="WEAVIATE_SCHEMA_", env_file=".env", extra="ignore")


@dataclass
class CollectionDefinition:
    """Complete definition for a Weaviate collection."""
    name: str
    collection_type: CollectionType
    description: str
    properties: list[PropertyConfig]
    distance_metric: VectorDistanceMetric = VectorDistanceMetric.COSINE
    multi_tenancy: bool = True
    replication_factor: int = 1

    def to_config(self) -> CollectionConfig:
        return CollectionConfig(
            name=self.name, description=self.description, properties=self.properties,
            distance_metric=self.distance_metric, multi_tenancy_enabled=self.multi_tenancy,
            replication_factor=self.replication_factor,
        )


def _prop(name: str, dtype: PropertyDataType, desc: str, skip_vec: bool = False,
          filterable: bool = True, searchable: bool = True) -> PropertyConfig:
    return PropertyConfig(name=name, data_type=dtype, description=desc,
                         skip_vectorization=skip_vec, index_filterable=filterable, index_searchable=searchable)


class SolaceCollections:
    """Factory for Solace-AI Weaviate collection definitions."""

    @staticmethod
    def conversation_memory() -> CollectionDefinition:
        """Collection for storing conversation messages and embeddings."""
        return CollectionDefinition(
            name="ConversationMemory",
            collection_type=CollectionType.CONVERSATION,
            description="Stores individual conversation messages with semantic embeddings",
            properties=[
                _prop("content", PropertyDataType.TEXT, "Message content"),
                _prop("user_id", PropertyDataType.TEXT, "User identifier", skip_vec=True),
                _prop("session_id", PropertyDataType.TEXT, "Session identifier", skip_vec=True),
                _prop("role", PropertyDataType.TEXT, "Message role", skip_vec=True),
                _prop("timestamp", PropertyDataType.DATE, "Message timestamp", skip_vec=True),
                _prop("emotion", PropertyDataType.TEXT, "Detected emotion", skip_vec=True),
                _prop("importance", PropertyDataType.NUMBER, "Importance score", skip_vec=True),
                _prop("metadata", PropertyDataType.OBJECT, "Additional metadata", skip_vec=True, searchable=False),
            ],
        )

    @staticmethod
    def session_summary() -> CollectionDefinition:
        """Collection for storing session summaries."""
        return CollectionDefinition(
            name="SessionSummary",
            collection_type=CollectionType.SESSION,
            description="Stores summarized session information for episodic memory",
            properties=[
                _prop("summary", PropertyDataType.TEXT, "Session summary text"),
                _prop("user_id", PropertyDataType.TEXT, "User identifier", skip_vec=True),
                _prop("session_id", PropertyDataType.TEXT, "Session identifier", skip_vec=True),
                _prop("session_number", PropertyDataType.INT, "Sequential session number", skip_vec=True),
                _prop("key_topics", PropertyDataType.TEXT_ARRAY, "Key topics discussed"),
                _prop("techniques_used", PropertyDataType.TEXT_ARRAY, "Therapeutic techniques", skip_vec=True),
                _prop("emotional_arc", PropertyDataType.NUMBER_ARRAY, "Emotional trajectory", skip_vec=True, searchable=False),
                _prop("session_date", PropertyDataType.DATE, "Date of session", skip_vec=True),
                _prop("duration_minutes", PropertyDataType.INT, "Session duration", skip_vec=True, searchable=False),
            ],
        )

    @staticmethod
    def therapeutic_insight() -> CollectionDefinition:
        """Collection for storing therapeutic insights and learnings."""
        return CollectionDefinition(
            name="TherapeuticInsight",
            collection_type=CollectionType.THERAPEUTIC,
            description="Stores therapeutic insights extracted from sessions",
            properties=[
                _prop("insight", PropertyDataType.TEXT, "Insight content"),
                _prop("user_id", PropertyDataType.TEXT, "User identifier", skip_vec=True),
                _prop("insight_type", PropertyDataType.TEXT, "Type of insight", skip_vec=True),
                _prop("confidence", PropertyDataType.NUMBER, "Confidence score", skip_vec=True),
                _prop("source_sessions", PropertyDataType.TEXT_ARRAY, "Source session IDs", skip_vec=True, searchable=False),
                _prop("created_at", PropertyDataType.DATE, "When insight was created", skip_vec=True),
            ],
        )

    @staticmethod
    def user_fact() -> CollectionDefinition:
        """Collection for storing user facts and knowledge graph entries."""
        return CollectionDefinition(
            name="UserFact",
            collection_type=CollectionType.USER_FACT,
            description="Stores extracted facts about users for semantic memory",
            properties=[
                _prop("fact", PropertyDataType.TEXT, "Fact statement"),
                _prop("user_id", PropertyDataType.TEXT, "User identifier", skip_vec=True),
                _prop("fact_type", PropertyDataType.TEXT, "Category of fact", skip_vec=True),
                _prop("subject", PropertyDataType.TEXT, "Subject of fact triple", skip_vec=True),
                _prop("predicate", PropertyDataType.TEXT, "Predicate of fact triple", skip_vec=True, searchable=False),
                _prop("object_value", PropertyDataType.TEXT, "Object of fact triple", skip_vec=True, searchable=False),
                _prop("confidence", PropertyDataType.NUMBER, "Confidence score", skip_vec=True, searchable=False),
                _prop("source_session_id", PropertyDataType.TEXT, "Source session", skip_vec=True, searchable=False),
                _prop("last_verified", PropertyDataType.DATE, "Last verification", skip_vec=True, searchable=False),
            ],
        )

    @staticmethod
    def crisis_event() -> CollectionDefinition:
        """Collection for storing crisis events (never deleted)."""
        return CollectionDefinition(
            name="CrisisEvent",
            collection_type=CollectionType.SAFETY,
            description="Stores crisis events for safety monitoring - NEVER DELETED",
            properties=[
                _prop("description", PropertyDataType.TEXT, "Crisis event description"),
                _prop("user_id", PropertyDataType.TEXT, "User identifier", skip_vec=True),
                _prop("severity_level", PropertyDataType.INT, "Severity level (1-5)", skip_vec=True),
                _prop("event_type", PropertyDataType.TEXT, "Type of crisis event", skip_vec=True),
                _prop("resolution_status", PropertyDataType.TEXT, "Current resolution status", skip_vec=True),
                _prop("occurred_at", PropertyDataType.DATE, "When event occurred", skip_vec=True),
                _prop("session_id", PropertyDataType.TEXT, "Associated session", skip_vec=True, searchable=False),
            ],
            multi_tenancy=False,
        )

    @classmethod
    def all_collections(cls) -> list[CollectionDefinition]:
        return [cls.conversation_memory(), cls.session_summary(), cls.therapeutic_insight(),
                cls.user_fact(), cls.crisis_event()]


class WeaviateSchemaManager:
    """Manages Weaviate schema lifecycle for Solace-AI."""

    def __init__(self, client: WeaviateClient, settings: WeaviateSchemaSettings | None = None) -> None:
        self._client = client
        self._settings = settings or WeaviateSchemaSettings()

    async def initialize_schema(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        if self._settings.delete_existing_on_init:
            await self.drop_all_collections()
        for definition in SolaceCollections.all_collections():
            config = definition.to_config()
            config.replication_factor = self._settings.replication_factor
            config.multi_tenancy_enabled = self._settings.multi_tenancy_enabled and definition.multi_tenancy
            try:
                created = await self._client.create_collection(config)
                results[definition.name] = created
                logger.info("weaviate_collection_setup", name=definition.name, created=created)
            except Exception as e:
                logger.error("weaviate_collection_failed", name=definition.name, error=str(e))
                results[definition.name] = False
        return results

    async def drop_all_collections(self) -> list[str]:
        dropped: list[str] = []
        for definition in SolaceCollections.all_collections():
            try:
                if await self._client.delete_collection(definition.name):
                    dropped.append(definition.name)
            except Exception as e:
                logger.warning("weaviate_collection_drop_failed", name=definition.name, error=str(e))
        return dropped

    async def verify_schema(self) -> dict[str, bool]:
        return {d.name: await self._client.collection_exists(d.name) for d in SolaceCollections.all_collections()}

    async def get_collection_stats(self) -> dict[str, int]:
        stats: dict[str, int] = {}
        for definition in SolaceCollections.all_collections():
            try:
                stats[definition.name] = await self._client.count(definition.name)
            except Exception:
                stats[definition.name] = -1
        return stats


async def setup_weaviate_schema(client: WeaviateClient, settings: WeaviateSchemaSettings | None = None) -> dict[str, bool]:
    manager = WeaviateSchemaManager(client, settings)
    return await manager.initialize_schema()
