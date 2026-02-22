"""
Solace-AI Memory Service - Weaviate Vector Repository.
Vector database repository for semantic memory storage and retrieval.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class WeaviateSettings(BaseSettings):
    """Weaviate connection configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=8080)
    grpc_port: int = Field(default=50051)
    use_https: bool = Field(default=False)
    api_key: SecretStr = Field(default=SecretStr(""))
    timeout_seconds: int = Field(default=30)
    batch_size: int = Field(default=100)
    embedding_dimension: int = Field(default=1536)
    model_config = SettingsConfigDict(env_prefix="WEAVIATE_", env_file=".env", extra="ignore")

    @property
    def http_url(self) -> str:
        return f"{'https' if self.use_https else 'http'}://{self.host}:{self.port}"


class CollectionName(str, Enum):
    """Weaviate collection names."""
    CONVERSATION_MEMORY = "ConversationMemory"
    SESSION_SUMMARY = "SessionSummary"
    THERAPEUTIC_INSIGHT = "TherapeuticInsight"
    USER_FACT = "UserFact"
    CRISIS_EVENT = "CrisisEvent"


@dataclass
class VectorRecord:
    """Record for vector storage."""
    record_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    session_id: UUID | None = None
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    collection: str = CollectionName.CONVERSATION_MEMORY.value
    importance: float = 0.5
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search."""
    record_id: UUID
    content: str
    score: float
    distance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class WeaviateRepository:
    """Weaviate vector repository for semantic memory storage."""

    def __init__(self, settings: WeaviateSettings | None = None) -> None:
        self._settings = settings or WeaviateSettings()
        self._client: Any = None
        self._collections: dict[str, Any] = {}
        self._stats = {"inserts": 0, "searches": 0, "deletes": 0, "batches": 0}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Weaviate client and create collections."""
        try:
            import weaviate
            import weaviate.classes as wvc
            connect_kwargs: dict[str, Any] = {
                "host": self._settings.host,
                "port": self._settings.port,
                "grpc_port": self._settings.grpc_port,
            }
            api_key_value = self._settings.api_key.get_secret_value()
            if api_key_value:
                connect_kwargs["auth_credentials"] = wvc.init.Auth.api_key(api_key_value)
            self._client = await asyncio.to_thread(weaviate.connect_to_local, **connect_kwargs)
            await self._ensure_collections()
            self._initialized = True
            logger.info("weaviate_initialized", host=self._settings.host,
                        authenticated=bool(api_key_value))
        except ImportError:
            import os
            logger.error("weaviate_client_not_installed",
                         hint="pip install weaviate-client")
            if os.environ.get("ENVIRONMENT", "").lower() == "production":
                raise RuntimeError("weaviate-client package required in production")
        except Exception as e:
            logger.error("weaviate_init_failed", error=str(e))

    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist."""
        if not self._client:
            return
        try:
            from weaviate.classes.config import Configure, Property, DataType, VectorDistances
            schemas = {
                CollectionName.CONVERSATION_MEMORY.value: [
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="role", data_type=DataType.TEXT),
                    Property(name="emotion", data_type=DataType.TEXT),
                    Property(name="importance", data_type=DataType.NUMBER),
                    Property(name="timestamp", data_type=DataType.DATE),
                ],
                CollectionName.SESSION_SUMMARY.value: [
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="session_id", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="session_number", data_type=DataType.INT),
                    Property(name="key_topics", data_type=DataType.TEXT_ARRAY),
                    Property(name="session_date", data_type=DataType.DATE),
                ],
                CollectionName.USER_FACT.value: [
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="category", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="confidence", data_type=DataType.NUMBER),
                    Property(name="importance", data_type=DataType.NUMBER),
                ],
            }
            for name, properties in schemas.items():
                if not await asyncio.to_thread(self._client.collections.exists, name):
                    await asyncio.to_thread(
                        self._client.collections.create, name=name, properties=properties,
                        vector_config=Configure.Vectors.none(name="default"),
                        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE))
                self._collections[name] = await asyncio.to_thread(self._client.collections.get, name)
        except Exception as e:
            logger.error("collection_creation_failed", error=str(e))

    async def close(self) -> None:
        """Close Weaviate client connection."""
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None
            self._initialized = False

    async def store_vector(self, record: VectorRecord) -> UUID:
        """Store a vector record."""
        if not self._initialized:
            return record.record_id
        self._stats["inserts"] += 1
        try:
            coll = self._collections.get(record.collection) or await asyncio.to_thread(self._client.collections.get, record.collection)
            self._collections[record.collection] = coll
            props = {"user_id": str(record.user_id), "content": record.content,
                     "importance": record.importance, "timestamp": record.timestamp.isoformat()}
            if record.session_id:
                props["session_id"] = str(record.session_id)
            await asyncio.to_thread(coll.data.insert, properties=props, uuid=record.record_id, vector=record.embedding or None)
        except Exception as e:
            logger.error("vector_store_failed", error=str(e))
        return record.record_id

    async def store_batch(self, records: list[VectorRecord]) -> int:
        """Store multiple vector records in batch."""
        if not self._initialized or not records:
            return 0
        self._stats["batches"] += 1
        stored = 0
        for i in range(0, len(records), self._settings.batch_size):
            batch = records[i:i + self._settings.batch_size]
            coll_name = batch[0].collection
            coll = self._collections.get(coll_name) or await asyncio.to_thread(self._client.collections.get, coll_name)
            self._collections[coll_name] = coll
            try:
                def _do_batch(coll, batch):
                    count = 0
                    with coll.batch.dynamic() as batch_obj:
                        for rec in batch:
                            props = {"user_id": str(rec.user_id), "content": rec.content,
                                     "importance": rec.importance, "timestamp": rec.timestamp.isoformat()}
                            if rec.session_id:
                                props["session_id"] = str(rec.session_id)
                            batch_obj.add_object(properties=props, uuid=rec.record_id, vector=rec.embedding or None)
                            count += 1
                    return count
                stored += await asyncio.to_thread(_do_batch, coll, batch)
            except Exception as e:
                logger.error("batch_store_failed", error=str(e))
        return stored

    async def search_similar(self, query_vector: list[float], user_id: UUID,
                              collection: str = CollectionName.CONVERSATION_MEMORY.value,
                              limit: int = 20, min_certainty: float = 0.7) -> list[SearchResult]:
        """Search for similar vectors."""
        if not self._initialized:
            return []
        self._stats["searches"] += 1
        results: list[SearchResult] = []
        try:
            from weaviate.classes.query import Filter, MetadataQuery
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            self._collections[collection] = coll
            response = await asyncio.to_thread(
                coll.query.near_vector,
                near_vector=query_vector, limit=limit, certainty=min_certainty,
                filters=Filter.by_property("user_id").equal(str(user_id)),
                return_metadata=MetadataQuery(certainty=True, distance=True),
            )
            for obj in response.objects:
                results.append(SearchResult(record_id=obj.uuid, content=obj.properties.get("content", ""),
                    score=obj.metadata.certainty or 0.0, distance=obj.metadata.distance or 0.0, metadata=dict(obj.properties)))
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
        return results

    async def search_by_keyword(self, keyword: str, user_id: UUID,
                                 collection: str = CollectionName.CONVERSATION_MEMORY.value,
                                 limit: int = 20) -> list[SearchResult]:
        """Search using BM25 keyword search."""
        if not self._initialized:
            return []
        self._stats["searches"] += 1
        results: list[SearchResult] = []
        try:
            from weaviate.classes.query import Filter, MetadataQuery
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            self._collections[collection] = coll
            response = await asyncio.to_thread(
                coll.query.bm25, query=keyword, limit=limit,
                filters=Filter.by_property("user_id").equal(str(user_id)), return_metadata=MetadataQuery(score=True))
            for obj in response.objects:
                results.append(SearchResult(record_id=obj.uuid, content=obj.properties.get("content", ""),
                    score=obj.metadata.score or 0.0, distance=0.0, metadata=dict(obj.properties)))
        except Exception as e:
            logger.error("keyword_search_failed", error=str(e))
        return results

    async def hybrid_search(self, query: str, query_vector: list[float], user_id: UUID,
                             collection: str = CollectionName.CONVERSATION_MEMORY.value,
                             limit: int = 20, alpha: float = 0.5) -> list[SearchResult]:
        """Perform hybrid search (BM25 + vector)."""
        if not self._initialized:
            return []
        self._stats["searches"] += 1
        results: list[SearchResult] = []
        try:
            from weaviate.classes.query import Filter, MetadataQuery, HybridFusion
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            self._collections[collection] = coll
            response = await asyncio.to_thread(
                coll.query.hybrid, query=query, vector=query_vector, alpha=alpha, limit=limit,
                filters=Filter.by_property("user_id").equal(str(user_id)),
                fusion_type=HybridFusion.RELATIVE_SCORE, return_metadata=MetadataQuery(score=True))
            for obj in response.objects:
                results.append(SearchResult(record_id=obj.uuid, content=obj.properties.get("content", ""),
                    score=obj.metadata.score or 0.0, distance=0.0, metadata=dict(obj.properties)))
        except Exception as e:
            logger.error("hybrid_search_failed", error=str(e))
        return results

    async def get_by_session(self, session_id: UUID, user_id: UUID,
                              collection: str = CollectionName.CONVERSATION_MEMORY.value,
                              limit: int = 100) -> list[SearchResult]:
        """Get all records for a session."""
        if not self._initialized:
            return []
        self._stats["searches"] += 1
        results: list[SearchResult] = []
        try:
            from weaviate.classes.query import Filter
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            response = await asyncio.to_thread(
                coll.query.fetch_objects, limit=limit,
                filters=Filter.by_property("session_id").equal(str(session_id)) & Filter.by_property("user_id").equal(str(user_id)))
            for obj in response.objects:
                results.append(SearchResult(record_id=obj.uuid, content=obj.properties.get("content", ""),
                    score=1.0, distance=0.0, metadata=dict(obj.properties)))
        except Exception as e:
            logger.error("session_fetch_failed", error=str(e))
        return results

    async def delete_record(self, record_id: UUID, collection: str) -> bool:
        """Delete a vector record."""
        if not self._initialized:
            return False
        self._stats["deletes"] += 1
        try:
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            await asyncio.to_thread(coll.data.delete_by_id, record_id)
            return True
        except Exception as e:
            logger.error("vector_delete_failed", error=str(e))
            return False

    async def delete_user_data(self, user_id: UUID) -> int:
        """Delete all vectors for a user (GDPR compliance)."""
        if not self._initialized:
            return 0
        deleted = 0
        try:
            from weaviate.classes.query import Filter
            for coll_name in CollectionName:
                coll = await asyncio.to_thread(self._client.collections.get, coll_name.value)
                await asyncio.to_thread(coll.data.delete_many, where=Filter.by_property("user_id").equal(str(user_id)))
                deleted += 1
        except Exception as e:
            logger.error("user_delete_failed", error=str(e))
        return deleted

    async def get_collection_count(self, collection: str, user_id: UUID | None = None) -> int:
        """Get count of records in a collection."""
        if not self._initialized:
            return 0
        try:
            coll = self._collections.get(collection) or await asyncio.to_thread(self._client.collections.get, collection)
            if user_id:
                from weaviate.classes.query import Filter
                agg = await asyncio.to_thread(coll.aggregate.over_all, filters=Filter.by_property("user_id").equal(str(user_id)), total_count=True)
            else:
                agg = await asyncio.to_thread(coll.aggregate.over_all, total_count=True)
            return agg.total_count or 0
        except Exception as e:
            logger.error("count_failed", error=str(e))
            return 0

    def is_initialized(self) -> bool:
        return self._initialized

    def get_statistics(self) -> dict[str, Any]:
        return {**self._stats, "initialized": self._initialized, "host": self._settings.host,
                "collections": list(self._collections.keys())}
