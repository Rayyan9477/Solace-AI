"""Solace-AI Weaviate Client - Vector database operations and schema management."""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.data import DataObject
import structlog
from solace_common.exceptions import InfrastructureError

logger = structlog.get_logger(__name__)


class VectorDistanceMetric(str, Enum):
    """Vector similarity distance metrics."""

    COSINE = "cosine"
    DOT = "dot"
    L2_SQUARED = "l2-squared"
    HAMMING = "hamming"
    MANHATTAN = "manhattan"


class PropertyDataType(str, Enum):
    """Weaviate property data types."""

    TEXT = "text"
    TEXT_ARRAY = "text[]"
    INT = "int"
    INT_ARRAY = "int[]"
    NUMBER = "number"
    NUMBER_ARRAY = "number[]"
    BOOLEAN = "boolean"
    BOOLEAN_ARRAY = "boolean[]"
    DATE = "date"
    DATE_ARRAY = "date[]"
    UUID = "uuid"
    UUID_ARRAY = "uuid[]"
    BLOB = "blob"
    OBJECT = "object"
    OBJECT_ARRAY = "object[]"


class WeaviateSettings(BaseSettings):
    """Weaviate connection settings from environment."""

    host: str = Field(default="localhost")
    port: int = Field(default=8080, ge=1, le=65535)
    grpc_port: int = Field(default=50051, ge=1, le=65535)
    scheme: str = Field(default="http")
    api_key: SecretStr | None = Field(default=None)
    cluster_url: str | None = Field(default=None)
    timeout_config: tuple[int, int] = Field(default=(5, 60))
    startup_period: int = Field(default=5)
    embedded: bool = Field(default=False)
    model_config = SettingsConfigDict(
        env_prefix="WEAVIATE_", env_file=".env", extra="ignore"
    )

    def get_url(self) -> str:
        """Build Weaviate connection URL."""
        if self.cluster_url:
            return self.cluster_url
        return f"{self.scheme}://{self.host}:{self.port}"


@dataclass
class PropertyConfig:
    """Configuration for a collection property."""

    name: str
    data_type: PropertyDataType
    description: str | None = None
    tokenization: str | None = None
    skip_vectorization: bool = False
    index_filterable: bool = True
    index_searchable: bool = True


@dataclass
class CollectionConfig:
    """Configuration for a Weaviate collection (class)."""

    name: str
    description: str | None = None
    properties: list[PropertyConfig] = field(default_factory=list)
    vectorizer: str | None = None
    vector_index_type: str = "hnsw"
    distance_metric: VectorDistanceMetric = VectorDistanceMetric.COSINE
    replication_factor: int = 1
    multi_tenancy_enabled: bool = False


@dataclass
class SearchResult:
    """Container for search results."""

    uuid: UUID
    properties: dict[str, Any]
    vector: list[float] | None = None
    distance: float | None = None
    certainty: float | None = None
    score: float | None = None


class WeaviateClient:
    """Async-compatible Weaviate client for vector operations and schema management."""

    def __init__(self, settings: WeaviateSettings | None = None) -> None:
        self._settings = settings or WeaviateSettings()
        self._client: weaviate.WeaviateClient | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_ready()

    async def connect(self) -> None:
        """Initialize Weaviate connection."""
        if self._client is not None:
            return
        try:
            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(None, self._create_client)
            await loop.run_in_executor(None, self._client.connect)
            logger.info("weaviate_connected", url=self._settings.get_url())
        except Exception as e:
            raise InfrastructureError(f"Failed to connect to Weaviate: {e}", cause=e)

    def _create_client(self) -> weaviate.WeaviateClient:
        """Create Weaviate client instance."""
        if self._settings.cluster_url:
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=self._settings.cluster_url,
                auth_credentials=AuthApiKey(self._settings.api_key.get_secret_value())
                if self._settings.api_key
                else None,
            )
        if self._settings.embedded:
            return weaviate.connect_to_embedded()
        return weaviate.connect_to_local(
            host=self._settings.host,
            port=self._settings.port,
            grpc_port=self._settings.grpc_port,
            auth_credentials=AuthApiKey(self._settings.api_key.get_secret_value())
            if self._settings.api_key
            else None,
        )

    async def disconnect(self) -> None:
        """Close Weaviate connection gracefully."""
        if self._client:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._client.close)
            self._client = None
            logger.info("weaviate_disconnected")

    def _ensure_connected(self) -> weaviate.WeaviateClient:
        if not self._client:
            raise InfrastructureError("Weaviate client not connected")
        return self._client

    async def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new collection with specified configuration."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()
        try:

            def _create() -> bool:
                if client.collections.exists(config.name):
                    return False
                properties = [self._build_property(p) for p in config.properties]
                distance = self._get_distance_enum(config.distance_metric)
                client.collections.create(
                    name=config.name,
                    description=config.description,
                    properties=properties,
                    vectorizer_config=Configure.Vectorizer.none()
                    if not config.vectorizer
                    else None,
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=distance
                    ),
                    replication_config=Configure.replication(
                        factor=config.replication_factor
                    ),
                    multi_tenancy_config=Configure.multi_tenancy(
                        enabled=config.multi_tenancy_enabled
                    ),
                )
                return True

            created = await loop.run_in_executor(None, _create)
            if created:
                logger.info("weaviate_collection_created", name=config.name)
            return created
        except Exception as e:
            raise InfrastructureError(f"Failed to create collection: {e}", cause=e)

    def _build_property(self, prop: PropertyConfig) -> Property:
        """Build Weaviate property from config."""
        dtype = getattr(DataType, prop.data_type.name, DataType.TEXT)
        return Property(
            name=prop.name,
            data_type=dtype,
            description=prop.description,
            skip_vectorization=prop.skip_vectorization,
            index_filterable=prop.index_filterable,
            index_searchable=prop.index_searchable,
        )

    def _get_distance_enum(self, metric: VectorDistanceMetric) -> VectorDistances:
        """Convert distance metric enum."""
        mapping = {
            VectorDistanceMetric.COSINE: VectorDistances.COSINE,
            VectorDistanceMetric.DOT: VectorDistances.DOT,
            VectorDistanceMetric.L2_SQUARED: VectorDistances.L2_SQUARED,
            VectorDistanceMetric.HAMMING: VectorDistances.HAMMING,
            VectorDistanceMetric.MANHATTAN: VectorDistances.MANHATTAN,
        }
        return mapping.get(metric, VectorDistances.COSINE)

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _delete() -> bool:
            if not client.collections.exists(name):
                return False
            client.collections.delete(name)
            return True

        deleted = await loop.run_in_executor(None, _delete)
        if deleted:
            logger.info("weaviate_collection_deleted", name=name)
        return deleted

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, client.collections.exists, name)

    async def insert(
        self,
        collection: str,
        properties: dict[str, Any],
        vector: list[float] | None = None,
        uuid: UUID | None = None,
    ) -> UUID:
        """Insert object into collection."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()
        obj_uuid = uuid or uuid4()

        def _insert() -> UUID:
            coll = client.collections.get(collection)
            coll.data.insert(properties=properties, vector=vector, uuid=obj_uuid)
            return obj_uuid

        result = await loop.run_in_executor(None, _insert)
        logger.debug("weaviate_insert", collection=collection, uuid=str(result))
        return result

    async def insert_batch(
        self,
        collection: str,
        objects: list[dict[str, Any]],
        vectors: list[list[float]] | None = None,
    ) -> list[UUID]:
        """Batch insert multiple objects."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _batch_insert() -> list[UUID]:
            coll = client.collections.get(collection)
            uuids = []
            with coll.batch.dynamic() as batch:
                for i, obj in enumerate(objects):
                    obj_uuid = uuid4()
                    vec = vectors[i] if vectors else None
                    batch.add_object(properties=obj, vector=vec, uuid=obj_uuid)
                    uuids.append(obj_uuid)
            return uuids

        result = await loop.run_in_executor(None, _batch_insert)
        logger.info("weaviate_batch_insert", collection=collection, count=len(result))
        return result

    async def get_by_id(self, collection: str, uuid: UUID) -> SearchResult | None:
        """Get object by UUID."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _get() -> SearchResult | None:
            coll = client.collections.get(collection)
            obj = coll.query.fetch_object_by_id(uuid, include_vector=True)
            if not obj:
                return None
            return SearchResult(
                uuid=obj.uuid, properties=obj.properties, vector=obj.vector
            )

        return await loop.run_in_executor(None, _get)

    async def delete_by_id(self, collection: str, uuid: UUID) -> bool:
        """Delete object by UUID."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _delete() -> bool:
            coll = client.collections.get(collection)
            coll.data.delete_by_id(uuid)
            return True

        return await loop.run_in_executor(None, _delete)

    async def vector_search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        return_vector: bool = False,
    ) -> list[SearchResult]:
        """Perform vector similarity search."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _search() -> list[SearchResult]:
            coll = client.collections.get(collection)
            query = coll.query.near_vector(
                near_vector=vector,
                limit=limit,
                include_vector=return_vector,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
            return [
                SearchResult(
                    uuid=obj.uuid,
                    properties=obj.properties,
                    vector=obj.vector if return_vector else None,
                    distance=obj.metadata.distance,
                    certainty=obj.metadata.certainty,
                )
                for obj in query.objects
            ]

        return await loop.run_in_executor(None, _search)

    async def hybrid_search(
        self,
        collection: str,
        query: str,
        vector: list[float] | None = None,
        limit: int = 10,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Perform hybrid (keyword + vector) search."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _search() -> list[SearchResult]:
            coll = client.collections.get(collection)
            result = coll.query.hybrid(
                query=query,
                vector=vector,
                limit=limit,
                alpha=alpha,
                return_metadata=MetadataQuery(score=True),
            )
            return [
                SearchResult(
                    uuid=obj.uuid, properties=obj.properties, score=obj.metadata.score
                )
                for obj in result.objects
            ]

        return await loop.run_in_executor(None, _search)

    async def update(
        self,
        collection: str,
        uuid: UUID,
        properties: dict[str, Any],
        vector: list[float] | None = None,
    ) -> bool:
        """Update object properties and optionally vector."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _update() -> bool:
            coll = client.collections.get(collection)
            coll.data.update(uuid=uuid, properties=properties, vector=vector)
            return True

        return await loop.run_in_executor(None, _update)

    async def count(self, collection: str) -> int:
        """Count objects in collection."""
        client = self._ensure_connected()
        loop = asyncio.get_running_loop()

        def _count() -> int:
            coll = client.collections.get(collection)
            agg = coll.aggregate.over_all(total_count=True)
            return agg.total_count or 0

        return await loop.run_in_executor(None, _count)

    async def check_health(self) -> dict[str, Any]:
        """Check Weaviate health status."""
        try:
            client = self._ensure_connected()
            loop = asyncio.get_running_loop()
            is_ready = await loop.run_in_executor(None, client.is_ready)
            meta = await loop.run_in_executor(None, lambda: client.get_meta())
            return {
                "status": "healthy" if is_ready else "unhealthy",
                "version": meta.get("version", "unknown"),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


async def create_weaviate_client(
    settings: WeaviateSettings | None = None,
) -> WeaviateClient:
    """Factory function to create and connect a Weaviate client."""
    client = WeaviateClient(settings)
    await client.connect()
    return client
