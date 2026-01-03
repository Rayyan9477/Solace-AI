"""Unit tests for Weaviate client module."""
from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
from solace_infrastructure.weaviate import (
    WeaviateClient,
    WeaviateSettings,
    CollectionConfig,
    PropertyConfig,
    PropertyDataType,
    VectorDistanceMetric,
    SearchResult,
    create_weaviate_client,
)


class TestWeaviateSettings:
    """Tests for WeaviateSettings configuration."""

    def test_default_settings(self):
        settings = WeaviateSettings()
        assert settings.host == "localhost"
        assert settings.port == 8080
        assert settings.grpc_port == 50051
        assert settings.scheme == "http"

    def test_get_url_local(self):
        settings = WeaviateSettings()
        url = settings.get_url()
        assert url == "http://localhost:8080"

    def test_get_url_cloud(self):
        settings = WeaviateSettings(cluster_url="https://my-cluster.weaviate.cloud")
        url = settings.get_url()
        assert url == "https://my-cluster.weaviate.cloud"

    def test_custom_settings(self):
        settings = WeaviateSettings(
            host="weaviate.example.com", port=8081, scheme="https"
        )
        assert settings.host == "weaviate.example.com"
        assert settings.port == 8081
        assert settings.scheme == "https"


class TestPropertyDataType:
    """Tests for PropertyDataType enum."""

    def test_data_types(self):
        assert PropertyDataType.TEXT.value == "text"
        assert PropertyDataType.INT.value == "int"
        assert PropertyDataType.NUMBER.value == "number"
        assert PropertyDataType.BOOLEAN.value == "boolean"
        assert PropertyDataType.UUID.value == "uuid"


class TestVectorDistanceMetric:
    """Tests for VectorDistanceMetric enum."""

    def test_distance_metrics(self):
        assert VectorDistanceMetric.COSINE.value == "cosine"
        assert VectorDistanceMetric.DOT.value == "dot"
        assert VectorDistanceMetric.L2_SQUARED.value == "l2-squared"


class TestPropertyConfig:
    """Tests for PropertyConfig dataclass."""

    def test_basic_property(self):
        prop = PropertyConfig(name="title", data_type=PropertyDataType.TEXT)
        assert prop.name == "title"
        assert prop.data_type == PropertyDataType.TEXT
        assert prop.skip_vectorization is False

    def test_property_with_options(self):
        prop = PropertyConfig(
            name="content", data_type=PropertyDataType.TEXT,
            description="Main content", skip_vectorization=True,
            index_filterable=False
        )
        assert prop.description == "Main content"
        assert prop.skip_vectorization is True
        assert prop.index_filterable is False


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_basic_collection(self):
        config = CollectionConfig(name="Articles")
        assert config.name == "Articles"
        assert config.distance_metric == VectorDistanceMetric.COSINE
        assert config.replication_factor == 1

    def test_collection_with_properties(self):
        config = CollectionConfig(
            name="Documents",
            description="Document storage",
            properties=[
                PropertyConfig(name="title", data_type=PropertyDataType.TEXT),
                PropertyConfig(name="content", data_type=PropertyDataType.TEXT),
            ],
            distance_metric=VectorDistanceMetric.DOT
        )
        assert len(config.properties) == 2
        assert config.distance_metric == VectorDistanceMetric.DOT


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_basic_result(self):
        result = SearchResult(
            uuid=uuid4(),
            properties={"title": "Test", "content": "Content"}
        )
        assert result.properties["title"] == "Test"
        assert result.distance is None

    def test_result_with_metadata(self):
        result = SearchResult(
            uuid=uuid4(),
            properties={"title": "Test"},
            vector=[0.1, 0.2, 0.3],
            distance=0.15,
            certainty=0.85
        )
        assert result.distance == 0.15
        assert result.certainty == 0.85
        assert len(result.vector) == 3


class TestWeaviateClient:
    """Tests for WeaviateClient operations."""

    @pytest.fixture
    def mock_weaviate(self):
        mock = MagicMock()
        mock.is_ready.return_value = True
        mock.collections = MagicMock()
        mock.get_meta.return_value = {"version": "1.25.0"}
        return mock

    @pytest.fixture
    def client(self):
        return WeaviateClient(WeaviateSettings())

    def test_initial_state(self, client):
        assert not client.is_connected
        assert client._client is None

    @pytest.mark.asyncio
    async def test_connect(self, client, mock_weaviate):
        with patch.object(client, "_create_client", return_value=mock_weaviate):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_executor = MagicMock()
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_weaviate)
                await client.connect()
                assert client._client is mock_weaviate

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_weaviate):
        client._client = mock_weaviate
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock()
            await client.disconnect()
            assert client._client is None

    @pytest.mark.asyncio
    async def test_collection_exists(self, client, mock_weaviate):
        client._client = mock_weaviate
        mock_weaviate.collections.exists.return_value = True
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
            result = await client.collection_exists("Articles")
            assert result is True

    @pytest.mark.asyncio
    async def test_create_collection(self, client, mock_weaviate):
        client._client = mock_weaviate
        mock_weaviate.collections.exists.return_value = False
        config = CollectionConfig(
            name="TestCollection",
            properties=[PropertyConfig(name="title", data_type=PropertyDataType.TEXT)]
        )
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
            result = await client.create_collection(config)
            assert result is True

    @pytest.mark.asyncio
    async def test_create_collection_already_exists(self, client, mock_weaviate):
        client._client = mock_weaviate
        mock_weaviate.collections.exists.return_value = True
        config = CollectionConfig(name="ExistingCollection")
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=False)
            result = await client.create_collection(config)
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_collection(self, client, mock_weaviate):
        client._client = mock_weaviate
        mock_weaviate.collections.exists.return_value = True
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
            result = await client.delete_collection("TestCollection")
            assert result is True

    @pytest.mark.asyncio
    async def test_insert(self, client, mock_weaviate):
        client._client = mock_weaviate
        obj_uuid = uuid4()
        mock_collection = MagicMock()
        mock_weaviate.collections.get.return_value = mock_collection
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=obj_uuid)
            result = await client.insert(
                "Articles",
                properties={"title": "Test", "content": "Content"},
                vector=[0.1, 0.2, 0.3],
                uuid=obj_uuid
            )
            assert result == obj_uuid

    @pytest.mark.asyncio
    async def test_insert_batch(self, client, mock_weaviate):
        client._client = mock_weaviate
        objects = [{"title": "Doc1"}, {"title": "Doc2"}, {"title": "Doc3"}]
        vectors = [[0.1]*3, [0.2]*3, [0.3]*3]
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_uuids = [uuid4() for _ in objects]
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_uuids)
            result = await client.insert_batch("Articles", objects, vectors)
            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_by_id(self, client, mock_weaviate):
        client._client = mock_weaviate
        obj_uuid = uuid4()
        mock_obj = MagicMock()
        mock_obj.uuid = obj_uuid
        mock_obj.properties = {"title": "Found"}
        mock_obj.vector = [0.1, 0.2]
        expected = SearchResult(uuid=obj_uuid, properties={"title": "Found"}, vector=[0.1, 0.2])
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=expected)
            result = await client.get_by_id("Articles", obj_uuid)
            assert result.uuid == obj_uuid

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, client, mock_weaviate):
        client._client = mock_weaviate
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
            result = await client.get_by_id("Articles", uuid4())
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_by_id(self, client, mock_weaviate):
        client._client = mock_weaviate
        obj_uuid = uuid4()
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
            result = await client.delete_by_id("Articles", obj_uuid)
            assert result is True

    @pytest.mark.asyncio
    async def test_vector_search(self, client, mock_weaviate):
        client._client = mock_weaviate
        search_results = [
            SearchResult(uuid=uuid4(), properties={"title": "Result1"}, distance=0.1),
            SearchResult(uuid=uuid4(), properties={"title": "Result2"}, distance=0.2),
        ]
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=search_results)
            result = await client.vector_search("Articles", [0.1, 0.2, 0.3], limit=5)
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_hybrid_search(self, client, mock_weaviate):
        client._client = mock_weaviate
        search_results = [
            SearchResult(uuid=uuid4(), properties={"title": "Hybrid1"}, score=0.9),
        ]
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=search_results)
            result = await client.hybrid_search("Articles", "test query", limit=10)
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update(self, client, mock_weaviate):
        client._client = mock_weaviate
        obj_uuid = uuid4()
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)
            result = await client.update("Articles", obj_uuid, {"title": "Updated"})
            assert result is True

    @pytest.mark.asyncio
    async def test_count(self, client, mock_weaviate):
        client._client = mock_weaviate
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=42)
            result = await client.count("Articles")
            assert result == 42

    @pytest.mark.asyncio
    async def test_check_health_healthy(self, client, mock_weaviate):
        client._client = mock_weaviate
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(side_effect=[True, {"version": "1.25.0"}])
            health = await client.check_health()
            assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_check_health_unhealthy(self, client):
        client._client = None
        health = await client.check_health()
        assert health["status"] == "unhealthy"


class TestDistanceMetricMapping:
    """Tests for distance metric conversion."""

    def test_get_distance_enum(self):
        client = WeaviateClient()
        from weaviate.classes.config import VectorDistances
        assert client._get_distance_enum(VectorDistanceMetric.COSINE) == VectorDistances.COSINE
        assert client._get_distance_enum(VectorDistanceMetric.DOT) == VectorDistances.DOT


class TestFactoryFunction:
    """Tests for factory function."""

    @pytest.mark.asyncio
    async def test_create_weaviate_client(self):
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True
        with patch("weaviate.connect_to_local", return_value=mock_client):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_client)
                client = await create_weaviate_client(WeaviateSettings())
                assert client._client is mock_client
