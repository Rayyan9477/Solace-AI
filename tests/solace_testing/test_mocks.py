"""Unit tests for Solace-AI Testing Library - Mocks module."""

from __future__ import annotations

import pytest

from solace_testing.mocks import (
    MockEventPublisher,
    MockHTTPClient,
    MockLLMClient,
    MockLLMResponse,
    MockPostgresClient,
    MockQueryResult,
    MockRedisClient,
    MockWeaviateClient,
)


class TestMockPostgresClient:
    """Tests for MockPostgresClient."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        client = MockPostgresClient()
        await client.connect()
        assert client._connected is True
        await client.disconnect()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_execute_select(self) -> None:
        client = MockPostgresClient()
        await client.connect()
        client.insert_test_data("users", [{"id": "1", "name": "Test"}])
        result = await client.execute("SELECT * FROM users")
        assert isinstance(result, MockQueryResult)
        assert len(result.rows) == 1
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_execute_insert(self) -> None:
        client = MockPostgresClient()
        await client.connect()
        result = await client.execute("INSERT INTO users VALUES ($1)", ("John",))
        assert result.row_count == 1
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_transaction(self) -> None:
        client = MockPostgresClient()
        await client.connect()
        await client.begin_transaction()
        client.insert_test_data("test", [{"id": "1"}])
        await client.rollback()
        data = client.get_table_data("test")
        assert len(data) == 0
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_not_connected_error(self) -> None:
        client = MockPostgresClient()
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.execute("SELECT 1")


class TestMockRedisClient:
    """Tests for MockRedisClient."""

    @pytest.mark.asyncio
    async def test_get_set(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.set("key1", "value1")
        result = await client.get("key1")
        assert result == "value1"
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.set("key1", "value1")
        count = await client.delete("key1")
        assert count == 1
        result = await client.get("key1")
        assert result is None
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.set("key1", "value1")
        count = await client.exists("key1", "key2")
        assert count == 1
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_incr(self) -> None:
        client = MockRedisClient()
        await client.connect()
        val1 = await client.incr("counter")
        val2 = await client.incr("counter")
        assert val1 == 1
        assert val2 == 2
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_set_operations(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.sadd("myset", "a", "b", "c")
        members = await client.smembers("myset")
        assert members == {"a", "b", "c"}
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_list_operations(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.lpush("mylist", "a", "b")
        items = await client.lrange("mylist", 0, -1)
        assert "a" in items and "b" in items
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_hash_operations(self) -> None:
        client = MockRedisClient()
        await client.connect()
        await client.hset("myhash", "field1", "value1")
        val = await client.hget("myhash", "field1")
        assert val == "value1"
        all_vals = await client.hgetall("myhash")
        assert all_vals == {"field1": "value1"}
        await client.disconnect()


class TestMockWeaviateClient:
    """Tests for MockWeaviateClient."""

    @pytest.mark.asyncio
    async def test_create_collection(self) -> None:
        client = MockWeaviateClient()
        await client.connect()
        await client.create_collection("Test", [{"name": "title"}])
        assert "Test" in client._schemas
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_insert_get(self) -> None:
        client = MockWeaviateClient()
        await client.connect()
        await client.create_collection("Test", [])
        obj_id = await client.insert("Test", {"title": "test"}, [0.1, 0.2])
        retrieved = await client.get_by_id("Test", obj_id)
        assert retrieved is not None
        assert retrieved["properties"]["title"] == "test"
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_batch_insert(self) -> None:
        client = MockWeaviateClient()
        await client.connect()
        await client.create_collection("Test", [])
        objects = [{"properties": {"title": f"item{i}"}, "vector": [0.1]} for i in range(5)]
        ids = await client.batch_insert("Test", objects)
        assert len(ids) == 5
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_search(self) -> None:
        client = MockWeaviateClient()
        await client.connect()
        await client.create_collection("Test", [])
        await client.insert("Test", {"title": "test1"}, [1.0, 0.0])
        await client.insert("Test", {"title": "test2"}, [0.0, 1.0])
        results = await client.search("Test", [1.0, 0.0], limit=1)
        assert len(results) == 1
        await client.disconnect()


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_default_response(self) -> None:
        client = MockLLMClient()
        client.set_default_response("Default reply")
        response = await client.complete([{"role": "user", "content": "Hello"}])
        assert response.content == "Default reply"

    @pytest.mark.asyncio
    async def test_sequential_responses(self) -> None:
        client = MockLLMClient()
        client.set_responses([
            MockLLMResponse(content="First"),
            MockLLMResponse(content="Second"),
        ])
        r1 = await client.complete([{"role": "user", "content": "1"}])
        r2 = await client.complete([{"role": "user", "content": "2"}])
        assert r1.content == "First"
        assert r2.content == "Second"

    @pytest.mark.asyncio
    async def test_request_recording(self) -> None:
        client = MockLLMClient()
        client.set_default_response("OK")
        await client.complete(
            [{"role": "user", "content": "Test"}],
            tools=[{"name": "test_tool"}],
            system_prompt="Be helpful",
        )
        requests = client.get_requests()
        assert len(requests) == 1
        assert requests[0]["system_prompt"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_stream(self) -> None:
        client = MockLLMClient()
        client.set_default_response("Hello world")
        chunks = []
        async for chunk in client.stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
        assert len(chunks) > 0

    def test_reset(self) -> None:
        client = MockLLMClient()
        client.set_responses([MockLLMResponse(content="Test")])
        client.reset()
        assert len(client._responses) == 0
        assert len(client._requests) == 0


class TestMockEventPublisher:
    """Tests for MockEventPublisher."""

    @pytest.mark.asyncio
    async def test_publish(self) -> None:
        publisher = MockEventPublisher()
        event_id = await publisher.publish("test-topic", {"type": "test", "data": 123})
        assert event_id is not None
        events = publisher.get_events("test-topic")
        assert len(events) == 1
        assert events[0]["type"] == "test"

    @pytest.mark.asyncio
    async def test_subscribe(self) -> None:
        publisher = MockEventPublisher()
        received = []

        async def handler(event: dict) -> None:
            received.append(event)

        publisher.subscribe("test-topic", handler)
        await publisher.publish("test-topic", {"msg": "hello"})
        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    def test_clear(self) -> None:
        publisher = MockEventPublisher()
        publisher._events["topic"] = [{"event": {}}]
        publisher.clear()
        assert len(publisher._events) == 0


class TestMockHTTPClient:
    """Tests for MockHTTPClient."""

    @pytest.mark.asyncio
    async def test_register_response(self) -> None:
        client = MockHTTPClient()
        client.register_response("GET", "/api/test", status=200, json_data={"ok": True})
        response = await client.request("GET", "/api/test")
        assert response["status"] == 200
        assert response["json"]["ok"] is True

    @pytest.mark.asyncio
    async def test_register_error(self) -> None:
        client = MockHTTPClient()
        client.register_error("GET", "/api/fail", ValueError("Test error"))
        with pytest.raises(ValueError, match="Test error"):
            await client.request("GET", "/api/fail")

    @pytest.mark.asyncio
    async def test_get_post_shortcuts(self) -> None:
        client = MockHTTPClient()
        client.register_response("GET", "/test", status=200)
        client.register_response("POST", "/test", status=201)
        get_resp = await client.get("/test")
        post_resp = await client.post("/test", json_data={"data": 1})
        assert get_resp["status"] == 200
        assert post_resp["status"] == 201

    @pytest.mark.asyncio
    async def test_request_recording(self) -> None:
        client = MockHTTPClient()
        await client.request("POST", "/api/create", json_data={"name": "test"})
        requests = client.get_requests("POST")
        assert len(requests) == 1
        assert requests[0]["json"]["name"] == "test"

    def test_clear(self) -> None:
        client = MockHTTPClient()
        client.register_response("GET", "/test", status=200)
        client.clear()
        assert len(client._responses) == 0
