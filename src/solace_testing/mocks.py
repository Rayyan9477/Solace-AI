"""Solace-AI Testing Library - Mock services and clients."""

from __future__ import annotations

import copy
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class MockQueryResult(BaseModel):
    """Mock database query result."""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    columns: list[str] = Field(default_factory=list)


class MockPostgresClient:
    """In-memory PostgreSQL mock for testing."""

    def __init__(self) -> None:
        self._tables: dict[str, list[dict[str, Any]]] = {}
        self._sequences: dict[str, int] = {}
        self._transaction_stack: list[dict[str, list[dict[str, Any]]]] = []
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        logger.debug("Mock PostgreSQL connected")

    async def disconnect(self) -> None:
        self._connected = False
        logger.debug("Mock PostgreSQL disconnected")

    def _check_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to database")

    async def execute(self, query: str, params: tuple[Any, ...] | None = None) -> MockQueryResult:
        self._check_connected()
        q = query.lower().strip()
        if q.startswith("select"):
            return await self._handle_select(query, params)
        elif q.startswith("insert"):
            return await self._handle_insert(query, params)
        elif q.startswith("update"):
            return await self._handle_update(query, params)
        elif q.startswith("delete"):
            return await self._handle_delete(query, params)
        return MockQueryResult()

    async def _handle_select(self, query: str, params: tuple[Any, ...] | None) -> MockQueryResult:
        table = self._extract_table_name(query, "from")
        rows = self._tables.get(table, [])
        return MockQueryResult(rows=rows, row_count=len(rows))

    async def _handle_insert(self, query: str, params: tuple[Any, ...] | None) -> MockQueryResult:
        table = self._extract_table_name(query, "into")
        row = {"id": str(uuid.uuid4())}
        if params:
            for i, val in enumerate(params):
                row[f"col_{i}"] = val
        self._tables.setdefault(table, []).append(row)
        return MockQueryResult(rows=[row], row_count=1)

    async def _handle_update(self, query: str, params: tuple[Any, ...] | None) -> MockQueryResult:
        table = self._extract_table_name(query, "update")
        return MockQueryResult(row_count=len(self._tables.get(table, [])))

    async def _handle_delete(self, query: str, params: tuple[Any, ...] | None) -> MockQueryResult:
        table = self._extract_table_name(query, "from")
        deleted = len(self._tables.get(table, []))
        self._tables[table] = []
        return MockQueryResult(row_count=deleted)

    def _extract_table_name(self, query: str, keyword: str) -> str:
        idx = query.lower().find(keyword)
        if idx == -1:
            return "unknown"
        remaining = query[idx + len(keyword):].strip()
        return remaining.split()[0].strip(";").strip("(")

    async def begin_transaction(self) -> None:
        self._transaction_stack.append(copy.deepcopy(self._tables))

    async def commit(self) -> None:
        if self._transaction_stack:
            self._transaction_stack.pop()

    async def rollback(self) -> None:
        if self._transaction_stack:
            self._tables = self._transaction_stack.pop()

    def insert_test_data(self, table: str, rows: list[dict[str, Any]]) -> None:
        self._tables.setdefault(table, []).extend(rows)

    def get_table_data(self, table: str) -> list[dict[str, Any]]:
        return self._tables.get(table, [])


class MockRedisClient:
    """In-memory Redis mock for testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._ttls: dict[str, float] = {}
        self._sets: dict[str, set[str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._data[key] = value
        if ex:
            self._ttls[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                count += 1
        return count

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self._data)

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._data:
            self._ttls[key] = seconds
            return True
        return False

    async def ttl(self, key: str) -> int:
        return int(self._ttls.get(key, -1))

    async def incr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    async def sadd(self, key: str, *members: str) -> int:
        self._sets.setdefault(key, set()).update(members)
        return len(members)

    async def smembers(self, key: str) -> set[str]:
        return self._sets.get(key, set())

    async def lpush(self, key: str, *values: str) -> int:
        self._lists.setdefault(key, []).extend(reversed(values))
        return len(self._lists[key])

    async def lrange(self, key: str, start: int, stop: int) -> list[str]:
        lst = self._lists.get(key, [])
        return lst[start:stop + 1 if stop >= 0 else None]

    async def hset(self, key: str, field: str, value: str) -> int:
        h = self._hashes.setdefault(key, {})
        is_new = field not in h
        h[field] = value
        return 1 if is_new else 0

    async def hget(self, key: str, field: str) -> str | None:
        return self._hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        return self._hashes.get(key, {})

    async def flushdb(self) -> None:
        self._data.clear()
        self._ttls.clear()
        self._sets.clear()
        self._lists.clear()
        self._hashes.clear()


class MockWeaviateClient:
    """In-memory Weaviate mock for testing."""

    def __init__(self) -> None:
        self._collections: dict[str, list[dict[str, Any]]] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def create_collection(self, name: str, properties: list[dict[str, Any]],
                                 vector_config: dict[str, Any] | None = None) -> None:
        self._schemas[name] = {"properties": properties, "vector_config": vector_config}
        self._collections[name] = []

    async def delete_collection(self, name: str) -> None:
        self._schemas.pop(name, None)
        self._collections.pop(name, None)

    async def insert(self, collection: str, properties: dict[str, Any],
                     vector: list[float] | None = None) -> str:
        obj_id = str(uuid.uuid4())
        self._collections.setdefault(collection, []).append(
            {"id": obj_id, "properties": properties, "vector": vector}
        )
        return obj_id

    async def batch_insert(self, collection: str, objects: list[dict[str, Any]]) -> list[str]:
        ids = []
        for obj in objects:
            obj_id = await self.insert(collection, obj.get("properties", obj), obj.get("vector"))
            ids.append(obj_id)
        return ids

    async def get_by_id(self, collection: str, obj_id: str) -> dict[str, Any] | None:
        for obj in self._collections.get(collection, []):
            if obj["id"] == obj_id:
                return obj
        return None

    async def search(self, collection: str, vector: list[float], limit: int = 10,
                     filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        objects = self._collections.get(collection, [])
        scored = [(obj, self._cosine_similarity(vector, obj.get("vector", []))) for obj in objects]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"object": obj, "score": score} for obj, score in scored[:limit]]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a, norm_b = sum(x * x for x in a) ** 0.5, sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class MockLLMResponse(BaseModel):
    """Mock LLM response structure."""
    content: str = ""
    finish_reason: str = "stop"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str = "mock-model"
    latency_ms: float = 10.0


class MockLLMClient:
    """Mock LLM client for deterministic testing."""

    def __init__(self) -> None:
        self._responses: list[MockLLMResponse] = []
        self._response_index = 0
        self._requests: list[dict[str, Any]] = []
        self._default_response: MockLLMResponse | None = None

    def set_response(self, response: MockLLMResponse) -> None:
        self._responses.append(response)

    def set_responses(self, responses: list[MockLLMResponse]) -> None:
        self._responses.extend(responses)

    def set_default_response(self, content: str) -> None:
        self._default_response = MockLLMResponse(content=content)

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                       system_prompt: str | None = None) -> MockLLMResponse:
        self._requests.append({"messages": messages, "tools": tools, "system_prompt": system_prompt})
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            return response
        if self._default_response:
            return self._default_response
        return MockLLMResponse(content=f"Mock response for: {messages[-1].get('content', '')[:50]}")

    async def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                     system_prompt: str | None = None) -> AsyncIterator[dict[str, Any]]:
        response = await self.complete(messages, tools, system_prompt)
        words = response.content.split()
        for i, word in enumerate(words):
            yield {"content": word + " ", "is_final": i == len(words) - 1}

    def get_requests(self) -> list[dict[str, Any]]:
        return self._requests.copy()

    def reset(self) -> None:
        self._responses.clear()
        self._response_index = 0
        self._requests.clear()


class MockEventPublisher:
    """Mock event publisher for testing."""

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._handlers: dict[str, list[Callable[..., Any]]] = {}

    async def publish(self, topic: str, event: dict[str, Any], key: str | None = None) -> str:
        event_id = str(uuid.uuid4())
        self._events.setdefault(topic, []).append(
            {"id": event_id, "key": key, "event": event, "published_at": datetime.now(timezone.utc)}
        )
        for handler in self._handlers.get(topic, []):
            await handler(event)
        return event_id

    def subscribe(self, topic: str, handler: Callable[..., Any]) -> None:
        self._handlers.setdefault(topic, []).append(handler)

    def get_events(self, topic: str) -> list[dict[str, Any]]:
        return [e["event"] for e in self._events.get(topic, [])]

    def clear(self) -> None:
        self._events.clear()


class MockHTTPClient:
    """Mock HTTP client for testing external API calls."""

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}
        self._requests: list[dict[str, Any]] = []
        self._error_routes: dict[str, Exception] = {}

    def register_response(self, method: str, url: str, status: int = 200,
                          json_data: dict[str, Any] | None = None, text: str | None = None,
                          headers: dict[str, str] | None = None) -> None:
        key = f"{method.upper()}:{url}"
        self._responses[key] = {"status": status, "json": json_data, "text": text, "headers": headers or {}}

    def register_error(self, method: str, url: str, error: Exception) -> None:
        self._error_routes[f"{method.upper()}:{url}"] = error

    async def request(self, method: str, url: str, headers: dict[str, str] | None = None,
                      json_data: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        key = f"{method.upper()}:{url}"
        self._requests.append({"method": method, "url": url, "headers": headers, "json": json_data, "params": params})
        if key in self._error_routes:
            raise self._error_routes[key]
        return self._responses.get(key, {"status": 404, "json": None, "text": "Not Found", "headers": {}})

    async def get(self, url: str, **kwargs: Any) -> dict[str, Any]:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> dict[str, Any]:
        return await self.request("POST", url, **kwargs)

    def get_requests(self, method: str | None = None) -> list[dict[str, Any]]:
        if method:
            return [r for r in self._requests if r["method"] == method.upper()]
        return self._requests.copy()

    def clear(self) -> None:
        self._responses.clear()
        self._requests.clear()
        self._error_routes.clear()
