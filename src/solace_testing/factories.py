"""Solace-AI Testing Library - Test data factories."""

from __future__ import annotations

import random
import string
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)
T = TypeVar("T")


class FactoryConfig(BaseModel):
    """Configuration for data factories."""
    seed: int | None = None
    locale: str = "en_US"
    default_batch_size: int = 10
    use_realistic_data: bool = True


class FactorySequence:
    """Auto-incrementing sequence generator."""

    def __init__(self, start: int = 1, prefix: str = "") -> None:
        self._current = start
        self._prefix = prefix

    def next(self) -> str:
        value = f"{self._prefix}{self._current}"
        self._current += 1
        return value

    def reset(self, start: int = 1) -> None:
        self._current = start


class BaseFactory(ABC, Generic[T]):
    """Abstract base factory with builder pattern support."""
    _sequences: dict[str, FactorySequence] = {}
    _config: FactoryConfig = FactoryConfig()

    def __init__(self, config: FactoryConfig | None = None) -> None:
        if config:
            self._config = config
        if self._config.seed is not None:
            random.seed(self._config.seed)

    @abstractmethod
    def create(self, **overrides: Any) -> T:
        raise NotImplementedError

    def create_batch(self, count: int, **overrides: Any) -> list[T]:
        return [self.create(**overrides) for _ in range(count)]

    @classmethod
    def get_sequence(cls, name: str, start: int = 1, prefix: str = "") -> FactorySequence:
        key = f"{cls.__name__}:{name}"
        if key not in cls._sequences:
            cls._sequences[key] = FactorySequence(start, prefix)
        return cls._sequences[key]

    @classmethod
    def reset_sequences(cls) -> None:
        prefix = f"{cls.__name__}:"
        for key in [k for k in cls._sequences if k.startswith(prefix)]:
            cls._sequences[key].reset()

    @staticmethod
    def random_string(length: int = 10, chars: str | None = None) -> str:
        return "".join(random.choices(chars or string.ascii_lowercase, k=length))

    @staticmethod
    def random_email(domain: str = "test.com") -> str:
        return f"{BaseFactory.random_string(8)}@{domain}"

    @staticmethod
    def random_uuid() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def random_datetime(start: datetime | None = None, end: datetime | None = None) -> datetime:
        start = start or datetime.now(timezone.utc) - timedelta(days=365)
        end = end or datetime.now(timezone.utc)
        delta = (end - start).total_seconds()
        return start + timedelta(seconds=random.uniform(0, delta))


class EntityIdFactory(BaseFactory[str]):
    """Factory for generating entity IDs."""

    def __init__(self, prefix: str = "", config: FactoryConfig | None = None) -> None:
        super().__init__(config)
        self._prefix = prefix

    def create(self, **overrides: Any) -> str:
        return overrides.get("value", f"{self._prefix}{self.random_uuid()}")


class EntityMetadataDict(BaseModel):
    """Entity metadata structure."""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    created_by: str | None = None
    updated_by: str | None = None


class EntityFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating domain entities."""

    def __init__(self, entity_type: str = "Entity", config: FactoryConfig | None = None) -> None:
        super().__init__(config)
        self._entity_type = entity_type
        self._id_factory = EntityIdFactory(prefix=f"{entity_type.lower()}-")

    def create(self, **overrides: Any) -> dict[str, Any]:
        seq = self.get_sequence("id")
        entity = {
            "id": self._id_factory.create(),
            "name": overrides.get("name", f"{self._entity_type}-{seq.next()}"),
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": 1, "created_by": overrides.get("created_by"), "updated_by": None,
            },
        }
        entity.update({k: v for k, v in overrides.items() if k not in ("name", "created_by")})
        return entity


class UserFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating user data."""
    SAMPLE_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    def create(self, **overrides: Any) -> dict[str, Any]:
        seq = self.get_sequence("user")
        name = overrides.get("name", random.choice(self.SAMPLE_NAMES))
        return {
            "id": overrides.get("id", f"user-{self.random_uuid()}"),
            "name": name,
            "email": overrides.get("email", f"{name.lower()}{seq.next()}@test.com"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": overrides.get("status", "active"),
            "preferences": overrides.get("preferences", {}),
            "metadata": overrides.get("metadata", {}),
        }


class SessionFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating session data."""

    def __init__(self, config: FactoryConfig | None = None) -> None:
        super().__init__(config)
        self._user_factory = UserFactory(config)

    def create(self, **overrides: Any) -> dict[str, Any]:
        user_id = overrides.get("user_id") or self._user_factory.create()["id"]
        return {
            "id": overrides.get("id", f"session-{self.random_uuid()}"),
            "user_id": user_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": overrides.get("ended_at"),
            "status": overrides.get("status", "active"),
            "correlation_id": overrides.get("correlation_id", f"corr-{self.random_uuid()[:8]}"),
            "context": overrides.get("context", {}),
        }


class MessageFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating chat messages."""
    SAMPLE_USER_MESSAGES = ["I've been feeling anxious lately.", "I'm having trouble sleeping.",
                            "Work has been really stressful.", "I need help managing my emotions.", "Can you help me understand this?"]
    SAMPLE_ASSISTANT_MESSAGES = ["I understand. Can you tell me more about that?", "That sounds challenging. How does it make you feel?",
                                  "Let's explore this together.", "I'm here to help. What specifically concerns you?"]

    def create(self, **overrides: Any) -> dict[str, Any]:
        role = overrides.get("role", "user")
        content = overrides.get("content", random.choice(self.SAMPLE_USER_MESSAGES if role == "user" else self.SAMPLE_ASSISTANT_MESSAGES))
        return {
            "id": overrides.get("id", f"msg-{self.random_uuid()}"),
            "role": role, "content": content,
            "session_id": overrides.get("session_id", f"session-{self.random_uuid()}"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": overrides.get("metadata", {}),
        }

    def create_conversation(self, turns: int = 3, session_id: str | None = None) -> list[dict[str, Any]]:
        session_id = session_id or f"session-{self.random_uuid()}"
        return [self.create(role="user" if i % 2 == 0 else "assistant", session_id=session_id) for i in range(turns * 2)]


class EventFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating domain events."""

    def __init__(self, event_type: str = "BaseEvent", config: FactoryConfig | None = None) -> None:
        super().__init__(config)
        self._event_type = event_type

    def create(self, **overrides: Any) -> dict[str, Any]:
        return {
            "event_id": overrides.get("event_id", f"evt-{self.random_uuid()}"),
            "event_type": overrides.get("event_type", self._event_type),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": overrides.get("correlation_id", f"corr-{self.random_uuid()[:8]}"),
            "source_service": overrides.get("source_service", "test-service"),
            "payload": overrides.get("payload", {}),
            "metadata": overrides.get("metadata", {}),
        }


class DiagnosisFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating clinical diagnosis data."""
    SAMPLE_CONDITIONS = [("GAD", "Generalized Anxiety Disorder", "F41.1"), ("MDD", "Major Depressive Disorder", "F33.0"),
                         ("PTSD", "Post-Traumatic Stress Disorder", "F43.10"), ("OCD", "Obsessive-Compulsive Disorder", "F42.2"),
                         ("SAD", "Social Anxiety Disorder", "F40.10")]

    def create(self, **overrides: Any) -> dict[str, Any]:
        condition = overrides.get("condition") or random.choice(self.SAMPLE_CONDITIONS)
        return {
            "id": overrides.get("id", f"diag-{self.random_uuid()}"),
            "user_id": overrides.get("user_id", f"user-{self.random_uuid()}"),
            "session_id": overrides.get("session_id", f"session-{self.random_uuid()}"),
            "condition_code": condition[0], "condition_name": condition[1], "icd_code": condition[2],
            "confidence": overrides.get("confidence", random.uniform(0.6, 0.95)),
            "severity": overrides.get("severity", random.choice(["mild", "moderate", "severe"])),
            "supporting_evidence": overrides.get("supporting_evidence", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


class SafetyAssessmentFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating safety assessment data."""

    def create(self, **overrides: Any) -> dict[str, Any]:
        risk_level = overrides.get("risk_level", random.choice(["LOW", "ELEVATED", "HIGH"]))
        return {
            "id": overrides.get("id", f"safety-{self.random_uuid()}"),
            "user_id": overrides.get("user_id", f"user-{self.random_uuid()}"),
            "session_id": overrides.get("session_id", f"session-{self.random_uuid()}"),
            "message_id": overrides.get("message_id", f"msg-{self.random_uuid()}"),
            "risk_level": risk_level, "crisis_detected": risk_level == "HIGH",
            "risk_factors": overrides.get("risk_factors", []),
            "recommended_actions": overrides.get("recommended_actions", []),
            "assessed_at": datetime.now(timezone.utc).isoformat(),
        }


class VectorFactory(BaseFactory[list[float]]):
    """Factory for generating embedding vectors."""

    def __init__(self, dimensions: int = 1536, config: FactoryConfig | None = None) -> None:
        super().__init__(config)
        self._dimensions = dimensions

    def create(self, **overrides: Any) -> list[float]:
        dims = overrides.get("dimensions", self._dimensions)
        vector = [random.gauss(0, 1) for _ in range(dims)]
        norm = sum(x * x for x in vector) ** 0.5
        return [x / norm for x in vector] if norm > 0 else vector


class ToolCallFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating LLM tool calls."""

    def create(self, **overrides: Any) -> dict[str, Any]:
        return {
            "id": overrides.get("id", f"tc-{self.random_uuid()[:8]}"),
            "name": overrides.get("name", "sample_tool"),
            "arguments": overrides.get("arguments", {"param1": "value1"}),
        }


class LLMResponseFactory(BaseFactory[dict[str, Any]]):
    """Factory for generating LLM response data."""

    def create(self, **overrides: Any) -> dict[str, Any]:
        return {
            "content": overrides.get("content", "This is a test response from the LLM."),
            "finish_reason": overrides.get("finish_reason", "stop"),
            "tool_calls": overrides.get("tool_calls", []),
            "usage": overrides.get("usage", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}),
            "model": overrides.get("model", "test-model"),
            "latency_ms": overrides.get("latency_ms", random.uniform(50, 500)),
        }


class FactoryRegistry:
    """Registry for managing and accessing factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BaseFactory[Any]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self._factories.update({
            "entity": EntityFactory(), "user": UserFactory(), "session": SessionFactory(),
            "message": MessageFactory(), "event": EventFactory(), "diagnosis": DiagnosisFactory(),
            "safety": SafetyAssessmentFactory(), "vector": VectorFactory(),
            "tool_call": ToolCallFactory(), "llm_response": LLMResponseFactory(),
        })

    def register(self, name: str, factory: BaseFactory[Any]) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> BaseFactory[Any]:
        if name not in self._factories:
            raise KeyError(f"Factory '{name}' not registered")
        return self._factories[name]

    def create(self, factory_name: str, **overrides: Any) -> Any:
        return self.get(factory_name).create(**overrides)

    def create_batch(self, factory_name: str, count: int, **overrides: Any) -> list[Any]:
        return self.get(factory_name).create_batch(count, **overrides)

    def reset_all(self) -> None:
        for factory in self._factories.values():
            factory.reset_sequences()


_default_registry = FactoryRegistry()


def get_factory_registry() -> FactoryRegistry:
    return _default_registry
