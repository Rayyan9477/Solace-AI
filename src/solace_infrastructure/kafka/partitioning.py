"""
Solace-AI Kafka Partitioning Strategies - Custom message partitioning.
Ensures consistent routing for user sessions, safety events, and analytics.
"""
from __future__ import annotations

import hashlib
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)


class PartitionStrategy(str, Enum):
    """Available partitioning strategies."""
    ROUND_ROBIN = "round_robin"
    HASH_KEY = "hash_key"
    USER_AFFINITY = "user_affinity"
    PRIORITY_BASED = "priority_based"
    STICKY = "sticky"
    RANDOM = "random"


@dataclass
class PartitionResult:
    """Result of partition calculation."""
    partition: int
    strategy_used: PartitionStrategy
    key_hash: int | None = None
    metadata: dict[str, Any] | None = None


class PartitionerConfig(BaseModel):
    """Configuration for partitioner behavior."""

    default_strategy: PartitionStrategy = Field(default=PartitionStrategy.HASH_KEY)
    sticky_batch_size: int = Field(default=100, ge=1)
    priority_partition_count: int = Field(default=2, ge=1)
    hash_algorithm: str = Field(default="murmur2")
    enable_logging: bool = Field(default=False)


class Partitioner(ABC):
    """Abstract base class for message partitioners."""

    @abstractmethod
    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Calculate partition for a message."""

    @property
    @abstractmethod
    def strategy(self) -> PartitionStrategy:
        """Return the partitioning strategy."""


class RoundRobinPartitioner(Partitioner):
    """Distributes messages evenly across partitions."""

    def __init__(self) -> None:
        self._counter: int = 0

    @property
    def strategy(self) -> PartitionStrategy:
        return PartitionStrategy.ROUND_ROBIN

    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Assign partition in round-robin fashion."""
        partition = self._counter % num_partitions
        self._counter += 1
        return PartitionResult(partition=partition, strategy_used=self.strategy)


class HashKeyPartitioner(Partitioner):
    """Partitions based on key hash (Kafka default behavior)."""

    def __init__(self, algorithm: str = "murmur2") -> None:
        self._algorithm = algorithm

    @property
    def strategy(self) -> PartitionStrategy:
        return PartitionStrategy.HASH_KEY

    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Calculate partition using key hash."""
        if key is None:
            partition = 0
            key_hash = None
        else:
            key_hash = self._hash_key(key)
            partition = abs(key_hash) % num_partitions
        return PartitionResult(partition=partition, strategy_used=self.strategy, key_hash=key_hash)

    def _hash_key(self, key: bytes) -> int:
        """Hash key using configured algorithm."""
        if self._algorithm == "murmur2":
            return self._murmur2(key)
        return int(hashlib.md5(key).hexdigest(), 16)

    @staticmethod
    def _murmur2(data: bytes) -> int:
        """Murmur2 hash (Kafka's default partitioner algorithm)."""
        seed = 0x9747b28c
        m = 0x5bd1e995
        r = 24
        length = len(data)
        h = seed ^ length
        offset = 0
        while length >= 4:
            k = struct.unpack("<I", data[offset:offset + 4])[0]
            k = (k * m) & 0xFFFFFFFF
            k ^= (k >> r)
            k = (k * m) & 0xFFFFFFFF
            h = (h * m) & 0xFFFFFFFF
            h ^= k
            offset += 4
            length -= 4
        if length == 3:
            h ^= data[offset + 2] << 16
        if length >= 2:
            h ^= data[offset + 1] << 8
        if length >= 1:
            h ^= data[offset]
            h = (h * m) & 0xFFFFFFFF
        h ^= (h >> 13)
        h = (h * m) & 0xFFFFFFFF
        h ^= (h >> 15)
        return h if h < 0x80000000 else h - 0x100000000


class UserAffinityPartitioner(Partitioner):
    """Routes all messages for a user to the same partition."""

    def __init__(self) -> None:
        self._hash_partitioner = HashKeyPartitioner()

    @property
    def strategy(self) -> PartitionStrategy:
        return PartitionStrategy.USER_AFFINITY

    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Partition based on user ID for session affinity."""
        user_id = kwargs.get("user_id")
        if user_id:
            if isinstance(user_id, UUID):
                user_key = str(user_id).encode()
            else:
                user_key = str(user_id).encode()
            result = self._hash_partitioner.partition(user_key, num_partitions)
            return PartitionResult(
                partition=result.partition,
                strategy_used=self.strategy,
                key_hash=result.key_hash,
                metadata={"user_id": str(user_id)},
            )
        return self._hash_partitioner.partition(key, num_partitions)


class PriorityPartitioner(Partitioner):
    """Reserves specific partitions for high-priority messages."""

    def __init__(self, priority_partition_count: int = 2) -> None:
        self._priority_count = priority_partition_count
        self._hash_partitioner = HashKeyPartitioner()
        self._counter = 0

    @property
    def strategy(self) -> PartitionStrategy:
        return PartitionStrategy.PRIORITY_BASED

    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Route high-priority messages to dedicated partitions."""
        priority = kwargs.get("priority", "normal")
        is_critical = priority in ("critical", "high")
        if is_critical and self._priority_count > 0:
            priority_partition = self._counter % min(self._priority_count, num_partitions)
            self._counter += 1
            return PartitionResult(
                partition=priority_partition,
                strategy_used=self.strategy,
                metadata={"priority": priority, "dedicated": True},
            )
        if key:
            result = self._hash_partitioner.partition(key, num_partitions - self._priority_count)
            adjusted = result.partition + self._priority_count
            return PartitionResult(
                partition=min(adjusted, num_partitions - 1),
                strategy_used=self.strategy,
                key_hash=result.key_hash,
                metadata={"priority": priority, "dedicated": False},
            )
        partition = self._priority_count + (self._counter % (num_partitions - self._priority_count))
        self._counter += 1
        return PartitionResult(partition=partition, strategy_used=self.strategy,
                              metadata={"priority": priority, "dedicated": False})


class StickyPartitioner(Partitioner):
    """Batches messages to same partition until batch size reached."""

    def __init__(self, batch_size: int = 100) -> None:
        self._batch_size = batch_size
        self._current_partition: int = 0
        self._batch_count: int = 0

    @property
    def strategy(self) -> PartitionStrategy:
        return PartitionStrategy.STICKY

    def partition(self, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Stick to partition until batch is full."""
        if key is not None:
            hash_p = HashKeyPartitioner()
            return hash_p.partition(key, num_partitions, **kwargs)
        partition = self._current_partition
        self._batch_count += 1
        if self._batch_count >= self._batch_size:
            self._current_partition = (self._current_partition + 1) % num_partitions
            self._batch_count = 0
        return PartitionResult(
            partition=partition,
            strategy_used=self.strategy,
            metadata={"batch_count": self._batch_count},
        )


class PartitionerFactory:
    """Factory for creating partitioner instances."""

    _partitioners: dict[PartitionStrategy, type[Partitioner]] = {
        PartitionStrategy.ROUND_ROBIN: RoundRobinPartitioner,
        PartitionStrategy.HASH_KEY: HashKeyPartitioner,
        PartitionStrategy.USER_AFFINITY: UserAffinityPartitioner,
        PartitionStrategy.PRIORITY_BASED: PriorityPartitioner,
        PartitionStrategy.STICKY: StickyPartitioner,
    }

    @classmethod
    def create(cls, strategy: PartitionStrategy, **kwargs: Any) -> Partitioner:
        """Create partitioner instance for strategy."""
        partitioner_cls = cls._partitioners.get(strategy)
        if not partitioner_cls:
            raise ValueError(f"Unknown partition strategy: {strategy}")
        return partitioner_cls(**kwargs)

    @classmethod
    def register(cls, strategy: PartitionStrategy, partitioner_cls: type[Partitioner]) -> None:
        """Register custom partitioner."""
        cls._partitioners[strategy] = partitioner_cls


class TopicPartitionRouter:
    """Routes messages to partitions based on topic-specific strategies."""

    def __init__(self, config: PartitionerConfig | None = None) -> None:
        self._config = config or PartitionerConfig()
        self._topic_strategies: dict[str, PartitionStrategy] = {}
        self._partitioners: dict[PartitionStrategy, Partitioner] = {}
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Configure default strategies for Solace topics."""
        self._topic_strategies = {
            "solace.sessions": PartitionStrategy.USER_AFFINITY,
            "solace.safety": PartitionStrategy.PRIORITY_BASED,
            "solace.assessments": PartitionStrategy.USER_AFFINITY,
            "solace.therapy": PartitionStrategy.USER_AFFINITY,
            "solace.memory": PartitionStrategy.USER_AFFINITY,
            "solace.analytics": PartitionStrategy.ROUND_ROBIN,
            "solace.personality": PartitionStrategy.USER_AFFINITY,
        }

    def set_strategy(self, topic: str, strategy: PartitionStrategy) -> None:
        """Set partitioning strategy for a topic."""
        self._topic_strategies[topic] = strategy
        logger.debug("partition_strategy_set", topic=topic, strategy=strategy.value)

    def get_strategy(self, topic: str) -> PartitionStrategy:
        """Get partitioning strategy for a topic."""
        return self._topic_strategies.get(topic, self._config.default_strategy)

    def get_partitioner(self, strategy: PartitionStrategy) -> Partitioner:
        """Get or create partitioner for strategy."""
        if strategy not in self._partitioners:
            kwargs = {}
            if strategy == PartitionStrategy.STICKY:
                kwargs["batch_size"] = self._config.sticky_batch_size
            if strategy == PartitionStrategy.PRIORITY_BASED:
                kwargs["priority_partition_count"] = self._config.priority_partition_count
            self._partitioners[strategy] = PartitionerFactory.create(strategy, **kwargs)
        return self._partitioners[strategy]

    def route(self, topic: str, key: bytes | None, num_partitions: int, **kwargs: Any) -> PartitionResult:
        """Route message to partition based on topic strategy."""
        strategy = self.get_strategy(topic)
        partitioner = self.get_partitioner(strategy)
        result = partitioner.partition(key, num_partitions, **kwargs)
        if self._config.enable_logging:
            logger.debug("partition_routed", topic=topic, partition=result.partition,
                        strategy=result.strategy_used.value)
        return result


def create_partition_router(config: PartitionerConfig | None = None) -> TopicPartitionRouter:
    """Factory function to create partition router."""
    return TopicPartitionRouter(config)
