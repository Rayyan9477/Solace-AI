"""Unit tests for Kafka Partitioning Strategies module."""
from __future__ import annotations

import pytest
from uuid import uuid4
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from solace_infrastructure.kafka.partitioning import (
    PartitionStrategy,
    PartitionResult,
    PartitionerConfig,
    RoundRobinPartitioner,
    HashKeyPartitioner,
    UserAffinityPartitioner,
    PriorityPartitioner,
    StickyPartitioner,
    PartitionerFactory,
    TopicPartitionRouter,
    create_partition_router,
)


class TestPartitionStrategy:
    """Tests for PartitionStrategy enum."""

    def test_strategy_values(self) -> None:
        assert PartitionStrategy.ROUND_ROBIN.value == "round_robin"
        assert PartitionStrategy.HASH_KEY.value == "hash_key"
        assert PartitionStrategy.USER_AFFINITY.value == "user_affinity"
        assert PartitionStrategy.PRIORITY_BASED.value == "priority_based"
        assert PartitionStrategy.STICKY.value == "sticky"


class TestPartitionResult:
    """Tests for PartitionResult dataclass."""

    def test_result_creation(self) -> None:
        result = PartitionResult(
            partition=2,
            strategy_used=PartitionStrategy.HASH_KEY,
            key_hash=12345,
        )
        assert result.partition == 2
        assert result.strategy_used == PartitionStrategy.HASH_KEY
        assert result.key_hash == 12345

    def test_result_with_metadata(self) -> None:
        result = PartitionResult(
            partition=0,
            strategy_used=PartitionStrategy.PRIORITY_BASED,
            metadata={"priority": "critical"},
        )
        assert result.metadata["priority"] == "critical"


class TestPartitionerConfig:
    """Tests for PartitionerConfig."""

    def test_default_config(self) -> None:
        config = PartitionerConfig()
        assert config.default_strategy == PartitionStrategy.HASH_KEY
        assert config.sticky_batch_size == 100
        assert config.priority_partition_count == 2

    def test_custom_config(self) -> None:
        config = PartitionerConfig(
            default_strategy=PartitionStrategy.ROUND_ROBIN,
            sticky_batch_size=50,
            enable_logging=True,
        )
        assert config.default_strategy == PartitionStrategy.ROUND_ROBIN
        assert config.sticky_batch_size == 50
        assert config.enable_logging is True


class TestRoundRobinPartitioner:
    """Tests for RoundRobinPartitioner."""

    @pytest.fixture
    def partitioner(self) -> RoundRobinPartitioner:
        return RoundRobinPartitioner()

    def test_strategy(self, partitioner: RoundRobinPartitioner) -> None:
        assert partitioner.strategy == PartitionStrategy.ROUND_ROBIN

    def test_distributes_evenly(self, partitioner: RoundRobinPartitioner) -> None:
        num_partitions = 4
        counts = {i: 0 for i in range(num_partitions)}
        for _ in range(100):
            result = partitioner.partition(None, num_partitions)
            counts[result.partition] += 1
        # Each partition should get 25 messages
        for count in counts.values():
            assert count == 25

    def test_ignores_key(self, partitioner: RoundRobinPartitioner) -> None:
        result1 = partitioner.partition(b"key1", 4)
        result2 = partitioner.partition(b"key2", 4)
        # Should still increment sequentially regardless of key
        assert result2.partition == (result1.partition + 1) % 4


class TestHashKeyPartitioner:
    """Tests for HashKeyPartitioner."""

    @pytest.fixture
    def partitioner(self) -> HashKeyPartitioner:
        return HashKeyPartitioner()

    def test_strategy(self, partitioner: HashKeyPartitioner) -> None:
        assert partitioner.strategy == PartitionStrategy.HASH_KEY

    def test_consistent_hashing(self, partitioner: HashKeyPartitioner) -> None:
        key = b"consistent-key"
        num_partitions = 8
        results = [partitioner.partition(key, num_partitions) for _ in range(10)]
        # All should go to same partition
        assert all(r.partition == results[0].partition for r in results)

    def test_different_keys_distribute(self, partitioner: HashKeyPartitioner) -> None:
        num_partitions = 4
        partitions = set()
        for i in range(100):
            result = partitioner.partition(f"key-{i}".encode(), num_partitions)
            partitions.add(result.partition)
        # Should use multiple partitions
        assert len(partitions) > 1

    def test_null_key_goes_to_partition_zero(self, partitioner: HashKeyPartitioner) -> None:
        result = partitioner.partition(None, 4)
        assert result.partition == 0
        assert result.key_hash is None

    def test_murmur2_algorithm(self) -> None:
        partitioner = HashKeyPartitioner(algorithm="murmur2")
        result = partitioner.partition(b"test", 4)
        assert result.key_hash is not None


class TestUserAffinityPartitioner:
    """Tests for UserAffinityPartitioner."""

    @pytest.fixture
    def partitioner(self) -> UserAffinityPartitioner:
        return UserAffinityPartitioner()

    def test_strategy(self, partitioner: UserAffinityPartitioner) -> None:
        assert partitioner.strategy == PartitionStrategy.USER_AFFINITY

    def test_same_user_same_partition(self, partitioner: UserAffinityPartitioner) -> None:
        user_id = uuid4()
        num_partitions = 8
        results = [
            partitioner.partition(None, num_partitions, user_id=user_id)
            for _ in range(10)
        ]
        assert all(r.partition == results[0].partition for r in results)

    def test_user_id_in_metadata(self, partitioner: UserAffinityPartitioner) -> None:
        user_id = uuid4()
        result = partitioner.partition(None, 4, user_id=user_id)
        assert result.metadata is not None
        assert result.metadata["user_id"] == str(user_id)

    def test_fallback_to_key_hash(self, partitioner: UserAffinityPartitioner) -> None:
        # Without user_id, should use key
        result = partitioner.partition(b"some-key", 4)
        assert result.key_hash is not None

    def test_string_user_id(self, partitioner: UserAffinityPartitioner) -> None:
        user_id = "user-123"
        result = partitioner.partition(None, 4, user_id=user_id)
        assert result.metadata["user_id"] == user_id


class TestPriorityPartitioner:
    """Tests for PriorityPartitioner."""

    @pytest.fixture
    def partitioner(self) -> PriorityPartitioner:
        return PriorityPartitioner(priority_partition_count=2)

    def test_strategy(self, partitioner: PriorityPartitioner) -> None:
        assert partitioner.strategy == PartitionStrategy.PRIORITY_BASED

    def test_critical_uses_dedicated_partitions(self, partitioner: PriorityPartitioner) -> None:
        num_partitions = 8
        partitions = set()
        for _ in range(10):
            result = partitioner.partition(None, num_partitions, priority="critical")
            partitions.add(result.partition)
        # Should only use first 2 partitions (priority count)
        assert max(partitions) < 2

    def test_high_uses_dedicated_partitions(self, partitioner: PriorityPartitioner) -> None:
        result = partitioner.partition(None, 8, priority="high")
        assert result.partition < 2
        assert result.metadata["dedicated"] is True

    def test_normal_avoids_priority_partitions(self, partitioner: PriorityPartitioner) -> None:
        result = partitioner.partition(b"key", 8, priority="normal")
        assert result.partition >= 2
        assert result.metadata["dedicated"] is False

    def test_metadata_includes_priority(self, partitioner: PriorityPartitioner) -> None:
        result = partitioner.partition(None, 4, priority="critical")
        assert result.metadata["priority"] == "critical"


class TestStickyPartitioner:
    """Tests for StickyPartitioner."""

    @pytest.fixture
    def partitioner(self) -> StickyPartitioner:
        return StickyPartitioner(batch_size=5)

    def test_strategy(self, partitioner: StickyPartitioner) -> None:
        assert partitioner.strategy == PartitionStrategy.STICKY

    def test_sticks_to_partition(self, partitioner: StickyPartitioner) -> None:
        num_partitions = 4
        results = [partitioner.partition(None, num_partitions) for _ in range(4)]
        # All should go to same partition (batch size is 5)
        assert all(r.partition == results[0].partition for r in results)

    def test_switches_after_batch(self, partitioner: StickyPartitioner) -> None:
        num_partitions = 4
        first_partition = partitioner.partition(None, num_partitions).partition
        # Send batch_size messages
        for _ in range(4):
            partitioner.partition(None, num_partitions)
        # 5th message triggers switch
        new_result = partitioner.partition(None, num_partitions)
        assert new_result.partition != first_partition

    def test_keyed_messages_use_hash(self, partitioner: StickyPartitioner) -> None:
        result = partitioner.partition(b"specific-key", 4)
        # Should use hash partitioner for keyed messages
        assert result.strategy_used == PartitionStrategy.HASH_KEY


class TestPartitionerFactory:
    """Tests for PartitionerFactory."""

    def test_create_round_robin(self) -> None:
        partitioner = PartitionerFactory.create(PartitionStrategy.ROUND_ROBIN)
        assert isinstance(partitioner, RoundRobinPartitioner)

    def test_create_hash_key(self) -> None:
        partitioner = PartitionerFactory.create(PartitionStrategy.HASH_KEY)
        assert isinstance(partitioner, HashKeyPartitioner)

    def test_create_user_affinity(self) -> None:
        partitioner = PartitionerFactory.create(PartitionStrategy.USER_AFFINITY)
        assert isinstance(partitioner, UserAffinityPartitioner)

    def test_create_priority(self) -> None:
        partitioner = PartitionerFactory.create(
            PartitionStrategy.PRIORITY_BASED,
            priority_partition_count=3,
        )
        assert isinstance(partitioner, PriorityPartitioner)

    def test_create_sticky(self) -> None:
        partitioner = PartitionerFactory.create(
            PartitionStrategy.STICKY,
            batch_size=50,
        )
        assert isinstance(partitioner, StickyPartitioner)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown partition strategy"):
            PartitionerFactory.create(PartitionStrategy.RANDOM)


class TestTopicPartitionRouter:
    """Tests for TopicPartitionRouter."""

    @pytest.fixture
    def router(self) -> TopicPartitionRouter:
        return TopicPartitionRouter()

    def test_default_strategies(self, router: TopicPartitionRouter) -> None:
        assert router.get_strategy("solace.sessions") == PartitionStrategy.USER_AFFINITY
        assert router.get_strategy("solace.safety") == PartitionStrategy.PRIORITY_BASED
        assert router.get_strategy("solace.analytics") == PartitionStrategy.ROUND_ROBIN

    def test_set_strategy(self, router: TopicPartitionRouter) -> None:
        router.set_strategy("custom.topic", PartitionStrategy.STICKY)
        assert router.get_strategy("custom.topic") == PartitionStrategy.STICKY

    def test_unknown_topic_uses_default(self, router: TopicPartitionRouter) -> None:
        config = PartitionerConfig(default_strategy=PartitionStrategy.HASH_KEY)
        router = TopicPartitionRouter(config)
        assert router.get_strategy("unknown.topic") == PartitionStrategy.HASH_KEY

    def test_route_sessions_user_affinity(self, router: TopicPartitionRouter) -> None:
        user_id = uuid4()
        result = router.route("solace.sessions", None, 4, user_id=user_id)
        assert result.strategy_used == PartitionStrategy.USER_AFFINITY
        assert result.metadata is not None

    def test_route_safety_priority(self, router: TopicPartitionRouter) -> None:
        result = router.route("solace.safety", None, 4, priority="critical")
        assert result.strategy_used == PartitionStrategy.PRIORITY_BASED

    def test_route_analytics_round_robin(self, router: TopicPartitionRouter) -> None:
        router = TopicPartitionRouter()
        results = [router.route("solace.analytics", None, 4) for _ in range(4)]
        partitions = [r.partition for r in results]
        # Round robin should distribute across partitions
        assert len(set(partitions)) == 4

    def test_partitioner_caching(self, router: TopicPartitionRouter) -> None:
        # First call creates partitioner
        p1 = router.get_partitioner(PartitionStrategy.HASH_KEY)
        # Second call should return same instance
        p2 = router.get_partitioner(PartitionStrategy.HASH_KEY)
        assert p1 is p2


class TestFactoryFunction:
    """Tests for factory functions."""

    def test_create_partition_router(self) -> None:
        router = create_partition_router()
        assert isinstance(router, TopicPartitionRouter)

    def test_create_partition_router_with_config(self) -> None:
        config = PartitionerConfig(
            default_strategy=PartitionStrategy.ROUND_ROBIN,
            enable_logging=True,
        )
        router = create_partition_router(config)
        assert router.get_strategy("unknown.topic") == PartitionStrategy.ROUND_ROBIN
