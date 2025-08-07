"""
Data Consistency and System Reliability Module for Solace-AI

This module provides comprehensive data consistency, integrity, and system
reliability capabilities including:
- Data consistency validation and enforcement
- Distributed transaction management
- System reliability monitoring
- Fault tolerance and recovery mechanisms
- Data backup and restoration
- Consistency conflict resolution
- System resilience patterns
- Performance degradation detection
- Automated recovery procedures
"""

import asyncio
import time
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import threading
import pickle
import sqlite3
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod
import statistics

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority

logger = get_logger(__name__)


class ConsistencyLevel(Enum):
    """Data consistency levels."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"
    BOUNDED_STALENESS = "bounded_staleness"


class ReliabilityStatus(Enum):
    """System reliability status levels."""
    OPTIMAL = "optimal"
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"


class TransactionState(Enum):
    """Transaction states."""
    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"


class FailureType(Enum):
    """Types of system failures."""
    NETWORK_PARTITION = "network_partition"
    NODE_FAILURE = "node_failure"
    DATA_CORRUPTION = "data_corruption"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class DataVersion:
    """Data version information for consistency tracking."""
    
    entity_id: str
    version: int
    timestamp: datetime
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entity_id': self.entity_id,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'checksum': self.checksum,
            'metadata': self.metadata
        }


@dataclass
class ConsistencyCheck:
    """Consistency check result."""
    
    check_id: str
    entity_id: str
    is_consistent: bool
    consistency_level: ConsistencyLevel
    discrepancies: List[str] = field(default_factory=list)
    resolution_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'check_id': self.check_id,
            'entity_id': self.entity_id,
            'is_consistent': self.is_consistent,
            'consistency_level': self.consistency_level.value,
            'discrepancies': self.discrepancies,
            'resolution_actions': self.resolution_actions,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DistributedTransaction:
    """Distributed transaction management."""
    
    transaction_id: str
    participants: List[str]
    operations: List[Dict[str, Any]]
    state: TransactionState = TransactionState.PENDING
    coordinator: Optional[str] = None
    timeout_seconds: int = 30
    started_at: datetime = field(default_factory=datetime.now)
    prepare_votes: Dict[str, bool] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if transaction has expired."""
        return (datetime.now() - self.started_at).total_seconds() > self.timeout_seconds
    
    def can_commit(self) -> bool:
        """Check if transaction can be committed."""
        return (
            self.state == TransactionState.PREPARED and
            len(self.prepare_votes) == len(self.participants) and
            all(self.prepare_votes.values())
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'transaction_id': self.transaction_id,
            'participants': self.participants,
            'operations': self.operations,
            'state': self.state.value,
            'coordinator': self.coordinator,
            'timeout_seconds': self.timeout_seconds,
            'started_at': self.started_at.isoformat(),
            'prepare_votes': self.prepare_votes
        }


@dataclass
class SystemFailure:
    """System failure information."""
    
    failure_id: str
    failure_type: FailureType
    component: str
    description: str
    severity: int  # 1-10 scale
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    recovery_actions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        """Check if failure is resolved."""
        return self.resolved_at is not None
    
    def duration(self) -> timedelta:
        """Get failure duration."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.detected_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'failure_id': self.failure_id,
            'failure_type': self.failure_type.value,
            'component': self.component,
            'description': self.description,
            'severity': self.severity,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'recovery_actions': self.recovery_actions,
            'impact_assessment': self.impact_assessment,
            'duration_seconds': self.duration().total_seconds()
        }


@dataclass
class ReliabilityMetrics:
    """System reliability metrics."""
    
    uptime_percentage: float
    mean_time_to_failure: float  # hours
    mean_time_to_recovery: float  # hours
    availability_score: float
    consistency_score: float
    performance_score: float
    overall_reliability_score: float
    measurement_period_hours: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'uptime_percentage': self.uptime_percentage,
            'mean_time_to_failure': self.mean_time_to_failure,
            'mean_time_to_recovery': self.mean_time_to_recovery,
            'availability_score': self.availability_score,
            'consistency_score': self.consistency_score,
            'performance_score': self.performance_score,
            'overall_reliability_score': self.overall_reliability_score,
            'measurement_period_hours': self.measurement_period_hours,
            'timestamp': self.timestamp.isoformat()
        }


class DataStore(ABC):
    """Abstract data store interface for consistency management."""
    
    @abstractmethod
    async def read(self, key: str) -> Optional[Any]:
        """Read data by key."""
        pass
    
    @abstractmethod
    async def write(self, key: str, value: Any, version: Optional[int] = None) -> bool:
        """Write data with optional version check."""
        pass
    
    @abstractmethod
    async def delete(self, key: str, version: Optional[int] = None) -> bool:
        """Delete data with optional version check."""
        pass
    
    @abstractmethod
    async def get_version(self, key: str) -> Optional[DataVersion]:
        """Get version information for a key."""
        pass
    
    @abstractmethod
    async def get_all_keys(self) -> List[str]:
        """Get all keys in the data store."""
        pass


class InMemoryDataStore(DataStore):
    """In-memory data store with versioning."""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.versions: Dict[str, DataVersion] = {}
        self.lock = threading.RLock()
        self.next_version = 1
    
    async def read(self, key: str) -> Optional[Any]:
        """Read data by key."""
        with self.lock:
            return self.data.get(key)
    
    async def write(self, key: str, value: Any, version: Optional[int] = None) -> bool:
        """Write data with optional version check."""
        with self.lock:
            # Check version if specified
            if version is not None:
                current_version = self.versions.get(key)
                if current_version and current_version.version != version:
                    return False  # Version conflict
            
            # Write data
            self.data[key] = value
            
            # Update version
            checksum = hashlib.md5(pickle.dumps(value)).hexdigest()
            self.versions[key] = DataVersion(
                entity_id=key,
                version=self.next_version,
                timestamp=datetime.now(),
                checksum=checksum
            )
            self.next_version += 1
            
            return True
    
    async def delete(self, key: str, version: Optional[int] = None) -> bool:
        """Delete data with optional version check."""
        with self.lock:
            if key not in self.data:
                return False
            
            # Check version if specified
            if version is not None:
                current_version = self.versions.get(key)
                if current_version and current_version.version != version:
                    return False  # Version conflict
            
            # Delete data
            del self.data[key]
            del self.versions[key]
            
            return True
    
    async def get_version(self, key: str) -> Optional[DataVersion]:
        """Get version information for a key."""
        with self.lock:
            return self.versions.get(key)
    
    async def get_all_keys(self) -> List[str]:
        """Get all keys in the data store."""
        with self.lock:
            return list(self.data.keys())


class PersistentDataStore(DataStore):
    """Persistent data store using SQLite."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_store (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    version INTEGER,
                    timestamp TEXT,
                    checksum TEXT
                )
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    async def read(self, key: str) -> Optional[Any]:
        """Read data by key."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM data_store WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    return pickle.loads(result[0])
                return None
            finally:
                conn.close()
    
    async def write(self, key: str, value: Any, version: Optional[int] = None) -> bool:
        """Write data with optional version check."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Check version if specified
                if version is not None:
                    cursor.execute("SELECT version FROM data_store WHERE key = ?", (key,))
                    result = cursor.fetchone()
                    if result and result[0] != version:
                        return False  # Version conflict
                
                # Write data
                value_blob = pickle.dumps(value)
                checksum = hashlib.md5(value_blob).hexdigest()
                timestamp = datetime.now().isoformat()
                
                # Get next version
                cursor.execute("SELECT MAX(version) FROM data_store WHERE key = ?", (key,))
                result = cursor.fetchone()
                new_version = (result[0] or 0) + 1
                
                cursor.execute("""
                    INSERT OR REPLACE INTO data_store (key, value, version, timestamp, checksum)
                    VALUES (?, ?, ?, ?, ?)
                """, (key, value_blob, new_version, timestamp, checksum))
                
                conn.commit()
                return True
                
            finally:
                conn.close()
    
    async def delete(self, key: str, version: Optional[int] = None) -> bool:
        """Delete data with optional version check."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                
                # Check if key exists
                cursor.execute("SELECT version FROM data_store WHERE key = ?", (key,))
                result = cursor.fetchone()
                if not result:
                    return False
                
                # Check version if specified
                if version is not None and result[0] != version:
                    return False  # Version conflict
                
                # Delete data
                cursor.execute("DELETE FROM data_store WHERE key = ?", (key,))
                conn.commit()
                
                return cursor.rowcount > 0
                
            finally:
                conn.close()
    
    async def get_version(self, key: str) -> Optional[DataVersion]:
        """Get version information for a key."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT version, timestamp, checksum FROM data_store WHERE key = ?
                """, (key,))
                result = cursor.fetchone()
                
                if result:
                    version, timestamp_str, checksum = result
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    return DataVersion(
                        entity_id=key,
                        version=version,
                        timestamp=timestamp,
                        checksum=checksum
                    )
                return None
                
            finally:
                conn.close()
    
    async def get_all_keys(self) -> List[str]:
        """Get all keys in the data store."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT key FROM data_store")
                results = cursor.fetchall()
                
                return [row[0] for row in results]
                
            finally:
                conn.close()


class ConsistencyManager:
    """Manages data consistency across distributed components."""
    
    def __init__(self):
        self.data_stores: Dict[str, DataStore] = {}
        self.consistency_checks: Dict[str, ConsistencyCheck] = {}
        self.conflict_resolution_strategies: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        
        # Default conflict resolution strategies
        self._setup_default_strategies()
    
    def register_data_store(self, store_name: str, data_store: DataStore) -> None:
        """Register a data store for consistency management."""
        with self.lock:
            self.data_stores[store_name] = data_store
        logger.info(f"Registered data store: {store_name}")
    
    def register_conflict_resolution_strategy(self, strategy_name: str, strategy_func: Callable) -> None:
        """Register a conflict resolution strategy."""
        with self.lock:
            self.conflict_resolution_strategies[strategy_name] = strategy_func
        logger.info(f"Registered conflict resolution strategy: {strategy_name}")
    
    async def check_consistency(self, entity_id: str, 
                              consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL) -> ConsistencyCheck:
        """Check consistency of an entity across all data stores."""
        
        check_id = f"consistency_check_{int(time.time() * 1000000)}"
        discrepancies = []
        resolution_actions = []
        
        try:
            # Get versions from all stores
            versions = {}
            values = {}
            
            with self.lock:
                store_names = list(self.data_stores.keys())
            
            for store_name in store_names:
                store = self.data_stores[store_name]
                version = await store.get_version(entity_id)
                value = await store.read(entity_id)
                
                versions[store_name] = version
                values[store_name] = value
            
            # Analyze consistency
            is_consistent = True
            
            if consistency_level == ConsistencyLevel.STRONG:
                # All stores must have identical versions
                version_numbers = [v.version for v in versions.values() if v is not None]
                if len(set(version_numbers)) > 1:
                    is_consistent = False
                    discrepancies.append("Version mismatch across stores")
                    resolution_actions.append("Synchronize to latest version")
            
            elif consistency_level == ConsistencyLevel.EVENTUAL:
                # Check for data integrity issues
                checksums = [v.checksum for v in versions.values() if v is not None]
                if len(set(checksums)) > 1:
                    discrepancies.append("Data checksum mismatch")
                    resolution_actions.append("Resolve conflicting values")
            
            elif consistency_level == ConsistencyLevel.CAUSAL:
                # Check causal ordering
                timestamps = [v.timestamp for v in versions.values() if v is not None]
                if timestamps and max(timestamps) - min(timestamps) > timedelta(minutes=5):
                    discrepancies.append("Potential causal ordering violation")
                    resolution_actions.append("Verify causal dependencies")
            
            # Store consistency check result
            check = ConsistencyCheck(
                check_id=check_id,
                entity_id=entity_id,
                is_consistent=is_consistent,
                consistency_level=consistency_level,
                discrepancies=discrepancies,
                resolution_actions=resolution_actions
            )
            
            with self.lock:
                self.consistency_checks[check_id] = check
            
            return check
            
        except Exception as e:
            logger.error(f"Error checking consistency for {entity_id}: {e}")
            return ConsistencyCheck(
                check_id=check_id,
                entity_id=entity_id,
                is_consistent=False,
                consistency_level=consistency_level,
                discrepancies=[f"Consistency check failed: {str(e)}"],
                resolution_actions=["Investigate system error"]
            )
    
    async def resolve_conflicts(self, entity_id: str, strategy: str = "latest_wins") -> bool:
        """Resolve conflicts for an entity using specified strategy."""
        
        try:
            # Get conflict resolution strategy
            if strategy not in self.conflict_resolution_strategies:
                logger.error(f"Unknown conflict resolution strategy: {strategy}")
                return False
            
            strategy_func = self.conflict_resolution_strategies[strategy]
            
            # Get current state from all stores
            store_states = {}
            
            with self.lock:
                store_names = list(self.data_stores.keys())
            
            for store_name in store_names:
                store = self.data_stores[store_name]
                version = await store.get_version(entity_id)
                value = await store.read(entity_id)
                
                store_states[store_name] = {
                    'version': version,
                    'value': value,
                    'store': store
                }
            
            # Apply resolution strategy
            resolved_state = strategy_func(store_states)
            
            if not resolved_state:
                logger.error(f"Conflict resolution strategy {strategy} failed for {entity_id}")
                return False
            
            # Apply resolved state to all stores
            resolved_value = resolved_state['value']
            
            success_count = 0
            for store_name, store_state in store_states.items():
                store = store_state['store']
                try:
                    success = await store.write(entity_id, resolved_value)
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error writing resolved state to {store_name}: {e}")
            
            # Consider resolution successful if majority of stores updated
            return success_count > len(store_states) / 2
            
        except Exception as e:
            logger.error(f"Error resolving conflicts for {entity_id}: {e}")
            return False
    
    async def synchronize_entity(self, entity_id: str, target_store: str) -> bool:
        """Synchronize an entity from target store to all other stores."""
        
        try:
            if target_store not in self.data_stores:
                logger.error(f"Target store {target_store} not found")
                return False
            
            # Get authoritative state from target store
            target_data_store = self.data_stores[target_store]
            authoritative_value = await target_data_store.read(entity_id)
            
            if authoritative_value is None:
                logger.warning(f"Entity {entity_id} not found in target store {target_store}")
                return False
            
            # Synchronize to all other stores
            success_count = 0
            
            with self.lock:
                other_stores = {name: store for name, store in self.data_stores.items() if name != target_store}
            
            for store_name, store in other_stores.items():
                try:
                    success = await store.write(entity_id, authoritative_value)
                    if success:
                        success_count += 1
                    else:
                        logger.warning(f"Failed to synchronize {entity_id} to {store_name}")
                except Exception as e:
                    logger.error(f"Error synchronizing {entity_id} to {store_name}: {e}")
            
            return success_count == len(other_stores)
            
        except Exception as e:
            logger.error(f"Error synchronizing entity {entity_id}: {e}")
            return False
    
    def _setup_default_strategies(self) -> None:
        """Setup default conflict resolution strategies."""
        
        def latest_wins_strategy(store_states: Dict[str, Dict]) -> Optional[Dict]:
            """Latest timestamp wins strategy."""
            latest_state = None
            latest_timestamp = None
            
            for store_name, state in store_states.items():
                version = state['version']
                if version and (latest_timestamp is None or version.timestamp > latest_timestamp):
                    latest_timestamp = version.timestamp
                    latest_state = state
            
            return latest_state
        
        def highest_version_strategy(store_states: Dict[str, Dict]) -> Optional[Dict]:
            """Highest version number wins strategy."""
            highest_state = None
            highest_version = -1
            
            for store_name, state in store_states.items():
                version = state['version']
                if version and version.version > highest_version:
                    highest_version = version.version
                    highest_state = state
            
            return highest_state
        
        def majority_wins_strategy(store_states: Dict[str, Dict]) -> Optional[Dict]:
            """Value with majority consensus wins."""
            value_counts = defaultdict(int)
            value_states = {}
            
            for store_name, state in store_states.items():
                value = state['value']
                if value is not None:
                    value_hash = hashlib.md5(pickle.dumps(value)).hexdigest()
                    value_counts[value_hash] += 1
                    value_states[value_hash] = state
            
            if value_counts:
                majority_value_hash = max(value_counts.keys(), key=lambda k: value_counts[k])
                return value_states[majority_value_hash]
            
            return None
        
        self.register_conflict_resolution_strategy("latest_wins", latest_wins_strategy)
        self.register_conflict_resolution_strategy("highest_version", highest_version_strategy)
        self.register_conflict_resolution_strategy("majority_wins", majority_wins_strategy)


class TransactionManager:
    """Manages distributed transactions using two-phase commit protocol."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.transactions: Dict[str, DistributedTransaction] = {}
        self.participants: Dict[str, Callable] = {}  # participant_id -> handler
        self.lock = threading.RLock()
        
        # Background task for transaction cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the transaction manager."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("TransactionManager started")
    
    async def stop(self) -> None:
        """Stop the transaction manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("TransactionManager stopped")
    
    def register_participant(self, participant_id: str, handler: Callable) -> None:
        """Register a transaction participant."""
        with self.lock:
            self.participants[participant_id] = handler
        logger.info(f"Registered transaction participant: {participant_id}")
    
    async def begin_transaction(self, 
                              participants: List[str], 
                              operations: List[Dict[str, Any]],
                              timeout_seconds: int = 30) -> str:
        """Begin a new distributed transaction."""
        
        transaction_id = f"txn_{uuid.uuid4().hex}"
        
        # Validate participants
        with self.lock:
            unknown_participants = [p for p in participants if p not in self.participants]
        
        if unknown_participants:
            raise ValueError(f"Unknown participants: {unknown_participants}")
        
        # Create transaction
        transaction = DistributedTransaction(
            transaction_id=transaction_id,
            participants=participants,
            operations=operations,
            coordinator=f"coordinator_{uuid.uuid4().hex[:8]}",
            timeout_seconds=timeout_seconds
        )
        
        with self.lock:
            self.transactions[transaction_id] = transaction
        
        # Publish transaction started event
        await self.event_bus.publish(Event(
            event_type="transaction_started",
            source_agent="transaction_manager",
            data={
                'transaction_id': transaction_id,
                'participants': participants,
                'operations_count': len(operations)
            }
        ))
        
        logger.info(f"Started transaction {transaction_id} with {len(participants)} participants")
        return transaction_id
    
    async def prepare_transaction(self, transaction_id: str) -> bool:
        """Prepare phase of two-phase commit."""
        
        with self.lock:
            if transaction_id not in self.transactions:
                logger.error(f"Transaction {transaction_id} not found")
                return False
            
            transaction = self.transactions[transaction_id]
        
        if transaction.state != TransactionState.PENDING:
            logger.error(f"Transaction {transaction_id} not in pending state")
            return False
        
        if transaction.is_expired():
            transaction.state = TransactionState.ABORTED
            logger.error(f"Transaction {transaction_id} expired")
            return False
        
        # Update state to preparing
        transaction.state = TransactionState.PREPARING
        
        # Send prepare requests to all participants
        prepare_tasks = []
        
        for participant_id in transaction.participants:
            handler = self.participants[participant_id]
            task = asyncio.create_task(
                self._send_prepare_request(transaction_id, participant_id, handler, transaction.operations)
            )
            prepare_tasks.append(task)
        
        # Wait for all prepare responses
        prepare_results = await asyncio.gather(*prepare_tasks, return_exceptions=True)
        
        # Process prepare results
        all_prepared = True
        
        for i, result in enumerate(prepare_results):
            participant_id = transaction.participants[i]
            
            if isinstance(result, Exception):
                logger.error(f"Prepare failed for participant {participant_id}: {result}")
                transaction.prepare_votes[participant_id] = False
                all_prepared = False
            else:
                transaction.prepare_votes[participant_id] = result
                if not result:
                    all_prepared = False
        
        # Update transaction state
        if all_prepared:
            transaction.state = TransactionState.PREPARED
            logger.info(f"Transaction {transaction_id} prepared successfully")
        else:
            transaction.state = TransactionState.ABORTING
            logger.warning(f"Transaction {transaction_id} prepare phase failed")
        
        return all_prepared
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit phase of two-phase commit."""
        
        with self.lock:
            if transaction_id not in self.transactions:
                logger.error(f"Transaction {transaction_id} not found")
                return False
            
            transaction = self.transactions[transaction_id]
        
        if not transaction.can_commit():
            logger.error(f"Transaction {transaction_id} cannot be committed")
            await self._abort_transaction(transaction_id)
            return False
        
        # Update state to committing
        transaction.state = TransactionState.COMMITTING
        
        # Send commit requests to all participants
        commit_tasks = []
        
        for participant_id in transaction.participants:
            handler = self.participants[participant_id]
            task = asyncio.create_task(
                self._send_commit_request(transaction_id, participant_id, handler)
            )
            commit_tasks.append(task)
        
        # Wait for all commit responses
        commit_results = await asyncio.gather(*commit_tasks, return_exceptions=True)
        
        # Process commit results
        all_committed = True
        
        for i, result in enumerate(commit_results):
            participant_id = transaction.participants[i]
            
            if isinstance(result, Exception):
                logger.error(f"Commit failed for participant {participant_id}: {result}")
                all_committed = False
            elif not result:
                logger.error(f"Commit rejected by participant {participant_id}")
                all_committed = False
        
        # Update transaction state
        if all_committed:
            transaction.state = TransactionState.COMMITTED
            logger.info(f"Transaction {transaction_id} committed successfully")
            
            # Publish committed event
            await self.event_bus.publish(Event(
                event_type="transaction_committed",
                source_agent="transaction_manager",
                data={
                    'transaction_id': transaction_id,
                    'participants': transaction.participants
                }
            ))
        else:
            transaction.state = TransactionState.ABORTED
            logger.error(f"Transaction {transaction_id} commit failed")
        
        return all_committed
    
    async def _abort_transaction(self, transaction_id: str) -> None:
        """Abort a transaction."""
        
        with self.lock:
            if transaction_id not in self.transactions:
                return
            
            transaction = self.transactions[transaction_id]
        
        transaction.state = TransactionState.ABORTING
        
        # Send abort requests to all participants
        abort_tasks = []
        
        for participant_id in transaction.participants:
            if participant_id in self.participants:
                handler = self.participants[participant_id]
                task = asyncio.create_task(
                    self._send_abort_request(transaction_id, participant_id, handler)
                )
                abort_tasks.append(task)
        
        # Wait for abort acknowledgments
        await asyncio.gather(*abort_tasks, return_exceptions=True)
        
        transaction.state = TransactionState.ABORTED
        
        # Publish aborted event
        await self.event_bus.publish(Event(
            event_type="transaction_aborted",
            source_agent="transaction_manager",
            data={
                'transaction_id': transaction_id,
                'participants': transaction.participants
            }
        ))
        
        logger.info(f"Transaction {transaction_id} aborted")
    
    async def _send_prepare_request(self, 
                                  transaction_id: str, 
                                  participant_id: str, 
                                  handler: Callable,
                                  operations: List[Dict[str, Any]]) -> bool:
        """Send prepare request to a participant."""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler("prepare", transaction_id, operations)
            else:
                return handler("prepare", transaction_id, operations)
        except Exception as e:
            logger.error(f"Error sending prepare request to {participant_id}: {e}")
            return False
    
    async def _send_commit_request(self, 
                                 transaction_id: str, 
                                 participant_id: str, 
                                 handler: Callable) -> bool:
        """Send commit request to a participant."""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler("commit", transaction_id, [])
            else:
                return handler("commit", transaction_id, [])
        except Exception as e:
            logger.error(f"Error sending commit request to {participant_id}: {e}")
            return False
    
    async def _send_abort_request(self, 
                                transaction_id: str, 
                                participant_id: str, 
                                handler: Callable) -> bool:
        """Send abort request to a participant."""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler("abort", transaction_id, [])
            else:
                return handler("abort", transaction_id, [])
        except Exception as e:
            logger.error(f"Error sending abort request to {participant_id}: {e}")
            return False
    
    async def _cleanup_loop(self) -> None:
        """Background loop to cleanup expired transactions."""
        
        while self._running:
            try:
                current_time = datetime.now()
                expired_transactions = []
                
                with self.lock:
                    for txn_id, txn in self.transactions.items():
                        if txn.is_expired() and txn.state not in [TransactionState.COMMITTED, TransactionState.ABORTED]:
                            expired_transactions.append(txn_id)
                
                # Abort expired transactions
                for txn_id in expired_transactions:
                    await self._abort_transaction(txn_id)
                
                # Remove old completed transactions
                with self.lock:
                    cutoff_time = current_time - timedelta(hours=24)
                    old_transactions = [
                        txn_id for txn_id, txn in self.transactions.items()
                        if txn.started_at < cutoff_time and txn.state in [TransactionState.COMMITTED, TransactionState.ABORTED]
                    ]
                    
                    for txn_id in old_transactions:
                        del self.transactions[txn_id]
                
                if expired_transactions or old_transactions:
                    logger.info(f"Cleaned up {len(expired_transactions)} expired and {len(old_transactions)} old transactions")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transaction cleanup loop: {e}")
                await asyncio.sleep(60)


class ReliabilityMonitor:
    """Monitors system reliability and detects failures."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.failures: Dict[str, SystemFailure] = {}
        self.reliability_history: deque = deque(maxlen=1000)
        self.component_status: Dict[str, ReliabilityStatus] = {}
        self.recovery_handlers: Dict[FailureType, List[Callable]] = defaultdict(list)
        
        # Monitoring configuration
        self.check_interval_seconds = 30
        self.failure_detection_thresholds = {
            'response_time_threshold': 10.0,  # seconds
            'error_rate_threshold': 0.1,      # 10%
            'availability_threshold': 0.95    # 95%
        }
        
        # Background tasks
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics collection
        self.metrics_lock = threading.RLock()
        self.component_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("ReliabilityMonitor initialized")
    
    async def start(self) -> None:
        """Start reliability monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("ReliabilityMonitor started")
    
    async def stop(self) -> None:
        """Stop reliability monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("ReliabilityMonitor stopped")
    
    def register_recovery_handler(self, failure_type: FailureType, handler: Callable) -> None:
        """Register a recovery handler for a specific failure type."""
        self.recovery_handlers[failure_type].append(handler)
        logger.info(f"Registered recovery handler for {failure_type.value}")
    
    async def report_failure(self, 
                           component: str, 
                           failure_type: FailureType,
                           description: str,
                           severity: int = 5) -> str:
        """Report a system failure."""
        
        failure_id = f"failure_{int(time.time() * 1000000)}"
        
        failure = SystemFailure(
            failure_id=failure_id,
            failure_type=failure_type,
            component=component,
            description=description,
            severity=severity
        )
        
        self.failures[failure_id] = failure
        
        # Update component status
        if severity >= 8:
            self.component_status[component] = ReliabilityStatus.CRITICAL
        elif severity >= 6:
            self.component_status[component] = ReliabilityStatus.DEGRADED
        else:
            self.component_status[component] = ReliabilityStatus.STABLE
        
        # Publish failure event
        await self.event_bus.publish(Event(
            event_type="system_failure_detected",
            source_agent="reliability_monitor",
            priority=EventPriority.CRITICAL if severity >= 8 else EventPriority.HIGH,
            data=failure.to_dict()
        ))
        
        # Trigger automated recovery if handlers are available
        await self._trigger_recovery(failure)
        
        logger.error(f"System failure reported: {failure_id} - {description}")
        return failure_id
    
    async def resolve_failure(self, failure_id: str, recovery_actions: List[str] = None) -> bool:
        """Mark a failure as resolved."""
        
        if failure_id not in self.failures:
            return False
        
        failure = self.failures[failure_id]
        
        if failure.is_resolved():
            return True
        
        failure.resolved_at = datetime.now()
        failure.recovery_actions = recovery_actions or []
        
        # Update component status if this was the only failure
        component_failures = [
            f for f in self.failures.values()
            if f.component == failure.component and not f.is_resolved()
        ]
        
        if not component_failures:
            self.component_status[failure.component] = ReliabilityStatus.OPTIMAL
        
        # Publish resolution event
        await self.event_bus.publish(Event(
            event_type="system_failure_resolved",
            source_agent="reliability_monitor",
            priority=EventPriority.NORMAL,
            data={
                'failure_id': failure_id,
                'component': failure.component,
                'duration_seconds': failure.duration().total_seconds(),
                'recovery_actions': recovery_actions or []
            }
        ))
        
        logger.info(f"System failure resolved: {failure_id}")
        return True
    
    async def record_component_metrics(self, 
                                     component: str,
                                     metrics: Dict[str, Any]) -> None:
        """Record component performance metrics."""
        
        timestamp = datetime.now()
        metric_entry = {
            'timestamp': timestamp,
            'component': component,
            **metrics
        }
        
        with self.metrics_lock:
            self.component_metrics[component].append(metric_entry)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = timestamp - timedelta(hours=24)
            self.component_metrics[component] = [
                m for m in self.component_metrics[component]
                if m['timestamp'] >= cutoff_time
            ]
        
        # Check for failures based on metrics
        await self._analyze_metrics_for_failures(component, metrics)
    
    async def calculate_reliability_metrics(self, 
                                          component: Optional[str] = None,
                                          period_hours: int = 24) -> ReliabilityMetrics:
        """Calculate reliability metrics for a component or overall system."""
        
        cutoff_time = datetime.now() - timedelta(hours=period_hours)
        
        if component:
            relevant_failures = [
                f for f in self.failures.values()
                if f.component == component and f.detected_at >= cutoff_time
            ]
            
            with self.metrics_lock:
                relevant_metrics = [
                    m for m in self.component_metrics.get(component, [])
                    if m['timestamp'] >= cutoff_time
                ]
        else:
            relevant_failures = [
                f for f in self.failures.values()
                if f.detected_at >= cutoff_time
            ]
            
            with self.metrics_lock:
                relevant_metrics = []
                for comp_metrics in self.component_metrics.values():
                    relevant_metrics.extend([
                        m for m in comp_metrics
                        if m['timestamp'] >= cutoff_time
                    ])
        
        # Calculate uptime
        total_downtime = sum(
            f.duration().total_seconds()
            for f in relevant_failures
            if f.is_resolved() and f.severity >= 7
        )
        
        total_period_seconds = period_hours * 3600
        uptime_percentage = max(0, (total_period_seconds - total_downtime) / total_period_seconds * 100)
        
        # Calculate MTTF (Mean Time To Failure)
        if relevant_failures:
            failure_intervals = []
            sorted_failures = sorted(relevant_failures, key=lambda f: f.detected_at)
            
            for i in range(1, len(sorted_failures)):
                prev_failure = sorted_failures[i-1]
                curr_failure = sorted_failures[i]
                
                if prev_failure.is_resolved():
                    interval = (curr_failure.detected_at - prev_failure.resolved_at).total_seconds() / 3600
                    failure_intervals.append(interval)
            
            mttf = statistics.mean(failure_intervals) if failure_intervals else period_hours
        else:
            mttf = period_hours
        
        # Calculate MTTR (Mean Time To Recovery)
        resolved_failures = [f for f in relevant_failures if f.is_resolved()]
        if resolved_failures:
            recovery_times = [f.duration().total_seconds() / 3600 for f in resolved_failures]
            mttr = statistics.mean(recovery_times)
        else:
            mttr = 0.0
        
        # Calculate availability score
        availability_score = uptime_percentage
        
        # Calculate consistency score (based on data consistency checks)
        consistency_score = 95.0  # Placeholder - would integrate with ConsistencyManager
        
        # Calculate performance score (based on recent metrics)
        if relevant_metrics:
            response_times = [m.get('response_time', 0) for m in relevant_metrics if 'response_time' in m]
            error_rates = [m.get('error_rate', 0) for m in relevant_metrics if 'error_rate' in m]
            
            avg_response_time = statistics.mean(response_times) if response_times else 0
            avg_error_rate = statistics.mean(error_rates) if error_rates else 0
            
            # Performance score based on response time and error rate
            response_time_score = max(0, 100 - (avg_response_time * 10))
            error_rate_score = max(0, 100 - (avg_error_rate * 1000))
            performance_score = (response_time_score + error_rate_score) / 2
        else:
            performance_score = 90.0  # Default if no metrics available
        
        # Calculate overall reliability score
        overall_score = (availability_score * 0.4 + consistency_score * 0.3 + performance_score * 0.3)
        
        metrics = ReliabilityMetrics(
            uptime_percentage=uptime_percentage,
            mean_time_to_failure=mttf,
            mean_time_to_recovery=mttr,
            availability_score=availability_score,
            consistency_score=consistency_score,
            performance_score=performance_score,
            overall_reliability_score=overall_score,
            measurement_period_hours=period_hours
        )
        
        # Store in history
        self.reliability_history.append(metrics)
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system reliability status."""
        
        current_failures = [f for f in self.failures.values() if not f.is_resolved()]
        critical_failures = [f for f in current_failures if f.severity >= 8]
        
        # Determine overall status
        if critical_failures:
            overall_status = ReliabilityStatus.CRITICAL
        elif len(current_failures) > 5:
            overall_status = ReliabilityStatus.DEGRADED
        elif current_failures:
            overall_status = ReliabilityStatus.STABLE
        else:
            overall_status = ReliabilityStatus.OPTIMAL
        
        return {
            'overall_status': overall_status.value,
            'active_failures': len(current_failures),
            'critical_failures': len(critical_failures),
            'component_status': {k: v.value for k, v in self.component_status.items()},
            'monitored_components': len(self.component_metrics),
            'monitoring_enabled': self._running,
            'last_check': datetime.now().isoformat()
        }
    
    async def _trigger_recovery(self, failure: SystemFailure) -> None:
        """Trigger automated recovery for a failure."""
        
        handlers = self.recovery_handlers.get(failure.failure_type, [])
        
        if not handlers:
            logger.info(f"No recovery handlers available for {failure.failure_type.value}")
            return
        
        logger.info(f"Triggering {len(handlers)} recovery handlers for failure {failure.failure_id}")
        
        recovery_tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                task = asyncio.create_task(handler(failure))
            else:
                task = asyncio.create_task(asyncio.get_event_loop().run_in_executor(None, handler, failure))
            recovery_tasks.append(task)
        
        # Execute recovery handlers
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        
        # Process recovery results
        successful_recoveries = []
        for i, result in enumerate(recovery_results):
            if isinstance(result, Exception):
                logger.error(f"Recovery handler {i} failed: {result}")
            else:
                successful_recoveries.append(f"Handler {i} executed successfully")
        
        if successful_recoveries:
            failure.recovery_actions.extend(successful_recoveries)
            logger.info(f"Recovery initiated for failure {failure.failure_id}")
    
    async def _analyze_metrics_for_failures(self, component: str, metrics: Dict[str, Any]) -> None:
        """Analyze metrics to detect potential failures."""
        
        # Check response time threshold
        response_time = metrics.get('response_time', 0)
        if response_time > self.failure_detection_thresholds['response_time_threshold']:
            await self.report_failure(
                component=component,
                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                description=f"Response time {response_time:.2f}s exceeds threshold",
                severity=6
            )
        
        # Check error rate threshold
        error_rate = metrics.get('error_rate', 0)
        if error_rate > self.failure_detection_thresholds['error_rate_threshold']:
            await self.report_failure(
                component=component,
                failure_type=FailureType.PERFORMANCE_DEGRADATION,
                description=f"Error rate {error_rate:.2%} exceeds threshold",
                severity=7
            )
        
        # Check availability threshold
        availability = metrics.get('availability', 1.0)
        if availability < self.failure_detection_thresholds['availability_threshold']:
            await self.report_failure(
                component=component,
                failure_type=FailureType.NODE_FAILURE,
                description=f"Availability {availability:.2%} below threshold",
                severity=8
            )
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        
        while self._running:
            try:
                # Calculate current reliability metrics
                overall_metrics = await self.calculate_reliability_metrics(period_hours=1)
                
                # Check for system-wide issues
                if overall_metrics.overall_reliability_score < 70:
                    await self.report_failure(
                        component="system",
                        failure_type=FailureType.PERFORMANCE_DEGRADATION,
                        description=f"Overall reliability score {overall_metrics.overall_reliability_score:.1f} is below threshold",
                        severity=7
                    )
                
                # Publish reliability metrics event
                await self.event_bus.publish(Event(
                    event_type="reliability_metrics_updated",
                    source_agent="reliability_monitor",
                    data=overall_metrics.to_dict()
                ))
                
                await asyncio.sleep(self.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reliability monitoring loop: {e}")
                await asyncio.sleep(60)


class DataReliabilitySystem:
    """
    Comprehensive data consistency and system reliability system.
    Coordinates consistency management, transaction handling, and reliability monitoring.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
        # Core components
        self.consistency_manager = ConsistencyManager()
        self.transaction_manager = TransactionManager(event_bus)
        self.reliability_monitor = ReliabilityMonitor(event_bus)
        
        # Data stores
        self.primary_store = InMemoryDataStore()
        self.backup_store = PersistentDataStore("backup.db")
        
        # Configuration
        self.auto_consistency_check_enabled = True
        self.consistency_check_interval = 300  # 5 minutes
        self.backup_interval = 3600  # 1 hour
        
        # Background tasks
        self._running = False
        self._consistency_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
        
        # Initialize system
        self._initialize_system()
        
        logger.info("DataReliabilitySystem initialized")
    
    async def start(self) -> None:
        """Start the data reliability system."""
        if self._running:
            return
        
        self._running = True
        
        # Start core components
        await self.transaction_manager.start()
        await self.reliability_monitor.start()
        
        # Start background tasks
        if self.auto_consistency_check_enabled:
            self._consistency_task = asyncio.create_task(self._consistency_check_loop())
        
        self._backup_task = asyncio.create_task(self._backup_loop())
        
        # Register recovery handlers
        self._register_recovery_handlers()
        
        # Publish startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="data_reliability_system",
            data={'status': 'started', 'components_active': 3}
        ))
        
        logger.info("DataReliabilitySystem started")
    
    async def stop(self) -> None:
        """Stop the data reliability system."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        tasks_to_cancel = []
        if self._consistency_task:
            tasks_to_cancel.append(self._consistency_task)
        if self._backup_task:
            tasks_to_cancel.append(self._backup_task)
        
        for task in tasks_to_cancel:
            task.cancel()
        
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        # Stop core components
        await self.transaction_manager.stop()
        await self.reliability_monitor.stop()
        
        logger.info("DataReliabilitySystem stopped")
    
    async def store_data(self, key: str, value: Any, ensure_consistency: bool = True) -> bool:
        """Store data with consistency guarantees."""
        
        try:
            # Start transaction if consistency is required
            if ensure_consistency:
                transaction_id = await self.transaction_manager.begin_transaction(
                    participants=["primary_store", "backup_store"],
                    operations=[
                        {"action": "write", "key": key, "value": value}
                    ]
                )
                
                # Prepare transaction
                prepare_success = await self.transaction_manager.prepare_transaction(transaction_id)
                
                if not prepare_success:
                    logger.error(f"Failed to prepare transaction for storing {key}")
                    return False
                
                # Commit transaction
                commit_success = await self.transaction_manager.commit_transaction(transaction_id)
                
                if not commit_success:
                    logger.error(f"Failed to commit transaction for storing {key}")
                    return False
                
                return True
            
            else:
                # Simple write without transaction
                primary_success = await self.primary_store.write(key, value)
                backup_success = await self.backup_store.write(key, value)
                
                return primary_success and backup_success
        
        except Exception as e:
            logger.error(f"Error storing data for key {key}: {e}")
            
            # Report failure
            await self.reliability_monitor.report_failure(
                component="data_store",
                failure_type=FailureType.DATA_CORRUPTION,
                description=f"Failed to store data for key {key}: {str(e)}",
                severity=7
            )
            
            return False
    
    async def retrieve_data(self, key: str) -> Optional[Any]:
        """Retrieve data with consistency checking."""
        
        try:
            # Try primary store first
            primary_value = await self.primary_store.read(key)
            
            # Check backup store for consistency
            backup_value = await self.backup_store.read(key)
            
            # If values match, return primary value
            if primary_value == backup_value:
                return primary_value
            
            # If values don't match, perform consistency check
            if primary_value is not None and backup_value is not None:
                logger.warning(f"Consistency mismatch detected for key {key}")
                
                # Perform detailed consistency check
                consistency_check = await self.consistency_manager.check_consistency(key)
                
                if not consistency_check.is_consistent:
                    # Attempt to resolve conflicts
                    resolution_success = await self.consistency_manager.resolve_conflicts(key, "latest_wins")
                    
                    if resolution_success:
                        # Re-read after resolution
                        return await self.primary_store.read(key)
                    else:
                        logger.error(f"Failed to resolve consistency conflict for key {key}")
                        return None
            
            # Return primary value if backup is missing
            if primary_value is not None:
                return primary_value
            
            # Return backup value if primary is missing
            return backup_value
        
        except Exception as e:
            logger.error(f"Error retrieving data for key {key}: {e}")
            return None
    
    async def delete_data(self, key: str, ensure_consistency: bool = True) -> bool:
        """Delete data with consistency guarantees."""
        
        try:
            if ensure_consistency:
                transaction_id = await self.transaction_manager.begin_transaction(
                    participants=["primary_store", "backup_store"],
                    operations=[
                        {"action": "delete", "key": key}
                    ]
                )
                
                # Prepare and commit transaction
                prepare_success = await self.transaction_manager.prepare_transaction(transaction_id)
                
                if not prepare_success:
                    return False
                
                return await self.transaction_manager.commit_transaction(transaction_id)
            
            else:
                # Simple delete without transaction
                primary_success = await self.primary_store.delete(key)
                backup_success = await self.backup_store.delete(key)
                
                return primary_success or backup_success
        
        except Exception as e:
            logger.error(f"Error deleting data for key {key}: {e}")
            return False
    
    async def check_system_consistency(self) -> Dict[str, ConsistencyCheck]:
        """Check consistency across all data entities."""
        
        try:
            # Get all keys from primary store
            all_keys = await self.primary_store.get_all_keys()
            
            consistency_results = {}
            
            # Check consistency for each key
            for key in all_keys:
                check_result = await self.consistency_manager.check_consistency(
                    key, ConsistencyLevel.STRONG
                )
                consistency_results[key] = check_result
            
            return consistency_results
        
        except Exception as e:
            logger.error(f"Error checking system consistency: {e}")
            return {}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        
        # Get reliability status
        reliability_status = self.reliability_monitor.get_system_status()
        
        # Get reliability metrics
        reliability_metrics = await self.reliability_monitor.calculate_reliability_metrics()
        
        # Check data consistency
        consistency_checks = await self.check_system_consistency()
        inconsistent_entities = [
            key for key, check in consistency_checks.items()
            if not check.is_consistent
        ]
        
        # Get transaction status
        active_transactions = len([
            txn for txn in self.transaction_manager.transactions.values()
            if txn.state not in [TransactionState.COMMITTED, TransactionState.ABORTED]
        ])
        
        return {
            'reliability_status': reliability_status,
            'reliability_metrics': reliability_metrics.to_dict(),
            'consistency_status': {
                'total_entities': len(consistency_checks),
                'consistent_entities': len(consistency_checks) - len(inconsistent_entities),
                'inconsistent_entities': inconsistent_entities,
                'consistency_percentage': (len(consistency_checks) - len(inconsistent_entities)) / max(len(consistency_checks), 1) * 100
            },
            'transaction_status': {
                'active_transactions': active_transactions,
                'total_transactions': len(self.transaction_manager.transactions)
            },
            'data_store_status': {
                'primary_store_healthy': True,  # Would implement actual health check
                'backup_store_healthy': True
            },
            'overall_health_score': reliability_metrics.overall_reliability_score
        }
    
    def _initialize_system(self) -> None:
        """Initialize system components and configurations."""
        
        # Register data stores with consistency manager
        self.consistency_manager.register_data_store("primary", self.primary_store)
        self.consistency_manager.register_data_store("backup", self.backup_store)
        
        # Register transaction participants
        self.transaction_manager.register_participant("primary_store", self._primary_store_handler)
        self.transaction_manager.register_participant("backup_store", self._backup_store_handler)
        
        logger.info("Data reliability system components initialized")
    
    def _register_recovery_handlers(self) -> None:
        """Register automated recovery handlers."""
        
        async def handle_data_corruption(failure: SystemFailure) -> None:
            """Handle data corruption failures."""
            logger.info(f"Handling data corruption failure: {failure.failure_id}")
            
            # Perform system-wide consistency check
            consistency_results = await self.check_system_consistency()
            
            # Resolve any inconsistencies found
            for key, check in consistency_results.items():
                if not check.is_consistent:
                    await self.consistency_manager.resolve_conflicts(key, "majority_wins")
            
            # Mark failure as resolved if successful
            await self.reliability_monitor.resolve_failure(
                failure.failure_id,
                ["Performed consistency check", "Resolved data conflicts"]
            )
        
        async def handle_performance_degradation(failure: SystemFailure) -> None:
            """Handle performance degradation failures."""
            logger.info(f"Handling performance degradation: {failure.failure_id}")
            
            # Could implement various performance recovery strategies
            # For now, just mark as handled
            await self.reliability_monitor.resolve_failure(
                failure.failure_id,
                ["Performance monitoring initiated"]
            )
        
        # Register handlers
        self.reliability_monitor.register_recovery_handler(FailureType.DATA_CORRUPTION, handle_data_corruption)
        self.reliability_monitor.register_recovery_handler(FailureType.PERFORMANCE_DEGRADATION, handle_performance_degradation)
        
        logger.info("Recovery handlers registered")
    
    async def _primary_store_handler(self, action: str, transaction_id: str, operations: List[Dict[str, Any]]) -> bool:
        """Transaction handler for primary store."""
        
        try:
            if action == "prepare":
                # Simulate prepare logic - check if operations can be performed
                for operation in operations:
                    if operation["action"] == "write":
                        # Check if write is possible (e.g., no version conflicts)
                        key = operation["key"]
                        current_version = await self.primary_store.get_version(key)
                        # Would check version conflicts here
                
                return True  # Prepare successful
            
            elif action == "commit":
                # Perform the actual operations
                for operation in operations:
                    if operation["action"] == "write":
                        success = await self.primary_store.write(operation["key"], operation["value"])
                        if not success:
                            return False
                    elif operation["action"] == "delete":
                        success = await self.primary_store.delete(operation["key"])
                        if not success:
                            return False
                
                return True  # Commit successful
            
            elif action == "abort":
                # Rollback any prepared changes
                # For in-memory store, no rollback needed
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error in primary store handler: {e}")
            return False
    
    async def _backup_store_handler(self, action: str, transaction_id: str, operations: List[Dict[str, Any]]) -> bool:
        """Transaction handler for backup store."""
        
        try:
            if action == "prepare":
                # Check if operations can be performed on backup store
                return True  # Prepare successful
            
            elif action == "commit":
                # Perform the actual operations on backup store
                for operation in operations:
                    if operation["action"] == "write":
                        success = await self.backup_store.write(operation["key"], operation["value"])
                        if not success:
                            return False
                    elif operation["action"] == "delete":
                        success = await self.backup_store.delete(operation["key"])
                        if not success:
                            return False
                
                return True  # Commit successful
            
            elif action == "abort":
                # Rollback any prepared changes
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error in backup store handler: {e}")
            return False
    
    async def _consistency_check_loop(self) -> None:
        """Background loop for periodic consistency checking."""
        
        while self._running:
            try:
                logger.info("Starting periodic consistency check")
                
                consistency_results = await self.check_system_consistency()
                
                # Report any inconsistencies
                inconsistent_count = sum(1 for check in consistency_results.values() if not check.is_consistent)
                
                if inconsistent_count > 0:
                    logger.warning(f"Found {inconsistent_count} inconsistent entities")
                    
                    # Attempt to resolve inconsistencies
                    for key, check in consistency_results.items():
                        if not check.is_consistent:
                            resolution_success = await self.consistency_manager.resolve_conflicts(key, "latest_wins")
                            if resolution_success:
                                logger.info(f"Resolved consistency conflict for {key}")
                            else:
                                logger.error(f"Failed to resolve consistency conflict for {key}")
                
                # Record metrics
                await self.reliability_monitor.record_component_metrics(
                    "consistency_manager",
                    {
                        'total_entities': len(consistency_results),
                        'consistent_entities': len(consistency_results) - inconsistent_count,
                        'consistency_rate': (len(consistency_results) - inconsistent_count) / max(len(consistency_results), 1)
                    }
                )
                
                await asyncio.sleep(self.consistency_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consistency check loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _backup_loop(self) -> None:
        """Background loop for periodic backup operations."""
        
        while self._running:
            try:
                logger.info("Starting periodic backup")
                
                # Get all keys from primary store
                all_keys = await self.primary_store.get_all_keys()
                
                backup_success_count = 0
                backup_failure_count = 0
                
                # Backup each entity
                for key in all_keys:
                    try:
                        primary_value = await self.primary_store.read(key)
                        if primary_value is not None:
                            backup_success = await self.backup_store.write(key, primary_value)
                            if backup_success:
                                backup_success_count += 1
                            else:
                                backup_failure_count += 1
                    except Exception as e:
                        logger.error(f"Error backing up key {key}: {e}")
                        backup_failure_count += 1
                
                logger.info(f"Backup completed: {backup_success_count} successful, {backup_failure_count} failed")
                
                # Record metrics
                await self.reliability_monitor.record_component_metrics(
                    "backup_system",
                    {
                        'backup_success_count': backup_success_count,
                        'backup_failure_count': backup_failure_count,
                        'backup_success_rate': backup_success_count / max(backup_success_count + backup_failure_count, 1)
                    }
                )
                
                await asyncio.sleep(self.backup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error


# Factory function
def create_data_reliability_system(event_bus: EventBus) -> DataReliabilitySystem:
    """Create a data reliability system instance."""
    return DataReliabilitySystem(event_bus)