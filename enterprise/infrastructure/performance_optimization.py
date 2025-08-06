"""
Performance Optimization and Scalability Infrastructure
Implements caching, load balancing, and performance monitoring
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
import aiohttp
import hashlib
import threading
from collections import defaultdict, deque
import psutil
import concurrent.futures
from functools import wraps
import weakref

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_BASED = "health_based"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    timestamp: datetime
    response_time: float
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    active_connections: int
    cache_hit_rate: float
    error_rate: float
    throughput: float
    queue_size: int


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[int]
    size_bytes: int


@dataclass
class ServerNode:
    """Server node for load balancing"""
    node_id: str
    host: str
    port: int
    weight: float
    active_connections: int
    health_score: float
    last_health_check: datetime
    response_times: deque
    is_healthy: bool = True
    total_requests: int = 0


class InMemoryCache:
    """High-performance in-memory cache with multiple strategies"""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.access_counts = defaultdict(int)  # For LFU
        self._lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
                
            entry = self.cache[key]
            
            # Check TTL
            if entry.ttl and (datetime.utcnow() - entry.created_at).seconds > entry.ttl:
                await self._remove_key(key)
                self.stats["misses"] += 1
                return None
                
            # Update access tracking
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            self.access_counts[key] += 1
            
            if self.strategy == CacheStrategy.LRU:
                self.access_order.remove(key)
                self.access_order.append(key)
                
            self.stats["hits"] += 1
            return entry.value
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        async with self._lock:
            # Calculate size
            size_bytes = len(str(value).encode('utf-8'))
            
            # Check if we need to evict
            while len(self.cache) >= self.max_size:
                await self._evict_one()
                
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # Remove old entry if exists
            if key in self.cache:
                await self._remove_key(key)
                
            self.cache[key] = entry
            self.access_order.append(key)
            self.access_counts[key] = 1
            self.stats["total_size"] += size_bytes
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self.cache:
                await self._remove_key(key)
                return True
            return False
            
    async def _evict_one(self):
        """Evict one entry based on strategy"""
        if not self.cache:
            return
            
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key_to_remove = self.access_order.popleft()
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key_to_remove = min(self.cache.keys(), key=lambda k: self.access_counts[k])
        else:
            # Default to LRU
            key_to_remove = self.access_order.popleft()
            
        await self._remove_key(key_to_remove)
        self.stats["evictions"] += 1
        
    async def _remove_key(self, key: str):
        """Remove key and update tracking"""
        if key in self.cache:
            entry = self.cache[key]
            self.stats["total_size"] -= entry.size_bytes
            del self.cache[key]
            
        if key in self.access_order:
            self.access_order.remove(key)
            
        if key in self.access_counts:
            del self.access_counts[key]
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "total_entries": len(self.cache),
                "total_size_bytes": self.stats["total_size"],
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"]
            }
            
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_size": 0}


class DistributedCache:
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache = InMemoryCache(max_size=1000)  # L1 cache
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        await self.redis_client.ping()
        logger.info("Distributed cache initialized")
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value with L1 and L2 cache"""
        # Try L1 cache first
        value = await self.local_cache.get(key)
        if value is not None:
            return value
            
        # Try Redis (L2 cache)
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    value = json.loads(cached_data)
                    # Store in L1 cache
                    await self.local_cache.set(key, value, ttl=300)  # 5 minute TTL in L1
                    return value
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                
        return None
        
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in both caches"""
        # Set in L1 cache
        await self.local_cache.set(key, value, ttl=min(ttl, 300))
        
        # Set in Redis (L2 cache)
        if self.redis_client:
            try:
                serialized_value = json.dumps(value, default=str)
                await self.redis_client.setex(key, ttl, serialized_value)
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                
    async def delete(self, key: str):
        """Delete from both caches"""
        await self.local_cache.delete(key)
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                
    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis pattern invalidation error: {e}")


class LoadBalancer:
    """Intelligent load balancer with health monitoring"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.nodes: Dict[str, ServerNode] = {}
        self.current_index = 0
        self._lock = asyncio.Lock()
        
    async def add_node(self, node_id: str, host: str, port: int, weight: float = 1.0):
        """Add server node to load balancer"""
        async with self._lock:
            node = ServerNode(
                node_id=node_id,
                host=host,
                port=port,
                weight=weight,
                active_connections=0,
                health_score=1.0,
                last_health_check=datetime.utcnow(),
                response_times=deque(maxlen=100)
            )
            self.nodes[node_id] = node
            logger.info(f"Added node {node_id} ({host}:{port}) to load balancer")
            
    async def remove_node(self, node_id: str):
        """Remove server node from load balancer"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed node {node_id} from load balancer")
                
    async def get_next_node(self) -> Optional[ServerNode]:
        """Get next available node based on strategy"""
        async with self._lock:
            healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]
            
            if not healthy_nodes:
                return None
                
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                node = healthy_nodes[self.current_index % len(healthy_nodes)]
                self.current_index += 1
                
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                # Weighted selection based on node weights
                total_weight = sum(node.weight for node in healthy_nodes)
                if total_weight == 0:
                    node = healthy_nodes[0]
                else:
                    # Simplified weighted selection
                    weights = [node.weight / total_weight for node in healthy_nodes]
                    import random
                    node = random.choices(healthy_nodes, weights=weights)[0]
                    
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                node = min(healthy_nodes, key=lambda n: n.active_connections)
                
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                node = min(healthy_nodes, 
                         key=lambda n: sum(n.response_times) / len(n.response_times) 
                         if n.response_times else float('inf'))
                         
            elif self.strategy == LoadBalancingStrategy.HEALTH_BASED:
                node = max(healthy_nodes, key=lambda n: n.health_score)
                
            else:
                node = healthy_nodes[0]
                
            return node
            
    async def record_request(self, node_id: str, response_time: float, success: bool = True):
        """Record request metrics for a node"""
        async with self._lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.response_times.append(response_time)
                node.total_requests += 1
                
                if success:
                    # Improve health score for successful requests
                    node.health_score = min(1.0, node.health_score + 0.01)
                else:
                    # Degrade health score for failures
                    node.health_score = max(0.0, node.health_score - 0.1)
                    
    async def health_check_all(self):
        """Perform health checks on all nodes"""
        for node in self.nodes.values():
            await self._health_check_node(node)
            
    async def _health_check_node(self, node: ServerNode):
        """Perform health check on a single node"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{node.host}:{node.port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        node.is_healthy = True
                        node.health_score = min(1.0, node.health_score + 0.05)
                        node.response_times.append(response_time)
                    else:
                        node.is_healthy = False
                        node.health_score = max(0.0, node.health_score - 0.2)
                        
        except Exception as e:
            logger.warning(f"Health check failed for node {node.node_id}: {e}")
            node.is_healthy = False
            node.health_score = max(0.0, node.health_score - 0.3)
            
        node.last_health_check = datetime.utcnow()
        
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        async with self._lock:
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
            total_requests = sum(node.total_requests for node in self.nodes.values())
            avg_response_time = 0
            
            if self.nodes:
                total_response_times = []
                for node in self.nodes.values():
                    total_response_times.extend(node.response_times)
                if total_response_times:
                    avg_response_time = sum(total_response_times) / len(total_response_times)
                    
            return {
                "total_nodes": len(self.nodes),
                "healthy_nodes": healthy_nodes,
                "total_requests": total_requests,
                "average_response_time": avg_response_time,
                "strategy": self.strategy.value,
                "node_details": {
                    node_id: {
                        "is_healthy": node.is_healthy,
                        "health_score": node.health_score,
                        "active_connections": node.active_connections,
                        "total_requests": node.total_requests,
                        "avg_response_time": sum(node.response_times) / len(node.response_times) 
                                          if node.response_times else 0
                    }
                    for node_id, node in self.nodes.items()
                }
            }


class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, sampling_interval: int = 60):
        self.sampling_interval = sampling_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.is_monitoring = False
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "response_time": 2.0,
            "error_rate": 5.0
        }
        self.alert_callbacks: List[Callable] = []
        
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
        
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        logger.info("Performance monitoring stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                await asyncio.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
                
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Application metrics (these would be populated by the application)
        response_time = 0.0  # Would be calculated from request tracking
        active_connections = 0  # Would be tracked by connection pool
        cache_hit_rate = 0.0  # Would be provided by cache
        error_rate = 0.0  # Would be calculated from error tracking
        throughput = 0.0  # Would be calculated from request rate
        queue_size = 0  # Would be provided by task queue
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            response_time=response_time,
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_io=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0,
            network_io=network_io.bytes_sent + network_io.bytes_recv if network_io else 0,
            active_connections=active_connections,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate,
            throughput=throughput,
            queue_size=queue_size
        )
        
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check metrics against alert thresholds"""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            
        if metrics.response_time > self.alert_thresholds["response_time"]:
            alerts.append(f"High response time: {metrics.response_time:.2f}s")
            
        if metrics.error_rate > self.alert_thresholds["error_rate"]:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
            
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    await callback(alert, metrics)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
    def add_alert_callback(self, callback: Callable):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
        
    async def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for the specified time period"}
            
        return {
            "time_period_minutes": minutes,
            "sample_count": len(recent_metrics),
            "cpu_usage": {
                "avg": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
                "max": max(m.cpu_usage for m in recent_metrics),
                "min": min(m.cpu_usage for m in recent_metrics)
            },
            "memory_usage": {
                "avg": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
                "max": max(m.memory_usage for m in recent_metrics),
                "min": min(m.memory_usage for m in recent_metrics)
            },
            "response_time": {
                "avg": sum(m.response_time for m in recent_metrics) / len(recent_metrics),
                "max": max(m.response_time for m in recent_metrics),
                "min": min(m.response_time for m in recent_metrics)
            },
            "error_rate": {
                "avg": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
                "max": max(m.error_rate for m in recent_metrics)
            },
            "throughput": {
                "avg": sum(m.throughput for m in recent_metrics) / len(recent_metrics),
                "max": max(m.throughput for m in recent_metrics)
            }
        }


class ConnectionPoolManager:
    """Manages connection pools for database and external services"""
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self.pool_configs: Dict[str, Dict] = {}
        
    async def create_pool(self, pool_name: str, connection_string: str,
                         min_size: int = 1, max_size: int = 10, **kwargs):
        """Create a connection pool"""
        try:
            if "postgresql" in connection_string:
                import asyncpg
                pool = await asyncpg.create_pool(
                    connection_string,
                    min_size=min_size,
                    max_size=max_size,
                    **kwargs
                )
            elif "redis" in connection_string:
                pool = redis.ConnectionPool.from_url(
                    connection_string,
                    max_connections=max_size,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported connection type: {connection_string}")
                
            self.pools[pool_name] = pool
            self.pool_configs[pool_name] = {
                "connection_string": connection_string,
                "min_size": min_size,
                "max_size": max_size,
                "created_at": datetime.utcnow()
            }
            
            logger.info(f"Created connection pool '{pool_name}' (size: {min_size}-{max_size})")
            
        except Exception as e:
            logger.error(f"Failed to create connection pool '{pool_name}': {e}")
            raise
            
    async def get_connection(self, pool_name: str):
        """Get connection from pool"""
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
            
        pool = self.pools[pool_name]
        
        if hasattr(pool, 'acquire'):
            # asyncpg pool
            return pool.acquire()
        else:
            # Redis pool
            return redis.Redis(connection_pool=pool)
            
    async def close_pool(self, pool_name: str):
        """Close and remove connection pool"""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            
            if hasattr(pool, 'close'):
                await pool.close()
                
            del self.pools[pool_name]
            del self.pool_configs[pool_name]
            
            logger.info(f"Closed connection pool '{pool_name}'")
            
    async def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        stats = {}
        
        for pool_name, pool in self.pools.items():
            config = self.pool_configs[pool_name]
            
            if hasattr(pool, 'size'):
                # asyncpg pool
                stats[pool_name] = {
                    "current_size": pool.size,
                    "max_size": config["max_size"],
                    "min_size": config["min_size"],
                    "created_at": config["created_at"].isoformat()
                }
            else:
                # Redis or other pools
                stats[pool_name] = {
                    "max_connections": config["max_size"],
                    "created_at": config["created_at"].isoformat()
                }
                
        return stats


def performance_cache(ttl: int = 3600, cache_key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        # Use a weak reference to cache to allow garbage collection
        cache_ref = weakref.WeakValueDictionary()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                # Default cache key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
            # Check cache
            if cache_key in cache_ref:
                entry = cache_ref[cache_key]
                if (datetime.utcnow() - entry["timestamp"]).seconds < ttl:
                    return entry["value"]
                    
            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_ref[cache_key] = {
                "value": result,
                "timestamp": datetime.utcnow()
            }
            
            return result
            
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting function calls"""
    def decorator(func: Callable):
        call_times = deque()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls outside the time window
            while call_times and call_times[0] < now - 60:
                call_times.popleft()
                
            # Check rate limit
            if len(call_times) >= requests_per_minute:
                raise Exception(f"Rate limit exceeded: {requests_per_minute} requests per minute")
                
            call_times.append(now)
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator


class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.cache = DistributedCache()
        self.load_balancer = LoadBalancer()
        self.performance_monitor = PerformanceMonitor()
        self.connection_pools = ConnectionPoolManager()
        self._optimization_tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize performance optimization systems"""
        await self.cache.initialize()
        await self.performance_monitor.start_monitoring()
        
        # Start background optimization tasks
        self._optimization_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cache_cleanup_loop()),
            asyncio.create_task(self._performance_tuning_loop())
        ]
        
        logger.info("Performance optimizer initialized")
        
    async def shutdown(self):
        """Shutdown performance optimization systems"""
        await self.performance_monitor.stop_monitoring()
        
        # Cancel background tasks
        for task in self._optimization_tasks:
            task.cancel()
            
        # Close connection pools
        for pool_name in list(self.connection_pools.pools.keys()):
            await self.connection_pools.close_pool(pool_name)
            
        logger.info("Performance optimizer shutdown complete")
        
    async def _health_check_loop(self):
        """Periodic health checks for load balancer nodes"""
        while True:
            try:
                await self.load_balancer.health_check_all()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
                
    async def _cache_cleanup_loop(self):
        """Periodic cache cleanup and optimization"""
        while True:
            try:
                # Get cache stats
                stats = await self.cache.local_cache.get_stats()
                
                # If hit rate is low, consider adjusting cache size or strategy
                if stats["hit_rate"] < 0.3 and stats["total_entries"] > 100:
                    logger.info(f"Low cache hit rate: {stats['hit_rate']:.2f}")
                    # Could implement cache size adjustment here
                    
                await asyncio.sleep(300)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {e}")
                await asyncio.sleep(300)
                
    async def _performance_tuning_loop(self):
        """Automatic performance tuning based on metrics"""
        while True:
            try:
                # Get recent performance metrics
                summary = await self.performance_monitor.get_performance_summary(minutes=10)
                
                if "error" not in summary:
                    # Adjust based on performance metrics
                    if summary["cpu_usage"]["avg"] > 70:
                        logger.info("High CPU usage detected - consider scaling")
                        
                    if summary["memory_usage"]["avg"] > 80:
                        logger.info("High memory usage detected - consider cache cleanup")
                        
                    if summary["response_time"]["avg"] > 1.0:
                        logger.info("High response time detected - check bottlenecks")
                        
                await asyncio.sleep(600)  # Check every 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance tuning loop: {e}")
                await asyncio.sleep(600)
                
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get overall optimization system status"""
        cache_stats = await self.cache.local_cache.get_stats()
        lb_stats = await self.load_balancer.get_load_balancer_stats()
        perf_stats = await self.performance_monitor.get_performance_summary(minutes=30)
        pool_stats = await self.connection_pools.get_pool_stats()
        
        return {
            "cache": cache_stats,
            "load_balancer": lb_stats,
            "performance": perf_stats,
            "connection_pools": pool_stats,
            "optimization_tasks_running": len([t for t in self._optimization_tasks if not t.done()]),
            "status": "healthy"
        }