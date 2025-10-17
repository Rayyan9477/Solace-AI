"""
Performance Monitoring and Health Check System
Implements comprehensive monitoring, metrics collection, and health checks
"""

import os
import sys
import time
import psutil
import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["status"] = self.status.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    thread_count: int
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

class MetricsCollector:
    """Collects and stores application metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._make_key(name, tags)
            self.counters[key] = self.counters.get(key, 0) + value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        with self._lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.max_history)
            
            self.metrics[key].append({
                "timestamp": datetime.utcnow().isoformat(),
                "value": duration_ms,
                "type": "timing"
            })
    
    def record_value(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a generic value metric"""
        with self._lock:
            key = self._make_key(name, tags)
            if key not in self.metrics:
                self.metrics[key] = deque(maxlen=self.max_history)
            
            self.metrics[key].append({
                "timestamp": datetime.utcnow().isoformat(),
                "value": value,
                "type": "value"
            })
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create metric key with tags"""
        if not tags:
            return name
        
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            summary = {
                "counters": self.counters.copy(),
                "gauges": self.gauges.copy(),
                "timings": {},
                "values": {}
            }
            
            for key, history in self.metrics.items():
                if not history:
                    continue
                
                values = [item["value"] for item in history]
                last_item = list(history)[-1]
                
                metric_summary = {
                    "count": len(values),
                    "latest": values[-1] if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "last_updated": last_item["timestamp"]
                }
                
                if last_item["type"] == "timing":
                    summary["timings"][key] = metric_summary
                else:
                    summary["values"][key] = metric_summary
            
            return summary

class SystemHealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checks: Dict[str, Callable] = {}
        self.last_health_check: Optional[datetime] = None
        self.health_results: List[HealthCheckResult] = []
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks"""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("process_health", self._check_process_health)
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """Register a health check function"""
        self.health_checks[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started health monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped health monitoring")
    
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._record_system_metrics(system_metrics)
                
                # Run health checks
                self.run_health_checks()
                
                # Sleep until next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_bytes_sent = network.bytes_sent
        network_bytes_recv = network.bytes_recv
        
        # Process metrics
        current_process = psutil.Process()
        active_connections = len(current_process.connections())
        thread_count = current_process.num_threads()
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            active_connections=active_connections,
            thread_count=thread_count,
            process_count=process_count
        )
    
    def _record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics"""
        self.metrics_collector.set_gauge("system.cpu_percent", metrics.cpu_percent)
        self.metrics_collector.set_gauge("system.memory_percent", metrics.memory_percent)
        self.metrics_collector.set_gauge("system.memory_used_mb", metrics.memory_used_mb)
        self.metrics_collector.set_gauge("system.disk_usage_percent", metrics.disk_usage_percent)
        self.metrics_collector.set_gauge("system.disk_free_gb", metrics.disk_free_gb)
        self.metrics_collector.set_gauge("system.active_connections", metrics.active_connections)
        self.metrics_collector.set_gauge("system.thread_count", metrics.thread_count)
    
    def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks"""
        results = []
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                end_time = time.time()
                
                if not isinstance(result, HealthCheckResult):
                    # Convert simple return to HealthCheckResult
                    if isinstance(result, bool):
                        status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                        message = "OK" if result else "Check failed"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = str(result)
                    
                    result = HealthCheckResult(
                        name=name,
                        status=status,
                        message=message,
                        response_time_ms=(end_time - start_time) * 1000,
                        timestamp=datetime.utcnow()
                    )
                
                results.append(result)
                
                # Record metrics
                self.metrics_collector.record_timing(f"health_check.{name}", result.response_time_ms)
                self.metrics_collector.set_gauge(f"health_check.{name}.status", 
                                               1 if result.status == HealthStatus.HEALTHY else 0)
                
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check error: {str(e)}",
                    response_time_ms=0,
                    timestamp=datetime.utcnow()
                )
                results.append(result)
        
        self.health_results = results
        self.last_health_check = datetime.utcnow()
        return results
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            elif cpu_percent > 75 or memory_percent > 75 or disk_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Normal resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow()
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024 ** 3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            if free_gb < 1:  # Less than 1GB free
                status = HealthStatus.CRITICAL
                message = f"Critical: Only {free_gb:.2f}GB disk space remaining"
            elif free_gb < 5:  # Less than 5GB free
                status = HealthStatus.DEGRADED
                message = f"Warning: Only {free_gb:.2f}GB disk space remaining"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.2f}GB available ({usage_percent:.1f}% used)"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={
                    "free_gb": free_gb,
                    "usage_percent": usage_percent
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow()
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            available_gb = memory.available / (1024 ** 3)
            
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical: Memory usage at {usage_percent}%"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {usage_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {usage_percent}% ({available_gb:.2f}GB available)"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={
                    "usage_percent": usage_percent,
                    "available_gb": available_gb
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow()
            )
    
    def _check_process_health(self) -> HealthCheckResult:
        """Check process health"""
        try:
            current_process = psutil.Process()
            
            # Check if process is responsive
            cpu_times = current_process.cpu_times()
            memory_info = current_process.memory_info()
            thread_count = current_process.num_threads()
            
            # Basic health checks
            memory_mb = memory_info.rss / (1024 * 1024)
            
            if memory_mb > 1000:  # More than 1GB memory usage
                status = HealthStatus.DEGRADED
                message = f"High memory usage by process: {memory_mb:.2f}MB"
            elif thread_count > 100:  # Too many threads
                status = HealthStatus.DEGRADED
                message = f"High thread count: {thread_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process healthy - Memory: {memory_mb:.2f}MB, Threads: {thread_count}"
            
            return HealthCheckResult(
                name="process_health",
                status=status,
                message=message,
                response_time_ms=0,
                timestamp=datetime.utcnow(),
                details={
                    "memory_mb": memory_mb,
                    "thread_count": thread_count,
                    "cpu_user_time": cpu_times.user,
                    "cpu_system_time": cpu_times.system
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check process health: {str(e)}",
                response_time_ms=0,
                timestamp=datetime.utcnow()
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        if not self.health_results:
            return {
                "overall_status": "UNKNOWN",
                "last_check": None,
                "checks": []
            }
        
        # Determine overall status
        statuses = [result.status for result in self.health_results]
        if HealthStatus.CRITICAL in statuses:
            overall_status = "CRITICAL"
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = "UNHEALTHY"
        elif HealthStatus.DEGRADED in statuses:
            overall_status = "DEGRADED"
        else:
            overall_status = "HEALTHY"
        
        return {
            "overall_status": overall_status,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "checks": [result.to_dict() for result in self.health_results],
            "metrics_summary": self.metrics_collector.get_metrics_summary()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        summary = self.get_health_summary()
        
        if format.lower() == "json":
            return json.dumps(summary, indent=2, default=str)
        elif format.lower() == "prometheus":
            # Convert to Prometheus format
            lines = []
            metrics = self.metrics_collector.get_metrics_summary()
            
            for name, value in metrics.get("gauges", {}).items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
            
            for name, value in metrics.get("counters", {}).items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global health monitor instance
health_monitor = SystemHealthMonitor()

# Context manager for timing operations
class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.metric_name = metric_name
        self.tags = tags
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            health_monitor.metrics_collector.record_timing(
                self.metric_name, duration_ms, self.tags
            )

def timing_decorator(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with TimingContext(metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator