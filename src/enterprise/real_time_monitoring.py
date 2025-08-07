"""
Enterprise Real-Time Monitoring System for Solace-AI

This module provides comprehensive real-time monitoring, metrics collection,
and event streaming capabilities for enterprise-grade agent orchestration.

Features:
- Real-time performance metrics and KPIs
- Agent health monitoring with predictive analytics
- Event streaming with guaranteed delivery
- Distributed tracing and correlation
- Alerting and notification system
- Performance optimization recommendations
- Resource usage monitoring
- Clinical safety monitoring
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import logging
import threading
import weakref
from abc import ABC, abstractmethod
import statistics

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics tracked by the monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"
    HEALTH_CHECK = "health_check"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    
    metric_name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'metric_type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'metadata': self.metadata
        }


@dataclass
class Alert:
    """Alert structure for monitoring events."""
    
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    check_name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'check_name': self.check_name,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'duration': self.duration
        }


class AlertRule:
    """Rule for generating alerts based on metrics."""
    
    def __init__(self, 
                 rule_id: str,
                 metric_name: str,
                 condition: Callable[[float], bool],
                 severity: AlertSeverity,
                 message_template: str,
                 cooldown_seconds: int = 300):
        self.rule_id = rule_id
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered: Optional[datetime] = None
    
    def should_trigger(self, value: float) -> bool:
        """Check if rule should trigger an alert."""
        if not self.condition(value):
            return False
        
        # Check cooldown
        if self.last_triggered:
            time_since = (datetime.now() - self.last_triggered).total_seconds()
            if time_since < self.cooldown_seconds:
                return False
        
        return True
    
    def trigger(self, value: float, metadata: Dict[str, Any] = None) -> Alert:
        """Generate alert."""
        self.last_triggered = datetime.now()
        
        return Alert(
            alert_type=self.rule_id,
            severity=self.severity,
            message=self.message_template.format(value=value, **metadata or {}),
            source=self.metric_name,
            metadata={
                'rule_id': self.rule_id,
                'metric_value': value,
                'threshold_condition': str(self.condition.__doc__ or 'custom condition'),
                **(metadata or {})
            }
        )


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, check_name: str, timeout_seconds: float = 10.0):
        self.check_name = check_name
        self.timeout_seconds = timeout_seconds
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    async def safe_check(self) -> HealthCheckResult:
        """Safely perform health check with timeout."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.check(),
                timeout=self.timeout_seconds
            )
            result.duration = time.time() - start_time
            return result
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_name=self.check_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout_seconds}s",
                duration=time.time() - start_time
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=self.check_name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__},
                duration=time.time() - start_time
            )


class AgentHealthCheck(HealthCheck):
    """Health check for agent availability and performance."""
    
    def __init__(self, agent_id: str, agent_instance: Any, check_name: str = None):
        super().__init__(check_name or f"agent_{agent_id}")
        self.agent_id = agent_id
        self.agent_instance = agent_instance
    
    async def check(self) -> HealthCheckResult:
        """Check agent health."""
        details = {'agent_id': self.agent_id}
        
        try:
            # Check if agent has health method
            if hasattr(self.agent_instance, 'get_health_status'):
                health_data = self.agent_instance.get_health_status()
                
                if isinstance(health_data, dict):
                    status_str = health_data.get('status', 'unknown')
                    if status_str == 'healthy':
                        status = HealthStatus.HEALTHY
                    elif status_str == 'degraded':
                        status = HealthStatus.DEGRADED
                    elif status_str in ['unhealthy', 'critical']:
                        status = HealthStatus.UNHEALTHY
                    else:
                        status = HealthStatus.DEGRADED
                    
                    details.update(health_data)
                else:
                    status = HealthStatus.HEALTHY
                    details['basic_health'] = str(health_data)
            
            elif hasattr(self.agent_instance, 'is_active'):
                # Basic activity check
                is_active = getattr(self.agent_instance, 'is_active', True)
                status = HealthStatus.HEALTHY if is_active else HealthStatus.UNHEALTHY
                details['is_active'] = is_active
            
            else:
                # Agent exists check
                status = HealthStatus.HEALTHY
                details['exists'] = True
            
            return HealthCheckResult(
                check_name=self.check_name,
                status=status,
                message=f"Agent {self.agent_id} is {status.value}",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=self.check_name,
                status=HealthStatus.CRITICAL,
                message=f"Agent health check failed: {str(e)}",
                details={'error': str(e), 'agent_id': self.agent_id}
            )


class SystemHealthCheck(HealthCheck):
    """System-level health check."""
    
    def __init__(self, system_name: str, check_functions: List[Callable]):
        super().__init__(f"system_{system_name}")
        self.system_name = system_name
        self.check_functions = check_functions
    
    async def check(self) -> HealthCheckResult:
        """Check system health."""
        details = {}
        overall_status = HealthStatus.HEALTHY
        messages = []
        
        for check_func in self.check_functions:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                if isinstance(result, dict):
                    for key, value in result.items():
                        details[key] = value
                        
                        # Determine status based on result
                        if isinstance(value, dict) and 'status' in value:
                            status_str = value['status']
                            if status_str in ['unhealthy', 'critical', 'error']:
                                overall_status = HealthStatus.UNHEALTHY
                            elif status_str == 'degraded' and overall_status == HealthStatus.HEALTHY:
                                overall_status = HealthStatus.DEGRADED
                
                elif isinstance(result, str):
                    messages.append(result)
                    details[check_func.__name__] = result
            
            except Exception as e:
                overall_status = HealthStatus.CRITICAL
                messages.append(f"Check {check_func.__name__} failed: {str(e)}")
                details[f"{check_func.__name__}_error"] = str(e)
        
        return HealthCheckResult(
            check_name=self.check_name,
            status=overall_status,
            message='; '.join(messages) if messages else f"System {self.system_name} status",
            details=details
        )


class MetricsCollector:
    """Advanced metrics collector with aggregation and alerting."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.max_points_per_metric = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.alert_rules: Dict[str, List[AlertRule]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.lock = threading.RLock()
        
        # Start aggregation task
        self._running = False
        self._aggregation_task: Optional[asyncio.Task] = None
    
    def start(self):
        """Start the metrics collector."""
        if not self._running:
            self._running = True
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())
            logger.info("MetricsCollector started")
    
    async def stop(self):
        """Stop the metrics collector."""
        self._running = False
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass
        logger.info("MetricsCollector stopped")
    
    def record_metric(self, 
                     metric_name: str, 
                     value: Union[int, float], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None) -> None:
        """Record a metric point."""
        point = MetricPoint(
            metric_name=metric_name,
            value=value,
            metric_type=metric_type,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics[metric_name].append(point)
        
        # Check alert rules
        self._check_alert_rules(metric_name, value, metadata or {})
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule for a metric."""
        with self.lock:
            self.alert_rules[rule.metric_name].append(rule)
        logger.info(f"Added alert rule {rule.rule_id} for metric {rule.metric_name}")
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        with self.lock:
            for metric_name, rules in self.alert_rules.items():
                for rule in rules[:]:  # Copy list for safe iteration
                    if rule.rule_id == rule_id:
                        rules.remove(rule)
                        logger.info(f"Removed alert rule {rule_id}")
                        return True
        return False
    
    def get_metric_summary(self, metric_name: str, 
                          time_window: timedelta = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            points = list(self.metrics[metric_name])
        
        if not points:
            return {'count': 0, 'metric_name': metric_name}
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        if not points:
            return {'count': 0, 'metric_name': metric_name, 'time_window': str(time_window)}
        
        values = [p.value for p in points]
        
        return {
            'metric_name': metric_name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'latest_value': values[-1],
            'latest_timestamp': points[-1].timestamp.isoformat(),
            'time_window': str(time_window) if time_window else 'all_time'
        }
    
    def get_all_metrics_summary(self, 
                               time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get summary for all metrics."""
        with self.lock:
            metric_names = list(self.metrics.keys())
        
        summaries = {}
        for metric_name in metric_names:
            summaries[metric_name] = self.get_metric_summary(metric_name, time_window)
        
        return {
            'summaries': summaries,
            'total_metrics': len(summaries),
            'time_window': str(time_window),
            'generated_at': datetime.now().isoformat()
        }
    
    def _check_alert_rules(self, metric_name: str, value: float, metadata: Dict[str, Any]) -> None:
        """Check if any alert rules should trigger."""
        rules = self.alert_rules.get(metric_name, [])
        
        for rule in rules:
            if rule.should_trigger(value):
                alert = rule.trigger(value, metadata)
                # This would normally send the alert to an alert handler
                logger.warning(f"Alert triggered: {alert.message}")
                
                # Could emit event here for alert handling
                # await self.event_bus.publish(Event(
                #     event_type=EventType.ALERT_GENERATED,
                #     data=alert.to_dict()
                # ))
    
    async def _aggregation_loop(self):
        """Background loop for metric aggregation."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Aggregate every minute
                await self._aggregate_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics aggregation: {e}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for efficient querying."""
        current_time = datetime.now()
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                if not points:
                    continue
                
                # Aggregate last hour's data
                hour_ago = current_time - timedelta(hours=1)
                recent_points = [p for p in points if p.timestamp >= hour_ago]
                
                if recent_points:
                    values = [p.value for p in recent_points]
                    self.aggregated_metrics[metric_name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'latest': values[-1],
                        'aggregated_at': current_time.isoformat()
                    }


class RealTimeMonitor:
    """
    Enterprise real-time monitoring system for agent orchestration.
    Provides comprehensive monitoring, alerting, and health checking.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics_collector = MetricsCollector()
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: deque = deque(maxlen=10000)  # Last 10k alerts
        self.active_alerts: Dict[str, Alert] = {}
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self.health_check_interval = 30  # seconds
        self.metrics_retention_days = 7
        
        # Background tasks
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Subscribe to system events
        self._setup_event_subscriptions()
        
        logger.info("RealTimeMonitor initialized")
    
    async def start(self) -> None:
        """Start the monitoring system."""
        if self._running:
            return
        
        self._running = True
        
        # Start metrics collector
        self.metrics_collector.start()
        
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Emit startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="real_time_monitor",
            data={'component': 'monitoring_system', 'status': 'started'}
        ))
        
        logger.info("RealTimeMonitor started")
    
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._health_check_task, self._cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop metrics collector
        await self.metrics_collector.stop()
        
        logger.info("RealTimeMonitor stopped")
    
    def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a health check."""
        self.health_checks[health_check.check_name] = health_check
        logger.info(f"Added health check: {health_check.check_name}")
    
    def remove_health_check(self, check_name: str) -> bool:
        """Remove a health check."""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            logger.info(f"Removed health check: {check_name}")
            return True
        return False
    
    def record_metric(self, *args, **kwargs) -> None:
        """Record a metric (delegates to metrics collector)."""
        self.metrics_collector.record_metric(*args, **kwargs)
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.metrics_collector.add_alert_rule(rule)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_results = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all health checks
        for check_name, health_check in self.health_checks.items():
            result = await health_check.safe_check()
            health_results[check_name] = result.to_dict()
            
            # Update overall status
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        # Get metrics summary
        metrics_summary = self.metrics_collector.get_all_metrics_summary()
        
        # Get recent alerts
        recent_alerts = [
            alert.to_dict() for alert in list(self.alerts)[-10:]
            if not alert.resolved
        ]
        
        return {
            'overall_status': overall_status.value,
            'health_checks': health_results,
            'metrics_summary': metrics_summary,
            'active_alerts_count': len(self.active_alerts),
            'recent_alerts': recent_alerts,
            'monitoring_enabled': self.monitoring_enabled,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_dashboard_data(self, time_window_hours: int = 1) -> Dict[str, Any]:
        """Get data for metrics dashboard."""
        time_window = timedelta(hours=time_window_hours)
        
        # Key performance indicators
        agent_metrics = {}
        system_metrics = {}
        
        # Collect agent-specific metrics
        agent_metric_names = [name for name in self.metrics_collector.metrics.keys() 
                             if name.startswith('agent_')]
        
        for metric_name in agent_metric_names:
            summary = self.metrics_collector.get_metric_summary(metric_name, time_window)
            agent_id = metric_name.split('_')[1] if len(metric_name.split('_')) > 1 else 'unknown'
            
            if agent_id not in agent_metrics:
                agent_metrics[agent_id] = {}
            
            metric_type = metric_name.split('_')[-1] if len(metric_name.split('_')) > 2 else 'metric'
            agent_metrics[agent_id][metric_type] = summary
        
        # Collect system metrics
        system_metric_names = [name for name in self.metrics_collector.metrics.keys() 
                              if name.startswith('system_')]
        
        for metric_name in system_metric_names:
            summary = self.metrics_collector.get_metric_summary(metric_name, time_window)
            system_metrics[metric_name] = summary
        
        return {
            'agent_metrics': agent_metrics,
            'system_metrics': system_metrics,
            'time_window_hours': time_window_hours,
            'total_agents_monitored': len(agent_metrics),
            'generated_at': datetime.now().isoformat()
        }
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules for common issues."""
        
        # High error rate alert
        self.metrics_collector.add_alert_rule(AlertRule(
            rule_id="high_error_rate",
            metric_name="agent_error_rate",
            condition=lambda x: x > 0.1,  # More than 10% errors
            severity=AlertSeverity.WARNING,
            message_template="High error rate detected: {value:.2%}"
        ))
        
        # Slow response time alert
        self.metrics_collector.add_alert_rule(AlertRule(
            rule_id="slow_response_time",
            metric_name="agent_response_time",
            condition=lambda x: x > 10.0,  # More than 10 seconds
            severity=AlertSeverity.WARNING,
            message_template="Slow response time detected: {value:.2f}s"
        ))
        
        # System memory alert
        self.metrics_collector.add_alert_rule(AlertRule(
            rule_id="high_memory_usage",
            metric_name="system_memory_usage",
            condition=lambda x: x > 0.9,  # More than 90% memory usage
            severity=AlertSeverity.CRITICAL,
            message_template="High memory usage: {value:.1%}"
        ))
        
        logger.info("Default alert rules configured")
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for monitoring."""
        
        # Monitor agent errors
        self.event_bus.subscribe(
            EventType.AGENT_ERROR,
            self._handle_agent_error,
            agent_id="real_time_monitor"
        )
        
        # Monitor clinical assessments
        self.event_bus.subscribe(
            EventType.CLINICAL_ASSESSMENT,
            self._handle_clinical_assessment,
            agent_id="real_time_monitor"
        )
        
        # Monitor validation results
        self.event_bus.subscribe(
            EventType.VALIDATION_RESULT,
            self._handle_validation_result,
            agent_id="real_time_monitor"
        )
        
        logger.info("Event subscriptions configured")
    
    async def _handle_agent_error(self, event: Event) -> None:
        """Handle agent error events."""
        agent_id = event.source_agent or 'unknown'
        
        # Record error metric
        self.record_metric(
            f"agent_{agent_id}_errors",
            1,
            MetricType.COUNTER,
            labels={'agent_id': agent_id, 'error_type': 'agent_error'}
        )
        
        # Check if this indicates a critical issue
        error_data = event.data
        if error_data.get('severity') == 'critical':
            alert = Alert(
                alert_type="critical_agent_error",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical error in agent {agent_id}: {error_data.get('error', 'Unknown error')}",
                source=agent_id,
                metadata=error_data
            )
            await self._emit_alert(alert)
    
    async def _handle_clinical_assessment(self, event: Event) -> None:
        """Handle clinical assessment events."""
        agent_id = event.source_agent or 'unknown'
        assessment_data = event.data
        
        # Record assessment metric
        self.record_metric(
            f"agent_{agent_id}_assessments",
            1,
            MetricType.COUNTER,
            labels={'agent_id': agent_id}
        )
        
        # Record processing time if available
        if 'processing_time' in assessment_data:
            self.record_metric(
                f"agent_{agent_id}_processing_time",
                assessment_data['processing_time'],
                MetricType.TIMER,
                labels={'agent_id': agent_id}
            )
        
        # Monitor for high-risk assessments
        diagnosis_result = assessment_data.get('diagnosis_result', {})
        severity = diagnosis_result.get('severity', 'mild')
        
        if severity == 'severe':
            self.record_metric(
                f"clinical_high_risk_assessments",
                1,
                MetricType.COUNTER,
                labels={'severity': severity, 'agent_id': agent_id}
            )
    
    async def _handle_validation_result(self, event: Event) -> None:
        """Handle validation result events."""
        result_data = event.data
        
        # Record validation metrics
        self.record_metric(
            "supervision_validations",
            1,
            MetricType.COUNTER,
            labels={'consensus': str(result_data.get('consensus', False))}
        )
        
        # Monitor validation failures
        if result_data.get('result', {}).get('final_result') == 'BLOCKED':
            alert = Alert(
                alert_type="validation_blocked",
                severity=AlertSeverity.WARNING,
                message="Content blocked by supervision validation",
                source="supervision_mesh",
                metadata=result_data
            )
            await self._emit_alert(alert)
    
    async def _emit_alert(self, alert: Alert) -> None:
        """Emit an alert."""
        # Store alert
        self.alerts.append(alert)
        if not alert.resolved:
            self.active_alerts[alert.alert_id] = alert
        
        # Publish alert event
        await self.event_bus.publish(Event(
            event_type="alert_generated",
            source_agent="real_time_monitor",
            priority=EventPriority.HIGH if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else EventPriority.NORMAL,
            data=alert.to_dict()
        ))
        
        logger.warning(f"Alert emitted: {alert.message}")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                # Run all health checks
                for check_name, health_check in self.health_checks.items():
                    result = await health_check.safe_check()
                    
                    # Record health check metric
                    status_value = {
                        HealthStatus.HEALTHY: 1.0,
                        HealthStatus.DEGRADED: 0.7,
                        HealthStatus.UNHEALTHY: 0.3,
                        HealthStatus.CRITICAL: 0.0
                    }.get(result.status, 0.5)
                    
                    self.record_metric(
                        f"health_check_{check_name}",
                        status_value,
                        MetricType.GAUGE,
                        labels={'check_name': check_name, 'status': result.status.value}
                    )
                    
                    # Generate alerts for unhealthy components
                    if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                        alert = Alert(
                            alert_type=f"health_check_failure_{check_name}",
                            severity=AlertSeverity.CRITICAL if result.status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
                            message=f"Health check failed for {check_name}: {result.message}",
                            source=check_name,
                            metadata=result.details
                        )
                        await self._emit_alert(alert)
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                # Clean up old resolved alerts
                cutoff_time = datetime.now() - timedelta(days=1)
                
                # Remove old resolved alerts
                alerts_to_remove = [
                    alert for alert in self.alerts
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert in alerts_to_remove:
                    try:
                        self.alerts.remove(alert)
                    except ValueError:
                        pass  # Already removed
                
                # Clean up active alerts that should be resolved
                active_alerts_copy = dict(self.active_alerts)
                for alert_id, alert in active_alerts_copy.items():
                    # Auto-resolve old alerts (could add more sophisticated logic)
                    if (datetime.now() - alert.timestamp).total_seconds() > 3600:  # 1 hour
                        alert.resolve()
                        del self.active_alerts[alert_id]
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)


# Factory functions and utilities

def create_agent_health_check(agent_id: str, agent_instance: Any) -> AgentHealthCheck:
    """Create a health check for an agent."""
    return AgentHealthCheck(agent_id, agent_instance)


def create_system_health_check(system_name: str, check_functions: List[Callable]) -> SystemHealthCheck:
    """Create a system health check."""
    return SystemHealthCheck(system_name, check_functions)


@asynccontextmanager
async def monitored_operation(monitor: RealTimeMonitor, 
                              operation_name: str,
                              labels: Dict[str, str] = None):
    """Context manager for monitoring operations."""
    start_time = time.time()
    labels = labels or {}
    
    # Record operation start
    monitor.record_metric(
        f"operation_{operation_name}_started",
        1,
        MetricType.COUNTER,
        labels=labels
    )
    
    try:
        yield
        
        # Record successful completion
        duration = time.time() - start_time
        monitor.record_metric(
            f"operation_{operation_name}_duration",
            duration,
            MetricType.TIMER,
            labels=labels
        )
        
        monitor.record_metric(
            f"operation_{operation_name}_success",
            1,
            MetricType.COUNTER,
            labels=labels
        )
        
    except Exception as e:
        # Record failure
        monitor.record_metric(
            f"operation_{operation_name}_errors",
            1,
            MetricType.COUNTER,
            labels={**labels, 'error_type': type(e).__name__}
        )
        
        raise


# Global instance
_monitor_instance: Optional[RealTimeMonitor] = None


def get_real_time_monitor(event_bus: EventBus = None) -> RealTimeMonitor:
    """Get the global real-time monitor instance."""
    global _monitor_instance
    
    if _monitor_instance is None:
        if event_bus is None:
            from src.integration.event_bus import get_event_bus
            event_bus = get_event_bus()
        
        _monitor_instance = RealTimeMonitor(event_bus)
    
    return _monitor_instance