"""
Comprehensive Monitoring and Alerting System
Real-time monitoring of all enterprise components with intelligent alerting
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import psutil
import aiohttp
from collections import deque, defaultdict
import numpy as np
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import slack_sdk
from abc import ABC, abstractmethod
import threading
import pickle

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack" 
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class Metric:
    """System metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    unit: str = ""


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source_component: str
    metric_name: Optional[str]
    current_value: Optional[Union[int, float]]
    threshold_value: Optional[Union[int, float]]
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    is_active: bool = True
    notification_channels: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "equals", "not_equals"
    threshold: Union[int, float]
    severity: AlertSeverity
    evaluation_interval: int = 60  # seconds
    for_duration: int = 300  # seconds - how long condition must be true
    labels: Dict[str, str] = field(default_factory=dict)
    notification_channels: List[AlertChannel] = field(default_factory=list)
    enabled: bool = True
    cooldown_period: int = 900  # 15 minutes between repeated alerts


@dataclass
class HealthCheck:
    """Component health check configuration"""
    component_name: str
    check_type: str  # "http", "tcp", "custom"
    endpoint: Optional[str] = None
    timeout: int = 30
    interval: int = 60
    retries: int = 3
    expected_status: int = 200
    custom_check: Optional[Callable] = None


@dataclass
class ComponentHealth:
    """Component health status"""
    component_name: str
    status: HealthStatus
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, retention_days: int = 30):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_days = retention_days
        self.collection_interval = 30  # seconds
        self.running = False
        self.custom_collectors: Dict[str, Callable] = {}
        
    async def start_collection(self):
        """Start metrics collection"""
        if self.running:
            return
            
        self.running = True
        asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")
        
    async def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        logger.info("Metrics collection stopped")
        
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await self._collect_custom_metrics()
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
                
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        now = datetime.utcnow()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric(Metric(
            name="system_cpu_usage_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE,
            timestamp=now,
            description="CPU usage percentage",
            unit="percent"
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.add_metric(Metric(
            name="system_memory_usage_percent",
            value=memory.percent,
            metric_type=MetricType.GAUGE,
            timestamp=now,
            description="Memory usage percentage",
            unit="percent"
        ))
        
        self.add_metric(Metric(
            name="system_memory_available_bytes",
            value=memory.available,
            metric_type=MetricType.GAUGE,
            timestamp=now,
            description="Available memory in bytes",
            unit="bytes"
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.add_metric(Metric(
            name="system_disk_usage_percent",
            value=(disk.used / disk.total) * 100,
            metric_type=MetricType.GAUGE,
            timestamp=now,
            description="Disk usage percentage",
            unit="percent"
        ))
        
        # Network metrics
        network = psutil.net_io_counters()
        if network:
            self.add_metric(Metric(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                metric_type=MetricType.COUNTER,
                timestamp=now,
                description="Total bytes sent",
                unit="bytes"
            ))
            
            self.add_metric(Metric(
                name="system_network_bytes_received",
                value=network.bytes_recv,
                metric_type=MetricType.COUNTER,
                timestamp=now,
                description="Total bytes received",
                unit="bytes"
            ))
            
    async def _collect_application_metrics(self):
        """Collect application-specific metrics"""
        now = datetime.utcnow()
        
        # Example application metrics
        # These would be populated by the actual application components
        
        # Request metrics (placeholder)
        self.add_metric(Metric(
            name="http_requests_total",
            value=0,  # Would be actual count
            metric_type=MetricType.COUNTER,
            timestamp=now,
            labels={"method": "GET", "status": "200"},
            description="Total HTTP requests",
            unit="requests"
        ))
        
        # Response time metrics (placeholder)
        self.add_metric(Metric(
            name="http_request_duration_seconds",
            value=0.5,  # Would be actual response time
            metric_type=MetricType.HISTOGRAM,
            timestamp=now,
            description="HTTP request duration",
            unit="seconds"
        ))
        
        # Active sessions (placeholder)
        self.add_metric(Metric(
            name="active_therapy_sessions",
            value=0,  # Would be actual count
            metric_type=MetricType.GAUGE,
            timestamp=now,
            description="Active therapy sessions",
            unit="sessions"
        ))
        
    async def _collect_custom_metrics(self):
        """Collect custom application metrics"""
        now = datetime.utcnow()
        
        for name, collector in self.custom_collectors.items():
            try:
                result = await collector() if asyncio.iscoroutinefunction(collector) else collector()
                
                if isinstance(result, Metric):
                    self.add_metric(result)
                elif isinstance(result, list):
                    for metric in result:
                        if isinstance(metric, Metric):
                            self.add_metric(metric)
                            
            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")
                
    def add_metric(self, metric: Metric):
        """Add a metric to the collection"""
        self.metrics[metric.name].append(metric)
        
    def add_custom_collector(self, name: str, collector: Callable):
        """Add custom metrics collector"""
        self.custom_collectors[name] = collector
        
    def get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
        
    def get_metric_history(self, metric_name: str, 
                          time_range: Optional[timedelta] = None) -> List[Metric]:
        """Get metric history within time range"""
        if metric_name not in self.metrics:
            return []
            
        metrics = list(self.metrics[metric_name])
        
        if time_range:
            cutoff_time = datetime.utcnow() - time_range
            metrics = [m for m in metrics if m.timestamp > cutoff_time]
            
        return metrics
        
    async def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
        
        for metric_name, metric_deque in self.metrics.items():
            # Remove old metrics from the front of the deque
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[AlertChannel, Any] = {}
        self.running = False
        self.evaluation_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_monitoring(self):
        """Start alert monitoring"""
        if self.running:
            return
            
        self.running = True
        
        # Start evaluation tasks for each alert rule
        for rule_id, rule in self.alert_rules.items():
            if rule.enabled:
                task = asyncio.create_task(self._evaluate_rule_loop(rule))
                self.evaluation_tasks[rule_id] = task
                
        logger.info("Alert monitoring started")
        
    async def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        
        # Cancel evaluation tasks
        for task in self.evaluation_tasks.values():
            task.cancel()
            
        self.evaluation_tasks.clear()
        logger.info("Alert monitoring stopped")
        
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.rule_id] = rule
        
        # Start evaluation task if monitoring is running
        if self.running and rule.enabled:
            task = asyncio.create_task(self._evaluate_rule_loop(rule))
            self.evaluation_tasks[rule.rule_id] = task
            
        logger.info(f"Added alert rule: {rule.name}")
        
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            
            # Cancel evaluation task
            if rule_id in self.evaluation_tasks:
                self.evaluation_tasks[rule_id].cancel()
                del self.evaluation_tasks[rule_id]
                
            logger.info(f"Removed alert rule: {rule_id}")
            
    async def _evaluate_rule_loop(self, rule: AlertRule):
        """Continuously evaluate an alert rule"""
        condition_start_time = None
        last_alert_time = None
        
        while self.running:
            try:
                # Get latest metric value
                from .comprehensive_monitoring import metrics_collector
                latest_metric = metrics_collector.get_latest_metric(rule.metric_name)
                
                if latest_metric is None:
                    await asyncio.sleep(rule.evaluation_interval)
                    continue
                    
                # Evaluate condition
                condition_met = self._evaluate_condition(
                    latest_metric.value, rule.condition, rule.threshold
                )
                
                if condition_met:
                    if condition_start_time is None:
                        condition_start_time = datetime.utcnow()
                    elif (datetime.utcnow() - condition_start_time).seconds >= rule.for_duration:
                        # Condition has been true for required duration
                        if (last_alert_time is None or 
                            (datetime.utcnow() - last_alert_time).seconds >= rule.cooldown_period):
                            
                            await self._create_alert(rule, latest_metric)
                            last_alert_time = datetime.utcnow()
                            condition_start_time = None
                else:
                    condition_start_time = None
                    
                await asyncio.sleep(rule.evaluation_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
                await asyncio.sleep(rule.evaluation_interval)
                
    def _evaluate_condition(self, value: Union[int, float], 
                          condition: str, threshold: Union[int, float]) -> bool:
        """Evaluate alert condition"""
        if condition == "greater_than":
            return value > threshold
        elif condition == "less_than":
            return value < threshold
        elif condition == "equals":
            return value == threshold
        elif condition == "not_equals":
            return value != threshold
        elif condition == "greater_than_or_equal":
            return value >= threshold
        elif condition == "less_than_or_equal":
            return value <= threshold
        else:
            return False
            
    async def _create_alert(self, rule: AlertRule, metric: Metric):
        """Create and send alert"""
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            title=f"Alert: {rule.name}",
            description=f"{rule.description}. Current value: {metric.value}, Threshold: {rule.threshold}",
            severity=rule.severity,
            source_component=rule.labels.get("component", "unknown"),
            metric_name=rule.metric_name,
            current_value=metric.value,
            threshold_value=rule.threshold,
            notification_channels=rule.notification_channels,
            metadata={"rule_id": rule.rule_id, "metric_timestamp": metric.timestamp.isoformat()}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.warning(f"Alert created: {alert.title}")
        
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in alert.notification_channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_notification(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_notification(alert)
                elif channel == AlertChannel.CONSOLE:
                    await self._send_console_notification(alert)
                    
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification for alert {alert.alert_id}: {e}")
                
    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        if AlertChannel.EMAIL not in self.notification_channels:
            return
            
        email_config = self.notification_channels[AlertChannel.EMAIL]
        
        msg = MimeMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = ', '.join(email_config['to_emails'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
        Alert Details:
        - Severity: {alert.severity.value.upper()}
        - Component: {alert.source_component}
        - Description: {alert.description}
        - Time: {alert.created_at.isoformat()}
        - Alert ID: {alert.alert_id}
        
        Current Value: {alert.current_value}
        Threshold: {alert.threshold_value}
        """
        
        msg.attach(MimeText(body, 'plain'))
        
        # Send email (this would use actual SMTP in production)
        logger.info(f"Email notification sent for alert: {alert.alert_id}")
        
    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        if AlertChannel.SLACK not in self.notification_channels:
            return
            
        slack_config = self.notification_channels[AlertChannel.SLACK]
        
        # Slack color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900", 
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#cc0000",
            AlertSeverity.EMERGENCY: "#990000"
        }
        
        message = {
            "channel": slack_config['channel'],
            "username": "Solace-AI Monitor",
            "attachments": [{
                "color": color_map.get(alert.severity, "#cccccc"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Component", "value": alert.source_component, "short": True},
                    {"title": "Current Value", "value": str(alert.current_value), "short": True},
                    {"title": "Threshold", "value": str(alert.threshold_value), "short": True}
                ],
                "footer": "Solace-AI Monitoring",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        # Send to Slack (would use actual Slack client in production)
        logger.info(f"Slack notification sent for alert: {alert.alert_id}")
        
    async def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        if AlertChannel.WEBHOOK not in self.notification_channels:
            return
            
        webhook_config = self.notification_channels[AlertChannel.WEBHOOK]
        
        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "source_component": alert.source_component,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "created_at": alert.created_at.isoformat(),
            "metadata": alert.metadata
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {})
            ) as response:
                if response.status == 200:
                    logger.info(f"Webhook notification sent for alert: {alert.alert_id}")
                else:
                    logger.error(f"Webhook notification failed: {response.status}")
                    
    async def _send_console_notification(self, alert: Alert):
        """Send console notification"""
        severity_colors = {
            AlertSeverity.INFO: "\033[92m",      # Green
            AlertSeverity.WARNING: "\033[93m",   # Yellow
            AlertSeverity.ERROR: "\033[91m",     # Red
            AlertSeverity.CRITICAL: "\033[95m",  # Magenta
            AlertSeverity.EMERGENCY: "\033[41m"  # Red background
        }
        
        color = severity_colors.get(alert.severity, "")
        reset_color = "\033[0m"
        
        print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
        print(f"Component: {alert.source_component}")
        print(f"Description: {alert.description}")
        print(f"Time: {alert.created_at}")
        print("-" * 50)
        
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            alert.is_active = False
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
            
    def configure_notification_channel(self, channel: AlertChannel, config: Dict[str, Any]):
        """Configure notification channel"""
        self.notification_channels[channel] = config
        logger.info(f"Configured {channel.value} notification channel")


class HealthMonitor:
    """Monitors component health"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.running = False
        self.check_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.running:
            return
            
        self.running = True
        
        # Start health check tasks
        for component_name, health_check in self.health_checks.items():
            task = asyncio.create_task(self._health_check_loop(health_check))
            self.check_tasks[component_name] = task
            
        logger.info("Health monitoring started")
        
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        
        # Cancel health check tasks
        for task in self.check_tasks.values():
            task.cancel()
            
        self.check_tasks.clear()
        logger.info("Health monitoring stopped")
        
    def add_health_check(self, health_check: HealthCheck):
        """Add health check for a component"""
        self.health_checks[health_check.component_name] = health_check
        
        # Initialize component health
        self.component_health[health_check.component_name] = ComponentHealth(
            component_name=health_check.component_name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.utcnow()
        )
        
        # Start health check task if monitoring is running
        if self.running:
            task = asyncio.create_task(self._health_check_loop(health_check))
            self.check_tasks[health_check.component_name] = task
            
        logger.info(f"Added health check for component: {health_check.component_name}")
        
    async def _health_check_loop(self, health_check: HealthCheck):
        """Continuously perform health checks for a component"""
        while self.running:
            try:
                await self._perform_health_check(health_check)
                await asyncio.sleep(health_check.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {health_check.component_name}: {e}")
                await asyncio.sleep(health_check.interval)
                
    async def _perform_health_check(self, health_check: HealthCheck):
        """Perform a single health check"""
        component_health = self.component_health[health_check.component_name]
        
        start_time = time.time()
        check_passed = False
        error_message = None
        
        try:
            if health_check.check_type == "http":
                check_passed = await self._http_health_check(health_check)
            elif health_check.check_type == "tcp":
                check_passed = await self._tcp_health_check(health_check)
            elif health_check.check_type == "custom" and health_check.custom_check:
                check_passed = await self._custom_health_check(health_check)
            else:
                error_message = f"Unknown health check type: {health_check.check_type}"
                
        except Exception as e:
            error_message = str(e)
            
        response_time = time.time() - start_time
        
        # Update component health
        component_health.last_check = datetime.utcnow()
        component_health.response_time = response_time
        component_health.error_message = error_message
        
        if check_passed:
            component_health.status = HealthStatus.HEALTHY
            component_health.consecutive_failures = 0
        else:
            component_health.consecutive_failures += 1
            
            if component_health.consecutive_failures >= health_check.retries:
                if component_health.consecutive_failures >= health_check.retries * 2:
                    component_health.status = HealthStatus.CRITICAL
                else:
                    component_health.status = HealthStatus.UNHEALTHY
            else:
                component_health.status = HealthStatus.DEGRADED
                
        # Calculate uptime percentage (simplified)
        # In production, this would be based on historical data
        if component_health.status == HealthStatus.HEALTHY:
            component_health.uptime_percentage = min(100.0, component_health.uptime_percentage + 0.1)
        else:
            component_health.uptime_percentage = max(0.0, component_health.uptime_percentage - 1.0)
            
    async def _http_health_check(self, health_check: HealthCheck) -> bool:
        """Perform HTTP health check"""
        if not health_check.endpoint:
            return False
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_check.endpoint,
                    timeout=aiohttp.ClientTimeout(total=health_check.timeout)
                ) as response:
                    return response.status == health_check.expected_status
                    
        except Exception:
            return False
            
    async def _tcp_health_check(self, health_check: HealthCheck) -> bool:
        """Perform TCP health check"""
        # Implementation would check TCP connection
        # For now, return True as placeholder
        return True
        
    async def _custom_health_check(self, health_check: HealthCheck) -> bool:
        """Perform custom health check"""
        try:
            if asyncio.iscoroutinefunction(health_check.custom_check):
                return await health_check.custom_check()
            else:
                return health_check.custom_check()
        except Exception:
            return False
            
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a component"""
        return self.component_health.get(component_name)
        
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        if not self.component_health:
            return {"status": "unknown", "components": {}}
            
        healthy_count = sum(1 for h in self.component_health.values() 
                          if h.status == HealthStatus.HEALTHY)
        total_count = len(self.component_health)
        
        if healthy_count == total_count:
            overall_status = "healthy"
        elif healthy_count >= total_count * 0.8:
            overall_status = "degraded"
        elif healthy_count >= total_count * 0.5:
            overall_status = "unhealthy"
        else:
            overall_status = "critical"
            
        return {
            "status": overall_status,
            "healthy_components": healthy_count,
            "total_components": total_count,
            "uptime_percentage": sum(h.uptime_percentage for h in self.component_health.values()) / total_count,
            "components": {
                name: {
                    "status": health.status.value,
                    "last_check": health.last_check.isoformat(),
                    "response_time": health.response_time,
                    "consecutive_failures": health.consecutive_failures,
                    "uptime_percentage": health.uptime_percentage
                }
                for name, health in self.component_health.items()
            }
        }


class ComprehensiveMonitoringSystem:
    """Main monitoring system that coordinates all monitoring components"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = HealthMonitor()
        self.dashboard_data: Dict[str, Any] = {}
        self.running = False
        
    async def initialize(self):
        """Initialize monitoring system"""
        # Setup default alert rules
        await self._setup_default_alert_rules()
        
        # Setup default health checks
        await self._setup_default_health_checks()
        
        # Configure notification channels
        await self._setup_notification_channels()
        
        logger.info("Comprehensive monitoring system initialized")
        
    async def start(self):
        """Start all monitoring components"""
        if self.running:
            return
            
        self.running = True
        
        await self.metrics_collector.start_collection()
        await self.alert_manager.start_monitoring()
        await self.health_monitor.start_monitoring()
        
        # Start dashboard update task
        asyncio.create_task(self._update_dashboard_loop())
        
        logger.info("Comprehensive monitoring system started")
        
    async def stop(self):
        """Stop all monitoring components"""
        self.running = False
        
        await self.metrics_collector.stop_collection()
        await self.alert_manager.stop_monitoring()
        await self.health_monitor.stop_monitoring()
        
        logger.info("Comprehensive monitoring system stopped")
        
    async def _setup_default_alert_rules(self):
        """Setup default alert rules for common issues"""
        
        # High CPU usage alert
        cpu_alert = AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 80%",
            metric_name="system_cpu_usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            notification_channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        self.alert_manager.add_alert_rule(cpu_alert)
        
        # High memory usage alert
        memory_alert = AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",  
            description="Memory usage is above 85%",
            metric_name="system_memory_usage_percent",
            condition="greater_than",
            threshold=85.0,
            severity=AlertSeverity.WARNING,
            notification_channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        self.alert_manager.add_alert_rule(memory_alert)
        
        # Critical memory usage alert
        critical_memory_alert = AlertRule(
            rule_id="critical_memory_usage",
            name="Critical Memory Usage",
            description="Memory usage is above 95%",
            metric_name="system_memory_usage_percent", 
            condition="greater_than",
            threshold=95.0,
            severity=AlertSeverity.CRITICAL,
            notification_channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.CONSOLE],
            for_duration=60  # Alert after 1 minute
        )
        self.alert_manager.add_alert_rule(critical_memory_alert)
        
        # Disk space alert
        disk_alert = AlertRule(
            rule_id="low_disk_space",
            name="Low Disk Space",
            description="Disk usage is above 90%",
            metric_name="system_disk_usage_percent",
            condition="greater_than",
            threshold=90.0,
            severity=AlertSeverity.ERROR,
            notification_channels=[AlertChannel.EMAIL, AlertChannel.SLACK]
        )
        self.alert_manager.add_alert_rule(disk_alert)
        
    async def _setup_default_health_checks(self):
        """Setup default health checks for system components"""
        
        # API health check
        api_check = HealthCheck(
            component_name="api_server",
            check_type="http",
            endpoint="http://localhost:8000/api/health",
            interval=30,
            timeout=10
        )
        self.health_monitor.add_health_check(api_check)
        
        # Database health check (placeholder)
        db_check = HealthCheck(
            component_name="database",
            check_type="custom",
            custom_check=self._database_health_check,
            interval=60,
            timeout=15
        )
        self.health_monitor.add_health_check(db_check)
        
        # Memory system health check
        memory_check = HealthCheck(
            component_name="memory_system",
            check_type="custom",
            custom_check=self._memory_system_health_check,
            interval=120,
            timeout=30
        )
        self.health_monitor.add_health_check(memory_check)
        
    async def _database_health_check(self) -> bool:
        """Custom health check for database"""
        try:
            # In production, this would check actual database connectivity
            # For now, return True as placeholder
            return True
        except Exception:
            return False
            
    async def _memory_system_health_check(self) -> bool:
        """Custom health check for memory system"""
        try:
            # Check if memory system is responding
            # This would test actual memory system in production
            return True
        except Exception:
            return False
            
    async def _setup_notification_channels(self):
        """Setup notification channels"""
        
        # Email configuration (placeholder)
        email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'from_email': 'alerts@solace-ai.com',
            'to_emails': ['ops@solace-ai.com', 'admin@solace-ai.com'],
            'username': 'alerts@solace-ai.com',
            'password': 'password'  # Use environment variable in production
        }
        self.alert_manager.configure_notification_channel(AlertChannel.EMAIL, email_config)
        
        # Slack configuration (placeholder)
        slack_config = {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'channel': '#solace-ai-alerts',  
            'username': 'Solace-AI Monitor'
        }
        self.alert_manager.configure_notification_channel(AlertChannel.SLACK, slack_config)
        
        # Webhook configuration (placeholder)
        webhook_config = {
            'url': 'https://your-webhook-endpoint.com/alerts',
            'headers': {'Authorization': 'Bearer your-token'}
        }
        self.alert_manager.configure_notification_channel(AlertChannel.WEBHOOK, webhook_config)
        
    async def _update_dashboard_loop(self):
        """Update dashboard data periodically"""
        while self.running:
            try:
                await self._update_dashboard_data()
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(30)
                
    async def _update_dashboard_data(self):
        """Update dashboard data"""
        now = datetime.utcnow()
        
        # Get latest metrics
        cpu_metric = self.metrics_collector.get_latest_metric("system_cpu_usage_percent")
        memory_metric = self.metrics_collector.get_latest_metric("system_memory_usage_percent")
        disk_metric = self.metrics_collector.get_latest_metric("system_disk_usage_percent")
        
        # Get health status
        overall_health = self.health_monitor.get_overall_health()
        
        # Get active alerts
        active_alerts = len(self.alert_manager.active_alerts)
        
        self.dashboard_data = {
            "timestamp": now.isoformat(),
            "system_metrics": {
                "cpu_usage": cpu_metric.value if cpu_metric else 0,
                "memory_usage": memory_metric.value if memory_metric else 0,
                "disk_usage": disk_metric.value if disk_metric else 0
            },
            "health_status": overall_health,
            "alerts": {
                "active_count": active_alerts,
                "recent_alerts": [
                    {
                        "title": alert.title,
                        "severity": alert.severity.value,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in sorted(
                        self.alert_manager.alert_history[-10:], 
                        key=lambda a: a.created_at, 
                        reverse=True
                    )
                ]
            },
            "performance": {
                "uptime_percentage": overall_health.get("uptime_percentage", 0),
                "response_time": 0.5  # Would calculate actual response time
            }
        }
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data
        
    async def create_custom_alert(self, rule_config: Dict[str, Any]) -> str:
        """Create custom alert rule"""
        rule_id = str(uuid.uuid4())
        
        rule = AlertRule(
            rule_id=rule_id,
            name=rule_config["name"],
            description=rule_config["description"],
            metric_name=rule_config["metric_name"],
            condition=rule_config["condition"],
            threshold=rule_config["threshold"],
            severity=AlertSeverity(rule_config["severity"]),
            evaluation_interval=rule_config.get("evaluation_interval", 60),
            for_duration=rule_config.get("for_duration", 300),
            notification_channels=[AlertChannel(ch) for ch in rule_config.get("channels", ["email"])]
        )
        
        self.alert_manager.add_alert_rule(rule)
        return rule_id
        
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring system status"""
        return {
            "system_running": self.running,
            "metrics_collector": {
                "running": self.metrics_collector.running,
                "total_metrics": sum(len(deque) for deque in self.metrics_collector.metrics.values()),
                "metric_types": list(self.metrics_collector.metrics.keys())
            },
            "alert_manager": {
                "running": self.alert_manager.running,
                "total_rules": len(self.alert_manager.alert_rules),
                "active_alerts": len(self.alert_manager.active_alerts),
                "total_alerts_history": len(self.alert_manager.alert_history)
            },
            "health_monitor": {
                "running": self.health_monitor.running,
                "total_components": len(self.health_monitor.health_checks),
                "healthy_components": sum(1 for h in self.health_monitor.component_health.values() 
                                        if h.status == HealthStatus.HEALTHY)
            },
            "dashboard_last_updated": self.dashboard_data.get("timestamp", "never")
        }


# Global monitoring system instance
monitoring_system = ComprehensiveMonitoringSystem()

# Convenience function to get the global instance
def get_monitoring_system() -> ComprehensiveMonitoringSystem:
    """Get the global monitoring system instance"""
    return monitoring_system