"""
Clinical Safety and Compliance Reporting System for Solace-AI

This module provides comprehensive clinical safety monitoring and compliance
reporting capabilities including:
- Clinical safety monitoring and alerting
- Compliance audit trails and reporting
- Regulatory compliance validation
- Clinical risk assessment and mitigation
- Patient safety incident tracking
- Clinical quality metrics
- Regulatory reporting automation
- Clinical decision support audit
- Safety event correlation and analysis
- Compliance dashboard and reporting
"""

import asyncio
import time
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import threading
from abc import ABC, abstractmethod
import statistics

from src.utils.logger import get_logger
from src.integration.event_bus import EventBus, Event, EventType, EventPriority

logger = get_logger(__name__)


class RiskLevel(Enum):
    """Clinical risk levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    EXEMPT = "exempt"


class SafetyEventType(Enum):
    """Types of safety events."""
    CLINICAL_ERROR = "clinical_error"
    RISK_ESCALATION = "risk_escalation"
    BOUNDARY_VIOLATION = "boundary_violation"
    INAPPROPRIATE_RESPONSE = "inappropriate_response"
    CRISIS_MISHANDLING = "crisis_mishandling"
    DATA_BREACH = "data_breach"
    SYSTEM_MALFUNCTION = "system_malfunction"
    SUPERVISION_FAILURE = "supervision_failure"


class RegulatoryFramework(Enum):
    """Regulatory frameworks for compliance."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA = "fda"
    NICE = "nice"
    APA_GUIDELINES = "apa_guidelines"
    ISO_27001 = "iso_27001"
    SOC_2 = "soc_2"
    HITECH = "hitech"


@dataclass
class SafetyEvent:
    """Clinical safety event record."""
    
    event_id: str = field(default_factory=lambda: f"safety_{uuid.uuid4().hex}")
    event_type: SafetyEventType = SafetyEventType.CLINICAL_ERROR
    severity: RiskLevel = RiskLevel.LOW
    description: str = ""
    affected_user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    mitigation_actions: List[str] = field(default_factory=list)
    root_cause: Optional[str] = None
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    regulatory_implications: List[RegulatoryFramework] = field(default_factory=list)
    follow_up_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_resolved(self) -> bool:
        """Check if event is resolved."""
        return self.resolved_at is not None
    
    def duration(self) -> timedelta:
        """Get event duration."""
        end_time = self.resolved_at or datetime.now()
        return end_time - self.detected_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'affected_user_id': self.affected_user_id,
            'session_id': self.session_id,
            'agent_id': self.agent_id,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'mitigation_actions': self.mitigation_actions,
            'root_cause': self.root_cause,
            'impact_assessment': self.impact_assessment,
            'regulatory_implications': [r.value for r in self.regulatory_implications],
            'follow_up_required': self.follow_up_required,
            'metadata': self.metadata,
            'duration_minutes': self.duration().total_seconds() / 60
        }


@dataclass
class ComplianceCheck:
    """Compliance check result."""
    
    check_id: str = field(default_factory=lambda: f"compliance_{uuid.uuid4().hex}")
    framework: RegulatoryFramework = RegulatoryFramework.HIPAA
    requirement_id: str = ""
    requirement_description: str = ""
    status: ComplianceStatus = ComplianceStatus.COMPLIANT
    evidence: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    assessed_at: datetime = field(default_factory=datetime.now)
    assessed_by: str = "automated_system"
    next_assessment_due: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'check_id': self.check_id,
            'framework': self.framework.value,
            'requirement_id': self.requirement_id,
            'requirement_description': self.requirement_description,
            'status': self.status.value,
            'evidence': self.evidence,
            'violations': self.violations,
            'recommendations': self.recommendations,
            'assessed_at': self.assessed_at.isoformat(),
            'assessed_by': self.assessed_by,
            'next_assessment_due': self.next_assessment_due.isoformat() if self.next_assessment_due else None,
            'metadata': self.metadata
        }


@dataclass
class ClinicalMetric:
    """Clinical quality metric."""
    
    metric_id: str
    metric_name: str
    value: float
    target_value: Optional[float] = None
    unit: str = ""
    category: str = "quality"
    risk_level: RiskLevel = RiskLevel.LOW
    measurement_period: str = "daily"
    measured_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_within_target(self) -> bool:
        """Check if metric is within target range."""
        if self.target_value is None:
            return True
        return abs(self.value - self.target_value) / max(self.target_value, 0.001) <= 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_id': self.metric_id,
            'metric_name': self.metric_name,
            'value': self.value,
            'target_value': self.target_value,
            'unit': self.unit,
            'category': self.category,
            'risk_level': self.risk_level.value,
            'measurement_period': self.measurement_period,
            'measured_at': self.measured_at.isoformat(),
            'within_target': self.is_within_target(),
            'metadata': self.metadata
        }


@dataclass
class AuditTrail:
    """Audit trail record."""
    
    record_id: str = field(default_factory=lambda: f"audit_{uuid.uuid4().hex}")
    action: str = ""
    actor_type: str = "system"  # system, user, agent
    actor_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'record_id': self.record_id,
            'action': self.action,
            'actor_type': self.actor_type,
            'actor_id': self.actor_id,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'outcome': self.outcome,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    
    report_id: str = field(default_factory=lambda: f"report_{uuid.uuid4().hex}")
    report_type: str = "compliance_summary"
    framework: RegulatoryFramework = RegulatoryFramework.HIPAA
    reporting_period_start: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    reporting_period_end: datetime = field(default_factory=datetime.now)
    overall_compliance_score: float = 0.0
    compliance_checks: List[ComplianceCheck] = field(default_factory=list)
    safety_events: List[SafetyEvent] = field(default_factory=list)
    clinical_metrics: List[ClinicalMetric] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = "automated_system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'report_id': self.report_id,
            'report_type': self.report_type,
            'framework': self.framework.value,
            'reporting_period_start': self.reporting_period_start.isoformat(),
            'reporting_period_end': self.reporting_period_end.isoformat(),
            'overall_compliance_score': self.overall_compliance_score,
            'compliance_checks': [check.to_dict() for check in self.compliance_checks],
            'safety_events': [event.to_dict() for event in self.safety_events],
            'clinical_metrics': [metric.to_dict() for metric in self.clinical_metrics],
            'recommendations': self.recommendations,
            'action_items': self.action_items,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by
        }


class ComplianceRule(ABC):
    """Abstract base class for compliance rules."""
    
    def __init__(self, rule_id: str, framework: RegulatoryFramework, requirement_id: str):
        self.rule_id = rule_id
        self.framework = framework
        self.requirement_id = requirement_id
    
    @abstractmethod
    async def evaluate(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Evaluate compliance rule."""
        pass


class HIPAAAccessControlRule(ComplianceRule):
    """HIPAA access control compliance rule."""
    
    def __init__(self):
        super().__init__(
            "hipaa_access_control",
            RegulatoryFramework.HIPAA,
            "164.312(a)(1)"
        )
    
    async def evaluate(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Evaluate HIPAA access control compliance."""
        
        # Check access control measures
        audit_records = context.get('audit_records', [])
        access_violations = []
        evidence = []
        
        # Check for unauthorized access attempts
        unauthorized_attempts = [
            record for record in audit_records
            if record.get('outcome') == 'unauthorized_access'
        ]
        
        if unauthorized_attempts:
            access_violations.append(f"Found {len(unauthorized_attempts)} unauthorized access attempts")
        else:
            evidence.append("No unauthorized access attempts detected")
        
        # Check for proper authentication
        authenticated_sessions = [
            record for record in audit_records
            if record.get('action') == 'login' and record.get('outcome') == 'success'
        ]
        
        if authenticated_sessions:
            evidence.append(f"Proper authentication verified for {len(authenticated_sessions)} sessions")
        
        # Determine compliance status
        if access_violations:
            status = ComplianceStatus.NON_COMPLIANT
            recommendations = [
                "Review access control procedures",
                "Implement stronger authentication measures",
                "Monitor access attempts more closely"
            ]
        else:
            status = ComplianceStatus.COMPLIANT
            recommendations = ["Continue monitoring access controls"]
        
        return ComplianceCheck(
            framework=self.framework,
            requirement_id=self.requirement_id,
            requirement_description="Implement access control procedures",
            status=status,
            evidence=evidence,
            violations=access_violations,
            recommendations=recommendations,
            next_assessment_due=datetime.now() + timedelta(days=30)
        )


class HIPAADataIntegrityRule(ComplianceRule):
    """HIPAA data integrity compliance rule."""
    
    def __init__(self):
        super().__init__(
            "hipaa_data_integrity",
            RegulatoryFramework.HIPAA,
            "164.312(c)(1)"
        )
    
    async def evaluate(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Evaluate HIPAA data integrity compliance."""
        
        data_integrity_checks = context.get('data_integrity_checks', [])
        evidence = []
        violations = []
        
        # Check for data corruption
        corrupted_records = [
            check for check in data_integrity_checks
            if not check.get('is_consistent', True)
        ]
        
        if corrupted_records:
            violations.append(f"Found {len(corrupted_records)} data integrity issues")
        else:
            evidence.append("Data integrity checks passed")
        
        # Check for backup procedures
        backup_status = context.get('backup_status', {})
        if backup_status.get('last_backup'):
            evidence.append(f"Last backup completed: {backup_status['last_backup']}")
        else:
            violations.append("No recent backup found")
        
        # Determine compliance status
        status = ComplianceStatus.COMPLIANT if not violations else ComplianceStatus.NON_COMPLIANT
        
        recommendations = []
        if violations:
            recommendations.extend([
                "Implement data integrity monitoring",
                "Ensure regular backups are performed",
                "Review data handling procedures"
            ])
        else:
            recommendations.append("Continue monitoring data integrity")
        
        return ComplianceCheck(
            framework=self.framework,
            requirement_id=self.requirement_id,
            requirement_description="Protect against improper alteration or destruction of PHI",
            status=status,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )


class ClinicalSafetyRule(ComplianceRule):
    """Clinical safety compliance rule."""
    
    def __init__(self):
        super().__init__(
            "clinical_safety",
            RegulatoryFramework.APA_GUIDELINES,
            "ETHICAL_GUIDELINES"
        )
    
    async def evaluate(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Evaluate clinical safety compliance."""
        
        safety_events = context.get('safety_events', [])
        clinical_assessments = context.get('clinical_assessments', [])
        
        evidence = []
        violations = []
        
        # Check for high-risk safety events
        high_risk_events = [
            event for event in safety_events
            if event.get('severity') in ['high', 'critical']
        ]
        
        if high_risk_events:
            violations.append(f"Found {len(high_risk_events)} high-risk safety events")
        else:
            evidence.append("No high-risk safety events detected")
        
        # Check clinical assessment quality
        poor_quality_assessments = [
            assessment for assessment in clinical_assessments
            if assessment.get('quality_score', 100) < 70
        ]
        
        if poor_quality_assessments:
            violations.append(f"Found {len(poor_quality_assessments)} poor quality clinical assessments")
        else:
            evidence.append("Clinical assessment quality within acceptable range")
        
        # Check for crisis intervention compliance
        crisis_events = [
            event for event in safety_events
            if event.get('event_type') == 'crisis_mishandling'
        ]
        
        if crisis_events:
            violations.append(f"Found {len(crisis_events)} crisis mishandling events")
        else:
            evidence.append("Crisis intervention protocols followed")
        
        # Determine compliance status
        if high_risk_events or crisis_events:
            status = ComplianceStatus.NON_COMPLIANT
        elif poor_quality_assessments:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.COMPLIANT
        
        recommendations = []
        if violations:
            recommendations.extend([
                "Review clinical safety protocols",
                "Provide additional training on crisis intervention",
                "Implement enhanced quality monitoring"
            ])
        else:
            recommendations.append("Continue monitoring clinical safety")
        
        return ComplianceCheck(
            framework=self.framework,
            requirement_id=self.requirement_id,
            requirement_description="Ensure clinical safety and ethical practice",
            status=status,
            evidence=evidence,
            violations=violations,
            recommendations=recommendations
        )


class SafetyMonitor:
    """Clinical safety event monitoring system."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.safety_events: Dict[str, SafetyEvent] = {}
        self.event_patterns: Dict[str, List[SafetyEvent]] = defaultdict(list)
        self.escalation_thresholds = {
            RiskLevel.CRITICAL: 1,    # Immediate escalation
            RiskLevel.HIGH: 3,        # Escalate after 3 events
            RiskLevel.MODERATE: 10,   # Escalate after 10 events
            RiskLevel.LOW: 50         # Escalate after 50 events
        }
        
        # Background monitoring
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("SafetyMonitor initialized")
    
    async def start(self) -> None:
        """Start safety monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("SafetyMonitor started")
    
    async def stop(self) -> None:
        """Stop safety monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("SafetyMonitor stopped")
    
    async def report_safety_event(self, 
                                 event_type: SafetyEventType,
                                 severity: RiskLevel,
                                 description: str,
                                 affected_user_id: Optional[str] = None,
                                 session_id: Optional[str] = None,
                                 agent_id: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Report a safety event."""
        
        safety_event = SafetyEvent(
            event_type=event_type,
            severity=severity,
            description=description,
            affected_user_id=affected_user_id,
            session_id=session_id,
            agent_id=agent_id,
            metadata=metadata or {}
        )
        
        # Store event
        self.safety_events[safety_event.event_id] = safety_event
        
        # Add to pattern tracking
        pattern_key = f"{event_type.value}_{agent_id or 'system'}"
        self.event_patterns[pattern_key].append(safety_event)
        
        # Check for escalation
        await self._check_escalation(safety_event, pattern_key)
        
        # Publish safety event
        await self.event_bus.publish(Event(
            event_type="safety_event_reported",
            source_agent="safety_monitor",
            priority=self._get_event_priority(severity),
            data=safety_event.to_dict()
        ))
        
        logger.warning(f"Safety event reported: {safety_event.event_id} - {description}")
        return safety_event.event_id
    
    async def resolve_safety_event(self, 
                                  event_id: str,
                                  mitigation_actions: List[str],
                                  root_cause: Optional[str] = None) -> bool:
        """Resolve a safety event."""
        
        if event_id not in self.safety_events:
            return False
        
        safety_event = self.safety_events[event_id]
        
        if safety_event.is_resolved():
            return True
        
        safety_event.resolved_at = datetime.now()
        safety_event.mitigation_actions = mitigation_actions
        safety_event.root_cause = root_cause
        
        # Publish resolution event
        await self.event_bus.publish(Event(
            event_type="safety_event_resolved",
            source_agent="safety_monitor",
            priority=EventPriority.NORMAL,
            data={
                'event_id': event_id,
                'resolution_time_minutes': safety_event.duration().total_seconds() / 60,
                'mitigation_actions': mitigation_actions,
                'root_cause': root_cause
            }
        ))
        
        logger.info(f"Safety event resolved: {event_id}")
        return True
    
    def get_safety_events(self, 
                         severity: Optional[RiskLevel] = None,
                         event_type: Optional[SafetyEventType] = None,
                         resolved: Optional[bool] = None,
                         last_n_days: int = 30) -> List[SafetyEvent]:
        """Get safety events with filtering."""
        
        cutoff_time = datetime.now() - timedelta(days=last_n_days)
        
        filtered_events = []
        for event in self.safety_events.values():
            if event.detected_at < cutoff_time:
                continue
            
            if severity and event.severity != severity:
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            if resolved is not None and event.is_resolved() != resolved:
                continue
            
            filtered_events.append(event)
        
        return sorted(filtered_events, key=lambda e: e.detected_at, reverse=True)
    
    def get_safety_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get safety event statistics."""
        
        events = self.get_safety_events(last_n_days=days)
        
        if not events:
            return {
                'total_events': 0,
                'events_by_severity': {},
                'events_by_type': {},
                'resolution_rate': 0.0,
                'average_resolution_time_hours': 0.0
            }
        
        # Count by severity
        severity_counts = defaultdict(int)
        for event in events:
            severity_counts[event.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for event in events:
            type_counts[event.event_type.value] += 1
        
        # Calculate resolution metrics
        resolved_events = [e for e in events if e.is_resolved()]
        resolution_rate = len(resolved_events) / len(events) * 100
        
        if resolved_events:
            resolution_times = [e.duration().total_seconds() / 3600 for e in resolved_events]
            avg_resolution_time = statistics.mean(resolution_times)
        else:
            avg_resolution_time = 0.0
        
        return {
            'total_events': len(events),
            'events_by_severity': dict(severity_counts),
            'events_by_type': dict(type_counts),
            'resolution_rate': resolution_rate,
            'average_resolution_time_hours': avg_resolution_time,
            'unresolved_events': len(events) - len(resolved_events)
        }
    
    async def _check_escalation(self, event: SafetyEvent, pattern_key: str) -> None:
        """Check if event should trigger escalation."""
        
        # Immediate escalation for critical events
        if event.severity == RiskLevel.CRITICAL:
            await self._escalate_event(event, "Critical severity event")
            return
        
        # Pattern-based escalation
        recent_events = [
            e for e in self.event_patterns[pattern_key]
            if (datetime.now() - e.detected_at).total_seconds() < 3600  # Last hour
        ]
        
        threshold = self.escalation_thresholds.get(event.severity, 10)
        
        if len(recent_events) >= threshold:
            await self._escalate_event(
                event, 
                f"Pattern detected: {len(recent_events)} {event.event_type.value} events in last hour"
            )
    
    async def _escalate_event(self, event: SafetyEvent, reason: str) -> None:
        """Escalate a safety event."""
        
        # Publish escalation event
        await self.event_bus.publish(Event(
            event_type="safety_escalation",
            source_agent="safety_monitor",
            priority=EventPriority.CRITICAL,
            data={
                'event_id': event.event_id,
                'escalation_reason': reason,
                'event_details': event.to_dict()
            }
        ))
        
        logger.critical(f"Safety event escalated: {event.event_id} - {reason}")
    
    def _get_event_priority(self, severity: RiskLevel) -> EventPriority:
        """Get event priority based on severity."""
        priority_mapping = {
            RiskLevel.CRITICAL: EventPriority.CRITICAL,
            RiskLevel.HIGH: EventPriority.HIGH,
            RiskLevel.MODERATE: EventPriority.NORMAL,
            RiskLevel.LOW: EventPriority.NORMAL,
            RiskLevel.MINIMAL: EventPriority.LOW
        }
        return priority_mapping.get(severity, EventPriority.NORMAL)
    
    def _setup_event_subscriptions(self) -> None:
        """Setup event subscriptions for safety monitoring."""
        
        # Monitor agent errors for safety implications
        self.event_bus.subscribe(
            EventType.AGENT_ERROR,
            self._handle_agent_error,
            agent_id="safety_monitor"
        )
        
        # Monitor clinical assessments for safety issues
        self.event_bus.subscribe(
            EventType.CLINICAL_ASSESSMENT,
            self._handle_clinical_assessment,
            agent_id="safety_monitor"
        )
        
        # Monitor validation results for safety blocks
        self.event_bus.subscribe(
            EventType.VALIDATION_RESULT,
            self._handle_validation_result,
            agent_id="safety_monitor"
        )
        
        logger.info("Safety monitoring event subscriptions configured")
    
    async def _handle_agent_error(self, event: Event) -> None:
        """Handle agent error events for safety monitoring."""
        try:
            error_data = event.data
            severity_str = error_data.get('severity', 'low')
            
            # Map severity
            severity_mapping = {
                'low': RiskLevel.LOW,
                'medium': RiskLevel.MODERATE,
                'high': RiskLevel.HIGH,
                'critical': RiskLevel.CRITICAL
            }
            severity = severity_mapping.get(severity_str, RiskLevel.LOW)
            
            # Report safety event if significant
            if severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                await self.report_safety_event(
                    event_type=SafetyEventType.SYSTEM_MALFUNCTION,
                    severity=severity,
                    description=f"Agent error: {error_data.get('error', 'Unknown error')}",
                    agent_id=event.source_agent,
                    session_id=event.session_id,
                    metadata=error_data
                )
        
        except Exception as e:
            logger.error(f"Error handling agent error event: {e}")
    
    async def _handle_clinical_assessment(self, event: Event) -> None:
        """Handle clinical assessment events for safety monitoring."""
        try:
            assessment_data = event.data
            diagnosis_result = assessment_data.get('diagnosis_result', {})
            
            # Check for high-risk assessments
            severity = diagnosis_result.get('severity', 'mild')
            if severity == 'severe':
                # Check if appropriate safety measures were taken
                safety_validations = diagnosis_result.get('supervision_validated', False)
                
                if not safety_validations:
                    await self.report_safety_event(
                        event_type=SafetyEventType.SUPERVISION_FAILURE,
                        severity=RiskLevel.HIGH,
                        description="High-risk assessment without proper supervision validation",
                        affected_user_id=event.user_id,
                        session_id=event.session_id,
                        agent_id=event.source_agent,
                        metadata={'diagnosis_result': diagnosis_result}
                    )
        
        except Exception as e:
            logger.error(f"Error handling clinical assessment event: {e}")
    
    async def _handle_validation_result(self, event: Event) -> None:
        """Handle validation result events for safety monitoring."""
        try:
            result_data = event.data
            result = result_data.get('result', {})
            
            # Check for blocked content due to safety concerns
            if result.get('final_result') == 'BLOCKED':
                await self.report_safety_event(
                    event_type=SafetyEventType.INAPPROPRIATE_RESPONSE,
                    severity=RiskLevel.MODERATE,
                    description="Content blocked by safety validation",
                    metadata=result_data
                )
        
        except Exception as e:
            logger.error(f"Error handling validation result event: {e}")
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for pattern detection."""
        
        while self._running:
            try:
                # Clean up old events from pattern tracking
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for pattern_key, events in self.event_patterns.items():
                    self.event_patterns[pattern_key] = [
                        e for e in events if e.detected_at >= cutoff_time
                    ]
                
                # Check for unresolved high-severity events
                unresolved_high_severity = [
                    event for event in self.safety_events.values()
                    if not event.is_resolved() and 
                    event.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] and
                    event.duration().total_seconds() > 3600  # Unresolved for > 1 hour
                ]
                
                for event in unresolved_high_severity:
                    await self._escalate_event(
                        event,
                        f"High-severity event unresolved for {event.duration().total_seconds() / 3600:.1f} hours"
                    )
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in safety monitoring loop: {e}")
                await asyncio.sleep(300)


class ComplianceEngine:
    """Compliance monitoring and reporting engine."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.compliance_checks: Dict[str, ComplianceCheck] = {}
        self.audit_trail: List[AuditTrail] = []
        self.clinical_metrics: Dict[str, ClinicalMetric] = {}
        
        # Configuration
        self.assessment_interval_hours = 24
        self.audit_retention_days = 90
        
        # Background tasks
        self._running = False
        self._assessment_task: Optional[asyncio.Task] = None
        self._audit_cleanup_task: Optional[asyncio.Task] = None
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("ComplianceEngine initialized")
    
    async def start(self) -> None:
        """Start compliance monitoring."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._assessment_task = asyncio.create_task(self._assessment_loop())
        self._audit_cleanup_task = asyncio.create_task(self._audit_cleanup_loop())
        
        logger.info("ComplianceEngine started")
    
    async def stop(self) -> None:
        """Stop compliance monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop background tasks
        tasks = [self._assessment_task, self._audit_cleanup_task]
        for task in tasks:
            if task:
                task.cancel()
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("ComplianceEngine stopped")
    
    def add_compliance_rule(self, rule: ComplianceRule) -> None:
        """Add a compliance rule."""
        self.compliance_rules[rule.rule_id] = rule
        logger.info(f"Added compliance rule: {rule.rule_id}")
    
    async def assess_compliance(self, 
                              framework: Optional[RegulatoryFramework] = None,
                              force_assessment: bool = False) -> Dict[str, ComplianceCheck]:
        """Assess compliance against rules."""
        
        rules_to_assess = []
        
        for rule in self.compliance_rules.values():
            if framework and rule.framework != framework:
                continue
            
            # Check if assessment is due
            existing_check = self.compliance_checks.get(rule.rule_id)
            if not force_assessment and existing_check:
                if existing_check.next_assessment_due and \
                   existing_check.next_assessment_due > datetime.now():
                    continue
            
            rules_to_assess.append(rule)
        
        # Prepare assessment context
        context = await self._prepare_assessment_context()
        
        # Run assessments
        assessment_results = {}
        
        for rule in rules_to_assess:
            try:
                check_result = await rule.evaluate(context)
                self.compliance_checks[rule.rule_id] = check_result
                assessment_results[rule.rule_id] = check_result
                
                # Publish compliance event
                await self.event_bus.publish(Event(
                    event_type="compliance_check_completed",
                    source_agent="compliance_engine",
                    priority=EventPriority.HIGH if check_result.status == ComplianceStatus.NON_COMPLIANT else EventPriority.NORMAL,
                    data=check_result.to_dict()
                ))
                
            except Exception as e:
                logger.error(f"Error assessing compliance rule {rule.rule_id}: {e}")
        
        logger.info(f"Completed compliance assessment for {len(assessment_results)} rules")
        return assessment_results
    
    async def record_audit_event(self,
                                action: str,
                                actor_type: str,
                                actor_id: str,
                                resource_type: str,
                                resource_id: str,
                                details: Optional[Dict[str, Any]] = None,
                                outcome: str = "success",
                                session_id: Optional[str] = None) -> str:
        """Record an audit event."""
        
        audit_record = AuditTrail(
            action=action,
            actor_type=actor_type,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            outcome=outcome,
            session_id=session_id
        )
        
        self.audit_trail.append(audit_record)
        
        # Publish audit event
        await self.event_bus.publish(Event(
            event_type="audit_event_recorded",
            source_agent="compliance_engine",
            priority=EventPriority.LOW,
            data=audit_record.to_dict()
        ))
        
        return audit_record.record_id
    
    def record_clinical_metric(self, metric: ClinicalMetric) -> None:
        """Record a clinical quality metric."""
        self.clinical_metrics[metric.metric_id] = metric
        
        # Check for concerning metrics
        if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(f"High-risk clinical metric recorded: {metric.metric_name} = {metric.value}")
    
    async def generate_compliance_report(self,
                                       framework: RegulatoryFramework,
                                       start_date: Optional[datetime] = None,
                                       end_date: Optional[datetime] = None) -> ComplianceReport:
        """Generate comprehensive compliance report."""
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        # Get relevant compliance checks
        relevant_checks = [
            check for check in self.compliance_checks.values()
            if check.framework == framework and 
            start_date <= check.assessed_at <= end_date
        ]
        
        # Get relevant safety events (assuming we have access to safety monitor)
        safety_events = []  # Would be populated from safety monitor
        
        # Get relevant clinical metrics
        relevant_metrics = [
            metric for metric in self.clinical_metrics.values()
            if start_date <= metric.measured_at <= end_date
        ]
        
        # Calculate overall compliance score
        if relevant_checks:
            compliant_checks = len([c for c in relevant_checks if c.status == ComplianceStatus.COMPLIANT])
            overall_score = (compliant_checks / len(relevant_checks)) * 100
        else:
            overall_score = 0.0
        
        # Generate recommendations
        recommendations = []
        action_items = []
        
        non_compliant_checks = [c for c in relevant_checks if c.status == ComplianceStatus.NON_COMPLIANT]
        for check in non_compliant_checks:
            recommendations.extend(check.recommendations)
            action_items.extend([f"Address {check.requirement_id}: {violation}" for violation in check.violations])
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        action_items = list(set(action_items))
        
        report = ComplianceReport(
            framework=framework,
            reporting_period_start=start_date,
            reporting_period_end=end_date,
            overall_compliance_score=overall_score,
            compliance_checks=relevant_checks,
            safety_events=safety_events,
            clinical_metrics=relevant_metrics,
            recommendations=recommendations,
            action_items=action_items
        )
        
        # Publish report generated event
        await self.event_bus.publish(Event(
            event_type="compliance_report_generated",
            source_agent="compliance_engine",
            priority=EventPriority.NORMAL,
            data={
                'report_id': report.report_id,
                'framework': framework.value,
                'compliance_score': overall_score,
                'recommendations_count': len(recommendations),
                'action_items_count': len(action_items)
            }
        ))
        
        logger.info(f"Generated compliance report {report.report_id} for {framework.value}")
        return report
    
    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard."""
        
        # Overall compliance status
        total_checks = len(self.compliance_checks)
        if total_checks > 0:
            compliant_checks = len([c for c in self.compliance_checks.values() if c.status == ComplianceStatus.COMPLIANT])
            compliance_rate = (compliant_checks / total_checks) * 100
        else:
            compliance_rate = 0.0
        
        # Compliance by framework
        framework_compliance = defaultdict(lambda: {'total': 0, 'compliant': 0})
        for check in self.compliance_checks.values():
            framework_compliance[check.framework.value]['total'] += 1
            if check.status == ComplianceStatus.COMPLIANT:
                framework_compliance[check.framework.value]['compliant'] += 1
        
        # Calculate rates
        for framework_data in framework_compliance.values():
            if framework_data['total'] > 0:
                framework_data['rate'] = (framework_data['compliant'] / framework_data['total']) * 100
            else:
                framework_data['rate'] = 0.0
        
        # Recent audit activity
        recent_audits = len([
            audit for audit in self.audit_trail
            if (datetime.now() - audit.timestamp).total_seconds() < 86400  # Last 24 hours
        ])
        
        # Clinical metrics summary
        high_risk_metrics = len([
            metric for metric in self.clinical_metrics.values()
            if metric.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ])
        
        return {
            'overall_compliance_rate': compliance_rate,
            'total_compliance_checks': total_checks,
            'compliance_by_framework': dict(framework_compliance),
            'recent_audit_events': recent_audits,
            'high_risk_clinical_metrics': high_risk_metrics,
            'last_assessment': max(
                (check.assessed_at for check in self.compliance_checks.values()),
                default=None
            ).isoformat() if self.compliance_checks else None
        }
    
    async def _prepare_assessment_context(self) -> Dict[str, Any]:
        """Prepare context for compliance assessment."""
        
        # Convert audit trail to dict format for rules
        audit_records = [audit.to_dict() for audit in self.audit_trail]
        
        # Get recent clinical metrics
        clinical_metrics = [metric.to_dict() for metric in self.clinical_metrics.values()]
        
        # Would integrate with other systems for additional context
        context = {
            'audit_records': audit_records,
            'clinical_metrics': clinical_metrics,
            'assessment_timestamp': datetime.now().isoformat(),
            'data_integrity_checks': [],  # Would be populated from data reliability system
            'backup_status': {},  # Would be populated from backup system
            'safety_events': [],  # Would be populated from safety monitor
            'clinical_assessments': []  # Would be populated from clinical data
        }
        
        return context
    
    def _initialize_default_rules(self) -> None:
        """Initialize default compliance rules."""
        
        # Add HIPAA rules
        self.add_compliance_rule(HIPAAAccessControlRule())
        self.add_compliance_rule(HIPAADataIntegrityRule())
        
        # Add clinical safety rule
        self.add_compliance_rule(ClinicalSafetyRule())
        
        logger.info("Default compliance rules initialized")
    
    async def _assessment_loop(self) -> None:
        """Background loop for periodic compliance assessment."""
        
        while self._running:
            try:
                # Run compliance assessment
                await self.assess_compliance()
                
                # Wait for next assessment
                await asyncio.sleep(self.assessment_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in compliance assessment loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def _audit_cleanup_loop(self) -> None:
        """Background loop for audit trail cleanup."""
        
        while self._running:
            try:
                # Clean up old audit records
                cutoff_time = datetime.now() - timedelta(days=self.audit_retention_days)
                
                old_count = len(self.audit_trail)
                self.audit_trail = [
                    audit for audit in self.audit_trail
                    if audit.timestamp >= cutoff_time
                ]
                
                cleaned_count = old_count - len(self.audit_trail)
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old audit records")
                
                # Wait 24 hours before next cleanup
                await asyncio.sleep(86400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audit cleanup loop: {e}")
                await asyncio.sleep(86400)


class ClinicalComplianceSystem:
    """
    Comprehensive clinical safety and compliance system.
    Integrates safety monitoring, compliance assessment, and reporting.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.safety_monitor = SafetyMonitor(event_bus)
        self.compliance_engine = ComplianceEngine(event_bus)
        
        # Configuration
        self.integration_enabled = True
        
        logger.info("ClinicalComplianceSystem initialized")
    
    async def start(self) -> None:
        """Start the clinical compliance system."""
        
        # Start components
        await self.safety_monitor.start()
        await self.compliance_engine.start()
        
        # Setup integration if enabled
        if self.integration_enabled:
            self._setup_component_integration()
        
        # Publish startup event
        await self.event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source_agent="clinical_compliance_system",
            data={'status': 'started', 'components': ['safety_monitor', 'compliance_engine']}
        ))
        
        logger.info("ClinicalComplianceSystem started")
    
    async def stop(self) -> None:
        """Stop the clinical compliance system."""
        
        # Stop components
        await self.safety_monitor.stop()
        await self.compliance_engine.stop()
        
        logger.info("ClinicalComplianceSystem stopped")
    
    async def generate_comprehensive_report(self,
                                          framework: RegulatoryFramework = RegulatoryFramework.HIPAA,
                                          days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive compliance and safety report."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate compliance report
        compliance_report = await self.compliance_engine.generate_compliance_report(
            framework, start_date, end_date
        )
        
        # Get safety statistics
        safety_stats = self.safety_monitor.get_safety_statistics(days)
        
        # Get safety events
        safety_events = self.safety_monitor.get_safety_events(last_n_days=days)
        
        # Get dashboard data
        compliance_dashboard = self.compliance_engine.get_compliance_dashboard_data()
        
        comprehensive_report = {
            'report_metadata': {
                'report_id': f"comprehensive_{uuid.uuid4().hex[:8]}",
                'generated_at': datetime.now().isoformat(),
                'reporting_period_days': days,
                'framework': framework.value
            },
            'compliance_report': compliance_report.to_dict(),
            'safety_statistics': safety_stats,
            'safety_events_summary': {
                'total_events': len(safety_events),
                'unresolved_events': len([e for e in safety_events if not e.is_resolved()]),
                'high_risk_events': len([e for e in safety_events if e.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
                'events_by_type': {
                    event_type.value: len([e for e in safety_events if e.event_type == event_type])
                    for event_type in SafetyEventType
                }
            },
            'compliance_dashboard': compliance_dashboard,
            'recommendations': self._generate_integrated_recommendations(
                compliance_report, safety_stats, safety_events
            )
        }
        
        return comprehensive_report
    
    def _setup_component_integration(self) -> None:
        """Setup integration between safety monitor and compliance engine."""
        
        # Subscribe compliance engine to safety events
        self.event_bus.subscribe(
            "safety_event_reported",
            self._handle_safety_event_for_compliance,
            agent_id="clinical_compliance_integration"
        )
        
        # Subscribe safety monitor to compliance violations
        self.event_bus.subscribe(
            "compliance_check_completed",
            self._handle_compliance_result_for_safety,
            agent_id="clinical_compliance_integration"
        )
        
        logger.info("Component integration configured")
    
    async def _handle_safety_event_for_compliance(self, event: Event) -> None:
        """Handle safety events for compliance tracking."""
        try:
            safety_event_data = event.data
            
            # Record audit event for safety incident
            await self.compliance_engine.record_audit_event(
                action="safety_incident",
                actor_type="system",
                actor_id=safety_event_data.get('agent_id', 'unknown'),
                resource_type="safety_event",
                resource_id=safety_event_data['event_id'],
                details=safety_event_data,
                outcome="safety_event_reported",
                session_id=safety_event_data.get('session_id')
            )
            
            # Record clinical metric if applicable
            severity = safety_event_data.get('severity', 'low')
            severity_score = {
                'minimal': 1, 'low': 2, 'moderate': 5, 'high': 8, 'critical': 10
            }.get(severity, 2)
            
            clinical_metric = ClinicalMetric(
                metric_id=f"safety_severity_{safety_event_data['event_id']}",
                metric_name="safety_event_severity",
                value=severity_score,
                target_value=3,  # Target is to keep severity low
                unit="severity_score",
                category="safety",
                risk_level=RiskLevel(severity),
                measurement_period="incident"
            )
            
            self.compliance_engine.record_clinical_metric(clinical_metric)
            
        except Exception as e:
            logger.error(f"Error handling safety event for compliance: {e}")
    
    async def _handle_compliance_result_for_safety(self, event: Event) -> None:
        """Handle compliance results for safety monitoring."""
        try:
            compliance_data = event.data
            
            # Check for compliance violations that indicate safety risks
            if compliance_data.get('status') == 'non_compliant':
                violations = compliance_data.get('violations', [])
                
                for violation in violations:
                    # Determine if violation indicates a safety risk
                    if any(keyword in violation.lower() for keyword in ['access', 'breach', 'unauthorized', 'integrity']):
                        await self.safety_monitor.report_safety_event(
                            event_type=SafetyEventType.DATA_BREACH,
                            severity=RiskLevel.HIGH,
                            description=f"Compliance violation with safety implications: {violation}",
                            metadata={'compliance_check_id': compliance_data['check_id']}
                        )
        
        except Exception as e:
            logger.error(f"Error handling compliance result for safety: {e}")
    
    def _generate_integrated_recommendations(self,
                                           compliance_report: ComplianceReport,
                                           safety_stats: Dict[str, Any],
                                           safety_events: List[SafetyEvent]) -> List[str]:
        """Generate integrated recommendations based on compliance and safety data."""
        
        recommendations = []
        
        # Add compliance recommendations
        recommendations.extend(compliance_report.recommendations)
        
        # Add safety-based recommendations
        if safety_stats['total_events'] > 10:
            recommendations.append("High number of safety events - review system processes")
        
        if safety_stats['resolution_rate'] < 80:
            recommendations.append("Low safety event resolution rate - improve incident response")
        
        # Pattern-based recommendations
        critical_events = [e for e in safety_events if e.severity == RiskLevel.CRITICAL]
        if critical_events:
            recommendations.append("Critical safety events detected - immediate review required")
        
        # Integration recommendations
        if safety_stats['total_events'] > 0 and compliance_report.overall_compliance_score < 90:
            recommendations.append("Correlation between safety events and compliance issues - comprehensive review needed")
        
        # Remove duplicates and sort
        unique_recommendations = list(set(recommendations))
        return sorted(unique_recommendations)


# Factory function
def create_clinical_compliance_system(event_bus: EventBus) -> ClinicalComplianceSystem:
    """Create a clinical compliance system instance."""
    return ClinicalComplianceSystem(event_bus)