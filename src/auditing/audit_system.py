"""
Comprehensive Audit Trail and Logging System for Mental Health AI.

This module provides detailed audit trails, compliance logging, and forensic
capabilities for all agent interactions and supervisory activities.
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict
import sqlite3
import os
import gzip

from src.utils.logger import get_logger
from src.utils.vector_db_integration import add_user_data

logger = get_logger(__name__)

class AuditEventType(Enum):
    """Types of audit events."""
    AGENT_INTERACTION = "agent_interaction"
    VALIDATION_PERFORMED = "validation_performed"
    RESPONSE_BLOCKED = "response_blocked"
    CRISIS_DETECTED = "crisis_detected"
    BOUNDARY_VIOLATION = "boundary_violation"
    ETHICAL_CONCERN = "ethical_concern"
    USER_FEEDBACK = "user_feedback"
    SYSTEM_ALERT = "system_alert"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    EXPORT_PERFORMED = "export_performed"

class ComplianceStandard(Enum):
    """Compliance standards for audit requirements."""
    HIPAA = "hipaa"
    GDPR = "gdpr"
    SOC2 = "soc2"
    CLINICAL_TRIALS = "clinical_trials"
    FDA_SOFTWARE = "fda_software"
    PROFESSIONAL_ETHICS = "professional_ethics"

class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AuditEvent:
    """Comprehensive audit event record."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    session_id: str
    user_id: str
    agent_name: str
    event_description: str
    event_data: Dict[str, Any]
    metadata: Dict[str, Any]
    compliance_flags: List[ComplianceStandard]
    data_hash: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    retention_policy: str = "7_years"  # Default retention for healthcare
    
    def __post_init__(self):
        """Generate data hash for integrity verification."""
        if not self.data_hash:
            event_str = json.dumps(asdict(self), sort_keys=True, default=str)
            self.data_hash = hashlib.sha256(event_str.encode()).hexdigest()

@dataclass
class ComplianceReport:
    """Compliance audit report."""
    report_id: str
    compliance_standard: ComplianceStandard
    reporting_period: Dict[str, datetime]
    total_events: int
    violations_found: int
    compliance_score: float
    critical_findings: List[str]
    recommendations: List[str]
    generated_timestamp: datetime
    generated_by: str

class AuditTrail:
    """Comprehensive audit trail system."""
    
    def __init__(self, storage_path: str = None, enable_encryption: bool = True):
        """Initialize audit trail system.
        
        Args:
            storage_path: Path for audit storage
            enable_encryption: Whether to encrypt audit data
        """
        self.storage_path = Path(storage_path or "src/data/audit")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_encryption = enable_encryption
        self.lock = threading.RLock()
        
        # Initialize database
        self.db_path = self.storage_path / "audit.db"
        # On Windows under pytest, use an in-memory shared DB to avoid file locks
        self._db_uri: Optional[str] = None
        try:
            if os.name == 'nt' and os.environ.get('PYTEST_CURRENT_TEST'):
                self._db_uri = f"file:audit_{id(self)}?mode=memory&cache=shared"
        except (KeyError, OSError, TypeError):
            self._db_uri = None
        self._initialize_database()
        
        # In-memory audit cache for real-time access
        self.audit_cache = defaultdict(list)
        self.cache_max_size = 1000
        
        # Compliance requirements mapping
        self.compliance_requirements = {
            ComplianceStandard.HIPAA: {
                "required_events": [AuditEventType.DATA_ACCESS, AuditEventType.EXPORT_PERFORMED],
                "retention_period": timedelta(days=2555),  # 7 years
                "encryption_required": True
            },
            ComplianceStandard.GDPR: {
                "required_events": [AuditEventType.DATA_ACCESS, AuditEventType.USER_FEEDBACK],
                "retention_period": timedelta(days=2190),  # 6 years
                "encryption_required": True
            },
            ComplianceStandard.SOC2: {
                "required_events": [AuditEventType.SYSTEM_ALERT, AuditEventType.CONFIGURATION_CHANGE],
                "retention_period": timedelta(days=1095),  # 3 years
                "encryption_required": True
            }
        }
        
        logger.info("Audit trail system initialized")
    
    def _connect(self) -> sqlite3.Connection:
        """Create a SQLite connection with safe defaults for Windows.

        Returns:
            sqlite3.Connection: configured connection
        """
        # Prefer memory DB during pytest on Windows
        if getattr(self, "_db_uri", None):
            conn = sqlite3.connect(self._db_uri, isolation_level=None, timeout=1.0, uri=True)
        else:
            # Normalize path and use URI with nolock on Windows to reduce file lock issues in tests
            db_posix = Path(self.db_path).absolute().as_posix()
            if Path(self.db_path).drive:
                uri = f"file:{db_posix}?mode=rwc&cache=private&nolock=1"
                conn = sqlite3.connect(uri, isolation_level=None, timeout=1.0, uri=True)
            else:
                conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=1.0)
        # Apply safe PRAGMAs
        try:
            if getattr(self, "_db_uri", None):
                # Tests: relax durability and avoid locks
                conn.execute("PRAGMA journal_mode=MEMORY;")
                conn.execute("PRAGMA synchronous=OFF;")
                conn.execute("PRAGMA busy_timeout=1000;")
            else:
                # File-backed: keep defaults, only set a busy timeout
                conn.execute("PRAGMA busy_timeout=1000;")
        except (sqlite3.Error, sqlite3.OperationalError, sqlite3.DatabaseError) as pragma_err:
            # PRAGMA failures are non-fatal - database will use defaults
            logger.debug(f"PRAGMA configuration skipped: {pragma_err}")
        return conn

    def _initialize_database(self):
        """Initialize SQLite database for audit storage."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    user_id TEXT,
                    agent_name TEXT,
                    event_description TEXT,
                    event_data TEXT,
                    metadata TEXT,
                    compliance_flags TEXT,
                    data_hash TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    correlation_id TEXT,
                    retention_policy TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)
                """
            )

            conn.commit()
    
    def log_event(self, event_type: AuditEventType, severity: AuditSeverity,
                  session_id: str, user_id: str, agent_name: str,
                  description: str, event_data: Dict[str, Any] = None,
                  metadata: Dict[str, Any] = None,
                  compliance_flags: List[ComplianceStandard] = None,
                  correlation_id: str = None) -> str:
        """Log an audit event.
        
        Args:
            event_type: Type of event being logged
            severity: Severity level of the event
            session_id: Session identifier
            user_id: User identifier
            agent_name: Name of the agent involved
            description: Human-readable description
            event_data: Structured event data
            metadata: Additional metadata
            compliance_flags: Relevant compliance standards
            correlation_id: Correlation ID for related events
            
        Returns:
            Event ID of the logged event
        """
        event_id = str(uuid.uuid4())
        
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            event_description=description,
            event_data=event_data or {},
            metadata=metadata or {},
            compliance_flags=compliance_flags or [],
            data_hash="",  # Will be generated in __post_init__
            correlation_id=correlation_id
        )
        
        # Store in database
        self._store_event(audit_event)
        
        # Add to cache
        with self.lock:
            self.audit_cache[session_id].append(audit_event)
            
            # Trim cache if too large
            if len(self.audit_cache[session_id]) > self.cache_max_size:
                self.audit_cache[session_id] = self.audit_cache[session_id][-self.cache_max_size//2:]
        
        # Log based on severity
        log_message = f"AUDIT [{event_type.value}] {description}"
        if severity == AuditSeverity.CRITICAL:
            logger.critical(log_message, {"event_id": event_id, "session_id": session_id})
        elif severity == AuditSeverity.HIGH:
            logger.error(log_message, {"event_id": event_id, "session_id": session_id})
        elif severity == AuditSeverity.WARNING:
            logger.warning(log_message, {"event_id": event_id, "session_id": session_id})
        else:
            logger.info(log_message, {"event_id": event_id, "session_id": session_id})
        
        return event_id
    
    def _store_event(self, event: AuditEvent):
        """Store audit event in database."""
        event_dict = asdict(event)
        # Serialize complex fields
        event_dict['event_data'] = json.dumps(event_dict['event_data'])
        event_dict['metadata'] = json.dumps(event_dict['metadata'])
        event_dict['compliance_flags'] = json.dumps([flag.value for flag in event.compliance_flags])
        event_dict['timestamp'] = event_dict['timestamp'].isoformat()
        event_dict['event_type'] = event_dict['event_type'].value
        event_dict['severity'] = event_dict['severity'].value

        placeholders = ', '.join(['?' for _ in event_dict])
        columns = ', '.join(event_dict.keys())

        def _insert_once() -> bool:
            with self._connect() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(
                        f"INSERT INTO audit_events ({columns}) VALUES ({placeholders})",
                        list(event_dict.values())
                    )
                    conn.commit()
                    return True
                finally:
                    cur.close()

        # Try insert; if schema missing (e.g., fresh in-memory DB), initialize and retry once
        try:
            if _insert_once():
                return
        except sqlite3.OperationalError as e:
            if "no such table: audit_events" in str(e).lower():
                self._initialize_database()
                # Retry once after initializing schema
                _insert_once()
            else:
                raise
    
    def log_agent_interaction(self, session_id: str, user_id: str, agent_name: str,
                            user_input: str, agent_response: str, 
                            validation_result: Any = None, processing_time: float = None):
        """Log a complete agent interaction."""
        event_data = {
            "user_input_length": len(user_input),
            "response_length": len(agent_response),
            "processing_time": processing_time,
            "validation_passed": validation_result.validation_level.value if validation_result else None,
            "risk_level": validation_result.clinical_risk.value if validation_result else None
        }
        
        # Determine severity based on validation result
        if validation_result:
            if validation_result.validation_level.value == "blocked":
                severity = AuditSeverity.CRITICAL
            elif validation_result.validation_level.value == "critical":
                severity = AuditSeverity.HIGH
            elif validation_result.validation_level.value == "warning":
                severity = AuditSeverity.WARNING
            else:
                severity = AuditSeverity.INFO
        else:
            severity = AuditSeverity.INFO
        
        # Add compliance flags based on content
        compliance_flags = [ComplianceStandard.HIPAA, ComplianceStandard.PROFESSIONAL_ETHICS]
        
        return self.log_event(
            event_type=AuditEventType.AGENT_INTERACTION,
            severity=severity,
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            description=f"Agent interaction with {len(user_input)} char input, {len(agent_response)} char response",
            event_data=event_data,
            compliance_flags=compliance_flags
        )
    
    def log_validation_event(self, session_id: str, user_id: str, agent_name: str,
                           validation_result: Any, validator_name: str):
        """Log a validation event."""
        event_data = {
            "validator": validator_name,
            "overall_score": validation_result.overall_score if hasattr(validation_result, 'overall_score') else None,
            "validation_level": validation_result.validation_level.value if hasattr(validation_result, 'validation_level') else None,
            "critical_issues": validation_result.critical_issues if hasattr(validation_result, 'critical_issues') else [],
            "blocking_issues": validation_result.blocking_issues if hasattr(validation_result, 'blocking_issues') else [],
            "recommendations": validation_result.recommendations if hasattr(validation_result, 'recommendations') else []
        }
        
        severity = AuditSeverity.INFO
        if hasattr(validation_result, 'validation_level'):
            if validation_result.validation_level.value == "blocked":
                severity = AuditSeverity.CRITICAL
            elif validation_result.validation_level.value == "critical":
                severity = AuditSeverity.HIGH
        
        return self.log_event(
            event_type=AuditEventType.VALIDATION_PERFORMED,
            severity=severity,
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            description=f"Validation performed by {validator_name}",
            event_data=event_data,
            compliance_flags=[ComplianceStandard.PROFESSIONAL_ETHICS, ComplianceStandard.SOC2]
        )
    
    def log_crisis_detection(self, session_id: str, user_id: str, agent_name: str,
                           crisis_type: str, crisis_indicators: List[str],
                           intervention_taken: str):
        """Log crisis detection and intervention."""
        event_data = {
            "crisis_type": crisis_type,
            "indicators": crisis_indicators,
            "intervention": intervention_taken,
            "requires_followup": True
        }
        
        return self.log_event(
            event_type=AuditEventType.CRISIS_DETECTED,
            severity=AuditSeverity.EMERGENCY,
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            description=f"Crisis detected: {crisis_type}",
            event_data=event_data,
            compliance_flags=[ComplianceStandard.HIPAA, ComplianceStandard.PROFESSIONAL_ETHICS]
        )
    
    def log_response_blocked(self, session_id: str, user_id: str, agent_name: str,
                           blocked_content: str, reason: str, alternative_provided: bool):
        """Log when a response is blocked."""
        # Hash the blocked content for audit trail without storing full content
        content_hash = hashlib.sha256(blocked_content.encode()).hexdigest()
        
        event_data = {
            "blocked_content_hash": content_hash,
            "blocked_content_length": len(blocked_content),
            "block_reason": reason,
            "alternative_provided": alternative_provided
        }
        
        return self.log_event(
            event_type=AuditEventType.RESPONSE_BLOCKED,
            severity=AuditSeverity.HIGH,
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            description=f"Response blocked due to: {reason}",
            event_data=event_data,
            compliance_flags=[ComplianceStandard.PROFESSIONAL_ETHICS]
        )
    
    def log_user_feedback(self, session_id: str, user_id: str, 
                         satisfaction_score: float, feedback_text: str = None):
        """Log user feedback."""
        event_data = {
            "satisfaction_score": satisfaction_score,
            "has_text_feedback": bool(feedback_text),
            "feedback_length": len(feedback_text) if feedback_text else 0
        }
        
        severity = AuditSeverity.WARNING if satisfaction_score < 3.0 else AuditSeverity.INFO
        
        return self.log_event(
            event_type=AuditEventType.USER_FEEDBACK,
            severity=severity,
            session_id=session_id,
            user_id=user_id,
            agent_name="system",
            description=f"User feedback received: {satisfaction_score}/5.0",
            event_data=event_data,
            compliance_flags=[ComplianceStandard.GDPR]
        )
    
    def get_session_audit_trail(self, session_id: str) -> List[AuditEvent]:
        """Get complete audit trail for a session."""
        # Check cache first
        with self.lock:
            if session_id in self.audit_cache:
                return self.audit_cache[session_id].copy()
        
        # Query database
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT * FROM audit_events WHERE session_id = ? ORDER BY timestamp",
                    (session_id,)
                )
                events: List[AuditEvent] = []
                for row in cur.fetchall():
                    event_dict = dict(row)
                    # Deserialize complex fields
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    event_dict['metadata'] = json.loads(event_dict['metadata'])
                    event_dict['compliance_flags'] = [
                        ComplianceStandard(flag)
                        for flag in json.loads(event_dict['compliance_flags'])
                    ]
                    event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                    event_dict['event_type'] = AuditEventType(event_dict['event_type'])
                    event_dict['severity'] = AuditSeverity(event_dict['severity'])
                    # Remove database-specific fields
                    event_dict.pop('created_at', None)
                    events.append(AuditEvent(**event_dict))
                return events
            finally:
                cur.close()
    
    def get_events_by_type(self, event_type: AuditEventType, 
                          start_time: datetime = None, 
                          end_time: datetime = None) -> List[AuditEvent]:
        """Get events by type within time range."""
        start_time = start_time or (datetime.now() - timedelta(days=30))
        end_time = end_time or datetime.now()

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute(
                    """SELECT * FROM audit_events 
                       WHERE event_type = ? AND timestamp BETWEEN ? AND ? 
                       ORDER BY timestamp DESC""",
                    (event_type.value, start_time.isoformat(), end_time.isoformat())
                )
                events: List[AuditEvent] = []
                for row in cur.fetchall():
                    event_dict = dict(row)
                    # Deserialize fields (same as above)
                    event_dict['event_data'] = json.loads(event_dict['event_data'])
                    event_dict['metadata'] = json.loads(event_dict['metadata'])
                    event_dict['compliance_flags'] = [
                        ComplianceStandard(flag)
                        for flag in json.loads(event_dict['compliance_flags'])
                    ]
                    event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                    event_dict['event_type'] = AuditEventType(event_dict['event_type'])
                    event_dict['severity'] = AuditSeverity(event_dict['severity'])
                    event_dict.pop('created_at', None)
                    events.append(AuditEvent(**event_dict))
                return events
            finally:
                cur.close()
    
    def generate_compliance_report(self, compliance_standard: ComplianceStandard,
                                 start_date: datetime, end_date: datetime) -> ComplianceReport:
        """Generate compliance audit report."""
        report_id = str(uuid.uuid4())

        # Get all events in the reporting period
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute(
                    """SELECT * FROM audit_events 
                       WHERE timestamp BETWEEN ? AND ?""",
                    (start_date.isoformat(), end_date.isoformat())
                )
                rows = cur.fetchall()
                all_events = [dict(row) for row in rows]
            finally:
                cur.close()
        
        # Filter events relevant to compliance standard
        relevant_events = []
        required_events = self.compliance_requirements[compliance_standard]["required_events"]
        
        for event in all_events:
            event_type = AuditEventType(event['event_type'])
            compliance_flags = json.loads(event['compliance_flags'])
            
            if (event_type in required_events or 
                compliance_standard.value in compliance_flags):
                relevant_events.append(event)
        
        # Analyze compliance
        total_events = len(relevant_events)
        violations = self._identify_violations(relevant_events, compliance_standard)
        compliance_score = max(0, 1.0 - (len(violations) / max(total_events, 1)))
        
        # Generate findings and recommendations
        critical_findings = [v["description"] for v in violations if v["severity"] == "critical"]
        recommendations = self._generate_compliance_recommendations(violations, compliance_standard)
        
        return ComplianceReport(
            report_id=report_id,
            compliance_standard=compliance_standard,
            reporting_period={"start": start_date, "end": end_date},
            total_events=total_events,
            violations_found=len(violations),
            compliance_score=compliance_score,
            critical_findings=critical_findings,
            recommendations=recommendations,
            generated_timestamp=datetime.now(),
            generated_by="audit_system"
        )
    
    def _identify_violations(self, events: List[Dict],
                             compliance_standard: ComplianceStandard) -> List[Dict[str, Any]]:
        """Identify compliance violations in events."""
        violations: List[Dict[str, Any]] = []

        # Check for missing required events
        event_types = {e['event_type'] for e in events}
        required_events = self.compliance_requirements[compliance_standard]["required_events"]

        for required_event in required_events:
            if required_event.value not in event_types:
                violations.append({
                    "type": "missing_required_event",
                    "description": f"Missing required event type: {required_event.value}",
                    "severity": "high",
                })

        # Check for high-severity events without proper follow-up
        high_severity_events = [e for e in events if e['severity'] in ['critical', 'emergency']]

        for ev in high_severity_events:
            # Check if there's a follow-up event within reasonable time
            event_time = datetime.fromisoformat(ev['timestamp'])
            followup_window = event_time + timedelta(hours=24)

            followup_exists = any(
                datetime.fromisoformat(e['timestamp']) <= followup_window and
                e.get('correlation_id') == ev.get('event_id')
                for e in events
            )

            if not followup_exists:
                violations.append({
                    "type": "missing_followup",
                    "description": f"High-severity event {ev.get('event_id')} lacks proper follow-up",
                    "severity": "moderate",
                })

        return violations
    
    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]],
                                           compliance_standard: ComplianceStandard) -> List[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []
        
        if any(v["type"] == "missing_required_event" for v in violations):
            recommendations.append("Ensure all required event types are being logged")
        
        if any(v["type"] == "missing_followup" for v in violations):
            recommendations.append("Implement systematic follow-up procedures for high-severity events")
        
        if compliance_standard == ComplianceStandard.HIPAA:
            recommendations.append("Review data access logging and encryption practices")
        
        if compliance_standard == ComplianceStandard.GDPR:
            recommendations.append("Verify user consent tracking and data retention policies")
        
        return recommendations
    
    def export_audit_data(self, output_path: str, start_date: datetime = None,
                         end_date: datetime = None, event_types: List[AuditEventType] = None,
                         compress: bool = True):
        """Export audit data for external analysis or compliance reporting."""
        start_date = start_date or (datetime.now() - timedelta(days=30))
        end_date = end_date or datetime.now()
        
        # Build query
        query = "SELECT * FROM audit_events WHERE timestamp BETWEEN ? AND ?"
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if event_types:
            query += " AND event_type IN ({})".format(','.join(['?' for _ in event_types]))
            params.extend([et.value for et in event_types])
        
        query += " ORDER BY timestamp"

        # Execute query and export
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute(query, params)
                export_data = {
                    "export_metadata": {
                        "export_timestamp": datetime.now().isoformat(),
                        "export_period": {
                            "start": start_date.isoformat(),
                            "end": end_date.isoformat()
                        },
                        "total_records": 0
                    },
                    "audit_events": []
                }
                for row in cur.fetchall():
                    event_dict = dict(row)
                    export_data["audit_events"].append(event_dict)
                export_data["export_metadata"]["total_records"] = len(export_data["audit_events"])
            finally:
                cur.close()
        
        # Write to file
        if compress:
            with gzip.open(f"{output_path}.gz", 'wt', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        # Log the export
        self.log_event(
            event_type=AuditEventType.EXPORT_PERFORMED,
            severity=AuditSeverity.INFO,
            session_id="system",
            user_id="system",
            agent_name="audit_system",
            description=f"Audit data exported to {output_path}",
            event_data={
                "records_exported": export_data["export_metadata"]["total_records"],
                "compressed": compress
            },
            compliance_flags=[ComplianceStandard.HIPAA, ComplianceStandard.SOC2]
        )
        
        logger.info(f"Audit data exported: {export_data['export_metadata']['total_records']} records")
    
    def cleanup_expired_records(self):
        """Clean up audit records that have exceeded their retention period."""
        cleaned_count = 0
        
        for standard, requirements in self.compliance_requirements.items():
            retention_period = requirements["retention_period"]
            cutoff_date = datetime.now() - retention_period

            with self._connect() as conn:
                cur = conn.cursor()
                try:
                    cur.execute(
                        """DELETE FROM audit_events 
                           WHERE timestamp < ? AND compliance_flags LIKE ?""",
                        (cutoff_date.isoformat(), f'%{standard.value}%')
                    )
                    cleaned_count += cur.rowcount
                    conn.commit()
                finally:
                    cur.close()
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired audit records")
        return cleaned_count

    def close(self):
        """Explicit close hook for compatibility. No persistent connections are maintained."""
        return None