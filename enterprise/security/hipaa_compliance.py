"""
HIPAA Compliance Framework for Mental Health Platform
Implements comprehensive HIPAA security and privacy controls
"""

import asyncio
import hashlib
import secrets
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import os
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PHIClassification(Enum):
    """Protected Health Information Classification"""
    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier" 
    SENSITIVE_HEALTH_INFO = "sensitive_health_info"
    CLINICAL_DATA = "clinical_data"
    DEMOGRAPHIC_DATA = "demographic_data"
    NON_PHI = "non_phi"


class AccessLevel(Enum):
    """HIPAA Access Levels"""
    NO_ACCESS = "no_access"
    VIEW_ONLY = "view_only"
    LIMITED_ACCESS = "limited_access"
    FULL_ACCESS = "full_access"
    EMERGENCY_ACCESS = "emergency_access"
    ADMINISTRATIVE = "administrative"


class AuditEventType(Enum):
    """HIPAA Audit Event Types"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PHI_ACCESS = "phi_access"
    PHI_CREATION = "phi_creation"
    PHI_MODIFICATION = "phi_modification"
    PHI_DELETION = "phi_deletion"
    PHI_EXPORT = "phi_export"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_ADMIN = "system_admin"
    EMERGENCY_ACCESS = "emergency_access"
    BREACH_DETECTION = "breach_detection"


@dataclass
class PHIElement:
    """Protected Health Information Element"""
    element_id: str
    element_type: str
    classification: PHIClassification
    content: Any
    encrypted_content: Optional[str] = None
    hash_value: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    patient_id: Optional[str] = None
    provider_id: Optional[str] = None
    retention_period_days: int = 2555  # 7 years default


@dataclass
class User:
    """HIPAA-compliant user entity"""
    user_id: str
    username: str
    email: str
    role: str
    access_level: AccessLevel
    permissions: Set[str]
    department: str
    is_covered_entity: bool
    is_business_associate: bool
    mfa_enabled: bool
    password_hash: str
    failed_login_attempts: int = 0
    account_locked: bool = False
    last_login: Optional[datetime] = None
    last_password_change: datetime = field(default_factory=datetime.utcnow)
    session_timeout: int = 900  # 15 minutes
    created_at: datetime = field(default_factory=datetime.utcnow)
    training_completed: Dict[str, datetime] = field(default_factory=dict)
    business_associate_agreement: Optional[str] = None


@dataclass
class AuditLog:
    """HIPAA Audit Log Entry"""
    audit_id: str
    event_type: AuditEventType
    user_id: str
    patient_id: Optional[str]
    phi_accessed: List[str]
    ip_address: str
    user_agent: str
    session_id: str
    action_description: str
    outcome: str  # success, failure, warning
    timestamp: datetime = field(default_factory=datetime.utcnow)
    risk_score: float = 0.0
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityIncident:
    """Security Incident Record"""
    incident_id: str
    incident_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_patients: List[str]
    affected_phi: List[str]
    detected_at: datetime
    reported_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    status: str = "open"  # open, investigating, resolved
    actions_taken: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    notification_required: bool = False
    notification_sent: bool = False


class EncryptionManager:
    """HIPAA-compliant encryption and key management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = Fernet.generate_key()
        self.master_key = master_key
        self.fernet = Fernet(master_key)
        self.key_rotation_interval = timedelta(days=90)
        self.encryption_keys: Dict[str, bytes] = {}
        
    def generate_patient_key(self, patient_id: str) -> bytes:
        """Generate patient-specific encryption key"""
        if patient_id not in self.encryption_keys:
            # Derive key from master key and patient ID
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=patient_id.encode(),
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            self.encryption_keys[patient_id] = key
            
        return self.encryption_keys[patient_id]
        
    def encrypt_phi(self, content: str, patient_id: str) -> str:
        """Encrypt PHI with patient-specific key"""
        patient_key = self.generate_patient_key(patient_id)
        patient_fernet = Fernet(patient_key)
        
        encrypted_content = patient_fernet.encrypt(content.encode())
        return base64.urlsafe_b64encode(encrypted_content).decode()
        
    def decrypt_phi(self, encrypted_content: str, patient_id: str) -> str:
        """Decrypt PHI with patient-specific key"""
        try:
            patient_key = self.generate_patient_key(patient_id)
            patient_fernet = Fernet(patient_key)
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_content.encode())
            decrypted_content = patient_fernet.decrypt(encrypted_bytes)
            return decrypted_content.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt PHI for patient {patient_id}: {e}")
            raise
            
    def hash_phi(self, content: str, salt: Optional[str] = None) -> str:
        """Create irreversible hash of PHI for indexing"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_input = f"{content}{salt}".encode()
        hash_value = hashlib.sha256(hash_input).hexdigest()
        return f"{salt}:{hash_value}"
        
    def verify_phi_hash(self, content: str, hash_value: str) -> bool:
        """Verify PHI against stored hash"""
        try:
            salt, stored_hash = hash_value.split(':', 1)
            computed_hash = hashlib.sha256(f"{content}{salt}".encode()).hexdigest()
            return secrets.compare_digest(stored_hash, computed_hash)
        except:
            return False
            
    def rotate_keys(self):
        """Rotate encryption keys (should be scheduled)"""
        # Implementation would re-encrypt all data with new keys
        logger.info("Key rotation initiated - this is a placeholder for production implementation")


class AccessControlManager:
    """HIPAA Access Control and Authorization"""
    
    def __init__(self):
        self.role_permissions = {
            "physician": {
                "view_patient_phi", "modify_patient_phi", "create_treatment_plan",
                "prescribe_medication", "view_full_medical_record"
            },
            "therapist": {
                "view_patient_phi", "modify_therapy_notes", "create_treatment_plan",
                "view_limited_medical_record"
            },
            "nurse": {
                "view_patient_phi", "modify_nursing_notes", "view_limited_medical_record"
            },
            "admin": {
                "view_system_logs", "manage_users", "system_configuration"
            },
            "patient": {
                "view_own_phi", "modify_contact_info", "view_treatment_history"
            },
            "researcher": {
                "view_deidentified_data", "export_aggregate_data"
            }
        }
        
    def check_permission(self, user: User, permission: str, resource: str = None) -> bool:
        """Check if user has specific permission"""
        if user.account_locked:
            return False
            
        # Check role-based permissions
        role_perms = self.role_permissions.get(user.role, set())
        if permission not in role_perms and permission not in user.permissions:
            return False
            
        # Additional checks for patient data access
        if resource and resource.startswith("patient:"):
            patient_id = resource.split(":", 1)[1]
            return self._check_patient_access(user, patient_id, permission)
            
        return True
        
    def _check_patient_access(self, user: User, patient_id: str, permission: str) -> bool:
        """Check patient-specific access permissions"""
        # Patients can only access their own data
        if user.role == "patient":
            return user.user_id == patient_id
            
        # Healthcare providers need minimum necessary access
        if user.role in ["physician", "therapist", "nurse"]:
            # Would check provider-patient relationships in real implementation
            return True
            
        # Researchers only get de-identified data
        if user.role == "researcher":
            return permission in ["view_deidentified_data", "export_aggregate_data"]
            
        return False
        
    def get_minimum_necessary_data(self, user: User, patient_id: str, 
                                 requested_data: List[str]) -> List[str]:
        """Return minimum necessary data based on user role and purpose"""
        allowed_data = []
        
        if user.role == "physician":
            # Physicians can access most data
            allowed_data = requested_data
        elif user.role == "therapist":
            # Therapists get mental health focused data
            therapy_relevant = [
                "mental_health_history", "therapy_notes", "medication_psychiatric",
                "treatment_plans", "session_notes", "psychological_assessments"
            ]
            allowed_data = [d for d in requested_data if any(tr in d for tr in therapy_relevant)]
        elif user.role == "nurse":
            # Nurses get care coordination data
            nursing_relevant = [
                "current_medications", "vital_signs", "nursing_notes", 
                "care_plans", "allergies"
            ]
            allowed_data = [d for d in requested_data if any(nr in d for nr in nursing_relevant)]
        elif user.role == "patient":
            # Patients get their own data
            if user.user_id == patient_id:
                allowed_data = requested_data
                
        return allowed_data


class AuditLogger:
    """HIPAA Audit Logging System"""
    
    def __init__(self):
        self.audit_logs: List[AuditLog] = []
        self.high_risk_events = {
            AuditEventType.UNAUTHORIZED_ACCESS,
            AuditEventType.PHI_DELETION,
            AuditEventType.EMERGENCY_ACCESS,
            AuditEventType.BREACH_DETECTION
        }
        
    async def log_event(self, event_type: AuditEventType, user_id: str,
                       ip_address: str, user_agent: str, session_id: str,
                       action_description: str, outcome: str = "success",
                       patient_id: Optional[str] = None,
                       phi_accessed: List[str] = None,
                       additional_data: Dict[str, Any] = None) -> str:
        """Log HIPAA audit event"""
        
        audit_id = str(uuid.uuid4())
        risk_score = self._calculate_risk_score(event_type, user_id, ip_address, outcome)
        
        audit_log = AuditLog(
            audit_id=audit_id,
            event_type=event_type,
            user_id=user_id,
            patient_id=patient_id,
            phi_accessed=phi_accessed or [],
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            action_description=action_description,
            outcome=outcome,
            risk_score=risk_score,
            additional_data=additional_data or {}
        )
        
        self.audit_logs.append(audit_log)
        
        # Alert on high-risk events
        if event_type in self.high_risk_events or risk_score > 7.0:
            await self._generate_security_alert(audit_log)
            
        logger.info(f"Audit event logged: {event_type.value} by {user_id} (Risk: {risk_score})")
        return audit_id
        
    def _calculate_risk_score(self, event_type: AuditEventType, user_id: str, 
                            ip_address: str, outcome: str) -> float:
        """Calculate risk score for audit event"""
        risk_score = 0.0
        
        # Base risk by event type
        event_risks = {
            AuditEventType.LOGIN_FAILURE: 2.0,
            AuditEventType.UNAUTHORIZED_ACCESS: 9.0,
            AuditEventType.PHI_ACCESS: 3.0,
            AuditEventType.PHI_MODIFICATION: 4.0,
            AuditEventType.PHI_DELETION: 8.0,
            AuditEventType.PHI_EXPORT: 5.0,
            AuditEventType.EMERGENCY_ACCESS: 6.0,
            AuditEventType.BREACH_DETECTION: 10.0
        }
        
        risk_score += event_risks.get(event_type, 1.0)
        
        # Outcome risk
        if outcome == "failure":
            risk_score += 2.0
        elif outcome == "warning":
            risk_score += 1.0
            
        # Time-based risk (off-hours access)
        current_hour = datetime.utcnow().hour
        if current_hour < 7 or current_hour > 19:  # Outside business hours
            risk_score += 1.0
            
        # Multiple failed attempts (would need session tracking)
        # This is simplified - real implementation would track patterns
        
        return min(10.0, risk_score)
        
    async def _generate_security_alert(self, audit_log: AuditLog):
        """Generate security alert for high-risk events"""
        alert_message = f"High-risk security event detected: {audit_log.event_type.value}"
        
        # In production, this would send alerts to security team
        logger.warning(f"SECURITY ALERT: {alert_message} - User: {audit_log.user_id}, Risk: {audit_log.risk_score}")
        
    async def search_audit_logs(self, criteria: Dict[str, Any]) -> List[AuditLog]:
        """Search audit logs with criteria"""
        filtered_logs = self.audit_logs
        
        if "user_id" in criteria:
            filtered_logs = [log for log in filtered_logs if log.user_id == criteria["user_id"]]
            
        if "patient_id" in criteria:
            filtered_logs = [log for log in filtered_logs if log.patient_id == criteria["patient_id"]]
            
        if "event_type" in criteria:
            filtered_logs = [log for log in filtered_logs if log.event_type == criteria["event_type"]]
            
        if "start_date" in criteria:
            start_date = criteria["start_date"]
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
            
        if "end_date" in criteria:
            end_date = criteria["end_date"]
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
            
        return filtered_logs
        
    async def generate_audit_report(self, patient_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate HIPAA audit report for patient"""
        patient_logs = await self.search_audit_logs({
            "patient_id": patient_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        # Summarize access patterns
        access_by_user = {}
        access_by_type = {}
        
        for log in patient_logs:
            if log.user_id not in access_by_user:
                access_by_user[log.user_id] = 0
            access_by_user[log.user_id] += 1
            
            if log.event_type not in access_by_type:
                access_by_type[log.event_type] = 0
            access_by_type[log.event_type] += 1
            
        return {
            "patient_id": patient_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_access_events": len(patient_logs),
            "access_by_user": access_by_user,
            "access_by_type": {k.value: v for k, v in access_by_type.items()},
            "high_risk_events": [
                {
                    "audit_id": log.audit_id,
                    "event_type": log.event_type.value,
                    "user_id": log.user_id,
                    "timestamp": log.timestamp.isoformat(),
                    "risk_score": log.risk_score
                }
                for log in patient_logs if log.risk_score > 5.0
            ]
        }


class BreachDetectionSystem:
    """HIPAA Breach Detection and Response"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.incidents: List[SecurityIncident] = []
        self.detection_rules = [
            self._detect_unusual_access_patterns,
            self._detect_bulk_data_export,
            self._detect_failed_login_patterns,
            self._detect_unauthorized_access,
            self._detect_off_hours_access
        ]
        
    async def monitor_for_breaches(self):
        """Continuously monitor for potential breaches"""
        while True:
            try:
                recent_logs = [log for log in self.audit_logger.audit_logs 
                             if (datetime.utcnow() - log.timestamp).seconds < 3600]  # Last hour
                
                for rule in self.detection_rules:
                    potential_incidents = await rule(recent_logs)
                    for incident in potential_incidents:
                        await self._handle_incident(incident)
                        
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in breach monitoring: {e}")
                await asyncio.sleep(60)
                
    async def _detect_unusual_access_patterns(self, recent_logs: List[AuditLog]) -> List[SecurityIncident]:
        """Detect unusual patient data access patterns"""
        incidents = []
        
        # Group by user
        user_access = {}
        for log in recent_logs:
            if log.event_type == AuditEventType.PHI_ACCESS and log.patient_id:
                if log.user_id not in user_access:
                    user_access[log.user_id] = set()
                user_access[log.user_id].add(log.patient_id)
                
        # Check for users accessing unusually many patients
        for user_id, patients in user_access.items():
            if len(patients) > 20:  # Threshold for unusual access
                incident = SecurityIncident(
                    incident_id=str(uuid.uuid4()),
                    incident_type="unusual_access_pattern",
                    severity="medium",
                    description=f"User {user_id} accessed {len(patients)} different patients in 1 hour",
                    affected_patients=list(patients),
                    affected_phi=["patient_records"],
                    detected_at=datetime.utcnow()
                )
                incidents.append(incident)
                
        return incidents
        
    async def _detect_bulk_data_export(self, recent_logs: List[AuditLog]) -> List[SecurityIncident]:
        """Detect bulk PHI export attempts"""
        incidents = []
        
        export_events = [log for log in recent_logs if log.event_type == AuditEventType.PHI_EXPORT]
        
        if len(export_events) > 5:  # More than 5 exports in an hour
            affected_patients = list(set(log.patient_id for log in export_events if log.patient_id))
            
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                incident_type="bulk_data_export",
                severity="high",
                description=f"Bulk PHI export detected: {len(export_events)} export events",
                affected_patients=affected_patients,
                affected_phi=["exported_records"],
                detected_at=datetime.utcnow(),
                notification_required=True
            )
            incidents.append(incident)
            
        return incidents
        
    async def _detect_failed_login_patterns(self, recent_logs: List[AuditLog]) -> List[SecurityIncident]:
        """Detect brute force login attempts"""
        incidents = []
        
        failed_logins = [log for log in recent_logs if log.event_type == AuditEventType.LOGIN_FAILURE]
        
        # Group by IP address
        ip_failures = {}
        for log in failed_logins:
            if log.ip_address not in ip_failures:
                ip_failures[log.ip_address] = 0
            ip_failures[log.ip_address] += 1
            
        for ip, count in ip_failures.items():
            if count > 10:  # More than 10 failures from same IP
                incident = SecurityIncident(
                    incident_id=str(uuid.uuid4()),
                    incident_type="brute_force_attack",
                    severity="high",
                    description=f"Potential brute force attack from IP {ip}: {count} failed logins",
                    affected_patients=[],
                    affected_phi=[],
                    detected_at=datetime.utcnow()
                )
                incidents.append(incident)
                
        return incidents
        
    async def _detect_unauthorized_access(self, recent_logs: List[AuditLog]) -> List[SecurityIncident]:
        """Detect unauthorized access attempts"""
        incidents = []
        
        unauthorized_events = [log for log in recent_logs 
                             if log.event_type == AuditEventType.UNAUTHORIZED_ACCESS]
        
        for event in unauthorized_events:
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                incident_type="unauthorized_access",
                severity="critical",
                description=f"Unauthorized access attempt by {event.user_id}",
                affected_patients=[event.patient_id] if event.patient_id else [],
                affected_phi=event.phi_accessed,
                detected_at=datetime.utcnow(),
                notification_required=True
            )
            incidents.append(incident)
            
        return incidents
        
    async def _detect_off_hours_access(self, recent_logs: List[AuditLog]) -> List[SecurityIncident]:
        """Detect unusual off-hours access"""
        incidents = []
        
        off_hours_access = []
        for log in recent_logs:
            if log.event_type == AuditEventType.PHI_ACCESS:
                hour = log.timestamp.hour
                if hour < 6 or hour > 22:  # Very late or very early
                    off_hours_access.append(log)
                    
        if len(off_hours_access) > 5:
            affected_patients = list(set(log.patient_id for log in off_hours_access if log.patient_id))
            
            incident = SecurityIncident(
                incident_id=str(uuid.uuid4()),
                incident_type="off_hours_access",
                severity="medium",
                description=f"Unusual off-hours PHI access: {len(off_hours_access)} events",  
                affected_patients=affected_patients,
                affected_phi=["patient_records"],
                detected_at=datetime.utcnow()
            )
            incidents.append(incident)
            
        return incidents
        
    async def _handle_incident(self, incident: SecurityIncident):
        """Handle detected security incident"""
        self.incidents.append(incident)
        
        # Log the incident
        await self.audit_logger.log_event(
            event_type=AuditEventType.BREACH_DETECTION,
            user_id="system",
            ip_address="127.0.0.1",
            user_agent="breach_detection_system",
            session_id="system",
            action_description=f"Security incident detected: {incident.incident_type}",
            outcome="warning",
            additional_data={
                "incident_id": incident.incident_id,
                "severity": incident.severity,
                "affected_patients_count": len(incident.affected_patients)
            }
        )
        
        # Immediate response based on severity
        if incident.severity == "critical":
            await self._immediate_response(incident)
        elif incident.severity == "high":
            await self._escalate_incident(incident)
            
        logger.warning(f"Security incident detected: {incident.incident_type} (Severity: {incident.severity})")
        
    async def _immediate_response(self, incident: SecurityIncident):
        """Immediate response for critical incidents"""
        # In production, this would:
        # 1. Lock affected accounts
        # 2. Notify security team immediately
        # 3. Preserve evidence
        # 4. Begin containment procedures
        
        incident.actions_taken.append("Immediate response initiated")
        incident.actions_taken.append("Security team notified")
        
    async def _escalate_incident(self, incident: SecurityIncident):
        """Escalate high-severity incidents"""
        # In production, this would:
        # 1. Notify security officer
        # 2. Begin investigation procedures
        # 3. Document evidence
        
        incident.actions_taken.append("Incident escalated to security officer")
        
    async def generate_breach_report(self, incident_id: str) -> Dict[str, Any]:
        """Generate HIPAA breach report"""
        incident = next((inc for inc in self.incidents if inc.incident_id == incident_id), None)
        
        if not incident:
            return {"error": "Incident not found"}
            
        return {
            "incident_id": incident.incident_id,
            "incident_type": incident.incident_type,
            "severity": incident.severity,
            "description": incident.description,
            "detected_at": incident.detected_at.isoformat(),
            "status": incident.status,
            "affected_patients_count": len(incident.affected_patients),
            "affected_phi_types": incident.affected_phi,
            "actions_taken": incident.actions_taken,
            "notification_required": incident.notification_required,
            "notification_sent": incident.notification_sent,
            "risk_assessment": incident.risk_assessment
        }


class HIPAAComplianceManager:
    """Main HIPAA Compliance Management System"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.access_control = AccessControlManager()
        self.audit_logger = AuditLogger()
        self.breach_detector = BreachDetectionSystem(self.audit_logger)
        self.users: Dict[str, User] = {}
        self.phi_elements: Dict[str, PHIElement] = {}
        
    async def initialize(self):
        """Initialize HIPAA compliance system"""
        # Start breach monitoring
        asyncio.create_task(self.breach_detector.monitor_for_breaches())
        logger.info("HIPAA Compliance Manager initialized")
        
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create HIPAA-compliant user"""
        user = User(
            user_id=str(uuid.uuid4()),
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"],
            access_level=AccessLevel(user_data.get("access_level", "limited_access")),
            permissions=set(user_data.get("permissions", [])),
            department=user_data["department"],
            is_covered_entity=user_data.get("is_covered_entity", False),
            is_business_associate=user_data.get("is_business_associate", False),
            mfa_enabled=user_data.get("mfa_enabled", True),
            password_hash=self._hash_password(user_data["password"])
        )
        
        self.users[user.user_id] = user
        
        await self.audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_ADMIN,
            user_id="system",
            ip_address="127.0.0.1",
            user_agent="system",
            session_id="system",
            action_description=f"User created: {user.username} ({user.role})"
        )
        
        return user
        
    def _hash_password(self, password: str) -> str:
        """Hash password securely"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
        
    async def store_phi(self, content: Any, classification: PHIClassification,
                       patient_id: str, element_type: str) -> str:
        """Store PHI with encryption and audit trail"""
        element_id = str(uuid.uuid4())
        
        # Encrypt sensitive content
        if classification in [PHIClassification.DIRECT_IDENTIFIER, 
                            PHIClassification.SENSITIVE_HEALTH_INFO]:
            encrypted_content = self.encryption_manager.encrypt_phi(str(content), patient_id)
            hash_value = self.encryption_manager.hash_phi(str(content))
        else:
            encrypted_content = None
            hash_value = None
            
        phi_element = PHIElement(
            element_id=element_id,
            element_type=element_type,
            classification=classification,
            content=content if encrypted_content is None else None,
            encrypted_content=encrypted_content,
            hash_value=hash_value,
            patient_id=patient_id
        )
        
        self.phi_elements[element_id] = phi_element
        
        await self.audit_logger.log_event(
            event_type=AuditEventType.PHI_CREATION,
            user_id="system",
            ip_address="127.0.0.1",
            user_agent="system", 
            session_id="system",
            action_description=f"PHI element created: {element_type}",
            patient_id=patient_id,
            phi_accessed=[element_id]
        )
        
        return element_id
        
    async def access_phi(self, element_id: str, user: User, 
                        session_info: Dict[str, str]) -> Optional[Any]:
        """Access PHI with authorization and audit"""
        
        phi_element = self.phi_elements.get(element_id)
        if not phi_element:
            return None
            
        # Check authorization
        resource = f"patient:{phi_element.patient_id}"
        if not self.access_control.check_permission(user, "view_patient_phi", resource):
            await self.audit_logger.log_event(
                event_type=AuditEventType.UNAUTHORIZED_ACCESS,
                user_id=user.user_id,
                ip_address=session_info.get("ip_address", ""),
                user_agent=session_info.get("user_agent", ""),
                session_id=session_info.get("session_id", ""),
                action_description=f"Unauthorized PHI access attempt: {element_id}",
                outcome="failure",
                patient_id=phi_element.patient_id
            )
            return None
            
        # Decrypt if necessary
        content = phi_element.content
        if phi_element.encrypted_content:
            content = self.encryption_manager.decrypt_phi(
                phi_element.encrypted_content, 
                phi_element.patient_id
            )
            
        # Update access tracking
        phi_element.last_accessed = datetime.utcnow()
        phi_element.access_count += 1
        
        # Audit the access
        await self.audit_logger.log_event(
            event_type=AuditEventType.PHI_ACCESS,
            user_id=user.user_id,
            ip_address=session_info.get("ip_address", ""),
            user_agent=session_info.get("user_agent", ""),
            session_id=session_info.get("session_id", ""),
            action_description=f"PHI accessed: {phi_element.element_type}",
            patient_id=phi_element.patient_id,
            phi_accessed=[element_id]
        )
        
        return content
        
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall HIPAA compliance status"""
        total_users = len(self.users)
        mfa_enabled = sum(1 for user in self.users.values() if user.mfa_enabled)
        
        recent_incidents = [inc for inc in self.breach_detector.incidents 
                          if (datetime.utcnow() - inc.detected_at).days <= 30]
        
        recent_audits = [log for log in self.audit_logger.audit_logs
                        if (datetime.utcnow() - log.timestamp).days <= 30]
        
        return {
            "compliance_score": self._calculate_compliance_score(),
            "total_users": total_users,
            "mfa_adoption_rate": mfa_enabled / total_users if total_users > 0 else 0,
            "recent_incidents": len(recent_incidents),
            "critical_incidents": len([inc for inc in recent_incidents if inc.severity == "critical"]),
            "audit_events_30_days": len(recent_audits),
            "high_risk_events_30_days": len([log for log in recent_audits if log.risk_score > 7.0]),
            "phi_elements_encrypted": sum(1 for phi in self.phi_elements.values() if phi.encrypted_content),
            "total_phi_elements": len(self.phi_elements),
            "last_assessment": datetime.utcnow().isoformat()
        }
        
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)"""
        score = 100.0
        
        # Deduct for security issues
        critical_incidents = len([inc for inc in self.breach_detector.incidents 
                                if inc.severity == "critical" and inc.status == "open"])
        score -= critical_incidents * 20
        
        # Deduct for users without MFA
        users_without_mfa = sum(1 for user in self.users.values() if not user.mfa_enabled)
        if self.users:
            score -= (users_without_mfa / len(self.users)) * 15
            
        # Deduct for unencrypted PHI
        unencrypted_phi = sum(1 for phi in self.phi_elements.values() 
                            if phi.classification in [PHIClassification.DIRECT_IDENTIFIER,
                                                    PHIClassification.SENSITIVE_HEALTH_INFO]
                            and not phi.encrypted_content)
        if self.phi_elements:
            score -= (unencrypted_phi / len(self.phi_elements)) * 25
            
        return max(0.0, score)