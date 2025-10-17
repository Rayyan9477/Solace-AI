"""
HIPAA Compliance Validation System
Implements comprehensive HIPAA compliance checks and PHI protection
"""

import re
import logging
import hashlib
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

class PHIType(Enum):
    """Types of Protected Health Information"""
    NAME = "name"
    SSN = "ssn"
    DOB = "date_of_birth"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    MRN = "medical_record_number"
    ACCOUNT_NUMBER = "account_number"
    CERTIFICATE_NUMBER = "certificate_number"
    DEVICE_IDENTIFIER = "device_identifier"
    URL = "url"
    IP_ADDRESS = "ip_address"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"

class ComplianceLevel(Enum):
    """HIPAA compliance levels"""
    STRICT = "strict"          # Maximum protection
    STANDARD = "standard"      # Standard HIPAA requirements
    BASIC = "basic"           # Minimum requirements

@dataclass
class PHIDetection:
    """PHI detection result"""
    phi_type: PHIType
    value: str
    confidence: float
    start_position: int
    end_position: int
    context: str

@dataclass
class ComplianceViolation:
    """HIPAA compliance violation"""
    violation_type: str
    severity: str
    description: str
    location: str
    phi_detected: List[PHIDetection]
    timestamp: datetime
    remediation: str

class PHIDetector:
    """Detects Protected Health Information in text"""
    
    def __init__(self):
        # PHI detection patterns
        self.patterns = {
            PHIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{3}\s\d{2}\s\d{4}\b',
                r'\b\d{9}\b'
            ],
            PHIType.PHONE: [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\(\d{3}\)\s?\d{3}-\d{4}',
                r'\b\d{10}\b'
            ],
            PHIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PHIType.DOB: [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
                r'\b\d{4}-\d{1,2}-\d{1,2}\b'
            ],
            PHIType.MRN: [
                r'\bMRN:?\s*[A-Z0-9]{6,10}\b',
                r'\bMedical Record:?\s*[A-Z0-9]{6,10}\b'
            ],
            PHIType.IP_ADDRESS: [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ]
        }
        
        # Common name patterns (basic detection)
        self.name_patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b'  # Last, First
        ]
        
        # Medical terminology that might contain PHI
        self.medical_indicators = {
            'diagnosis', 'medication', 'prescription', 'treatment', 'therapy',
            'condition', 'disease', 'symptom', 'patient', 'doctor', 'physician'
        }
    
    def detect_phi(self, text: str, context: str = "") -> List[PHIDetection]:
        """Detect PHI in text"""
        detections = []
        
        # Check each PHI type pattern
        for phi_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    detection = PHIDetection(
                        phi_type=phi_type,
                        value=match.group(),
                        confidence=0.9,  # High confidence for pattern matches
                        start_position=match.start(),
                        end_position=match.end(),
                        context=context
                    )
                    detections.append(detection)
        
        # Check for potential names (lower confidence)
        for pattern in self.name_patterns:
            for match in re.finditer(pattern, text):
                # Only flag as PHI if in medical context
                surrounding_text = text[max(0, match.start()-50):match.end()+50].lower()
                has_medical_context = any(
                    indicator in surrounding_text 
                    for indicator in self.medical_indicators
                )
                
                if has_medical_context:
                    detection = PHIDetection(
                        phi_type=PHIType.NAME,
                        value=match.group(),
                        confidence=0.7,  # Lower confidence for name detection
                        start_position=match.start(),
                        end_position=match.end(),
                        context=context
                    )
                    detections.append(detection)
        
        return detections
    
    def scrub_phi(self, text: str, replacement: str = "[REDACTED]") -> Tuple[str, List[PHIDetection]]:
        """Remove PHI from text and return cleaned text with detections"""
        detections = self.detect_phi(text)
        
        # Sort detections by position (reverse order for safe replacement)
        detections.sort(key=lambda x: x.start_position, reverse=True)
        
        scrubbed_text = text
        for detection in detections:
            scrubbed_text = (
                scrubbed_text[:detection.start_position] + 
                replacement + 
                scrubbed_text[detection.end_position:]
            )
        
        return scrubbed_text, detections

class HIPAAValidator:
    """Comprehensive HIPAA compliance validator"""
    
    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.STANDARD):
        self.compliance_level = compliance_level
        self.phi_detector = PHIDetector()
        self.violation_log: List[ComplianceViolation] = []
        
        # Compliance requirements by level
        self.requirements = {
            ComplianceLevel.BASIC: {
                "encrypt_phi": True,
                "access_logging": False,
                "phi_detection": False,
                "data_retention_days": 2555,  # 7 years
                "minimum_password_length": 8
            },
            ComplianceLevel.STANDARD: {
                "encrypt_phi": True,
                "access_logging": True,
                "phi_detection": True,
                "data_retention_days": 2555,
                "minimum_password_length": 12,
                "require_2fa": False
            },
            ComplianceLevel.STRICT: {
                "encrypt_phi": True,
                "access_logging": True,
                "phi_detection": True,
                "data_retention_days": 2555,
                "minimum_password_length": 16,
                "require_2fa": True,
                "end_to_end_encryption": True
            }
        }
    
    def validate_data_input(self, data: Dict[str, Any], context: str = "api_input") -> Dict[str, Any]:
        """Validate input data for HIPAA compliance"""
        violations = []
        cleaned_data = data.copy()
        
        # Check for PHI in input data
        for key, value in data.items():
            if isinstance(value, str):
                detections = self.phi_detector.detect_phi(value, f"{context}.{key}")
                
                if detections:
                    violation = ComplianceViolation(
                        violation_type="PHI_IN_INPUT",
                        severity="HIGH",
                        description=f"PHI detected in input field '{key}'",
                        location=f"{context}.{key}",
                        phi_detected=detections,
                        timestamp=datetime.utcnow(),
                        remediation="Sanitize input or implement PHI handling procedures"
                    )
                    violations.append(violation)
                    
                    # Scrub PHI from data
                    scrubbed_value, _ = self.phi_detector.scrub_phi(value)
                    cleaned_data[key] = scrubbed_value
        
        # Log violations
        self.violation_log.extend(violations)
        
        return {
            "cleaned_data": cleaned_data,
            "violations": violations,
            "compliance_status": "VIOLATION" if violations else "COMPLIANT"
        }
    
    def validate_log_entry(self, log_message: str, level: str = "INFO") -> Dict[str, Any]:
        """Validate log entry for PHI exposure"""
        detections = self.phi_detector.detect_phi(log_message, "log_entry")
        violations = []
        
        if detections:
            violation = ComplianceViolation(
                violation_type="PHI_IN_LOGS",
                severity="CRITICAL",
                description="PHI detected in log entry - potential HIPAA violation",
                location="logging_system",
                phi_detected=detections,
                timestamp=datetime.utcnow(),
                remediation="Implement PHI scrubbing in logging system"
            )
            violations.append(violation)
        
        # Create safe log entry
        safe_message, _ = self.phi_detector.scrub_phi(log_message, "[PHI_REDACTED]")
        
        return {
            "safe_message": safe_message,
            "violations": violations,
            "phi_detected": len(detections) > 0
        }
    
    def validate_api_response(self, response_data: Any, endpoint: str = "unknown") -> Dict[str, Any]:
        """Validate API response for PHI exposure"""
        violations = []
        
        # Convert response to string for analysis
        if isinstance(response_data, dict):
            response_text = json.dumps(response_data, default=str)
        else:
            response_text = str(response_data)
        
        detections = self.phi_detector.detect_phi(response_text, f"api_response.{endpoint}")
        
        if detections:
            violation = ComplianceViolation(
                violation_type="PHI_IN_RESPONSE",
                severity="CRITICAL",
                description=f"PHI detected in API response from endpoint '{endpoint}'",
                location=f"api_response.{endpoint}",
                phi_detected=detections,
                timestamp=datetime.utcnow(),
                remediation="Implement response filtering to remove PHI"
            )
            violations.append(violation)
        
        self.violation_log.extend(violations)
        
        return {
            "violations": violations,
            "phi_count": len(detections),
            "compliance_status": "VIOLATION" if violations else "COMPLIANT"
        }
    
    def validate_user_session(self, user_data: Dict[str, Any], session_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user session for HIPAA compliance"""
        violations = []
        requirements = self.requirements[self.compliance_level]
        
        # Check session timeout
        if "last_activity" in session_info:
            last_activity = datetime.fromisoformat(session_info["last_activity"])
            if datetime.utcnow() - last_activity > timedelta(minutes=30):
                violation = ComplianceViolation(
                    violation_type="SESSION_TIMEOUT",
                    severity="MEDIUM",
                    description="Session timeout exceeded HIPAA requirements",
                    location="session_management",
                    phi_detected=[],
                    timestamp=datetime.utcnow(),
                    remediation="Implement automatic session timeout"
                )
                violations.append(violation)
        
        # Check password requirements
        if "password_strength" in user_data:
            min_length = requirements.get("minimum_password_length", 8)
            if user_data["password_strength"] < min_length:
                violation = ComplianceViolation(
                    violation_type="WEAK_PASSWORD",
                    severity="HIGH",
                    description=f"Password does not meet minimum length requirement ({min_length} chars)",
                    location="authentication",
                    phi_detected=[],
                    timestamp=datetime.utcnow(),
                    remediation=f"Require passwords with at least {min_length} characters"
                )
                violations.append(violation)
        
        # Check 2FA requirement for strict mode
        if requirements.get("require_2fa") and not user_data.get("has_2fa"):
            violation = ComplianceViolation(
                violation_type="MISSING_2FA",
                severity="HIGH",
                description="Two-factor authentication required but not enabled",
                location="authentication",
                phi_detected=[],
                timestamp=datetime.utcnow(),
                remediation="Enable two-factor authentication"
            )
            violations.append(violation)
        
        self.violation_log.extend(violations)
        
        return {
            "violations": violations,
            "compliance_status": "VIOLATION" if violations else "COMPLIANT",
            "session_valid": len(violations) == 0
        }
    
    def audit_data_access(self, user_id: str, resource: str, action: str, 
                         phi_accessed: bool = False) -> Dict[str, Any]:
        """Log data access for HIPAA audit trail"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:16],  # Anonymized
            "resource": resource,
            "action": action,
            "phi_accessed": phi_accessed,
            "ip_address_hash": "",  # Should be set by caller
            "user_agent": "",       # Should be set by caller
            "session_id_hash": ""   # Should be set by caller
        }
        
        # Write to audit log (implement persistent storage)
        self._write_audit_log(audit_entry)
        
        return {
            "audit_logged": True,
            "audit_id": audit_entry["timestamp"] + "_" + audit_entry["user_id"][:8]
        }
    
    def _write_audit_log(self, audit_entry: Dict[str, Any]):
        """Write audit entry to persistent storage"""
        # In production, this should write to a secure, tamper-proof audit log
        audit_dir = Path("logs/audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        audit_file = audit_dir / f"hipaa_audit_{datetime.utcnow().strftime('%Y%m%d')}.log"
        
        try:
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write HIPAA audit log: {e}")
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        total_violations = len(self.violation_log)
        violation_by_type = {}
        violation_by_severity = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        for violation in self.violation_log:
            # Count by type
            if violation.violation_type not in violation_by_type:
                violation_by_type[violation.violation_type] = 0
            violation_by_type[violation.violation_type] += 1
            
            # Count by severity
            violation_by_severity[violation.severity] += 1
        
        return {
            "compliance_level": self.compliance_level.value,
            "total_violations": total_violations,
            "violations_by_type": violation_by_type,
            "violations_by_severity": violation_by_severity,
            "recent_violations": [
                {
                    "type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "timestamp": v.timestamp.isoformat()
                }
                for v in self.violation_log[-10:]  # Last 10 violations
            ],
            "recommendations": self._get_compliance_recommendations()
        }
    
    def _get_compliance_recommendations(self) -> List[str]:
        """Get compliance improvement recommendations"""
        recommendations = []
        
        violation_types = set(v.violation_type for v in self.violation_log)
        
        if "PHI_IN_LOGS" in violation_types:
            recommendations.append("Implement automatic PHI scrubbing in logging system")
        
        if "PHI_IN_RESPONSE" in violation_types:
            recommendations.append("Add PHI filtering to API response handling")
        
        if "SESSION_TIMEOUT" in violation_types:
            recommendations.append("Implement stricter session timeout policies")
        
        if "WEAK_PASSWORD" in violation_types:
            recommendations.append("Enforce stronger password requirements")
        
        return recommendations
    
    def clear_violation_history(self):
        """Clear violation history (for testing/maintenance)"""
        self.violation_log.clear()

# Global HIPAA validator instance
try:
    # Determine compliance level from environment
    compliance_level_str = os.getenv("HIPAA_COMPLIANCE_LEVEL", "standard").lower()
    compliance_level = ComplianceLevel(compliance_level_str)
    
    hipaa_validator = HIPAAValidator(compliance_level)
    logger.info(f"HIPAA validator initialized with {compliance_level.value} compliance level")
    
except Exception as e:
    logger.error(f"Failed to initialize HIPAA validator: {e}")
    # Fallback to basic compliance
    hipaa_validator = HIPAAValidator(ComplianceLevel.BASIC)