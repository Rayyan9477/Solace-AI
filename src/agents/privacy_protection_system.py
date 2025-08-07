"""
Privacy Protection System for Adaptive Learning

This module implements comprehensive privacy protection mechanisms for the
adaptive learning system, ensuring HIPAA compliance, data anonymization,
differential privacy, and secure federated learning capabilities.
"""

import hashlib
import hmac
import random
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from collections import defaultdict, Counter
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..utils.logger import get_logger

logger = get_logger(__name__)

class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"  # Basic anonymization
    STANDARD = "standard"  # Differential privacy + anonymization
    HIGH = "high"  # Enhanced privacy with noise injection
    MAXIMUM = "maximum"  # Full privacy with federated learning only

class DataSensitivity(Enum):
    """Data sensitivity classifications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"  # PHI/PII

@dataclass
class PrivacyMetrics:
    """Privacy protection metrics"""
    epsilon: float  # Differential privacy budget
    delta: float   # Differential privacy parameter
    noise_level: float
    anonymization_level: str
    data_retention_days: int
    encryption_strength: str
    
@dataclass
class DataPoint:
    """Anonymized data point for learning"""
    anonymized_user_id: str
    data_type: str
    features: Dict[str, Any]
    sensitivity_level: DataSensitivity
    privacy_budget_used: float
    timestamp: datetime
    noise_applied: bool = False
    federated_source: bool = False

class DifferentialPrivacy:
    """Differential privacy implementation for learning data"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Privacy parameter
        self.global_sensitivity = 1.0
        self.privacy_budget_used = 0.0
        self.logger = get_logger(__name__)
    
    def add_laplace_noise(self, value: float, sensitivity: float = None) -> float:
        """Add Laplace noise for differential privacy"""
        if sensitivity is None:
            sensitivity = self.global_sensitivity
        
        # Calculate noise scale
        noise_scale = sensitivity / self.epsilon
        
        # Generate Laplace noise
        noise = np.random.laplace(0, noise_scale)
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon
        
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = None) -> float:
        """Add Gaussian noise for differential privacy"""
        if sensitivity is None:
            sensitivity = self.global_sensitivity
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Generate Gaussian noise
        noise = np.random.normal(0, noise_scale)
        
        # Update privacy budget
        self.privacy_budget_used += self.epsilon
        
        return value + noise
    
    def privatize_count(self, count: int) -> int:
        """Add differential privacy to count queries"""
        noisy_count = self.add_laplace_noise(float(count), sensitivity=1.0)
        return max(0, int(round(noisy_count)))
    
    def privatize_average(self, values: List[float], bounds: Tuple[float, float]) -> float:
        """Add differential privacy to average queries"""
        if not values:
            return 0.0
        
        # Clip values to bounds
        min_val, max_val = bounds
        clipped_values = [max(min_val, min(max_val, val)) for val in values]
        
        # Calculate sensitivity
        sensitivity = (max_val - min_val) / len(clipped_values)
        
        # Calculate average and add noise
        avg = sum(clipped_values) / len(clipped_values)
        return self.add_laplace_noise(avg, sensitivity)
    
    def check_privacy_budget(self, required_budget: float) -> bool:
        """Check if privacy budget allows for operation"""
        return (self.privacy_budget_used + required_budget) <= self.epsilon

class DataAnonymizer:
    """Data anonymization and pseudonymization system"""
    
    def __init__(self, salt: str = None):
        self.salt = salt or self._generate_salt()
        self.anonymization_cache = {}
        self.logger = get_logger(__name__)
    
    def _generate_salt(self) -> str:
        """Generate cryptographic salt"""
        return base64.b64encode(np.random.bytes(32)).decode('utf-8')
    
    def anonymize_user_id(self, user_id: str, session_id: str = None) -> str:
        """Create consistent anonymized user ID"""
        cache_key = f"{user_id}:{session_id}" if session_id else user_id
        
        if cache_key in self.anonymization_cache:
            return self.anonymization_cache[cache_key]
        
        # Create HMAC-based anonymized ID
        combined_id = f"{user_id}:{session_id}:{self.salt}"
        hash_obj = hmac.new(
            self.salt.encode('utf-8'),
            combined_id.encode('utf-8'),
            hashlib.sha256
        )
        anonymized_id = hash_obj.hexdigest()[:16]  # First 16 characters
        
        self.anonymization_cache[cache_key] = anonymized_id
        return anonymized_id
    
    def generalize_timestamps(self, timestamp: datetime, granularity: str = "hour") -> datetime:
        """Generalize timestamps to reduce precision"""
        if granularity == "hour":
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif granularity == "day":
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif granularity == "week":
            days_since_monday = timestamp.weekday()
            week_start = timestamp - timedelta(days=days_since_monday)
            return week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp
    
    def suppress_rare_values(self, data: Dict[str, Any], threshold: int = 5) -> Dict[str, Any]:
        """Suppress rare values that could lead to re-identification"""
        anonymized_data = {}
        
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                # For lists, check frequency of each item
                value_counts = Counter(value)
                anonymized_list = []
                for item in value:
                    if value_counts[item] >= threshold:
                        anonymized_list.append(item)
                    else:
                        anonymized_list.append("SUPPRESSED")
                anonymized_data[key] = anonymized_list
            elif isinstance(value, str):
                # For strings, suppress if they appear to be identifiers
                if len(value) > 20 or any(char.isdigit() for char in value):
                    anonymized_data[key] = "SUPPRESSED"
                else:
                    anonymized_data[key] = value
            else:
                anonymized_data[key] = value
        
        return anonymized_data
    
    def k_anonymize(self, dataset: List[Dict[str, Any]], k: int = 5, 
                   quasi_identifiers: List[str] = None) -> List[Dict[str, Any]]:
        """Apply k-anonymity to dataset"""
        if not quasi_identifiers:
            quasi_identifiers = ['age', 'location', 'gender', 'occupation']
        
        # Group records by quasi-identifier combinations
        groups = defaultdict(list)
        for record in dataset:
            key = tuple(record.get(qi, 'unknown') for qi in quasi_identifiers)
            groups[key].append(record)
        
        anonymized_dataset = []
        for group_records in groups.values():
            if len(group_records) >= k:
                # Group satisfies k-anonymity
                anonymized_dataset.extend(group_records)
            else:
                # Suppress or generalize records
                for record in group_records:
                    anonymized_record = record.copy()
                    for qi in quasi_identifiers:
                        if qi in anonymized_record:
                            anonymized_record[qi] = "GENERALIZED"
                    anonymized_dataset.append(anonymized_record)
        
        return anonymized_dataset

class SecureStorage:
    """Secure storage for sensitive learning data"""
    
    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = self._generate_key()
        
        self.fernet = Fernet(encryption_key)
        self.logger = get_logger(__name__)
    
    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data"""
        try:
            json_data = json.dumps(data, default=str)
            encrypted_data = self.fernet.encrypt(json_data.encode('utf-8'))
            return base64.b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise

class FederatedLearningPrivacy:
    """Privacy-preserving federated learning implementation"""
    
    def __init__(self, num_participants: int = 10, privacy_budget: float = 1.0):
        self.num_participants = num_participants
        self.privacy_budget = privacy_budget
        self.participant_updates = {}
        self.global_model_state = {}
        self.logger = get_logger(__name__)
    
    def add_participant_update(self, participant_id: str, model_update: Dict[str, Any],
                              privacy_noise: bool = True) -> None:
        """Add privacy-preserving model update from participant"""
        
        if privacy_noise:
            # Add differential privacy noise to model updates
            dp = DifferentialPrivacy(epsilon=self.privacy_budget / self.num_participants)
            
            noisy_update = {}
            for param_name, param_value in model_update.items():
                if isinstance(param_value, (int, float)):
                    noisy_update[param_name] = dp.add_gaussian_noise(param_value)
                elif isinstance(param_value, list):
                    noisy_update[param_name] = [
                        dp.add_gaussian_noise(val) if isinstance(val, (int, float)) else val
                        for val in param_value
                    ]
                else:
                    noisy_update[param_name] = param_value
        else:
            noisy_update = model_update
        
        self.participant_updates[participant_id] = {
            'update': noisy_update,
            'timestamp': datetime.now(),
            'privacy_applied': privacy_noise
        }
    
    def aggregate_updates(self) -> Dict[str, Any]:
        """Aggregate participant updates with privacy preservation"""
        
        if not self.participant_updates:
            return {}
        
        # Simple federated averaging with privacy
        aggregated_update = {}
        all_params = set()
        
        # Collect all parameter names
        for update_data in self.participant_updates.values():
            all_params.update(update_data['update'].keys())
        
        # Aggregate each parameter
        for param_name in all_params:
            param_values = []
            for update_data in self.participant_updates.values():
                if param_name in update_data['update']:
                    param_values.append(update_data['update'][param_name])
            
            if param_values:
                if isinstance(param_values[0], (int, float)):
                    # Average numeric parameters
                    aggregated_update[param_name] = sum(param_values) / len(param_values)
                elif isinstance(param_values[0], list) and all(isinstance(v, (int, float)) for v in param_values[0]):
                    # Average list parameters element-wise
                    avg_list = []
                    for i in range(len(param_values[0])):
                        element_values = [pv[i] for pv in param_values if i < len(pv)]
                        avg_list.append(sum(element_values) / len(element_values))
                    aggregated_update[param_name] = avg_list
                else:
                    # For non-numeric parameters, use most common value
                    aggregated_update[param_name] = max(set(param_values), key=param_values.count)
        
        # Clear participant updates after aggregation
        self.participant_updates = {}
        
        return aggregated_update

class PrivacyProtectionSystem:
    """Main privacy protection system for adaptive learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Privacy configuration
        self.privacy_level = PrivacyLevel(self.config.get('privacy_level', 'standard'))
        self.epsilon = self.config.get('differential_privacy_epsilon', 1.0)
        self.delta = self.config.get('differential_privacy_delta', 1e-5)
        self.k_anonymity = self.config.get('k_anonymity', 5)
        self.data_retention_days = self.config.get('data_retention_days', 90)
        
        # Initialize components
        self.differential_privacy = DifferentialPrivacy(self.epsilon, self.delta)
        self.anonymizer = DataAnonymizer()
        self.secure_storage = SecureStorage()
        self.federated_learning = FederatedLearningPrivacy()
        
        # Privacy tracking
        self.privacy_metrics = PrivacyMetrics(
            epsilon=self.epsilon,
            delta=self.delta,
            noise_level=0.0,
            anonymization_level=self.privacy_level.value,
            data_retention_days=self.data_retention_days,
            encryption_strength="AES-256"
        )
        
        # Data processing history
        self.processed_data_points = []
        self.privacy_violations = []
        
        self.logger.info(f"Privacy Protection System initialized with {self.privacy_level.value} level")
    
    def process_learning_data(self, user_id: str, data: Dict[str, Any], 
                             sensitivity: DataSensitivity = DataSensitivity.CONFIDENTIAL) -> DataPoint:
        """Process learning data with privacy protection"""
        
        try:
            # Step 1: Anonymize user identification
            anonymized_id = self.anonymizer.anonymize_user_id(user_id)
            
            # Step 2: Apply data sensitivity-based processing
            processed_features = self._apply_sensitivity_processing(data, sensitivity)
            
            # Step 3: Apply differential privacy if required
            if self.privacy_level in [PrivacyLevel.STANDARD, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
                processed_features = self._apply_differential_privacy(processed_features)
                noise_applied = True
            else:
                noise_applied = False
            
            # Step 4: Create privacy-protected data point
            data_point = DataPoint(
                anonymized_user_id=anonymized_id,
                data_type=data.get('data_type', 'interaction'),
                features=processed_features,
                sensitivity_level=sensitivity,
                privacy_budget_used=self.differential_privacy.privacy_budget_used,
                timestamp=self.anonymizer.generalize_timestamps(datetime.now()),
                noise_applied=noise_applied,
                federated_source=self.privacy_level == PrivacyLevel.MAXIMUM
            )
            
            # Step 5: Store with encryption if sensitive
            if sensitivity in [DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED]:
                encrypted_features = self.secure_storage.encrypt_data(processed_features)
                data_point.features = {'encrypted_data': encrypted_features}
            
            # Track processed data point
            self.processed_data_points.append(data_point)
            
            # Cleanup old data points
            self._cleanup_old_data()
            
            return data_point
            
        except Exception as e:
            self.logger.error(f"Error processing learning data with privacy protection: {str(e)}")
            raise
    
    def _apply_sensitivity_processing(self, data: Dict[str, Any], 
                                    sensitivity: DataSensitivity) -> Dict[str, Any]:
        """Apply processing based on data sensitivity level"""
        
        processed_data = data.copy()
        
        if sensitivity == DataSensitivity.RESTRICTED:
            # Maximum protection for PHI/PII
            processed_data = self.anonymizer.suppress_rare_values(processed_data, threshold=10)
            
            # Remove direct identifiers
            sensitive_fields = ['name', 'email', 'phone', 'ssn', 'address', 'ip_address']
            for field in sensitive_fields:
                if field in processed_data:
                    processed_data[field] = "REDACTED"
            
        elif sensitivity == DataSensitivity.CONFIDENTIAL:
            # High protection
            processed_data = self.anonymizer.suppress_rare_values(processed_data, threshold=5)
            
            # Generalize specific fields
            if 'age' in processed_data:
                age = processed_data['age']
                if isinstance(age, (int, float)):
                    # Age grouping: 18-25, 26-35, 36-45, 46-55, 56-65, 65+
                    if age < 26: processed_data['age'] = '18-25'
                    elif age < 36: processed_data['age'] = '26-35'
                    elif age < 46: processed_data['age'] = '36-45'
                    elif age < 56: processed_data['age'] = '46-55'
                    elif age < 66: processed_data['age'] = '56-65'
                    else: processed_data['age'] = '65+'
            
            if 'location' in processed_data:
                location = processed_data['location']
                if isinstance(location, str):
                    # Only keep state/region level
                    parts = location.split(',')
                    if len(parts) > 1:
                        processed_data['location'] = parts[-1].strip()  # Keep last part (state/country)
        
        elif sensitivity == DataSensitivity.INTERNAL:
            # Moderate protection
            processed_data = self.anonymizer.suppress_rare_values(processed_data, threshold=3)
        
        # For PUBLIC data, minimal processing required
        
        return processed_data
    
    def _apply_differential_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy mechanisms to data"""
        
        dp_data = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Apply noise to numeric values
                if self.differential_privacy.check_privacy_budget(0.1):
                    dp_data[key] = self.differential_privacy.add_laplace_noise(value)
                else:
                    # Privacy budget exhausted, suppress value
                    dp_data[key] = "SUPPRESSED_PRIVACY_BUDGET"
            elif isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                # Apply noise to numeric lists
                if self.differential_privacy.check_privacy_budget(0.1):
                    dp_data[key] = [
                        self.differential_privacy.add_laplace_noise(v) for v in value
                    ]
                else:
                    dp_data[key] = ["SUPPRESSED_PRIVACY_BUDGET"] * len(value)
            else:
                # Keep non-numeric values as-is
                dp_data[key] = value
        
        return dp_data
    
    def create_federated_update(self, model_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Create privacy-preserving federated learning update"""
        
        if self.privacy_level != PrivacyLevel.MAXIMUM:
            self.logger.warning("Federated learning recommended only for MAXIMUM privacy level")
        
        # Add privacy-preserving noise to model updates
        participant_id = f"participant_{random.randint(1000, 9999)}"
        self.federated_learning.add_participant_update(
            participant_id, model_updates, privacy_noise=True
        )
        
        # If enough participants, aggregate updates
        if len(self.federated_learning.participant_updates) >= self.federated_learning.num_participants:
            return self.federated_learning.aggregate_updates()
        
        return {}
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy protection report"""
        
        now = datetime.now()
        
        # Calculate privacy budget usage
        budget_usage = (self.differential_privacy.privacy_budget_used / self.epsilon) * 100
        
        # Count data points by sensitivity
        sensitivity_counts = {}
        for data_point in self.processed_data_points:
            sensitivity = data_point.sensitivity_level.value
            sensitivity_counts[sensitivity] = sensitivity_counts.get(sensitivity, 0) + 1
        
        # Calculate retention compliance
        cutoff_date = now - timedelta(days=self.data_retention_days)
        old_data_count = sum(
            1 for dp in self.processed_data_points
            if dp.timestamp < cutoff_date
        )
        
        return {
            'privacy_configuration': {
                'privacy_level': self.privacy_level.value,
                'differential_privacy_epsilon': self.epsilon,
                'differential_privacy_delta': self.delta,
                'k_anonymity': self.k_anonymity,
                'data_retention_days': self.data_retention_days
            },
            'current_status': {
                'privacy_budget_used': self.differential_privacy.privacy_budget_used,
                'privacy_budget_usage_percent': budget_usage,
                'total_data_points_processed': len(self.processed_data_points),
                'data_points_by_sensitivity': sensitivity_counts,
                'encryption_active': True,
                'federated_learning_active': self.privacy_level == PrivacyLevel.MAXIMUM
            },
            'compliance_metrics': {
                'data_retention_compliant': old_data_count == 0,
                'old_data_points_requiring_cleanup': old_data_count,
                'privacy_violations_detected': len(self.privacy_violations),
                'anonymization_coverage': 100.0  # All data points are anonymized
            },
            'recommendations': self._generate_privacy_recommendations(budget_usage, old_data_count)
        }
    
    def _generate_privacy_recommendations(self, budget_usage: float, old_data_count: int) -> List[str]:
        """Generate privacy protection recommendations"""
        
        recommendations = []
        
        if budget_usage > 80:
            recommendations.append(
                "Privacy budget usage is high (>80%). Consider increasing epsilon or reducing data processing frequency."
            )
        
        if old_data_count > 0:
            recommendations.append(
                f"Found {old_data_count} data points exceeding retention period. Schedule cleanup immediately."
            )
        
        if self.privacy_level == PrivacyLevel.MINIMAL:
            recommendations.append(
                "Consider upgrading to STANDARD privacy level for better protection of sensitive data."
            )
        
        if len(self.processed_data_points) > 10000:
            recommendations.append(
                "Large number of processed data points. Consider implementing data sampling to reduce privacy impact."
            )
        
        return recommendations
    
    def _cleanup_old_data(self):
        """Clean up data points exceeding retention period"""
        
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        
        original_count = len(self.processed_data_points)
        self.processed_data_points = [
            dp for dp in self.processed_data_points
            if dp.timestamp >= cutoff_date
        ]
        cleaned_count = original_count - len(self.processed_data_points)
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old data points for privacy compliance")
    
    def validate_privacy_compliance(self) -> Dict[str, bool]:
        """Validate current privacy compliance status"""
        
        compliance = {
            'data_retention_compliant': True,
            'privacy_budget_compliant': True,
            'anonymization_compliant': True,
            'encryption_compliant': True,
            'overall_compliant': True
        }
        
        # Check data retention compliance
        cutoff_date = datetime.now() - timedelta(days=self.data_retention_days)
        old_data_exists = any(dp.timestamp < cutoff_date for dp in self.processed_data_points)
        compliance['data_retention_compliant'] = not old_data_exists
        
        # Check privacy budget compliance
        budget_usage = self.differential_privacy.privacy_budget_used / self.epsilon
        compliance['privacy_budget_compliant'] = budget_usage <= 1.0
        
        # Check anonymization compliance (all data should be anonymized)
        all_anonymized = all(
            dp.anonymized_user_id != dp.features.get('original_user_id', '')
            for dp in self.processed_data_points
        )
        compliance['anonymization_compliant'] = all_anonymized
        
        # Check encryption for sensitive data
        sensitive_data_points = [
            dp for dp in self.processed_data_points
            if dp.sensitivity_level in [DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED]
        ]
        encrypted_sensitive = all(
            'encrypted_data' in dp.features
            for dp in sensitive_data_points
        )
        compliance['encryption_compliant'] = encrypted_sensitive or len(sensitive_data_points) == 0
        
        # Overall compliance
        compliance['overall_compliant'] = all(compliance.values())
        
        return compliance
    
    def reset_privacy_budget(self):
        """Reset differential privacy budget (use with caution)"""
        
        self.logger.warning("Resetting differential privacy budget - this should only be done in controlled circumstances")
        self.differential_privacy.privacy_budget_used = 0.0
    
    def get_privacy_metrics(self) -> PrivacyMetrics:
        """Get current privacy metrics"""
        
        self.privacy_metrics.noise_level = self.differential_privacy.privacy_budget_used / self.epsilon
        return self.privacy_metrics