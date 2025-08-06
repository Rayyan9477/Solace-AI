"""
Configuration Management for SupervisorAgent System.

This module provides comprehensive configuration management for the supervision
system, including clinical guidelines, validation thresholds, and monitoring settings.
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)

class SupervisionMode(Enum):
    """Supervision operation modes."""
    FULL = "full"  # Complete supervision with all features
    MONITORING_ONLY = "monitoring_only"  # Only metrics and monitoring
    AUDIT_ONLY = "audit_only"  # Only audit trail
    DISABLED = "disabled"  # No supervision

class ValidationStrictness(Enum):
    """Validation strictness levels."""
    STRICT = "strict"  # Strict validation with low thresholds
    MODERATE = "moderate"  # Balanced validation
    LENIENT = "lenient"  # Lenient validation for development

@dataclass
class ValidationThresholds:
    """Validation threshold configuration."""
    accuracy_warning: float = 0.7
    accuracy_critical: float = 0.5
    consistency_warning: float = 0.6
    consistency_critical: float = 0.4
    appropriateness_warning: float = 0.6
    appropriateness_critical: float = 0.4
    safety_warning: float = 0.8
    safety_critical: float = 0.6
    response_time_warning: float = 2.0
    response_time_critical: float = 5.0
    blocked_response_rate_warning: float = 0.1
    blocked_response_rate_critical: float = 0.25

@dataclass
class MonitoringConfig:
    """Monitoring system configuration."""
    metrics_collection_enabled: bool = True
    real_time_monitoring: bool = True
    alert_notifications: bool = True
    performance_dashboard: bool = True
    metrics_retention_days: int = 90
    alert_cooldown_minutes: int = 15
    batch_size: int = 100
    collection_interval_seconds: int = 30

@dataclass
class AuditConfig:
    """Audit system configuration."""
    audit_enabled: bool = True
    detailed_logging: bool = True
    compliance_reporting: bool = True
    data_encryption: bool = True
    retention_years: int = 7
    export_format: str = "json"
    compress_exports: bool = True
    audit_database_path: str = "src/data/audit"

@dataclass
class ClinicalConfig:
    """Clinical guidelines configuration."""
    guidelines_enabled: bool = True
    strict_boundary_enforcement: bool = True
    crisis_intervention_required: bool = True
    medication_advice_blocking: bool = True
    diagnostic_limitation_enforcement: bool = True
    cultural_sensitivity_checks: bool = True
    trauma_informed_validation: bool = True

@dataclass
class SupervisionConfig:
    """Complete supervision system configuration."""
    mode: SupervisionMode = SupervisionMode.FULL
    strictness: ValidationStrictness = ValidationStrictness.MODERATE
    validation_thresholds: ValidationThresholds = None
    monitoring: MonitoringConfig = None
    audit: AuditConfig = None
    clinical: ClinicalConfig = None
    
    # System settings
    max_concurrent_validations: int = 10
    validation_timeout_seconds: int = 30
    fallback_mode_on_error: bool = True
    debug_mode: bool = False
    
    def __post_init__(self):
        if self.validation_thresholds is None:
            self.validation_thresholds = ValidationThresholds()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.audit is None:
            self.audit = AuditConfig()
        if self.clinical is None:
            self.clinical = ClinicalConfig()

class SupervisionConfigManager:
    """Manages supervision system configuration."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path or "src/config")
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_path / "supervision.yaml"
        self.current_config: Optional[SupervisionConfig] = None
        
        # Environment-specific configurations
        self.env_configs = {
            "development": self._get_development_config(),
            "testing": self._get_testing_config(),
            "staging": self._get_staging_config(),
            "production": self._get_production_config()
        }
        
        logger.info(f"Configuration manager initialized with path: {self.config_path}")
    
    def load_config(self, environment: str = None) -> SupervisionConfig:
        """Load configuration from file or environment defaults.
        
        Args:
            environment: Environment name (development, testing, staging, production)
            
        Returns:
            SupervisionConfig instance
        """
        try:
            # Try to load from file first
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Convert to SupervisionConfig
                config = self._dict_to_config(config_data)
                logger.info(f"Configuration loaded from file: {self.config_file}")
                
            else:
                # Use environment-specific defaults
                env = environment or os.getenv("ENVIRONMENT", "development")
                config = self.env_configs.get(env, self._get_development_config())
                logger.info(f"Using default configuration for environment: {env}")
            
            self.current_config = config
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Return safe defaults
            return self._get_safe_defaults()
    
    def save_config(self, config: SupervisionConfig):
        """Save configuration to file.
        
        Args:
            config: Configuration to save
        """
        try:
            config_dict = self._config_to_dict(config)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.current_config = config
            logger.info(f"Configuration saved to: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def update_config(self, updates: Dict[str, Any]) -> SupervisionConfig:
        """Update current configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            Updated configuration
        """
        if not self.current_config:
            self.current_config = self.load_config()
        
        try:
            # Apply updates to current config
            updated_dict = self._config_to_dict(self.current_config)
            self._deep_update(updated_dict, updates)
            
            # Convert back to config object
            updated_config = self._dict_to_config(updated_dict)
            
            # Save updated configuration
            self.save_config(updated_config)
            
            logger.info("Configuration updated successfully")
            return updated_config
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise
    
    def get_validation_thresholds(self, strictness: ValidationStrictness = None) -> ValidationThresholds:
        """Get validation thresholds based on strictness level.
        
        Args:
            strictness: Validation strictness level
            
        Returns:
            ValidationThresholds instance
        """
        if not self.current_config:
            self.current_config = self.load_config()
        
        strictness = strictness or self.current_config.strictness
        
        if strictness == ValidationStrictness.STRICT:
            return ValidationThresholds(
                accuracy_warning=0.8,
                accuracy_critical=0.6,
                consistency_warning=0.7,
                consistency_critical=0.5,
                appropriateness_warning=0.7,
                appropriateness_critical=0.5,
                safety_warning=0.9,
                safety_critical=0.8,
                response_time_warning=1.5,
                response_time_critical=3.0,
                blocked_response_rate_warning=0.05,
                blocked_response_rate_critical=0.15
            )
        elif strictness == ValidationStrictness.LENIENT:
            return ValidationThresholds(
                accuracy_warning=0.6,
                accuracy_critical=0.4,
                consistency_warning=0.5,
                consistency_critical=0.3,
                appropriateness_warning=0.5,
                appropriateness_critical=0.3,
                safety_warning=0.7,
                safety_critical=0.5,
                response_time_warning=3.0,
                response_time_critical=7.0,
                blocked_response_rate_warning=0.15,
                blocked_response_rate_critical=0.35
            )
        else:  # MODERATE
            return self.current_config.validation_thresholds
    
    def _get_development_config(self) -> SupervisionConfig:
        """Get development environment configuration."""
        return SupervisionConfig(
            mode=SupervisionMode.FULL,
            strictness=ValidationStrictness.LENIENT,
            debug_mode=True,
            monitoring=MonitoringConfig(
                metrics_retention_days=30,
                real_time_monitoring=True,
                alert_notifications=False
            ),
            audit=AuditConfig(
                detailed_logging=True,
                retention_years=1,
                data_encryption=False
            ),
            clinical=ClinicalConfig(
                strict_boundary_enforcement=False,
                crisis_intervention_required=True
            )
        )
    
    def _get_testing_config(self) -> SupervisionConfig:
        """Get testing environment configuration."""
        return SupervisionConfig(
            mode=SupervisionMode.MONITORING_ONLY,
            strictness=ValidationStrictness.MODERATE,
            debug_mode=True,
            monitoring=MonitoringConfig(
                metrics_retention_days=7,
                real_time_monitoring=False,
                alert_notifications=False
            ),
            audit=AuditConfig(
                audit_enabled=False,
                data_encryption=False
            )
        )
    
    def _get_staging_config(self) -> SupervisionConfig:
        """Get staging environment configuration."""
        return SupervisionConfig(
            mode=SupervisionMode.FULL,
            strictness=ValidationStrictness.MODERATE,
            debug_mode=False,
            monitoring=MonitoringConfig(
                metrics_retention_days=60,
                real_time_monitoring=True,
                alert_notifications=True
            ),
            audit=AuditConfig(
                detailed_logging=True,
                retention_years=3,
                data_encryption=True
            ),
            clinical=ClinicalConfig(
                strict_boundary_enforcement=True,
                crisis_intervention_required=True
            )
        )
    
    def _get_production_config(self) -> SupervisionConfig:
        """Get production environment configuration."""
        return SupervisionConfig(
            mode=SupervisionMode.FULL,
            strictness=ValidationStrictness.STRICT,
            debug_mode=False,
            max_concurrent_validations=20,
            monitoring=MonitoringConfig(
                metrics_retention_days=365,
                real_time_monitoring=True,
                alert_notifications=True,
                performance_dashboard=True
            ),
            audit=AuditConfig(
                detailed_logging=True,
                compliance_reporting=True,
                retention_years=7,
                data_encryption=True,
                compress_exports=True
            ),
            clinical=ClinicalConfig(
                strict_boundary_enforcement=True,
                crisis_intervention_required=True,
                medication_advice_blocking=True,
                diagnostic_limitation_enforcement=True,
                cultural_sensitivity_checks=True,
                trauma_informed_validation=True
            )
        )
    
    def _get_safe_defaults(self) -> SupervisionConfig:
        """Get safe default configuration for error cases."""
        return SupervisionConfig(
            mode=SupervisionMode.MONITORING_ONLY,
            strictness=ValidationStrictness.MODERATE,
            fallback_mode_on_error=True
        )
    
    def _config_to_dict(self, config: SupervisionConfig) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        config_dict = asdict(config)
        
        # Convert enums to strings
        config_dict["mode"] = config.mode.value
        config_dict["strictness"] = config.strictness.value
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SupervisionConfig:
        """Convert dictionary to config object."""
        # Convert enum strings back to enums
        if "mode" in config_dict:
            config_dict["mode"] = SupervisionMode(config_dict["mode"])
        if "strictness" in config_dict:
            config_dict["strictness"] = ValidationStrictness(config_dict["strictness"])
        
        # Handle nested dataclasses
        if "validation_thresholds" in config_dict and config_dict["validation_thresholds"]:
            config_dict["validation_thresholds"] = ValidationThresholds(**config_dict["validation_thresholds"])
        
        if "monitoring" in config_dict and config_dict["monitoring"]:
            config_dict["monitoring"] = MonitoringConfig(**config_dict["monitoring"])
        
        if "audit" in config_dict and config_dict["audit"]:
            config_dict["audit"] = AuditConfig(**config_dict["audit"])
        
        if "clinical" in config_dict and config_dict["clinical"]:
            config_dict["clinical"] = ClinicalConfig(**config_dict["clinical"])
        
        return SupervisionConfig(**config_dict)
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep update target dictionary with source values."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def validate_config(self, config: SupervisionConfig) -> List[str]:
        """Validate configuration and return any errors.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate thresholds
        thresholds = config.validation_thresholds
        if thresholds.accuracy_critical >= thresholds.accuracy_warning:
            errors.append("Accuracy critical threshold must be less than warning threshold")
        
        if thresholds.consistency_critical >= thresholds.consistency_warning:
            errors.append("Consistency critical threshold must be less than warning threshold")
        
        if thresholds.safety_critical >= thresholds.safety_warning:
            errors.append("Safety critical threshold must be less than warning threshold")
        
        # Validate monitoring config
        if config.monitoring.metrics_retention_days < 1:
            errors.append("Metrics retention must be at least 1 day")
        
        if config.monitoring.collection_interval_seconds < 10:
            errors.append("Collection interval must be at least 10 seconds")
        
        # Validate audit config
        if config.audit.retention_years < 1:
            errors.append("Audit retention must be at least 1 year for compliance")
        
        # Validate system limits
        if config.max_concurrent_validations < 1:
            errors.append("Max concurrent validations must be at least 1")
        
        if config.validation_timeout_seconds < 5:
            errors.append("Validation timeout must be at least 5 seconds")
        
        return errors
    
    def export_config_template(self, output_path: str):
        """Export configuration template file.
        
        Args:
            output_path: Path to save template file
        """
        template_config = self._get_production_config()
        
        # Add comments and documentation
        template_dict = self._config_to_dict(template_config)
        template_dict["_documentation"] = {
            "description": "SupervisorAgent Configuration Template",
            "modes": {
                "full": "Complete supervision with all features enabled",
                "monitoring_only": "Only metrics collection and monitoring",
                "audit_only": "Only audit trail logging",
                "disabled": "No supervision features"
            },
            "strictness_levels": {
                "strict": "Strict validation with low error tolerance",
                "moderate": "Balanced validation for production use",
                "lenient": "Relaxed validation for development"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration template exported to: {output_path}")


# Global configuration manager instance
_config_manager = None

def get_supervision_config_manager() -> SupervisionConfigManager:
    """Get global supervision configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SupervisionConfigManager()
    return _config_manager

def load_supervision_config(environment: str = None) -> SupervisionConfig:
    """Load supervision configuration."""
    manager = get_supervision_config_manager()
    return manager.load_config(environment)