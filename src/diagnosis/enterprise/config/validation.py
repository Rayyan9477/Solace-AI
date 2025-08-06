"""
Configuration validation for Enterprise Diagnostic Pipeline
"""

from typing import Dict, Any, List, Tuple
import logging
from .base_config import EnterpriseConfig, ModelConfig, ClinicalConfig, PrivacyConfig, PerformanceConfig

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration parameters for the enterprise pipeline"""
    
    @staticmethod
    def validate_model_config(config: ModelConfig) -> Tuple[bool, List[str]]:
        """Validate model configuration parameters"""
        errors = []
        
        # Validate fusion_dim
        if config.fusion_dim <= 0 or config.fusion_dim % 64 != 0:
            errors.append("fusion_dim must be positive and divisible by 64")
        
        # Validate attention_heads
        if config.attention_heads <= 0 or config.fusion_dim % config.attention_heads != 0:
            errors.append("attention_heads must be positive and fusion_dim must be divisible by it")
        
        # Validate dropout
        if not 0.0 <= config.dropout <= 1.0:
            errors.append("dropout must be between 0.0 and 1.0")
        
        # Validate temporal config
        if config.temporal_hidden_dim <= 0:
            errors.append("temporal_hidden_dim must be positive")
        
        if config.temporal_layers <= 0:
            errors.append("temporal_layers must be positive")
        
        # Validate uncertainty samples
        if config.uncertainty_samples <= 0:
            errors.append("uncertainty_samples must be positive")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_clinical_config(config: ClinicalConfig) -> Tuple[bool, List[str]]:
        """Validate clinical configuration parameters"""
        errors = []
        
        # Validate confidence threshold
        if not 0.0 <= config.confidence_threshold <= 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        # Validate severity thresholds
        if not all(0.0 <= v <= 1.0 for v in config.severity_thresholds.values()):
            errors.append("All severity thresholds must be between 0.0 and 1.0")
        
        # Check threshold ordering
        thresholds = config.severity_thresholds
        if 'mild' in thresholds and 'moderate' in thresholds and 'severe' in thresholds:
            if not (thresholds['mild'] < thresholds['moderate'] < thresholds['severe']):
                errors.append("Severity thresholds must be in ascending order: mild < moderate < severe")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_privacy_config(config: PrivacyConfig) -> Tuple[bool, List[str]]:
        """Validate privacy configuration parameters"""
        errors = []
        
        # Validate data retention
        if config.data_retention_days <= 0:
            errors.append("data_retention_days must be positive")
        
        # HIPAA compliance requires certain settings
        if config.hipaa_compliance:
            if not config.enable_encryption:
                errors.append("HIPAA compliance requires encryption to be enabled")
            if not config.audit_logging:
                errors.append("HIPAA compliance requires audit logging to be enabled")
            if not config.anonymization:
                errors.append("HIPAA compliance requires anonymization to be enabled")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_performance_config(config: PerformanceConfig) -> Tuple[bool, List[str]]:
        """Validate performance configuration parameters"""
        errors = []
        
        # Validate batch size
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        # Validate sequence length
        if config.max_sequence_length <= 0:
            errors.append("max_sequence_length must be positive")
        
        # Validate cache size
        if config.cache_size <= 0:
            errors.append("cache_size must be positive")
        
        # Validate update frequency
        if config.model_update_frequency <= 0:
            errors.append("model_update_frequency must be positive")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_config(cls, config: EnterpriseConfig) -> Tuple[bool, List[str]]:
        """Validate complete enterprise configuration"""
        all_errors = []
        
        # Validate each config section
        model_valid, model_errors = cls.validate_model_config(config.model)
        clinical_valid, clinical_errors = cls.validate_clinical_config(config.clinical)
        privacy_valid, privacy_errors = cls.validate_privacy_config(config.privacy)
        performance_valid, performance_errors = cls.validate_performance_config(config.performance)
        
        # Collect all errors
        all_errors.extend([f"Model config: {err}" for err in model_errors])
        all_errors.extend([f"Clinical config: {err}" for err in clinical_errors])
        all_errors.extend([f"Privacy config: {err}" for err in privacy_errors])
        all_errors.extend([f"Performance config: {err}" for err in performance_errors])
        
        # Cross-validation checks
        cross_validation_errors = cls._cross_validate_configs(config)
        all_errors.extend(cross_validation_errors)
        
        is_valid = len(all_errors) == 0
        
        if not is_valid:
            logger.error(f"Configuration validation failed with {len(all_errors)} errors:")
            for error in all_errors:
                logger.error(f"  {error}")
        
        return is_valid, all_errors
    
    @staticmethod
    def _cross_validate_configs(config: EnterpriseConfig) -> List[str]:
        """Cross-validate settings across different config sections"""
        errors = []
        
        # Check if monitoring is enabled but A/B testing requires it
        if config.performance.enable_ab_testing and not config.performance.enable_monitoring:
            errors.append("A/B testing requires monitoring to be enabled")
        
        # Check if clinical features are compatible with privacy settings
        if config.clinical.enable_suicide_risk_assessment and not config.privacy.audit_logging:
            errors.append("Suicide risk assessment requires audit logging for safety compliance")
        
        # Check if uncertainty sampling is reasonable for batch size
        if config.model.uncertainty_samples > config.performance.batch_size * 10:
            errors.append("Uncertainty samples should not exceed 10x batch size for performance")
        
        return errors
    
    @staticmethod
    def get_validation_report(config: EnterpriseConfig) -> Dict[str, Any]:
        """Get detailed validation report"""
        is_valid, errors = ConfigValidator.validate_config(config)
        
        # Validate individual sections
        model_valid, model_errors = ConfigValidator.validate_model_config(config.model)
        clinical_valid, clinical_errors = ConfigValidator.validate_clinical_config(config.clinical)
        privacy_valid, privacy_errors = ConfigValidator.validate_privacy_config(config.privacy)
        performance_valid, performance_errors = ConfigValidator.validate_performance_config(config.performance)
        
        return {
            'overall_valid': is_valid,
            'total_errors': len(errors),
            'sections': {
                'model': {'valid': model_valid, 'errors': model_errors},
                'clinical': {'valid': clinical_valid, 'errors': clinical_errors},
                'privacy': {'valid': privacy_valid, 'errors': privacy_errors},
                'performance': {'valid': performance_valid, 'errors': performance_errors}
            },
            'all_errors': errors,
            'recommendations': ConfigValidator._generate_recommendations(config, errors)
        }
    
    @staticmethod
    def _generate_recommendations(config: EnterpriseConfig, errors: List[str]) -> List[str]:
        """Generate recommendations based on configuration and errors"""
        recommendations = []
        
        # Performance recommendations
        if config.model.uncertainty_samples > 50:
            recommendations.append("Consider reducing uncertainty_samples to 50 for faster inference")
        
        if config.performance.batch_size > 64:
            recommendations.append("Large batch sizes may cause memory issues, consider reducing")
        
        # Security recommendations
        if not config.privacy.enable_encryption:
            recommendations.append("Enable encryption for production deployments")
        
        if config.privacy.data_retention_days > 365:
            recommendations.append("Consider shorter data retention for privacy compliance")
        
        # Clinical recommendations
        if config.clinical.confidence_threshold < 0.7:
            recommendations.append("Consider higher confidence threshold for clinical use")
        
        return recommendations