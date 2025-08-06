"""
Base configuration classes for Enterprise Diagnostic Pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from .constants import (
    DEFAULT_MODEL_CONFIG, 
    DEFAULT_CLINICAL_CONFIG, 
    DEFAULT_PRIVACY_CONFIG,
    DEFAULT_PERFORMANCE_CONFIG
)

@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    fusion_dim: int = 512
    attention_heads: int = 8
    dropout: float = 0.1
    temporal_hidden_dim: int = 256
    temporal_layers: int = 2
    uncertainty_samples: int = 100
    device: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

@dataclass
class ClinicalConfig:
    """Configuration for clinical decision support"""
    confidence_threshold: float = 0.6
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "mild": 0.3,
        "moderate": 0.6,
        "severe": 0.8
    })
    dsm5_compliance: bool = True
    icd11_compliance: bool = True
    enable_suicide_risk_assessment: bool = True
    enable_treatment_recommendations: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClinicalConfig':
        """Create ClinicalConfig from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

@dataclass
class PrivacyConfig:
    """Configuration for privacy and security"""
    enable_encryption: bool = True
    audit_logging: bool = True
    data_retention_days: int = 90
    anonymization: bool = True
    hipaa_compliance: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PrivacyConfig':
        """Create PrivacyConfig from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

@dataclass
class PerformanceConfig:
    """Configuration for performance and monitoring"""
    batch_size: int = 32
    max_sequence_length: int = 512
    cache_size: int = 1000
    model_update_frequency: int = 24  # hours
    enable_monitoring: bool = True
    enable_ab_testing: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PerformanceConfig':
        """Create PerformanceConfig from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

@dataclass
class EnterpriseConfig:
    """Main configuration class for Enterprise Diagnostic Pipeline"""
    model: ModelConfig = field(default_factory=ModelConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnterpriseConfig':
        """Create EnterpriseConfig from dictionary"""
        return cls(
            model=ModelConfig.from_dict(config_dict.get('model', {})),
            clinical=ClinicalConfig.from_dict(config_dict.get('clinical', {})),
            privacy=PrivacyConfig.from_dict(config_dict.get('privacy', {})),
            performance=PerformanceConfig.from_dict(config_dict.get('performance', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'clinical': self.clinical.__dict__,
            'privacy': self.privacy.__dict__,
            'performance': self.performance.__dict__
        }
    
    @classmethod
    def get_default(cls) -> 'EnterpriseConfig':
        """Get default configuration"""
        return cls(
            model=ModelConfig(**DEFAULT_MODEL_CONFIG),
            clinical=ClinicalConfig(**DEFAULT_CLINICAL_CONFIG),
            privacy=PrivacyConfig(**DEFAULT_PRIVACY_CONFIG),
            performance=PerformanceConfig(**DEFAULT_PERFORMANCE_CONFIG)
        )