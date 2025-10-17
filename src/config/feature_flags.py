"""
Feature Flag System for MVP vs Enterprise Modes
Manages feature availability and configuration based on deployment mode
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class DeploymentMode(Enum):
    """Deployment modes"""
    MVP = "mvp"
    ENTERPRISE = "enterprise"
    DEVELOPMENT = "development"
    TESTING = "testing"

class FeatureCategory(Enum):
    """Feature categories for organization"""
    CORE = "core"
    AI_MODELS = "ai_models"
    DIAGNOSTICS = "diagnostics"
    VOICE = "voice"
    ANALYTICS = "analytics"
    INTEGRATIONS = "integrations"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    UI = "ui"

class FeatureFlagManager:
    """Manages feature flags for different deployment modes"""
    
    def __init__(self, mode: Optional[str] = None, config_file: Optional[str] = None):
        self.mode = self._determine_mode(mode)
        self.config_file = Path(config_file or "feature_config.json")
        self._flags = self._load_feature_configuration()
        
        logger.info(f"Feature flag manager initialized in {self.mode.value} mode")
    
    def _determine_mode(self, mode: Optional[str] = None) -> DeploymentMode:
        """Determine deployment mode from environment or parameter"""
        if mode:
            return DeploymentMode(mode.lower())
        
        env_mode = os.getenv("DEPLOYMENT_MODE", os.getenv("ENVIRONMENT", "mvp")).lower()
        
        # Map environment values to deployment modes
        mode_mapping = {
            "mvp": DeploymentMode.MVP,
            "production": DeploymentMode.ENTERPRISE,
            "enterprise": DeploymentMode.ENTERPRISE,
            "development": DeploymentMode.DEVELOPMENT,
            "dev": DeploymentMode.DEVELOPMENT,
            "testing": DeploymentMode.TESTING,
            "test": DeploymentMode.TESTING
        }
        
        return mode_mapping.get(env_mode, DeploymentMode.MVP)
    
    def _load_feature_configuration(self) -> Dict[str, Any]:
        """Load feature configuration from file or use defaults"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded feature configuration from {self.config_file}")
                return config
            except Exception as e:
                logger.error(f"Failed to load feature config: {e}")
        
        # Use default configuration
        return self._get_default_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default feature configuration for all modes"""
        return {
            "mvp": {
                "core": {
                    "chat_interface": True,
                    "user_authentication": True,
                    "basic_logging": True,
                    "health_check": True,
                    "file_upload": False,
                    "real_time_streaming": False
                },
                "ai_models": {
                    "basic_chat": True,
                    "advanced_diagnosis": False,
                    "multi_model_ensemble": False,
                    "custom_model_training": False,
                    "model_versioning": False
                },
                "diagnostics": {
                    "simple_assessment": True,
                    "comprehensive_diagnosis": False,
                    "differential_diagnosis": False,
                    "temporal_analysis": False,
                    "risk_stratification": False
                },
                "voice": {
                    "basic_tts": True,
                    "basic_stt": True,
                    "voice_cloning": False,
                    "emotion_analysis": False,
                    "multilingual_support": False
                },
                "analytics": {
                    "basic_metrics": True,
                    "user_analytics": False,
                    "advanced_reporting": False,
                    "predictive_analytics": False,
                    "real_time_dashboard": False
                },
                "integrations": {
                    "openai_integration": True,
                    "anthropic_integration": True,
                    "google_integration": False,
                    "external_apis": False,
                    "webhook_support": False
                },
                "security": {
                    "basic_auth": True,
                    "rate_limiting": True,
                    "input_sanitization": True,
                    "advanced_encryption": False,
                    "audit_logging": False
                },
                "compliance": {
                    "basic_privacy": True,
                    "hipaa_compliance": False,
                    "gdpr_compliance": False,
                    "sox_compliance": False,
                    "custom_compliance": False
                },
                "ui": {
                    "basic_interface": True,
                    "admin_panel": False,
                    "analytics_dashboard": False,
                    "custom_themes": False,
                    "mobile_app": False
                }
            },
            "enterprise": {
                "core": {
                    "chat_interface": True,
                    "user_authentication": True,
                    "basic_logging": True,
                    "health_check": True,
                    "file_upload": True,
                    "real_time_streaming": True
                },
                "ai_models": {
                    "basic_chat": True,
                    "advanced_diagnosis": True,
                    "multi_model_ensemble": True,
                    "custom_model_training": True,
                    "model_versioning": True
                },
                "diagnostics": {
                    "simple_assessment": True,
                    "comprehensive_diagnosis": True,
                    "differential_diagnosis": True,
                    "temporal_analysis": True,
                    "risk_stratification": True
                },
                "voice": {
                    "basic_tts": True,
                    "basic_stt": True,
                    "voice_cloning": True,
                    "emotion_analysis": True,
                    "multilingual_support": True
                },
                "analytics": {
                    "basic_metrics": True,
                    "user_analytics": True,
                    "advanced_reporting": True,
                    "predictive_analytics": True,
                    "real_time_dashboard": True
                },
                "integrations": {
                    "openai_integration": True,
                    "anthropic_integration": True,
                    "google_integration": True,
                    "external_apis": True,
                    "webhook_support": True
                },
                "security": {
                    "basic_auth": True,
                    "rate_limiting": True,
                    "input_sanitization": True,
                    "advanced_encryption": True,
                    "audit_logging": True
                },
                "compliance": {
                    "basic_privacy": True,
                    "hipaa_compliance": True,
                    "gdpr_compliance": True,
                    "sox_compliance": True,
                    "custom_compliance": True
                },
                "ui": {
                    "basic_interface": True,
                    "admin_panel": True,
                    "analytics_dashboard": True,
                    "custom_themes": True,
                    "mobile_app": True
                }
            },
            "development": {
                # Development mode inherits enterprise features + debug features
                **self._get_enterprise_config(),
                "debug": {
                    "debug_endpoints": True,
                    "performance_profiling": True,
                    "mock_services": True,
                    "test_data_generation": True,
                    "feature_toggle_ui": True
                }
            },
            "testing": {
                # Testing mode with minimal features for automated tests
                **self._get_mvp_config(),
                "testing": {
                    "mock_external_apis": True,
                    "test_fixtures": True,
                    "performance_testing": True,
                    "security_testing": True
                }
            }
        }
    
    def _get_mvp_config(self) -> Dict[str, Any]:
        """Get MVP configuration"""
        config = self._get_default_configuration()
        return config["mvp"]
    
    def _get_enterprise_config(self) -> Dict[str, Any]:
        """Get enterprise configuration"""
        config = self._get_default_configuration()
        return config["enterprise"]
    
    def is_enabled(self, feature: str, category: str = "core") -> bool:
        """Check if a feature is enabled in current mode"""
        try:
            mode_config = self._flags.get(self.mode.value, {})
            category_config = mode_config.get(category, {})
            return category_config.get(feature, False)
        except Exception as e:
            logger.error(f"Error checking feature flag {category}.{feature}: {e}")
            return False
    
    def get_feature_config(self, feature: str, category: str = "core") -> Any:
        """Get feature configuration value"""
        try:
            mode_config = self._flags.get(self.mode.value, {})
            category_config = mode_config.get(category, {})
            return category_config.get(feature)
        except Exception as e:
            logger.error(f"Error getting feature config {category}.{feature}: {e}")
            return None
    
    def get_enabled_features(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get all enabled features, optionally filtered by category"""
        try:
            mode_config = self._flags.get(self.mode.value, {})
            
            if category:
                category_config = mode_config.get(category, {})
                return {k: v for k, v in category_config.items() if v is True}
            
            enabled_features = {}
            for cat, features in mode_config.items():
                if isinstance(features, dict):
                    enabled_features[cat] = {k: v for k, v in features.items() if v is True}
            
            return enabled_features
        except Exception as e:
            logger.error(f"Error getting enabled features: {e}")
            return {}
    
    def set_feature(self, feature: str, enabled: bool, category: str = "core") -> bool:
        """Dynamically set feature flag (runtime only, not persistent)"""
        try:
            if self.mode.value not in self._flags:
                self._flags[self.mode.value] = {}
            
            if category not in self._flags[self.mode.value]:
                self._flags[self.mode.value][category] = {}
            
            self._flags[self.mode.value][category][feature] = enabled
            logger.info(f"Feature {category}.{feature} set to {enabled} in {self.mode.value} mode")
            return True
        except Exception as e:
            logger.error(f"Error setting feature flag {category}.{feature}: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._flags, f, indent=2)
            logger.info(f"Feature configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving feature configuration: {e}")
            return False
    
    def get_mode_summary(self) -> Dict[str, Any]:
        """Get summary of current mode and enabled features"""
        return {
            "deployment_mode": self.mode.value,
            "total_features": sum(
                len(features) for features in self._flags.get(self.mode.value, {}).values()
                if isinstance(features, dict)
            ),
            "enabled_features": len([
                feature for category in self._flags.get(self.mode.value, {}).values()
                if isinstance(category, dict)
                for feature, enabled in category.items()
                if enabled is True
            ]),
            "categories": list(self._flags.get(self.mode.value, {}).keys())
        }
    
    def require_feature(self, feature: str, category: str = "core"):
        """Decorator to require a feature to be enabled"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.is_enabled(feature, category):
                    raise FeatureDisabledError(
                        f"Feature {category}.{feature} is not enabled in {self.mode.value} mode"
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator

class FeatureDisabledError(Exception):
    """Raised when attempting to use a disabled feature"""
    pass

# Feature-specific helper classes
class AIModelFeatures:
    """Helper for AI model feature checks"""
    
    def __init__(self, flag_manager: FeatureFlagManager):
        self.flags = flag_manager
    
    def can_use_advanced_diagnosis(self) -> bool:
        return self.flags.is_enabled("advanced_diagnosis", "ai_models")
    
    def can_use_multi_model(self) -> bool:
        return self.flags.is_enabled("multi_model_ensemble", "ai_models")
    
    def can_train_custom_models(self) -> bool:
        return self.flags.is_enabled("custom_model_training", "ai_models")

class ComplianceFeatures:
    """Helper for compliance feature checks"""
    
    def __init__(self, flag_manager: FeatureFlagManager):
        self.flags = flag_manager
    
    def is_hipaa_required(self) -> bool:
        return self.flags.is_enabled("hipaa_compliance", "compliance")
    
    def is_gdpr_required(self) -> bool:
        return self.flags.is_enabled("gdpr_compliance", "compliance")
    
    def needs_audit_logging(self) -> bool:
        return self.flags.is_enabled("audit_logging", "security")

# Global feature flag manager instance
try:
    feature_flags = FeatureFlagManager()
    ai_features = AIModelFeatures(feature_flags)
    compliance_features = ComplianceFeatures(feature_flags)
    
    logger.info(f"Feature flags initialized: {feature_flags.get_mode_summary()}")
except Exception as e:
    logger.error(f"Failed to initialize feature flags: {e}")
    # Fallback to MVP mode
    feature_flags = FeatureFlagManager(mode="mvp")
    ai_features = AIModelFeatures(feature_flags)
    compliance_features = ComplianceFeatures(feature_flags)