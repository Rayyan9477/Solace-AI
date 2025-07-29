"""
Abstract interface for Configuration management.

This interface provides a contract for configuration providers,
enabling flexible configuration sources and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum


class ConfigSource(Enum):
    """Enum for different configuration sources."""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigInterface(ABC):
    """
    Abstract base class for configuration providers.
    
    This interface ensures all configuration providers implement
    consistent methods for loading, validating, and managing configuration.
    """
    
    def __init__(self, source: ConfigSource, source_path: Optional[str] = None):
        """Initialize the configuration provider."""
        self.source = source
        self.source_path = source_path
        self._config_data = {}
        self._initialized = False
    
    @abstractmethod
    async def load_config(self) -> bool:
        """
        Load configuration from the source.
        
        Returns:
            bool: True if loading was successful
        """
        pass
    
    @abstractmethod
    async def save_config(self) -> bool:
        """
        Save current configuration to the source.
        
        Returns:
            bool: True if saving was successful
        """
        pass
    
    @abstractmethod
    def get_value(
        self,
        key: str,
        default: Any = None,
        value_type: Optional[type] = None
    ) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            value_type: Expected type for type checking
            
        Returns:
            Configuration value or default
        """
        pass
    
    @abstractmethod
    def set_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        pass
    
    @abstractmethod
    def has_key(self, key: str) -> bool:
        """
        Check if a configuration key exists.
        
        Args:
            key: Configuration key to check
            
        Returns:
            bool: True if key exists
        """
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dict containing section configuration
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> ConfigValidationResult:
        """
        Validate the current configuration.
        
        Returns:
            ConfigValidationResult: Validation results
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the configuration schema.
        
        Returns:
            Dict containing the configuration schema
        """
        pass
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration data."""
        return self._config_data.copy()
    
    def merge_config(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration into this one.
        
        Args:
            other_config: Configuration to merge
        """
        def deep_merge(base: Dict, overlay: Dict) -> Dict:
            """Recursively merge two dictionaries."""
            for key, value in overlay.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        deep_merge(self._config_data, other_config)
    
    def get_nested_value(self, key: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_nested_value(self, key: str, value: Any) -> None:
        """
        Set a nested configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'database.host')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config_data
        
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    @property
    def is_initialized(self) -> bool:
        """Check if configuration is initialized."""
        return self._initialized
    
    @property
    def config_source(self) -> ConfigSource:
        """Get the configuration source."""
        return self.source
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the configuration provider.
        
        Returns:
            Dict containing health status
        """
        try:
            validation_result = self.validate_config()
            
            return {
                "status": "healthy" if validation_result.is_valid else "degraded",
                "source": self.source.value,
                "initialized": self.is_initialized,
                "validation_errors": validation_result.errors,
                "validation_warnings": validation_result.warnings,
                "config_keys": len(self._config_data)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "source": self.source.value,
                "error": str(e),
                "initialized": self.is_initialized
            }