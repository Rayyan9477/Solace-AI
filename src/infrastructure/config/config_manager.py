"""
Configuration Manager implementation.

This module provides a centralized configuration management system
that supports multiple configuration sources and validation.
"""

from typing import Dict, Any, List, Optional, Type, Union
import asyncio
from pathlib import Path

from ...core.interfaces.config_interface import (
    ConfigInterface,
    ConfigSource,
    ConfigValidationResult
)
from ...core.exceptions.base_exceptions import ConfigurationError
from .schema import ConfigSchema


class ConfigManager:
    """
    Centralized configuration manager.
    
    Manages multiple configuration providers and provides a unified
    interface for accessing configuration values with validation.
    """
    
    def __init__(self):
        """Initialize the configuration manager."""
        self._providers: List[ConfigInterface] = []
        self._schemas: Dict[str, ConfigSchema] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = True
    
    def add_provider(self, provider: ConfigInterface, priority: int = 0) -> None:
        """
        Add a configuration provider.
        
        Args:
            provider: Configuration provider to add
            priority: Priority of the provider (higher = more priority)
        """
        self._providers.append((provider, priority))
        # Sort by priority (highest first)
        self._providers.sort(key=lambda x: x[1], reverse=True)
    
    def register_schema(self, schema_name: str, schema: ConfigSchema) -> None:
        """
        Register a configuration schema for validation.
        
        Args:
            schema_name: Name of the schema
            schema: Configuration schema
        """
        self._schemas[schema_name] = schema
    
    async def load_all(self) -> bool:
        """
        Load configuration from all providers.
        
        Returns:
            bool: True if all providers loaded successfully
        """
        success = True
        
        for provider, _ in self._providers:
            try:
                if not await provider.load_config():
                    success = False
            except Exception as e:
                print(f"Error loading config from {provider.__class__.__name__}: {e}")
                success = False
        
        # Clear cache after loading
        self._cache.clear()
        
        return success
    
    def get_value(
        self,
        key: str,
        default: Any = None,
        value_type: Optional[Type] = None
    ) -> Any:
        """
        Get a configuration value from providers.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            value_type: Expected type for validation
            
        Returns:
            Configuration value or default
        """
        # Check cache first
        if self._cache_enabled and key in self._cache:
            return self._cache[key]
        
        # Try each provider in priority order
        for provider, _ in self._providers:
            if provider.has_key(key):
                value = provider.get_value(key, default, value_type)
                
                # Cache the value
                if self._cache_enabled:
                    self._cache[key] = value
                
                return value
        
        # No provider has the key, return default
        return default
    
    def set_value(self, key: str, value: Any, provider_index: int = 0) -> None:
        """
        Set a configuration value in a specific provider.
        
        Args:
            key: Configuration key
            value: Value to set
            provider_index: Index of provider to use (default: highest priority)
        """
        if provider_index >= len(self._providers):
            raise ConfigurationError(
                f"Provider index {provider_index} out of range",
                config_key=key
            )
        
        provider, _ = self._providers[provider_index]
        provider.set_value(key, value)
        
        # Update cache
        if self._cache_enabled:
            self._cache[key] = value
    
    def has_key(self, key: str) -> bool:
        """
        Check if any provider has the specified key.
        
        Args:
            key: Configuration key to check
            
        Returns:
            bool: True if key exists in any provider
        """
        for provider, _ in self._providers:
            if provider.has_key(key):
                return True
        return False
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: Section name
            
        Returns:
            Dict containing merged section from all providers
        """
        merged_section = {}
        
        # Merge sections from all providers (reverse priority order for overrides)
        for provider, _ in reversed(self._providers):
            try:
                provider_section = provider.get_section(section)
                if provider_section:
                    merged_section.update(provider_section)
            except Exception:
                continue
        
        return merged_section
    
    def validate_schema(self, schema_name: str) -> ConfigValidationResult:
        """
        Validate configuration against a registered schema.
        
        Args:
            schema_name: Name of the schema to validate against
            
        Returns:
            ConfigValidationResult: Validation results
        """
        if schema_name not in self._schemas:
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Schema '{schema_name}' not registered"],
                warnings=[]
            )
        
        schema = self._schemas[schema_name]
        return schema.validate(self)
    
    def validate_all_schemas(self) -> Dict[str, ConfigValidationResult]:
        """
        Validate configuration against all registered schemas.
        
        Returns:
            Dict mapping schema names to validation results
        """
        results = {}
        
        for schema_name, schema in self._schemas.items():
            results[schema_name] = schema.validate(self)
        
        return results
    
    async def save_all(self) -> bool:
        """
        Save configuration to all providers that support saving.
        
        Returns:
            bool: True if all saves were successful
        """
        success = True
        
        for provider, _ in self._providers:
            try:
                if not await provider.save_config():
                    success = False
            except Exception as e:
                print(f"Error saving config to {provider.__class__.__name__}: {e}")
                success = False
        
        return success
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration from all providers.
        
        Returns:
            Dict containing merged configuration from all providers
        """
        merged_config = {}
        
        # Merge config from all providers (reverse priority for overrides)
        for provider, _ in reversed(self._providers):
            provider_config = provider.get_all_config()
            if provider_config:
                self._deep_merge(merged_config, provider_config)
        
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def enable_cache(self) -> None:
        """Enable configuration value caching."""
        self._cache_enabled = True
    
    def disable_cache(self) -> None:
        """Disable configuration value caching."""
        self._cache_enabled = False
        self._cache.clear()
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
    
    def get_provider_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered providers.
        
        Returns:
            List of provider information
        """
        provider_info = []
        
        for i, (provider, priority) in enumerate(self._providers):
            provider_info.append({
                'index': i,
                'class_name': provider.__class__.__name__,
                'source': provider.config_source.value,
                'priority': priority,
                'initialized': provider.is_initialized
            })
        
        return provider_info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all providers.
        
        Returns:
            Dict containing health status
        """
        provider_health = []
        overall_healthy = True
        
        for provider, priority in self._providers:
            try:
                health_info = await provider.health_check()
                provider_health.append({
                    'provider': provider.__class__.__name__,
                    'priority': priority,
                    'health': health_info
                })
                
                if health_info.get('status') != 'healthy':
                    overall_healthy = False
                    
            except Exception as e:
                provider_health.append({
                    'provider': provider.__class__.__name__,
                    'priority': priority,
                    'health': {
                        'status': 'error',
                        'error': str(e)
                    }
                })
                overall_healthy = False
        
        # Validate all schemas
        schema_validations = self.validate_all_schemas()
        schema_valid = all(result.is_valid for result in schema_validations.values())
        
        return {
            'status': 'healthy' if overall_healthy and schema_valid else 'degraded',
            'providers': provider_health,
            'schemas': {
                name: {
                    'valid': result.is_valid,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                for name, result in schema_validations.items()
            },
            'cache_enabled': self._cache_enabled,
            'cached_values': len(self._cache)
        }


# Global configuration manager instance
_global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return _global_config_manager


def reset_config_manager() -> None:
    """Reset the global configuration manager (useful for testing)."""
    global _global_config_manager
    _global_config_manager = ConfigManager()