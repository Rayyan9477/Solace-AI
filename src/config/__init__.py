"""
Configuration module for the Mental Health Support Bot.

This module provides centralized configuration management including
application settings, model configurations, and optimization settings.
"""

from .settings import AppConfig

# Import optimization config if available
try:
    from .optimization_config import OptimizationConfig
    OPTIMIZATION_CONFIG_AVAILABLE = True
except ImportError:
    OPTIMIZATION_CONFIG_AVAILABLE = False
    OptimizationConfig = None

__all__ = ['AppConfig', 'OptimizationConfig', 'OPTIMIZATION_CONFIG_AVAILABLE']