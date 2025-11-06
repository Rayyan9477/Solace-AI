"""
Optimization Configuration for Multi-Agent System

This module provides configuration settings for performance optimization,
caching, parallel execution, and resource management.
"""

import os
from typing import Dict, Any, List
from pathlib import Path

class OptimizationConfig:
    """
    Configuration settings for system optimization.

    These settings control caching, parallel execution, context management,
    and performance profiling for the multi-agent system.
    """

    # Enable/Disable optimization features
    OPTIMIZATION_ENABLED = os.getenv("OPTIMIZATION_ENABLED", "True").lower() == "true"
    PARALLEL_EXECUTION_ENABLED = os.getenv("PARALLEL_EXECUTION_ENABLED", "True").lower() == "true"
    CACHING_ENABLED = os.getenv("CACHING_ENABLED", "True").lower() == "true"
    CONTEXT_COMPRESSION_ENABLED = os.getenv("CONTEXT_COMPRESSION_ENABLED", "True").lower() == "true"
    PERFORMANCE_MONITORING_ENABLED = os.getenv("PERFORMANCE_MONITORING_ENABLED", "True").lower() == "true"

    # Parallel Execution Configuration
    PARALLEL_GROUPS = {
        "initial_assessment": ["emotion_agent", "personality_agent", "safety_agent"],
        "information_gathering": ["search_agent", "crawler_agent"],
        "clinical_assessment": ["diagnosis_agent", "therapy_agent"]
    }

    # Maximum workers for parallel execution
    MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "5"))

    # Caching Configuration
    CACHE_CONFIG = {
        "ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "300")),  # 5 minutes default
        "max_cache_size_mb": int(os.getenv("MAX_CACHE_SIZE_MB", "100")),
        "cache_dir": os.getenv("CACHE_DIR", str(Path(__file__).parent.parent / "cache")),

        # Agent-specific cache TTLs (in seconds)
        "agent_ttls": {
            "emotion_agent": 180,      # 3 minutes - emotions change quickly
            "personality_agent": 3600,  # 1 hour - personality is stable
            "safety_agent": 60,         # 1 minute - safety critical
            "diagnosis_agent": 600,     # 10 minutes - diagnosis is stable
            "therapy_agent": 300,       # 5 minutes - therapy recommendations
            "search_agent": 1800,       # 30 minutes - search results cache
            "crawler_agent": 1800,      # 30 minutes - crawled content cache
            "chat_agent": 0,            # No caching for chat responses
        },

        # Bypass cache for certain conditions
        "bypass_conditions": [
            "crisis",
            "emergency",
            "suicidal",
            "harm",
            "severe"
        ]
    }

    # Context Window Management
    CONTEXT_CONFIG = {
        "default_window_size": int(os.getenv("DEFAULT_CONTEXT_WINDOW", "4000")),

        # Agent-specific context windows (in tokens)
        "agent_windows": {
            "chat_agent": 8000,
            "therapy_agent": 6000,
            "diagnosis_agent": 6000,
            "emotion_agent": 4000,
            "personality_agent": 4000,
            "safety_agent": 3000,
            "search_agent": 2000,
            "crawler_agent": 2000
        },

        # Context compression settings
        "compression": {
            "enabled": CONTEXT_COMPRESSION_ENABLED,
            "max_compression_ratio": 0.7,  # Compress to 70% of original
            "preserve_critical": True,      # Always preserve safety/emotion context
            "semantic_importance_threshold": 0.6
        }
    }

    # Performance Profiling Configuration
    PROFILING_CONFIG = {
        "enabled": PERFORMANCE_MONITORING_ENABLED,
        "sample_rate": float(os.getenv("PROFILING_SAMPLE_RATE", "0.1")),  # Sample 10% of requests
        "metrics_export_interval": int(os.getenv("METRICS_EXPORT_INTERVAL", "300")),  # 5 minutes
        "metrics_retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "30")),

        # Performance thresholds
        "thresholds": {
            "max_response_time": 5.0,       # 5 seconds max response time
            "max_token_usage": 10000,       # Max tokens per request
            "max_memory_usage_mb": 500,     # Max memory usage per agent
            "max_error_rate": 0.1            # 10% error rate threshold
        },

        # Metrics to track
        "tracked_metrics": [
            "response_time",
            "token_usage",
            "cache_hit_rate",
            "error_rate",
            "memory_usage",
            "parallel_speedup",
            "context_compression_ratio"
        ]
    }

    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_CONFIG = {
        "enabled": os.getenv("CIRCUIT_BREAKER_ENABLED", "True").lower() == "true",
        "failure_threshold": int(os.getenv("CB_FAILURE_THRESHOLD", "5")),
        "reset_timeout": int(os.getenv("CB_RESET_TIMEOUT", "60")),  # seconds
        "success_threshold": int(os.getenv("CB_SUCCESS_THRESHOLD", "2")),
        "timeout": int(os.getenv("CB_TIMEOUT", "30"))  # seconds
    }

    # Retry Configuration
    RETRY_CONFIG = {
        "enabled": os.getenv("RETRY_ENABLED", "True").lower() == "true",
        "max_retries": int(os.getenv("MAX_RETRIES", "3")),
        "base_delay": float(os.getenv("RETRY_BASE_DELAY", "1.0")),  # seconds
        "max_delay": float(os.getenv("RETRY_MAX_DELAY", "60.0")),   # seconds
        "exponential_backoff": True
    }

    # Cost Optimization Settings
    COST_OPTIMIZATION = {
        "enabled": os.getenv("COST_OPTIMIZATION_ENABLED", "True").lower() == "true",

        # Model selection strategy
        "model_strategy": os.getenv("MODEL_STRATEGY", "balanced"),  # "performance", "cost", "balanced"

        # Cost thresholds (in USD)
        "daily_budget": float(os.getenv("DAILY_BUDGET", "10.0")),
        "monthly_budget": float(os.getenv("MONTHLY_BUDGET", "280.0")),

        # Model cost per 1K tokens (example values, adjust based on actual costs)
        "model_costs": {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3-sonnet": 0.015,
            "claude-3-haiku": 0.001,
            "gemini-pro": 0.001,
            "gemini-flash": 0.0001
        }
    }

    @classmethod
    def get_cache_config(cls) -> Dict[str, Any]:
        """Get caching configuration"""
        return cls.CACHE_CONFIG

    @classmethod
    def get_context_config(cls) -> Dict[str, Any]:
        """Get context management configuration"""
        return cls.CONTEXT_CONFIG

    @classmethod
    def get_parallel_groups(cls) -> Dict[str, List[str]]:
        """Get parallel execution groups"""
        return cls.PARALLEL_GROUPS if cls.PARALLEL_EXECUTION_ENABLED else {}

    @classmethod
    def get_profiling_config(cls) -> Dict[str, Any]:
        """Get performance profiling configuration"""
        return cls.PROFILING_CONFIG

    @classmethod
    def should_bypass_cache(cls, context: Dict[str, Any]) -> bool:
        """
        Check if cache should be bypassed based on context.

        Args:
            context: Current execution context

        Returns:
            True if cache should be bypassed
        """
        if not cls.CACHING_ENABLED:
            return True

        # Check for bypass conditions in context
        for condition in cls.CACHE_CONFIG.get("bypass_conditions", []):
            # Check in emotion context
            if context.get("emotion", {}).get("primary_emotion") == condition:
                return True

            # Check in safety context
            if context.get("safety", {}).get("risk_level", "").lower() == condition:
                return True

            # Check in message content
            message = context.get("message", "").lower()
            if condition in message:
                return True

        return False

    @classmethod
    def get_agent_cache_ttl(cls, agent_name: str) -> int:
        """
        Get cache TTL for specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            TTL in seconds
        """
        return cls.CACHE_CONFIG.get("agent_ttls", {}).get(
            agent_name,
            cls.CACHE_CONFIG.get("ttl_seconds", 300)
        )

    @classmethod
    def get_agent_context_window(cls, agent_name: str) -> int:
        """
        Get context window size for specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Context window size in tokens
        """
        return cls.CONTEXT_CONFIG.get("agent_windows", {}).get(
            agent_name,
            cls.CONTEXT_CONFIG.get("default_window_size", 4000)
        )

    @classmethod
    def is_optimization_enabled(cls) -> bool:
        """Check if optimization is enabled"""
        return cls.OPTIMIZATION_ENABLED

    @classmethod
    def get_optimization_summary(cls) -> Dict[str, Any]:
        """Get summary of optimization settings"""
        return {
            "optimization_enabled": cls.OPTIMIZATION_ENABLED,
            "features": {
                "parallel_execution": cls.PARALLEL_EXECUTION_ENABLED,
                "caching": cls.CACHING_ENABLED,
                "context_compression": cls.CONTEXT_COMPRESSION_ENABLED,
                "performance_monitoring": cls.PERFORMANCE_MONITORING_ENABLED,
                "circuit_breaker": cls.CIRCUIT_BREAKER_CONFIG["enabled"],
                "retry": cls.RETRY_CONFIG["enabled"],
                "cost_optimization": cls.COST_OPTIMIZATION["enabled"]
            },
            "parallel_workers": cls.MAX_PARALLEL_WORKERS,
            "cache_ttl": cls.CACHE_CONFIG["ttl_seconds"],
            "model_strategy": cls.COST_OPTIMIZATION["model_strategy"],
            "daily_budget": cls.COST_OPTIMIZATION["daily_budget"]
        }