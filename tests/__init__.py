"""
Comprehensive Testing Suite for Solace-AI

This module provides a complete testing framework including:
- Unit tests for individual components
- Integration tests for system interactions
- Performance tests for scalability
- Clinical validation tests
- Security and compliance tests
"""

import pytest
import asyncio
import logging
from typing import Dict, Any

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "database": {
        "use_test_db": True,
        "test_db_url": "sqlite:///test_solace.db"
    },
    "models": {
        "use_mock_models": True,
        "model_cache_size": 10
    },
    "timeout": {
        "unit_test": 30,
        "integration_test": 120,
        "performance_test": 300
    }
}

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "clinical: mark test as a clinical validation test"    
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )