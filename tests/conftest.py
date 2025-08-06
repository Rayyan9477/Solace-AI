"""
Pytest configuration and fixtures for Solace-AI testing
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch
import logging

# Set up test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config():
    """Mock configuration for tests"""
    return {
        "app": {
            "debug": True,
            "environment": "test"
        },
        "database": {
            "url": "sqlite:///:memory:",
            "echo": False
        },
        "models": {
            "use_mock": True,
            "cache_size": 10
        },
        "logging": {
            "level": "DEBUG"
        }
    }

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock.generate.return_value = "Mock response"
    mock.generate_async.return_value = asyncio.coroutine(lambda: "Mock async response")()
    return mock

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock = Mock()
    mock.add_documents.return_value = True
    mock.search.return_value = [{"content": "test", "score": 0.9}]
    return mock

@pytest.fixture
def sample_patient_data():
    """Sample patient data for testing"""
    return {
        "patient_id": "test_patient_001",
        "age": 25,
        "gender": "female",
        "symptoms": ["anxiety", "sleep problems", "concentration issues"],
        "duration": "3 months",
        "severity": "moderate",
        "previous_conditions": [],
        "medications": [],
        "session_data": {
            "messages": [
                {"role": "user", "content": "I've been feeling anxious lately"},
                {"role": "assistant", "content": "Can you tell me more about your anxiety?"}
            ]
        }
    }

@pytest.fixture
def sample_clinical_data():
    """Sample clinical assessment data"""
    return {
        "phq9_score": 12,
        "gad7_score": 10,
        "assessment_date": "2024-01-15",
        "clinician_notes": "Patient reports moderate anxiety and depression symptoms",
        "risk_factors": ["work stress", "relationship issues"],
        "protective_factors": ["good social support", "regular exercise"]
    }

@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor"""
    mock = Mock()
    mock.extract.return_value = {
        "features": [0.1, 0.2, 0.3, 0.4, 0.5],
        "confidence": 0.85,
        "metadata": {"extraction_method": "mock"}
    }
    return mock

@pytest.fixture
def mock_diagnostic_pipeline():
    """Mock diagnostic pipeline"""
    mock = Mock()
    mock.process.return_value = {
        "conditions": [
            {"name": "Generalized Anxiety Disorder", "probability": 0.75, "confidence": "high"},
            {"name": "Major Depressive Disorder", "probability": 0.45, "confidence": "moderate"}
        ],
        "severity": "moderate",
        "recommendations": ["therapy", "lifestyle changes"]
    }
    return mock

@pytest.fixture
def clinical_test_cases():
    """Clinical test cases for validation"""
    return [
        {
            "case_id": "case_001",
            "description": "Young adult with GAD symptoms",
            "input": {
                "age": 22,
                "symptoms": ["excessive worry", "restlessness", "fatigue"],
                "duration": "6 months",
                "gad7_score": 15
            },
            "expected_output": {
                "primary_diagnosis": "Generalized Anxiety Disorder",
                "confidence": "high",
                "severity": "moderate"
            }
        },
        {
            "case_id": "case_002", 
            "description": "Adult with MDD symptoms",
            "input": {
                "age": 35,
                "symptoms": ["persistent sadness", "loss of interest", "sleep problems"],
                "duration": "8 weeks",
                "phq9_score": 18
            },
            "expected_output": {
                "primary_diagnosis": "Major Depressive Disorder",
                "confidence": "high",
                "severity": "moderate-severe"
            }
        }
    ]

@pytest.fixture
def performance_benchmarks():
    """Performance benchmarks for testing"""
    return {
        "response_time": {
            "diagnostic_analysis": 2.0,  # seconds
            "feature_extraction": 0.5,
            "model_inference": 1.0
        },
        "throughput": {
            "concurrent_users": 100,
            "requests_per_second": 50
        },
        "resource_usage": {
            "max_memory_mb": 1024,
            "max_cpu_percent": 80
        }
    }

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically clean up temporary files after each test"""
    temp_files = []
    yield temp_files
    
    # Clean up any temporary files created during tests
    for file_path in temp_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")

@pytest.fixture
def security_test_data():
    """Security test data for penetration testing"""
    return {
        "sql_injection_payloads": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users"
        ],
        "xss_payloads": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ],
        "sensitive_data_patterns": [
            r"\d{3}-\d{2}-\d{4}",  # SSN pattern
            r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",  # Credit card pattern
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # Email pattern
        ]
    }

# Pytest markers for different test categories
def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location"""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests in performance/ directory as performance tests
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark tests in clinical/ directory as clinical tests
        elif "clinical" in str(item.fspath):
            item.add_marker(pytest.mark.clinical)
        
        # Mark tests in security/ directory as security tests
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)

# Custom pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.clinical = pytest.mark.clinical
pytest.mark.security = pytest.mark.security
pytest.mark.slow = pytest.mark.slow