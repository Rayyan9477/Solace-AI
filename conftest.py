"""
Root pytest configuration for Solace-AI.

Sets up Python path and test environment variables for all test directories.
"""

import os
import sys
from pathlib import Path

# Set test environment variables before anything else imports settings
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "5")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "1")
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")

# Service database/redis passwords (required by pydantic-settings configs)
_TEST_PASSWORD = "test_password_for_pytest_only"
for prefix in [
    "PERSONALITY_DB_", "PERSONALITY_REDIS_",
    "USER_DB_", "USER_REDIS_",
    "NOTIFICATION_DB_", "NOTIFICATION_REDIS_",
    "ANALYTICS_DB_", "ANALYTICS_REDIS_",
    "SAFETY_DB_", "SAFETY_REDIS_",
    "DIAGNOSIS_DB_", "DIAGNOSIS_REDIS_",
    "THERAPY_DB_", "THERAPY_REDIS_",
    "MEMORY_DB_", "MEMORY_REDIS_",
    "ORCHESTRATOR_DB_", "ORCHESTRATOR_REDIS_",
]:
    os.environ.setdefault(f"{prefix}PASSWORD", _TEST_PASSWORD)

# User-service specific required fields
os.environ.setdefault("USER_FIELD_ENCRYPTION_KEY", "dGVzdF9lbmNyeXB0aW9uX2tleV8zMmJ5dGVz")
os.environ.setdefault("USER_JWT_SECRET_KEY", "test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars")

# Project root
project_root = Path(__file__).parent

# Add src directory to path for imports (solace_common, solace_security, etc.)
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
