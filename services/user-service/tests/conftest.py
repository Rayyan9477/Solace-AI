"""Pytest configuration for user-service tests."""
import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet

# Add user-service root to path so 'from src.main import ...' works
service_root = Path(__file__).parent.parent
if str(service_root) not in sys.path:
    sys.path.insert(0, str(service_root))

# User-service specific env vars required by UserServiceSettings (env_prefix="USER_SERVICE_")
_test_secret = "test_secret_key_for_pytest_only_not_for_production_use_minimum_32_chars"
os.environ.setdefault("USER_SERVICE_JWT_SECRET_KEY", _test_secret)
os.environ.setdefault("USER_SERVICE_FIELD_ENCRYPTION_KEY", _test_secret)

# SecurityConfig (env_prefix="SECURITY_")
os.environ.setdefault("SECURITY_JWT_SECRET", _test_secret)

# DatabaseConfig requires USER_DB_PASSWORD; override root conftest value since
# _is_unsafe_secret rejects strings containing "password"
os.environ["USER_DB_PASSWORD"] = "test_db_credential_for_pytest_only"

# Fernet keys required by lifespan() in main.py
os.environ.setdefault("FERNET_TOKEN_KEY", Fernet.generate_key().decode())
os.environ.setdefault("FERNET_FIELD_KEY", Fernet.generate_key().decode())
