"""
Pytest configuration for tests/ directory.

Path and env setup is handled by the root conftest.py.
"""

# Every test under tests/ honours the expected_failures.yaml of its directory (MVP-039).
from solace_testing.e2e_manifest import (  # noqa: F401
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_runtest_makereport,
)
