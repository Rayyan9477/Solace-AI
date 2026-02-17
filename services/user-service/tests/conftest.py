"""Pytest configuration for user-service tests."""
import sys
from pathlib import Path

# Add user-service root to path so 'from src.main import ...' works
service_root = Path(__file__).parent.parent
if str(service_root) not in sys.path:
    sys.path.insert(0, str(service_root))
