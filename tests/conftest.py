"""
Pytest configuration for Solace-AI tests.

Adds the src directory to the Python path for proper imports.
"""

import sys
from pathlib import Path

# Project root (where tests/ and src/ directories are)
project_root = Path(__file__).parent.parent

# Add src directory to path for imports
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
