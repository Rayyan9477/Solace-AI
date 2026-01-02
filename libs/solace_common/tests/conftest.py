"""
Pytest configuration for solace-common tests.

Adds the libs directory to the Python path for proper imports.
"""

import sys
from pathlib import Path

# Add libs directory to path for imports
libs_path = Path(__file__).parent.parent.parent
if str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

# Add project root to path
project_root = libs_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
