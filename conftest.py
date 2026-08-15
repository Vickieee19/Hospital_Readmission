"""
conftest.py
-----------
Pytest configuration for the Hospital Readmission project.

Ensures the project root is on sys.path so that `src.*`
imports resolve correctly regardless of which directory pytest
is invoked from.
"""

import sys
from pathlib import Path

# Always add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
