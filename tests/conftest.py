"""
Shared test fixtures and configuration for pytest.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so that `core`, `domains`,
# `api`, `dashboard`, etc. are importable during test runs.
# We append to the *end* of sys.path so that the Python standard
# library is preferred in case of any name collision.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)

# TODO: Add shared fixtures as the project grows.
