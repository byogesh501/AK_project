"""
Shared test fixtures and configuration for pytest.
"""

import sys
from pathlib import Path

# We do NOT inject the root directly at index 0 because our `platform`
# folder shadows the Python standard library `platform` module, which
# breaks third-party tools like pytest and httpx during import.
#
# Tests will run via `python -m pytest tests/` which safely adds the cwd
# without strictly overriding standard library imports too early, OR
# developers should install via `pip install -e .`

# TODO: Add shared fixtures as the project grows.
