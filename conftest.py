"""Root pytest configuration.

Ensures that the project root directory is always on sys.path regardless of how
pytest is invoked (direct CLI `pytest`, `python -m pytest`, IDE test runner, CI).
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
