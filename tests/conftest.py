"""Pytest fixtures + sys.path setup for portable imports.

Adds the package parent to sys.path so tests can use `from seg_pose.X import Y`
regardless of where the package directory is located. This works both inside
the project (parents[2] = tools/m2/) and after a directory copy (parents[2] =
wherever the user pasted it).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests that load real SAM 3.1 model (deselect with -m 'not slow')"
    )
