"""Resolve the repo root and put it on sys.path WITHOUT shadowing the sam3 package.

The repo root holds a module `sam3.py` that shadows the installed `sam3` package
when the root is sys.path[0]. We therefore append the root to the END of sys.path
so site-packages (the real sam3 package) win, while `import seg_pose` still works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root(explicit: str | None = None) -> Path:
    """Return the TargetGeo repo root. Falls back to two parents up from this file."""
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def ensure_seg_pose_importable(root: Path) -> None:
    """Append `root` to sys.path so `import seg_pose` works (root appended, not prepended)."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
