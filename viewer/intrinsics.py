"""Build a camera intrinsics matrix K from CLI-style parameters."""

from __future__ import annotations

import numpy as np

DEFAULT_HFOV_DEG = 60.0


def fov_to_K(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """K from horizontal FOV; principal point at image center, fx=fy, no skew."""
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def build_K(
    width: int,
    height: int,
    *,
    hfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Build K. Explicit fx & fy win; otherwise derive from hfov (default 60deg).

    When fx/fy are given, cx/cy default to the image center.
    """
    if fx is not None and fy is not None:
        cx = width / 2.0 if cx is None else cx
        cy = height / 2.0 if cy is None else cy
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=float)
    return fov_to_K(DEFAULT_HFOV_DEG if hfov_deg is None else hfov_deg, width, height)
