"""Pure drawing functions for the viewer. Each operates on a BGR ndarray.

Colors are BGR. No I/O, no global state — directly unit-testable.
"""

from __future__ import annotations

import cv2
import numpy as np

BBOX_COLOR = (0, 255, 0)
ELLIPSE_COLOR = (255, 0, 255)
MASK_COLOR = (0, 255, 200)
NORMAL_COLOR = (0, 0, 255)
NORMAL_REJECT_COLOR = (120, 120, 255)
HUD_BG = (0, 0, 0)
HUD_FG = (255, 255, 255)


def project_point(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    """Pinhole-project a camera-frame point. Returns (u, v) float or None if z<=0."""
    p = np.asarray(p_cam, dtype=float)
    if p[2] <= 1e-6:
        return None
    u = K[0, 0] * p[0] / p[2] + K[0, 2]
    v = K[1, 1] * p[1] / p[2] + K[1, 2]
    return (float(u), float(v))


def project_normal_arrow(
    c_cam: np.ndarray,
    n_cam: np.ndarray,
    K: np.ndarray,
    length_m: float,
    px_cap: float | None = 150.0,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Project the 3D segment c_cam -> c_cam + length_m * n_cam to image pixels.

    Optionally caps the on-screen arrow length to px_cap pixels so near-frontal
    or distant normals stay visible. Returns (p0, p1) int pixel tuples, or
    (None, None) if either endpoint is behind the camera.
    """
    c = np.asarray(c_cam, dtype=float)
    n = np.asarray(n_cam, dtype=float)
    a = project_point(c, K)
    b = project_point(c + length_m * n, K)
    if a is None or b is None:
        return None, None
    du, dv = b[0] - a[0], b[1] - a[1]
    L = float(np.hypot(du, dv))
    if px_cap is not None and L > px_cap and L > 0:
        s = px_cap / L
        b = (a[0] + du * s, a[1] + dv * s)
    return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
