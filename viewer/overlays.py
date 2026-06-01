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


def draw_bbox(img, bbox, color=BBOX_COLOR, thickness=2) -> None:
    """Draw an (x1, y1, x2, y2) rectangle. No-op if bbox is None."""
    if bbox is None:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_ellipse(img, ellipse, color=ELLIPSE_COLOR, thickness=2) -> None:
    """Draw the fitted ellipse outline + center dot. ellipse: dict cx,cy,major,minor,theta."""
    if ellipse is None:
        return
    center = (int(round(ellipse["cx"])), int(round(ellipse["cy"])))
    axes = (int(round(ellipse["major"] / 2.0)), int(round(ellipse["minor"] / 2.0)))
    cv2.ellipse(img, center, axes, float(ellipse["theta"]), 0.0, 360.0,
                color, thickness, cv2.LINE_AA)
    cv2.circle(img, center, 3, color, -1, cv2.LINE_AA)


def draw_mask(img, contour, color=MASK_COLOR, alpha=0.35) -> None:
    """Translucent fill of the mask contour. No-op if contour is None."""
    if contour is None:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.asarray(contour, dtype=np.int32).reshape(-1, 2)], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
