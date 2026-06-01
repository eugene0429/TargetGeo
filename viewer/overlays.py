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


def draw_normal(img, result, K, arrow_len_m: float, px_cap: float = 150.0) -> None:
    """Draw the chosen normal (solid) and the rejected candidate (faint).

    Anchors each arrow at its own camera-frame center, projected through K.
    No-op if there are no candidates.
    """
    candidates = getattr(result, "candidates", None)
    if not candidates:
        return
    chosen = getattr(result, "chosen_idx", None)
    for idx, (c_cam, n_cam) in enumerate(candidates):
        is_chosen = (idx == chosen)
        color = NORMAL_COLOR if is_chosen else NORMAL_REJECT_COLOR
        thickness = 2 if is_chosen else 1
        p0, p1 = project_normal_arrow(c_cam, n_cam, K, arrow_len_m, px_cap=px_cap)
        if p0 is None or p1 is None:
            continue
        cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.2)


def _fmt_vec(v, nd=3):
    if v is None:
        return "N/A"
    return "(" + ", ".join(f"{x:+.{nd}f}" for x in v) + ")"


def draw_hud(img, result, origin=(8, 8), line_h=20) -> None:
    """Draw a translucent text panel with pipeline diagnostics."""
    has_geo = getattr(result, "lat", None) is not None
    geo = (f"lat {result.lat:.6f}  lon {result.lon:.6f}  alt {result.alt_m:.2f} m"
           if has_geo else "lat/lon/alt: N/A - no telemetry")
    rng = "N/A" if getattr(result, "range_m", None) is None else f"{result.range_m:.2f} m"
    cone = "N/A" if getattr(result, "cone_deg", None) is None else f"{result.cone_deg:.2f} deg"
    lines = [
        f"status: {result.status}   sam={getattr(result, 'sam_score', 0.0):.3f}"
        f"   fit={getattr(result, 'fit_method', '-')}"
        f"   disambig={getattr(result, 'disambiguation_method', '-')}",
        f"range: {rng}    normal_cone: {cone}",
        f"normal_camera: {_fmt_vec(getattr(result, 'normal_camera', None))}",
        f"normal_world : {_fmt_vec(getattr(result, 'normal_world', None))}",
        f"geodetic: {geo}",
        f"flags: {', '.join(getattr(result, 'flags', []) or []) or '-'}",
    ]
    x0, y0 = origin
    w = 560
    h = line_h * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0, dst=img)
    for i, line in enumerate(lines):
        y = y0 + 18 + i * line_h
        cv2.putText(img, line, (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, HUD_FG, 1, cv2.LINE_AA)


def render(img, result, K, layers: dict, arrow_len_m: float) -> np.ndarray:
    """Composite enabled overlay layers onto a copy of img. Returns the copy."""
    out = img.copy()
    if result is None:
        return out
    if layers.get("mask"):
        draw_mask(out, getattr(result, "mask_contour", None))
    if layers.get("bbox"):
        draw_bbox(out, getattr(result, "bbox", None))
    if layers.get("ellipse"):
        draw_ellipse(out, getattr(result, "ellipse", None))
    if layers.get("normal"):
        draw_normal(out, result, K, arrow_len_m)
    if layers.get("hud"):
        draw_hud(out, result)
    return out
