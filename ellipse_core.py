"""Mask -> ellipse parameters -> conic matrix."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EllipseFit:
    center_x: float
    center_y: float
    major: float          # major axis length (full, not semi)
    minor: float          # minor axis length (full)
    angle_deg: float      # CCW rotation of major axis from +x
    residual: float       # rms algebraic residual of x^T Q x (NOT pixel distance)
    valid: bool


def fit_ellipse_to_mask(mask: np.ndarray, min_area: int = 30) -> EllipseFit:
    """Extract a single ellipse from a binary mask via outer contour + cv2.fitEllipse."""
    if mask.dtype != np.uint8:
        m = mask.astype(np.uint8)
    else:
        m = mask
    if m.sum() < min_area:
        return EllipseFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return EllipseFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    # Use largest contour
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return EllipseFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    (cx, cy), (mn, mj), ang = cv2.fitEllipse(cnt)
    if mn < 1e-3 or mj < 1e-3:
        return EllipseFit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    # cv2.fitEllipse returns (axes[0], axes[1], angle) where `angle` is the
    # orientation (CCW from +x, in degrees) of axes[0] -- the rectangle's
    # "width" side. With OpenCV >= 4 the convention is mj >= mn (axes[1] is
    # the major), so `ang` is the orientation of the MINOR axis. We want
    # `angle_deg` to mean orientation of the MAJOR axis -> subtract 90°.
    # Defensively handle the (unlikely) mn > mj case by swapping first.
    if mn > mj:
        mn, mj = mj, mn
        ang = (ang + 90.0) % 180.0
    ang = (ang - 90.0) % 180.0
    # rms residual: project contour points onto algebraic conic
    pts = cnt.reshape(-1, 2).astype(np.float64)
    fit = EllipseFit(cx, cy, mj, mn, ang, 0.0, True)
    Q = ellipse_params_to_conic(fit)
    homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    algebraic = (homog @ Q * homog).sum(axis=1)
    fit.residual = float(np.sqrt((algebraic ** 2).mean()))
    return fit


def ellipse_params_to_conic(fit: EllipseFit) -> np.ndarray:
    """Build 3x3 symmetric conic Q such that x^T Q x = 0 for points (u, v, 1) on the ellipse.

    Convention: Q is normalized so the constant term is 1, matching standard implicit form
    A u^2 + B u v + C v^2 + D u + E v + F = 0.
    """
    cx, cy, mj, mn, ang_deg = fit.center_x, fit.center_y, fit.major, fit.minor, fit.angle_deg
    a = mj / 2.0   # semi-major
    b = mn / 2.0   # semi-minor
    th = np.deg2rad(ang_deg)
    ct, st = np.cos(th), np.sin(th)
    # Rotated, translated implicit ellipse:
    #   (((u-cx)*ct + (v-cy)*st) / a)^2 + ((-(u-cx)*st + (v-cy)*ct) / b)^2 = 1
    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)
    A = inv_a2 * ct * ct + inv_b2 * st * st
    B = 2.0 * (inv_a2 - inv_b2) * ct * st
    C = inv_a2 * st * st + inv_b2 * ct * ct
    D = -2.0 * (A * cx + 0.5 * B * cy)
    E = -2.0 * (0.5 * B * cx + C * cy)
    F = A * cx * cx + B * cx * cy + C * cy * cy - 1.0
    return np.array([[A,        B / 2.0, D / 2.0],
                     [B / 2.0,  C,       E / 2.0],
                     [D / 2.0,  E / 2.0, F]])


# ----------------------------------------------------------------------------
# Hull-based robust ellipse fit (added for seg_pose).
# Convex hull bridges over concave dents caused by overlay occlusions (HUD
# crosshair, IR insets, text), so cv2.fitEllipse is not pulled inward.
# ----------------------------------------------------------------------------

def fit_ellipse_hull(
    mask: np.ndarray,
    *,
    min_contour_points: int = 5,
    min_minor_axis_px: float = 4.0,
) -> tuple[EllipseFit | None, str]:
    """SAM mask → robust ellipse via convex hull.

    Returns (fit, method) where method ∈ {"hull", "direct", "(failed)"}.
    Falls back to direct fit only if hull cannot form >=5 points.
    """
    if mask is None or mask.sum() == 0:
        return None, "(failed)"

    m8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, "(failed)"

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < min_contour_points:
        return None, "(failed)"

    # Primary: convex hull then fitEllipse.
    hull = cv2.convexHull(largest)
    if len(hull) >= min_contour_points:
        try:
            params = cv2.fitEllipse(hull)
            fit = _params_to_ellipse_fit(params)
            if fit.minor >= min_minor_axis_px:
                return fit, "hull"
        except cv2.error:
            pass

    # Fallback: direct fit on raw contour (only reached if hull <5 points).
    try:
        params = cv2.fitEllipse(largest)
        fit = _params_to_ellipse_fit(params)
        if fit.minor >= min_minor_axis_px:
            return fit, "direct"
    except cv2.error:
        pass

    return None, "(failed)"


def _params_to_ellipse_fit(params) -> EllipseFit:
    (cx, cy), (a, b), angle = params
    # cv2.fitEllipse may return (major, minor) in either order; canonicalize.
    if a >= b:
        major, minor = a, b
    else:
        major, minor = b, a
        angle = angle + 90.0
    return EllipseFit(
        center_x=float(cx), center_y=float(cy),
        major=float(major), minor=float(minor),
        angle_deg=float(angle) % 180.0,
        residual=0.0, valid=True,
    )
