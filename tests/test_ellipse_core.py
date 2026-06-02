"""Hull-based ellipse fit tests."""
import numpy as np
import cv2

from targetgeo.ellipse_core import (
    EllipseFit, ellipse_params_to_conic, fit_ellipse_hull,
)


def _make_disk_mask(H, W, cx, cy, r, dtype=bool):
    yy, xx = np.ogrid[:H, :W]
    return ((xx - cx)**2 + (yy - cy)**2 <= r**2)


def test_hull_fit_clean_circle():
    mask = _make_disk_mask(400, 400, 200, 200, 80)
    fit, method = fit_ellipse_hull(mask)
    assert fit is not None
    assert method == "hull"
    assert abs(fit.center_x - 200) < 2
    assert abs(fit.center_y - 200) < 2
    # major and minor should be near diameter (~160)
    assert 150 < fit.major < 170
    assert 150 < fit.minor < 170


def test_hull_fit_recovers_from_notch():
    # Disk with a rectangular notch cut out (simulating HUD overlay).
    mask = _make_disk_mask(400, 400, 200, 200, 80).copy()
    mask[195:205, 120:280] = False  # horizontal slit across disk
    fit_hull, method = fit_ellipse_hull(mask)
    fit_direct_only = _direct_fit_for_comparison(mask)
    assert fit_hull is not None
    assert method == "hull"
    # Hull fit should be CLOSER to the true circle than direct fit.
    # True major ~160; hull should be close to 160; direct will be pulled smaller.
    assert abs(fit_hull.major - 160) < abs(fit_direct_only.major - 160) + 1


def _direct_fit_for_comparison(mask):
    m8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    params = cv2.fitEllipse(largest)
    (cx, cy), (major, minor), angle = params
    if major < minor:
        major, minor, angle = minor, major, angle + 90
    return EllipseFit(
        center_x=cx, center_y=cy, major=major, minor=minor,
        angle_deg=angle, residual=0.0, valid=True,
    )


def test_hull_fit_returns_none_for_empty_mask():
    mask = np.zeros((100, 100), dtype=bool)
    fit, method = fit_ellipse_hull(mask)
    assert fit is None
    assert method == "(failed)"


def test_hull_fit_returns_none_for_tiny_mask():
    mask = np.zeros((100, 100), dtype=bool)
    mask[50:52, 50:53] = True  # 2x3 pixels
    fit, method = fit_ellipse_hull(mask, min_minor_axis_px=4.0)
    assert fit is None or fit.minor < 4.0
