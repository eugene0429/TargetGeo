import numpy as np

from seg_pose.viewer.overlays import project_point, project_normal_arrow


def _K():
    return np.array([[100.0, 0.0, 50.0],
                     [0.0, 100.0, 50.0],
                     [0.0, 0.0, 1.0]])


def test_project_point_center():
    # point on optical axis at z=10 -> principal point
    assert project_point(np.array([0.0, 0.0, 10.0]), _K()) == (50.0, 50.0)


def test_project_point_behind_camera_returns_none():
    assert project_point(np.array([0.0, 0.0, -1.0]), _K()) is None


def test_project_normal_arrow_sideways():
    c = np.array([0.0, 0.0, 10.0])
    n = np.array([1.0, 0.0, 0.0])  # +x in camera frame
    p0, p1 = project_normal_arrow(c, n, _K(), length_m=2.0, px_cap=None)
    # base projects to center; tip = (2,0,10) -> u = 100*2/10 + 50 = 70
    assert p0 == (50, 50)
    assert p1 == (70, 50)


def test_project_normal_arrow_caps_pixel_length():
    c = np.array([0.0, 0.0, 10.0])
    n = np.array([1.0, 0.0, 0.0])
    p0, p1 = project_normal_arrow(c, n, _K(), length_m=2.0, px_cap=10.0)
    # uncapped dx would be 20px; capped to 10px
    assert p0 == (50, 50)
    assert p1 == (60, 50)


from seg_pose.viewer.overlays import (
    draw_bbox, draw_ellipse, draw_mask,
    BBOX_COLOR, ELLIPSE_COLOR,
)


def _blank():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_draw_bbox_colors_edge():
    img = _blank()
    draw_bbox(img, (10, 10, 40, 40))
    # top edge pixel should be the bbox color
    assert tuple(int(c) for c in img[10, 25]) == BBOX_COLOR


def test_draw_bbox_none_is_noop():
    img = _blank()
    draw_bbox(img, None)
    assert img.sum() == 0


def test_draw_ellipse_marks_center():
    img = _blank()
    ellipse = {"cx": 50.0, "cy": 50.0, "major": 40.0, "minor": 20.0, "theta": 0.0}
    draw_ellipse(img, ellipse)
    # center dot drawn in ELLIPSE_COLOR
    assert tuple(int(c) for c in img[50, 50]) == ELLIPSE_COLOR


def test_draw_mask_blends_region():
    img = _blank()
    # a square contour
    contour = np.array([[[20, 20]], [[20, 60]], [[60, 60]], [[60, 20]]], dtype=np.int32)
    draw_mask(img, contour, alpha=0.5)
    assert img[40, 40].sum() > 0  # interior tinted


def test_draw_mask_none_is_noop():
    img = _blank()
    draw_mask(img, None)
    assert img.sum() == 0
