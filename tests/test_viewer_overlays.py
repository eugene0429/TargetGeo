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
