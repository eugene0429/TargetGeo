import numpy as np

from targetgeo.viewer.overlays import project_point, project_normal_arrow


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


from targetgeo.viewer.overlays import (
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


from types import SimpleNamespace
from targetgeo.viewer.overlays import draw_normal, draw_hud, render


def _result_ok():
    return SimpleNamespace(
        status="ok", valid=True,
        bbox=(20, 20, 80, 80),
        mask_contour=np.array([[[20, 20]], [[20, 80]], [[80, 80]], [[80, 20]]], dtype=np.int32),
        ellipse={"cx": 50.0, "cy": 50.0, "major": 40.0, "minor": 30.0, "theta": 10.0},
        candidates=[(np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0])),
                    (np.array([0.0, 0.0, 10.0]), np.array([0.0, 1.0, 0.0]))],
        chosen_idx=0,
        normal_camera=(1.0, 0.0, 0.0),
        offset_camera_m=(0.0, 0.0, 10.0),
        range_m=10.0, cone_deg=1.5,
        disambiguation_method="visibility", fit_method="hull", sam_score=0.9,
        flags=[], lat=None, lon=None, alt_m=None, normal_world=None,
    )


def _K200():
    return np.array([[200.0, 0.0, 50.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]])


def test_draw_normal_draws_something():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_normal(img, _result_ok(), _K200(), arrow_len_m=2.0)
    assert img.sum() > 0


def test_draw_hud_returns_same_shape_and_writes_text():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, _result_ok())
    assert img.shape == (480, 640, 3)
    assert img.sum() > 0


def test_draw_hud_shows_na_without_telemetry():
    # Smoke: no telemetry -> still renders without error
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, _result_ok())  # lat/lon None
    assert img.sum() > 0


def test_render_all_layers_changes_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = render(img, _result_ok(), _K200(),
                 layers={"bbox": True, "ellipse": True, "mask": True,
                         "normal": True, "hud": True},
                 arrow_len_m=2.0)
    assert out.shape == img.shape
    assert out.sum() > 0


def test_render_respects_layer_toggles():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = render(img, _result_ok(), _K200(),
                 layers={"bbox": False, "ellipse": False, "mask": False,
                         "normal": False, "hud": False},
                 arrow_len_m=2.0)
    assert out.sum() == 0
