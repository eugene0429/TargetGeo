import numpy as np

from benchmarks.sam_compare.metrics import iou, ellipse_summary


def test_iou_identical_masks_is_one():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    assert iou(m, m) == 1.0


def test_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool); a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool); b[6:9, 6:9] = True
    assert iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.zeros((10, 10), dtype=bool); a[0:4, 0:4] = True   # 16 px
    b = np.zeros((10, 10), dtype=bool); b[2:6, 0:4] = True   # 16 px, overlap rows 2-3 = 8 px
    # intersection 8, union 24 -> 1/3
    assert abs(iou(a, b) - (8.0 / 24.0)) < 1e-9


def test_iou_both_empty_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    assert iou(a, a) == 0.0


def test_ellipse_summary_on_filled_disk_is_ok():
    mask = np.zeros((100, 100), dtype=bool)
    yy, xx = np.ogrid[:100, :100]
    mask[(xx - 50) ** 2 + (yy - 50) ** 2 <= 20 ** 2] = True
    s = ellipse_summary(mask)
    assert s["ok"] is True
    assert abs(s["centroid"][0] - 50) < 2 and abs(s["centroid"][1] - 50) < 2
    assert s["area"] > 0


def test_ellipse_summary_on_empty_is_not_ok():
    s = ellipse_summary(np.zeros((50, 50), dtype=bool))
    assert s["ok"] is False
    assert s["area"] == 0
