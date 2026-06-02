"""Estimator pipeline tests with mocked segmenter."""
import numpy as np

from targetgeo.estimator import TargetGeoEstimator
from targetgeo.pose_types import DroneStateUe


def _make_disk_crop_mask(crop_h, crop_w, r=40):
    """Return a clean disk mask the size of a crop."""
    cy, cx = crop_h // 2, crop_w // 2
    yy, xx = np.ogrid[:crop_h, :crop_w]
    return ((xx - cx)**2 + (yy - cy)**2 <= r**2)


class _MockSegmenter:
    """Returns a clean disk mask sized to the input crop."""
    def __init__(self, score=0.85):
        self.score = score

    def segment(self, crop_bgr, text_prompts):
        h, w = crop_bgr.shape[:2]
        return _make_disk_crop_mask(h, w, r=min(h, w) // 3), self.score, text_prompts[0]


class _NullDetector:
    """Detector stub used when estimate(rgb, bbox, state) is called explicitly.
    estimate_from_image() isn't exercised in these tests."""
    def detect(self, _rgb):
        return None


def _make_state_ue(focal=1500.0):
    K = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1]], dtype=float)
    return DroneStateUe(
        camera_xyz_ue_m=(0.0, 0.0, 10.0),
        camera_pyr_deg=(0.0, 0.0, 0.0),
        K=K,
    )


def test_estimate_returns_ellipse_when_mask_present():
    est = TargetGeoEstimator(segmenter=_MockSegmenter(), detector=_NullDetector(), target_radius_m=2.5)
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (800, 400, 1100, 700)
    state = _make_state_ue()

    result = est.estimate(rgb, bbox, state)
    # We don't expect valid=True yet (pose/disambiguate/etc. still missing),
    # but ellipse and mask area should be populated as far as we got.
    assert result.disk_mask_area_px > 0
    assert result.ellipse is not None
    assert result.sam3_score == 0.85
    assert result.fit_method in ("hull", "direct")


def test_estimate_fails_when_segmenter_returns_none():
    class _Null:
        def segment(self, *_a, **_k):
            return None, -1.0, ""
    est = TargetGeoEstimator(segmenter=_Null(), detector=_NullDetector())
    result = est.estimate(np.zeros((1080, 1920, 3), dtype=np.uint8),
                          (800, 400, 1100, 700), _make_state_ue())
    assert not result.valid
    assert result.status == "no_mask"


def test_estimate_runs_chen_and_disambiguates():
    est = TargetGeoEstimator(segmenter=_MockSegmenter(), detector=_NullDetector(), target_radius_m=2.5)
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (800, 400, 1100, 700)
    state = _make_state_ue()
    result = est.estimate(rgb, bbox, state)
    assert result.offset_camera_m is not None
    assert result.normal_camera is not None
    # range is positive (target is in front of camera)
    assert result.range_m > 0
    assert result.disambiguation_method in ("world_up_axis", "visibility", "fallback")


def test_estimate_ue_path_produces_target_xyz():
    est = TargetGeoEstimator(segmenter=_MockSegmenter(), detector=_NullDetector(), target_radius_m=2.5)
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (800, 400, 1100, 700)
    state = _make_state_ue()
    result = est.estimate(rgb, bbox, state)
    assert result.target_xyz_ue_m is not None
    assert result.target_lat is None
    assert result.normal_world is not None


def test_estimate_gps_path_produces_target_lat_lon():
    from targetgeo.pose_types import DroneStateGps
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=float)
    state = DroneStateGps(
        camera_lat=37.5, camera_lon=127.0, camera_alt_m=100.0,
        camera_pyr_deg=(0.0, 0.0, 0.0),
        K=K,
    )
    est = TargetGeoEstimator(segmenter=_MockSegmenter(), detector=_NullDetector(), target_radius_m=2.5)
    result = est.estimate(np.zeros((1080, 1920, 3), dtype=np.uint8),
                          (800, 400, 1100, 700), state)
    assert result.target_xyz_ue_m is None
    assert result.target_lat is not None and 37.0 < result.target_lat < 38.0
    assert result.target_lon is not None and 126.5 < result.target_lon < 127.5
    assert result.target_alt_m is not None
    assert result.normal_world is not None


def test_estimate_returns_valid_true_with_covariance():
    est = TargetGeoEstimator(segmenter=_MockSegmenter(), detector=_NullDetector(), target_radius_m=2.5)
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (800, 400, 1100, 700)
    state = _make_state_ue()
    result = est.estimate(rgb, bbox, state)
    assert result.valid
    assert result.status == "ok"
    assert result.pos_cov_3x3 is not None
    assert result.pos_cov_3x3.shape == (3, 3)
    assert 0.0 <= result.normal_cone_deg <= 90.0


def test_estimate_flags_high_normal_cone():
    """When normal_cone_deg exceeds threshold, flag is added."""
    est = TargetGeoEstimator(
        segmenter=_MockSegmenter(),
        detector=_NullDetector(),
        target_radius_m=2.5,
        max_normal_cone_deg=0.0001,   # nearly impossible threshold
        pixel_sigma_px=10.0,           # large pixel noise → large cone
    )
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (800, 400, 1100, 700)
    result = est.estimate(rgb, bbox, _make_state_ue())
    assert "high_normal_cone" in result.flags


class _MockDetector:
    """Returns a fixed bbox."""
    def __init__(self, bbox=(800, 400, 1100, 700)):
        self.bbox = bbox

    def detect(self, _rgb):
        return self.bbox


def test_estimate_from_image_runs_detector_then_estimate():
    est = TargetGeoEstimator(
        segmenter=_MockSegmenter(),
        detector=_MockDetector(bbox=(800, 400, 1100, 700)),
        target_radius_m=2.5,
    )
    result = est.estimate_from_image(
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        _make_state_ue(),
    )
    assert result.valid
    assert result.target_xyz_ue_m is not None


def test_estimate_from_image_returns_no_detection_when_detector_empty():
    est = TargetGeoEstimator(
        segmenter=_MockSegmenter(),
        detector=_NullDetector(),
    )
    result = est.estimate_from_image(
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        _make_state_ue(),
    )
    assert not result.valid
    assert result.status == "no_detection"
