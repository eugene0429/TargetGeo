import numpy as np

from targetgeo.viewer.inference import FrameAnalyzer, FrameResult


class _FakeDetector:
    """Returns a fixed bbox covering the synthetic disk."""
    def __init__(self, bbox):
        self.bbox = bbox

    def detect(self, rgb):
        return self.bbox


class _FakeSegmenter:
    """Returns a filled-circle mask for the crop region it is given."""
    def __init__(self, score=0.9):
        self.score = score

    def segment(self, crop_bgr, text_prompts):
        h, w = crop_bgr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r = min(h, w) * 0.4
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        return mask, self.score, text_prompts[0]


def _synthetic_frame():
    # 400x400 black frame; a bright disk centered at (200,200), radius ~60
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:400, :400]
    disk = ((xx - 200) ** 2 + (yy - 200) ** 2) <= 60 ** 2
    img[disk] = 255
    return img


def _K():
    return np.array([[600.0, 0.0, 200.0], [0.0, 600.0, 200.0], [0.0, 0.0, 1.0]])


def test_analyzer_no_telemetry_produces_camera_frame_normal():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector((140, 140, 260, 260)),
        segmenter=_FakeSegmenter(),
    )
    res = analyzer.analyze(_synthetic_frame(), _K(), radius=2.5, n_prompts=1)
    assert isinstance(res, FrameResult)
    assert res.valid is True and res.status == "ok"
    assert res.bbox == (140, 140, 260, 260)
    assert res.ellipse is not None and res.mask_contour is not None
    assert res.normal_camera is not None and len(res.normal_camera) == 3
    assert res.range_m is not None and res.range_m > 0
    assert res.cone_deg is not None
    assert len(res.candidates) == 2 and res.chosen_idx in (0, 1)
    assert res.disambiguation_method in ("visibility", "fallback")
    # no telemetry -> geodetic fields stay None
    assert res.lat is None and res.lon is None and res.alt_m is None
    assert res.normal_world is None


def test_analyzer_no_detection_returns_status():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector(None),
        segmenter=_FakeSegmenter(),
    )
    res = analyzer.analyze(_synthetic_frame(), _K(), radius=2.5, n_prompts=1)
    assert res.valid is False
    assert res.status == "no_detection"
    assert res.bbox is None


from targetgeo.pose_types import DroneStateGps


def test_analyzer_gps_telemetry_fills_geodetic():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector((140, 140, 260, 260)),
        segmenter=_FakeSegmenter(),
    )
    K = _K()
    state = DroneStateGps(
        camera_lat=37.5063, camera_lon=127.0125, camera_alt_m=80.0,
        camera_pyr_deg=(-30.0, 45.0, 0.0), K=K,
    )
    res = analyzer.analyze(_synthetic_frame(), K, radius=2.5, n_prompts=1, telemetry=state)
    assert res.valid is True
    assert res.lat is not None and res.lon is not None and res.alt_m is not None
    assert res.normal_world is not None and len(res.normal_world) == 3
    # world-up prior should be used for disambiguation
    assert res.disambiguation_method in ("world_up_axis", "fallback")
