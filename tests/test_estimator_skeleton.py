"""Estimator skeleton + DI tests (no estimate() logic yet)."""
import numpy as np

from targetgeo.estimator import TargetGeoEstimator


class _DummySegmenter:
    def segment(self, *_args, **_kw):
        return None, 0.0, ""


class _DummyDetector:
    def detect(self, _rgb):
        return None


def _make_estimator(**overrides):
    return TargetGeoEstimator(
        segmenter=_DummySegmenter(),
        detector=_DummyDetector(),
        **overrides,
    )


def test_estimator_accepts_injected_segmenter_and_detector():
    est = _make_estimator(target_radius_m=2.5)
    assert est.target_radius_m == 2.5
    assert est.segmenter is not None
    assert est.detector is not None


def test_estimator_init_defaults():
    est = _make_estimator()
    assert est.target_radius_m == 2.5
    assert est.min_mask_area_px > 0
    assert est.max_normal_cone_deg > 0
    assert len(est.text_prompts) >= 1


def test_estimator_estimate_callable():
    est = _make_estimator()
    assert callable(est.estimate)
    assert callable(est.estimate_from_image)
