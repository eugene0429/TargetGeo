"""Verify public API is reachable via top-level package import."""

def test_public_api_imports():
    from targetgeo import (
        TargetGeoEstimator,
        TargetDetector,
        DroneStateUe,
        DroneStateGps,
        TargetGeoEstimate,
        PoseSigma,
        IntrinsicSigma,
    )
    assert TargetGeoEstimator is not None
    assert TargetDetector is not None
    assert DroneStateUe is not None
    assert DroneStateGps is not None
    assert TargetGeoEstimate is not None
