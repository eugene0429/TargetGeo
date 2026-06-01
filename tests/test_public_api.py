"""Verify public API is reachable via top-level package import."""

def test_public_api_imports():
    from seg_pose import (
        SegPoseEstimator,
        TargetDetector,
        DroneStateUe,
        DroneStateGps,
        TargetGeoEstimate,
        PoseSigma,
        IntrinsicSigma,
    )
    assert SegPoseEstimator is not None
    assert TargetDetector is not None
    assert DroneStateUe is not None
    assert DroneStateGps is not None
    assert TargetGeoEstimate is not None
