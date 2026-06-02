"""Dataclass tests for targetgeo types."""
import numpy as np

from targetgeo.pose_types import (
    DroneStateUe, DroneStateGps, TargetGeoEstimate,
    PoseSigma, IntrinsicSigma,
    DEFAULT_POSE_SIGMA, DEFAULT_INTRINSIC_SIGMA,
)


def test_drone_state_ue_basic():
    K = np.eye(3)
    s = DroneStateUe(
        camera_xyz_ue_m=(0.0, 0.0, 10.0),
        camera_pyr_deg=(0.0, 0.0, 0.0),
        K=K,
    )
    assert s.frame_id == -1
    assert s.timestamp_s == 0.0
    assert s.K is K


def test_drone_state_gps_basic():
    K = np.eye(3)
    s = DroneStateGps(
        camera_lat=37.5, camera_lon=127.0, camera_alt_m=100.0,
        camera_pyr_deg=(0.0, 0.0, 0.0),
        K=K,
    )
    assert s.camera_lat == 37.5


def test_target_geo_estimate_default_invalid():
    t = TargetGeoEstimate(
        target_xyz_ue_m=None, target_lat=None, target_lon=None, target_alt_m=None,
        offset_camera_m=None, range_m=None,
        normal_camera=None, normal_world=None,
        pos_cov_3x3=None, normal_cone_deg=0.0,
        ellipse=None, disk_mask_area_px=0, sam3_score=0.0,
        fit_method="(failed)", disambiguation_method="(none)",
        valid=False, status="not_run",
    )
    assert not t.valid
    assert t.flags == []


def test_default_sigmas_exist():
    assert isinstance(DEFAULT_POSE_SIGMA, PoseSigma)
    assert isinstance(DEFAULT_INTRINSIC_SIGMA, IntrinsicSigma)
    assert DEFAULT_POSE_SIGMA.pos_m > 0
    assert DEFAULT_POSE_SIGMA.att_deg > 0
