"""End-to-end UE path with real SAM 3.1 model. Marked slow."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from targetgeo import TargetGeoEstimator, DroneStateUe


REF_FRAME = Path("/home/sim2real/drone/data_real/frame_002850_t0095.00s.png")
# Approximate M1 rec_bbox for the reference frame (from Task 1 of the earlier PoC plan).
REF_BBOX = (780.0, 368.0, 1151.0, 741.0)


@pytest.mark.slow
def test_e2e_ue_path_real_sam3():
    if not REF_FRAME.exists():
        pytest.skip(f"Reference frame missing: {REF_FRAME}")

    rgb = cv2.imread(str(REF_FRAME))
    assert rgb is not None
    H, W = rgb.shape[:2]
    fov_deg = 60.0
    fx = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=float)

    state = DroneStateUe(
        camera_xyz_ue_m=(0.0, 0.0, 80.0),    # drone at 80m altitude, world origin
        camera_pyr_deg=(-30.0, 0.0, 0.0),    # pitched 30° down
        K=K,
    )

    est = TargetGeoEstimator(target_radius_m=2.5)  # 5x5 m target → radius 2.5 m
    result = est.estimate(rgb, REF_BBOX, state)

    assert result.valid, f"status={result.status} flags={result.flags}"
    assert result.target_xyz_ue_m is not None
    assert result.range_m is not None and result.range_m > 0
    assert result.fit_method in ("hull", "direct")

    # Sanity: target should be in front of the drone, within plausible distance.
    assert 10.0 < result.range_m < 500.0
