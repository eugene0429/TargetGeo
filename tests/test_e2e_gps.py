"""End-to-end GPS path with real SAM 3.1. Marked slow."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from targetgeo import TargetGeoEstimator, DroneStateGps


REF_FRAME = Path("/home/sim2real/drone/data_real/frame_002850_t0095.00s.png")
REF_BBOX = (780.0, 368.0, 1151.0, 741.0)


@pytest.mark.slow
def test_e2e_gps_path_real_sam3():
    if not REF_FRAME.exists():
        pytest.skip(f"Reference frame missing: {REF_FRAME}")

    rgb = cv2.imread(str(REF_FRAME))
    H, W = rgb.shape[:2]
    fov_deg = 60.0
    fx = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=float)

    # Synthetic GPS telemetry near the C++ node reference (Korean coords).
    state = DroneStateGps(
        camera_lat=38.069328,
        camera_lon=127.360708,
        camera_alt_m=80.0,
        camera_pyr_deg=(-30.0, 0.0, 0.0),
        K=K,
    )

    est = TargetGeoEstimator(target_radius_m=2.5)
    result = est.estimate(rgb, REF_BBOX, state)

    assert result.valid, f"status={result.status} flags={result.flags}"
    assert result.target_lat is not None
    assert result.target_lon is not None
    assert result.target_alt_m is not None
    # Target within ~500m of camera
    assert 37.9 < result.target_lat < 38.2
    assert 127.2 < result.target_lon < 127.5
