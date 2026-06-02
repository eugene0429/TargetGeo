"""Verify vendored covariance.py with new relative imports."""
import numpy as np
import pytest

from targetgeo.covariance import compute_position_covariance


class _Sigma:
    """Minimal stub matching PoseSigma / IntrinsicSigma duck type."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _DronePose:
    def __init__(self, loc_xyz_ue, pyr_deg):
        self.loc_xyz_ue = loc_xyz_ue
        self.pyr_deg = pyr_deg


def test_covariance_returns_psd_3x3_and_cone_deg():
    # Realistic-ish ellipse parameters.
    ellipse_params = dict(cx=960.0, cy=540.0, major=120.0, minor=100.0, theta=15.0)
    fx = fy = 1500.0
    K = np.array([[fx, 0, 960], [0, fy, 540], [0, 0, 1]], dtype=float)
    drone_pose = _DronePose(loc_xyz_ue=(0.0, 0.0, 10.0), pyr_deg=(0.0, 0.0, 0.0))
    pose_sigma = _Sigma(pos_m=1.0, att_deg=0.5)
    intrinsic_sigma = _Sigma(fx_px=1.0, cxy_px=0.5)

    pos_cov, cone_deg = compute_position_covariance(
        ellipse_params=ellipse_params,
        K=K,
        radius=2.5,
        drone_pose=drone_pose,
        pixel_sigma_px=0.5,
        pose_sigma=pose_sigma,
        intrinsic_sigma=intrinsic_sigma,
        chosen_idx=0,
    )
    assert pos_cov.shape == (3, 3)
    np.testing.assert_allclose(pos_cov, pos_cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(pos_cov)
    assert (eigvals >= -1e-9).all(), f"pos_cov not PSD: {eigvals}"
    assert 0.0 <= cone_deg <= 90.0
