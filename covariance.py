"""Numerical Jacobian + covariance propagation for the localization pipeline."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .ellipse_core import EllipseFit, ellipse_params_to_conic
from .pose_solver import solve_circle_pose
from .transforms import (
    M_UE2CV,
    cam_cv_to_world,
    ue_rotation_matrix,
)


def _numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Forward finite-difference Jacobian. ~2x faster than central, lower accuracy."""
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    n_in = x.size
    n_out = f0.size
    J = np.zeros((n_out, n_in))
    for i in range(n_in):
        x_plus = x.copy(); x_plus[i] += eps
        J[:, i] = (f(x_plus) - f0) / eps
    return J


def _forward(
    ellipse_p: np.ndarray,
    fx_fy_cxy: np.ndarray,
    pose_p: np.ndarray,
    *,
    radius: float,
    chosen_idx: int,
) -> np.ndarray:
    """ellipse -> conic -> Chen 2004 -> chosen candidate -> world frame.

    Returns 6-vector [world_xyz; normal_world]. chosen_idx is fixed
    (not re-disambiguated) to keep the function continuous for finite
    differences. Caller must pass the idx selected at the un-perturbed
    operating point.
    """
    cx, cy, major, minor, theta = ellipse_p
    fit = EllipseFit(
        center_x=float(cx), center_y=float(cy),
        major=float(major), minor=float(minor),
        angle_deg=float(theta), residual=0.0, valid=True,
    )
    Q = ellipse_params_to_conic(fit)
    fx, fy, kcx, kcy = fx_fy_cxy
    K = np.array([[fx, 0.0, kcx], [0.0, fy, kcy], [0.0, 0.0, 1.0]])
    cands = solve_circle_pose(Q, K, float(radius))
    c_cv, n_cv = cands[int(chosen_idx)]

    loc = pose_p[:3]
    pyr = pose_p[3:]
    world_xyz = cam_cv_to_world(c_cv, loc, pyr)
    R_cam = ue_rotation_matrix(pyr)
    n_ue_local = M_UE2CV.T @ n_cv
    n_world = R_cam @ n_ue_local
    n_world = n_world / np.linalg.norm(n_world)
    return np.concatenate([world_xyz, n_world])


def compute_position_covariance(
    *,
    ellipse_params: dict,
    K: np.ndarray,
    radius: float,
    drone_pose,                       # DronePose
    pixel_sigma_px: float,
    pose_sigma,                       # PoseSigma
    intrinsic_sigma,                  # IntrinsicSigma
    chosen_idx: int,
) -> tuple[np.ndarray, float]:
    """Returns (pos_cov_3x3, normal_cone_deg)."""
    ellipse_p = np.array([
        ellipse_params["cx"], ellipse_params["cy"],
        ellipse_params["major"], ellipse_params["minor"],
        ellipse_params["theta"],
    ], dtype=float)
    fx_fy_cxy = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=float)
    pose_p = np.concatenate([
        np.asarray(drone_pose.loc_xyz_ue, dtype=float),
        np.asarray(drone_pose.pyr_deg, dtype=float),
    ])

    def f_all(x):
        e = x[:5]
        k = x[5:9]
        p = x[9:15]
        return _forward(e, k, p, radius=radius, chosen_idx=chosen_idx)

    x0 = np.concatenate([ellipse_p, fx_fy_cxy, pose_p])
    J = _numerical_jacobian(f_all, x0, eps=1e-4)         # (6, 15)

    # Σ_in: block-diagonal independence assumption
    var_ellipse = (pixel_sigma_px ** 2) * np.ones(5)     # cx,cy,major,minor,theta in px / rad
    var_K = np.array([
        intrinsic_sigma.fx_px ** 2,
        intrinsic_sigma.fx_px ** 2,
        intrinsic_sigma.cxy_px ** 2,
        intrinsic_sigma.cxy_px ** 2,
    ])
    var_pose = np.array([
        pose_sigma.pos_m ** 2, pose_sigma.pos_m ** 2, pose_sigma.pos_m ** 2,
        pose_sigma.att_deg ** 2, pose_sigma.att_deg ** 2, pose_sigma.att_deg ** 2,
    ])
    Sigma_in = np.diag(np.concatenate([var_ellipse, var_K, var_pose]))

    Sigma_out = J @ Sigma_in @ J.T                        # (6, 6)
    pos_cov = Sigma_out[:3, :3]
    pos_cov = 0.5 * (pos_cov + pos_cov.T)                 # symmetrize numerical drift

    normal_cov = Sigma_out[3:, 3:]
    normal_cov = 0.5 * (normal_cov + normal_cov.T)
    # Project onto tangent plane: tangent variance ~ largest eigenvalue
    eigvals = np.linalg.eigvalsh(normal_cov)
    var_tangent = max(float(eigvals[-1]), 0.0)
    cone_rad = np.arctan(np.sqrt(var_tangent))
    cone_deg = float(np.degrees(cone_rad))
    return pos_cov, cone_deg
