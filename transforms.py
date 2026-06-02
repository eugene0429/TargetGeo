"""UE world <-> OpenCV camera coordinate transforms.

Constants and helpers locked from sanity_check_coords.py findings:
- UE Rotator: R @ (1,0,0) = (cP*cY, cP*sY, sP), composition Rz(Y)@Ry_ue(P)@Rx(R)
- UE camera local -> OpenCV: cv = (uy, -uz, ux)
- FOV is horizontal, distortion = 0
- Target mesh forward axis = +X local (sign convention; mesh is double-sided)
"""

from __future__ import annotations

import numpy as np

# UE camera local -> OpenCV camera local mapping. det = -1 (handedness flip).
M_UE2CV = np.array([[0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0]])

# Target mesh local forward axis convention (line; sign arbitrary because the
# UE mesh is double-sided per sanity check). Targets in this dataset are flat
# disks lying on the ground (T_RotP=T_RotR=0, only Yaw varies); their face
# normal in world coords is vertical +Z, which equals the target-actor's
# local +Z under yaw-only rotation. (sanity_check_coords.py earlier picked +X
# as a statistical artifact — its "face-on" filter via mask-aspect was
# dominated by small/aliased masks rather than true face-on geometry, biasing
# the mean|cos| in favor of axes that happen to track camera yaw rather than
# the true mesh forward.)
TARGET_LOCAL_FORWARD = np.array([0.0, 0.0, 1.0])


def ue_rotation_matrix(pyr_deg) -> np.ndarray:
    """3x3 rotation matrix in UE LH coords (X fwd, Y right, Z up).

    Composition R = Rz(yaw) @ Ry_ue(pitch) @ Rx(roll) such that
    R @ (1,0,0) = (cP*cY, cP*sY, sP). Pitch sign is flipped vs RH-standard
    Ry to match UE FRotator convention.
    """
    p, y, r = np.deg2rad(np.asarray(pyr_deg, dtype=float))
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    cr, sr = np.cos(r), np.sin(r)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def fov_to_intrinsics(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Build OpenCV K matrix assuming horizontal FOV, principal point at image center,
    no distortion, square pixels (fx = fy)."""
    f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]])


def world_to_cam_cv(world_pt, cam_loc, cam_rot_pyr) -> np.ndarray:
    """Transform a UE world point to OpenCV camera coords."""
    R_cam = ue_rotation_matrix(cam_rot_pyr)
    local_ue = R_cam.T @ (np.asarray(world_pt, float) - np.asarray(cam_loc, float))
    return M_UE2CV @ local_ue


def cam_cv_to_world(p_cv, cam_loc, cam_rot_pyr) -> np.ndarray:
    """Inverse of world_to_cam_cv: OpenCV cam coords back into UE world coords."""
    R_cam = ue_rotation_matrix(cam_rot_pyr)
    local_ue = M_UE2CV.T @ np.asarray(p_cv, dtype=float)
    return R_cam @ local_ue + np.asarray(cam_loc, dtype=float)


def project_point_to_image(world_pt, cam_loc, cam_rot_pyr, K):
    """Project UE world point to image pixel. Returns None if behind camera."""
    p_cv = world_to_cam_cv(world_pt, cam_loc, cam_rot_pyr)
    if p_cv[2] <= 0:
        return None
    return np.array([(K[0, 0] * p_cv[0]) / p_cv[2] + K[0, 2],
                     (K[1, 1] * p_cv[1]) / p_cv[2] + K[1, 2]])


def gt_normal_in_cam(cam_loc, cam_rot, target_rot) -> np.ndarray:
    """Compute target normal (using +Z local convention) in OpenCV cam coords.

    Mesh is double-sided so the LINE matters, not the sign. Returns unit vector.
    """
    R_cam = ue_rotation_matrix(cam_rot)
    R_tgt = ue_rotation_matrix(target_rot)
    n_world = R_tgt @ TARGET_LOCAL_FORWARD
    n_cam_ue = R_cam.T @ n_world
    n_cv = M_UE2CV @ n_cam_ue
    return n_cv / np.linalg.norm(n_cv)


def world_up_in_cam_cv(cam_rot) -> np.ndarray:
    """Express UE world up (+Z) in OpenCV cam coords. Unit vector.

    Useful as a prior for disambiguating Chen-2004 candidates when targets are
    known to be flat on the ground (their face normal aligned with world up).
    """
    R_cam = ue_rotation_matrix(cam_rot)
    up_cam_ue = R_cam.T @ np.array([0.0, 0.0, 1.0])
    up_cv = M_UE2CV @ up_cam_ue
    return up_cv / np.linalg.norm(up_cv)


def line_angle_error(n1, n2) -> float:
    """Angle between two unit-vector LINES (sign-agnostic), in radians."""
    cos_abs = float(np.abs(np.dot(n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2))))
    return float(np.arccos(min(1.0, cos_abs)))


# ----------------------------------------------------------------------------
# ENU convention (added for targetgeo GPS path).
# Camera Euler (pitch, yaw, roll) interpreted in ENU world frame:
#   - yaw around Up (+Z), pitch around East (+X), roll around forward.
# Returned matrix maps camera-frame point → ENU world-frame point:
#   p_enu = R_enu @ p_cam
# ----------------------------------------------------------------------------

def enu_rotation_matrix(pyr_deg) -> np.ndarray:
    """ENU rotation: camera frame → ENU world frame.

    Uses ZYX intrinsic Euler (yaw around Z, pitch around Y, roll around X)
    in ENU (East/North/Up) frame, matching standard aerospace gimbal telemetry.
    """
    pitch, yaw, roll = [float(v) for v in pyr_deg]
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cy, sy = np.cos(np.radians(yaw)),   np.sin(np.radians(yaw))
    cr, sr = np.cos(np.radians(roll)),  np.sin(np.radians(roll))
    # R_z(yaw) @ R_y(pitch) @ R_x(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    return Rz @ Ry @ Rx


def world_up_in_cam_enu(cam_pyr_deg) -> np.ndarray:
    """World-up (ENU +Z) expressed in OpenCV camera coords.

    Returned unit vector for `disambiguate_visibility(world_up_cv=...)`.
    """
    R_enu = enu_rotation_matrix(cam_pyr_deg)  # cam → enu
    up_enu = np.array([0.0, 0.0, 1.0])         # ENU up
    up_cam = R_enu.T @ up_enu                  # enu → cam (transpose = inverse for rotation)
    n = np.linalg.norm(up_cam)
    return up_cam / n if n > 0 else up_cam
