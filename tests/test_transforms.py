"""Verify vendored transforms + new ENU helpers."""
import numpy as np

from targetgeo.transforms import (
    ue_rotation_matrix,
    cam_cv_to_world,
    world_up_in_cam_cv,
    enu_rotation_matrix,
    world_up_in_cam_enu,
    M_UE2CV,
)


def test_ue_rotation_matrix_is_orthogonal():
    R = ue_rotation_matrix((10.0, 20.0, 30.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_enu_rotation_matrix_is_orthogonal():
    R = enu_rotation_matrix((10.0, 20.0, 30.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


def test_enu_identity_at_zero_angles():
    R = enu_rotation_matrix((0.0, 0.0, 0.0))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


def test_world_up_in_cam_enu_unit_vector():
    up = world_up_in_cam_enu((30.0, 0.0, 0.0))
    np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-10)
