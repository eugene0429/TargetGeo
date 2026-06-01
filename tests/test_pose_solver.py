"""Verify vendored pose_solver imports and the Chen 2004 algorithm still works."""
import numpy as np

from seg_pose.pose_solver import solve_circle_pose, PoseSolverError


def test_solve_circle_pose_returns_two_candidates():
    # Synthetic circle at z=10m, radius 2.5m, frontoparallel view.
    # Conic of a circle at distance d, radius r, on principal axis is:
    #   Q = diag(1, 1, -(r/d)^2 * fx^2)
    # in normalized coords. With K = identity-ish, this should give two
    # candidates with center near (0, 0, 10) and normal near (0, 0, ±1).
    fx = fy = 1000.0
    cx, cy = 500.0, 500.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    r = 2.5
    d = 10.0
    # Circle on principal axis: image conic is circle of radius r*fx/d at (cx,cy).
    s = r * fx / d
    Q = np.array([
        [1, 0, -cx],
        [0, 1, -cy],
        [-cx, -cy, cx**2 + cy**2 - s**2],
    ], dtype=float)
    cands = solve_circle_pose(Q, K, r)
    assert len(cands) == 2
    for c, n in cands:
        assert c.shape == (3,)
        assert n.shape == (3,)
        assert c[2] > 0  # in front of camera
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-6)


def test_pose_solver_error_on_invalid_shape():
    import pytest
    with pytest.raises(PoseSolverError):
        solve_circle_pose(np.zeros((3, 4)), np.eye(3), 1.0)
