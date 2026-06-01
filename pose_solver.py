"""Chen 2004: image conic + intrinsics + circle radius -> 2 candidate (center, normal).

Reference:
- Chen, Q., Wu, H., Wada, T. (2004) "Camera Calibration with Two Arbitrary Coplanar
  Circles", ECCV.
- Heikkilae, J. (1997) "Geometric Camera Calibration Using Circular Control Points",
  IEEE TPAMI.

Notation matches Forsyth/Mundy/Zisserman tradition: 3x3 symmetric Q in image
pixels (x^T Q x = 0). A = K^T Q K is the cone matrix in camera-ray space.
"""

from __future__ import annotations

import numpy as np


class PoseSolverError(ValueError):
    pass


def solve_circle_pose(Q_image: np.ndarray, K: np.ndarray, radius: float):
    """Recover two candidate (center, normal) pairs in camera coords.

    Returns: list of two tuples [(c1, n1), (c2, n2)] where c is (3,) and n is unit (3,).
    Both candidates are physically plausible (z > 0, normal facing camera adjusted).
    """
    if not (Q_image.shape == (3, 3) and K.shape == (3, 3)):
        raise PoseSolverError("Q and K must be 3x3")

    A = K.T @ Q_image @ K
    A = 0.5 * (A + A.T)  # numerical symmetry
    w, V = np.linalg.eigh(A)  # ascending
    # Sort to ensure (lambda1 >= lambda2 > 0 > lambda3) by absolute scaling.
    # Cone matrices have signature (++,-) up to overall sign; flip if needed.
    if (w > 0).sum() == 1 and (w < 0).sum() == 2:
        w = -w
        # eigenvectors stay the same (eigenvectors of -A are eigenvectors of A)
    if not ((w > 0).sum() == 2 and (w < 0).sum() == 1):
        raise PoseSolverError(f"Conic does not represent valid ellipse cone: eigvals={w}")

    # Index assignment: l3 = most negative; l1, l2 = positives with l1 >= l2
    idx_neg = int(np.argmin(w))
    pos_idx = [i for i in range(3) if i != idx_neg]
    if w[pos_idx[0]] >= w[pos_idx[1]]:
        idx_l1, idx_l2 = pos_idx[0], pos_idx[1]
    else:
        idx_l1, idx_l2 = pos_idx[1], pos_idx[0]
    l1, l2, l3 = w[idx_l1], w[idx_l2], w[idx_neg]
    v1 = V[:, idx_l1]
    v3 = V[:, idx_neg]

    # Canonicalize eigenvector signs so candidate ordering is stable under
    # small perturbations of A. np.linalg.eigh returns eigenvectors with
    # arbitrary signs; perturbations can flip them discontinuously and swap
    # the (n_a, c_a) vs (n_b, c_b) labels.
    def _pin_sign(v: np.ndarray) -> np.ndarray:
        k = int(np.argmax(np.abs(v)))
        return v if v[k] >= 0.0 else -v

    v1 = _pin_sign(v1)
    v3 = _pin_sign(v3)

    g = float(np.sqrt((l1 - l2) / (l1 - l3)))
    h = float(np.sqrt((l2 - l3) / (l1 - l3)))

    # Candidate normals in camera coords (Chen 2004 / Heikkila 1997):
    #   n = g * e1 +/- h * e3
    # where e1 is the eigenvector of the largest positive eigenvalue (l1)
    # and e3 is that of the negative eigenvalue (l3).
    n_a = g * v1 + h * v3
    n_b = g * v1 - h * v3

    # Center coordinates derived from substituting the parametrized circle into
    # the cone equation lambda_i * P_i^2 = 0 and zeroing the cos(theta) /
    # cos(2 theta) coefficients (radius r enforces overall scale):
    #   |X_c| = r * g * sqrt(-l3 / l1),  |Z_c| = r * h * sqrt(l1 / -l3)
    # The cos(theta) constraint pairs sign(X_c) with sign(Z_c) according to the
    # sign of the v3 component of the normal:
    #   for n_a (= g v1 + h v3): X_c and Z_c have OPPOSITE signs
    #   for n_b (= g v1 - h v3): X_c and Z_c have the SAME sign
    a_coeff = radius * g * np.sqrt(-l3 / l1)
    b_coeff = radius * h * np.sqrt(l1 / -l3)
    c_a = a_coeff * v1 - b_coeff * v3   # paired with n_a
    c_b = a_coeff * v1 + b_coeff * v3   # paired with n_b

    out = []
    for c, n in [(c_a, n_a), (c_b, n_b)]:
        n = n / np.linalg.norm(n)
        # Ensure z > 0 (target in front of camera). Center and normal flip
        # together so the (center, normal) pair stays geometrically consistent.
        if c[2] < 0:
            c = -c
            n = -n
        out.append((c, n))
    return out
