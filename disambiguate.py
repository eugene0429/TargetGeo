"""2-fold ambiguity disambiguation for Chen2004 outputs.

v1 strategies:
- world-up axis prior (when caller can supply world_up_cv): pick the candidate
  whose plane is more "axis-aligned" with world up (i.e., normal is either
  parallel OR perpendicular to world up — flat OR vertical disk geometry). Uses
  score = max(|n·u|, 1 - |n·u|), highest at the extremes (0 or 1). This works
  for BOTH flat targets (true normal = world up, |n·u|≈1) AND vertical targets
  (true normal ⊥ world up, |n·u|≈0) without needing target-type metadata. The
  WRONG Chen-2004 candidate is the "tilted-the-other-way" mirror solution
  whose plane is oblique — score < 1.
- visibility (fallback when world_up not provided): pick the candidate with
  most negative n_z. Fragile when two candidates have similar n_z; flagged
  'fallback'.

v2 (deferred): multi-target coplanarity, temporal smoothness, multi-view.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DisambiguationResult:
    center: np.ndarray
    normal: np.ndarray
    chosen_idx: int      # 0 or 1
    method: str          # 'world_up_axis' | 'visibility' | 'fallback'


def disambiguate_visibility(
    candidates,
    close_threshold: float = 0.05,
    world_up_cv: np.ndarray | None = None,
) -> DisambiguationResult:
    """Pick the more physically plausible of two Chen-2004 candidates.

    Args:
        candidates: list of two (center, normal) tuples in OpenCV cam coords.
        close_threshold: gap below which the choice is unreliable; labeled
            'fallback'. Compared against the relevant scoring metric.
        world_up_cv: optional unit vector — world-up in OpenCV cam coords
            (e.g., world_up_in_cam_cv(cam_rot)). When provided, uses the
            world-up axis prior (works for both flat and vertical targets).
            When omitted, falls back to the visibility-via-n_z rule.
    """
    if len(candidates) != 2:
        raise ValueError(f"expected 2 candidates, got {len(candidates)}")

    if world_up_cv is not None:
        u = np.asarray(world_up_cv, dtype=float)
        u = u / np.linalg.norm(u)
        # Score: prefer candidates whose plane is axis-aligned with world up
        # (normal either parallel or perpendicular to up). For flat targets
        # the true normal is parallel to up (|n.u|≈1); for vertical targets
        # it's perpendicular (|n.u|≈0). The wrong Chen-2004 mirror has its
        # plane tilted obliquely — score strictly less than 1.
        a0 = abs(float(np.dot(candidates[0][1], u)))
        a1 = abs(float(np.dot(candidates[1][1], u)))
        s0 = max(a0, 1.0 - a0)
        s1 = max(a1, 1.0 - a1)
        if s0 >= s1:
            idx = 0
        else:
            idx = 1
        method = "world_up_axis"
        if abs(s0 - s1) < close_threshold:
            method = "fallback"
        c, n = candidates[idx]
        return DisambiguationResult(center=c, normal=n, chosen_idx=idx, method=method)

    # Fallback: visibility constraint via n_z only.
    nz0 = candidates[0][1][2]
    nz1 = candidates[1][1][2]
    if nz0 < nz1:
        idx = 0
    else:
        idx = 1
    method = "visibility"
    if nz0 >= 0 and nz1 >= 0:
        method = "fallback"
    elif abs(nz0 - nz1) < close_threshold:
        method = "fallback"
    c, n = candidates[idx]
    return DisambiguationResult(center=c, normal=n, chosen_idx=idx, method=method)
