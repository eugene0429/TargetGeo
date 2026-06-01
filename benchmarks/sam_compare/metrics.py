"""Mask-comparison metrics for the SAM variant benchmark."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .paths import repo_root, ensure_seg_pose_importable

ensure_seg_pose_importable(repo_root())
from seg_pose.ellipse_core import fit_ellipse_to_mask  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. Returns 0.0 if union is empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def ellipse_summary(mask: np.ndarray) -> Dict:
    """Fit an ellipse to a boolean mask and summarize fit success + key params."""
    fit = fit_ellipse_to_mask(mask.astype(np.uint8))
    area = float(mask.astype(bool).sum())
    return {
        "ok": bool(fit.valid),
        "centroid": (float(fit.center_x), float(fit.center_y)),
        "area": area,
        "major": float(fit.major),
        "minor": float(fit.minor),
    }
