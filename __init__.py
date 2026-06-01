"""SAM 3.1 + hull-based disk segmentation + pose estimation.

Self-contained module — portable to other machines via directory copy.
"""

from .estimator import SegPoseEstimator
from .detector import TargetDetector, DEFAULT_DETECTOR_PATH
from .pose_types import (
    DroneStateUe,
    DroneStateGps,
    TargetGeoEstimate,
    PoseSigma,
    IntrinsicSigma,
    DEFAULT_POSE_SIGMA,
    DEFAULT_INTRINSIC_SIGMA,
)

__all__ = [
    "SegPoseEstimator",
    "TargetDetector",
    "DEFAULT_DETECTOR_PATH",
    "DroneStateUe",
    "DroneStateGps",
    "TargetGeoEstimate",
    "PoseSigma",
    "IntrinsicSigma",
    "DEFAULT_POSE_SIGMA",
    "DEFAULT_INTRINSIC_SIGMA",
]
