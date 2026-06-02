"""Public dataclasses for targetgeo: DroneStateUe, DroneStateGps, TargetGeoEstimate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PoseSigma:
    pos_m: float = 1.0
    att_deg: float = 0.5


@dataclass(frozen=True)
class IntrinsicSigma:
    fx_px: float = 1.0
    cxy_px: float = 0.5


DEFAULT_POSE_SIGMA = PoseSigma()
DEFAULT_INTRINSIC_SIGMA = IntrinsicSigma()


@dataclass(frozen=True)
class DroneStateUe:
    """UE synthetic test path."""
    camera_xyz_ue_m: tuple[float, float, float]
    camera_pyr_deg: tuple[float, float, float]
    K: np.ndarray
    frame_id: int = -1
    timestamp_s: float = 0.0


@dataclass(frozen=True)
class DroneStateGps:
    """Real flight test path."""
    camera_lat: float
    camera_lon: float
    camera_alt_m: float
    camera_pyr_deg: tuple[float, float, float]
    K: np.ndarray
    frame_id: int = -1
    timestamp_s: float = 0.0


# Type alias — caller picks one.
DroneState = "DroneStateUe | DroneStateGps"


@dataclass(frozen=True)
class TargetGeoEstimate:
    """Output of TargetGeoEstimator.estimate(). Frame-agnostic + per-mode fields."""
    # Absolute position — exactly one of these is populated based on input mode
    target_xyz_ue_m: tuple[float, float, float] | None
    target_lat: float | None
    target_lon: float | None
    target_alt_m: float | None

    # Frame-agnostic
    offset_camera_m: tuple[float, float, float] | None
    range_m: float | None
    normal_camera: tuple[float, float, float] | None
    normal_world: tuple[float, float, float] | None

    # Uncertainty
    pos_cov_3x3: np.ndarray | None
    normal_cone_deg: float

    # Diagnostics
    ellipse: dict | None
    disk_mask_area_px: int
    sam3_score: float
    fit_method: str
    disambiguation_method: str

    # Status
    valid: bool
    status: str
    flags: list[str] = field(default_factory=list)
