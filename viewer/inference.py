"""FrameAnalyzer: run the seg_pose pipeline on one frame -> FrameResult.

Reuses the same building blocks as SegPoseEstimator but (a) keeps the SAM mask
so the viewer can draw it, and (b) supports a no-telemetry path that yields
camera-frame normal/range/cone with visibility-based disambiguation. When a
DroneState is supplied, world-frame geodetic fields are filled in too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from seg_pose.sam3 import crop_to_bbox, pad_mask_to_full
from seg_pose.ellipse_core import fit_ellipse_hull, ellipse_params_to_conic
from seg_pose.pose_solver import solve_circle_pose, PoseSolverError
from seg_pose.disambiguate import disambiguate_visibility
from seg_pose.covariance import compute_position_covariance
from seg_pose.pose_types import (
    DroneStateUe, DroneStateGps, DEFAULT_POSE_SIGMA, DEFAULT_INTRINSIC_SIGMA,
)
from seg_pose.transforms import (
    M_UE2CV, ue_rotation_matrix, enu_rotation_matrix,
    world_up_in_cam_cv, world_up_in_cam_enu,
)
from seg_pose.gps import offset_to_target_gps

DEFAULT_TEXT_PROMPTS = (
    "concentric circular target",
    "round archery target on white background",
    "black outer ring of archery target",
)


@dataclass
class FrameResult:
    status: str
    valid: bool = False
    bbox: tuple | None = None
    mask_contour: np.ndarray | None = None
    ellipse: dict | None = None
    candidates: list = field(default_factory=list)
    chosen_idx: int | None = None
    normal_camera: tuple | None = None
    offset_camera_m: tuple | None = None
    range_m: float | None = None
    disambiguation_method: str = "-"
    sam_score: float = 0.0
    fit_method: str = "-"
    cone_deg: float | None = None
    flags: list = field(default_factory=list)
    # world-frame (telemetry only)
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    normal_world: tuple | None = None


class _DronePose:
    """Minimal DronePose shape for covariance.compute_position_covariance."""
    def __init__(self, loc_xyz_ue, pyr_deg):
        self.loc_xyz_ue = loc_xyz_ue
        self.pyr_deg = pyr_deg


def _largest_contour(mask_bool: np.ndarray):
    m8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class FrameAnalyzer:
    """Holds detector + segmenter; analyzes single frames."""

    def __init__(
        self,
        *,
        detector,
        segmenter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
        crop_pad_ratio=0.15,
        min_mask_area_px=200,
        min_minor_axis_px=4.0,
        max_normal_cone_deg=45.0,
        pixel_sigma_px=0.5,
    ):
        self._detector = detector
        self._segmenter = segmenter
        self._text_prompts = tuple(text_prompts)
        self.crop_pad_ratio = float(crop_pad_ratio)
        self.min_mask_area_px = int(min_mask_area_px)
        self.min_minor_axis_px = float(min_minor_axis_px)
        self.max_normal_cone_deg = float(max_normal_cone_deg)
        self.pixel_sigma_px = float(pixel_sigma_px)

    def analyze(self, frame, K, radius, n_prompts=3, telemetry=None,
                need_sam=True) -> FrameResult:
        H, W = frame.shape[:2]
        bbox = self._detector.detect(frame)
        if bbox is None:
            return FrameResult(status="no_detection")

        # Fast path: only the detector bbox is needed (mask/ellipse/normal/HUD
        # off). Skip the heavy SAM + ellipse + pose stages entirely (~19 ms vs
        # ~0.5 s/frame), so bbox-only playback runs at full speed.
        if not need_sam:
            return FrameResult(status="ok", valid=True, bbox=tuple(bbox))

        try:
            crop, crop_xyxy = crop_to_bbox(frame, bbox, pad_ratio=self.crop_pad_ratio)
        except Exception:
            return FrameResult(status="bbox_too_small", bbox=tuple(bbox))
        if crop.size == 0 or min(crop.shape[:2]) < 16:
            return FrameResult(status="bbox_too_small", bbox=tuple(bbox))

        prompts = self._text_prompts[:max(1, int(n_prompts))]
        crop_mask, score, _ = self._segmenter.segment(crop, prompts)
        if crop_mask is None:
            return FrameResult(status="no_mask", bbox=tuple(bbox))

        full_mask = pad_mask_to_full(crop_mask, crop_xyxy, (H, W))
        if int(full_mask.sum()) < self.min_mask_area_px:
            return FrameResult(status="no_mask", bbox=tuple(bbox), sam_score=float(score))
        contour = _largest_contour(full_mask)

        fit, fit_method = fit_ellipse_hull(full_mask, min_minor_axis_px=self.min_minor_axis_px)
        if fit is None:
            return FrameResult(status="fit_failed", bbox=tuple(bbox),
                               mask_contour=contour, sam_score=float(score),
                               fit_method=fit_method)

        ellipse = {"cx": fit.center_x, "cy": fit.center_y,
                   "major": fit.major, "minor": fit.minor, "theta": fit.angle_deg}
        Q = ellipse_params_to_conic(fit)
        try:
            candidates = solve_circle_pose(Q, K, float(radius))
        except PoseSolverError:
            return FrameResult(status="pose_failed", bbox=tuple(bbox),
                               mask_contour=contour, ellipse=ellipse,
                               sam_score=float(score), fit_method=fit_method)

        # world-up prior only when telemetry is available
        world_up = None
        if isinstance(telemetry, DroneStateUe):
            world_up = world_up_in_cam_cv(telemetry.camera_pyr_deg)
        elif isinstance(telemetry, DroneStateGps):
            world_up = world_up_in_cam_enu(telemetry.camera_pyr_deg)

        dr = disambiguate_visibility(candidates, world_up_cv=world_up)
        c_cam = np.asarray(dr.center, dtype=float)
        n_cam = np.asarray(dr.normal, dtype=float)

        # uncertainty cone — magnitude is rotation-frame invariant, so a dummy
        # zero pose is fine when no telemetry is present.
        pyr = telemetry.camera_pyr_deg if telemetry is not None else (0.0, 0.0, 0.0)
        try:
            _, cone_deg = compute_position_covariance(
                ellipse_params=ellipse, K=K, radius=float(radius),
                drone_pose=_DronePose((0.0, 0.0, 0.0), pyr),
                pixel_sigma_px=self.pixel_sigma_px,
                pose_sigma=DEFAULT_POSE_SIGMA, intrinsic_sigma=DEFAULT_INTRINSIC_SIGMA,
                chosen_idx=dr.chosen_idx,
            )
            cone_deg = float(cone_deg)
        except Exception:
            cone_deg = None

        flags = []
        if cone_deg is not None and cone_deg > self.max_normal_cone_deg:
            flags.append("high_normal_cone")
        if dr.method == "fallback":
            flags.append("disambig_fallback")

        result = FrameResult(
            status="ok", valid=True, bbox=tuple(bbox), mask_contour=contour,
            ellipse=ellipse,
            candidates=[(np.asarray(c, float), np.asarray(n, float)) for c, n in candidates],
            chosen_idx=int(dr.chosen_idx),
            normal_camera=tuple(float(v) for v in n_cam),
            offset_camera_m=tuple(float(v) for v in c_cam),
            range_m=float(np.linalg.norm(c_cam)),
            disambiguation_method=dr.method, sam_score=float(score),
            fit_method=fit_method, cone_deg=cone_deg, flags=flags,
        )

        if telemetry is not None:
            self._fill_world(result, c_cam, n_cam, telemetry)
        return result

    @staticmethod
    def _fill_world(result, c_cam, n_cam, telemetry):
        if isinstance(telemetry, DroneStateUe):
            R = ue_rotation_matrix(telemetry.camera_pyr_deg)
            offset_ue = R @ (M_UE2CV.T @ c_cam)
            tx, ty, tz = telemetry.camera_xyz_ue_m
            n_ue = R @ (M_UE2CV.T @ n_cam)
            n_ue = n_ue / np.linalg.norm(n_ue)
            result.normal_world = tuple(float(v) for v in n_ue)
            # UE absolute position isn't geodetic; expose via normal_world + range only.
        elif isinstance(telemetry, DroneStateGps):
            R = enu_rotation_matrix(telemetry.camera_pyr_deg)
            offset_enu = R @ c_cam
            lat, lon, alt = offset_to_target_gps(
                east_m=float(offset_enu[0]), north_m=float(offset_enu[1]),
                up_m=float(offset_enu[2]),
                cam_lat=telemetry.camera_lat, cam_lon=telemetry.camera_lon,
                cam_alt_m=telemetry.camera_alt_m,
            )
            result.lat, result.lon, result.alt_m = lat, lon, alt
            n_enu = R @ n_cam
            n_enu = n_enu / np.linalg.norm(n_enu)
            result.normal_world = tuple(float(v) for v in n_enu)
