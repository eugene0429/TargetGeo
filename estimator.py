"""SegPoseEstimator — top-level orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .detector import TargetDetector, DEFAULT_DETECTOR_PATH
from .sam3 import Sam3DiskSegmenter
from .pose_types import (
    DroneStateUe, DroneStateGps, TargetGeoEstimate,
    PoseSigma, IntrinsicSigma, DEFAULT_POSE_SIGMA, DEFAULT_INTRINSIC_SIGMA,
)


DEFAULT_TEXT_PROMPTS = (
    "concentric circular target",
    "round archery target on white background",
    "black outer ring of archery target",
)


class SegPoseEstimator:
    """Single-frame target detection + disk segmentation + pose estimation."""

    def __init__(
        self,
        *,
        sam3_checkpoint: str | Path = "hf",
        detector_checkpoint: str | Path = DEFAULT_DETECTOR_PATH,
        target_radius_m: float = 2.5,
        device: str = "cuda",
        text_prompts: tuple[str, ...] = DEFAULT_TEXT_PROMPTS,
        crop_pad_ratio: float = 0.15,
        detector_conf_threshold: float = 0.25,
        min_mask_area_px: int = 200,
        min_minor_axis_px: float = 4.0,
        max_normal_cone_deg: float = 45.0,
        pixel_sigma_px: float = 0.5,
        pose_sigma: PoseSigma = DEFAULT_POSE_SIGMA,
        intrinsic_sigma: IntrinsicSigma = DEFAULT_INTRINSIC_SIGMA,
        segmenter: Optional[object] = None,   # for DI/testing
        detector: Optional[TargetDetector] = None,  # for DI/testing
    ):
        self.target_radius_m = float(target_radius_m)
        self.text_prompts = tuple(text_prompts)
        self.crop_pad_ratio = float(crop_pad_ratio)
        self.min_mask_area_px = int(min_mask_area_px)
        self.min_minor_axis_px = float(min_minor_axis_px)
        self.max_normal_cone_deg = float(max_normal_cone_deg)
        self.pixel_sigma_px = float(pixel_sigma_px)
        self.pose_sigma = pose_sigma
        self.intrinsic_sigma = intrinsic_sigma
        if segmenter is not None:
            self.segmenter = segmenter
        else:
            self.segmenter = Sam3DiskSegmenter(checkpoint=sam3_checkpoint, device=device)
        if detector is not None:
            self.detector = detector
        else:
            self.detector = TargetDetector(
                checkpoint=detector_checkpoint,
                conf_threshold=detector_conf_threshold,
                device=device,
            )

    def estimate(
        self,
        rgb: np.ndarray,
        rec_bbox: tuple[float, float, float, float],
        drone_state,
    ) -> TargetGeoEstimate:
        # Step 1: Crop to bbox with padding
        from .sam3 import crop_to_bbox, pad_mask_to_full
        H, W = rgb.shape[:2]
        try:
            crop, crop_xyxy = crop_to_bbox(rgb, rec_bbox, pad_ratio=self.crop_pad_ratio)
        except Exception:
            return _empty_failure("bbox_too_small")
        if crop.size == 0 or min(crop.shape[:2]) < 16:
            return _empty_failure("bbox_too_small")

        # Step 2: SAM 3.1 on crop
        crop_mask, score, winner = self.segmenter.segment(crop, self.text_prompts)
        if crop_mask is None:
            return _empty_failure("no_mask")
        area = int(crop_mask.sum())
        if area < self.min_mask_area_px:
            r = _empty_failure("no_mask")
            return TargetGeoEstimate(**{**r.__dict__, "disk_mask_area_px": area, "sam3_score": float(score)})

        # Step 3: Pad to full-image mask
        full_mask = pad_mask_to_full(crop_mask, crop_xyxy, (H, W))

        # Step 4: Hull-based ellipse fit
        from .ellipse_core import fit_ellipse_hull
        fit, fit_method = fit_ellipse_hull(
            full_mask,
            min_minor_axis_px=self.min_minor_axis_px,
        )
        if fit is None:
            r = _empty_failure("fit_failed")
            return TargetGeoEstimate(**{
                **r.__dict__,
                "disk_mask_area_px": int(full_mask.sum()),
                "sam3_score": float(score),
                "fit_method": fit_method,
            })

        ellipse_dict = {
            "cx": fit.center_x, "cy": fit.center_y,
            "major": fit.major, "minor": fit.minor,
            "theta": fit.angle_deg,
        }

        # Step 5: Ellipse → conic Q
        from .ellipse_core import ellipse_params_to_conic
        Q = ellipse_params_to_conic(fit)

        # Step 6: Chen 2004 — 2 (center, normal) candidates in camera frame
        from .pose_solver import solve_circle_pose, PoseSolverError
        try:
            candidates = solve_circle_pose(Q, drone_state.K, self.target_radius_m)
        except PoseSolverError:
            r = _empty_failure("pose_failed")
            return TargetGeoEstimate(**{
                **r.__dict__,
                "ellipse": ellipse_dict,
                "disk_mask_area_px": int(full_mask.sum()),
                "sam3_score": float(score),
                "fit_method": fit_method,
            })

        # Step 7: Disambiguation — compute world_up in camera frame per input mode
        from .disambiguate import disambiguate_visibility
        from .transforms import world_up_in_cam_cv, world_up_in_cam_enu
        if isinstance(drone_state, DroneStateUe):
            world_up_cv = world_up_in_cam_cv(drone_state.camera_pyr_deg)
        elif isinstance(drone_state, DroneStateGps):
            world_up_cv = world_up_in_cam_enu(drone_state.camera_pyr_deg)
        else:
            r = _empty_failure("unknown_state_type")
            return TargetGeoEstimate(**{**r.__dict__, "ellipse": ellipse_dict})

        dr = disambiguate_visibility(candidates, world_up_cv=world_up_cv)
        c_cam = np.asarray(dr.center, dtype=float)
        n_cam = np.asarray(dr.normal, dtype=float)
        offset_camera_m = tuple(float(v) for v in c_cam)
        range_m = float(np.linalg.norm(c_cam))

        # Step 8-9: Camera frame → world frame absolute position.
        # UE path: target_xyz_ue = camera_xyz_ue + R_cam_to_ue @ offset
        # GPS path: ENU offset → enu2geodetic(camera_gps)
        from .transforms import M_UE2CV, ue_rotation_matrix
        target_xyz_ue_m: tuple[float, float, float] | None = None
        target_lat: float | None = None
        target_lon: float | None = None
        target_alt_m: float | None = None
        normal_world: tuple[float, float, float] | None = None

        if isinstance(drone_state, DroneStateUe):
            R_cam_to_world = ue_rotation_matrix(drone_state.camera_pyr_deg)
            # offset_world (UE coords, meters): rotate camera-frame offset using
            # the camera→UE-world rotation. Same convention as M_UE2CV.T applied
            # after ue_rotation_matrix.
            offset_ue = R_cam_to_world @ (M_UE2CV.T @ c_cam)
            tx, ty, tz = drone_state.camera_xyz_ue_m
            target_xyz_ue_m = (
                float(tx + offset_ue[0]),
                float(ty + offset_ue[1]),
                float(tz + offset_ue[2]),
            )
            n_ue = R_cam_to_world @ (M_UE2CV.T @ n_cam)
            n_ue = n_ue / np.linalg.norm(n_ue)
            normal_world = tuple(float(v) for v in n_ue)
        elif isinstance(drone_state, DroneStateGps):
            from .transforms import enu_rotation_matrix
            # ENU rotation: cam-frame offset → ENU world-frame offset
            R_cam_to_enu = enu_rotation_matrix(drone_state.camera_pyr_deg)
            offset_enu = R_cam_to_enu @ c_cam
            east, north, up = float(offset_enu[0]), float(offset_enu[1]), float(offset_enu[2])
            from .gps import offset_to_target_gps
            target_lat, target_lon, target_alt_m = offset_to_target_gps(
                east_m=east, north_m=north, up_m=up,
                cam_lat=drone_state.camera_lat,
                cam_lon=drone_state.camera_lon,
                cam_alt_m=drone_state.camera_alt_m,
            )
            n_enu = R_cam_to_enu @ n_cam
            n_enu = n_enu / np.linalg.norm(n_enu)
            normal_world = tuple(float(v) for v in n_enu)

        # Step 10: Covariance propagation via numerical Jacobian.
        # Note: covariance.py is hardcoded to UE convention. For GPS path
        # we construct a DronePose with the same pyr_deg (treated as UE
        # convention for the purpose of Jacobian shape). Covariance magnitude
        # is rotation-frame invariant; axes interpretation matches the input
        # mode.
        from .covariance import compute_position_covariance

        # Adapter — supply a DronePose-shaped object.
        class _DronePoseAdapter:
            def __init__(self, loc, pyr):
                self.loc_xyz_ue = loc
                self.pyr_deg = pyr

        if isinstance(drone_state, DroneStateUe):
            dp = _DronePoseAdapter(drone_state.camera_xyz_ue_m, drone_state.camera_pyr_deg)
        else:
            # GPS path: use zero origin (translation doesn't affect covariance shape)
            dp = _DronePoseAdapter((0.0, 0.0, 0.0), drone_state.camera_pyr_deg)

        try:
            pos_cov, cone_deg = compute_position_covariance(
                ellipse_params=ellipse_dict,
                K=drone_state.K,
                radius=self.target_radius_m,
                drone_pose=dp,
                pixel_sigma_px=self.pixel_sigma_px,
                pose_sigma=self.pose_sigma,
                intrinsic_sigma=self.intrinsic_sigma,
                chosen_idx=dr.chosen_idx,
            )
        except Exception:
            pos_cov, cone_deg = None, 0.0

        flags: list[str] = []
        if cone_deg > self.max_normal_cone_deg:
            flags.append("high_normal_cone")

        return TargetGeoEstimate(
            target_xyz_ue_m=target_xyz_ue_m,
            target_lat=target_lat,
            target_lon=target_lon,
            target_alt_m=target_alt_m,
            offset_camera_m=offset_camera_m,
            range_m=range_m,
            normal_camera=tuple(float(v) for v in n_cam),
            normal_world=normal_world,
            pos_cov_3x3=pos_cov,
            normal_cone_deg=float(cone_deg),
            ellipse=ellipse_dict,
            disk_mask_area_px=int(full_mask.sum()),
            sam3_score=float(score),
            fit_method=fit_method,
            disambiguation_method=dr.method,
            valid=True,
            status="ok",
            flags=flags,
        )

    def estimate_from_image(
        self,
        rgb: np.ndarray,
        drone_state,
    ) -> TargetGeoEstimate:
        """Primary entry point: target detection → segment → fit → pose.

        Returns valid=False, status='no_detection' if the detector finds nothing.
        """
        if self.detector is None:
            return _empty_failure("no_detector")
        bbox = self.detector.detect(rgb)
        if bbox is None:
            return _empty_failure("no_detection")
        return self.estimate(rgb, bbox, drone_state)


def _empty_failure(status: str, flags: list[str] | None = None) -> TargetGeoEstimate:
    return TargetGeoEstimate(
        target_xyz_ue_m=None, target_lat=None, target_lon=None, target_alt_m=None,
        offset_camera_m=None, range_m=None,
        normal_camera=None, normal_world=None,
        pos_cov_3x3=None, normal_cone_deg=0.0,
        ellipse=None, disk_mask_area_px=0, sam3_score=0.0,
        fit_method="(failed)", disambiguation_method="(none)",
        valid=False, status=status,
        flags=list(flags or []),
    )
