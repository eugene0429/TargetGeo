"""Crop-based disk segmenters for the SAM variant benchmark.

Mirrors the production pipeline: the YOLO detector bbox is cropped (with padding)
and each model segments the disk *inside that crop*.

Every adapter exposes:
    name: str
    available: bool
    segment(crop_bgr, box_in_crop) -> Optional[np.ndarray]   # bool mask, crop HxW

- SAM3.1 and FastSAM use TEXT prompts (ignore box_in_crop).
- MobileSAM / EdgeSAM use box_in_crop as a box prompt (no text encoder).

Construction performs the (heavy) model load. If deps/weights are missing the
adapter sets available=False and segment() returns None; the runner skips it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


class BoxSegmenter:
    name: str = "base"

    def __init__(self) -> None:
        self.available: bool = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        raise NotImplementedError


def _ultra_mask_to_full(result, h: int, w: int) -> Optional[np.ndarray]:
    """Build a crop-sized bool mask from an ultralytics Results object via polygons.

    Polygons (`masks.xy`) are already in input-image (crop) pixel coords, which
    avoids letterbox-resolution mismatches in `masks.data`.
    """
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None or len(masks.xy) == 0:
        return None
    full = np.zeros((h, w), dtype=np.uint8)
    for poly in masks.xy:
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(full, [poly.astype(np.int32)], 1)
    if full.sum() == 0:
        return None
    return full.astype(bool)


class Sam31Adapter(BoxSegmenter):
    """Production SAM3.1 path: text prompts on the crop. Box is ignored.

    Wraps seg_pose.sam3.Sam3DiskSegmenter for exact production parity.
    """

    name = "sam3.1"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        try:
            from seg_pose.sam3 import Sam3DiskSegmenter
            from seg_pose.estimator import DEFAULT_TEXT_PROMPTS
            self._seg = Sam3DiskSegmenter(device=device)
            self._prompts = tuple(DEFAULT_TEXT_PROMPTS)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[sam3.1] unavailable: {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        mask, _score, _winner = self._seg.segment(crop_bgr, self._prompts)
        if mask is None:
            return None
        return mask.astype(bool)


class FastSamAdapter(BoxSegmenter):
    """FastSAM text path (CLIP): segments everything in the crop, then CLIP picks
    the mask matching each text prompt. Uses the same production prompts as SAM3.1
    and keeps the highest-similarity mask. Ignores box.
    """

    name = "fastsam"

    def __init__(self, device: str = "cuda", weights: str = "FastSAM-s.pt") -> None:
        super().__init__()
        self.device = device
        try:
            from ultralytics import FastSAM
            from seg_pose.estimator import DEFAULT_TEXT_PROMPTS
            self._model = FastSAM(weights)
            self._prompts = tuple(DEFAULT_TEXT_PROMPTS)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[fastsam] unavailable: {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = crop_bgr.shape[:2]
        best_mask, best_conf = None, -1.0
        for text in self._prompts:
            res = self._model(
                crop_bgr, texts=text, device=self.device, verbose=False,
            )
            if not res:
                continue
            r = res[0]
            if r.masks is None or r.boxes is None or len(r.boxes) == 0:
                continue
            conf = float(r.boxes.conf.max().item())
            if conf > best_conf:
                m = _ultra_mask_to_full(r, h, w)
                if m is not None:
                    best_mask, best_conf = m, conf
        return best_mask


class MobileSamAdapter(BoxSegmenter):
    name = "mobilesam"

    def __init__(self, device: str = "cuda", weights: str = "mobile_sam.pt") -> None:
        super().__init__()
        self.device = device
        try:
            from ultralytics import SAM
            self._model = SAM(weights)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[mobilesam] unavailable: {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = crop_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        res = self._model(
            crop_bgr, bboxes=[[x1, y1, x2, y2]], device=self.device, verbose=False,
        )
        if not res:
            return None
        return _ultra_mask_to_full(res[0], h, w)


class EdgeSamAdapter(BoxSegmenter):
    """EdgeSAM via the chongzhou96/EdgeSAM `edge_sam` package + SamPredictor API.

    Set EDGE_SAM_CHECKPOINT to the weight path (default ./weights/edge_sam_3x.pth).
    """

    name = "edgesam"

    def __init__(self, device: str = "cuda",
                 checkpoint: Optional[str] = None,
                 model_type: str = "edge_sam") -> None:
        super().__init__()
        self.device = device
        ckpt = checkpoint or os.environ.get(
            "EDGE_SAM_CHECKPOINT", "weights/edge_sam_3x.pth")
        try:
            from edge_sam import sam_model_registry, SamPredictor
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"EdgeSAM checkpoint not found: {ckpt}")
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(device=device)
            self._predictor = SamPredictor(sam)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[edgesam] unavailable (skipping): {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (int(v) for v in box)
        self._predictor.set_image(rgb)
        masks, scores, _ = self._predictor.predict(
            box=np.array([x1, y1, x2, y2]), multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None
        i = int(np.argmax(scores))
        m = masks[i]
        if m.ndim == 3:
            m = m[0]
        return m.astype(bool)
