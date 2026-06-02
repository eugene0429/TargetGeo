"""Crop-based disk segmenters for the SAM variant benchmark.

Mirrors the production pipeline: the YOLO detector bbox is cropped (with padding)
and each model segments the disk *inside that crop*.

Every adapter exposes:
    name: str
    available: bool
    segment(crop_bgr, box_in_crop) -> Optional[np.ndarray]   # bool mask, crop HxW

Prompting (the YOLO box covers the WHOLE target, so a box prompt tends to grab
the whole target rather than the disk; a center POINT lands on the disk):
- sam3.1        : production TEXT prompts (ignores box).
- fastsam-text  : same TEXT prompts via CLIP (ignores box).
- fastsam-point : POINT prompt at the box center.
- mobilesam-point / edgesam-point : POINT prompt at the box center (no text encoder).

Construction performs the (heavy) model load. If deps/weights are missing the
adapter sets available=False and segment() returns None; the runner skips it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def _box_center(box: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = (int(v) for v in box)
    return (x1 + x2) // 2, (y1 + y2) // 2


class BoxSegmenter:
    name: str = "base"

    def __init__(self) -> None:
        self.available: bool = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        raise NotImplementedError


def _ultra_best_mask(result, h: int, w: int) -> Optional[np.ndarray]:
    """Single highest-confidence mask from an ultralytics Results object.

    Picks ONE object (top conf) rather than unioning all polygons — for point/box
    prompts that avoids merging the disk with the surrounding target. Polygons
    (`masks.xy`) are already in input-image (crop) pixel coords.
    """
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None or len(masks.xy) == 0:
        return None
    boxes = getattr(result, "boxes", None)
    if (boxes is not None and boxes.conf is not None
            and len(boxes.conf) == len(masks.xy)):
        i = int(boxes.conf.argmax())
    else:
        i = 0
    poly = masks.xy[i]
    if poly is None or len(poly) < 3:
        return None
    full = np.zeros((h, w), dtype=np.uint8)
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
    """FastSAM on the crop, prompted by text and/or a center point.

    prompt_mode:
      "text"        -> name "fastsam-text"       : text only (same input as sam3.1)
      "point"       -> name "fastsam-point"      : center point only
      "point+text"  -> name "fastsam-point+text" : center point AND text in one call
    """

    def __init__(self, device: str = "cuda", weights: str = "FastSAM-s.pt",
                 prompt_mode: str = "text") -> None:
        super().__init__()
        self.device = device
        self.prompt_mode = prompt_mode
        self.name = f"fastsam-{prompt_mode}"
        try:
            from ultralytics import FastSAM
            self._model = FastSAM(weights)
            if "text" in prompt_mode:
                from seg_pose.estimator import DEFAULT_TEXT_PROMPTS
                self._prompts = tuple(DEFAULT_TEXT_PROMPTS)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[{self.name}] unavailable: {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = crop_bgr.shape[:2]
        if self.prompt_mode == "point":
            px, py = _box_center(box)
            res = self._model(
                crop_bgr, points=[[px, py]], labels=[1],
                device=self.device, verbose=False,
            )
            return _ultra_best_mask(res[0], h, w) if res else None
        # text or point+text: try each production prompt (optionally with the
        # center point), keep the highest-CLIP-conf mask.
        pt_kwargs = {}
        if self.prompt_mode == "point+text":
            px, py = _box_center(box)
            pt_kwargs = {"points": [[px, py]], "labels": [1]}
        best_mask, best_conf = None, -1.0
        for text in self._prompts:
            res = self._model(
                crop_bgr, texts=text, device=self.device, verbose=False, **pt_kwargs,
            )
            if not res:
                continue
            r = res[0]
            if r.masks is None or r.boxes is None or len(r.boxes) == 0:
                continue
            conf = float(r.boxes.conf.max().item())
            if conf > best_conf:
                m = _ultra_best_mask(r, h, w)
                if m is not None:
                    best_mask, best_conf = m, conf
        return best_mask


class MobileSamAdapter(BoxSegmenter):
    """MobileSAM with a POINT prompt at the box center (no text encoder)."""

    name = "mobilesam-point"

    def __init__(self, device: str = "cuda", weights: str = "mobile_sam.pt") -> None:
        super().__init__()
        self.device = device
        try:
            from ultralytics import SAM
            self._model = SAM(weights)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[mobilesam-point] unavailable: {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = crop_bgr.shape[:2]
        px, py = _box_center(box)
        res = self._model(
            crop_bgr, points=[[px, py]], labels=[1],
            device=self.device, verbose=False,
        )
        return _ultra_best_mask(res[0], h, w) if res else None


class EdgeSamAdapter(BoxSegmenter):
    """EdgeSAM (chongzhou96/EdgeSAM `edge_sam` package) with a POINT prompt.

    Set EDGE_SAM_CHECKPOINT to the weight path (default ./weights/edge_sam_3x.pth).
    """

    name = "edgesam-point"

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
            print(f"[edgesam-point] unavailable (skipping): {e}")
            self.available = False

    def segment(self, crop_bgr: np.ndarray, box: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        px, py = _box_center(box)
        self._predictor.set_image(rgb)
        masks, scores, _ = self._predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            return None
        i = int(np.argmax(scores))
        m = masks[i]
        if m.ndim == 3:
            m = m[0]
        return m.astype(bool)
