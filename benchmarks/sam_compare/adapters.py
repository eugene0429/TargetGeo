"""Uniform box-prompted segmenters for the SAM variant benchmark.

Every adapter exposes:
    name: str
    available: bool
    segment(rgb_bgr, bbox_xyxy) -> Optional[np.ndarray]   # full-image bool mask HxW

Construction performs the (heavy) model load. If deps/weights are missing the
adapter sets available=False and segment() returns None; the runner skips it.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


class BoxSegmenter:
    name: str = "base"

    def __init__(self) -> None:
        self.available: bool = False

    def segment(self, rgb_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        raise NotImplementedError


def _ultra_mask_to_full(result, h: int, w: int) -> Optional[np.ndarray]:
    """Build a full-image bool mask from an ultralytics Results object via polygons.

    Polygons (`masks.xy`) are already in original-image pixel coords, which avoids
    letterbox-resolution mismatches in `masks.data`.
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
    name = "sam3.1"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            model = build_sam3_image_model(device=device)
            self._processor = Sam3Processor(model, device=device)
            import torch
            self._torch = torch
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[sam3.1] unavailable: {e}")
            self.available = False

    def segment(self, rgb_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = (int(v) for v in bbox)
        torch = self._torch
        with torch.autocast(self.device if self.device == "cuda" else "cpu",
                            dtype=torch.bfloat16):
            state = self._processor.set_image(pil)
            out = self._processor.add_geometric_prompt(
                box=[x1, y1, x2, y2], label=True, state=state,
            )
            masks = out.get("masks")
            scores = out.get("scores")
            self._processor.reset_all_prompts(state)
        if masks is None or len(masks) == 0:
            return None
        s_arr = self._to_numpy(scores)
        m_arr = self._to_numpy(masks)
        i = int(np.argmax(s_arr))
        m = m_arr[i]
        if m.ndim == 3:
            m = m[0]
        return m.astype(bool)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
        except ImportError:
            pass
        return np.asarray(x)
