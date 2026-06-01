"""SAM 3.1 disk segmenter — crop image to M1 bbox, run text prompts, return mask.

Heavy: SAM 3.1 model loads ~3 GB to GPU on construction. Hold a single
Sam3DiskSegmenter instance for the lifetime of the application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch


def crop_to_bbox(
    img: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.15,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop with padding. Returns (crop, (cx1, cy1, cx2, cy2)) in full-image coords."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    cx1 = max(0, int(x1) - pad)
    cy1 = max(0, int(y1) - pad)
    cx2 = min(W, int(x2) + pad)
    cy2 = min(H, int(y2) + pad)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def pad_mask_to_full(
    crop_mask: np.ndarray,
    crop_xyxy: Tuple[int, int, int, int],
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """Pad a crop-coord mask back to full-image coords."""
    H, W = full_hw
    full = np.zeros((H, W), dtype=bool)
    cx1, cy1, cx2, cy2 = crop_xyxy
    full[cy1:cy2, cx1:cx2] = crop_mask.astype(bool)
    return full


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class Sam3DiskSegmenter:
    """SAM 3.1 wrapper with crop+text disk segmentation.

    Construction loads the model. Hold ONE instance for the app lifetime.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path = "hf",
        device: str = "cuda",
    ):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise SystemExit(
                "SAM 3 not installed. pip install -r requirements.txt\n"
                f"(original error: {e})"
            ) from e

        ckpt_str = str(checkpoint)
        if ckpt_str == "hf":
            model = build_sam3_image_model(device=device)
        else:
            model = build_sam3_image_model(checkpoint_path=ckpt_str, device=device)
        self._processor = Sam3Processor(model, device=device)
        self._device = device

    def segment(
        self,
        crop_bgr: np.ndarray,
        text_prompts: tuple[str, ...],
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """Run text prompts, return (mask, score, winning_prompt). mask is None if all fail."""
        pil = _bgr_to_pil(crop_bgr)
        # bfloat16 autocast required by SAM 3 fused kernels.
        with torch.autocast(self._device if self._device == "cuda" else "cpu",
                            dtype=torch.bfloat16):
            state = self._processor.set_image(pil)
            best: Tuple[Optional[np.ndarray], float, str] = (None, -1.0, "")
            for text in text_prompts:
                out = self._processor.set_text_prompt(prompt=text, state=state)
                masks = out.get("masks")
                scores = out.get("scores")
                if masks is None or len(masks) == 0:
                    continue
                s_arr = self._to_numpy(scores)
                m_arr = self._to_numpy(masks)
                i = int(np.argmax(s_arr))
                s = float(s_arr[i])
                # SAM 3 mask shape: (N, 1, H, W) → squeeze leading channel
                m = m_arr[i]
                if m.ndim == 3:
                    m = m[0]
                m_bool = m.astype(bool)
                if s > best[1]:
                    best = (m_bool, s, text)
                self._processor.reset_all_prompts(state)
        return best

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x)
