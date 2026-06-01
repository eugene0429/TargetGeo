"""Visualization: per-frame side-by-side panels of each model's disk mask.

Layout (left to right): the raw crop, then one panel per model showing its mask
overlaid on the crop, labelled with the model name and IoU vs SAM3.1. Saved so a
human can eyeball whether each model actually segmented the disk.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# Distinct BGR overlay colors per panel (cycled if more models).
_COLORS: Sequence[Tuple[int, int, int]] = (
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
)


def _overlay(crop_bgr: np.ndarray, mask: Optional[np.ndarray],
             color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
    """Return a copy of the crop with mask tinted and its contour outlined."""
    out = crop_bgr.copy()
    if mask is None or mask.sum() == 0:
        return out
    m = mask.astype(bool)
    tint = np.zeros_like(out)
    tint[m] = color
    out = cv2.addWeighted(out, 1.0, tint, alpha, 0.0)
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def _label(img: np.ndarray, lines: List[str]) -> np.ndarray:
    """Draw a small dark banner with text lines at the top-left."""
    out = img.copy()
    y = 6
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (4, y), (4 + tw + 6, y + th + 8), (0, 0, 0), -1)
        cv2.putText(out, ln, (7, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + 12
    return out


def _upscale(crop_bgr: np.ndarray, masks: List[Optional[np.ndarray]],
             box: Tuple[int, int, int, int], cell_h: int):
    """Scale a small crop (and its masks/box) up to cell_h for readable panels."""
    h0, w0 = crop_bgr.shape[:2]
    if h0 >= cell_h:
        return crop_bgr, masks, box
    s = cell_h / float(h0)
    new_wh = (max(1, int(round(w0 * s))), cell_h)
    crop_up = cv2.resize(crop_bgr, new_wh, interpolation=cv2.INTER_LINEAR)
    masks_up = [
        None if m is None
        else cv2.resize(m.astype(np.uint8), new_wh,
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        for m in masks
    ]
    box_up = tuple(int(round(v * s)) for v in box)
    return crop_up, masks_up, box_up


def make_panel(crop_bgr: np.ndarray,
               box_in_crop: Tuple[int, int, int, int],
               model_results: List[Tuple[str, Optional[np.ndarray], float]],
               cell_h: int = 256,
               ) -> np.ndarray:
    """Build one horizontal panel image for a single frame.

    model_results: list of (model_name, mask_or_None, iou_vs_sam3).
    The first cell is the raw crop with the detector box drawn. Small crops are
    upscaled to cell_h pixels tall so the masks are easy to inspect by eye.
    """
    names = [n for n, _, _ in model_results]
    masks_in = [m for _, m, _ in model_results]
    ious = [v for _, _, v in model_results]
    # Areas reported at ORIGINAL resolution (consistent with results.csv).
    areas = [0 if m is None else int(m.sum()) for m in masks_in]
    crop_bgr, masks_in, box_in_crop = _upscale(
        crop_bgr, masks_in, box_in_crop, cell_h)

    h = crop_bgr.shape[0]
    cells: List[np.ndarray] = []

    raw = crop_bgr.copy()
    x1, y1, x2, y2 = (int(v) for v in box_in_crop)
    cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange detector box
    cells.append(_label(raw, ["crop + det box"]))

    for i, (name, mask, iou_val) in enumerate(zip(names, masks_in, ious)):
        color = _COLORS[i % len(_COLORS)]
        cell = _overlay(crop_bgr, mask, color)
        lines = [name, f"IoU={iou_val:.3f}", f"area={areas[i]}"]
        cells.append(_label(cell, lines))

    # Pad every cell to the same height before hstack (crops share height already,
    # but guard against off-by-one from copies).
    cells = [c if c.shape[0] == h else cv2.resize(c, (c.shape[1], h)) for c in cells]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    row: List[np.ndarray] = []
    for i, c in enumerate(cells):
        if i:
            row.append(sep)
        row.append(c)
    return np.hstack(row)


def save_panel(path, panel: np.ndarray) -> None:
    cv2.imwrite(str(path), panel)
