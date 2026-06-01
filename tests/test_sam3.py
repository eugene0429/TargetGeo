"""Unit tests for sam3.py — pure helpers + injectable processor."""
import numpy as np

from seg_pose.sam3 import (
    crop_to_bbox, pad_mask_to_full, Sam3DiskSegmenter,
)


def test_crop_to_bbox_padded():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bbox = (100, 200, 300, 400)
    crop, (cx1, cy1, cx2, cy2) = crop_to_bbox(img, bbox, pad_ratio=0.1)
    # max(w, h) = 200, pad = 20
    assert (cx1, cy1, cx2, cy2) == (80, 180, 320, 420)
    assert crop.shape == (240, 240, 3)


def test_crop_to_bbox_clamps_at_edges():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (90, 5, 99, 25)
    crop, (cx1, cy1, cx2, cy2) = crop_to_bbox(img, bbox, pad_ratio=0.5)
    assert cx1 >= 0 and cy1 >= 0
    assert cx2 <= 100 and cy2 <= 100


def test_pad_mask_to_full():
    crop_mask = np.zeros((40, 40), dtype=bool)
    crop_mask[10:20, 15:25] = True
    # crop_xyxy is (cx1, cy1, cx2, cy2) — x-first, consistent with crop_to_bbox.
    # Place the 40x40 crop at full-image (cols 200:240, rows 100:140).
    # Then crop_mask[10:20, 15:25] lands at full[110:120, 215:225].
    full = pad_mask_to_full(crop_mask, (200, 100, 240, 140), (480, 640))
    assert full.shape == (480, 640)
    assert full[110:120, 215:225].all()
    assert not full[0:50, 0:50].any()


class _FakeProcessor:
    """Returns a fixed mask & score for any text prompt."""
    def __init__(self, mask_HxW, score=0.8):
        self.mask = mask_HxW
        self.score = score
        self.prompts_called = []

    def set_image(self, img):
        return {"_image_set": True}

    def set_text_prompt(self, *, prompt, state):
        self.prompts_called.append(prompt)
        return {
            "masks": [self.mask],
            "scores": [self.score],
        }

    def reset_all_prompts(self, state):
        pass


def test_segmenter_picks_highest_scoring_prompt():
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 40:60] = True
    proc = _FakeProcessor(mask, score=0.9)

    seg = Sam3DiskSegmenter.__new__(Sam3DiskSegmenter)  # bypass __init__/model load
    seg._processor = proc
    seg._device = "cpu"  # required by segment()

    crop_mask, score, winner = seg.segment(crop, text_prompts=("disk", "circle"))
    assert crop_mask is not None
    assert crop_mask.sum() == 400  # 20x20
    assert score == 0.9
    assert winner in ("disk", "circle")
    assert proc.prompts_called == ["disk", "circle"]
