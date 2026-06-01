"""Target detection sub-module — YOLO wrapper returning rec_bbox per frame."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO
except ImportError as _e:  # noqa: F841
    YOLO = None  # imported lazily so unit tests can mock without ultralytics installed


DEFAULT_DETECTOR_PATH = Path(__file__).parent / "models" / "target_detector.pt"


class TargetDetector:
    """YOLO-based archery target rec_bbox detector.

    Loads weights once on construction. Each `.detect(rgb)` call returns the
    single highest-confidence detection's bbox in pixel xyxy format, or None
    if the model finds nothing above the confidence threshold.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path = DEFAULT_DETECTOR_PATH,
        conf_threshold: float = 0.25,
        device: str = "cuda",
    ):
        if YOLO is None:
            raise SystemExit(
                "ultralytics not installed. pip install -r requirements.txt"
            )
        self.model = YOLO(str(checkpoint))
        self.conf_threshold = float(conf_threshold)
        self.device = device

    def detect(self, rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Run YOLO on one frame. Returns highest-confidence rec_bbox or None."""
        res = self.model.predict(
            rgb, conf=self.conf_threshold, device=self.device, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        confs = res.boxes.conf.cpu().numpy()
        idx = int(np.argmax(confs))
        xyxy = res.boxes.xyxy[idx].cpu().numpy().tolist()
        return int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
