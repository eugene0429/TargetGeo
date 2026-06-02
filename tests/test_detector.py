"""TargetDetector unit tests with a mock YOLO model."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from targetgeo.detector import TargetDetector, DEFAULT_DETECTOR_PATH


def _mock_yolo_result(bboxes_xyxy, confs):
    """Build a mock ultralytics result object."""
    mock_box = MagicMock()
    mock_box.xyxy = _mock_tensor(np.array(bboxes_xyxy, dtype=float))
    mock_box.conf = _mock_tensor(np.array(confs, dtype=float))
    mock_box.__len__ = lambda self: len(bboxes_xyxy)
    mock_result = MagicMock()
    mock_result.boxes = mock_box if len(bboxes_xyxy) > 0 else None
    return [mock_result]


def _mock_tensor(arr):
    t = MagicMock()
    t.cpu.return_value = t
    t.numpy.return_value = arr
    # support len() and indexing for xyxy[idx].cpu().numpy().tolist()
    t.__len__ = lambda self: len(arr)
    t.__getitem__ = lambda self, idx: _mock_tensor(arr[idx]) if arr.ndim > 1 else arr[idx]
    t.tolist = lambda: arr.tolist()
    return t


def _make_detector_with_mock(mock_predict_return):
    with patch("targetgeo.detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_predict_return
        mock_yolo_cls.return_value = mock_model
        det = TargetDetector(checkpoint="/fake/path.pt")
    det._mock_model = mock_model  # for assertions
    return det


def test_detect_returns_highest_confidence_bbox():
    result = _mock_yolo_result(
        bboxes_xyxy=[[100, 200, 300, 400], [50, 50, 80, 80]],
        confs=[0.6, 0.9],   # second is higher
    )
    det = _make_detector_with_mock(result)
    bbox = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert bbox == (50, 50, 80, 80)


def test_detect_returns_none_when_no_detections():
    result = _mock_yolo_result(bboxes_xyxy=[], confs=[])
    det = _make_detector_with_mock(result)
    bbox = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert bbox is None


def test_default_checkpoint_path_under_models_dir():
    assert DEFAULT_DETECTOR_PATH.name == "target_detector.pt"
    assert DEFAULT_DETECTOR_PATH.parent.name == "models"
