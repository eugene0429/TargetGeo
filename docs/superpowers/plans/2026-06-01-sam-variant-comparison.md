# SAM Variant Comparison Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-repo benchmark that compares SAM3.1, FastSAM, MobileSAM, and EdgeSAM on the TargetGeo disk-segmentation task, measuring per-frame latency and mask quality using a shared YOLO bbox prompt.

**Architecture:** An adapter pattern wraps each model behind a uniform `segment(rgb_bgr, bbox_xyxy) -> bool mask` interface. A model-agnostic runner detects a box per frame with the existing YOLO detector, box-prompts every available model on the full frame, times the per-frame call, and scores each mask against SAM3.1's mask (IoU) plus an ellipse fit. Results are written to CSV + a markdown summary.

**Tech Stack:** Python 3.10, PyTorch 2.12 (cuda), `sam3`, `ultralytics` (FastSAM + MobileSAM), optional `edge_sam`, OpenCV, NumPy. Reuses `seg_pose.ellipse_core` and `seg_pose.detector`.

---

## Environment notes (read before starting)

- **Interpreter:** always `/home/sim2real/TargetGeo/.venv/bin/python` (and its `pytest`). Never system python.
- **`sam3` import shadowing:** the repo root contains `sam3.py`, which shadows the installed `sam3` *package*. Any code that imports the `sam3` package must NOT run with the repo root as `sys.path[0]`. The benchmark code lives in `benchmarks/sam_compare/` and is always run as a module from a neutral cwd, e.g.:
  ```bash
  cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m benchmarks.sam_compare.bench --root /home/sim2real/TargetGeo --limit 5
  ```
  The runner inserts the repo root onto `sys.path` *after* the stdlib/site-packages so `import seg_pose` works without shadowing `sam3`. (The `seg_pose` symlink already points at the repo root.)
- **Detector weights:** `/home/sim2real/TargetGeo/models/target_detector.pt` (real file, present).
- **Data:** `/home/sim2real/TargetGeo/data/real/*.png` (51 frames).
- **ultralytics numpy convention:** passing a `cv2`-loaded BGR array to an ultralytics model is correct (it expects BGR). SAM3 and EdgeSAM need RGB PIL/array — convert with `cv2.cvtColor(..., COLOR_BGR2RGB)`.

---

## File Structure

```
benchmarks/
  __init__.py              # empty, makes benchmarks a package
  sam_compare/
    __init__.py            # empty
    paths.py               # repo-root resolution + sys.path setup helper
    metrics.py             # iou() + ellipse_summary()
    adapters.py            # BoxSegmenter base + Sam31/FastSam/MobileSam/EdgeSam adapters
    bench.py               # CLI runner: detect -> loop models -> time -> score -> report
    README.md              # how to run / how to add a model
    results/               # gitignored output dir (created at runtime)
tests/
  test_sam_compare_metrics.py   # unit tests for metrics.py
```

`.gitignore` gains `benchmarks/sam_compare/results/`.

---

### Task 1: Package skeleton + gitignore + paths helper

**Files:**
- Create: `benchmarks/__init__.py` (empty)
- Create: `benchmarks/sam_compare/__init__.py` (empty)
- Create: `benchmarks/sam_compare/paths.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create empty package files**

```bash
cd /home/sim2real/TargetGeo
mkdir -p benchmarks/sam_compare
: > benchmarks/__init__.py
: > benchmarks/sam_compare/__init__.py
```

- [ ] **Step 2: Write `benchmarks/sam_compare/paths.py`**

```python
"""Resolve the repo root and put it on sys.path WITHOUT shadowing the sam3 package.

The repo root holds a module `sam3.py` that shadows the installed `sam3` package
when the root is sys.path[0]. We therefore append the root to the END of sys.path
so site-packages (the real sam3 package) win, while `import seg_pose` still works.
"""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root(explicit: str | None = None) -> Path:
    """Return the TargetGeo repo root. Falls back to two parents up from this file."""
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[2]


def ensure_seg_pose_importable(root: Path) -> None:
    """Append `root` to sys.path so `import seg_pose` works (root appended, not prepended)."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
```

- [ ] **Step 3: Add results dir to `.gitignore`**

Append this line to `.gitignore`:

```
benchmarks/sam_compare/results/
```

- [ ] **Step 4: Verify seg_pose still imports from neutral cwd**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "
import sys; sys.path.insert(0, '/home/sim2real/TargetGeo')
from benchmarks.sam_compare.paths import repo_root, ensure_seg_pose_importable
r = repo_root('/home/sim2real/TargetGeo'); ensure_seg_pose_importable(r)
import seg_pose; from sam3.model_builder import build_sam3_image_model
print('imports ok', r)
"
```
Expected: `imports ok /home/sim2real/TargetGeo` (FutureWarning lines from timm are fine).

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/__init__.py benchmarks/sam_compare/__init__.py benchmarks/sam_compare/paths.py .gitignore
git commit -m "feat(bench): sam_compare package skeleton + path helper"
```

---

### Task 2: metrics.py (IoU + ellipse summary) — TDD

**Files:**
- Create: `benchmarks/sam_compare/metrics.py`
- Test: `tests/test_sam_compare_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sam_compare_metrics.py`:

```python
import numpy as np

from benchmarks.sam_compare.metrics import iou, ellipse_summary


def test_iou_identical_masks_is_one():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    assert iou(m, m) == 1.0


def test_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool); a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool); b[6:9, 6:9] = True
    assert iou(a, b) == 0.0


def test_iou_half_overlap():
    a = np.zeros((10, 10), dtype=bool); a[0:4, 0:4] = True   # 16 px
    b = np.zeros((10, 10), dtype=bool); b[2:6, 0:4] = True   # 16 px, overlap rows 2-3 = 8 px
    # intersection 8, union 24 -> 1/3
    assert abs(iou(a, b) - (8.0 / 24.0)) < 1e-9


def test_iou_both_empty_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    assert iou(a, a) == 0.0


def test_ellipse_summary_on_filled_disk_is_ok():
    mask = np.zeros((100, 100), dtype=bool)
    yy, xx = np.ogrid[:100, :100]
    mask[(xx - 50) ** 2 + (yy - 50) ** 2 <= 20 ** 2] = True
    s = ellipse_summary(mask)
    assert s["ok"] is True
    assert abs(s["centroid"][0] - 50) < 2 and abs(s["centroid"][1] - 50) < 2
    assert s["area"] > 0


def test_ellipse_summary_on_empty_is_not_ok():
    s = ellipse_summary(np.zeros((50, 50), dtype=bool))
    assert s["ok"] is False
    assert s["area"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/sim2real/TargetGeo && .venv/bin/python -m pytest tests/test_sam_compare_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.sam_compare.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `benchmarks/sam_compare/metrics.py`:

```python
"""Mask-comparison metrics for the SAM variant benchmark."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .paths import repo_root, ensure_seg_pose_importable

ensure_seg_pose_importable(repo_root())
from seg_pose.ellipse_core import fit_ellipse_to_mask  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks. Returns 0.0 if union is empty."""
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def ellipse_summary(mask: np.ndarray) -> Dict:
    """Fit an ellipse to a boolean mask and summarize fit success + key params."""
    fit = fit_ellipse_to_mask(mask.astype(np.uint8))
    area = float(mask.astype(bool).sum())
    return {
        "ok": bool(fit.valid),
        "centroid": (float(fit.center_x), float(fit.center_y)),
        "area": area,
        "major": float(fit.major),
        "minor": float(fit.minor),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/sim2real/TargetGeo && .venv/bin/python -m pytest tests/test_sam_compare_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/metrics.py tests/test_sam_compare_metrics.py
git commit -m "feat(bench): IoU + ellipse summary metrics with tests"
```

---

### Task 3: adapters.py — base class + SAM3.1 adapter

**Files:**
- Create: `benchmarks/sam_compare/adapters.py`

- [ ] **Step 1: Write the base class + shared helpers + Sam31Adapter**

Create `benchmarks/sam_compare/adapters.py`:

```python
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
```

- [ ] **Step 2: Verify SAM3.1 adapter loads and segments one frame**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "
import sys; sys.path.insert(0, '/home/sim2real/TargetGeo')
import cv2, glob
from benchmarks.sam_compare.adapters import Sam31Adapter
f = sorted(glob.glob('/home/sim2real/TargetGeo/data/real/*.png'))[0]
img = cv2.imread(f); h, w = img.shape[:2]
a = Sam31Adapter()
m = a.segment(img, (w//2-100, h//2-100, w//2+100, h//2+100))
print('available', a.available, 'mask', None if m is None else (m.shape, int(m.sum())))
" 2>&1 | grep -vE "FutureWarning|warnings.warn"
```
Expected: `available True mask ((H, W), <positive int>)` (mask may be small/empty if no object in the test box, but shape should be full image and no crash).

- [ ] **Step 3: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/adapters.py
git commit -m "feat(bench): BoxSegmenter base + SAM3.1 box-prompt adapter"
```

---

### Task 4: FastSAM + MobileSAM adapters

**Files:**
- Modify: `benchmarks/sam_compare/adapters.py`

- [ ] **Step 1: Append FastSamAdapter and MobileSamAdapter**

Add to the end of `benchmarks/sam_compare/adapters.py`:

```python
class FastSamAdapter(BoxSegmenter):
    name = "fastsam"

    def __init__(self, device: str = "cuda", weights: str = "FastSAM-s.pt") -> None:
        super().__init__()
        self.device = device
        try:
            from ultralytics import FastSAM
            self._model = FastSAM(weights)
            self.available = True
        except Exception as e:  # noqa: BLE001
            print(f"[fastsam] unavailable: {e}")
            self.available = False

    def segment(self, rgb_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = rgb_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        res = self._model(
            rgb_bgr, bboxes=[[x1, y1, x2, y2]], device=self.device, verbose=False,
        )
        if not res:
            return None
        return _ultra_mask_to_full(res[0], h, w)


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

    def segment(self, rgb_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        h, w = rgb_bgr.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        res = self._model(
            rgb_bgr, bboxes=[[x1, y1, x2, y2]], device=self.device, verbose=False,
        )
        if not res:
            return None
        return _ultra_mask_to_full(res[0], h, w)
```

- [ ] **Step 2: Verify both adapters load and segment one frame (weights auto-download)**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "
import sys; sys.path.insert(0, '/home/sim2real/TargetGeo')
import cv2, glob
from benchmarks.sam_compare.adapters import FastSamAdapter, MobileSamAdapter
f = sorted(glob.glob('/home/sim2real/TargetGeo/data/real/*.png'))[0]
img = cv2.imread(f); h, w = img.shape[:2]
box = (w//2-150, h//2-150, w//2+150, h//2+150)
for A in (FastSamAdapter, MobileSamAdapter):
    a = A(); m = a.segment(img, box)
    print(a.name, 'available', a.available, 'mask', None if m is None else (m.shape, int(m.sum())))
" 2>&1 | grep -vE "FutureWarning|warnings.warn"
```
Expected: each line prints `available True` and a full-image `(H, W)` mask shape (sum may be 0 if nothing in the box, no crash). First run downloads `FastSAM-s.pt` and `mobile_sam.pt` to the ultralytics cache.

- [ ] **Step 3: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/adapters.py
git commit -m "feat(bench): FastSAM + MobileSAM box-prompt adapters"
```

---

### Task 5: EdgeSAM adapter (pluggable / skippable)

**Files:**
- Modify: `benchmarks/sam_compare/adapters.py`

EdgeSAM is not in ultralytics. It installs from `chongzhou96/EdgeSAM` and needs a weight file (`edge_sam_3x.pth`). The adapter must degrade gracefully: missing package or weight → `available=False`, never a crash.

- [ ] **Step 1: Append EdgeSamAdapter**

Add to the end of `benchmarks/sam_compare/adapters.py`:

```python
import os


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

    def segment(self, rgb_bgr: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        if not self.available:
            return None
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (int(v) for v in bbox)
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
```

- [ ] **Step 2: Verify graceful skip when not installed**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "
import sys; sys.path.insert(0, '/home/sim2real/TargetGeo')
from benchmarks.sam_compare.adapters import EdgeSamAdapter
a = EdgeSamAdapter()
print('edgesam available:', a.available)
assert a.available is False or a.available is True  # must not raise
print('graceful skip ok')
" 2>&1 | grep -vE "FutureWarning|warnings.warn"
```
Expected: prints `[edgesam] unavailable (skipping): ...` then `edgesam available: False` then `graceful skip ok` (no traceback). If `edge_sam` happens to be installed with weights, `available: True` is also acceptable.

- [ ] **Step 3: Document optional EdgeSAM install in README (added fully in Task 7)** — no code change here, just a note that the install steps will be:
  ```bash
  /home/sim2real/TargetGeo/.venv/bin/pip install git+https://github.com/chongzhou96/EdgeSAM.git
  mkdir -p /home/sim2real/TargetGeo/weights
  curl -L -o /home/sim2real/TargetGeo/weights/edge_sam_3x.pth \
    https://huggingface.co/chongzhou/EdgeSAM/resolve/main/edge_sam_3x.pth
  ```

- [ ] **Step 4: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/adapters.py
git commit -m "feat(bench): pluggable/skippable EdgeSAM adapter"
```

---

### Task 6: bench.py runner

**Files:**
- Create: `benchmarks/sam_compare/bench.py`

- [ ] **Step 1: Write the runner**

Create `benchmarks/sam_compare/bench.py`:

```python
"""SAM variant comparison runner.

Usage (run from a NEUTRAL cwd so the local sam3.py does not shadow the package):
    cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m benchmarks.sam_compare.bench \
        --root /home/sim2real/TargetGeo --limit 5
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .paths import repo_root, ensure_seg_pose_importable
from .metrics import iou, ellipse_summary


def detect_boxes(detector, frames: List[Path]) -> Dict[Path, Tuple[int, int, int, int]]:
    """Run the YOLO detector once per frame; keep frames with a detection."""
    boxes: Dict[Path, Tuple[int, int, int, int]] = {}
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[detect] unreadable: {f.name}")
            continue
        box = detector.detect(img)
        if box is None:
            print(f"[detect] no detection: {f.name}")
            continue
        boxes[f] = box
    return boxes


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def run(root: Path, limit: Optional[int], warmup: int, device: str) -> None:
    ensure_seg_pose_importable(root)
    from seg_pose.detector import TargetDetector
    from .adapters import (
        Sam31Adapter, FastSamAdapter, MobileSamAdapter, EdgeSamAdapter,
    )

    data_dir = root / "data" / "real"
    frames = sorted(data_dir.glob("*.png"))
    if limit:
        frames = frames[:limit]
    if not frames:
        raise SystemExit(f"no frames found in {data_dir}")

    print(f"[bench] {len(frames)} frames from {data_dir}")
    detector = TargetDetector(checkpoint=root / "models" / "target_detector.pt",
                              device=device)
    boxes = detect_boxes(detector, frames)
    used = [f for f in frames if f in boxes]
    print(f"[bench] {len(used)}/{len(frames)} frames have a detection")
    if not used:
        raise SystemExit("no detections — nothing to benchmark")

    adapters = [
        Sam31Adapter(device=device),
        FastSamAdapter(device=device),
        MobileSamAdapter(device=device),
        EdgeSamAdapter(device=device),
    ]
    adapters = [a for a in adapters if a.available]
    print(f"[bench] models: {[a.name for a in adapters]}")

    # Pass 1: compute SAM3.1 reference masks (and time them) so IoU has a reference.
    cache_img = {f: cv2.imread(str(f)) for f in used}
    ref_masks: Dict[Path, Optional[np.ndarray]] = {}

    rows: List[Dict] = []
    per_model_latency: Dict[str, List[float]] = {}

    for a in adapters:
        latencies: List[float] = []
        for idx, f in enumerate(used):
            img = cache_img[f]
            box = boxes[f]
            t0 = time.perf_counter()
            mask = a.segment(img, box)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if idx >= warmup:
                latencies.append(dt_ms)
            if a.name == "sam3.1":
                ref_masks[f] = mask
            ref = ref_masks.get(f)
            iou_val = iou(mask, ref) if (mask is not None and ref is not None) else 0.0
            es = ellipse_summary(mask) if mask is not None else {
                "ok": False, "centroid": (0.0, 0.0), "area": 0.0}
            rows.append({
                "model": a.name,
                "frame": f.name,
                "latency_ms": round(dt_ms, 3),
                "warmup": idx < warmup,
                "mask_area": int(0 if mask is None else mask.sum()),
                "iou_vs_sam3": round(iou_val, 4),
                "ellipse_ok": es["ok"],
                "centroid_x": round(es["centroid"][0], 2),
                "centroid_y": round(es["centroid"][1], 2),
            })
        per_model_latency[a.name] = latencies

    out_dir = root / "benchmarks" / "sam_compare" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "results.csv", rows)
    summary = _build_summary(rows, per_model_latency, used)
    _write_summary(out_dir / "summary.md", summary)
    print("\n" + summary)
    print(f"\n[bench] wrote {out_dir / 'results.csv'} and {out_dir / 'summary.md'}")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_summary(rows: List[Dict], lat: Dict[str, List[float]],
                   used: List[Path]) -> str:
    models = list(lat.keys())
    lines = [
        f"# SAM variant comparison ({len(used)} frames, IoU reference = sam3.1)",
        "",
        "| model | mean ms | median ms | p90 ms | mean IoU vs sam3.1 | ellipse ok % |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        mrows = [r for r in rows if r["model"] == m and not r["warmup"]]
        l = lat[m]
        ious = [r["iou_vs_sam3"] for r in mrows]
        ell = [1.0 if r["ellipse_ok"] else 0.0 for r in mrows]
        lines.append(
            f"| {m} | {mean(l):.1f} | {median(l):.1f} | {percentile(l, 90):.1f} | "
            f"{(mean(ious) if ious else float('nan')):.3f} | "
            f"{(100 * mean(ell) if ell else float('nan')):.0f} |"
        )
    return "\n".join(lines)


def _write_summary(path: Path, text: str) -> None:
    with open(path, "w") as fh:
        fh.write(text + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="SAM variant comparison benchmark")
    p.add_argument("--root", default=None, help="TargetGeo repo root")
    p.add_argument("--limit", type=int, default=None, help="max frames")
    p.add_argument("--warmup", type=int, default=3, help="warmup frames excluded from timing")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(repo_root(args.root), args.limit, args.warmup, args.device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run on 5 frames**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m benchmarks.sam_compare.bench \
    --root /home/sim2real/TargetGeo --limit 5 2>&1 | grep -vE "FutureWarning|warnings.warn"
```
Expected: prints model list (at least `sam3.1`, `fastsam`, `mobilesam`; `edgesam` only if installed), a markdown summary table with one row per available model, and a "wrote .../results.csv" line. No traceback.

- [ ] **Step 3: Verify output files exist and are well-formed**

Run:
```bash
ls -la /home/sim2real/TargetGeo/benchmarks/sam_compare/results/
head -3 /home/sim2real/TargetGeo/benchmarks/sam_compare/results/results.csv
cat /home/sim2real/TargetGeo/benchmarks/sam_compare/results/summary.md
```
Expected: `results.csv` (header + rows) and `summary.md` (table) both present.

- [ ] **Step 4: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/bench.py
git commit -m "feat(bench): runner — detect, time, score, CSV + summary report"
```

---

### Task 7: README + full-run verification

**Files:**
- Create: `benchmarks/sam_compare/README.md`

- [ ] **Step 1: Write the README**

Create `benchmarks/sam_compare/README.md`:

````markdown
# SAM variant comparison

Benchmarks SAM3.1, FastSAM, MobileSAM, and (optionally) EdgeSAM on the TargetGeo
disk-segmentation task. Every model is box-prompted with the same YOLO detector
bbox on the full frame; latency and mask quality (IoU vs SAM3.1 + ellipse fit)
are reported.

## Run

Run from a NEUTRAL cwd (the repo root's `sam3.py` shadows the `sam3` package):

```bash
cd /tmp
/home/sim2real/TargetGeo/.venv/bin/python -m benchmarks.sam_compare.bench \
    --root /home/sim2real/TargetGeo
```

Options: `--limit N` (first N frames), `--warmup K` (timing-excluded frames,
default 3), `--device cuda|cpu`.

Outputs land in `benchmarks/sam_compare/results/` (gitignored):
`results.csv` (per model+frame) and `summary.md` (per-model table, also printed).

FastSAM (`FastSAM-s.pt`) and MobileSAM (`mobile_sam.pt`) weights auto-download
from the ultralytics cache on first run.

## EdgeSAM (optional)

EdgeSAM is skipped automatically if not installed. To enable it:

```bash
/home/sim2real/TargetGeo/.venv/bin/pip install git+https://github.com/chongzhou96/EdgeSAM.git
mkdir -p /home/sim2real/TargetGeo/weights
curl -L -o /home/sim2real/TargetGeo/weights/edge_sam_3x.pth \
    https://huggingface.co/chongzhou/EdgeSAM/resolve/main/edge_sam_3x.pth
export EDGE_SAM_CHECKPOINT=/home/sim2real/TargetGeo/weights/edge_sam_3x.pth
```

## Adding a model

Subclass `BoxSegmenter` in `adapters.py`: set `name`, do the lazy load in
`__init__` (set `available=False` on failure), and implement
`segment(rgb_bgr, bbox) -> Optional[bool mask HxW]`. Add it to the `adapters`
list in `bench.py`.
````

- [ ] **Step 2: Full run over all 51 frames**

Run:
```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m benchmarks.sam_compare.bench \
    --root /home/sim2real/TargetGeo 2>&1 | grep -vE "FutureWarning|warnings.warn" | tail -20
```
Expected: detection count line (`N/51 frames have a detection`), model list, and the final summary table with sensible latencies (SAM3.1 slowest, FastSAM/MobileSAM faster) and IoU column (sam3.1 row = 1.000 since it is its own reference). No traceback.

- [ ] **Step 3: Run the metrics unit tests once more (regression check)**

Run: `cd /home/sim2real/TargetGeo && .venv/bin/python -m pytest tests/test_sam_compare_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 4: Commit**

```bash
cd /home/sim2real/TargetGeo
git add benchmarks/sam_compare/README.md
git commit -m "docs(bench): sam_compare README + usage"
```

---

## Self-Review Notes

- **Spec coverage:** speed (latency mean/median/p90) ✓ Task 6; mask quality IoU-vs-SAM3.1 ✓ Task 6; ellipse comparison ✓ Tasks 2+6; box-prompt-all on full frame ✓ Tasks 3–5; benchmarks/ location + gitignored results ✓ Tasks 1+6; EdgeSAM pluggable/skippable ✓ Task 5; data/real 51 frames + YOLO box ✓ Task 6; warmup=3, cuda, set_image included in timing ✓ Task 6; outputs results.csv + summary.md ✓ Task 6; testing (metrics unit test + smoke run + adapter degradation) ✓ Tasks 2,5,6,7.
- **Type consistency:** `BoxSegmenter.segment(rgb_bgr, bbox) -> Optional[np.ndarray]` and `.name`/`.available` used identically across all adapters and the runner; `metrics.iou(a,b)` and `metrics.ellipse_summary(mask)` signatures match their call sites in `bench.py`.
- **Non-goals respected:** no pose-error pipeline, no model tuning, no change to production seg_pose path.
