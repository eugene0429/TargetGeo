# Interactive Target Video Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive OpenCV viewer (`seg_pose/viewer/`) that runs the seg_pose pipeline on video files and RTSP streams and overlays bbox, ellipse, SAM mask, camera-frame normal vector, range, and uncertainty cone — with geodetic fields shown only when optional telemetry is supplied.

**Architecture:** A `viewer/` subpackage with focused modules: `intrinsics` (K building), `overlays` (pure drawing fns), `inference` (FrameAnalyzer wrapping the pipeline → FrameResult), `source` (FileSource + StreamSource with dependency injection), `telemetry` (optional CSV loader), `app` (threaded cv2 window), `__main__` (CLI). Heavy SAM inference runs lazily on a background worker thread; results are cached per frame (file) or kept as latest (stream).

**Tech Stack:** Python 3.10, OpenCV (cv2), NumPy, the existing seg_pose pipeline (detector, sam3, ellipse_core, pose_solver, disambiguate, covariance, transforms). Runs in the repo-local venv `/home/sim2real/TargetGeo/.venv`.

---

## Prerequisites & conventions (read first)

- **Environment is already set up** via `./setup_env.sh`: venv at `/home/sim2real/TargetGeo/.venv`, package importable as `seg_pose` (symlink inside the venv's site-packages), detector weights copied into `models/`.
- **Python:** always `/home/sim2real/TargetGeo/.venv/bin/python`.
- **Run tests from a NEUTRAL cwd** (not the repo dir) so the repo's `sam3.py` doesn't shadow the installed `sam3` package. Canonical test command:
  ```bash
  cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/<file> -v -p no:cacheprovider
  ```
- **Imports in tests and code:** use the canonical package name, e.g. `from seg_pose.viewer.overlays import draw_bbox`. Viewer modules import siblings via the same `seg_pose.*` path (matches the existing test suite).
- **Commit after each task** with the venv-run tests green. Push at the end (SSH remote already configured).
- Viewer test files do NOT need a GPU or SAM weights — `FrameAnalyzer` takes injected detector/segmenter fakes (same DI pattern as `tests/test_sam3.py`).

## File structure

```
seg_pose/viewer/
  __init__.py        # empty package marker (must NOT import sam3-using modules)
  intrinsics.py      # build_K() from --hfov-deg or --fx/--fy/--cx/--cy
  overlays.py        # pure drawing: project helpers, bbox, ellipse, mask, normal, hud, render
  inference.py       # FrameResult dataclass + FrameAnalyzer (pipeline wrapper)
  telemetry.py       # load_telemetry() optional CSV -> dict[idx] -> pose dict
  source.py          # FrameSource, FileSource, StreamSource, open_source()
  app.py             # ViewerApp: threaded cv2 window + event loop
  __main__.py        # CLI (argparse) -> build K/analyzer/source -> ViewerApp.run()
run_viewer.sh        # launcher: .venv python -m seg_pose.viewer "$@"
tests/
  test_viewer_intrinsics.py
  test_viewer_overlays.py
  test_viewer_inference.py
  test_viewer_telemetry.py
  test_viewer_source.py
```

---

### Task 1: Package skeleton + intrinsics

**Files:**
- Create: `seg_pose/viewer/__init__.py`
- Create: `seg_pose/viewer/intrinsics.py`
- Test: `tests/test_viewer_intrinsics.py`

- [ ] **Step 1: Create the empty package marker**

Create `seg_pose/viewer/__init__.py` with exactly:

```python
"""Interactive video viewer for the seg_pose pipeline."""
```

(Keep it empty of imports — `python -m seg_pose.viewer` imports this before the
sys.path scrub in `__main__`, so it must not pull in sam3-using modules.)

- [ ] **Step 2: Write the failing test**

Create `tests/test_viewer_intrinsics.py`:

```python
import numpy as np
import pytest

from seg_pose.viewer.intrinsics import build_K


def test_build_K_from_explicit_fx_fy():
    K = build_K(1920, 1080, fx=1500.0, fy=1500.0, cx=960.0, cy=540.0)
    assert K.shape == (3, 3)
    assert K[0, 0] == 1500.0 and K[1, 1] == 1500.0
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0
    assert K[2, 2] == 1.0


def test_build_K_explicit_defaults_principal_point_to_center():
    K = build_K(640, 480, fx=600.0, fy=600.0)
    assert K[0, 2] == 320.0 and K[1, 2] == 240.0


def test_build_K_from_hfov():
    # hfov=90 deg, width=1920 -> f = (1920/2)/tan(45deg) = 960
    K = build_K(1920, 1080, hfov_deg=90.0)
    assert K[0, 0] == pytest.approx(960.0, rel=1e-6)
    assert K[1, 1] == pytest.approx(960.0, rel=1e-6)
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0


def test_build_K_defaults_to_60deg_when_nothing_given():
    K = build_K(1000, 1000)
    expected_f = (1000 / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    assert K[0, 0] == pytest.approx(expected_f, rel=1e-6)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_intrinsics.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'seg_pose.viewer.intrinsics'`

- [ ] **Step 4: Write the implementation**

Create `seg_pose/viewer/intrinsics.py`:

```python
"""Build a camera intrinsics matrix K from CLI-style parameters."""

from __future__ import annotations

import numpy as np

DEFAULT_HFOV_DEG = 60.0


def fov_to_K(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """K from horizontal FOV; principal point at image center, fx=fy, no skew."""
    f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def build_K(
    width: int,
    height: int,
    *,
    hfov_deg: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """Build K. Explicit fx & fy win; otherwise derive from hfov (default 60deg).

    When fx/fy are given, cx/cy default to the image center.
    """
    if fx is not None and fy is not None:
        cx = width / 2.0 if cx is None else cx
        cy = height / 2.0 if cy is None else cy
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=float)
    return fov_to_K(DEFAULT_HFOV_DEG if hfov_deg is None else hfov_deg, width, height)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_intrinsics.py -v -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/__init__.py seg_pose/viewer/intrinsics.py tests/test_viewer_intrinsics.py
git commit -m "feat(viewer): K intrinsics builder + package skeleton"
```

---

### Task 2: Overlay projection helpers

**Files:**
- Create: `seg_pose/viewer/overlays.py`
- Test: `tests/test_viewer_overlays.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer_overlays.py`:

```python
import numpy as np

from seg_pose.viewer.overlays import project_point, project_normal_arrow


def _K():
    return np.array([[100.0, 0.0, 50.0],
                     [0.0, 100.0, 50.0],
                     [0.0, 0.0, 1.0]])


def test_project_point_center():
    # point on optical axis at z=10 -> principal point
    assert project_point(np.array([0.0, 0.0, 10.0]), _K()) == (50.0, 50.0)


def test_project_point_behind_camera_returns_none():
    assert project_point(np.array([0.0, 0.0, -1.0]), _K()) is None


def test_project_normal_arrow_sideways():
    c = np.array([0.0, 0.0, 10.0])
    n = np.array([1.0, 0.0, 0.0])  # +x in camera frame
    p0, p1 = project_normal_arrow(c, n, _K(), length_m=2.0, px_cap=None)
    # base projects to center; tip = (2,0,10) -> u = 100*2/10 + 50 = 70
    assert p0 == (50, 50)
    assert p1 == (70, 50)


def test_project_normal_arrow_caps_pixel_length():
    c = np.array([0.0, 0.0, 10.0])
    n = np.array([1.0, 0.0, 0.0])
    p0, p1 = project_normal_arrow(c, n, _K(), length_m=2.0, px_cap=10.0)
    # uncapped dx would be 20px; capped to 10px
    assert p0 == (50, 50)
    assert p1 == (60, 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'project_point'`

- [ ] **Step 3: Write the implementation**

Create `seg_pose/viewer/overlays.py`:

```python
"""Pure drawing functions for the viewer. Each operates on a BGR ndarray.

Colors are BGR. No I/O, no global state — directly unit-testable.
"""

from __future__ import annotations

import cv2
import numpy as np

BBOX_COLOR = (0, 255, 0)
ELLIPSE_COLOR = (255, 0, 255)
MASK_COLOR = (0, 255, 200)
NORMAL_COLOR = (0, 0, 255)
NORMAL_REJECT_COLOR = (120, 120, 255)
HUD_BG = (0, 0, 0)
HUD_FG = (255, 255, 255)


def project_point(p_cam: np.ndarray, K: np.ndarray) -> tuple[float, float] | None:
    """Pinhole-project a camera-frame point. Returns (u, v) float or None if z<=0."""
    p = np.asarray(p_cam, dtype=float)
    if p[2] <= 1e-6:
        return None
    u = K[0, 0] * p[0] / p[2] + K[0, 2]
    v = K[1, 1] * p[1] / p[2] + K[1, 2]
    return (float(u), float(v))


def project_normal_arrow(
    c_cam: np.ndarray,
    n_cam: np.ndarray,
    K: np.ndarray,
    length_m: float,
    px_cap: float | None = 150.0,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Project the 3D segment c_cam -> c_cam + length_m * n_cam to image pixels.

    Optionally caps the on-screen arrow length to px_cap pixels so near-frontal
    or distant normals stay visible. Returns (p0, p1) int pixel tuples, or
    (None, None) if either endpoint is behind the camera.
    """
    c = np.asarray(c_cam, dtype=float)
    n = np.asarray(n_cam, dtype=float)
    a = project_point(c, K)
    b = project_point(c + length_m * n, K)
    if a is None or b is None:
        return None, None
    du, dv = b[0] - a[0], b[1] - a[1]
    L = float(np.hypot(du, dv))
    if px_cap is not None and L > px_cap and L > 0:
        s = px_cap / L
        b = (a[0] + du * s, a[1] + dv * s)
    return (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1])))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/overlays.py tests/test_viewer_overlays.py
git commit -m "feat(viewer): pinhole projection helpers for overlays"
```

---

### Task 3: Overlay primitives — bbox, ellipse, mask

**Files:**
- Modify: `seg_pose/viewer/overlays.py` (append functions)
- Test: `tests/test_viewer_overlays.py` (append tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_viewer_overlays.py`:

```python
from seg_pose.viewer.overlays import (
    draw_bbox, draw_ellipse, draw_mask,
    BBOX_COLOR, ELLIPSE_COLOR,
)


def _blank():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_draw_bbox_colors_edge():
    img = _blank()
    draw_bbox(img, (10, 10, 40, 40))
    # top edge pixel should be the bbox color
    assert tuple(int(c) for c in img[10, 25]) == BBOX_COLOR


def test_draw_bbox_none_is_noop():
    img = _blank()
    draw_bbox(img, None)
    assert img.sum() == 0


def test_draw_ellipse_marks_center():
    img = _blank()
    ellipse = {"cx": 50.0, "cy": 50.0, "major": 40.0, "minor": 20.0, "theta": 0.0}
    draw_ellipse(img, ellipse)
    # center dot drawn in ELLIPSE_COLOR
    assert tuple(int(c) for c in img[50, 50]) == ELLIPSE_COLOR


def test_draw_mask_blends_region():
    img = _blank()
    # a square contour
    contour = np.array([[[20, 20]], [[20, 60]], [[60, 60]], [[60, 20]]], dtype=np.int32)
    draw_mask(img, contour, alpha=0.5)
    assert img[40, 40].sum() > 0  # interior tinted


def test_draw_mask_none_is_noop():
    img = _blank()
    draw_mask(img, None)
    assert img.sum() == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'draw_bbox'`

- [ ] **Step 3: Implement**

Append to `seg_pose/viewer/overlays.py`:

```python
def draw_bbox(img, bbox, color=BBOX_COLOR, thickness=2) -> None:
    """Draw an (x1, y1, x2, y2) rectangle. No-op if bbox is None."""
    if bbox is None:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def draw_ellipse(img, ellipse, color=ELLIPSE_COLOR, thickness=2) -> None:
    """Draw the fitted ellipse outline + center dot. ellipse: dict cx,cy,major,minor,theta."""
    if ellipse is None:
        return
    center = (int(round(ellipse["cx"])), int(round(ellipse["cy"])))
    axes = (int(round(ellipse["major"] / 2.0)), int(round(ellipse["minor"] / 2.0)))
    cv2.ellipse(img, center, axes, float(ellipse["theta"]), 0.0, 360.0,
                color, thickness, cv2.LINE_AA)
    cv2.circle(img, center, 3, color, -1, cv2.LINE_AA)


def draw_mask(img, contour, color=MASK_COLOR, alpha=0.35) -> None:
    """Translucent fill of the mask contour. No-op if contour is None."""
    if contour is None:
        return
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.asarray(contour, dtype=np.int32).reshape(-1, 2)], color)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/overlays.py tests/test_viewer_overlays.py
git commit -m "feat(viewer): bbox/ellipse/mask drawing primitives"
```

---

### Task 4: Overlay — normal arrows, HUD, and render compositor

**Files:**
- Modify: `seg_pose/viewer/overlays.py` (append)
- Test: `tests/test_viewer_overlays.py` (append)

This task uses the `FrameResult` shape defined in Task 5. The fields referenced
here (`bbox`, `mask_contour`, `ellipse`, `candidates`, `chosen_idx`,
`normal_camera`, `range_m`, `cone_deg`, `disambiguation_method`, `fit_method`,
`sam_score`, `status`, `flags`, `lat`, `lon`, `alt_m`) are all defined there.

- [ ] **Step 1: Add failing tests**

Append to `tests/test_viewer_overlays.py`:

```python
from types import SimpleNamespace
from seg_pose.viewer.overlays import draw_normal, draw_hud, render


def _result_ok():
    return SimpleNamespace(
        status="ok", valid=True,
        bbox=(20, 20, 80, 80),
        mask_contour=np.array([[[20, 20]], [[20, 80]], [[80, 80]], [[80, 20]]], dtype=np.int32),
        ellipse={"cx": 50.0, "cy": 50.0, "major": 40.0, "minor": 30.0, "theta": 10.0},
        candidates=[(np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0])),
                    (np.array([0.0, 0.0, 10.0]), np.array([0.0, 1.0, 0.0]))],
        chosen_idx=0,
        normal_camera=(1.0, 0.0, 0.0),
        offset_camera_m=(0.0, 0.0, 10.0),
        range_m=10.0, cone_deg=1.5,
        disambiguation_method="visibility", fit_method="hull", sam_score=0.9,
        flags=[], lat=None, lon=None, alt_m=None, normal_world=None,
    )


def _K200():
    return np.array([[200.0, 0.0, 50.0], [0.0, 200.0, 50.0], [0.0, 0.0, 1.0]])


def test_draw_normal_draws_something():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_normal(img, _result_ok(), _K200(), arrow_len_m=2.0)
    assert img.sum() > 0


def test_draw_hud_returns_same_shape_and_writes_text():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, _result_ok())
    assert img.shape == (480, 640, 3)
    assert img.sum() > 0


def test_draw_hud_shows_na_without_telemetry():
    # Smoke: no telemetry -> still renders without error
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, _result_ok())  # lat/lon None
    assert img.sum() > 0


def test_render_all_layers_changes_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = render(img, _result_ok(), _K200(),
                 layers={"bbox": True, "ellipse": True, "mask": True,
                         "normal": True, "hud": True},
                 arrow_len_m=2.0)
    assert out.shape == img.shape
    assert out.sum() > 0


def test_render_respects_layer_toggles():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = render(img, _result_ok(), _K200(),
                 layers={"bbox": False, "ellipse": False, "mask": False,
                         "normal": False, "hud": False},
                 arrow_len_m=2.0)
    assert out.sum() == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'draw_normal'`

- [ ] **Step 3: Implement**

Append to `seg_pose/viewer/overlays.py`:

```python
def draw_normal(img, result, K, arrow_len_m: float, px_cap: float = 150.0) -> None:
    """Draw the chosen normal (solid) and the rejected candidate (faint).

    Anchors each arrow at its own camera-frame center, projected through K.
    No-op if there are no candidates.
    """
    candidates = getattr(result, "candidates", None)
    if not candidates:
        return
    chosen = getattr(result, "chosen_idx", None)
    for idx, (c_cam, n_cam) in enumerate(candidates):
        is_chosen = (idx == chosen)
        color = NORMAL_COLOR if is_chosen else NORMAL_REJECT_COLOR
        thickness = 2 if is_chosen else 1
        p0, p1 = project_normal_arrow(c_cam, n_cam, K, arrow_len_m, px_cap=px_cap)
        if p0 is None or p1 is None:
            continue
        cv2.arrowedLine(img, p0, p1, color, thickness, cv2.LINE_AA, tipLength=0.2)


def _fmt_vec(v, nd=3):
    if v is None:
        return "N/A"
    return "(" + ", ".join(f"{x:+.{nd}f}" for x in v) + ")"


def draw_hud(img, result, origin=(8, 8), line_h=20) -> None:
    """Draw a translucent text panel with pipeline diagnostics."""
    has_geo = getattr(result, "lat", None) is not None
    geo = (f"lat {result.lat:.6f}  lon {result.lon:.6f}  alt {result.alt_m:.2f} m"
           if has_geo else "lat/lon/alt: N/A - no telemetry")
    rng = "N/A" if getattr(result, "range_m", None) is None else f"{result.range_m:.2f} m"
    cone = "N/A" if getattr(result, "cone_deg", None) is None else f"{result.cone_deg:.2f} deg"
    lines = [
        f"status: {result.status}   sam={getattr(result, 'sam_score', 0.0):.3f}"
        f"   fit={getattr(result, 'fit_method', '-')}"
        f"   disambig={getattr(result, 'disambiguation_method', '-')}",
        f"range: {rng}    normal_cone: {cone}",
        f"normal_camera: {_fmt_vec(getattr(result, 'normal_camera', None))}",
        f"normal_world : {_fmt_vec(getattr(result, 'normal_world', None))}",
        f"geodetic: {geo}",
        f"flags: {', '.join(getattr(result, 'flags', []) or []) or '-'}",
    ]
    x0, y0 = origin
    w = 560
    h = line_h * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0.0, dst=img)
    for i, line in enumerate(lines):
        y = y0 + 18 + i * line_h
        cv2.putText(img, line, (x0 + 6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, HUD_FG, 1, cv2.LINE_AA)


def render(img, result, K, layers: dict, arrow_len_m: float) -> np.ndarray:
    """Composite enabled overlay layers onto a copy of img. Returns the copy."""
    out = img.copy()
    if result is None:
        return out
    if layers.get("mask"):
        draw_mask(out, getattr(result, "mask_contour", None))
    if layers.get("bbox"):
        draw_bbox(out, getattr(result, "bbox", None))
    if layers.get("ellipse"):
        draw_ellipse(out, getattr(result, "ellipse", None))
    if layers.get("normal"):
        draw_normal(out, result, K, arrow_len_m)
    if layers.get("hud"):
        draw_hud(out, result)
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_overlays.py -v -p no:cacheprovider`
Expected: PASS (14 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/overlays.py tests/test_viewer_overlays.py
git commit -m "feat(viewer): normal arrows, HUD panel, layer compositor"
```

---

### Task 5: FrameResult + FrameAnalyzer (camera-frame / no-telemetry path)

**Files:**
- Create: `seg_pose/viewer/inference.py`
- Test: `tests/test_viewer_inference.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer_inference.py`:

```python
import numpy as np

from seg_pose.viewer.inference import FrameAnalyzer, FrameResult


class _FakeDetector:
    """Returns a fixed bbox covering the synthetic disk."""
    def __init__(self, bbox):
        self.bbox = bbox

    def detect(self, rgb):
        return self.bbox


class _FakeSegmenter:
    """Returns a filled-circle mask for the crop region it is given."""
    def __init__(self, score=0.9):
        self.score = score

    def segment(self, crop_bgr, text_prompts):
        h, w = crop_bgr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        r = min(h, w) * 0.4
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        return mask, self.score, text_prompts[0]


def _synthetic_frame():
    # 400x400 black frame; a bright disk centered at (200,200), radius ~60
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:400, :400]
    disk = ((xx - 200) ** 2 + (yy - 200) ** 2) <= 60 ** 2
    img[disk] = 255
    return img


def _K():
    return np.array([[600.0, 0.0, 200.0], [0.0, 600.0, 200.0], [0.0, 0.0, 1.0]])


def test_analyzer_no_telemetry_produces_camera_frame_normal():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector((140, 140, 260, 260)),
        segmenter=_FakeSegmenter(),
    )
    res = analyzer.analyze(_synthetic_frame(), _K(), radius=2.5, n_prompts=1)
    assert isinstance(res, FrameResult)
    assert res.valid is True and res.status == "ok"
    assert res.bbox == (140, 140, 260, 260)
    assert res.ellipse is not None and res.mask_contour is not None
    assert res.normal_camera is not None and len(res.normal_camera) == 3
    assert res.range_m is not None and res.range_m > 0
    assert res.cone_deg is not None
    assert len(res.candidates) == 2 and res.chosen_idx in (0, 1)
    assert res.disambiguation_method in ("visibility", "fallback")
    # no telemetry -> geodetic fields stay None
    assert res.lat is None and res.lon is None and res.alt_m is None
    assert res.normal_world is None


def test_analyzer_no_detection_returns_status():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector(None),
        segmenter=_FakeSegmenter(),
    )
    res = analyzer.analyze(_synthetic_frame(), _K(), radius=2.5, n_prompts=1)
    assert res.valid is False
    assert res.status == "no_detection"
    assert res.bbox is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_inference.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'seg_pose.viewer.inference'`

- [ ] **Step 3: Implement**

Create `seg_pose/viewer/inference.py`:

```python
"""FrameAnalyzer: run the seg_pose pipeline on one frame -> FrameResult.

Reuses the same building blocks as SegPoseEstimator but (a) keeps the SAM mask
so the viewer can draw it, and (b) supports a no-telemetry path that yields
camera-frame normal/range/cone with visibility-based disambiguation. When a
DroneState is supplied, world-frame geodetic fields are filled in too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from seg_pose.sam3 import crop_to_bbox, pad_mask_to_full
from seg_pose.ellipse_core import fit_ellipse_hull, ellipse_params_to_conic
from seg_pose.pose_solver import solve_circle_pose, PoseSolverError
from seg_pose.disambiguate import disambiguate_visibility
from seg_pose.covariance import compute_position_covariance
from seg_pose.pose_types import (
    DroneStateUe, DroneStateGps, DEFAULT_POSE_SIGMA, DEFAULT_INTRINSIC_SIGMA,
)
from seg_pose.transforms import (
    M_UE2CV, ue_rotation_matrix, enu_rotation_matrix,
    world_up_in_cam_cv, world_up_in_cam_enu,
)
from seg_pose.gps import offset_to_target_gps

DEFAULT_TEXT_PROMPTS = (
    "concentric circular target",
    "round archery target on white background",
    "black outer ring of archery target",
)


@dataclass
class FrameResult:
    status: str
    valid: bool = False
    bbox: tuple | None = None
    mask_contour: np.ndarray | None = None
    ellipse: dict | None = None
    candidates: list = field(default_factory=list)
    chosen_idx: int | None = None
    normal_camera: tuple | None = None
    offset_camera_m: tuple | None = None
    range_m: float | None = None
    disambiguation_method: str = "-"
    sam_score: float = 0.0
    fit_method: str = "-"
    cone_deg: float | None = None
    flags: list = field(default_factory=list)
    # world-frame (telemetry only)
    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None
    normal_world: tuple | None = None


class _DronePose:
    """Minimal DronePose shape for covariance.compute_position_covariance."""
    def __init__(self, loc_xyz_ue, pyr_deg):
        self.loc_xyz_ue = loc_xyz_ue
        self.pyr_deg = pyr_deg


def _largest_contour(mask_bool: np.ndarray):
    m8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class FrameAnalyzer:
    """Holds detector + segmenter; analyzes single frames."""

    def __init__(
        self,
        *,
        detector,
        segmenter,
        text_prompts=DEFAULT_TEXT_PROMPTS,
        crop_pad_ratio=0.15,
        min_mask_area_px=200,
        min_minor_axis_px=4.0,
        max_normal_cone_deg=45.0,
        pixel_sigma_px=0.5,
    ):
        self._detector = detector
        self._segmenter = segmenter
        self._text_prompts = tuple(text_prompts)
        self.crop_pad_ratio = float(crop_pad_ratio)
        self.min_mask_area_px = int(min_mask_area_px)
        self.min_minor_axis_px = float(min_minor_axis_px)
        self.max_normal_cone_deg = float(max_normal_cone_deg)
        self.pixel_sigma_px = float(pixel_sigma_px)

    def analyze(self, frame, K, radius, n_prompts=3, telemetry=None) -> FrameResult:
        H, W = frame.shape[:2]
        bbox = self._detector.detect(frame)
        if bbox is None:
            return FrameResult(status="no_detection")

        try:
            crop, crop_xyxy = crop_to_bbox(frame, bbox, pad_ratio=self.crop_pad_ratio)
        except Exception:
            return FrameResult(status="bbox_too_small", bbox=tuple(bbox))
        if crop.size == 0 or min(crop.shape[:2]) < 16:
            return FrameResult(status="bbox_too_small", bbox=tuple(bbox))

        prompts = self._text_prompts[:max(1, int(n_prompts))]
        crop_mask, score, _ = self._segmenter.segment(crop, prompts)
        if crop_mask is None:
            return FrameResult(status="no_mask", bbox=tuple(bbox))

        full_mask = pad_mask_to_full(crop_mask, crop_xyxy, (H, W))
        if int(full_mask.sum()) < self.min_mask_area_px:
            return FrameResult(status="no_mask", bbox=tuple(bbox), sam_score=float(score))

        contour = _largest_contour(full_mask)

        fit, fit_method = fit_ellipse_hull(full_mask, min_minor_axis_px=self.min_minor_axis_px)
        if fit is None:
            return FrameResult(status="fit_failed", bbox=tuple(bbox),
                               mask_contour=contour, sam_score=float(score),
                               fit_method=fit_method)

        ellipse = {"cx": fit.center_x, "cy": fit.center_y,
                   "major": fit.major, "minor": fit.minor, "theta": fit.angle_deg}
        Q = ellipse_params_to_conic(fit)
        try:
            candidates = solve_circle_pose(Q, K, float(radius))
        except PoseSolverError:
            return FrameResult(status="pose_failed", bbox=tuple(bbox),
                               mask_contour=contour, ellipse=ellipse,
                               sam_score=float(score), fit_method=fit_method)

        # world-up prior only when telemetry is available
        world_up = None
        if isinstance(telemetry, DroneStateUe):
            world_up = world_up_in_cam_cv(telemetry.camera_pyr_deg)
        elif isinstance(telemetry, DroneStateGps):
            world_up = world_up_in_cam_enu(telemetry.camera_pyr_deg)

        dr = disambiguate_visibility(candidates, world_up_cv=world_up)
        c_cam = np.asarray(dr.center, dtype=float)
        n_cam = np.asarray(dr.normal, dtype=float)

        # uncertainty cone — magnitude is rotation-frame invariant, so a dummy
        # zero pose is fine when no telemetry is present.
        pyr = telemetry.camera_pyr_deg if telemetry is not None else (0.0, 0.0, 0.0)
        try:
            _, cone_deg = compute_position_covariance(
                ellipse_params=ellipse, K=K, radius=float(radius),
                drone_pose=_DronePose((0.0, 0.0, 0.0), pyr),
                pixel_sigma_px=self.pixel_sigma_px,
                pose_sigma=DEFAULT_POSE_SIGMA, intrinsic_sigma=DEFAULT_INTRINSIC_SIGMA,
                chosen_idx=dr.chosen_idx,
            )
            cone_deg = float(cone_deg)
        except Exception:
            cone_deg = None

        flags = []
        if cone_deg is not None and cone_deg > self.max_normal_cone_deg:
            flags.append("high_normal_cone")
        if dr.method == "fallback":
            flags.append("disambig_fallback")

        result = FrameResult(
            status="ok", valid=True, bbox=tuple(bbox), mask_contour=contour,
            ellipse=ellipse,
            candidates=[(np.asarray(c, float), np.asarray(n, float)) for c, n in candidates],
            chosen_idx=int(dr.chosen_idx),
            normal_camera=tuple(float(v) for v in n_cam),
            offset_camera_m=tuple(float(v) for v in c_cam),
            range_m=float(np.linalg.norm(c_cam)),
            disambiguation_method=dr.method, sam_score=float(score),
            fit_method=fit_method, cone_deg=cone_deg, flags=flags,
        )

        if telemetry is not None:
            self._fill_world(result, c_cam, n_cam, telemetry)
        return result

    @staticmethod
    def _fill_world(result, c_cam, n_cam, telemetry):
        if isinstance(telemetry, DroneStateUe):
            R = ue_rotation_matrix(telemetry.camera_pyr_deg)
            offset_ue = R @ (M_UE2CV.T @ c_cam)
            tx, ty, tz = telemetry.camera_xyz_ue_m
            n_ue = R @ (M_UE2CV.T @ n_cam)
            n_ue = n_ue / np.linalg.norm(n_ue)
            result.normal_world = tuple(float(v) for v in n_ue)
            # UE absolute position isn't geodetic; expose via normal_world + range only.
        elif isinstance(telemetry, DroneStateGps):
            R = enu_rotation_matrix(telemetry.camera_pyr_deg)
            offset_enu = R @ c_cam
            lat, lon, alt = offset_to_target_gps(
                east_m=float(offset_enu[0]), north_m=float(offset_enu[1]),
                up_m=float(offset_enu[2]),
                cam_lat=telemetry.camera_lat, cam_lon=telemetry.camera_lon,
                cam_alt_m=telemetry.camera_alt_m,
            )
            result.lat, result.lon, result.alt_m = lat, lon, alt
            n_enu = R @ n_cam
            n_enu = n_enu / np.linalg.norm(n_enu)
            result.normal_world = tuple(float(v) for v in n_enu)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_inference.py -v -p no:cacheprovider`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/inference.py tests/test_viewer_inference.py
git commit -m "feat(viewer): FrameAnalyzer camera-frame pipeline + FrameResult"
```

---

### Task 6: FrameAnalyzer telemetry (GPS) path

**Files:**
- Test: `tests/test_viewer_inference.py` (append)
- (No code change expected — `_fill_world` was implemented in Task 5; this task verifies it.)

- [ ] **Step 1: Add failing/verification test**

Append to `tests/test_viewer_inference.py`:

```python
import numpy as np
from seg_pose.pose_types import DroneStateGps


def test_analyzer_gps_telemetry_fills_geodetic():
    analyzer = FrameAnalyzer(
        detector=_FakeDetector((140, 140, 260, 260)),
        segmenter=_FakeSegmenter(),
    )
    K = _K()
    state = DroneStateGps(
        camera_lat=37.5063, camera_lon=127.0125, camera_alt_m=80.0,
        camera_pyr_deg=(-30.0, 45.0, 0.0), K=K,
    )
    res = analyzer.analyze(_synthetic_frame(), K, radius=2.5, n_prompts=1, telemetry=state)
    assert res.valid is True
    assert res.lat is not None and res.lon is not None and res.alt_m is not None
    assert res.normal_world is not None and len(res.normal_world) == 3
    # world-up prior should be used for disambiguation
    assert res.disambiguation_method in ("world_up_axis", "fallback")
```

- [ ] **Step 2: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_inference.py -v -p no:cacheprovider`
Expected: PASS (3 passed). If it fails, inspect the `_fill_world` GPS branch and `offset_to_target_gps` signature in `seg_pose/gps.py`.

- [ ] **Step 3: Commit**

```bash
cd /home/sim2real/TargetGeo
git add tests/test_viewer_inference.py
git commit -m "test(viewer): GPS telemetry path fills geodetic fields"
```

---

### Task 7: Frame sources — FileSource + StreamSource

**Files:**
- Create: `seg_pose/viewer/source.py`
- Test: `tests/test_viewer_source.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer_source.py`:

```python
import time
import numpy as np

from seg_pose.viewer.source import FileSource, StreamSource, open_source


class _FakeCapture:
    """Deterministic stand-in for cv2.VideoCapture over a list of frames."""
    def __init__(self, frames, fps=30.0, loop=False):
        self.frames = frames
        self._pos = 0
        self._fps = fps
        self._loop = loop
        self._open = True

    def isOpened(self):
        return self._open

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, val):
        import cv2
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(val)
            return True
        return False

    def read(self):
        if self._pos >= len(self.frames):
            if self._loop:
                self._pos = 0
            else:
                return False, None
        f = self.frames[self._pos]
        self._pos += 1
        return True, f.copy()

    def release(self):
        self._open = False


def _frames(n):
    # frame i is a solid image whose pixel value == i (so identity is checkable)
    return [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]


def test_filesource_frame_count_and_seek():
    cap = _FakeCapture(_frames(5))
    src = FileSource("dummy.mp4", capture=cap)
    assert src.is_stream is False
    assert src.frame_count == 5
    f2 = src.get(2)
    assert int(f2[0, 0, 0]) == 2
    f0 = src.get(0)
    assert int(f0[0, 0, 0]) == 0


def test_streamsource_keeps_latest_frame():
    cap = _FakeCapture(_frames(100), loop=True)
    src = StreamSource("rtsp://x", capture_factory=lambda: cap)
    src.start()
    try:
        # let the grab thread advance through several frames
        deadline = time.time() + 2.0
        first = None
        while time.time() < deadline:
            f = src.latest()
            if f is not None:
                first = int(f[0, 0, 0])
                break
            time.sleep(0.01)
        assert first is not None
        time.sleep(0.2)
        later = src.latest()
        assert later is not None
        # latest() returns a recent frame, and the source is marked streaming
        assert src.is_stream is True
    finally:
        src.release()


def test_open_source_dispatches_by_scheme(monkeypatch):
    import seg_pose.viewer.source as S
    assert isinstance(open_source("rtsp://host/stream",
                                  capture_factory=lambda: _FakeCapture(_frames(3), loop=True)),
                      StreamSource)
    assert isinstance(open_source("/path/clip.mp4",
                                  capture=_FakeCapture(_frames(3))),
                      FileSource)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_source.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'seg_pose.viewer.source'`

- [ ] **Step 3: Implement**

Create `seg_pose/viewer/source.py`:

```python
"""Frame sources: seekable files and live RTSP/UDP streams.

Both accept injected captures/factories so they can be unit-tested without real
media. `cv2.VideoCapture` is the production default.
"""

from __future__ import annotations

import threading

import cv2


class FrameSource:
    is_stream = False

    def release(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class FileSource(FrameSource):
    """Seekable video file."""
    is_stream = False

    def __init__(self, path, capture=None):
        self.path = str(path)
        self._cap = capture if capture is not None else cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {self.path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def get(self, idx: int):
        """Return frame at index idx (BGR ndarray) or None."""
        idx = max(0, int(idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class StreamSource(FrameSource):
    """Live stream. A background thread always holds the most recent frame."""
    is_stream = True

    def __init__(self, url, capture_factory=None, reconnect_delay_s=1.0):
        self.url = str(url)
        self._factory = capture_factory if capture_factory is not None \
            else (lambda: cv2.VideoCapture(self.url))
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = self._factory()
        self.connected = bool(cap and cap.isOpened())
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                self.connected = False
                self._stop.wait(self._reconnect_delay_s)
                cap = self._factory()
                self.connected = bool(cap and cap.isOpened())
                continue
            ok, frame = cap.read()
            if not ok:
                self.connected = False
                cap.release()
                cap = None
                continue
            self.connected = True
            with self._lock:
                self._latest = frame
        if cap is not None:
            cap.release()

    def latest(self):
        """Most recent frame (BGR ndarray) or None if none received yet."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def open_source(uri: str, *, capture=None, capture_factory=None) -> FrameSource:
    """Dispatch by URI scheme: rtsp://, udp://, rtmp:// -> StreamSource; else FileSource."""
    low = str(uri).lower()
    if low.startswith(("rtsp://", "udp://", "rtmp://", "http://", "https://")):
        return StreamSource(uri, capture_factory=capture_factory)
    return FileSource(uri, capture=capture)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_source.py -v -p no:cacheprovider`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/source.py tests/test_viewer_source.py
git commit -m "feat(viewer): FileSource (seek) + StreamSource (latest-frame)"
```

---

### Task 8: Optional telemetry CSV loader

**Files:**
- Create: `seg_pose/viewer/telemetry.py`
- Test: `tests/test_viewer_telemetry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_viewer_telemetry.py`:

```python
import numpy as np

from seg_pose.viewer.telemetry import load_telemetry, build_state


def test_load_telemetry_parses_rows(tmp_path):
    csv = tmp_path / "tele.csv"
    csv.write_text(
        "frame,lat,lon,alt_m,pitch,yaw,roll\n"
        "0,37.5,127.0,80.0,-30.0,45.0,0.0\n"
        "10,37.6,127.1,81.0,-20.0,40.0,1.0\n"
    )
    table = load_telemetry(str(csv))
    assert set(table.keys()) == {0, 10}
    assert table[0]["lat"] == 37.5
    assert table[10]["pyr"] == (-20.0, 40.0, 1.0)


def test_build_state_makes_gps_drone_state():
    row = {"lat": 37.5, "lon": 127.0, "alt_m": 80.0, "pyr": (-30.0, 45.0, 0.0)}
    K = np.eye(3)
    state = build_state(row, K)
    assert state.camera_lat == 37.5
    assert state.camera_pyr_deg == (-30.0, 45.0, 0.0)
    assert state.K is K
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_telemetry.py -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'seg_pose.viewer.telemetry'`

- [ ] **Step 3: Implement**

Create `seg_pose/viewer/telemetry.py`:

```python
"""Optional per-frame telemetry loader.

CSV schema (header required):
    frame,lat,lon,alt_m,pitch,yaw,roll

Returns a dict keyed by integer frame index. K is supplied separately by the
caller (it is a fixed calibration, not per-frame telemetry).
"""

from __future__ import annotations

import csv

from seg_pose.pose_types import DroneStateGps


def load_telemetry(path: str) -> dict[int, dict]:
    table: dict[int, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(float(row["frame"]))
            table[idx] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "alt_m": float(row["alt_m"]),
                "pyr": (float(row["pitch"]), float(row["yaw"]), float(row["roll"])),
            }
    return table


def build_state(row: dict, K) -> DroneStateGps:
    """Construct a DroneStateGps from a telemetry row + intrinsics K."""
    return DroneStateGps(
        camera_lat=row["lat"], camera_lon=row["lon"], camera_alt_m=row["alt_m"],
        camera_pyr_deg=row["pyr"], K=K,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests/test_viewer_telemetry.py -v -p no:cacheprovider`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/telemetry.py tests/test_viewer_telemetry.py
git commit -m "feat(viewer): optional telemetry CSV loader"
```

---

### Task 9: ViewerApp (threaded cv2 window) — manual verification

**Files:**
- Create: `seg_pose/viewer/app.py`

No unit test (GUI + GPU). Verified manually in Task 11.

- [ ] **Step 1: Implement the app**

Create `seg_pose/viewer/app.py`:

```python
"""Interactive cv2 window with a background inference worker thread."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from seg_pose.viewer.overlays import render

WINDOW = "TargetGeo viewer"
DEFAULT_LAYERS = {"bbox": True, "ellipse": True, "mask": True, "normal": True, "hud": True}
KEY_TOGGLES = {ord("b"): "bbox", ord("e"): "ellipse", ord("m"): "mask",
               ord("n"): "normal", ord("h"): "hud"}


class ViewerApp:
    def __init__(self, analyzer, source, K, radius, *, n_prompts, telemetry=None,
                 arrow_len_m=None):
        self.analyzer = analyzer
        self.source = source
        self.K = K
        self.radius = float(radius)
        self.n_prompts = int(n_prompts)
        self.telemetry = telemetry or {}
        self.arrow_len_m = float(arrow_len_m if arrow_len_m is not None else radius)
        self.layers = dict(DEFAULT_LAYERS)
        self._cache: dict[int, object] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._req_idx = 0
        self._req_frame = None
        self._worker = None

    # ---- worker -----------------------------------------------------------
    def _worker_loop(self):
        last_done = object()
        while not self._stop.is_set():
            with self._lock:
                idx, frame = self._req_idx, self._req_frame
            key = idx if not self.source.is_stream else id(frame)
            if frame is None or key == last_done or (not self.source.is_stream and idx in self._cache):
                time.sleep(0.005)
                continue
            tele = self._telemetry_for(idx)
            try:
                result = self.analyzer.analyze(frame, self.K, self.radius,
                                               n_prompts=self.n_prompts, telemetry=tele)
            except Exception as exc:  # keep the UI alive
                from seg_pose.viewer.inference import FrameResult
                result = FrameResult(status=f"error: {exc}")
            with self._lock:
                self._cache[key] = (frame, result)
            last_done = key

    def _telemetry_for(self, idx):
        from seg_pose.viewer.telemetry import build_state
        row = self.telemetry.get(idx)
        return build_state(row, self.K) if row else None

    # ---- run loops --------------------------------------------------------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        try:
            if self.source.is_stream:
                self.source.start()
                self._run_stream()
            else:
                self._run_file()
        finally:
            self._stop.set()
            self.source.release()
            cv2.destroyAllWindows()

    def _request(self, idx, frame):
        with self._lock:
            self._req_idx, self._req_frame = idx, frame

    def _cached(self, key):
        with self._lock:
            return self._cache.get(key)

    def _run_file(self):
        n = max(1, self.source.frame_count)
        idx, playing = 0, False
        cv2.createTrackbar("frame", WINDOW, 0, n - 1, lambda v: None)
        while True:
            idx = cv2.getTrackbarPos("frame", WINDOW)
            frame = self.source.get(idx)
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self._request(idx, frame)
            entry = self._cached(idx)
            if entry is not None:
                base, result = entry
                shown = render(base, result, self.K, self.layers, self.arrow_len_m)
            else:
                shown = frame.copy()
                cv2.putText(shown, "analyzing...", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            if not self._handle_common_key(key):
                if key == ord(" "):
                    playing = not playing
                elif key in (81, ord(",")):  # left
                    idx = max(0, idx - 1); cv2.setTrackbarPos("frame", WINDOW, idx)
                elif key in (83, ord(".")):  # right
                    idx = min(n - 1, idx + 1); cv2.setTrackbarPos("frame", WINDOW, idx)
            if playing and idx in self._cache:
                idx = min(n - 1, idx + 1)
                cv2.setTrackbarPos("frame", WINDOW, idx)

    def _run_stream(self):
        while True:
            frame = self.source.latest()
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "connecting to stream...", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW, blank)
            else:
                self._request(0, frame)
                # show the most recently analyzed frame together with its overlay
                latest_entry = None
                with self._lock:
                    if self._cache:
                        latest_entry = list(self._cache.values())[-1]
                        # keep the cache bounded
                        if len(self._cache) > 4:
                            for k in list(self._cache.keys())[:-2]:
                                self._cache.pop(k, None)
                if latest_entry is not None:
                    base, result = latest_entry
                    shown = render(base, result, self.K, self.layers, self.arrow_len_m)
                else:
                    shown = frame.copy()
                cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(15) & 0xFF
            if key == ord("q"):
                break
            self._handle_common_key(key)

    def _handle_common_key(self, key) -> bool:
        if key in KEY_TOGGLES:
            layer = KEY_TOGGLES[key]
            self.layers[layer] = not self.layers[layer]
            return True
        return False
```

- [ ] **Step 2: Byte-compile check (no runtime GUI yet)**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "import seg_pose.viewer.app; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/app.py
git commit -m "feat(viewer): threaded cv2 ViewerApp (file + stream loops)"
```

---

### Task 10: CLI entrypoint + launcher

**Files:**
- Create: `seg_pose/viewer/__main__.py`
- Create: `run_viewer.sh`

- [ ] **Step 1: Implement `__main__.py`**

Create `seg_pose/viewer/__main__.py`:

```python
"""CLI: python -m seg_pose.viewer <video|rtsp> [options]."""

from __future__ import annotations

import argparse
import os
import sys

# Drop the repo dir (and cwd if it IS the repo) from sys.path so the repo's own
# sam3.py cannot shadow the installed top-level `sam3` package.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.curdir) != _REPO]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="seg_pose.viewer",
                                 description="Interactive target video viewer")
    ap.add_argument("source", help="video file path or rtsp://... stream URL")
    ap.add_argument("--hfov-deg", type=float, default=None,
                    help="horizontal FOV to derive K (default 60 if no fx/fy)")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--radius", type=float, default=2.5, help="target radius (m)")
    ap.add_argument("--prompts", type=int, default=None,
                    help="number of SAM text prompts (default: file=3, stream=1)")
    ap.add_argument("--telemetry", default=None, help="optional telemetry CSV")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arrow-len-m", type=float, default=None,
                    help="normal arrow physical length (default = radius)")
    args = ap.parse_args(argv)

    import cv2
    from seg_pose.sam3 import Sam3DiskSegmenter
    from seg_pose.detector import TargetDetector, DEFAULT_DETECTOR_PATH
    from seg_pose.viewer.intrinsics import build_K
    from seg_pose.viewer.inference import FrameAnalyzer
    from seg_pose.viewer.source import open_source, StreamSource
    from seg_pose.viewer.app import ViewerApp

    src = open_source(args.source)

    # Probe one frame for image size -> K
    if isinstance(src, StreamSource):
        src.start()
        frame = None
        for _ in range(200):
            frame = src.latest()
            if frame is not None:
                break
            cv2.waitKey(20)
        if frame is None:
            print("ERROR: no frames from stream", file=sys.stderr)
            return 2
        H, W = frame.shape[:2]
        default_prompts = 1
    else:
        frame = src.get(0)
        if frame is None:
            print("ERROR: cannot read first frame", file=sys.stderr)
            return 2
        H, W = frame.shape[:2]
        default_prompts = 3

    K = build_K(W, H, hfov_deg=args.hfov_deg, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    n_prompts = args.prompts if args.prompts is not None else default_prompts

    telemetry = {}
    if args.telemetry:
        from seg_pose.viewer.telemetry import load_telemetry
        telemetry = load_telemetry(args.telemetry)

    print(f"loading models on {args.device} (SAM ~12s)...", flush=True)
    detector = TargetDetector(checkpoint=DEFAULT_DETECTOR_PATH,
                              conf_threshold=args.conf, device=args.device)
    segmenter = Sam3DiskSegmenter(checkpoint="hf", device=args.device)
    analyzer = FrameAnalyzer(detector=detector, segmenter=segmenter)

    app = ViewerApp(analyzer, src, K, args.radius, n_prompts=n_prompts,
                    telemetry=telemetry, arrow_len_m=args.arrow_len_m)
    print("controls: space=play  <-/->=step  b/e/m/n/h=toggle layers  q=quit", flush=True)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Implement `run_viewer.sh`**

Create `run_viewer.sh`:

```bash
#!/usr/bin/env bash
# Launch the interactive viewer in the repo-local venv.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv missing - run ./setup_env.sh first" >&2
  exit 1
fi
exec "$PY" -m seg_pose.viewer "$@"
```

- [ ] **Step 3: Make launcher executable + import check**

```bash
cd /home/sim2real/TargetGeo
chmod +x run_viewer.sh
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -c "import seg_pose.viewer.__main__; print('cli import ok')"
```
Expected: `cli import ok`

- [ ] **Step 4: Commit**

```bash
cd /home/sim2real/TargetGeo
git add seg_pose/viewer/__main__.py run_viewer.sh
git commit -m "feat(viewer): CLI entrypoint + run_viewer.sh launcher"
```

---

### Task 11: Full-suite regression + manual smoke verification + docs

**Files:**
- Modify: `README.md` (append a Viewer section)

- [ ] **Step 1: Run the entire test suite (regression)**

Run: `cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m pytest /home/sim2real/TargetGeo/tests -q -m "not slow" -p no:cacheprovider`
Expected: PASS — original 42 + new viewer tests (≈25 new), 0 failures.

- [ ] **Step 2: Build a sample clip from the local frames**

The repo has frames under `data/real/` (gitignored). Build a short mp4:

```bash
cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python - <<'PY'
import cv2, glob, os
frames = sorted(glob.glob("/home/sim2real/TargetGeo/data/real/frame_*.png"))[:40]
img = cv2.imread(frames[0]); H, W = img.shape[:2]
vw = cv2.VideoWriter("/tmp/sample_targets.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
for f in frames:
    vw.write(cv2.imread(f))
vw.release()
print("wrote /tmp/sample_targets.mp4", len(frames), "frames")
PY
```

- [ ] **Step 3: Manual smoke — file mode**

Run (needs the X display, `DISPLAY=:1`):
```bash
cd /home/sim2real/TargetGeo && ./run_viewer.sh /tmp/sample_targets.mp4 --hfov-deg 90 --radius 2.5
```
Verify by observation:
- window opens; after the model-load pause, the first frame shows a green bbox, magenta ellipse, translucent mask, a red normal arrow, and the HUD panel.
- HUD shows `range`, `normal_camera`, `normal_cone`, and `geodetic: N/A - no telemetry`.
- trackbar scrubs; `space` plays; `b/e/m/n/h` toggle layers; `q` quits.

Record the outcome (pass/fail + a note). If the window cannot open, re-run over SSH with X forwarding or on the console session.

- [ ] **Step 4: Manual smoke — stream mode (optional, if an RTSP URL is available)**

```bash
cd /home/sim2real/TargetGeo && ./run_viewer.sh rtsp://<host>/<path> --hfov-deg 90
```
Verify the analyzed frame + overlay updates at ~2–6 Hz and `q` quits. If no RTSP
source is available, note it as "not tested - no stream".

- [ ] **Step 5: Document usage in README**

Append to `README.md`:

```markdown
## Interactive viewer

Visualize the pipeline on a video file or RTSP stream:

```bash
./run_viewer.sh <video.mp4 | rtsp://host/stream> \
  [--hfov-deg 60 | --fx FX --fy FY --cx CX --cy CY] \
  [--radius 2.5] [--prompts N] [--telemetry tele.csv] [--conf 0.25]
```

Overlays: detection bbox, fitted ellipse, SAM mask, camera-frame normal
(chosen solid, rejected candidate faint), and a HUD with range and the
uncertainty cone. Without telemetry the geodetic fields show "N/A"; supply a
`--telemetry` CSV (`frame,lat,lon,alt_m,pitch,yaw,roll`) to fill lat/lon/alt and
the world-frame normal.

Controls: `space` play/pause, `←/→` step, `b/e/m/n/h` toggle layers, `q` quit.
File mode runs lazily and caches per frame; stream mode shows the most recently
analyzed frame (SAM cannot keep up with full stream rate). Requires
`./setup_env.sh` to have been run.
```

- [ ] **Step 6: Commit + push**

```bash
cd /home/sim2real/TargetGeo
git add README.md
git commit -m "docs: document the interactive viewer"
git push
```

---

## Self-Review

**1. Spec coverage**
- bbox / ellipse / mask / normal / HUD overlays → Tasks 3–4. ✓
- camera-frame normal/range/cone, no telemetry, visibility fallback → Task 5. ✓
- geodetic "N/A" + optional telemetry full path → Tasks 4 (HUD N/A), 5–6, 8. ✓
- uncertainty cone via dummy pose → Task 5. ✓
- K from hfov or fx/fy/cx/cy → Task 1. ✓
- FileSource lazy + StreamSource latest-frame + scheme auto-detect → Task 7. ✓
- mode-based prompt defaults (file=3, stream=1), CLI override → Task 10. ✓
- threaded cv2 window, trackbar (file), analyzed-frame display (stream), layer toggles → Task 9. ✓
- run via repo-local venv, package-name independence → Tasks 10 (sys.path scrub, run_viewer.sh). ✓
- tests for overlays/source/analyzer/telemetry; GUI manual → Tasks 2–8, 9/11. ✓
- recording (`r`) — spec listed it RTSP-only; **deliberately deferred** (YAGNI: not required for the core test goal). Noted here so it isn't mistaken for an omission.

**2. Placeholder scan:** none — every code/test step contains complete code; every run step has an exact command + expected output.

**3. Type consistency:** `FrameResult` fields are defined once in Task 5 and the same names are used by `overlays.render`/`draw_hud`/`draw_normal` (Task 4) and `app.render` (Task 9). `analyze(frame, K, radius, n_prompts, telemetry)` signature is identical across Tasks 5, 6, 9, 10. `open_source`/`FileSource(capture=)`/`StreamSource(capture_factory=)` match between Task 7 code and tests. `build_K(width, height, *, hfov_deg, fx, fy, cx, cy)` matches between Task 1 and Task 10.
