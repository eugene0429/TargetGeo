# Interactive Target Video Viewer — Design

**Date:** 2026-06-01
**Status:** Approved (design phase)
**Module under test:** `seg_pose` (this repo)

## 1. Purpose & Scope

An interactive OpenCV-based viewer to test/debug the `seg_pose` pipeline on
video input (file or RTSP stream). For each frame it runs
`detector → crop → SAM 3.1 mask → hull ellipse → Chen 2004 pose` and overlays:

- detection **bbox**
- **ellipse** fit (outline + center)
- SAM **mask** (translucent)
- **normal vector** (camera-frame 3D arrow, projected to image)
- a **HUD** with range, normal components, tilt, uncertainty cone, and status

### Telemetry constraint

No per-frame camera telemetry (pose/intrinsics) exists yet. Therefore:

- The viewer computes **camera-frame** quantities only: `normal_camera`,
  `range_m`, `offset_camera_m`, and the uncertainty `cone_deg`.
- `lat/lon/alt` and `normal_world` are shown as **"N/A — no telemetry"**.
- Normal disambiguation uses the **visibility (n_z) fallback** (no world-up
  prior available), drawing the chosen candidate solid and the rejected one
  dashed.
- The uncertainty `cone_deg` is still computed via
  `compute_position_covariance` with a **dummy zero pose** — the covariance
  *magnitude* is rotation-frame invariant (per `covariance.py` notes), so the
  cone is meaningful without telemetry.
- **Forward-compat:** an optional `--telemetry` file, when present, supplies
  per-frame pose so that frame runs the full `estimate()` path and the HUD
  fills in `lat/lon/alt` + `normal_world`. Default off.

### Required inputs (since intrinsics aren't in telemetry)

- **K (camera intrinsics):** via `--fx/--fy/--cx/--cy`, or `--hfov-deg`
  (fx=fy derived from horizontal FOV; cx,cy default to frame center).
- **target_radius_m:** via `--radius` (default 2.5).

## 2. Benchmark basis (RTX 3090, `drone/.venv`)

| Stage | Time |
|---|---|
| detector (YOLO, 1920×1080) | ~19 ms |
| SAM 3.1 **per prompt** | ~156 ms (fixed; independent of crop size) |
| detect + SAM (3 prompts) | ~488 ms → ~2.1 fps |
| detect + SAM (1 prompt) | ~175 ms → ~5.7 fps |
| SAM model load (once) | ~12 s |

SAM resizes input to a fixed internal resolution, so crop size does **not**
affect latency — **prompt count is the speed lever**. This drives the
mode-based prompt defaults below.

## 3. Architecture

New `viewer/` subpackage inside the module.

### 3.1 `source.py` — frame source abstraction

- **`FileSource`**: seekable. Exposes total frame count and `get(idx)`.
  Supports trackbar scrub. Inference is **lazy on demand**, results cached by
  `frame_idx`. Default **3 prompts**.
- **`StreamSource`** (RTSP/UDP): a dedicated capture thread continuously reads
  and **keeps only the latest frame** (drops backlog to avoid lag). No seek.
  Default **1 prompt** (~6 fps).
- **Auto-detect:** URL scheme `rtsp://` / `udp://` → `StreamSource`; any other
  path → `FileSource`.

### 3.2 `inference.py` — `FrameAnalyzer`

Loads the detector + SAM segmenter **once** and holds them.

`analyze(frame, K, radius, prompts, telemetry=None) -> FrameResult`

Reuses `estimator` building blocks (steps 1–6: crop, segment, hull ellipse,
conic, `solve_circle_pose`, `disambiguate_visibility(world_up=None)`), then the
dummy-pose covariance for the cone. With telemetry, calls the full
`estimate()` and copies world fields.

**`FrameResult`** fields: `bbox`, `mask_contour`, `ellipse` (cx,cy,major,minor,
theta), `normal_camera`, `range_m`, `offset_camera_m`, both pose candidates,
`disambiguation_method`, `sam_score`, `fit_method`, `cone_deg`, `flags`,
`status`, and optional `lat/lon/alt` + `normal_world`.

### 3.3 `overlays.py` — pure drawing functions

`draw_bbox`, `draw_ellipse`, `draw_mask` (translucent), `draw_normal_arrow`
(project the 3D segment `c_cam → c_cam + radius·n_cam` through K, `cv2.arrowedLine`;
chosen solid, rejected candidate dashed/faint), `draw_hud` (text panel).
Each takes a BGR image and a `FrameResult`, returns/draws onto the image. No
state, no I/O — directly unit-testable.

### 3.4 `app.py` — interactive window

- One **background worker thread** runs inference; the main thread renders so
  the UI never blocks. Cache: `dict[frame_idx → FrameResult]`.
- **File mode:** trackbar (frame index), `Space` play/pause (advances only to
  already-analyzed frames — "analyze-and-play"), `←/→` step, `Home/End`.
- **RTSP mode:** no trackbar. Displays the **analyzed frame together with its
  overlay** (alignment guaranteed), refreshed at ~2–6 Hz. `r` toggles
  recording the overlaid output to mp4.
- **Common toggles:** `b` bbox, `e` ellipse, `m` mask, `n` normal, `h` HUD,
  `q` quit.
- Model load shows a "loading…" splash.

### 3.5 `__main__.py` + `run_viewer.sh`

```bash
./run_viewer.sh <video.mp4 | rtsp://...> \
  [--hfov-deg 60 | --fx --fy --cx --cy] [--radius 2.5] \
  [--prompts N] [--telemetry f.csv] [--conf 0.25] [--device cuda]
```

`run_viewer.sh` invokes **`/home/sim2real/drone/.venv/bin/python`** (the venv
with `sam3` installed — not `hailo_dfc_venv`), runs as `python -m <pkg>.viewer`
from the parent dir, and derives `<pkg>` from the directory name so a rename
to `seg_pose` still works.

## 4. Data flow

```
source(frame) ──► worker: FrameAnalyzer.analyze ──► FrameResult ──► cache
                                                                      │
main loop: get frame + cached FrameResult ──► overlays.* ──► cv2.imshow
```

## 5. Error handling

- Detector returns no bbox → overlay "no detection", HUD status, raw frame shown.
- SAM/ellipse/pose failure → reuse `estimate()` failure statuses; HUD shows
  `status` (e.g. `no_mask`, `fit_failed`, `pose_failed`), no normal drawn.
- RTSP disconnect → capture thread retries with backoff; HUD shows "stream
  reconnecting".
- Bad/missing K args → fail fast at startup with a clear message.

## 6. Testing

- **`overlays.py`**: feed a synthetic `FrameResult` + blank image, assert pixels
  changed in expected regions / arrow endpoints land where projected.
- **`source.py`**: `FileSource` seek/count on a tiny generated clip;
  `StreamSource` latest-frame-drop logic with a fake capture.
- **`FrameAnalyzer`**: inject mocked detector/segmenter (as existing tests do),
  assert `FrameResult` fields and the no-telemetry vs telemetry branches.
- **GUI (`app.py`)**: manual verification.

## 7. Out of scope (YAGNI)

- Multi-target tracking across frames.
- Temporal smoothing of pose/normal.
- Recording for file mode (file already has the source); recording is RTSP-only.
- Editing/annotating bboxes in-viewer.
