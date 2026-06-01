# TargetGeo

SAM 3.1 disk segmentation + convex-hull ellipse fit + Chen 2004 pose → drone target 3D localization.

Self-contained, portable: copy this directory to any machine, install requirements, and run.

## Install

```bash
pip install -r requirements.txt
```

SAM 3.1 weights auto-download from `huggingface.co/facebook/sam3.1` on first call. Requires:
1. Accepted license at https://huggingface.co/facebook/sam3.1
2. `hf auth login` configured (or `HF_TOKEN` environment variable)

## Detector weights

Place YOLO rec_bbox weights at `seg_pose/models/target_detector.pt` (default), or pass
an explicit `detector_checkpoint=` to `SegPoseEstimator(...)`. Weights are NOT included
in the repo — supply them per deployment.

## Usage

### Primary: single-call from image (auto-detection)

```python
import cv2
import numpy as np
from seg_pose import SegPoseEstimator, DroneStateUe

estimator = SegPoseEstimator(target_radius_m=2.5)  # loads SAM 3.1 + detector

rgb = cv2.imread("frame.png")  # HxWx3 BGR
K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=float)
state = DroneStateUe(
    camera_xyz_ue_m=(100.0, -200.0, 80.0),
    camera_pyr_deg=(-30.0, 45.0, 0.0),
    K=K,
)

result = estimator.estimate_from_image(rgb, state)
if result.valid:
    print("target UE:", result.target_xyz_ue_m)
    print("normal:", result.normal_world)
    print("cone_deg:", result.normal_cone_deg)
elif result.status == "no_detection":
    print("no target found")
```

### Real flight data

```python
from seg_pose import DroneStateGps

state = DroneStateGps(
    camera_lat=37.5063,
    camera_lon=127.0125,
    camera_alt_m=80.0,
    camera_pyr_deg=(-30.0, 45.0, 0.0),
    K=K,
)
result = estimator.estimate_from_image(rgb, state)
if result.valid:
    print(f"target: lat={result.target_lat:.6f} lon={result.target_lon:.6f} alt={result.target_alt_m:.2f}")
```

### Bypass detection (explicit bbox)

For debugging or when you already have a bbox from another source:

```python
rec_bbox = (780.0, 368.0, 1151.0, 741.0)  # x1, y1, x2, y2 in pixels
result = estimator.estimate(rgb, rec_bbox, state)
```

### Detector alone (no SAM/pose)

```python
from seg_pose import TargetDetector
det = TargetDetector(checkpoint="seg_pose/models/target_detector.pt")
bbox = det.detect(rgb)  # (x1, y1, x2, y2) or None
```

## Output: TargetGeoEstimate

Frame-mode-specific fields are populated based on `DroneStateUe` vs `DroneStateGps`:

| Field | UE input | GPS input |
|---|---|---|
| `target_xyz_ue_m` | populated | None |
| `target_lat/lon/alt_m` | None | populated |
| `normal_world` | UE world frame | ENU frame |
| `offset_camera_m`, `range_m`, `normal_camera`, `ellipse`, `pos_cov_3x3`, `normal_cone_deg`, etc. | populated | populated |

`flags` lists quality warnings (e.g., `high_normal_cone` when `normal_cone_deg > max_normal_cone_deg`).

## Tests

Run from the project root (i.e., the parent of the `seg_pose/` directory):

```bash
pytest seg_pose/tests/                       # fast tests
pytest seg_pose/tests/ -m slow               # E2E with real SAM 3.1 (requires GPU + checkpoint)
pytest seg_pose/tests/ -m 'not slow'         # default: skips slow
```

## Portability check

```bash
cp -r seg_pose /tmp/seg_pose
cd /tmp                           # run from the PARENT of seg_pose/
pip install -r seg_pose/requirements.txt
pytest seg_pose/tests/ -m 'not slow'
```

Run pytest from the **parent** directory (not from inside `seg_pose/`).
Python adds the working directory to `sys.path`, so running from inside the
package would shadow stdlib modules with the same name.

All non-slow tests should pass standalone.

## Architecture

```
┌──────────────────────────────────────────────────┐
│ image + M1 rec_bbox + DroneState (UE or GPS)     │
└──────────────────┬───────────────────────────────┘
                   │ crop with padding
                   ▼
              SAM 3.1 text prompt → disk mask
                   │ pad to full image
                   ▼
              Convex hull → cv2.fitEllipse
                   │ ellipse_params_to_conic
                   ▼
              Chen 2004 → 2 (center, normal) candidates
                   │ disambiguate by world-up axis
                   ▼
              Camera frame → world frame rotation
                   │
        ┌──────────┴──────────┐
        │ UE path             │ GPS path
        │ camera + offset_ue  │ enu2geodetic(cam_gps, offset_enu)
        ▼                     ▼
   target_xyz_ue          target_lat/lon/alt
                   │
                   ▼
              Covariance + quality gates
                   │
                   ▼
              TargetGeoEstimate
```

## Known limitations

- Multi-frame tracking/Kalman is caller's responsibility — this module is stateless per frame.
- Covariance is computed using the UE-convention numerical Jacobian; for GPS path the magnitudes are valid but the axes are interpreted as ENU (translation doesn't affect covariance).
- Gimbal Euler convention differs between UE and real-flight telemetry — caller adapts in the input dataclass.
- SAM 3.0 fallback not implemented — explicit checkpoint required.
