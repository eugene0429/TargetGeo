# SAM variant comparison

Benchmarks SAM3.1, FastSAM, MobileSAM, and (optionally) EdgeSAM on the TargetGeo
disk-segmentation task. Every model is box-prompted with the same YOLO detector
bbox on the full frame; latency and mask quality (IoU vs SAM3.1 + ellipse fit)
are reported.

## Run

Run from a NEUTRAL cwd. The repo root holds a `sam3.py` module that shadows the
installed `sam3` package if the repo root is on the front of `sys.path`, so the
runner is invoked through the `seg_pose` namespace (the `seg_pose` symlink to the
repo root is already on `sys.path` via site-packages, which keeps the real `sam3`
package importable):

```bash
cd /tmp
/home/sim2real/TargetGeo/.venv/bin/python -m seg_pose.benchmarks.sam_compare.bench \
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
