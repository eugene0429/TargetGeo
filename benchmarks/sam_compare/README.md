# SAM variant comparison

Benchmarks SAM3.1, FastSAM, MobileSAM, and (optionally) EdgeSAM on the TargetGeo
disk-segmentation task.

**Pipeline-faithful:** mirrors production — the YOLO detector bbox is cropped
(15% padding via `crop_to_bbox`) and each model segments the disk *inside that
crop*. Prompting follows each model's capability:

| model | prompt |
|---|---|
| sam3.1 (reference) | text — the production `DEFAULT_TEXT_PROMPTS` |
| fastsam | text — same prompts, CLIP-matched (FastSAM has a text encoder) |
| mobilesam | box — the detector bbox in crop-local coords (no text encoder) |
| edgesam | box — same as mobilesam (no text encoder) |

Reported per model: latency (mean/median/p90, warmup-excluded), mean IoU vs the
SAM3.1 reference mask, and ellipse-fit success rate. Per-frame side-by-side
visualization panels are saved so segmentation quality can be checked by eye.

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
default 3), `--device cuda|cpu`, `--viz-limit N` (max panels to save; `0`
disables, default = all frames).

Outputs land in `benchmarks/sam_compare/results/` (gitignored):
- `results.csv` — one row per (model, frame)
- `summary.md` — per-model table (also printed to console)
- `viz/<frame>.png` — side-by-side panel: crop+detector-box, then each model's
  mask overlaid on the crop, labelled with IoU vs SAM3.1 and mask area

FastSAM (`FastSAM-s.pt` + CLIP weights for text) and MobileSAM (`mobile_sam.pt`)
weights auto-download on first run.

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
`segment(crop_bgr, box_in_crop) -> Optional[bool mask (crop HxW)]` — use the box
or run your own text prompt as appropriate. Add it to the `adapters` list in
`bench.py`.
