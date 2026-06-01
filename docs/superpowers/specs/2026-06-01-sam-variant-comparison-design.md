# SAM Variant Comparison Harness — Design

**Date:** 2026-06-01
**Status:** Approved, pending implementation plan

## Purpose

SAM 3.1 inference feels slow for the TargetGeo disk-segmentation task. Before
committing to a model swap, benchmark **SAM3.1, FastSAM, MobileSAM, and EdgeSAM**
on the same data, measuring both **per-frame latency** and **mask quality**, so we
can decide whether a lighter model is fast enough *and* accurate enough for the
disk → ellipse → pose pipeline.

This is a throwaway-grade benchmarking harness that lives in-repo for reuse; it is
not part of the production `seg_pose` package.

## Scope decisions

- **Metrics:** speed *and* mask quality (latency + IoU agreement vs SAM3.1 + ellipse-fit comparison).
- **Prompt mode:** box-prompt **all** models with the same YOLO detector bbox (apples-to-apples). SAM3.1 is box-prompted too (not its production text path).
- **Location:** `benchmarks/sam_compare/` tracked in git; weights and results gitignored.
- **EdgeSAM:** pluggable and skippable — installed from `chongzhou96/EdgeSAM`; if the install or weights are missing the adapter reports `available=False` and the run proceeds with the other three.
- **Data:** `data/real` (51 real drone frames). The YOLO detector supplies one box per frame.

## Architecture

Adapter pattern: all four models hide behind one uniform call so the runner is model-agnostic.

```
benchmarks/sam_compare/
  adapters.py    # BoxSegmenter base + 4 concrete adapters
  metrics.py     # IoU + ellipse-comparison helpers (reuses seg_pose.ellipse_core)
  bench.py       # runner: detect -> loop models -> warmup+time -> metrics -> report
  README.md      # how to run / how to add a model
  results/       # gitignored: results.csv, summary.md, optional overlay PNGs
```

### `adapters.py` — uniform interface

```python
class BoxSegmenter:
    name: str
    available: bool   # False if deps/weights missing
    def segment(self, rgb_bgr, bbox_xyxy) -> Optional[np.ndarray]:  # bool mask HxW, full-image coords
        ...
```

Concrete adapters:

- **`Sam31Adapter`** — `processor.set_image(pil)` then `processor.add_geometric_prompt(box=bbox, label=True, state=state)`; extract highest-score mask. bfloat16 autocast on cuda (matches existing `sam3.py`).
- **`FastSamAdapter`** — `from ultralytics import FastSAM`; `model(img, bboxes=[bbox], device="cuda", verbose=False)`; take the result mask.
- **`MobileSamAdapter`** — `from ultralytics import SAM`; `SAM("mobile_sam.pt")(img, bboxes=[bbox], ...)`.
- **`EdgeSamAdapter`** — lazy import wrapped in `try/except`; on `ImportError`/missing weight set `available=False`. Box-prompted via EdgeSAM's predictor API.

Rules for every adapter:
- Lazy import + weight load happens on construction; failures degrade to `available=False`, never crash the run.
- Input is the **full frame + bbox** (no per-model cropping) for direct comparability.
- Output is a full-image boolean mask (`HxW`) or `None` if the model returns nothing.

### `metrics.py`

- `iou(mask_a, mask_b) -> float` — intersection-over-union of two boolean masks.
- `ellipse_summary(mask) -> {ok: bool, centroid: (x,y), area: float, axes, angle}` — fits an ellipse via `seg_pose.ellipse_core` and returns fit success + key params. Used to compare downstream-relevant quality, not just raw IoU.

### `bench.py` — runner

1. Load YOLO `models/target_detector.pt`; run once per frame over `data/real`. Cache one bbox per frame (highest confidence). Frames with no detection are skipped and logged.
2. Construct all four adapters; report which are `available`.
3. For each available model:
   - **Warmup:** first 3 frames, excluded from timing stats.
   - **Timed:** `time.perf_counter` around the per-frame `segment()` call only (model load excluded). The timed window includes the image-embedding/`set_image` step, since that is the real per-frame cost.
   - Record mask + latency per frame.
4. **Quality:** SAM3.1's mask is the reference. For every model+frame compute `iou_vs_sam3` and `ellipse_summary`.
5. Write outputs and print the summary table.

### Output

- **`results/results.csv`** — one row per (model, frame): `model, frame, latency_ms, mask_area, iou_vs_sam3, ellipse_ok, centroid_x, centroid_y`.
- **`results/summary.md`** — per-model table: mean / median / p90 latency_ms, mean IoU vs SAM3.1, ellipse success rate, # detections used.
- **Console** — prints the same summary table at the end of the run.

## Defaults

- **Full-frame input** to every model (most directly comparable). A `--crop` flag (segment within the padded bbox crop, matching SAM3's production path) is a possible later extension, not in initial scope.
- **Warmup = 3 frames** excluded from latency stats.
- **Device = `cuda`.**
- Latency includes the image-embedding step (`set_image`).

## Error handling

- Missing model deps/weights → adapter `available=False`, logged, run continues.
- Frame with no detection → skipped, logged, excluded from all stats.
- Model returns no mask for a frame → row recorded with `mask_area=0`, `iou_vs_sam3=0`, `ellipse_ok=False`.

## Testing

This is a benchmarking tool, not production code. Verification is:
- A smoke run on a small subset (e.g. `--limit 5`) completes and produces `results.csv` + `summary.md`.
- Unit test for `metrics.iou` on hand-constructed masks (known IoU values).
- Adapter `available` degradation verified by importing with a deliberately bad weight path.

## Non-goals

- End-to-end pose-error comparison (deferred; can be added by feeding masks through the full estimator later).
- Tuning/optimizing any individual model.
- Changing the production `seg_pose` segmentation path.
