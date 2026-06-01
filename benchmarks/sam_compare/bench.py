"""SAM variant comparison runner.

Mirrors the production pipeline: YOLO detector bbox -> crop (15% pad) -> each
model segments the disk inside the crop. SAM3.1/FastSAM use text prompts;
MobileSAM/EdgeSAM use the detector bbox (in crop coords) as a box prompt.

Usage (run from a NEUTRAL cwd so the local sam3.py does not shadow the package;
`seg_pose` resolves via its site-packages symlink, so benchmarks is a subpackage):
    cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python -m seg_pose.benchmarks.sam_compare.bench \
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


def run(root: Path, limit: Optional[int], warmup: int, device: str,
        viz_limit: Optional[int]) -> None:
    ensure_seg_pose_importable(root)
    from seg_pose.detector import TargetDetector
    from seg_pose.sam3 import crop_to_bbox
    from .adapters import (
        Sam31Adapter, FastSamAdapter, MobileSamAdapter, EdgeSamAdapter,
    )
    from .viz import make_panel, save_panel

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

    # Precompute the crop + detector-box-in-crop coords for every used frame,
    # mirroring the production crop_to_bbox(pad_ratio=0.15) step.
    crops: Dict[Path, Tuple[np.ndarray, Tuple[int, int, int, int]]] = {}
    for f in used:
        img = cv2.imread(str(f))
        crop, (cx1, cy1, cx2, cy2) = crop_to_bbox(img, boxes[f], pad_ratio=0.15)
        x1, y1, x2, y2 = boxes[f]
        box_in_crop = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        crops[f] = (crop, box_in_crop)

    adapters = [
        Sam31Adapter(device=device),
        FastSamAdapter(device=device, prompt_mode="text"),
        FastSamAdapter(device=device, prompt_mode="point"),
        MobileSamAdapter(device=device),
        EdgeSamAdapter(device=device),
    ]
    adapters = [a for a in adapters if a.available]
    print(f"[bench] models: {[a.name for a in adapters]}")

    ref_masks: Dict[Path, Optional[np.ndarray]] = {}
    rows: List[Dict] = []
    per_model_latency: Dict[str, List[float]] = {}
    # frame -> ordered list of (model_name, mask, iou) for visualization.
    frame_results: Dict[Path, List[Tuple[str, Optional[np.ndarray], float]]] = {
        f: [] for f in used}

    for a in adapters:
        latencies: List[float] = []
        for idx, f in enumerate(used):
            crop, box_in_crop = crops[f]
            t0 = time.perf_counter()
            mask = a.segment(crop, box_in_crop)
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
            frame_results[f].append((a.name, mask, iou_val))
        per_model_latency[a.name] = latencies

    out_dir = root / "benchmarks" / "sam_compare" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "results.csv", rows)
    summary = _build_summary(rows, per_model_latency, used)
    _write_summary(out_dir / "summary.md", summary)
    print("\n" + summary)
    print(f"\n[bench] wrote {out_dir / 'results.csv'} and {out_dir / 'summary.md'}")

    if viz_limit != 0:
        viz_dir = out_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        viz_frames = used if viz_limit is None else used[:viz_limit]
        for f in viz_frames:
            crop, box_in_crop = crops[f]
            panel = make_panel(crop, box_in_crop, frame_results[f])
            save_panel(viz_dir / f"{f.stem}.png", panel)
        print(f"[bench] wrote {len(viz_frames)} visualization panels to {viz_dir}")


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
        "Pipeline-faithful: YOLO bbox -> crop (15% pad) -> segment disk in crop.",
        "Prompts: sam3.1 & fastsam-text = text; fastsam-point/mobilesam/edgesam = point at box center.",
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
    p.add_argument("--viz-limit", type=int, default=None,
                   help="max visualization panels to save (0=disable, default=all)")
    args = p.parse_args()
    run(repo_root(args.root), args.limit, args.warmup, args.device, args.viz_limit)


if __name__ == "__main__":
    main()
