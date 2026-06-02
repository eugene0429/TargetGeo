"""Per-prompt analysis for SAM3.1 — pick a single text prompt to fix.

Production currently runs all of DEFAULT_TEXT_PROMPTS on each crop and keeps the
highest-score mask (text-head cost x N). This script runs each prompt alone on
every crop and reports, per prompt:
  - win rate     : % of frames where this prompt has the top score (= what the
                   multi-prompt pipeline actually picks)
  - mean score   : SAM3 confidence
  - mean IoU vs multi : how well this prompt ALONE reproduces the multi-prompt
                   output (the current production mask)
  - fail %, ellipse ok %

It also separates set_image (shared encoder) latency from per-prompt (text head)
latency, so we can estimate the saving from fixing one prompt:
    multi  ~= set_image + N * per_prompt
    single ~= set_image + 1 * per_prompt

Run from a neutral cwd via the targetgeo namespace:
    cd /tmp && /home/sim2real/TargetGeo/.venv/bin/python \
        -m targetgeo.benchmarks.sam_compare.prompt_analysis --root /home/sim2real/TargetGeo
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

from .paths import repo_root, ensure_targetgeo_importable
from .metrics import iou, ellipse_summary


def _to_numpy(x) -> np.ndarray:
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _extract(out) -> Tuple[Optional[np.ndarray], float]:
    """Best (mask, score) from a SAM3 set_text_prompt output, mirroring sam3.py."""
    masks = out.get("masks")
    scores = out.get("scores")
    if masks is None or len(masks) == 0:
        return None, -1.0
    s_arr = _to_numpy(scores)
    m_arr = _to_numpy(masks)
    i = int(np.argmax(s_arr))
    m = m_arr[i]
    if m.ndim == 3:
        m = m[0]
    return m.astype(bool), float(s_arr[i])


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def run(root: Path, limit: Optional[int], warmup: int, device: str) -> None:
    ensure_targetgeo_importable(root)
    from targetgeo.detector import TargetDetector
    from targetgeo.sam3 import crop_to_bbox, _bgr_to_pil
    from targetgeo.estimator import DEFAULT_TEXT_PROMPTS
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    prompts = list(DEFAULT_TEXT_PROMPTS)
    data_dir = root / "data" / "real"
    frames = sorted(data_dir.glob("*.png"))
    if limit:
        frames = frames[:limit]
    if not frames:
        raise SystemExit(f"no frames found in {data_dir}")

    detector = TargetDetector(checkpoint=root / "models" / "target_detector.pt",
                              device=device)
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device)

    # Precompute crops for frames with a detection.
    crops: List[Tuple[str, np.ndarray]] = []
    for f in frames:
        img = cv2.imread(str(f))
        box = detector.detect(img)
        if box is None:
            print(f"[detect] no detection: {f.name}")
            continue
        crop, _ = crop_to_bbox(img, box, pad_ratio=0.15)
        crops.append((f.name, crop))
    print(f"[prompt] {len(crops)}/{len(frames)} frames have a detection")
    if not crops:
        raise SystemExit("no detections")

    rows: List[Dict] = []
    img_latency: List[float] = []
    prompt_latency: Dict[str, List[float]] = {p: [] for p in prompts}

    for idx, (name, crop) in enumerate(crops):
        pil = _bgr_to_pil(crop)
        with torch.autocast(device if device == "cuda" else "cpu",
                            dtype=torch.bfloat16):
            t0 = time.perf_counter()
            state = processor.set_image(pil)
            t_img = (time.perf_counter() - t0) * 1000.0

            per_prompt: Dict[str, Tuple[Optional[np.ndarray], float, float]] = {}
            for p in prompts:
                t1 = time.perf_counter()
                out = processor.set_text_prompt(prompt=p, state=state)
                mask, score = _extract(out)
                dt = (time.perf_counter() - t1) * 1000.0
                processor.reset_all_prompts(state)
                per_prompt[p] = (mask, score, dt)

        if idx >= warmup:
            img_latency.append(t_img)
            for p in prompts:
                prompt_latency[p].append(per_prompt[p][2])

        # Multi-prompt reference = highest-score prompt this frame (production pick).
        winner = max(prompts, key=lambda p: per_prompt[p][1])
        ref_mask = per_prompt[winner][0]

        for p in prompts:
            mask, score, dt = per_prompt[p]
            iou_vs_multi = (iou(mask, ref_mask)
                            if (mask is not None and ref_mask is not None) else 0.0)
            es = ellipse_summary(mask) if mask is not None else {"ok": False}
            rows.append({
                "frame": name,
                "prompt": p,
                "score": round(score, 4),
                "is_winner": p == winner,
                "iou_vs_multi": round(iou_vs_multi, 4),
                "mask_area": int(0 if mask is None else mask.sum()),
                "ellipse_ok": es["ok"],
                "set_image_ms": round(t_img, 2),
                "prompt_ms": round(dt, 2),
                "warmup": idx < warmup,
            })

    out_dir = root / "benchmarks" / "sam_compare" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "prompt_analysis.csv", rows)
    summary = _build_summary(rows, prompts, img_latency, prompt_latency, len(crops))
    (out_dir / "prompt_summary.md").write_text(summary + "\n")
    print("\n" + summary)
    print(f"\n[prompt] wrote {out_dir / 'prompt_analysis.csv'} and "
          f"{out_dir / 'prompt_summary.md'}")


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _build_summary(rows: List[Dict], prompts: List[str], img_lat: List[float],
                   prompt_lat: Dict[str, List[float]], n_frames: int) -> str:
    scored = [r for r in rows if not r["warmup"]]
    img_mean = mean(img_lat) if img_lat else float("nan")

    lines = [
        f"# SAM3.1 prompt analysis ({n_frames} frames)",
        "",
        f"Shared set_image (encoder): mean {img_mean:.1f} ms/frame.",
        "Reference for IoU = the multi-prompt pick (current production output).",
        "",
        "| prompt | win % | mean score | mean IoU vs multi | fail % | ellipse ok % | prompt_ms |",
        "|---|---|---|---|---|---|---|",
    ]
    per_prompt_ms: Dict[str, float] = {}
    for p in prompts:
        pr = [r for r in scored if r["prompt"] == p]
        n = len(pr)
        wins = 100.0 * sum(1 for r in pr if r["is_winner"]) / n if n else float("nan")
        msc = mean(r["score"] for r in pr) if n else float("nan")
        miou = mean(r["iou_vs_multi"] for r in pr) if n else float("nan")
        failp = 100.0 * sum(1 for r in pr if r["mask_area"] == 0) / n if n else float("nan")
        ellp = 100.0 * sum(1 for r in pr if r["ellipse_ok"]) / n if n else float("nan")
        pms = mean(prompt_lat[p]) if prompt_lat[p] else float("nan")
        per_prompt_ms[p] = pms
        lines.append(
            f"| {p} | {wins:.0f} | {msc:.3f} | {miou:.3f} | {failp:.0f} | "
            f"{ellp:.0f} | {pms:.1f} |"
        )

    avg_prompt_ms = mean(per_prompt_ms.values())
    multi_ms = img_mean + len(prompts) * avg_prompt_ms
    single_ms = img_mean + avg_prompt_ms
    # Best single prompt = highest mean IoU vs multi (ties broken by win rate).
    def keyf(p):
        pr = [r for r in scored if r["prompt"] == p]
        miou = mean(r["iou_vs_multi"] for r in pr) if pr else 0.0
        wins = sum(1 for r in pr if r["is_winner"])
        return (miou, wins)
    best = max(prompts, key=keyf)

    lines += [
        "",
        f"**Latency:** multi-prompt ~= {multi_ms:.0f} ms "
        f"(set_image {img_mean:.0f} + {len(prompts)} x {avg_prompt_ms:.0f}); "
        f"single-prompt ~= {single_ms:.0f} ms. "
        f"Saving ~{multi_ms - single_ms:.0f} ms/frame "
        f"({100 * (multi_ms - single_ms) / multi_ms:.0f}%).",
        "",
        f"**Recommended fixed prompt:** \"{best}\" "
        "(highest mean IoU vs the multi-prompt output).",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="SAM3.1 single-prompt analysis")
    p.add_argument("--root", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(repo_root(args.root), args.limit, args.warmup, args.device)


if __name__ == "__main__":
    main()
