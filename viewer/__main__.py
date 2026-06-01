"""CLI: python -m seg_pose.viewer <video|rtsp> [options]."""

from __future__ import annotations

import argparse
import os
import sys

# Drop the package root (and cwd if it IS that dir) from sys.path so the repo's
# own sam3.py cannot shadow the installed top-level `sam3` package. The package
# files (incl. sam3.py) live one level up from this viewer/ dir.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.curdir) != _PKG_ROOT]


class _LazySam3Segmenter:
    """Defers Sam3DiskSegmenter construction (and its ~3GB GPU load) until the
    first segment() call, i.e. the first SAM-dependent layer the user views."""

    def __init__(self, build):
        self._build = build
        self._seg = None

    def segment(self, crop_bgr, text_prompts):
        if self._seg is None:
            print("loading SAM 3.1 (~12s, ~3GB GPU) ...", flush=True)
            self._seg = self._build()
        return self._seg.segment(crop_bgr, text_prompts)


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

    print(f"loading detector on {args.device}...", flush=True)
    detector = TargetDetector(checkpoint=DEFAULT_DETECTOR_PATH,
                              conf_threshold=args.conf, device=args.device)
    # SAM 3.1 is ~3 GB of GPU memory and is only needed for mask/ellipse/normal/
    # HUD layers. Load it lazily on first use so a bbox-only session never pays
    # that cost.
    segmenter = _LazySam3Segmenter(
        lambda: Sam3DiskSegmenter(checkpoint="hf", device=args.device))
    analyzer = FrameAnalyzer(detector=detector, segmenter=segmenter)
    print("detector ready. SAM 3.1 loads on first mask/ellipse/normal/HUD use "
          "(~12s, ~3GB GPU).", flush=True)

    app = ViewerApp(analyzer, src, K, args.radius, n_prompts=n_prompts,
                    telemetry=telemetry, arrow_len_m=args.arrow_len_m)
    print("controls: space=play  <-/->=step  b/e/m/n/h=toggle layers  q=quit", flush=True)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
