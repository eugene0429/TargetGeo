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
