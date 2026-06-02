"""Tkinter video viewer for the targetgeo pipeline.

A real player UI: ▶/❚❚ play-pause, ◀ ▶ frame step, scrubbable slider, speed
buttons, frame/second jump, per-layer toggles, and a status bar — modelled on
the original tools/m2 viewer but driven by the targetgeo FrameAnalyzer.

SAM 3.1 is heavy (~0.5 s/frame), so a background worker prefetches and caches
per-frame results; the UI plays through cached frames at the chosen speed and
shows a "analyzing…" placeholder while the worker catches up. Layer toggles and
re-visits are instant (no re-inference).
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from targetgeo.viewer.overlays import render
from targetgeo.viewer.inference import FrameResult

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
LOOKAHEAD = 64  # frames the worker may prefetch ahead of the playhead

# Layers whose overlays require the heavy SAM+ellipse+pose stage. bbox comes
# from the detector alone (~19 ms), so a bbox-only view skips SAM entirely.
SAM_LAYERS = ("mask", "ellipse", "normal", "hud")
RANK_BBOX = 0   # detector only
RANK_FULL = 1   # detector + SAM + ellipse + pose


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
        self.is_stream = bool(getattr(source, "is_stream", False))

        # shared state (worker writes, UI reads)
        self._cache: dict[int, tuple] = {}
        self._stream_entry: tuple | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

        # playback state (UI thread only)
        self.current_idx = 0
        self.want_idx = 0
        self.playing = False
        self._slider_dragging = False
        self._tk_image = None
        self._needed_mirror = RANK_FULL  # plain int read by the worker thread

        if not self.is_stream:
            self.total = max(1, int(getattr(source, "frame_count", 1)))
            self.fps = float(getattr(source, "fps", 30.0)) or 30.0
        else:
            self.total = 0
            self.fps = float(getattr(source, "fps", 30.0)) or 30.0

    # ---- entry point --------------------------------------------------------
    def run(self):
        self.root = tk.Tk()
        self.speed_var = tk.DoubleVar(value=1.0)
        # All layers ON by default: the SAM-dependent layers (mask/ellipse/
        # normal/hud) are enabled at launch, so SAM 3.1 (~3GB GPU) loads up front
        # via need_sam=True. Toggle layers off to fall back to the light
        # detector-only (bbox) path.
        self.layer_vars = {
            "bbox": tk.BooleanVar(value=True),
            "mask": tk.BooleanVar(value=True),
            "ellipse": tk.BooleanVar(value=True),
            "normal": tk.BooleanVar(value=True),
            "hud": tk.BooleanVar(value=True),
        }

        first = self._grab_initial_frame()
        if first is None:
            self.root.destroy()
            raise SystemExit("no frame available from source")
        fh, fw = first.shape[:2]
        scale = min(MAX_DISPLAY_W / fw, MAX_DISPLAY_H / fh, 1.0)
        self.disp_w, self.disp_h = int(fw * scale), int(fh * scale)

        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._needed_mirror = self._needed_rank()  # seed before worker starts
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        if self.is_stream:
            self._set_playing(True)   # show incoming frames immediately
            self._poll_stream()
        else:
            self.want_idx = 0
            self._refresh_canvas()    # self-schedules until frame 0 is cached

        self.root.mainloop()

    def _grab_initial_frame(self):
        if self.is_stream:
            for _ in range(200):
                f = self.source.latest()
                if f is not None:
                    return f
                time.sleep(0.02)
            return None
        return self.source.get(0)

    # ---- UI ------------------------------------------------------------------
    def _build_ui(self):
        title = "TargetGeo viewer — " + ("stream" if self.is_stream else "file")
        self.root.title(title)

        # File-only widgets: built in all modes for a consistent layout, but
        # disabled (greyed out) in stream mode where seeking/speed make no sense.
        self._file_only = []

        top = ttk.Frame(self.root, padding=4)
        top.pack(side=tk.TOP, fill=tk.X)

        self.play_btn = ttk.Button(top, text="▶ Play", width=10, command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT)
        b_prev = ttk.Button(top, text="◀", width=3, command=lambda: self.step(-1))
        b_prev.pack(side=tk.LEFT, padx=2)
        b_next = ttk.Button(top, text="▶", width=3, command=lambda: self.step(+1))
        b_next.pack(side=tk.LEFT, padx=2)
        self._file_only += [b_prev, b_next]

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        for key, label in (("bbox", "Bbox"), ("mask", "Mask"), ("ellipse", "Ellipse"),
                           ("normal", "Normal"), ("hud", "HUD")):
            ttk.Checkbutton(top, text=label, variable=self.layer_vars[key],
                            command=self._refresh_canvas).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._speed_label = ttk.Label(top, text="Speed:")
        self._speed_label.pack(side=tk.LEFT)
        for s in SPEEDS:
            b = ttk.Button(top, text=f"{s:g}x", width=4,
                           command=lambda s=s: self.speed_var.set(s))
            b.pack(side=tk.LEFT, padx=1)
            self._file_only.append(b)

        self.canvas = tk.Canvas(self.root, width=self.disp_w, height=self.disp_h, bg="#222")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        mid = ttk.Frame(self.root, padding=4)
        mid.pack(side=tk.TOP, fill=tk.X)
        self.slider = ttk.Scale(mid, from_=0, to=max(self.total - 1, 0),
                                orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.slider.bind("<ButtonPress-1>", self._on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self._on_slider_release)
        ttk.Label(mid, text="Jump:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(mid, width=10)
        self.jump_entry.pack(side=tk.LEFT, padx=2)
        self.jump_entry.bind("<Return>", lambda e: self._jump_from_entry("frame"))
        b_jf = ttk.Button(mid, text="frame", width=6,
                          command=lambda: self._jump_from_entry("frame"))
        b_jf.pack(side=tk.LEFT)
        b_js = ttk.Button(mid, text="sec", width=5,
                          command=lambda: self._jump_from_entry("sec"))
        b_js.pack(side=tk.LEFT)
        self._file_only += [self.slider, self.jump_entry, b_jf, b_js]

        if self.is_stream:
            for w in self._file_only:
                w.state(["disabled"])
            self._speed_label.state(["disabled"])

        self.status = ttk.Label(self.root, text="", anchor="w", padding=4)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_keys(self):
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("q", lambda e: self._on_close())
        self.root.bind("<Escape>", lambda e: self._on_close())
        if not self.is_stream:
            self.root.bind("<Left>", lambda e: self.step(-1))
            self.root.bind("<Right>", lambda e: self.step(+1))
            self.root.bind("<Shift-Left>", lambda e: self.step(-int(self.fps)))
            self.root.bind("<Shift-Right>", lambda e: self.step(+int(self.fps)))
            self.root.bind("<Home>", lambda e: self.seek(0))
            self.root.bind("<End>", lambda e: self.seek(self.total - 1))

    # ---- background worker ---------------------------------------------------
    def _worker_loop(self):
        while not self._stop.is_set():
            needed = self._needed_mirror  # plain int; tk vars are read only on the UI thread
            if self.is_stream:
                frame = self.source.latest()
                if frame is None:
                    self._stop.wait(0.02)
                    continue
                res = self._analyze(frame, self._tele_for(self.current_idx), needed)
                with self._lock:
                    self._stream_entry = (frame, res, needed)
                self._stop.wait(0.005)
            else:
                idx = self._next_to_analyze(needed)
                if idx is None:
                    self._stop.wait(0.02)
                    continue
                frame = self.source.get(idx)
                if frame is None:
                    with self._lock:
                        self._cache[idx] = (None, FrameResult(status="read_failed"), needed)
                    continue
                res = self._analyze(frame, self._tele_for(idx), needed)
                with self._lock:
                    self._cache[idx] = (frame, res, needed)

    def _needed_rank(self) -> int:
        """RANK_FULL if any SAM-dependent layer is on, else RANK_BBOX.

        Reads tk vars, so it MUST be called only on the UI thread. It also
        refreshes the plain-int mirror the worker thread reads.
        """
        rank = RANK_FULL if any(self.layer_vars[k].get() for k in SAM_LAYERS) else RANK_BBOX
        self._needed_mirror = rank
        return rank

    def _analyze(self, frame, tele, rank):
        try:
            return self.analyzer.analyze(frame, self.K, self.radius,
                                         n_prompts=self.n_prompts, telemetry=tele,
                                         need_sam=(rank >= RANK_FULL))
        except Exception as exc:  # keep the worker alive
            return FrameResult(status=f"error: {exc}")

    def _next_to_analyze(self, needed):
        with self._lock:
            base = self.want_idx
            for i in range(base, min(self.total, base + LOOKAHEAD)):
                entry = self._cache.get(i)
                if entry is None or entry[2] < needed:
                    return i
        return None

    def _tele_for(self, idx):
        from targetgeo.viewer.telemetry import build_state
        row = self.telemetry.get(idx)
        return build_state(row, self.K) if row else None

    # ---- playback ------------------------------------------------------------
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            (self._tick_stream if self.is_stream else self._tick_file)()

    def _set_playing(self, on: bool):
        self.playing = on
        self.play_btn.config(text="❚❚ Pause" if on else "▶ Play")

    def _tick_file(self):
        if not self.playing:
            return
        nxt = self.current_idx + 1
        if nxt > self.total - 1:
            self._set_playing(False)
            return
        needed = self._needed_rank()
        with self._lock:
            entry = self._cache.get(nxt)
            ready = entry is not None and entry[2] >= needed
        self.want_idx = nxt
        if ready:
            self.current_idx = nxt
            self._refresh_canvas()
            self._sync_slider(nxt)
            delay = max(1, int(1000.0 / max(self.fps * self.speed_var.get(), 0.1)))
            self.root.after(delay, self._tick_file)
        else:
            self._set_status(f"buffering frame {nxt} …")
            self.root.after(20, self._tick_file)

    def _tick_stream(self):
        # stream playback is driven by _poll_stream; nothing to schedule here.
        return

    def step(self, delta: int):
        if self.playing:
            self.toggle_play()
        self.seek(self.current_idx + delta)

    def seek(self, idx: int):
        idx = max(0, min(self.total - 1, int(idx)))
        self.current_idx = idx
        self.want_idx = idx
        self._refresh_canvas()
        self._sync_slider(idx)

    # ---- slider / jump -------------------------------------------------------
    def _on_slider_press(self, _ev):
        self._slider_dragging = True
        if self.playing:
            self.toggle_play()

    def _on_slider_change(self, val):
        if not self._slider_dragging:
            return
        try:
            idx = int(float(val))
        except ValueError:
            return
        self.current_idx = max(0, min(self.total - 1, idx))
        self.want_idx = self.current_idx
        self._refresh_canvas()

    def _on_slider_release(self, _ev):
        self._slider_dragging = False
        self.want_idx = self.current_idx
        self._refresh_canvas()

    def _sync_slider(self, idx):
        if self.is_stream:
            return
        prev = self._slider_dragging
        self._slider_dragging = False
        self.slider.set(idx)
        self._slider_dragging = prev

    def _jump_from_entry(self, unit: str):
        s = self.jump_entry.get().strip()
        if not s:
            return
        try:
            v = float(s)
        except ValueError:
            self._set_status(f"invalid jump value: {s}")
            return
        idx = int(round(v * self.fps)) if unit == "sec" else int(round(v))
        self.seek(idx)

    # ---- rendering -----------------------------------------------------------
    def _poll_stream(self):
        if self._stop.is_set():
            return
        self._needed_rank()  # refresh the worker's rank mirror from current toggles
        if self.playing:
            with self._lock:
                entry = self._stream_entry
            if entry is not None:
                frame, res = entry[0], entry[1]
                self._draw(render(frame, res, self.K, self._layers(), self.arrow_len_m))
                self._set_status_result(res, prefix="stream")
            else:
                self._set_status("connecting to stream …")
        self.root.after(33, self._poll_stream)

    def _refresh_canvas(self):
        if self.is_stream:
            return
        needed = self._needed_rank()
        with self._lock:
            entry = self._cache.get(self.current_idx)
        if entry is None or entry[0] is None:
            self._draw_placeholder(f"analyzing frame {self.current_idx} …")
            self.want_idx = self.current_idx
            self.root.after(40, self._refresh_if_current)
            return
        frame, res, rank = entry
        self._draw(render(frame, res, self.K, self._layers(), self.arrow_len_m))
        self._set_status_result(res)
        if rank < needed:
            # show what we have (e.g. bbox) now; upgrade with SAM layers when ready
            self.want_idx = self.current_idx
            self.root.after(60, self._refresh_if_current)

    def _refresh_if_current(self):
        # re-render the frame the user is parked on once it reaches the needed rank
        if self._stop.is_set() or self.playing:
            return
        needed = self._needed_rank()
        with self._lock:
            entry = self._cache.get(self.current_idx)
        if entry is not None and entry[0] is not None and entry[2] >= needed:
            self._refresh_canvas()
        else:
            self.root.after(60, self._refresh_if_current)

    def _layers(self):
        return {k: v.get() for k, v in self.layer_vars.items()}

    def _draw(self, bgr):
        small = cv2.resize(bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self._tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.disp_w // 2, self.disp_h // 2,
                                 image=self._tk_image, anchor=tk.CENTER)

    def _draw_placeholder(self, msg):
        self.canvas.delete("all")
        self.canvas.create_text(self.disp_w // 2, self.disp_h // 2, text=msg,
                                 fill="#ddd", font=("TkDefaultFont", 16))

    def _set_status(self, text):
        self.status.config(text=text)

    def _set_status_result(self, res, prefix=None):
        if self.is_stream:
            head = "stream"
        else:
            t = self.current_idx / max(self.fps, 1.0)
            total_t = self.total / max(self.fps, 1.0)
            head = (f"frame {self.current_idx}/{self.total - 1} · {t:6.2f}/{total_t:.2f}s "
                    f"· {self.speed_var.get():g}x · cache {len(self._cache)}")
        rng = "N/A" if getattr(res, "range_m", None) is None else f"{res.range_m:.2f}m"
        cone = "N/A" if getattr(res, "cone_deg", None) is None else f"{res.cone_deg:.1f}°"
        n = getattr(res, "normal_camera", None)
        nstr = "N/A" if n is None else "(" + ",".join(f"{x:+.2f}" for x in n) + ")"
        geo = "" if getattr(res, "lat", None) is None else \
            f" · geo {res.lat:.5f},{res.lon:.5f},{res.alt_m:.1f}m"
        self.status.config(
            text=(f"{head} · status={res.status} · range={rng} · cone={cone} "
                  f"· n_cam={nstr}{geo}"))

    # ---- shutdown ------------------------------------------------------------
    def _on_close(self):
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        try:
            self.source.release()
        except Exception:
            pass
        self.root.destroy()
