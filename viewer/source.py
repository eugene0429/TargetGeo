"""Frame sources: seekable files and live RTSP/UDP streams.

Both accept injected captures/factories so they can be unit-tested without real
media. `cv2.VideoCapture` is the production default.
"""

from __future__ import annotations

import threading

import cv2


class FrameSource:
    is_stream = False

    def release(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class FileSource(FrameSource):
    """Seekable video file."""
    is_stream = False

    def __init__(self, path, capture=None):
        self.path = str(path)
        self._cap = capture if capture is not None else cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video file: {self.path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0

    def get(self, idx: int):
        """Return frame at index idx (BGR ndarray) or None."""
        idx = max(0, int(idx))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class StreamSource(FrameSource):
    """Live stream. A background thread always holds the most recent frame."""
    is_stream = True

    def __init__(self, url, capture_factory=None, reconnect_delay_s=1.0):
        self.url = str(url)
        self._factory = capture_factory if capture_factory is not None \
            else (lambda: cv2.VideoCapture(self.url))
        self._reconnect_delay_s = float(reconnect_delay_s)
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = self._factory()
        self.connected = bool(cap and cap.isOpened())
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                self.connected = False
                self._stop.wait(self._reconnect_delay_s)
                cap = self._factory()
                self.connected = bool(cap and cap.isOpened())
                continue
            ok, frame = cap.read()
            if not ok:
                self.connected = False
                cap.release()
                cap = None
                continue
            self.connected = True
            with self._lock:
                self._latest = frame
        if cap is not None:
            cap.release()

    def latest(self):
        """Most recent frame (BGR ndarray) or None if none received yet."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def open_source(uri: str, *, capture=None, capture_factory=None) -> FrameSource:
    """Dispatch by URI scheme: rtsp://, udp://, rtmp:// -> StreamSource; else FileSource."""
    low = str(uri).lower()
    if low.startswith(("rtsp://", "udp://", "rtmp://", "http://", "https://")):
        return StreamSource(uri, capture_factory=capture_factory)
    return FileSource(uri, capture=capture)
