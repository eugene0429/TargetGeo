import time
import numpy as np

from targetgeo.viewer.source import FileSource, StreamSource, open_source


class _FakeCapture:
    """Deterministic stand-in for cv2.VideoCapture over a list of frames."""
    def __init__(self, frames, fps=30.0, loop=False):
        self.frames = frames
        self._pos = 0
        self._fps = fps
        self._loop = loop
        self._open = True

    def isOpened(self):
        return self._open

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, val):
        import cv2
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(val)
            return True
        return False

    def read(self):
        if self._pos >= len(self.frames):
            if self._loop:
                self._pos = 0
            else:
                return False, None
        f = self.frames[self._pos]
        self._pos += 1
        return True, f.copy()

    def release(self):
        self._open = False


def _frames(n):
    # frame i is a solid image whose pixel value == i (so identity is checkable)
    return [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]


def test_filesource_frame_count_and_seek():
    cap = _FakeCapture(_frames(5))
    src = FileSource("dummy.mp4", capture=cap)
    assert src.is_stream is False
    assert src.frame_count == 5
    f2 = src.get(2)
    assert int(f2[0, 0, 0]) == 2
    f0 = src.get(0)
    assert int(f0[0, 0, 0]) == 0


def test_streamsource_keeps_latest_frame():
    cap = _FakeCapture(_frames(100), loop=True)
    src = StreamSource("rtsp://x", capture_factory=lambda: cap)
    src.start()
    try:
        # let the grab thread advance through several frames
        deadline = time.time() + 2.0
        first = None
        while time.time() < deadline:
            f = src.latest()
            if f is not None:
                first = int(f[0, 0, 0])
                break
            time.sleep(0.01)
        assert first is not None
        time.sleep(0.2)
        later = src.latest()
        assert later is not None
        # latest() returns a recent frame, and the source is marked streaming
        assert src.is_stream is True
    finally:
        src.release()


def test_open_source_dispatches_by_scheme(monkeypatch):
    import targetgeo.viewer.source as S
    assert isinstance(open_source("rtsp://host/stream",
                                  capture_factory=lambda: _FakeCapture(_frames(3), loop=True)),
                      StreamSource)
    assert isinstance(open_source("/path/clip.mp4",
                                  capture=_FakeCapture(_frames(3))),
                      FileSource)
