"""
video_feeder.py
---------------
Simulates an ESP32-CAM MJPEG stream.

Reads a local video file, resizes each frame to the chosen resolution,
throttles output to the chosen FPS, and pushes raw BGR frames onto
the shared frame_queue.

Does NOT know about detection, LED logic, or UI.

Queue contract
--------------
  frame_queue  ←  np.ndarray  BGR image at simulated resolution
"""

import queue
import threading
import time

import cv2

import config


class VideoFeeder:

    def __init__(self, frame_queue: queue.Queue):
        self._q          = frame_queue
        self._path       = None
        self._target_w   = 640
        self._target_h   = 360
        self._target_fps = config.DEFAULT_FPS

        self._running = False
        self._paused  = False
        self._thread  = None

        # Public stats — written by worker, read by UI
        self.actual_fps  = 0.0
        self.frame_count = 0
        self.video_fps   = 0.0
        self.video_w     = 0
        self.video_h     = 0
        self.duration_s  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, path: str) -> bool:
        """Open the file and read metadata. Returns False if unreadable."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        self._path      = path
        self.video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total           = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s = total / self.video_fps if self.video_fps else 0
        cap.release()
        return True

    def set_resolution(self, w: int, h: int):
        self._target_w, self._target_h = w, h

    def set_fps(self, fps: int):
        self._target_fps = max(1, int(fps))

    def start(self):
        if self._running or not self._path:
            return
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="VideoFeeder")
        self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._running = False
            return

        fps_count = 0
        fps_timer = time.perf_counter()
        self.frame_count = 0

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                frame = cv2.resize(
                    frame, (self._target_w, self._target_h),
                    interpolation=cv2.INTER_AREA)

            try:
                self._q.put_nowait(frame)
                self.frame_count += 1
            except queue.Full:
                pass

            fps_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                self.actual_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.perf_counter()

            sleep = (1.0 / self._target_fps) - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)

        cap.release()
        self._running = False
