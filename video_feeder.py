"""
video_feeder.py
---------------
Simulates an ESP32-CAM MJPEG stream by reading a local video file,
resizing it to the chosen resolution, and throttling to the chosen FPS.

Responsibilities:
  - Open / loop the video file
  - Resize each frame to the simulated camera resolution
  - Pace output to match the target FPS
  - Push raw BGR frames onto the shared `frame_queue`
  - Handle pause / stop signals cleanly

Does NOT know anything about detection or the UI.

Queue contract:
  frame_queue.put(frame)
    frame : np.ndarray  BGR image at simulated resolution
"""

import time
import queue
import threading

import cv2

import config


class VideoFeeder:
    """
    Reads a video file and pushes frames onto `frame_queue` at `target_fps`
    resized to `(target_w, target_h)`.

    Usage:
        feeder = VideoFeeder(frame_queue)
        feeder.load("road.mp4")
        feeder.set_resolution(640, 360)
        feeder.set_fps(15)
        feeder.start()
        ...
        feeder.stop()
    """

    def __init__(self, frame_queue: queue.Queue):
        self._q          = frame_queue
        self._path       = None
        self._target_w   = 640
        self._target_h   = 360
        self._target_fps = config.DEFAULT_FPS

        self._running    = False
        self._paused     = False
        self._thread     = None

        # Readable stats (updated by worker thread)
        self.actual_fps  = 0.0
        self.frame_count = 0          # total frames emitted since last start
        self.video_fps   = 0.0        # native FPS of the file
        self.video_w     = 0
        self.video_h     = 0
        self.duration_s  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, path: str) -> bool:
        """
        Load a video file. Returns True if the file is readable, False otherwise.
        Safe to call while stopped.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False

        self._path      = path
        self.video_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames    = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.duration_s = total_frames / self.video_fps if self.video_fps else 0
        cap.release()
        return True

    def set_resolution(self, w: int, h: int):
        """Change simulated output resolution. Takes effect immediately."""
        self._target_w = w
        self._target_h = h

    def set_fps(self, fps: int):
        """Change simulated FPS. Takes effect on next frame."""
        self._target_fps = max(1, int(fps))

    def start(self):
        """Start the feeder thread. Does nothing if already running."""
        if self._running or not self._path:
            return
        self._running = True
        self._paused  = False
        self._thread  = threading.Thread(target=self._run, daemon=True, name="VideoFeeder")
        self._thread.start()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        """Signal the thread to stop and wait for it."""
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

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._running = False
            return

        fps_frame_count = 0
        fps_timer       = time.perf_counter()
        self.frame_count = 0

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            # Calculate sleep duration BEFORE reading so we stay in sync
            frame_duration = 1.0 / self._target_fps

            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                # End of file → loop back to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize to simulated camera resolution
            if frame.shape[1] != self._target_w or frame.shape[0] != self._target_h:
                frame = cv2.resize(
                    frame,
                    (self._target_w, self._target_h),
                    interpolation=cv2.INTER_AREA,
                )

            # Push to queue — drop if full (stay real-time, don't buffer up)
            try:
                self._q.put_nowait(frame)
                self.frame_count += 1
            except queue.Full:
                pass  # detector is slow — skip this frame

            # FPS measurement (update every second)
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                self.actual_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_timer       = time.perf_counter()

            # Pace to target FPS
            processing_time = time.perf_counter() - t_start
            sleep_for = frame_duration - processing_time
            if sleep_for > 0:
                time.sleep(sleep_for)

        cap.release()
        self._running = False
