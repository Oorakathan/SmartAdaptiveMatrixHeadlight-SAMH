"""
ai_detector.py
--------------
Object detection worker + IoU-based cross-frame tracker.

Key improvements over previous version
---------------------------------------
* IoU-based object tracker (ObjectTracker) assigns detections to persistent
  tracks across frames. Each track has a smoothed (cx_norm, cy_norm) position
  via EMA. This means:
    - The shadow glides smoothly behind a moving car, not jumps pixel-by-pixel.
    - Brief misdetections (1-3 frames) don't cause the shadow to vanish and
      reappear ("disco flash"). The track stays alive for TRACKER_MAX_LOST_FRAMES.
    - New ghost detections require TRACKER_MIN_HIT_STREAK hits before they
      activate the LED controller — eliminates single-frame false positives.

* annotate_frame is a PURE FUNCTION called only once in simulator._ui_poll.

* cy_norm mapping is physically correct:
    - Glare (car headlights): TOP of bounding box (y1).
    - Hazard (pedestrian/animal): CENTRE of bounding box.

Queue contract
--------------
  Consumes: frame_queue  ← np.ndarray BGR
  Produces: result_queue ← (raw_frame, confirmed_tracks)
            confirmed_tracks: list[dict] — only tracks with hit_streak >= MIN_HIT_STREAK
            Each dict has all keys of a detection dict plus:
              "track_id"  : int    — stable ID across frames
              "cx_norm"   : float  — EMA-smoothed horizontal centre
              "cy_norm"   : float  — EMA-smoothed vertical position
              "conf_ema"  : float  — EMA-smoothed confidence
              "lost"      : int    — frames since last matched detection (0 if active)
"""

import math
import queue
import threading
import time

import cv2
import numpy as np

import config

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# IoU TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: tuple, b: tuple) -> float:
    """Compute Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class _Track:
    """Single persistent object track."""
    _id_counter = 0

    def __init__(self, det: dict):
        _Track._id_counter += 1
        self.id         = _Track._id_counter
        self.category   = det["category"]
        self.label      = det["label"]

        # EMA-smoothed position and size
        self.cx_norm    = det["cx_norm"]
        self.cy_norm    = det["cy_norm"]
        self.w_norm     = det["w_norm"]
        self.h_norm     = det["h_norm"]

        # EMA confidence — starts at raw detection conf
        self.conf_ema   = det["conf"]

        # Last known raw bounding box (used for IoU matching)
        self.box        = det["box"]

        # Lifecycle counters
        self.hit_streak = 1      # consecutive frames matched
        self.lost       = 0      # frames since last match

    def update(self, det: dict):
        """Incorporate a new matched detection via EMA."""
        a_pos  = config.TRACK_POS_ALPHA
        a_size = config.TRACK_SIZE_ALPHA
        a_conf = config.CONF_EMA_ALPHA

        self.cx_norm  = a_pos  * det["cx_norm"] + (1 - a_pos)  * self.cx_norm
        self.cy_norm  = a_pos  * det["cy_norm"] + (1 - a_pos)  * self.cy_norm
        self.w_norm   = a_size * det["w_norm"]  + (1 - a_size) * self.w_norm
        self.h_norm   = a_size * det["h_norm"]  + (1 - a_size) * self.h_norm
        self.conf_ema = a_conf * det["conf"]    + (1 - a_conf) * self.conf_ema
        self.box      = det["box"]
        self.label    = det["label"]
        self.hit_streak += 1
        self.lost = 0

    def predict(self):
        """Called when no match found — track keeps previous position, loses confidence."""
        decay = 1.0 - config.CONF_EMA_ALPHA   # gentle confidence decay while lost
        self.conf_ema *= decay
        self.lost += 1
        self.hit_streak = max(0, self.hit_streak - 1)

    def to_det_dict(self) -> dict:
        """Export as a detection-compatible dict for the LED controller."""
        return {
            "track_id":  self.id,
            "label":     self.label,
            "conf":      self.conf_ema,
            "box":       self.box,
            "cx_norm":   self.cx_norm,
            "cy_norm":   self.cy_norm,
            "w_norm":    self.w_norm,
            "h_norm":    self.h_norm,
            "category":  self.category,
            "lost":      self.lost,
        }


class ObjectTracker:
    """
    Greedy IoU-based multi-object tracker.

    For each new frame of raw detections:
      1. Compute IoU between every existing track and every new detection.
      2. Greedily assign (highest IoU first) if IoU ≥ TRACKER_IOU_THRESHOLD.
      3. Unmatched tracks → predict() (age them, decay confidence).
      4. Unmatched detections → create new tracks.
      5. Prune tracks dead for > TRACKER_MAX_LOST_FRAMES.
      6. Return only "confirmed" tracks (hit_streak ≥ MIN_HIT_STREAK).

    Why greedy and not Hungarian algorithm?
      At typical headlamp scene density (2-5 vehicles), greedy is both fast
      enough and accurate enough. Hungarian adds O(n^3) complexity for no
      practical gain at this scale.
    """

    def __init__(self):
        self._tracks: list[_Track] = []

    def update(self, detections: list) -> list:
        """
        Update tracker with new detections. Returns list of confirmed track dicts.
        """
        # ── 1. Build IoU matrix ───────────────────────────────────────────────
        unmatched_dets   = list(range(len(detections)))
        matched_track_ids = set()

        if self._tracks and detections:
            # Score matrix: tracks × detections
            iou_matrix = [
                [_iou(t.box, d["box"]) for d in detections]
                for t in self._tracks
            ]

            # Greedy match: best IoU pairs first
            pairs = []
            for ti, row in enumerate(iou_matrix):
                for di, score in enumerate(row):
                    if score >= config.TRACKER_IOU_THRESHOLD:
                        pairs.append((score, ti, di))
            pairs.sort(reverse=True)

            used_tracks = set()
            used_dets   = set()
            for score, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                # Only match same category to prevent car→person swap
                if self._tracks[ti].category == detections[di]["category"]:
                    self._tracks[ti].update(detections[di])
                    matched_track_ids.add(ti)
                    used_tracks.add(ti)
                    used_dets.add(di)

            unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]

        # ── 2. Predict unmatched tracks ───────────────────────────────────────
        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.predict()

        # ── 3. Create new tracks for unmatched detections ─────────────────────
        for di in unmatched_dets:
            self._tracks.append(_Track(detections[di]))

        # ── 4. Prune dead tracks ──────────────────────────────────────────────
        self._tracks = [
            t for t in self._tracks
            if t.lost <= config.TRACKER_MAX_LOST_FRAMES
        ]

        # ── 5. Return confirmed tracks only ───────────────────────────────────
        return [
            t.to_det_dict()
            for t in self._tracks
            if t.hit_streak >= config.TRACKER_MIN_HIT_STREAK
        ]

    def reset(self):
        self._tracks.clear()


# ══════════════════════════════════════════════════════════════════════════════
# DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

class MockDetector:
    """Synthetic moving detections — no model required."""

    def __init__(self):
        self._t = 0.0

    def detect(self, frame: np.ndarray) -> list:
        self._t += 0.025
        h, w = frame.shape[:2]
        results = []

        cx = int((math.sin(self._t * 0.6) * 0.38 + 0.5) * w)
        cy = int(h * 0.40)
        hw, hh = int(w * 0.10), int(h * 0.12)
        results.append(_make_detection(
            "car", 0.88, (cx - hw, cy - hh, cx + hw, cy + hh), w, h))

        if math.sin(self._t * 0.4 + 2.1) > 0.55:
            cx2, cy2 = int(w * 0.75), int(h * 0.38)
            hw2, hh2 = int(w * 0.08), int(h * 0.10)
            results.append(_make_detection(
                "car", 0.72, (cx2 - hw2, cy2 - hh2, cx2 + hw2, cy2 + hh2), w, h))

        if math.sin(self._t * 0.28 + 1.5) > 0.45:
            px, py = int(w * 0.80), int(h * 0.55)
            results.append(_make_detection(
                "person", 0.76, (px - 28, py - 70, px + 28, py + 30), w, h))

        return results

    @staticmethod
    def name() -> str:
        return "Mock detector (install ultralytics for YOLO)"


class YOLODetector:
    """YOLOv8-nano via Ultralytics."""

    def __init__(self, model_name):
        self._model = _YOLO(model_name)
        self._names = self._model.names

    def detect(self, frame: np.ndarray) -> list:
        h, w = frame.shape[:2]
        res = self._model(frame, verbose=False, conf=config.CONFIDENCE_THRESHOLD)[0]
        out = []
        for box in res.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            out.append(_make_detection(self._names[cls], conf, (x1, y1, x2, y2), w, h))
        return out

    def name(self) -> str:
        return "YOLOv8-nano"


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION DICT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_detection(label: str, conf: float,
                    box: tuple, frame_w: int, frame_h: int) -> dict:
    x1, y1, x2, y2 = box
    cx_norm = ((x1 + x2) / 2.0) / frame_w
    is_glare = label in config.GLARE_CLASSES
    cy_norm  = (y1 / frame_h) if is_glare else ((y1 + y2) / 2.0) / frame_h
    cy_norm  = max(0.0, min(1.0, cy_norm))
    w_norm   = (x2 - x1) / frame_w
    h_norm   = (y2 - y1) / frame_h
    category = (
        "glare"  if label in config.GLARE_CLASSES  else
        "hazard" if label in config.HAZARD_CLASSES else
        "other"
    )
    return {
        "label": label, "conf": conf, "box": (x1, y1, x2, y2),
        "cx_norm": cx_norm, "cy_norm": cy_norm,
        "w_norm": w_norm, "h_norm": h_norm, "category": category,
    }


def _scale_boxes(detections: list, sx: float, sy: float) -> list:
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        out.append({**d, "box": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATE FRAME  (pure function — called ONCE in simulator._ui_poll)
# ══════════════════════════════════════════════════════════════════════════════

def annotate_frame(
    frame:        np.ndarray,
    detections:   list,
    led_states:   list,
    zone_states:  list,
    show_boxes:   bool = True,
    show_zones:   bool = True,
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]
    cols = config.ZONE_COUNT
    rows = config.ROW_LEDS

    if show_zones:
        seg_w = fw / cols
        seg_h = fh / rows

        for r in range(rows):
            for c in range(cols):
                state = led_states[r][c]
                if state == "suppress":
                    color, alpha = config.CV_COLOR_GLARE,  config.CV_ALPHA_GLARE
                elif state == "boost":
                    color, alpha = config.CV_COLOR_HAZARD, config.CV_ALPHA_HAZARD
                else:
                    continue

                x1 = int(c * seg_w); y1 = int(r * seg_h)
                x2 = int((c + 1) * seg_w); y2 = int((r + 1) * seg_h)
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        for c in range(cols):
            x_label = int((c + 0.5) * seg_w)
            cv2.putText(out, f"Z{c+1}", (x_label - 8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1, cv2.LINE_AA)
        for c in range(1, cols):
            cv2.line(out, (int(c * seg_w), 0), (int(c * seg_w), fh), (40, 60, 80), 1)
        for r in range(1, rows):
            cv2.line(out, (0, int(r * seg_h)), (fw, int(r * seg_h)), (40, 60, 80), 1)

    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cat   = det["category"]
            label = det["label"]
            conf  = det.get("conf_ema", det["conf"])
            tid   = det.get("track_id", "")
            lost  = det.get("lost", 0)

            if cat == "glare":
                color, tag = (50, 60, 220), "GLARE"
            elif cat == "hazard":
                color, tag = (0, 160, 240), "HAZARD"
            else:
                color, tag = (140, 140, 140), ""

            # Dim box if track is coasting (lost but still alive)
            if lost > 0:
                color = tuple(int(v * 0.5) for v in color)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag_str = f"[{tag}]" if tag else ""
            lost_str = f" ~{lost}" if lost > 0 else ""
            text = f"#{tid}{lost_str} {tag_str} {label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y1 - 4, th + 4)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(out, text, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# WORKER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AIDetector:
    """
    Pulls raw frames from frame_queue, runs detection + tracking, pushes
    (raw_frame, confirmed_tracks) to result_queue.

    The tracker runs here in the AI thread so it processes every detected frame.
    Only confirmed tracks (hit_streak ≥ MIN_HIT_STREAK) reach the LED controller,
    preventing ghost flashes from single-frame false positives.
    """

    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue):
        self._fq      = frame_queue
        self._rq      = result_queue
        self._running = False
        self._thread  = None

        self._detector = YOLODetector(config.MODEL_NAME) if _YOLO_AVAILABLE else MockDetector()
        self._tracker  = ObjectTracker()

        self.frames_processed = 0
        self.actual_fps       = 0.0

    @property
    def detector_name(self) -> str:
        return self._detector.name()

    def start(self):
        if self._running:
            return
        self._tracker.reset()
        self._running = True
        self._thread  = threading.Thread(
            target=self._run, daemon=True, name="AIDetector")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def _run(self):
        fps_count = 0
        fps_timer = time.perf_counter()

        while self._running:
            try:
                frame = self._fq.get(timeout=0.1)
            except queue.Empty:
                continue

            disp_h, disp_w = frame.shape[:2]

            if disp_w != config.AI_FRAME_W or disp_h != config.AI_FRAME_H:
                ai_frame = cv2.resize(
                    frame, (config.AI_FRAME_W, config.AI_FRAME_H),
                    interpolation=cv2.INTER_AREA)
            else:
                ai_frame = frame

            # Raw detections from model (at AI resolution)
            raw_dets = self._detector.detect(ai_frame)

            # Filter by minimum confidence per category before tracking
            filtered = [
                d for d in raw_dets
                if (d["category"] == "glare"  and d["conf"] >= config.MIN_CONFIDENCE_GLARE)
                or (d["category"] == "hazard" and d["conf"] >= config.MIN_CONFIDENCE_HAZARD)
                or (d["category"] == "other")
            ]

            # Update tracker → get smoothed, confirmed tracks
            confirmed_tracks = self._tracker.update(filtered)

            # Scale boxes back to display-frame pixel coords
            sx = disp_w / config.AI_FRAME_W
            sy = disp_h / config.AI_FRAME_H
            scaled_tracks = _scale_boxes(confirmed_tracks, sx, sy)

            try:
                self._rq.put_nowait((frame, scaled_tracks))
            except queue.Full:
                pass

            fps_count += 1
            self.frames_processed += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                self.actual_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.perf_counter()

        self._running = False
