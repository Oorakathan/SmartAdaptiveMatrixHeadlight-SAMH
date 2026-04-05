"""
ai_detector.py
--------------
Object detection worker.

Key fixes vs previous version
------------------------------
* annotate_frame is a PURE FUNCTION called only once — in simulator._ui_poll.
  The AIDetector._run() thread no longer annotates anything. It just pushes
  (raw_frame, detections). The UI annotates after the controller has updated
  zone states, so overlays are always in sync with the LED matrix.

* cy_norm mapping is physically correct:
    - Glare (car headlights): use TOP of bounding box (y1). Headlights sit
      at the TOP of the oncoming vehicle body.  A far car has a small box
      near the top of the frame → y1/frame_h ≈ 0.1–0.3 → row 0–2 (far rows).
    - Hazard (pedestrian/animal): use CENTRE of bounding box. A person
      standing close fills the lower frame → cy ≈ 0.5–0.8 → mid/near rows.

* _make_detection() now also returns bounding-box dimensions (w_norm, h_norm)
  normalised by frame size — useful for future size-based distance estimation.

Queue contract
--------------
  Consumes: frame_queue  ← np.ndarray BGR (simulated camera resolution)
  Produces: result_queue ← (raw_frame, detections)
            raw_frame  : np.ndarray — unmodified frame for UI to annotate
            detections : list[dict]
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

        # Car sweeping left → right
        cx = int((math.sin(self._t * 0.6) * 0.38 + 0.5) * w)
        cy = int(h * 0.40)
        hw, hh = int(w * 0.10), int(h * 0.12)
        results.append(_make_detection(
            "car", 0.88,
            (cx - hw, cy - hh, cx + hw, cy + hh),
            w, h,
        ))

        # Second car occasionally on the other side
        if math.sin(self._t * 0.4 + 2.1) > 0.55:
            cx2 = int(w * 0.75)
            cy2 = int(h * 0.38)
            hw2, hh2 = int(w * 0.08), int(h * 0.10)
            results.append(_make_detection(
                "car", 0.72,
                (cx2 - hw2, cy2 - hh2, cx2 + hw2, cy2 + hh2),
                w, h,
            ))

        # Pedestrian on the right, intermittent
        if math.sin(self._t * 0.28 + 1.5) > 0.45:
            px = int(w * 0.80)
            py = int(h * 0.55)
            results.append(_make_detection(
                "person", 0.76,
                (px - 28, py - 70, px + 28, py + 30),
                w, h,
            ))

        return results

    @staticmethod
    def name() -> str:
        return "Mock detector (install ultralytics for YOLO)"


class YOLODetector:
    """YOLOv8-nano via Ultralytics. Model auto-downloads (~6 MB) on first use."""

    def __init__(self, model_name: str = "yolov8n.pt"):
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
            out.append(_make_detection(
                self._names[cls], conf,
                (x1, y1, x2, y2), w, h,
            ))
        return out

    def name(self) -> str:
        return "YOLOv8-nano"


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION DICT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def _make_detection(label: str, conf: float,
                    box: tuple, frame_w: int, frame_h: int) -> dict:
    """
    Build a normalised detection dict.

    cx_norm  — horizontal centre, 0 = left edge, 1 = right edge
    cy_norm  — vertical position used for row mapping:
                 glare → top of box  (headlights)
                 hazard → centre of box (body)
    w_norm, h_norm — box size as fraction of frame (proxy for distance)
    """
    x1, y1, x2, y2 = box

    cx_norm = ((x1 + x2) / 2.0) / frame_w

    is_glare = label in config.GLARE_CLASSES
    if is_glare:
        # Top of box = where headlights physically are on an oncoming vehicle
        cy_norm = y1 / frame_h
    else:
        # Centre for pedestrians / animals
        cy_norm = ((y1 + y2) / 2.0) / frame_h

    cy_norm = max(0.0, min(1.0, cy_norm))

    w_norm = (x2 - x1) / frame_w   # larger = closer (rough distance proxy)
    h_norm = (y2 - y1) / frame_h

    category = (
        "glare"  if label in config.GLARE_CLASSES  else
        "hazard" if label in config.HAZARD_CLASSES else
        "other"
    )

    return {
        "label":    label,
        "conf":     conf,
        "box":      (x1, y1, x2, y2),
        "cx_norm":  cx_norm,
        "cy_norm":  cy_norm,
        "w_norm":   w_norm,
        "h_norm":   h_norm,
        "category": category,
    }


def _scale_boxes(detections: list, sx: float, sy: float) -> list:
    """Scale bounding boxes from AI-resolution coords to display-resolution coords."""
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
    led_states:   list,         # ROW_LEDS × ZONE_COUNT strings
    zone_states:  list,         # ZONE_COUNT strings (for column labels)
    show_boxes:   bool = True,
    show_zones:   bool = True,
) -> np.ndarray:
    """
    Draw the LED grid overlay and bounding boxes onto a COPY of frame.

    The overlay uses the led_states grid so each cell of the video frame
    (segment = frame_w/ZONE_COUNT × frame_h/ROW_LEDS) reflects the true
    per-LED state — not just a column-wide band.

    overlay colour:
      suppress → red tint (glare shadow)
      boost    → amber tint (hazard alert)
      on       → faint teal tint (safe beam active)
    """
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
                    # Skip full-beam cells to keep overlay light and clear
                    continue

                x1 = int(c * seg_w)
                y1 = int(r * seg_h)
                x2 = int((c + 1) * seg_w)
                y2 = int((r + 1) * seg_h)

                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Column zone labels at top (Z1 … ZN)
        for c in range(cols):
            x_label = int((c + 0.5) * seg_w)
            cv2.putText(out, f"Z{c+1}", (x_label - 8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 190, 190), 1, cv2.LINE_AA)

        # Thin grid lines to show segment boundaries
        for c in range(1, cols):
            x = int(c * seg_w)
            cv2.line(out, (x, 0), (x, fh), (40, 60, 80), 1)
        for r in range(1, rows):
            y = int(r * seg_h)
            cv2.line(out, (0, y), (fw, y), (40, 60, 80), 1)

    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cat   = det["category"]
            label = det["label"]
            conf  = det["conf"]

            if cat == "glare":
                color, tag = (50, 60, 220), "GLARE"
            elif cat == "hazard":
                color, tag = (0, 160, 240), "HAZARD"
            else:
                color, tag = (140, 140, 140), ""

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"[{tag}] {label} {conf:.0%}" if tag else f"{label} {conf:.0%}"
            # Text background for readability
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
    Pulls raw frames from frame_queue, runs detection, pushes
    (raw_frame, detections) to result_queue.

    DOES NOT annotate — annotation happens in the UI thread after
    the LED controller has updated, so overlays are always correct.
    """

    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue):
        self._fq      = frame_queue
        self._rq      = result_queue
        self._running = False
        self._thread  = None

        self._detector = YOLODetector() if _YOLO_AVAILABLE else MockDetector()

        # Stats
        self.frames_processed = 0
        self.actual_fps       = 0.0

    @property
    def detector_name(self) -> str:
        return self._detector.name()

    def start(self):
        if self._running:
            return
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

            # Scale to AI resolution for inference
            if disp_w != config.AI_FRAME_W or disp_h != config.AI_FRAME_H:
                ai_frame = cv2.resize(
                    frame, (config.AI_FRAME_W, config.AI_FRAME_H),
                    interpolation=cv2.INTER_AREA)
            else:
                ai_frame = frame

            # Run detector on AI-resolution frame
            raw_dets = self._detector.detect(ai_frame)

            # Scale boxes back to display-frame pixel coords
            sx = disp_w / config.AI_FRAME_W
            sy = disp_h / config.AI_FRAME_H
            detections = _scale_boxes(raw_dets, sx, sy)

            # Push raw frame + detections — annotation happens in UI thread
            try:
                self._rq.put_nowait((frame, detections))
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
