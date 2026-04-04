"""
ai_detector.py
--------------
Runs object detection on incoming video frames and pushes results
onto the shared `result_queue`.

Responsibilities:
  - Scale frames to AI_FRAME_W × AI_FRAME_H for consistent zone mapping
  - Run YOLOv8-nano (or MockDetector if ultralytics is not installed)
  - Scale bounding boxes back to the original display frame size
  - Package (annotated_frame, detections) and push to result_queue
  - Does NOT update the LED matrix — that is led_controller's job
  - Does NOT touch the UI

Queue contract:
  Consumes from: frame_queue
    frame : np.ndarray  BGR image at simulated camera resolution

  Pushes to: result_queue
    (annotated_frame, detections, raw_frame)
    annotated_frame : np.ndarray  BGR image with overlays drawn
    detections      : list[Detection]  see Detection dataclass below
    raw_frame       : np.ndarray  original unmodified frame

Detection dict keys:
    label    : str    YOLO class name  e.g. "car", "person"
    conf     : float  confidence 0.0–1.0
    box      : tuple  (x1, y1, x2, y2) in display-frame pixels
    cx_norm  : float  normalised centre-x 0.0–1.0 (used for zone mapping)
    category : str    "glare" | "hazard" | "other"
"""

import math
import queue
import threading
import time

import cv2
import numpy as np

import config

# ── Optional YOLO import ──────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# ── Detectors ─────────────────────────────────────────────────────────────────

class MockDetector:
    """
    Generates synthetic moving detections — no model required.
    Good for demoing the full pipeline without any video content.

    Produces:
      - A car sweeping left → right across the frame
      - An intermittent pedestrian on the right third
    """

    def __init__(self):
        self._t = 0.0

    def detect(self, frame: np.ndarray) -> list:
        self._t += 0.03
        h, w = frame.shape[:2]
        results = []

        # Car sweeps left → right with a sine wave
        cx = int((math.sin(self._t * 0.7) * 0.4 + 0.5) * w)
        cy = int(h * 0.45)
        hw = int(w * 0.12)
        hh = int(h * 0.14)
        results.append(_make_detection(
            label="car", conf=0.88,
            box=(cx - hw, cy - hh, cx + hw, cy + hh),
            frame_w=w, frame_h=h,
        ))

        # Pedestrian appears on the right intermittently
        if math.sin(self._t * 0.3 + 1.5) > 0.4:
            px = int(w * 0.78)
            py = int(h * 0.50)
            results.append(_make_detection(
                label="person", conf=0.75,
                box=(px - 30, py - 60, px + 30, py + 60),
                frame_w=w, frame_h=h,
            ))

        return results

    @staticmethod
    def name() -> str:
        return "Mock detector (ultralytics not installed)"


class YOLODetector:
    """
    Real YOLOv8-nano detector via Ultralytics.
    Model file (~6 MB) is downloaded on first use.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self._model = _YOLO(model_name)
        self._names = self._model.names
        self._model_name = model_name

    def detect(self, frame: np.ndarray) -> list:
        results = self._model(
            frame,
            verbose=False,
            conf=config.CONFIDENCE_THRESHOLD,
        )[0]

        detections = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = self._names[cls]
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append(_make_detection(
                label=label, conf=conf,
                box=(x1, y1, x2, y2),
                frame_w=w, frame_h=h,
            ))
        return detections

    def name(self) -> str:
        return f"YOLOv8-nano ({self._model_name})"


def _make_detection(label: str, conf: float, box: tuple, frame_w: int, frame_h: int = None) -> dict:
    """Build a normalised detection dict."""
    x1, y1, x2, y2 = box
    cx_norm = ((x1 + x2) / 2) / frame_w
    
    # Calculate normalized y-coordinate (0.0 = top, 1.0 = bottom)
    cy_norm = 0.5  # default center
    if frame_h is not None and frame_h > 0:
        cy_norm = ((y1 + y2) / 2) / frame_h

    if label in config.GLARE_CLASSES:
        category = "glare"
    elif label in config.HAZARD_CLASSES:
        category = "hazard"
    else:
        category = "other"

    return {
        "label":    label,
        "conf":     conf,
        "box":      (x1, y1, x2, y2),
        "cx_norm":  cx_norm,
        "cy_norm":  cy_norm,
        "category": category,
    }


def _scale_boxes(detections: list, scale_x: float, scale_y: float) -> list:
    """Return a new list with boxes scaled from AI resolution → display resolution."""
    scaled = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        scaled.append({
            **d,
            "box": (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ),
        })
    return scaled


# ── Annotator (pure function — no state) ─────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: list,
    zone_states: list,       # list of 8 strings: "on" | "suppress" | "boost" (backward compat)
    led_states: list = None, # 8x8 matrix of LED states (optional, for individual LED overlay)
    show_boxes: bool = True,
    show_zones: bool = True,
) -> np.ndarray:
    """
    Draw zone overlays and bounding boxes onto `frame`.
    If led_states is provided, draws individual LED regions instead of full columns.
    Returns a new annotated copy; the original is not modified.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    if show_zones:
        if led_states is not None:
            # Individual LED region overlay (8x8 grid)
            zone_w = w // config.ZONE_COUNT
            row_h = h // 8
            
            for row in range(8):
                for col in range(config.ZONE_COUNT):
                    state = led_states[row][col]
                    
                    if state == "suppress":
                        color = config.CV_COLOR_GLARE
                        alpha = config.CV_ALPHA_GLARE
                    elif state == "boost":
                        color = config.CV_COLOR_HAZARD
                        alpha = config.CV_ALPHA_HAZARD
                    else:
                        # Don't draw "on" state to keep overlay clean
                        continue
                    
                    # Calculate region bounds
                    x1_z = col * zone_w
                    x2_z = x1_z + zone_w
                    y1_z = row * row_h
                    y2_z = y1_z + row_h
                    
                    overlay = out.copy()
                    cv2.rectangle(overlay, (x1_z, y1_z), (x2_z, y2_z), color, -1)
                    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
            
            # Zone numbers at the top
            zone_w = w // config.ZONE_COUNT
            for col in range(config.ZONE_COUNT):
                x1_z = col * zone_w
                cv2.putText(
                    out, f"Z{col + 1}",
                    (x1_z + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )
        else:
            # Fallback: column-based overlay (backward compatible)
            zone_w = w // config.ZONE_COUNT
            for col, state in enumerate(zone_states):
                if state == "suppress":
                    color = config.CV_COLOR_GLARE
                    alpha = config.CV_ALPHA_GLARE
                elif state == "boost":
                    color = config.CV_COLOR_HAZARD
                    alpha = config.CV_ALPHA_HAZARD
                else:
                    color = config.CV_COLOR_SAFE
                    alpha = config.CV_ALPHA_SAFE

                x1_z = col * zone_w
                x2_z = x1_z + zone_w
                overlay = out.copy()
                cv2.rectangle(overlay, (x1_z, 0), (x2_z, h), color, -1)
                cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

                # Zone number at the top of each column
                cv2.putText(
                    out, f"Z{col + 1}",
                    (x1_z + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )

    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cat  = det["category"]
            conf = det["conf"]
            label = det["label"]

            if cat == "glare":
                color = (60,  60, 220)
                tag   = "GLARE"
            elif cat == "hazard":
                color = (0,  160, 240)
                tag   = "HAZARD"
            else:
                color = (160, 160, 160)
                tag   = ""

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"[{tag}] {label} {conf:.0%}" if tag else f"{label} {conf:.0%}"
            cv2.putText(
                out, text,
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA,
            )

    return out


# ── Worker class ──────────────────────────────────────────────────────────────

class AIDetector:
    """
    Pulls frames from `frame_queue`, runs detection, pushes annotated
    results to `result_queue`.

    The LED controller and UI read from `result_queue` independently.

    Usage:
        detector = AIDetector(frame_queue, result_queue)
        detector.start()
        ...
        detector.stop()
    """

    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue):
        self._fq      = frame_queue
        self._rq      = result_queue
        self._running = False
        self._thread  = None

        # Display toggles — set by UI, read by worker
        self.show_boxes = True
        self.show_zones = True

        # Current zone states — updated by led_controller, read here for annotation
        # This is a simple shared list; the controller writes, we read (no lock needed
        # since Python list reads/writes on a single reference are GIL-atomic at this size)
        self.zone_states = ["on"] * config.ZONE_COUNT
        # LED states matrix (8x8) — also updated by led_controller for individual LED overlay
        self.led_states = [["on"] * config.ZONE_COUNT for _ in range(8)]

        # Build detector
        if _YOLO_AVAILABLE:
            self._detector = YOLODetector()
        else:
            self._detector = MockDetector()

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
        self._thread  = threading.Thread(target=self._run, daemon=True, name="AIDetector")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run(self):
        fps_count = 0
        fps_timer = time.perf_counter()

        while self._running:
            # Get next frame
            try:
                frame = self._fq.get(timeout=0.1)
            except queue.Empty:
                continue

            disp_h, disp_w = frame.shape[:2]

            # Scale down to AI resolution for inference
            if disp_w != config.AI_FRAME_W or disp_h != config.AI_FRAME_H:
                ai_frame = cv2.resize(
                    frame,
                    (config.AI_FRAME_W, config.AI_FRAME_H),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                ai_frame = frame

            # Run detection on AI-resolution frame
            raw_dets = self._detector.detect(ai_frame)

            # Scale boxes back to display frame size
            sx = disp_w / config.AI_FRAME_W
            sy = disp_h / config.AI_FRAME_H
            detections = _scale_boxes(raw_dets, sx, sy)

            # Annotate a copy of the display frame
            annotated = annotate_frame(
                frame, detections,
                zone_states=self.zone_states,
                led_states=self.led_states,
                show_boxes=self.show_boxes,
                show_zones=self.show_zones,
            )

            # Push result — drop if consumer is slow
            try:
                self._rq.put_nowait((annotated, detections, frame))
            except queue.Full:
                pass

            # Stats
            fps_count += 1
            self.frames_processed += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                self.actual_fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.perf_counter()

        self._running = False
