"""
ai_detector.py
--------------
Object detection worker + IoU tracker + ROI horizon mask + scene density.

New in this version
-------------------
ROI horizon mask
  Applied immediately after YOLO inference, before the tracker sees anything.
  Any detection whose road-relevant y coordinate (y1_norm for glare = top of
  box = headlamp position; cy_norm for hazard = body centre) falls ABOVE
  config.ROI_HORIZON_RATIO is silently discarded.
  Side margins (config.ROI_SIDE_MARGIN) clip bonnet reflections and mirror
  artifacts at the frame edges.

  Physical justification:
    - A car at 50m appears in the bottom half of the frame in a dash-cam.
    - Signboards, traffic lights, flyovers appear in the top portion.
    - The road surface never extends above the horizon.
    - Therefore, anything above the horizon cannot be a vehicle or pedestrian
      ON THE ROAD — it's infrastructure or sky. Safe to discard.

Scene density estimator
  Counts confirmed glare tracks each frame, feeds an EMA.
  Published as self.density_ema — read by the UI to select drive mode.
  The LED controller then reads the current mode and applies beam shaping.

Queue contract
--------------
  Consumes: frame_queue  ← np.ndarray BGR
  Produces: result_queue ← (raw_frame, confirmed_tracks, density_ema)
            confirmed_tracks: list[dict] with smoothed position + conf_ema
            density_ema:      float — EMA vehicle count this frame
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
# ROI FILTER
# ══════════════════════════════════════════════════════════════════════════════

def _roi_filter(detections: list) -> list:
    """
    Discard detections that cannot physically be on the road.

    Rules applied (all normalised 0–1):
      1. Glare (car/truck/bus): y1_norm (TOP of box = headlamp) must be
         >= ROI_HORIZON_RATIO.  A headlamp below the horizon = real vehicle.
      2. Hazard (person/bike): cy_norm (body centre) must be
         >= ROI_HORIZON_RATIO.
      3. Any category: cx_norm must be within [ROI_SIDE_MARGIN, 1-ROI_SIDE_MARGIN].

    Why y1 for glare and cy for hazard?
      Headlamps sit at the TOP of the car body. A far car has a small box near
      the top of the frame — its headlamp (y1) is near the horizon.
      A pedestrian's centre is their torso — mid-box is representative.
    """
    horizon = config.ROI_HORIZON_RATIO
    side    = config.ROI_SIDE_MARGIN
    out = []
    for d in detections:
        cx = d["cx_norm"]
        # Side margin: discard bonnet / mirror reflections
        if cx < side or cx > 1.0 - side:
            continue

        cat = d["category"]
        if cat == "glare":
            # y1_norm = top of bounding box, normalised to frame height
            x1, y1, x2, y2 = d["box_norm"]
            if y1 < horizon:
                continue   # headlamp above horizon → not a real road vehicle
        elif cat == "hazard":
            if d["cy_norm"] < horizon:
                continue   # person above horizon → sign, banner, not a pedestrian
        # "other" category: apply cy_norm check too
        else:
            if d["cy_norm"] < horizon:
                continue

        out.append(d)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# IoU TRACKER  (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a: tuple, b: tuple) -> float:
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
    _id_counter = 0

    def __init__(self, det: dict):
        _Track._id_counter += 1
        self.id         = _Track._id_counter
        self.category   = det["category"]
        self.label      = det["label"]
        self.cx_norm    = det["cx_norm"]
        self.cy_norm    = det["cy_norm"]
        self.w_norm     = det["w_norm"]
        self.h_norm     = det["h_norm"]
        self.conf_ema   = det["conf"]
        self.box        = det["box"]
        self.hit_streak = 1
        self.lost       = 0

    def update(self, det: dict):
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
        self.conf_ema *= (1.0 - config.CONF_EMA_ALPHA)
        self.lost += 1
        self.hit_streak = max(0, self.hit_streak - 1)

    def to_det_dict(self) -> dict:
        return {
            "track_id": self.id, "label": self.label,
            "conf": self.conf_ema, "box": self.box,
            "cx_norm": self.cx_norm, "cy_norm": self.cy_norm,
            "w_norm": self.w_norm, "h_norm": self.h_norm,
            "category": self.category, "lost": self.lost,
        }


class ObjectTracker:
    def __init__(self):
        self._tracks: list[_Track] = []

    def update(self, detections: list) -> list:
        unmatched_dets    = list(range(len(detections)))
        matched_track_ids = set()

        if self._tracks and detections:
            iou_matrix = [
                [_iou(t.box, d["box"]) for d in detections]
                for t in self._tracks
            ]
            pairs = []
            for ti, row in enumerate(iou_matrix):
                for di, score in enumerate(row):
                    if score >= config.TRACKER_IOU_THRESHOLD:
                        pairs.append((score, ti, di))
            pairs.sort(reverse=True)

            used_tracks = set(); used_dets = set()
            for score, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                if self._tracks[ti].category == detections[di]["category"]:
                    self._tracks[ti].update(detections[di])
                    matched_track_ids.add(ti)
                    used_tracks.add(ti); used_dets.add(di)

            unmatched_dets = [di for di in range(len(detections)) if di not in used_dets]

        for ti, track in enumerate(self._tracks):
            if ti not in matched_track_ids:
                track.predict()

        for di in unmatched_dets:
            self._tracks.append(_Track(detections[di]))

        self._tracks = [t for t in self._tracks if t.lost <= config.TRACKER_MAX_LOST_FRAMES]

        return [t.to_det_dict() for t in self._tracks if t.hit_streak >= config.TRACKER_MIN_HIT_STREAK]

    def reset(self):
        self._tracks.clear()


# ══════════════════════════════════════════════════════════════════════════════
# DETECTORS
# ══════════════════════════════════════════════════════════════════════════════

class MockDetector:
    def __init__(self):
        self._t = 0.0

    def detect(self, frame: np.ndarray) -> list:
        self._t += 0.025
        h, w = frame.shape[:2]
        results = []
        cx = int((math.sin(self._t * 0.6) * 0.38 + 0.5) * w)
        cy = int(h * 0.60)   # well below horizon
        hw, hh = int(w * 0.10), int(h * 0.12)
        results.append(_make_detection("car", 0.88, (cx-hw, cy-hh, cx+hw, cy+hh), w, h))
        if math.sin(self._t * 0.4 + 2.1) > 0.55:
            cx2, cy2 = int(w * 0.75), int(h * 0.65)
            hw2, hh2 = int(w * 0.08), int(h * 0.10)
            results.append(_make_detection("car", 0.72, (cx2-hw2, cy2-hh2, cx2+hw2, cy2+hh2), w, h))
        if math.sin(self._t * 0.28 + 1.5) > 0.45:
            px, py = int(w * 0.80), int(h * 0.70)
            results.append(_make_detection("person", 0.76, (px-28, py-70, px+28, py+30), w, h))
        return results

    @staticmethod
    def name() -> str:
        return "Mock detector (install ultralytics for YOLO)"


class YOLODetector:
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

def _make_detection(label: str, conf: float, box: tuple, frame_w: int, frame_h: int) -> dict:
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
        "label": label, "conf": conf,
        "box": (x1, y1, x2, y2),
        # Normalised box corners — used by ROI filter (avoids re-dividing)
        "box_norm": (x1/frame_w, y1/frame_h, x2/frame_w, y2/frame_h),
        "cx_norm": cx_norm, "cy_norm": cy_norm,
        "w_norm": w_norm, "h_norm": h_norm,
        "category": category,
    }


def _scale_boxes(detections: list, sx: float, sy: float) -> list:
    out = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        out.append({**d, "box": (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ANNOTATE FRAME
# ══════════════════════════════════════════════════════════════════════════════

def annotate_frame(
    frame:       np.ndarray,
    detections:  list,
    led_states:  list,
    zone_states: list,
    drive_mode:  str  = "highway",
    show_boxes:  bool = True,
    show_zones:  bool = True,
    show_horizon:bool = True,
) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]
    cols = config.ZONE_COUNT
    rows = config.ROW_LEDS

    # ── Horizon line ──────────────────────────────────────────────────────────
    if show_horizon:
        hy = int(config.ROI_HORIZON_RATIO * fh)
        cv2.line(out, (0, hy), (fw, hy), config.CV_COLOR_HORIZON, 1)
        cv2.putText(out, f"ROI horizon ({config.ROI_HORIZON_RATIO:.0%})",
                    (6, hy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    config.CV_COLOR_HORIZON, 1, cv2.LINE_AA)

        # Side margin lines
        lx = int(config.ROI_SIDE_MARGIN * fw)
        rx = int((1 - config.ROI_SIDE_MARGIN) * fw)
        cv2.line(out, (lx, hy), (lx, fh), config.CV_COLOR_HORIZON, 1)
        cv2.line(out, (rx, hy), (rx, fh), config.CV_COLOR_HORIZON, 1)

    # ── Zone overlay ──────────────────────────────────────────────────────────
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
                x2 = int((c+1) * seg_w); y2 = int((r+1) * seg_h)
                overlay = out.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, alpha, out, 1-alpha, 0, out)

        for c in range(cols):
            x_label = int((c+0.5)*seg_w)
            cv2.putText(out, f"Z{c+1}", (x_label-8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190,190,190), 1, cv2.LINE_AA)
        for c in range(1, cols):
            cv2.line(out, (int(c*seg_w),0), (int(c*seg_w),fh), (40,60,80), 1)
        for r in range(1, rows):
            cv2.line(out, (0,int(r*seg_h)), (fw,int(r*seg_h)), (40,60,80), 1)

    # ── Drive mode badge ──────────────────────────────────────────────────────
    mode_colors = {
        config.DRIVE_MODE_HIGHWAY:    (0, 200, 170),
        config.DRIVE_MODE_EXPRESSWAY: (58, 142, 246),
        config.DRIVE_MODE_CITY:       (245, 158, 11),
    }
    mc = mode_colors.get(drive_mode, (150, 150, 150))
    badge = f" {drive_mode.upper()} MODE "
    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(out, (fw - bw - 14, 4), (fw - 4, bh + 10), mc, -1)
    cv2.putText(out, badge, (fw - bw - 10, bh + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 15, 30), 1, cv2.LINE_AA)

    # ── Bounding boxes ────────────────────────────────────────────────────────
    if show_boxes:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cat   = det["category"]
            label = det["label"]
            conf  = det.get("conf", 0)
            tid   = det.get("track_id", "")
            lost  = det.get("lost", 0)

            if cat == "glare":
                color, tag = (50, 60, 220), "GLARE"
            elif cat == "hazard":
                color, tag = (0, 160, 240), "HAZARD"
            else:
                color, tag = (140, 140, 140), ""

            if lost > 0:
                color = tuple(int(v * 0.5) for v in color)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag_str  = f"[{tag}]" if tag else ""
            lost_str = f" ~{lost}" if lost > 0 else ""
            text = f"#{tid}{lost_str} {tag_str} {label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y1-4, th+4)
            cv2.rectangle(out, (x1, ty-th-4), (x1+tw+4, ty), color, -1)
            cv2.putText(out, text, (x1+2, ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240,240,240), 1, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# WORKER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class AIDetector:
    """
    Pipeline: raw frame → YOLO → ROI filter → confidence filter → tracker
              → confirmed tracks + density EMA → result_queue
    """

    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue):
        self._fq      = frame_queue
        self._rq      = result_queue
        self._running = False
        self._thread  = None

        self._detector   = YOLODetector(config.MODEL_NAME) if _YOLO_AVAILABLE else MockDetector()
        self._tracker    = ObjectTracker()

        # Scene density EMA — read by UI thread for mode switching
        self.density_ema      = 0.0
        self.frames_processed = 0
        self.actual_fps       = 0.0

    @property
    def detector_name(self) -> str:
        return self._detector.name()

    def start(self):
        if self._running:
            return
        self._tracker.reset()
        self.density_ema = 0.0
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

    def _run(self):
        fps_count = 0
        fps_timer = time.perf_counter()

        while self._running:
            try:
                frame = self._fq.get(timeout=0.1)
            except queue.Empty:
                continue

            disp_h, disp_w = frame.shape[:2]

            # Resize to AI resolution
            if disp_w != config.AI_FRAME_W or disp_h != config.AI_FRAME_H:
                ai_frame = cv2.resize(frame, (config.AI_FRAME_W, config.AI_FRAME_H),
                                      interpolation=cv2.INTER_AREA)
            else:
                ai_frame = frame

            # 1. Raw detections from model
            raw_dets = self._detector.detect(ai_frame)

            # 2. Confidence filter
            conf_filtered = [
                d for d in raw_dets
                if (d["category"] == "glare"  and d["conf"] >= config.MIN_CONFIDENCE_GLARE)
                or (d["category"] == "hazard" and d["conf"] >= config.MIN_CONFIDENCE_HAZARD)
                or (d["category"] == "other")
            ]

            # 3. ROI horizon mask — discard sky / signboard detections
            roi_filtered = _roi_filter(conf_filtered)

            # 4. Tracker → smoothed confirmed tracks
            confirmed = self._tracker.update(roi_filtered)

            # 5. Scene density EMA (glare tracks only = vehicles)
            glare_count = sum(1 for t in confirmed if t["category"] == "glare")
            alpha = config.DENSITY_EMA_ALPHA
            self.density_ema = alpha * glare_count + (1 - alpha) * self.density_ema

            # 6. Scale boxes to display resolution
            sx = disp_w / config.AI_FRAME_W
            sy = disp_h / config.AI_FRAME_H
            scaled = _scale_boxes(confirmed, sx, sy)

            try:
                self._rq.put_nowait((frame, scaled, self.density_ema))
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