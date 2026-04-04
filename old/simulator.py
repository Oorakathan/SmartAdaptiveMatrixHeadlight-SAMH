"""
Smart Adaptive Matrix Headlamp — Desktop Simulator
====================================================
Simulates ESP32-CAM video feed → YOLO detection → 8×8 LED beam matrix

Usage:
    python main.py

Requirements:
    pip install opencv-python ultralytics numpy pillow
    tkinter  (bundled with standard Python installs)

Controls:
    - Upload any video file (simulates ESP32-CAM stream)
    - Adjust resolution & FPS to simulate different ESP32-CAM configs
    - Watch the 8×8 LED matrix respond in real time
"""

import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
import time
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Try importing YOLO; fall back to mock detector if not installed ──────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
ZONE_COUNT   = 8          # number of horizontal angular zones
FRAME_W      = 640        # base processing width
FRAME_H      = 360        # base processing height

# Classes that trigger zone suppression (glare = oncoming vehicle)
GLARE_CLASSES  = {"car", "truck", "motorcycle", "bus"}
# Classes that trigger zone boost/flash (hazard alert)
HAZARD_CLASSES = {"person", "dog", "cat", "bird", "horse", "cow", "sheep"}

# Colour palette (hex strings)
C_BG        = "#0A0F1E"
C_PANEL     = "#111827"
C_CARD      = "#1a2235"
C_BORDER    = "#1f2d44"
C_ACCENT    = "#00C2A8"
C_ACCENT2   = "#3A8EF6"
C_HAZARD    = "#F59E0B"
C_SUPPRESS  = "#EF4444"
C_ON        = "#00C2A8"   # LED on (full beam)
C_DIM       = "#EF4444"   # LED suppressed (shadow zone)
C_BOOST     = "#F59E0B"   # LED boosted (hazard zone)
C_OFF       = "#0d1a2e"   # LED off
C_TEXT      = "#E2E8F0"
C_MUTED     = "#64748B"

# ── Utility ──────────────────────────────────────────────────────────────────
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def interpolate_color(c1, c2, t):
    r1,g1,b1 = hex_to_rgb(c1)
    r2,g2,b2 = hex_to_rgb(c2)
    r = int(r1 + (r2-r1)*t)
    g = int(g1 + (g2-g1)*t)
    b = int(b1 + (b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"

# ── Mock detector (when ultralytics not installed) ───────────────────────────
class MockDetector:
    """Returns synthetic detections that move across the frame — good for demo."""
    def __init__(self):
        self._t = 0.0

    def detect(self, frame):
        self._t += 0.03
        w = frame.shape[1]
        h = frame.shape[0]
        detections = []
        # Simulate car sweeping left→right
        cx = int((math.sin(self._t * 0.7) * 0.4 + 0.5) * w)
        cy = int(h * 0.45)
        hw, hh = int(w*0.12), int(h*0.14)
        detections.append({
            "label": "car",
            "conf": 0.88,
            "box": (cx-hw, cy-hh, cx+hw, cy+hh),
            "cx_norm": cx / w,
        })
        # Simulate pedestrian on right side occasionally
        if math.sin(self._t * 0.3 + 1.5) > 0.4:
            px = int(w * 0.78)
            py = int(h * 0.5)
            detections.append({
                "label": "person",
                "conf": 0.75,
                "box": (px-30, py-60, px+30, py+60),
                "cx_norm": px / w,
            })
        return detections

# ── Real YOLO detector ────────────────────────────────────────────────────────
class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self._names = self.model.names

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=0.40)[0]
        detections = []
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = self._names[cls]
            conf  = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            cx_norm = ((x1+x2)/2) / frame.shape[1]
            detections.append({
                "label": label,
                "conf":  conf,
                "box":   (x1,y1,x2,y2),
                "cx_norm": cx_norm,
            })
        return detections

# ── Zone matrix logic ─────────────────────────────────────────────────────────
class ZoneMatrix:
    """
    Converts detections → per-zone state for the 8×8 LED matrix.

    Zone states (per column 0–7):
        "on"       — full beam, no object in this zone
        "suppress" — shadow zone (glare vehicle present)
        "boost"    — hazard pre-alert (animal/pedestrian)
    
    Each zone column drives all 8 LED rows (= one distance band).
    Confidence-weighted dimming: adjacent zones to shadow are dimmed 30%.
    """

    STATES = ("on", "suppress", "boost")

    def __init__(self):
        self.zone_states   = ["on"] * ZONE_COUNT
        self.zone_conf     = [1.0]  * ZONE_COUNT   # 0.0–1.0 brightness
        self._flash_tick   = 0
        self._flash_on     = True

    def update(self, detections, frame_w):
        states = ["on"]   * ZONE_COUNT
        confs  = [1.0]    * ZONE_COUNT

        for det in detections:
            cx   = det["cx_norm"]
            conf = det["conf"]
            col  = int(cx * ZONE_COUNT)
            col  = max(0, min(ZONE_COUNT-1, col))

            if det["label"] in GLARE_CLASSES:
                # Suppress the column + ±1 adjacent with graduated dimming
                for dc, alpha in [(-1, 0.35), (0, 0.0), (1, 0.35)]:
                    c = col + dc
                    if 0 <= c < ZONE_COUNT:
                        if states[c] != "suppress":
                            states[c] = "suppress"
                        confs[c] = min(confs[c], alpha)

            elif det["label"] in HAZARD_CLASSES:
                states[col] = "boost"
                confs[col]  = 1.0

        self.zone_states = states
        self.zone_conf   = confs

    def tick_flash(self):
        """Call at ~4 Hz to animate hazard flash."""
        self._flash_tick += 1
        self._flash_on    = (self._flash_tick % 2 == 0)

    def led_color(self, col, row):
        """
        Returns the LED colour for a given column (zone) and row (distance band).
        Row 0 = far (100m+), Row 7 = close (flood).
        """
        state = self.zone_states[col]
        conf  = self.zone_conf[col]

        if state == "suppress":
            # Row 7 (wide flood) always stays on even in shadow zones
            if row == 7:
                return C_DIM, 0.3
            return C_DIM, conf   # red, low brightness
        elif state == "boost":
            if self._flash_on:
                return C_BOOST, 1.0
            else:
                return C_ON, 0.3   # dim during off-phase of flash
        else:
            # Normal beam — rows closer to far (row 0) are slightly brighter
            brightness = 0.7 + 0.3 * (1 - row / 7)
            return C_ON, brightness * conf

# ── Main Application ──────────────────────────────────────────────────────────
class SmartHeadlampApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Adaptive Matrix Headlamp — Simulator")
        self.root.configure(bg=C_BG)
        self.root.minsize(1100, 680)

        # State
        self.cap           = None
        self.video_path    = None
        self.running       = False
        self.paused        = False
        self.frame_queue   = queue.Queue(maxsize=2)
        self.det_queue     = queue.Queue(maxsize=2)

        self.sim_res       = tk.StringVar(value="640×360 (ESP32-CAM high)")
        self.sim_fps       = tk.IntVar(value=15)
        self.show_boxes    = tk.BooleanVar(value=True)
        self.show_zones    = tk.BooleanVar(value=True)

        self.zone_matrix   = ZoneMatrix()
        self.detector      = YOLODetector() if YOLO_AVAILABLE else MockDetector()
        self.detector_type = "YOLOv8-nano" if YOLO_AVAILABLE else "Mock (install ultralytics for real AI)"

        self._last_detections = []
        self._frame_count  = 0
        self._fps_actual   = 0.0
        self._fps_timer    = time.time()
        self._flash_timer  = None

        self._build_ui()
        self._start_flash_ticker()

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── Top bar
        topbar = tk.Frame(root, bg=C_PANEL, height=48)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="SMART ADAPTIVE MATRIX HEADLAMP", bg=C_PANEL,
                 fg=C_ACCENT, font=("Courier New", 13, "bold")).pack(side=tk.LEFT, padx=16, pady=12)
        tk.Label(topbar, text=f"AI: {self.detector_type}", bg=C_PANEL,
                 fg=C_MUTED, font=("Courier New", 9)).pack(side=tk.LEFT, padx=8)

        # Model status pill
        pill_color = C_ACCENT if YOLO_AVAILABLE else C_HAZARD
        pill_text  = "YOLO LIVE" if YOLO_AVAILABLE else "MOCK MODE"
        tk.Label(topbar, text=f"  {pill_text}  ", bg=pill_color,
                 fg=C_BG, font=("Courier New", 9, "bold")).pack(side=tk.LEFT, padx=4)

        # ── Main 3-column layout
        body = tk.Frame(root, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Left panel — controls
        left = tk.Frame(body, bg=C_PANEL, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        # Centre — video feed
        centre = tk.Frame(body, bg=C_BG)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
        self._build_video_panel(centre)

        # Right — 8×8 matrix
        right = tk.Frame(body, bg=C_PANEL, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4,8), pady=8)
        right.pack_propagate(False)
        self._build_matrix_panel(right)

        # ── Status bar
        status = tk.Frame(root, bg=C_PANEL, height=28)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)
        self.status_var = tk.StringVar(value="Ready — upload a video to begin")
        tk.Label(status, textvariable=self.status_var, bg=C_PANEL,
                 fg=C_MUTED, font=("Courier New", 8)).pack(side=tk.LEFT, padx=12, pady=6)
        self.fps_var = tk.StringVar(value="")
        tk.Label(status, textvariable=self.fps_var, bg=C_PANEL,
                 fg=C_ACCENT, font=("Courier New", 8)).pack(side=tk.RIGHT, padx=12)

    def _build_controls(self, parent):
        def section(text):
            f = tk.Frame(parent, bg=C_PANEL)
            f.pack(fill=tk.X, padx=10, pady=(14,2))
            tk.Label(f, text=text, bg=C_PANEL, fg=C_ACCENT,
                     font=("Courier New", 8, "bold")).pack(anchor=tk.W)
            tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, padx=10, pady=(0,6))

        # Upload
        section("VIDEO SOURCE")
        self.upload_btn = tk.Button(parent, text="Upload Video File",
            bg=C_ACCENT, fg=C_BG, font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2", command=self._upload_video,
            activebackground="#00a08a", activeforeground=C_BG)
        self.upload_btn.pack(fill=tk.X, padx=10, pady=2)

        self.file_label = tk.Label(parent, text="No file selected", bg=C_PANEL,
            fg=C_MUTED, font=("Courier New", 7), wraplength=190, justify=tk.LEFT)
        self.file_label.pack(fill=tk.X, padx=10, pady=2)

        # ESP32-CAM simulation settings
        section("ESP32-CAM SIMULATION")

        tk.Label(parent, text="Resolution", bg=C_PANEL, fg=C_TEXT,
                 font=("Courier New", 8)).pack(anchor=tk.W, padx=10)
        res_options = [
            "160×120 (QQVGA)",
            "320×240 (QVGA)",
            "640×360 (ESP32-CAM high)",
            "800×600 (SVGA)",
            "1280×720 (HD)",
        ]
        res_menu = ttk.Combobox(parent, textvariable=self.sim_res,
                                values=res_options, state="readonly",
                                font=("Courier New", 8))
        res_menu.pack(fill=tk.X, padx=10, pady=2)
        self._style_combobox(res_menu)

        tk.Label(parent, text="FPS (simulated)", bg=C_PANEL, fg=C_TEXT,
                 font=("Courier New", 8)).pack(anchor=tk.W, padx=10, pady=(6,0))
        fps_frame = tk.Frame(parent, bg=C_PANEL)
        fps_frame.pack(fill=tk.X, padx=10)
        self.fps_slider = tk.Scale(fps_frame, from_=1, to=30, orient=tk.HORIZONTAL,
                                   variable=self.sim_fps, bg=C_PANEL, fg=C_TEXT,
                                   troughcolor=C_BORDER, activebackground=C_ACCENT,
                                   highlightthickness=0, font=("Courier New", 7))
        self.fps_slider.pack(fill=tk.X)

        # Display options
        section("DISPLAY OPTIONS")
        def _chk(text, var):
            tk.Checkbutton(parent, text=text, variable=var, bg=C_PANEL, fg=C_TEXT,
                           selectcolor=C_CARD, activebackground=C_PANEL,
                           font=("Courier New", 8), anchor=tk.W).pack(fill=tk.X, padx=10, pady=1)
        _chk("Show bounding boxes", self.show_boxes)
        _chk("Show zone overlay", self.show_zones)

        # Playback
        section("PLAYBACK")
        btn_frame = tk.Frame(parent, bg=C_PANEL)
        btn_frame.pack(fill=tk.X, padx=10, pady=4)

        self.play_btn = tk.Button(btn_frame, text="▶ PLAY",
            bg=C_ACCENT2, fg=C_BG, font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2", command=self._toggle_play,
            activebackground="#2a7ae6")
        self.play_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = tk.Button(btn_frame, text="■ STOP",
            bg=C_CARD, fg=C_TEXT, font=("Courier New", 9),
            relief=tk.FLAT, cursor="hand2", command=self._stop,
            activebackground=C_BORDER)
        self.stop_btn.pack(fill=tk.X, pady=2)

        # Legend
        section("ZONE LEGEND")
        legend = [
            (C_ON,       "Full beam zone"),
            (C_DIM,      "Shadow zone (glare)"),
            (C_BOOST,    "Hazard boost / flash"),
        ]
        for color, label in legend:
            row = tk.Frame(parent, bg=C_PANEL)
            row.pack(fill=tk.X, padx=10, pady=1)
            tk.Canvas(row, width=14, height=14, bg=C_PANEL,
                      highlightthickness=0).pack(side=tk.LEFT)
            c = tk.Canvas(row, width=14, height=14, bg=C_PANEL, highlightthickness=0)
            c.pack(side=tk.LEFT)
            c.create_oval(1,1,13,13, fill=color, outline="")
            tk.Label(row, text=label, bg=C_PANEL, fg=C_MUTED,
                     font=("Courier New", 7)).pack(side=tk.LEFT, padx=4)

    def _style_combobox(self, cb):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
            fieldbackground=C_CARD, background=C_CARD,
            foreground=C_TEXT, bordercolor=C_BORDER,
            arrowcolor=C_ACCENT, insertcolor=C_TEXT)

    def _build_video_panel(self, parent):
        tk.Label(parent, text="VIDEO FEED  (simulated ESP32-CAM)",
                 bg=C_BG, fg=C_MUTED, font=("Courier New", 8, "bold")).pack(anchor=tk.W, pady=(0,4))

        self.video_canvas = tk.Canvas(parent, bg=C_CARD,
                                      highlightthickness=1, highlightbackground=C_BORDER)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text
        self.video_canvas.bind("<Configure>", self._on_canvas_resize)
        self._draw_placeholder()

        # Stats row
        stats = tk.Frame(parent, bg=C_BG)
        stats.pack(fill=tk.X, pady=(4,0))
        self.res_label  = tk.Label(stats, text="Resolution: —", bg=C_BG,
                                   fg=C_MUTED, font=("Courier New", 8))
        self.res_label.pack(side=tk.LEFT, padx=4)
        self.det_label  = tk.Label(stats, text="Detections: 0", bg=C_BG,
                                   fg=C_MUTED, font=("Courier New", 8))
        self.det_label.pack(side=tk.LEFT, padx=12)

    def _draw_placeholder(self):
        self.video_canvas.delete("placeholder")
        w = self.video_canvas.winfo_width()  or 640
        h = self.video_canvas.winfo_height() or 360
        self.video_canvas.create_text(w//2, h//2,
            text="Upload a video file to begin\n\nSimulates ESP32-CAM MJPEG stream",
            fill=C_MUTED, font=("Courier New", 11), justify=tk.CENTER,
            tags="placeholder")

    def _on_canvas_resize(self, event):
        if not self.running:
            self._draw_placeholder()

    def _build_matrix_panel(self, parent):
        tk.Label(parent, text="8×8 BEAM MATRIX",
                 bg=C_PANEL, fg=C_ACCENT, font=("Courier New", 10, "bold")).pack(pady=(12,2))
        tk.Label(parent, text="col = angular zone (15°/zone)  |  row = distance",
                 bg=C_PANEL, fg=C_MUTED, font=("Courier New", 7)).pack()

        # Zone angle labels (top)
        angle_frame = tk.Frame(parent, bg=C_PANEL)
        angle_frame.pack(pady=(8,0))
        angles = ["-60°","-45°","-30°","-15°","0°","+15°","+30°","+60°"]
        for a in angles:
            tk.Label(angle_frame, text=a, bg=C_PANEL, fg=C_MUTED,
                     font=("Courier New", 6), width=4).pack(side=tk.LEFT)

        # Matrix canvas
        self.matrix_canvas = tk.Canvas(parent, bg=C_PANEL, highlightthickness=0,
                                       width=288, height=288)
        self.matrix_canvas.pack(pady=4)
        self._led_items = {}
        self._draw_matrix_base()

        # Distance row labels (right side)
        dist_frame = tk.Frame(parent, bg=C_PANEL)
        dist_frame.pack()
        distances = ["100m+","80m","60m","40m","25m","15m","8m","wide"]
        for d in distances:
            tk.Label(dist_frame, text=d, bg=C_PANEL, fg=C_MUTED,
                     font=("Courier New", 7)).pack(anchor=tk.W)

        # Zone state bar
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=8)
        tk.Label(parent, text="ZONE STATES", bg=C_PANEL, fg=C_MUTED,
                 font=("Courier New", 7, "bold")).pack()

        self.zone_bar_canvas = tk.Canvas(parent, bg=C_PANEL, height=36,
                                         highlightthickness=0)
        self.zone_bar_canvas.pack(fill=tk.X, padx=10, pady=4)

        # Detection list
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=(4,8))
        tk.Label(parent, text="DETECTIONS", bg=C_PANEL, fg=C_MUTED,
                 font=("Courier New", 7, "bold")).pack()

        self.det_list_frame = tk.Frame(parent, bg=C_PANEL)
        self.det_list_frame.pack(fill=tk.X, padx=10, pady=4)

        # Zone byte readout
        tk.Frame(parent, bg=C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=4)
        self.zone_byte_var = tk.StringVar(value="zone byte: 0xFF")
        tk.Label(parent, textvariable=self.zone_byte_var, bg=C_PANEL,
                 fg=C_ACCENT2, font=("Courier New", 9, "bold")).pack(pady=4)

    def _draw_matrix_base(self):
        cell = 34
        pad  = 4
        for row in range(8):
            for col in range(8):
                x1 = pad + col * cell
                y1 = pad + row * cell
                x2 = x1 + cell - 4
                y2 = y1 + cell - 4
                oval = self.matrix_canvas.create_oval(
                    x1, y1, x2, y2,
                    fill=C_OFF, outline="#0f1f35", width=1)
                self._led_items[(col, row)] = oval

    # ── VIDEO UPLOAD & PLAYBACK ───────────────────────────────────────────────
    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("All files", "*.*")])
        if not path:
            return
        self.video_path = path
        fname = os.path.basename(path)
        self.file_label.config(text=fname)
        self.status_var.set(f"Loaded: {fname}")

    def _parse_resolution(self):
        res_str = self.sim_res.get()
        part = res_str.split("(")[0].strip()
        try:
            w, h = [int(x) for x in part.split("×")]
            return w, h
        except Exception:
            return 640, 360

    def _toggle_play(self):
        if not self.video_path:
            self.status_var.set("Please upload a video file first.")
            return
        if self.running and not self.paused:
            self.paused = True
            self.play_btn.config(text="▶ RESUME")
            self.status_var.set("Paused")
        elif self.paused:
            self.paused = False
            self.play_btn.config(text="⏸ PAUSE")
            self.status_var.set("Playing...")
        else:
            self.running = True
            self.paused  = False
            self.play_btn.config(text="⏸ PAUSE")
            self.status_var.set("Playing...")
            threading.Thread(target=self._video_thread, daemon=True).start()
            threading.Thread(target=self._detect_thread, daemon=True).start()
            self.root.after(33, self._update_ui)

    def _stop(self):
        self.running = False
        self.paused  = False
        self.play_btn.config(text="▶ PLAY")
        self.status_var.set("Stopped")
        self._draw_placeholder()
        # Reset matrix to full beam
        self.zone_matrix = ZoneMatrix()
        self._render_matrix()

    # ── WORKER THREADS ────────────────────────────────────────────────────────
    def _video_thread(self):
        """Reads video, resizes to simulated resolution, feeds frame_queue."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status_var.set("Error: cannot open video file")
            self.running = False
            return

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            target_fps = self.sim_fps.get()
            frame_dur  = 1.0 / max(1, target_fps)

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue

            # Resize to simulated resolution
            tw, th = self._parse_resolution()
            frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            # Put in queue (drop if full — real-time priority)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            time.sleep(frame_dur)

        cap.release()

    def _detect_thread(self):
        """Pulls frames, runs detector, pushes (frame, detections) to det_queue."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Scale to FRAME_W for detection (consistent zone mapping)
            det_frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            detections = self.detector.detect(det_frame)

            # Scale detection boxes back to display frame size
            sx = frame.shape[1] / FRAME_W
            sy = frame.shape[0] / FRAME_H
            for d in detections:
                x1,y1,x2,y2 = d["box"]
                d["box"] = (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

            # Update zone matrix
            self.zone_matrix.update(detections, FRAME_W)

            # Annotate frame
            annotated = self._annotate(frame.copy(), detections, frame.shape[1])

            try:
                self.det_queue.put_nowait((annotated, detections))
            except queue.Full:
                pass

    def _annotate(self, frame, detections, frame_w):
        """Draw bounding boxes and zone overlay on frame."""
        h, w = frame.shape[:2]

        # Zone overlay
        if self.show_zones.get():
            zone_w = w // ZONE_COUNT
            for col in range(ZONE_COUNT):
                state = self.zone_matrix.zone_states[col]
                if state == "suppress":
                    color = (50, 50, 220)    # red-ish overlay (BGR)
                    alpha = 0.25
                elif state == "boost":
                    color = (0, 160, 240)    # amber overlay
                    alpha = 0.20
                else:
                    color = (0, 180, 140)    # teal overlay
                    alpha = 0.08

                x1_z = col * zone_w
                x2_z = x1_z + zone_w
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1_z, 0), (x2_z, h), color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

                # Zone label at top
                label = f"Z{col+1}"
                cv2.putText(frame, label, (x1_z+4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

        # Bounding boxes
        if self.show_boxes.get():
            for det in detections:
                x1,y1,x2,y2 = det["box"]
                label  = det["label"]
                conf   = det["conf"]
                is_glare  = label in GLARE_CLASSES
                is_hazard = label in HAZARD_CLASSES

                if is_glare:
                    color = (60, 60, 220)
                    tag   = "GLARE"
                elif is_hazard:
                    color = (0, 160, 240)
                    tag   = "HAZARD"
                else:
                    color = (160, 160, 160)
                    tag   = ""

                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                txt = f"{label} {conf:.0%}"
                if tag:
                    txt = f"[{tag}] {txt}"
                cv2.putText(frame, txt, (x1, max(y1-6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        return frame

    # ── UI UPDATE (main thread) ───────────────────────────────────────────────
    def _update_ui(self):
        if not self.running:
            return

        try:
            frame, detections = self.det_queue.get_nowait()
        except queue.Empty:
            self.root.after(16, self._update_ui)
            return

        self._last_detections = detections
        self._frame_count += 1

        # FPS counter
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps_actual = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now
            self.fps_var.set(f"actual: {self._fps_actual:.1f} fps  |  sim: {self.sim_fps.get()} fps  |  res: {'×'.join(str(x) for x in self._parse_resolution())}")

        # Show frame
        self._show_frame(frame)

        # Update stats
        h, w = frame.shape[:2]
        self.res_label.config(text=f"Resolution: {w}×{h}")
        self.det_label.config(text=f"Detections: {len(detections)}")

        # Update matrix & detection list
        self._render_matrix()
        self._render_zone_bar()
        self._render_detection_list(detections)
        self._render_zone_byte()

        self.root.after(16, self._update_ui)

    def _show_frame(self, frame):
        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        # Convert BGR→RGB and resize to fit canvas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Maintain aspect ratio
        fh, fw = frame.shape[:2]
        scale = min(cw/fw, ch/fh)
        nw = int(fw * scale)
        nh = int(fh * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.video_canvas.delete("frame")
        self.video_canvas.delete("placeholder")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.video_canvas.create_image(ox, oy, anchor=tk.NW, image=photo, tags="frame")
        self.video_canvas._photo = photo  # keep reference

    def _render_matrix(self):
        zm = self.zone_matrix
        for col in range(8):
            for row in range(8):
                color, brightness = zm.led_color(col, row)
                # Interpolate between off and the zone colour
                display_color = interpolate_color(C_OFF, color, max(0.05, brightness))
                self.matrix_canvas.itemconfig(self._led_items[(col, row)],
                                              fill=display_color)

    def _render_zone_bar(self):
        c = self.zone_bar_canvas
        c.delete("all")
        cw = c.winfo_width() or 288
        cell_w = cw / ZONE_COUNT
        for col in range(ZONE_COUNT):
            state = self.zone_matrix.zone_states[col]
            color = {
                "on":       C_ON,
                "suppress": C_DIM,
                "boost":    C_BOOST,
            }.get(state, C_ON)
            x1 = col * cell_w + 1
            x2 = x1 + cell_w - 2
            c.create_rectangle(x1, 2, x2, 34, fill=color, outline="")
            c.create_text((x1+x2)//2, 18, text=str(col+1),
                          fill=C_BG, font=("Courier New", 8, "bold"))

    def _render_detection_list(self, detections):
        for w in self.det_list_frame.winfo_children():
            w.destroy()
        if not detections:
            tk.Label(self.det_list_frame, text="no detections", bg=C_PANEL,
                     fg=C_MUTED, font=("Courier New", 7)).pack()
            return
        for det in detections[:6]:  # max 6 shown
            label = det["label"]
            conf  = det["conf"]
            cat   = "GLARE" if label in GLARE_CLASSES else ("HAZARD" if label in HAZARD_CLASSES else "")
            col   = int(det["cx_norm"] * ZONE_COUNT)
            col   = max(0, min(7, col))
            color = C_DIM if label in GLARE_CLASSES else (C_BOOST if label in HAZARD_CLASSES else C_MUTED)
            row = tk.Frame(self.det_list_frame, bg=C_PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{label}", bg=C_PANEL, fg=C_TEXT,
                     font=("Courier New", 8)).pack(side=tk.LEFT)
            tk.Label(row, text=f" {conf:.0%}", bg=C_PANEL, fg=C_MUTED,
                     font=("Courier New", 7)).pack(side=tk.LEFT)
            if cat:
                tk.Label(row, text=f" [{cat}]", bg=C_PANEL, fg=color,
                         font=("Courier New", 7, "bold")).pack(side=tk.LEFT)
            tk.Label(row, text=f" Z{col+1}", bg=C_PANEL, fg=C_ACCENT2,
                     font=("Courier New", 7)).pack(side=tk.RIGHT)

    def _render_zone_byte(self):
        byte_val = 0
        for col in range(8):
            if self.zone_matrix.zone_states[col] != "suppress":
                byte_val |= (1 << col)
        binary = format(byte_val, "08b")
        self.zone_byte_var.set(f"zone byte: 0x{byte_val:02X}  ({binary})")

    # ── Flash ticker ──────────────────────────────────────────────────────────
    def _start_flash_ticker(self):
        self.zone_matrix.tick_flash()
        self._flash_timer = self.root.after(250, self._start_flash_ticker)

    def on_close(self):
        self.running = False
        if self._flash_timer:
            self.root.after_cancel(self._flash_timer)
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = SmartHeadlampApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()