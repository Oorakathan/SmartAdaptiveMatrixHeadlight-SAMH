"""
simulator.py
------------
Tkinter UI — wires VideoFeeder, AIDetector, and ZoneController together.

Responsibilities:
  - Build and own the UI (video canvas, LED matrix, controls panel)
  - Start / pause / stop the pipeline on user action
  - Read results from result_queue on the main thread (tkinter after() loop)
  - Pass detection results to ZoneController each frame
  - Render the 8×8 LED matrix using ZoneController.led_color()
  - Push current zone_states back to AIDetector for annotation overlay
  - Display stats, detection list, zone byte readout

Does NOT:
  - Run any CV or detection logic
  - Own the feeder/detector threads (those objects own themselves)
  - Block the main thread

Run this file directly:
    python simulator.py
"""

import os
import queue
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

import config
from video_feeder  import VideoFeeder
from ai_detector   import AIDetector
from led_controller import ZoneController


# ── Colour helpers ────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Main simulator window ─────────────────────────────────────────────────────

class Simulator:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Adaptive Matrix Headlamp — Simulator")
        self.root.configure(bg=config.C_BG)
        self.root.minsize(1100, 680)

        # ── Shared queues ─────────────────────────────────────────────────────
        self._frame_q  = queue.Queue(maxsize=2)   # feeder  → detector
        self._result_q = queue.Queue(maxsize=2)   # detector → UI

        # ── Component instances ───────────────────────────────────────────────
        self._feeder     = VideoFeeder(self._frame_q)
        self._detector   = AIDetector(self._frame_q, self._result_q)
        self._controller = ZoneController()

        # ── UI state vars (set before _build_ui) ─────────────────────────────
        self._sim_res  = tk.StringVar(value=config.DEFAULT_RESOLUTION)
        self._sim_fps  = tk.IntVar(value=config.DEFAULT_FPS)
        self._show_boxes = tk.BooleanVar(value=True)
        self._show_zones = tk.BooleanVar(value=True)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._pipeline_running = False
        self._frame_count  = 0
        self._fps_timer    = time.time()
        self._flash_job    = None

        # ── Build UI ─────────────────────────────────────────────────────────
        self._build_ui()
        self._start_flash_ticker()

        # Sync display options to detector immediately
        self._on_display_toggle()

    # ═══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self.root, bg=config.C_PANEL, height=48)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        tk.Label(
            topbar, text="SMART ADAPTIVE MATRIX HEADLAMP",
            bg=config.C_PANEL, fg=config.C_ACCENT,
            font=("Courier New", 13, "bold"),
        ).pack(side=tk.LEFT, padx=16, pady=12)

        pill_color = config.C_ACCENT if "YOLO" in self._detector.detector_name else config.C_LED_BOOST
        pill_text  = " YOLO LIVE " if "YOLO" in self._detector.detector_name else " MOCK MODE "
        tk.Label(
            topbar, text=pill_text,
            bg=pill_color, fg=config.C_BG,
            font=("Courier New", 9, "bold"),
        ).pack(side=tk.LEFT, padx=4)

        tk.Label(
            topbar, text=self._detector.detector_name,
            bg=config.C_PANEL, fg=config.C_MUTED,
            font=("Courier New", 8),
        ).pack(side=tk.LEFT, padx=6)

        # Body
        body = tk.Frame(self.root, bg=config.C_BG)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=config.C_PANEL, width=224)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        centre = tk.Frame(body, bg=config.C_BG)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
        self._build_video_panel(centre)

        right = tk.Frame(body, bg=config.C_PANEL, width=322)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        right.pack_propagate(False)
        self._build_matrix_panel(right)

        # Status bar
        sbar = tk.Frame(self.root, bg=config.C_PANEL, height=28)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        sbar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready — upload a video to begin")
        self._fps_var    = tk.StringVar(value="")
        tk.Label(sbar, textvariable=self._status_var, bg=config.C_PANEL,
                 fg=config.C_MUTED, font=("Courier New", 8)).pack(side=tk.LEFT, padx=12, pady=6)
        tk.Label(sbar, textvariable=self._fps_var, bg=config.C_PANEL,
                 fg=config.C_ACCENT, font=("Courier New", 8)).pack(side=tk.RIGHT, padx=12)

    # ── Controls panel ────────────────────────────────────────────────────────

    def _build_controls(self, parent):

        def section(label):
            tk.Frame(parent, bg=config.C_PANEL).pack(fill=tk.X, padx=10, pady=(14, 2))
            tk.Label(parent, text=label, bg=config.C_PANEL, fg=config.C_ACCENT,
                     font=("Courier New", 8, "bold")).pack(anchor=tk.W, padx=10)
            tk.Frame(parent, bg=config.C_BORDER, height=1).pack(fill=tk.X, padx=10, pady=(0, 6))

        # ── Video source
        section("VIDEO SOURCE")
        self._upload_btn = tk.Button(
            parent, text="Upload Video File",
            bg=config.C_ACCENT, fg=config.C_BG,
            font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_upload,
            activebackground="#00a08a", activeforeground=config.C_BG,
        )
        self._upload_btn.pack(fill=tk.X, padx=10, pady=2)

        self._file_label = tk.Label(
            parent, text="No file selected",
            bg=config.C_PANEL, fg=config.C_MUTED,
            font=("Courier New", 7), wraplength=196, justify=tk.LEFT,
        )
        self._file_label.pack(fill=tk.X, padx=10, pady=2)

        # ── ESP32-CAM simulation
        section("ESP32-CAM SIMULATION")

        tk.Label(parent, text="Resolution", bg=config.C_PANEL,
                 fg=config.C_TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=10)

        self._res_menu = ttk.Combobox(
            parent,
            textvariable=self._sim_res,
            values=list(config.RESOLUTION_PRESETS.keys()),
            state="readonly",
            font=("Courier New", 8),
        )
        self._res_menu.pack(fill=tk.X, padx=10, pady=2)
        self._res_menu.bind("<<ComboboxSelected>>", self._on_resolution_change)
        self._style_combobox()

        tk.Label(parent, text="FPS (simulated)", bg=config.C_PANEL,
                 fg=config.C_TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=10, pady=(6, 0))
        tk.Scale(
            parent, from_=1, to=30, orient=tk.HORIZONTAL,
            variable=self._sim_fps,
            bg=config.C_PANEL, fg=config.C_TEXT,
            troughcolor=config.C_BORDER, activebackground=config.C_ACCENT,
            highlightthickness=0, font=("Courier New", 7),
            command=self._on_fps_change,
        ).pack(fill=tk.X, padx=10)

        # ── Display options
        section("DISPLAY OPTIONS")
        for text, var in [
            ("Show bounding boxes", self._show_boxes),
            ("Show zone overlay",   self._show_zones),
        ]:
            tk.Checkbutton(
                parent, text=text, variable=var,
                bg=config.C_PANEL, fg=config.C_TEXT,
                selectcolor=config.C_CARD, activebackground=config.C_PANEL,
                font=("Courier New", 8), anchor=tk.W,
                command=self._on_display_toggle,
            ).pack(fill=tk.X, padx=10, pady=1)

        # ── Playback
        section("PLAYBACK")
        btn_f = tk.Frame(parent, bg=config.C_PANEL)
        btn_f.pack(fill=tk.X, padx=10, pady=4)

        self._play_btn = tk.Button(
            btn_f, text="▶  PLAY",
            bg=config.C_ACCENT2, fg=config.C_BG,
            font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_play_pause,
            activebackground="#2a7ae6",
        )
        self._play_btn.pack(fill=tk.X, pady=2)

        tk.Button(
            btn_f, text="■  STOP",
            bg=config.C_CARD, fg=config.C_TEXT,
            font=("Courier New", 9),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_stop,
            activebackground=config.C_BORDER,
        ).pack(fill=tk.X, pady=2)

        # ── Legend
        section("ZONE LEGEND")
        for color, label in [
            (config.C_LED_ON,       "Full beam"),
            (config.C_LED_SUPPRESS, "Shadow zone (glare)"),
            (config.C_LED_BOOST,    "Hazard boost / flash"),
        ]:
            row = tk.Frame(parent, bg=config.C_PANEL)
            row.pack(fill=tk.X, padx=10, pady=1)
            dot = tk.Canvas(row, width=14, height=14, bg=config.C_PANEL, highlightthickness=0)
            dot.pack(side=tk.LEFT)
            dot.create_oval(1, 1, 13, 13, fill=color, outline="")
            tk.Label(row, text=label, bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack(side=tk.LEFT, padx=4)

    def _style_combobox(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
            fieldbackground=config.C_CARD, background=config.C_CARD,
            foreground=config.C_TEXT, bordercolor=config.C_BORDER,
            arrowcolor=config.C_ACCENT, insertcolor=config.C_TEXT)

    # ── Video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self, parent):
        tk.Label(
            parent, text="VIDEO FEED  (simulated ESP32-CAM)",
            bg=config.C_BG, fg=config.C_MUTED,
            font=("Courier New", 8, "bold"),
        ).pack(anchor=tk.W, pady=(0, 4))

        self._video_canvas = tk.Canvas(
            parent, bg=config.C_CARD,
            highlightthickness=1, highlightbackground=config.C_BORDER,
        )
        self._video_canvas.pack(fill=tk.BOTH, expand=True)
        self._video_canvas.bind("<Configure>", lambda e: self._draw_placeholder() if not self._pipeline_running else None)
        self._draw_placeholder()

        stats_row = tk.Frame(parent, bg=config.C_BG)
        stats_row.pack(fill=tk.X, pady=(4, 0))
        self._res_stat  = tk.Label(stats_row, text="Resolution: —", bg=config.C_BG,
                                   fg=config.C_MUTED, font=("Courier New", 8))
        self._det_stat  = tk.Label(stats_row, text="Detections: 0", bg=config.C_BG,
                                   fg=config.C_MUTED, font=("Courier New", 8))
        self._res_stat.pack(side=tk.LEFT, padx=4)
        self._det_stat.pack(side=tk.LEFT, padx=12)

    def _draw_placeholder(self):
        self._video_canvas.delete("placeholder")
        w = self._video_canvas.winfo_width()  or 640
        h = self._video_canvas.winfo_height() or 360
        self._video_canvas.create_text(
            w // 2, h // 2,
            text="Upload a video file to begin\n\nSimulates ESP32-CAM MJPEG stream",
            fill=config.C_MUTED, font=("Courier New", 11), justify=tk.CENTER,
            tags="placeholder",
        )

    # ── Matrix panel ──────────────────────────────────────────────────────────

    def _build_matrix_panel(self, parent):
        tk.Label(parent, text="8×8 BEAM MATRIX",
                 bg=config.C_PANEL, fg=config.C_ACCENT,
                 font=("Courier New", 10, "bold")).pack(pady=(12, 2))
        tk.Label(parent, text="col = zone (15°/zone)   row = distance band",
                 bg=config.C_PANEL, fg=config.C_MUTED,
                 font=("Courier New", 7)).pack()

        # Angle labels above matrix
        angle_row = tk.Frame(parent, bg=config.C_PANEL)
        angle_row.pack(pady=(8, 0))
        for a in ["-60°","-45°","-30°","-15°","0°","+15°","+30°","+60°"]:
            tk.Label(angle_row, text=a, bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 6), width=4).pack(side=tk.LEFT)

        # Matrix canvas
        self._matrix_canvas = tk.Canvas(
            parent, bg=config.C_PANEL,
            highlightthickness=0, width=288, height=288,
        )
        self._matrix_canvas.pack(pady=4)
        self._led_ovals = {}
        self._init_led_ovals()

        # Distance labels
        dist_frame = tk.Frame(parent, bg=config.C_PANEL)
        dist_frame.pack()
        for d in ["100m+","80m","60m","40m","25m","15m","8m","wide"]:
            tk.Label(dist_frame, text=d, bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack(anchor=tk.W)

        # Zone state bar
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=8)
        tk.Label(parent, text="ZONE STATES", bg=config.C_PANEL, fg=config.C_MUTED,
                 font=("Courier New", 7, "bold")).pack()
        self._zone_bar = tk.Canvas(parent, bg=config.C_PANEL, height=36,
                                   highlightthickness=0)
        self._zone_bar.pack(fill=tk.X, padx=10, pady=4)

        # Detection list
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=(4, 8))
        tk.Label(parent, text="DETECTIONS", bg=config.C_PANEL, fg=config.C_MUTED,
                 font=("Courier New", 7, "bold")).pack()
        self._det_list = tk.Frame(parent, bg=config.C_PANEL)
        self._det_list.pack(fill=tk.X, padx=10, pady=4)

        # Zone byte readout
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(fill=tk.X, padx=12, pady=4)
        self._zone_byte_var = tk.StringVar(value="zone byte: 0xFF  (11111111)")
        tk.Label(parent, textvariable=self._zone_byte_var,
                 bg=config.C_PANEL, fg=config.C_ACCENT2,
                 font=("Courier New", 9, "bold")).pack(pady=4)

    def _init_led_ovals(self):
        cell = 34
        pad  = 4
        for row in range(8):
            for col in range(8):
                x1 = pad + col * cell
                y1 = pad + row * cell
                x2 = x1 + cell - 4
                y2 = y1 + cell - 4
                oval = self._matrix_canvas.create_oval(
                    x1, y1, x2, y2,
                    fill=config.C_LED_OFF, outline="#0f1f35", width=1,
                )
                self._led_ovals[(col, row)] = oval

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All files",   "*.*"),
            ],
        )
        if not path:
            return
        ok = self._feeder.load(path)
        if not ok:
            self._status_var.set(f"Error: cannot open {os.path.basename(path)}")
            return
        self._file_label.config(text=os.path.basename(path))
        self._status_var.set(
            f"Loaded: {os.path.basename(path)}  "
            f"({self._feeder.video_w}×{self._feeder.video_h}, "
            f"{self._feeder.video_fps:.0f} fps, "
            f"{self._feeder.duration_s:.0f}s)"
        )

    def _on_resolution_change(self, _event=None):
        key = self._sim_res.get()
        w, h = config.RESOLUTION_PRESETS.get(key, (640, 360))
        self._feeder.set_resolution(w, h)

    def _on_fps_change(self, _value=None):
        self._feeder.set_fps(self._sim_fps.get())

    def _on_display_toggle(self):
        self._detector.show_boxes = self._show_boxes.get()
        self._detector.show_zones = self._show_zones.get()

    def _on_play_pause(self):
        if not self._feeder._path:
            self._status_var.set("Please upload a video file first.")
            return

        if self._pipeline_running and not self._feeder.is_paused:
            # Pause
            self._feeder.pause()
            self._play_btn.config(text="▶  RESUME")
            self._status_var.set("Paused")

        elif self._pipeline_running and self._feeder.is_paused:
            # Resume
            self._feeder.resume()
            self._play_btn.config(text="⏸  PAUSE")
            self._status_var.set("Playing…")

        else:
            # Start fresh
            self._start_pipeline()

    def _on_stop(self):
        self._stop_pipeline()
        self._play_btn.config(text="▶  PLAY")
        self._status_var.set("Stopped")
        self._draw_placeholder()
        self._controller.reset()
        self._render_matrix()
        self._render_zone_bar()
        self._render_zone_byte()

    # ═══════════════════════════════════════════════════════════════════════════
    # PIPELINE CONTROL
    # ═══════════════════════════════════════════════════════════════════════════

    def _start_pipeline(self):
        # Apply current settings
        key  = self._sim_res.get()
        w, h = config.RESOLUTION_PRESETS.get(key, (640, 360))
        self._feeder.set_resolution(w, h)
        self._feeder.set_fps(self._sim_fps.get())
        self._on_display_toggle()

        self._feeder.start()
        self._detector.start()

        self._pipeline_running = True
        self._frame_count      = 0
        self._fps_timer        = time.time()
        self._play_btn.config(text="⏸  PAUSE")
        self._status_var.set("Playing…")

        # Kick off the UI poll loop
        self.root.after(16, self._ui_poll)

    def _stop_pipeline(self):
        self._pipeline_running = False
        self._feeder.stop()
        self._detector.stop()

        # Drain queues so next run starts clean
        for q in (self._frame_q, self._result_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    # ═══════════════════════════════════════════════════════════════════════════
    # UI POLL LOOP  (main thread only — called via root.after)
    # ═══════════════════════════════════════════════════════════════════════════

    def _ui_poll(self):
        if not self._pipeline_running:
            return

        # Pull latest result (non-blocking)
        try:
            annotated_frame, detections, _raw = self._result_q.get_nowait()
        except queue.Empty:
            self.root.after(16, self._ui_poll)
            return

        # ── 1. Feed detections to the zone controller
        self._controller.update(detections)

        # ── 2. Push current zone states back to detector (for annotation overlay)
        self._detector.zone_states = self._controller.zone_states[:]
        # Also sync the 8x8 LED states matrix for individual LED overlay
        self._detector.led_states = [row[:] for row in self._controller.led_states]

        # ── 3. Render everything
        self._show_frame(annotated_frame)
        self._render_matrix()
        self._render_zone_bar()
        self._render_detection_list(detections)
        self._render_zone_byte()

        # ── 4. Stats
        h, w = annotated_frame.shape[:2]
        self._res_stat.config(text=f"Resolution: {w}×{h}")
        self._det_stat.config(text=f"Detections: {len(detections)}")

        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            ui_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now
            self._fps_var.set(
                f"ui: {ui_fps:.1f} fps  "
                f"| cam: {self._feeder.actual_fps:.1f} fps  "
                f"| ai: {self._detector.actual_fps:.1f} fps  "
                f"| sim: {self._sim_fps.get()} fps"
            )

        self.root.after(16, self._ui_poll)

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDER HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_frame(self, frame):
        cw = self._video_canvas.winfo_width()
        ch = self._video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(rgb)
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        img   = img.resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        self._video_canvas.delete("frame", "placeholder")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._video_canvas.create_image(ox, oy, anchor=tk.NW, image=photo, tags="frame")
        self._video_canvas._photo = photo   # prevent GC

    def _render_matrix(self):
        for col in range(8):
            for row in range(8):
                color, brightness = self._controller.led_color(col, row)
                display = _blend(config.C_LED_OFF, color, max(0.05, brightness))
                self._matrix_canvas.itemconfig(self._led_ovals[(col, row)], fill=display)

    def _render_zone_bar(self):
        c  = self._zone_bar
        c.delete("all")
        cw = c.winfo_width() or 288
        cw = max(cw, 1)
        cell_w = cw / config.ZONE_COUNT
        color_map = {
            "on":       config.C_LED_ON,
            "suppress": config.C_LED_SUPPRESS,
            "boost":    config.C_LED_BOOST,
        }
        for col, state in enumerate(self._controller.zone_states):
            x1 = col * cell_w + 1
            x2 = x1 + cell_w - 2
            fill = color_map.get(state, config.C_LED_ON)
            c.create_rectangle(x1, 2, x2, 34, fill=fill, outline="")
            c.create_text(
                (x1 + x2) / 2, 18,
                text=str(col + 1),
                fill=config.C_BG, font=("Courier New", 8, "bold"),
            )

    def _render_detection_list(self, detections):
        for w in self._det_list.winfo_children():
            w.destroy()

        if not detections:
            tk.Label(self._det_list, text="no detections",
                     bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack()
            return

        color_map = {
            "glare":  config.C_LED_SUPPRESS,
            "hazard": config.C_LED_BOOST,
            "other":  config.C_MUTED,
        }
        for det in detections[:6]:
            col   = self._controller._cx_to_col(det["cx_norm"])
            cat   = det["category"]
            row_f = tk.Frame(self._det_list, bg=config.C_PANEL)
            row_f.pack(fill=tk.X, pady=1)
            tk.Label(row_f, text=det["label"], bg=config.C_PANEL,
                     fg=config.C_TEXT, font=("Courier New", 8)).pack(side=tk.LEFT)
            tk.Label(row_f, text=f" {det['conf']:.0%}", bg=config.C_PANEL,
                     fg=config.C_MUTED, font=("Courier New", 7)).pack(side=tk.LEFT)
            if cat != "other":
                tk.Label(row_f, text=f" [{cat.upper()}]",
                         bg=config.C_PANEL, fg=color_map[cat],
                         font=("Courier New", 7, "bold")).pack(side=tk.LEFT)
            tk.Label(row_f, text=f" Z{col + 1}", bg=config.C_PANEL,
                     fg=config.C_ACCENT2, font=("Courier New", 7)).pack(side=tk.RIGHT)

    def _render_zone_byte(self):
        byte_val = self._controller.zone_byte()
        binary   = format(byte_val, "08b")
        self._zone_byte_var.set(f"zone byte: 0x{byte_val:02X}  ({binary})")

    # ── Flash ticker ──────────────────────────────────────────────────────────

    def _start_flash_ticker(self):
        self._controller.tick_flash()
        self._flash_job = self.root.after(
            config.FLASH_INTERVAL_MS, self._start_flash_ticker
        )

    # ── Clean shutdown ────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_pipeline()
        if self._flash_job:
            self.root.after_cancel(self._flash_job)
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app  = Simulator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
