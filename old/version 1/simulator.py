"""
simulator.py
------------
Tkinter UI — wires VideoFeeder, AIDetector, and ZoneController together.

Key changes vs previous version
---------------------------------
* Matrix is fully dynamic — changing the preset rebuilds the canvas and
  resets the controller. The cell size is computed as canvas_px / matrix_dim,
  so the LED dots always fill the panel regardless of ZONE_COUNT × ROW_LEDS.

* annotate_frame is called EXACTLY ONCE per frame, here in _ui_poll, AFTER
  the controller has processed detections. This ensures LED overlay colours
  match the matrix state that is actually displayed.

* AIDetector now pushes (raw_frame, detections) — no pre-annotated frame.

* Zone byte display adapts to >8 columns (shows full hex, not just 8 bits).

Run:
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
from video_feeder   import VideoFeeder
from ai_detector    import AIDetector, annotate_frame
from led_controller import ZoneController


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blend(c1: str, c2: str, t: float) -> str:
    r1,g1,b1 = _hex_to_rgb(c1)
    r2,g2,b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2-r1)*t)
    g = int(g1 + (g2-g1)*t)
    b = int(b1 + (b2-b1)*t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class Simulator:

    # Matrix panel physical pixel size (fixed; LEDs scale inside it)
    _MATRIX_PX = 288

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Adaptive Matrix Headlamp — Simulator")
        self.root.configure(bg=config.C_BG)
        self.root.minsize(1160, 700)

        # Shared queues
        self._frame_q  = queue.Queue(maxsize=2)
        self._result_q = queue.Queue(maxsize=2)

        # Components
        self._feeder     = VideoFeeder(self._frame_q)
        self._detector   = AIDetector(self._frame_q, self._result_q)
        self._controller = ZoneController()

        # UI vars
        self._sim_res      = tk.StringVar(value=config.DEFAULT_RESOLUTION)
        self._sim_fps      = tk.IntVar(value=config.DEFAULT_FPS)
        self._matrix_preset= tk.StringVar(value=config.DEFAULT_MATRIX)
        self._show_boxes   = tk.BooleanVar(value=True)
        self._show_zones   = tk.BooleanVar(value=True)

        # Runtime
        self._pipeline_running = False
        self._frame_count  = 0
        self._fps_timer    = time.time()
        self._flash_job    = None
        self._led_ovals    = {}   # (col, row) → canvas oval id

        self._build_ui()
        self._rebuild_matrix_canvas()   # draw initial 8×8 grid
        self._start_flash_ticker()

    # ═══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Top bar
        topbar = tk.Frame(self.root, bg=config.C_PANEL, height=48)
        topbar.pack(fill=tk.X, side=tk.TOP)
        topbar.pack_propagate(False)

        tk.Label(topbar, text="SMART ADAPTIVE MATRIX HEADLAMP",
                 bg=config.C_PANEL, fg=config.C_ACCENT,
                 font=("Courier New", 13, "bold")).pack(side=tk.LEFT, padx=16, pady=12)

        pill_col  = config.C_ACCENT if "YOLO" in self._detector.detector_name else config.C_LED_BOOST
        pill_text = f" {config.MODEL_NAME} " if "YOLO" in self._detector.detector_name else " MOCK MODE "
        tk.Label(topbar, text=pill_text, bg=pill_col, fg=config.C_BG,
                 font=("Courier New", 9, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Label(topbar, text=self._detector.detector_name,
                 bg=config.C_PANEL, fg=config.C_MUTED,
                 font=("Courier New", 8)).pack(side=tk.LEFT, padx=6)

        # ── Body
        body = tk.Frame(self.root, bg=config.C_BG)
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=config.C_PANEL, width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        centre = tk.Frame(body, bg=config.C_BG)
        centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=8)
        self._build_video_panel(centre)

        right = tk.Frame(body, bg=config.C_PANEL, width=330)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(4,8), pady=8)
        right.pack_propagate(False)
        self._build_matrix_panel(right)

        # ── Status bar
        sbar = tk.Frame(self.root, bg=config.C_PANEL, height=28)
        sbar.pack(fill=tk.X, side=tk.BOTTOM)
        sbar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Ready — upload a video to begin")
        self._fps_var    = tk.StringVar(value="")
        tk.Label(sbar, textvariable=self._status_var,
                 bg=config.C_PANEL, fg=config.C_MUTED,
                 font=("Courier New", 8)).pack(side=tk.LEFT, padx=12, pady=6)
        tk.Label(sbar, textvariable=self._fps_var,
                 bg=config.C_PANEL, fg=config.C_ACCENT,
                 font=("Courier New", 8)).pack(side=tk.RIGHT, padx=12)

    # ── Controls ──────────────────────────────────────────────────────────────

    def _build_controls(self, parent):

        def section(label):
            tk.Frame(parent, bg=config.C_PANEL).pack(fill=tk.X, padx=10, pady=(12,2))
            tk.Label(parent, text=label, bg=config.C_PANEL, fg=config.C_ACCENT,
                     font=("Courier New", 8, "bold")).pack(anchor=tk.W, padx=10)
            tk.Frame(parent, bg=config.C_BORDER, height=1).pack(fill=tk.X, padx=10, pady=(0,5))

        # Video source
        section("VIDEO SOURCE")
        self._upload_btn = tk.Button(
            parent, text="Upload Video File",
            bg=config.C_ACCENT, fg=config.C_BG,
            font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_upload,
            activebackground="#00a08a", activeforeground=config.C_BG)
        self._upload_btn.pack(fill=tk.X, padx=10, pady=2)

        self._file_label = tk.Label(
            parent, text="No file selected",
            bg=config.C_PANEL, fg=config.C_MUTED,
            font=("Courier New", 7), wraplength=200, justify=tk.LEFT)
        self._file_label.pack(fill=tk.X, padx=10, pady=2)

        # Camera simulation
        section("ESP32-CAM SIMULATION")
        tk.Label(parent, text="Resolution", bg=config.C_PANEL,
                 fg=config.C_TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=10)
        self._res_menu = ttk.Combobox(
            parent, textvariable=self._sim_res,
            values=list(config.RESOLUTION_PRESETS.keys()),
            state="readonly", font=("Courier New", 8))
        self._res_menu.pack(fill=tk.X, padx=10, pady=2)
        self._res_menu.bind("<<ComboboxSelected>>", self._on_resolution_change)

        tk.Label(parent, text="FPS (simulated)", bg=config.C_PANEL,
                 fg=config.C_TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=10, pady=(5,0))
        tk.Scale(parent, from_=1, to=30, orient=tk.HORIZONTAL,
                 variable=self._sim_fps,
                 bg=config.C_PANEL, fg=config.C_TEXT,
                 troughcolor=config.C_BORDER, activebackground=config.C_ACCENT,
                 highlightthickness=0, font=("Courier New", 7),
                 command=self._on_fps_change).pack(fill=tk.X, padx=10)

        # LED matrix size
        section("LED MATRIX SIZE")
        tk.Label(parent, text="Columns × Rows", bg=config.C_PANEL,
                 fg=config.C_TEXT, font=("Courier New", 8)).pack(anchor=tk.W, padx=10)
        self._matrix_menu = ttk.Combobox(
            parent, textvariable=self._matrix_preset,
            values=list(config.MATRIX_PRESETS.keys()),
            state="readonly", font=("Courier New", 8))
        self._matrix_menu.pack(fill=tk.X, padx=10, pady=2)
        self._matrix_menu.bind("<<ComboboxSelected>>", self._on_matrix_change)

        # Note about segment size
        self._seg_label = tk.Label(
            parent, text="Segment: —",
            bg=config.C_PANEL, fg=config.C_MUTED,
            font=("Courier New", 7))
        self._seg_label.pack(anchor=tk.W, padx=10)

        # Display options
        section("DISPLAY OPTIONS")
        for text, var in [("Show bounding boxes", self._show_boxes),
                          ("Show zone overlay",   self._show_zones)]:
            tk.Checkbutton(parent, text=text, variable=var,
                           bg=config.C_PANEL, fg=config.C_TEXT,
                           selectcolor=config.C_CARD,
                           activebackground=config.C_PANEL,
                           font=("Courier New", 8), anchor=tk.W).pack(
                fill=tk.X, padx=10, pady=1)

        # Playback
        section("PLAYBACK")
        btn_f = tk.Frame(parent, bg=config.C_PANEL)
        btn_f.pack(fill=tk.X, padx=10, pady=4)

        self._play_btn = tk.Button(
            btn_f, text="▶  PLAY",
            bg=config.C_ACCENT2, fg=config.C_BG,
            font=("Courier New", 9, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_play_pause,
            activebackground="#2a7ae6")
        self._play_btn.pack(fill=tk.X, pady=2)

        tk.Button(btn_f, text="■  STOP",
                  bg=config.C_CARD, fg=config.C_TEXT,
                  font=("Courier New", 9), relief=tk.FLAT, cursor="hand2",
                  command=self._on_stop,
                  activebackground=config.C_BORDER).pack(fill=tk.X, pady=2)

        # Legend
        section("LEGEND")
        for color, label in [
            (config.C_LED_ON,       "Full beam"),
            (config.C_LED_SUPPRESS, "Shadow (glare)"),
            (config.C_LED_PENUMBRA, "Penumbra edge"),
            (config.C_LED_BOOST,    "Hazard / flash"),
        ]:
            row = tk.Frame(parent, bg=config.C_PANEL)
            row.pack(fill=tk.X, padx=10, pady=1)
            dot = tk.Canvas(row, width=14, height=14,
                            bg=config.C_PANEL, highlightthickness=0)
            dot.pack(side=tk.LEFT)
            dot.create_oval(1, 1, 13, 13, fill=color, outline="")
            tk.Label(row, text=label, bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack(side=tk.LEFT, padx=4)

        self._style_combobox()

    def _style_combobox(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
            fieldbackground=config.C_CARD, background=config.C_CARD,
            foreground=config.C_TEXT, bordercolor=config.C_BORDER,
            arrowcolor=config.C_ACCENT, insertcolor=config.C_TEXT)

    # ── Video panel ───────────────────────────────────────────────────────────

    def _build_video_panel(self, parent):
        tk.Label(parent, text="VIDEO FEED  (simulated ESP32-CAM)",
                 bg=config.C_BG, fg=config.C_MUTED,
                 font=("Courier New", 8, "bold")).pack(anchor=tk.W, pady=(0,4))

        self._video_canvas = tk.Canvas(
            parent, bg=config.C_CARD,
            highlightthickness=1, highlightbackground=config.C_BORDER)
        self._video_canvas.pack(fill=tk.BOTH, expand=True)
        self._video_canvas.bind("<Configure>",
            lambda e: self._draw_placeholder() if not self._pipeline_running else None)
        self._draw_placeholder()

        stats = tk.Frame(parent, bg=config.C_BG)
        stats.pack(fill=tk.X, pady=(4,0))
        self._res_stat = tk.Label(stats, text="Resolution: —",
                                  bg=config.C_BG, fg=config.C_MUTED,
                                  font=("Courier New", 8))
        self._det_stat = tk.Label(stats, text="Detections: 0",
                                  bg=config.C_BG, fg=config.C_MUTED,
                                  font=("Courier New", 8))
        self._res_stat.pack(side=tk.LEFT, padx=4)
        self._det_stat.pack(side=tk.LEFT, padx=12)

    def _draw_placeholder(self):
        self._video_canvas.delete("placeholder")
        w = self._video_canvas.winfo_width()  or 640
        h = self._video_canvas.winfo_height() or 360
        self._video_canvas.create_text(
            w//2, h//2,
            text="Upload a video file to begin\n\nSimulates ESP32-CAM MJPEG stream",
            fill=config.C_MUTED, font=("Courier New", 11),
            justify=tk.CENTER, tags="placeholder")

    # ── Matrix panel ──────────────────────────────────────────────────────────

    def _build_matrix_panel(self, parent):
        tk.Label(parent, text="LED BEAM MATRIX",
                 bg=config.C_PANEL, fg=config.C_ACCENT,
                 font=("Courier New", 10, "bold")).pack(pady=(12,0))

        self._matrix_dim_label = tk.Label(
            parent, text="",
            bg=config.C_PANEL, fg=config.C_MUTED,
            font=("Courier New", 7))
        self._matrix_dim_label.pack()

        # Angle labels row (rebuilt when matrix changes)
        self._angle_frame = tk.Frame(parent, bg=config.C_PANEL)
        self._angle_frame.pack(pady=(6,0))

        # Matrix canvas — fixed pixel size; cells scale inside it
        self._matrix_canvas = tk.Canvas(
            parent, bg=config.C_PANEL,
            highlightthickness=0,
            width=self._MATRIX_PX, height=self._MATRIX_PX)
        self._matrix_canvas.pack(pady=4)

        # Distance labels (rebuilt when matrix changes)
        self._dist_frame = tk.Frame(parent, bg=config.C_PANEL)
        self._dist_frame.pack()

        # Zone state bar
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(
            fill=tk.X, padx=12, pady=8)
        tk.Label(parent, text="ZONE STATES", bg=config.C_PANEL,
                 fg=config.C_MUTED, font=("Courier New", 7, "bold")).pack()
        self._zone_bar = tk.Canvas(parent, bg=config.C_PANEL,
                                   height=34, highlightthickness=0)
        self._zone_bar.pack(fill=tk.X, padx=10, pady=4)

        # Detection list
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(
            fill=tk.X, padx=12, pady=(4,8))
        tk.Label(parent, text="DETECTIONS", bg=config.C_PANEL,
                 fg=config.C_MUTED, font=("Courier New", 7, "bold")).pack()
        self._det_list = tk.Frame(parent, bg=config.C_PANEL)
        self._det_list.pack(fill=tk.X, padx=10, pady=4)

        # Zone byte
        tk.Frame(parent, bg=config.C_BORDER, height=1).pack(
            fill=tk.X, padx=12, pady=4)
        self._zone_byte_var = tk.StringVar(value="zone byte: 0xFF")
        tk.Label(parent, textvariable=self._zone_byte_var,
                 bg=config.C_PANEL, fg=config.C_ACCENT2,
                 font=("Courier New", 9, "bold")).pack(pady=4)

    def _rebuild_matrix_canvas(self):
        """
        Rebuild the LED oval grid whenever ZONE_COUNT or ROW_LEDS changes.
        Cell size = MATRIX_PX / dimension, so every matrix fits the same panel.
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        cell_w = self._MATRIX_PX / cols
        cell_h = self._MATRIX_PX / rows
        pad    = 0.12   # fraction of cell for gap between ovals

        self._matrix_canvas.delete("all")
        self._led_ovals.clear()

        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w + cell_w * pad
                y1 = r * cell_h + cell_h * pad
                x2 = (c+1) * cell_w - cell_w * pad
                y2 = (r+1) * cell_h - cell_h * pad
                oval = self._matrix_canvas.create_oval(
                    x1, y1, x2, y2,
                    fill=config.C_LED_OFF,
                    outline="#0f1f35", width=1)
                self._led_ovals[(c, r)] = oval

        # Rebuild angle labels
        for w in self._angle_frame.winfo_children():
            w.destroy()
        deg_per_zone = config.ZONE_SPREAD / cols
        for c in range(cols):
            angle = -config.ZONE_SPREAD/2 + deg_per_zone * (c + 0.5)
            lbl = f"{angle:+.0f}°" if angle != 0 else "0°"
            tk.Label(self._angle_frame, text=lbl,
                     bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 5),
                     width=max(3, 28 // cols)).pack(side=tk.LEFT)

        # Rebuild distance labels
        for w in self._dist_frame.winfo_children():
            w.destroy()
        distances = ["100m+", "80m", "60m", "40m", "25m", "15m", "8m", "wide"]
        for r in range(rows):
            label = distances[r] if r < len(distances) else f"r{r}"
            tk.Label(self._dist_frame, text=label,
                     bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 6)).pack(anchor=tk.W)

        # Update dimension label
        self._matrix_dim_label.config(
            text=f"{cols}×{rows} matrix  |  {config.ZONE_SPREAD/cols:.1f}° per zone")

        # Update segment size label
        self._update_segment_label()

    def _update_segment_label(self):
        key = self._sim_res.get()
        fw, fh = config.RESOLUTION_PRESETS.get(key, (640, 360))
        sw = fw / config.ZONE_COUNT
        sh = fh / config.ROW_LEDS
        self._seg_label.config(
            text=f"Segment: {sw:.0f}×{sh:.0f} px per LED")

    # ═══════════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                       ("All files", "*.*")])
        if not path:
            return
        if not self._feeder.load(path):
            self._status_var.set(f"Error: cannot open {os.path.basename(path)}")
            return
        fname = os.path.basename(path)
        self._file_label.config(text=fname)
        self._status_var.set(
            f"Loaded: {fname}  "
            f"({self._feeder.video_w}×{self._feeder.video_h}, "
            f"{self._feeder.video_fps:.0f} fps, "
            f"{self._feeder.duration_s:.0f}s)")
        self._update_segment_label()

    def _on_resolution_change(self, _=None):
        key = self._sim_res.get()
        w, h = config.RESOLUTION_PRESETS.get(key, (640, 360))
        self._feeder.set_resolution(w, h)
        self._update_segment_label()

    def _on_fps_change(self, _=None):
        self._feeder.set_fps(self._sim_fps.get())

    def _on_matrix_change(self, _=None):
        """User changed matrix size → update config, reset controller, rebuild canvas."""
        key = self._matrix_preset.get()
        cols, rows = config.MATRIX_PRESETS.get(key, (8, 8))
        config.ZONE_COUNT = cols
        config.ROW_LEDS   = rows

        # Controller must reinitialise with new dimensions
        self._controller.reset()

        # Rebuild the canvas grid and labels
        self._rebuild_matrix_canvas()

        # Reset zone bar width proportions
        self._render_zone_bar()

        self._status_var.set(
            f"Matrix changed to {cols}×{rows} — "
            f"segment size: {config.RESOLUTION_PRESETS.get(self._sim_res.get(),(640,360))[0]/cols:.0f}"
            f"×{config.RESOLUTION_PRESETS.get(self._sim_res.get(),(640,360))[1]/rows:.0f} px per LED")

    def _on_play_pause(self):
        if not self._feeder._path:
            self._status_var.set("Please upload a video file first.")
            return
        if self._pipeline_running and not self._feeder.is_paused:
            self._feeder.pause()
            self._play_btn.config(text="▶  RESUME")
            self._status_var.set("Paused")
        elif self._pipeline_running and self._feeder.is_paused:
            self._feeder.resume()
            self._play_btn.config(text="⏸  PAUSE")
            self._status_var.set("Playing…")
        else:
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
        key = self._sim_res.get()
        w, h = config.RESOLUTION_PRESETS.get(key, (640, 360))
        self._feeder.set_resolution(w, h)
        self._feeder.set_fps(self._sim_fps.get())

        self._feeder.start()
        self._detector.start()

        self._pipeline_running = True
        self._frame_count      = 0
        self._fps_timer        = time.time()
        self._play_btn.config(text="⏸  PAUSE")
        self._status_var.set("Playing…")

        self.root.after(16, self._ui_poll)

    def _stop_pipeline(self):
        self._pipeline_running = False
        self._feeder.stop()
        self._detector.stop()
        for q in (self._frame_q, self._result_q):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break

    # ═══════════════════════════════════════════════════════════════════════════
    # UI POLL LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    def _ui_poll(self):
        if not self._pipeline_running:
            return

        try:
            raw_frame, detections = self._result_q.get_nowait()
        except queue.Empty:
            self.root.after(16, self._ui_poll)
            return

        # 1. Update LED controller with new detections
        self._controller.update(detections)

        # 2. Annotate frame ONCE — after controller state is current
        annotated = annotate_frame(
            raw_frame, detections,
            led_states  = self._controller.led_states,
            zone_states = self._controller.zone_states,
            show_boxes  = self._show_boxes.get(),
            show_zones  = self._show_zones.get(),
        )

        # 3. Render
        self._show_frame(annotated)
        self._render_matrix()
        self._render_zone_bar()
        self._render_detection_list(detections)
        self._render_zone_byte()

        # 4. Stats
        fh, fw = raw_frame.shape[:2]
        self._res_stat.config(text=f"Resolution: {fw}×{fh}")
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
                f"| sim: {self._sim_fps.get()} fps")

        self.root.after(16, self._ui_poll)

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDER HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_frame(self, frame):
        cw = self._video_canvas.winfo_width()
        ch = self._video_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        fh, fw = frame.shape[:2]
        scale = min(cw/fw, ch/fh)
        nw, nh = int(fw*scale), int(fh*scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._video_canvas.delete("frame", "placeholder")
        ox = (cw-nw)//2
        oy = (ch-nh)//2
        self._video_canvas.create_image(ox, oy, anchor=tk.NW,
                                        image=photo, tags="frame")
        self._video_canvas._photo = photo   # keep reference

    def _render_matrix(self):
        for (c, r), oval_id in self._led_ovals.items():
            color, brightness = self._controller.led_color(c, r)
            fill = _blend(config.C_LED_OFF, color, max(0.04, brightness))
            self._matrix_canvas.itemconfig(oval_id, fill=fill)

    def _render_zone_bar(self):
        c = self._zone_bar
        c.delete("all")
        cw = max(c.winfo_width() or self._MATRIX_PX, 1)
        cols = config.ZONE_COUNT
        cell_w = cw / cols
        color_map = {
            "on":       config.C_LED_ON,
            "suppress": config.C_LED_SUPPRESS,
            "boost":    config.C_LED_BOOST,
        }
        for col_i, state in enumerate(self._controller.zone_states):
            x1 = col_i * cell_w + 1
            x2 = x1 + cell_w - 2
            fill = color_map.get(state, config.C_LED_ON)
            c.create_rectangle(x1, 2, x2, 32, fill=fill, outline="")
            if cell_w >= 18:   # only show number if cell is wide enough
                c.create_text((x1+x2)/2, 17, text=str(col_i+1),
                              fill=config.C_BG, font=("Courier New", 7, "bold"))

    def _render_detection_list(self, detections):
        for w in self._det_list.winfo_children():
            w.destroy()
        if not detections:
            tk.Label(self._det_list, text="no detections",
                     bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack()
            return
        color_map = {"glare": config.C_LED_SUPPRESS,
                     "hazard": config.C_LED_BOOST,
                     "other": config.C_MUTED}
        for det in detections[:8]:
            cat = det["category"]
            col = self._controller._cx_to_col(det["cx_norm"])
            row = self._controller._cy_to_row(det["cy_norm"])
            row_f = tk.Frame(self._det_list, bg=config.C_PANEL)
            row_f.pack(fill=tk.X, pady=1)
            tk.Label(row_f, text=det["label"],
                     bg=config.C_PANEL, fg=config.C_TEXT,
                     font=("Courier New", 8)).pack(side=tk.LEFT)
            tk.Label(row_f, text=f" {det['conf']:.0%}",
                     bg=config.C_PANEL, fg=config.C_MUTED,
                     font=("Courier New", 7)).pack(side=tk.LEFT)
            if cat != "other":
                tk.Label(row_f, text=f" [{cat.upper()}]",
                         bg=config.C_PANEL, fg=color_map[cat],
                         font=("Courier New", 7, "bold")).pack(side=tk.LEFT)
            tk.Label(row_f, text=f" C{col+1} R{row+1}",
                     bg=config.C_PANEL, fg=config.C_ACCENT2,
                     font=("Courier New", 7)).pack(side=tk.RIGHT)

    def _render_zone_byte(self):
        cols     = config.ZONE_COUNT
        byte_val = self._controller.zone_byte()
        hex_digits = (cols + 3) // 4
        binary   = format(byte_val, f"0{cols}b")
        # Split binary into groups of 4 for readability
        groups = [binary[i:i+4] for i in range(0, len(binary), 4)]
        self._zone_byte_var.set(
            f"0x{byte_val:0{hex_digits}X}  ({'_'.join(groups)})")

    # ── Flash ticker ──────────────────────────────────────────────────────────

    def _start_flash_ticker(self):
        self._controller.tick_flash()
        self._flash_job = self.root.after(
            config.FLASH_INTERVAL_MS, self._start_flash_ticker)

    # ── Shutdown ──────────────────────────────────────────────────────────────

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
