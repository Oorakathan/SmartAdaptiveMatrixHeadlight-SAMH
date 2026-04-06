"""
config.py
---------
All shared constants for the Smart Adaptive Matrix Headlamp simulator.

New in this version
-------------------
* ROI_HORIZON_RATIO  — detections above this fraction of frame height are
                       discarded (sky, signboards, overpass structures).
                       Calibrate once per car: mount camera, take a still,
                       measure where the road horizon falls as a fraction.
                       Typical dashcam on sedan: 0.35–0.45.

* Drive modes (HIGHWAY / EXPRESSWAY / CITY):
    HIGHWAY     — full adaptive matrix, all rows active
    EXPRESSWAY  — full matrix, slightly tighter ROI (flyovers, overpasses)
    CITY        — low beam: top CITY_DARK_ROW_FRACTION forced off,
                  pixel-wise control still works on remaining rows

* DENSITY_*  — parameters for the auto scene-density estimator that switches
               modes automatically. EMA-smoothed vehicle count prevents a
               single passing car from toggling city mode on and off.
"""

# ── Matrix dimensions ─────────────────────────────────────────────────────────
ZONE_COUNT = 8
ROW_LEDS   = 8

# ── Zone geometry ─────────────────────────────────────────────────────────────
ZONE_SPREAD = 120

# ── AI processing resolution ─────────────────────────────────────────────────
AI_FRAME_W = 640
AI_FRAME_H = 360

# ── Detection thresholds ─────────────────────────────────────────────────────
MODEL_NAME             = "foduucom.pt"
CONFIDENCE_THRESHOLD   = 0.40
MIN_CONFIDENCE_GLARE   = 0.50
MIN_CONFIDENCE_HAZARD  = 0.40

# ── ROI horizon mask ─────────────────────────────────────────────────────────
# Detections whose road-relevant y position (y1_norm for glare, cy_norm for
# hazard) falls ABOVE (i.e. numerically less than) this value are discarded.
# 0.0 = top of frame, 1.0 = bottom of frame.
# --- Calibration guide ---
# 1. Mount the camera in its final position in the car.
# 2. Take a still frame on a flat, empty road.
# 3. In any image editor, note the pixel row of the visible horizon.
# 4. ROI_HORIZON_RATIO = horizon_row_px / frame_height_px
# Typical values: 0.38 (sedan), 0.42 (SUV), 0.45 (truck)
ROI_HORIZON_RATIO  = 0.40    # fraction of frame height — tune per vehicle
ROI_SIDE_MARGIN    = 0.04    # fraction of frame width clipped on each side edge
                              # (removes bonnet/mirror reflections)

# ── Drive mode identifiers ────────────────────────────────────────────────────
DRIVE_MODE_HIGHWAY    = "highway"
DRIVE_MODE_EXPRESSWAY = "expressway"
DRIVE_MODE_CITY       = "city"

# ── City / expressway beam shaper ─────────────────────────────────────────────
# Top N rows forced to brightness 0 in each mode.
# Expressed as fraction of ROW_LEDS so it scales with any matrix size.
# city 0.50 on 8-row → top 4 off → matches ECE R112 class C low-beam cutoff.
CITY_DARK_ROW_FRACTION       = 0.50    # fraction of rows forced off in CITY
EXPRESSWAY_DARK_ROW_FRACTION = 0.25    # lighter cut in EXPRESSWAY

# ── Scene density estimator ───────────────────────────────────────────────────
# EMA of confirmed glare track count, used to drive automatic mode switching.
# alpha=0.10 → ~10-frame time constant (≈0.65 s at 15fps)
DENSITY_EMA_ALPHA        = 0.10
DENSITY_CITY_ENTER       = 2.5    # EMA count >= this → enter CITY
DENSITY_CITY_EXIT        = 1.2    # EMA count <  this → leave CITY
DENSITY_EXPRESSWAY_ENTER = 0.4    # EMA count >= this → enter EXPRESSWAY
DENSITY_EXPRESSWAY_EXIT  = 0.15   # EMA count <  this → leave EXPRESSWAY
# Debounce: must sustain new mode for N frames before committing the switch
DENSITY_MODE_HOLD_FRAMES = 20     # 20 frames @ 15fps ≈ 1.3 s

# ── Object tracking ───────────────────────────────────────────────────────────
TRACKER_IOU_THRESHOLD   = 0.25
TRACKER_MAX_LOST_FRAMES = 12
TRACKER_MIN_HIT_STREAK  = 2

# ── Position smoothing ────────────────────────────────────────────────────────
TRACK_POS_ALPHA  = 0.25
TRACK_SIZE_ALPHA = 0.20

# ── LED brightness EMA ────────────────────────────────────────────────────────
LED_BRIGHTNESS_ALPHA_DOWN = 0.10    # suppress activating (dims slowly)
LED_BRIGHTNESS_ALPHA_UP   = 0.18    # suppress releasing  (brightens slower than dims)

# ── EMA confidence thresholds ────────────────────────────────────────────────
CONF_EMA_ALPHA    = 0.30
SUPPRESS_CONF_ON  = 0.45
SUPPRESS_CONF_OFF = 0.20
BOOST_CONF_ON     = 0.38
BOOST_CONF_OFF    = 0.18

# Legacy names (kept for backward compat)
SUPPRESS_TEMPORAL_ON  = 2
SUPPRESS_TEMPORAL_OFF = 5
BOOST_TEMPORAL_ON     = 1
BOOST_TEMPORAL_OFF    = 8

# ── Object class → action mapping ────────────────────────────────────────────
GLARE_CLASSES  = {"car", "truck", "bus"}
HAZARD_CLASSES = {"person", "bicycle", "motorcycle",
                  "dog", "cat", "bird", "horse", "cow", "sheep"}

# ── LED brightness levels ─────────────────────────────────────────────────────
BRIGHTNESS_FULL     = 1.00
BRIGHTNESS_SUPPRESS = 0.00
BRIGHTNESS_PENUMBRA = 0.30
BRIGHTNESS_FAR_ROW  = 0.95
BRIGHTNESS_NEAR_ROW = 0.60

# ── Shadow cone geometry ──────────────────────────────────────────────────────
SHADOW_COL_RADIUS_BASE  = 0.12
SHADOW_ROW_RADIUS_BASE  = 0.12
SHADOW_COL_CONF_SCALE   = 0.06
SHADOW_ROW_DIST_SCALE   = 0.15
SHADOW_PENUMBRA_FALLOFF = 2.2
SHADOW_MAX_DIST_NORM    = 1.4

# ── Hazard boost geometry ─────────────────────────────────────────────────────
BOOST_COL_RADIUS = 0.15
BOOST_ROW_RADIUS = 0.15

# ── Flash settings ────────────────────────────────────────────────────────────
FLASH_HZ          = 3
FLASH_INTERVAL_MS = int(1000 / (FLASH_HZ * 2))

# ── UI colours ────────────────────────────────────────────────────────────────
C_BG      = "#0A0F1E"
C_PANEL   = "#111827"
C_CARD    = "#1a2235"
C_BORDER  = "#1f2d44"
C_ACCENT  = "#00C2A8"
C_ACCENT2 = "#3A8EF6"
C_TEXT    = "#E2E8F0"
C_MUTED   = "#64748B"
C_DANGER  = "#DC2626"

C_LED_ON       = "#00C2A8"
C_LED_SUPPRESS = "#EF4444"
C_LED_BOOST    = "#F59E0B"
C_LED_PENUMBRA = "#2DD4BF"
C_LED_OFF      = "#0d1a2e"

C_MODE_HIGHWAY    = "#00C2A8"
C_MODE_EXPRESSWAY = "#3A8EF6"
C_MODE_CITY       = "#F59E0B"

CV_COLOR_GLARE   = (50,  50, 220)
CV_COLOR_HAZARD  = (0,  160, 240)
CV_COLOR_SAFE    = (0,  180, 140)
CV_COLOR_HORIZON = (0,  220, 100)
CV_ALPHA_GLARE   = 0.22
CV_ALPHA_HAZARD  = 0.18
CV_ALPHA_SAFE    = 0.06
CV_ALPHA_HORIZON = 0.0   # set >0 to show horizon debug line on video

# ── Camera simulation presets ─────────────────────────────────────────────────
RESOLUTION_PRESETS = {
    "160×120  (QQVGA)":      (160, 120),
    "320×240  (QVGA)":       (320, 240),
    "640×360  (ESP32-high)": (640, 360),
    "800×600  (SVGA)":       (800, 600),
    "1280×720 (HD)":         (1280, 720),
}
DEFAULT_RESOLUTION = "640×360  (ESP32-high)"
DEFAULT_FPS        = 15

# ── Matrix size presets ───────────────────────────────────────────────────────
MATRIX_PRESETS = {
    "4×4   (minimal)":    (4,  4),
    "8×8   (prototype)":  (8,  8),
    "12×8  (wide)":       (12, 8),
    "16×8  (semi-pixel)": (16, 8),
    "16×16 (full pixel)": (16, 16),
}
DEFAULT_MATRIX = "8×8   (prototype)"