"""
config.py
---------
All shared constants for the Smart Adaptive Matrix Headlamp simulator.
Edit this file to tune behaviour — nothing else needs to change.
"""

# ── Zone geometry ─────────────────────────────────────────────────────────────
ZONE_COUNT = 8        # horizontal angular zones (columns on LED matrix)
ZONE_SPREAD = 120     # total horizontal FOV in degrees (±60°)
ZONE_DEG = ZONE_SPREAD / ZONE_COUNT  # degrees per zone = 15°

# ── AI processing resolution ─────────────────────────────────────────────────
# Frames are always scaled to this before running detection.
# Smaller = faster inference; keep aspect ratio same as camera.
AI_FRAME_W = 640
AI_FRAME_H = 360

# ── Detection thresholds ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40   # minimum YOLO confidence to accept a detection
SUPPRESS_TEMPORAL    = 3      # frames a glare object must appear before suppressing
                              # (prevents single-frame false positives)

# ── Object class → action mapping ────────────────────────────────────────────
# Add or remove class names to change what the system reacts to.
GLARE_CLASSES  = {"car", "truck", "motorcycle", "bus"}
HAZARD_CLASSES = {"person", "dog", "cat", "bird", "horse", "cow", "sheep"}

# ── LED brightness levels ─────────────────────────────────────────────────────
BRIGHTNESS_FULL     = 1.0    # full beam LED
BRIGHTNESS_ADJACENT = 0.35   # LED ±1 next to a shadow (soft cutoff)
BRIGHTNESS_SUPPRESS = 0.0    # shadow LED centre
BRIGHTNESS_FAR_ROW  = 1.0    # row 0 (100m+) boost
BRIGHTNESS_NEAR_ROW = 0.7    # row 7 (wide flood) base

# ── LED individual control parameters ─────────────────────────────────────────
# Defines how detections affect individual LEDs in the 8x8 matrix
LED_INFLUENCE_COL_RADIUS = 1   # how many columns away an LED is affected (±N columns)
LED_INFLUENCE_ROW_RADIUS = 1   # how many rows away an LED is affected (±N rows)
LED_ADJACENT_BRIGHTNESS  = 0.35  # brightness for LEDs adjacent to suppressed area
LED_MIN_ROW7_BRIGHTNESS  = 0.30  # minimum brightness for row 7 (wide flood, always partially on)

# ── Flash settings ───────────────────────────────────────────────────────────
FLASH_HZ         = 4         # hazard flash frequency in Hz
FLASH_INTERVAL_MS = int(1000 / (FLASH_HZ * 2))  # ms per half-cycle = 125ms

# ── UI colours (hex) ─────────────────────────────────────────────────────────
C_BG       = "#0A0F1E"
C_PANEL    = "#111827"
C_CARD     = "#1a2235"
C_BORDER   = "#1f2d44"
C_ACCENT   = "#00C2A8"
C_ACCENT2  = "#3A8EF6"
C_TEXT     = "#E2E8F0"
C_MUTED    = "#64748B"

# LED colours
C_LED_ON      = "#00C2A8"   # teal  — full beam
C_LED_SUPPRESS= "#EF4444"   # red   — shadow zone (glare)
C_LED_BOOST   = "#F59E0B"   # amber — hazard boost / flash
C_LED_OFF     = "#0d1a2e"   # near-black — LED unlit

# Overlay colours on the video feed (BGR for OpenCV)
CV_COLOR_GLARE  = (50,  50, 220)   # red tint
CV_COLOR_HAZARD = (0,  160, 240)   # amber tint
CV_COLOR_SAFE   = (0,  180, 140)   # teal tint
CV_ALPHA_GLARE  = 0.25
CV_ALPHA_HAZARD = 0.20
CV_ALPHA_SAFE   = 0.08

# ── ESP32-CAM simulation presets ──────────────────────────────────────────────
# label → (width, height)
RESOLUTION_PRESETS = {
    "160×120  (QQVGA — lowest)":     (160, 120),
    "320×240  (QVGA — standard)":    (320, 240),
    "640×360  (ESP32-CAM high)":     (640, 360),
    "800×600  (SVGA)":               (800, 600),
    "1280×720 (HD)":                 (1280, 720),
}
DEFAULT_RESOLUTION = "640×360  (ESP32-CAM high)"
DEFAULT_FPS        = 15
