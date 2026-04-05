"""
config.py
---------
All shared constants for the Smart Adaptive Matrix Headlamp simulator.

IMPORTANT — matrix dimensions (ZONE_COUNT, ROW_LEDS) are defined here as
DEFAULTS. At runtime they are overridden on the MatrixConfig singleton so
every module reads the live value. Do NOT do `from config import ZONE_COUNT`
and cache it — always read config.ZONE_COUNT at call time.
"""

# ── Matrix dimensions (defaults — overridden at runtime) ─────────────────────
# These represent the physical LED array:
#   ZONE_COUNT  = number of horizontal columns  (angular zones, left→right)
#   ROW_LEDS    = number of vertical rows       (distance bands, far→near)
#
# Segment size on the video frame = (frame_w / ZONE_COUNT, frame_h / ROW_LEDS)
# Works for any integer dimensions: 4×4, 8×8, 16×8, 16×16, etc.
ZONE_COUNT = 24
ROW_LEDS   = 8

# ── Zone geometry ─────────────────────────────────────────────────────────────
ZONE_SPREAD = 120           # total horizontal FOV in degrees (±60°)

# ── AI processing resolution ─────────────────────────────────────────────────
AI_FRAME_W = 640
AI_FRAME_H = 360

# ── Detection thresholds ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.40  # minimum YOLO confidence to accept any detection
MIN_CONFIDENCE_GLARE   = 0.50  # higher bar for glare (false positives kill vision)
MIN_CONFIDENCE_HAZARD  = 0.40  # lower bar for hazard (miss is worse than false positive)

# ── Temporal filtering ────────────────────────────────────────────────────────
# Hysteresis: detect must persist N frames to activate, decays slowly on absence
SUPPRESS_TEMPORAL_ON  = 2    # frames to confirm before suppressing
SUPPRESS_TEMPORAL_OFF = 5    # frames to hold after object disappears (decay)
BOOST_TEMPORAL_ON     = 1    # hazard alerts activate in 1 frame (urgent)
BOOST_TEMPORAL_OFF    = 8    # hazard hold after disappearing

# ── Object class → action mapping ────────────────────────────────────────────
# GLARE  : emits blinding headlights → suppress LED zone
# HAZARD : vulnerable road user      → boost LED zone + flash alert
GLARE_CLASSES  = {"car", "truck", "bus"}
HAZARD_CLASSES = {"person", "bicycle", "motorcycle",
                  "dog", "cat", "bird", "horse", "cow", "sheep"}

# ── LED brightness levels ─────────────────────────────────────────────────────
BRIGHTNESS_FULL     = 1.00   # full beam — LED at max
BRIGHTNESS_SUPPRESS = 0.00   # shadow centre — LED off
BRIGHTNESS_PENUMBRA = 0.30   # shadow soft edge (realistic penumbra)
BRIGHTNESS_FAR_ROW  = 0.95   # row 0 (100 m+) narrow high beam
BRIGHTNESS_NEAR_ROW = 0.60   # last row (wide flood) — dimmer for safety

# ── Shadow cone geometry (normalised to matrix size) ─────────────────────────
# All radii expressed as FRACTION of matrix dimensions so they scale with
# any matrix size.  e.g. COL_RADIUS_BASE = 0.20 on an 8-col matrix = 1.6 cols
#                                          on a 16-col matrix = 3.2 cols
SHADOW_COL_RADIUS_BASE  = 0.12   # base column influence (fraction of ZONE_COUNT)
SHADOW_ROW_RADIUS_BASE  = 0.12   # base row influence (fraction of ROW_LEDS)
SHADOW_COL_CONF_SCALE   = 0.06   # extra col radius per unit confidence above MIN
SHADOW_ROW_DIST_SCALE   = 0.15   # extra row radius for far objects (realistic cone)
SHADOW_PENUMBRA_FALLOFF = 2.2    # steepness of brightness gradient (higher = sharper)
SHADOW_MAX_DIST_NORM    = 1.4    # normalised distance beyond which LED stays on

# ── Hazard boost geometry ─────────────────────────────────────────────────────
BOOST_COL_RADIUS = 0.15   # fraction of ZONE_COUNT
BOOST_ROW_RADIUS = 0.15   # fraction of ROW_LEDS

# ── Flash settings ────────────────────────────────────────────────────────────
FLASH_HZ          = 3
FLASH_INTERVAL_MS = int(1000 / (FLASH_HZ * 2))   # ms per half-cycle ≈ 167 ms

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

# LED colours
C_LED_ON       = "#00C2A8"   # teal  — full beam
C_LED_SUPPRESS = "#EF4444"   # red   — shadow (glare suppression)
C_LED_BOOST    = "#F59E0B"   # amber — hazard alert
C_LED_PENUMBRA = "#2DD4BF"   # light teal — penumbra edge
C_LED_OFF      = "#0d1a2e"   # near-black — unlit

# OpenCV overlay colours (BGR)
CV_COLOR_GLARE  = (50,  50, 220)
CV_COLOR_HAZARD = (0,  160, 240)
CV_COLOR_SAFE   = (0,  180, 140)
CV_ALPHA_GLARE  = 0.22
CV_ALPHA_HAZARD = 0.18
CV_ALPHA_SAFE   = 0.06

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

# ── Matrix size presets (cols × rows) ────────────────────────────────────────
MATRIX_PRESETS = {
    "4×4   (minimal)":    (4,  4),
    "8×8   (prototype)":  (8,  8),
    "12×8  (wide)":       (12, 8),
    "16×8  (semi-pixel)": (16, 8),
    "16×16 (full pixel)": (16, 16),
}
DEFAULT_MATRIX = "8×8   (prototype)"
