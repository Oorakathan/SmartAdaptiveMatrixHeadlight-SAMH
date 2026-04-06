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
ZONE_COUNT = 24
ROW_LEDS   = 8

# ── Zone geometry ─────────────────────────────────────────────────────────────
ZONE_SPREAD = 120           # total horizontal FOV in degrees (±60°)

# ── AI processing resolution ─────────────────────────────────────────────────
AI_FRAME_W = 640
AI_FRAME_H = 360

# ── Detection thresholds ─────────────────────────────────────────────────────
MODEL_NAME = "foduucom.pt"
CONFIDENCE_THRESHOLD   = 0.40
MIN_CONFIDENCE_GLARE   = 0.50
MIN_CONFIDENCE_HAZARD  = 0.40

# ── Object tracking (IoU-based cross-frame assignment) ───────────────────────
# Prevents detections from "jumping" between frames, eliminating shadow jitter
TRACKER_IOU_THRESHOLD   = 0.25   # min IoU to link a detection to an existing track
TRACKER_MAX_LOST_FRAMES = 12     # frames to keep a track alive when not matched
                                  # (at 15fps = 0.8 s; prevents brief-occlusion flicker)
TRACKER_MIN_HIT_STREAK  = 2      # detections needed to "confirm" a new track
                                  # (prevents single-frame ghost detections)

# ── Position smoothing (exponential moving average on track centre) ───────────
# Smooths the shadow position so it glides rather than jumps
# alpha=0.0 → fully frozen; alpha=1.0 → no smoothing (raw detection)
# 0.25 ≈ ~4-frame time constant — visually smooth at 15-30 fps
TRACK_POS_ALPHA = 0.25     # EMA weight for cx_norm / cy_norm
TRACK_SIZE_ALPHA = 0.20    # EMA weight for w_norm / h_norm (slower = smoother size)

# ── LED brightness smoothing ─────────────────────────────────────────────────
# Per-LED brightness is an EMA so it fades in/out rather than snapping
# Lower = slower fade (smoother); higher = faster response
LED_BRIGHTNESS_ALPHA_DOWN = 0.10   # fade-OUT speed (suppress turning on = dims slowly)
LED_BRIGHTNESS_ALPHA_UP   = 0.18   # fade-IN speed  (suppress releasing = brightens slower)
                                    # UP slower than DOWN: don't flash bright on brief miss

# ── Temporal filtering ────────────────────────────────────────────────────────
# EMA-based confidence tracking replaces the old integer streak counter.
# A track's "LED influence" = EMA of confidence. Activates above ON_THRESHOLD,
# deactivates below OFF_THRESHOLD (hysteresis gap prevents chattering).
CONF_EMA_ALPHA       = 0.30         # EMA weight for per-track confidence
SUPPRESS_CONF_ON     = 0.45         # EMA conf must exceed this to activate shadow
SUPPRESS_CONF_OFF    = 0.20         # EMA conf must drop below this to release shadow
BOOST_CONF_ON        = 0.38
BOOST_CONF_OFF       = 0.18

# Legacy names kept for backward compat (used nowhere in new code but avoids import errors)
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

# ── Shadow cone geometry (normalised to matrix size) ─────────────────────────
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
