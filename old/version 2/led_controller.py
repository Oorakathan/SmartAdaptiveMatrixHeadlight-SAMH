"""
led_controller.py
-----------------
Per-LED beam control — the "brain" of the headlamp.

Key improvements over previous version
---------------------------------------
* EMA-based confidence tracking replaces integer streak counters.
  Each LED now has a smoothed influence value that decays gently when the
  object is absent, and rises smoothly when it returns. This eliminates
  the "detected → gap → detected" disco flash pattern.

* Per-LED brightness EMA.
  The actual rendered brightness of each LED is a running average that
  fades to the target value over several frames. Suppression fades IN
  slowly (smooth dim-to-black) and releases OUT even more slowly
  (black-to-bright is gradual so a 1-frame miss doesn't flash the full beam).

* Hysteresis on state transitions.
  A LED transitions to "suppress" only when its EMA influence exceeds
  SUPPRESS_CONF_ON, and releases only when it drops below SUPPRESS_CONF_OFF
  (a lower threshold). This hysteresis gap prevents chattering at the edge.

* Track-ID-aware accumulation.
  Each frame, per-LED influence is the maximum EMA influence projected onto
  that LED from any active track. Multiple overlapping cars → darker shadow.

* Shadow cone uses the SMOOTHED track position (from ObjectTracker EMA),
  so even if the raw detection jitters ±3 columns, the shadow stays steady.

Public interface
----------------
  update(tracks)        → call once per frame (tracks from AIDetector)
  tick_flash()          → call every FLASH_INTERVAL_MS from UI timer
  led_color(col, row)   → (hex_color, brightness 0–1) for rendering
  zone_byte()           → bitmask int for Arduino serial
  reset()               → full beam, clear all state
  summary()             → one-line debug string
"""

import math
import config


class ZoneController:

    def __init__(self):
        self._init_state()

    def _init_state(self):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        # Logical state per LED
        self.led_states     = [["on"] * cols for _ in range(rows)]

        # Rendered brightness per LED (EMA-smoothed, 0.0–1.0)
        self.led_brightness = [[config.BRIGHTNESS_FULL] * cols for _ in range(rows)]

        # EMA influence accumulators — one for glare, one for hazard, per LED
        # Range 0.0–1.0. Built fresh each frame from track projections.
        self._glare_ema  = [[0.0] * cols for _ in range(rows)]
        self._hazard_ema = [[0.0] * cols for _ in range(rows)]

        # Current logical state (for hysteresis): True = active
        self._glare_active  = [[False] * cols for _ in range(rows)]
        self._hazard_active = [[False] * cols for _ in range(rows)]

        # Shadow target brightness per LED (0.0 = full suppress, 1.0 = full bright)
        # This is what the EMA moves toward.
        self._brightness_target = [[config.BRIGHTNESS_FULL] * cols for _ in range(rows)]

        # Flash
        self._flash_tick = 0
        self._flash_on   = True

        # Column-level summary
        self.zone_states = ["on"] * cols
        self.zone_conf   = [config.BRIGHTNESS_FULL] * cols

    def reset(self):
        self._init_state()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC — called once per frame
    # ══════════════════════════════════════════════════════════════════════════

    def update(self, tracks: list):
        """
        Recompute LED states from confirmed track list for this frame.

        tracks: list of dicts from AIDetector (smoothed position, EMA confidence)
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        # ── 1. Compute per-LED shadow/boost influence from tracks ─────────────
        # These are the "raw" influence values projected from tracks this frame.
        # We then EMA-blend them into the running accumulators.
        glare_proj  = [[0.0]   * cols for _ in range(rows)]   # 0 = no influence
        boost_proj  = [[False] * cols for _ in range(rows)]

        for track in tracks:
            cat  = track["category"]
            conf = track["conf"]   # already EMA-smoothed by tracker

            if cat == "glare":
                self._project_glare(track, conf, glare_proj)
            elif cat == "hazard":
                self._project_hazard(track, boost_proj)

        # ── 2. EMA-update running influence accumulators ──────────────────────
        alpha = config.CONF_EMA_ALPHA
        for r in range(rows):
            for c in range(cols):
                # Glare influence: move toward projected value
                self._glare_ema[r][c] = (
                    alpha * glare_proj[r][c]
                    + (1 - alpha) * self._glare_ema[r][c]
                )
                # Hazard influence: binary projected, EMA-smoothed
                target_h = 1.0 if boost_proj[r][c] else 0.0
                self._hazard_ema[r][c] = (
                    alpha * target_h
                    + (1 - alpha) * self._hazard_ema[r][c]
                )

        # ── 3. Hysteresis state transitions ───────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                g_inf = self._glare_ema[r][c]
                h_inf = self._hazard_ema[r][c]

                # Glare hysteresis
                if not self._glare_active[r][c]:
                    if g_inf >= config.SUPPRESS_CONF_ON:
                        self._glare_active[r][c] = True
                else:
                    if g_inf < config.SUPPRESS_CONF_OFF:
                        self._glare_active[r][c] = False

                # Hazard hysteresis (only when not suppressed)
                if not self._hazard_active[r][c]:
                    if h_inf >= config.BOOST_CONF_ON:
                        self._hazard_active[r][c] = True
                else:
                    if h_inf < config.BOOST_CONF_OFF:
                        self._hazard_active[r][c] = False

        # ── 4. Compute brightness targets from logical state ──────────────────
        for r in range(rows):
            for c in range(cols):
                if self._glare_active[r][c]:
                    # Target brightness = shadow cone value (0.0 = dark centre, 0.3 = edge)
                    # glare_proj[r][c] encodes shadow brightness (0=suppressed, 1=untouched)
                    # But here we use the EMA-smoothed glare influence to set darkness depth
                    g = self._glare_ema[r][c]
                    # Map EMA influence → target brightness:
                    #   high influence → near-zero brightness (dark centre)
                    #   low influence  → penumbra brightness
                    t_brightness = config.BRIGHTNESS_PENUMBRA * (1.0 - g)
                    self._brightness_target[r][c] = max(
                        config.BRIGHTNESS_SUPPRESS, t_brightness)
                    self.led_states[r][c] = "suppress"

                elif self._hazard_active[r][c]:
                    self._brightness_target[r][c] = config.BRIGHTNESS_FULL
                    self.led_states[r][c] = "boost"

                else:
                    # Normal beam: vary by row (far = brighter, near = dimmer)
                    t = r / max(1, rows - 1)
                    b = config.BRIGHTNESS_FAR_ROW * (1-t) + config.BRIGHTNESS_NEAR_ROW * t
                    self._brightness_target[r][c] = b
                    self.led_states[r][c] = "on"

        # ── 5. EMA-smooth actual brightness toward target ─────────────────────
        for r in range(rows):
            for c in range(cols):
                current = self.led_brightness[r][c]
                target  = self._brightness_target[r][c]
                if target < current:
                    # Dimming (suppression activating): use DOWN alpha (faster)
                    alpha_b = config.LED_BRIGHTNESS_ALPHA_DOWN
                else:
                    # Brightening (suppression releasing): use UP alpha (slower)
                    # This prevents a 1-frame miss from flashing bright
                    alpha_b = config.LED_BRIGHTNESS_ALPHA_UP
                self.led_brightness[r][c] = (
                    alpha_b * target + (1 - alpha_b) * current
                )

        # ── 6. Update column summary ──────────────────────────────────────────
        self._update_zone_states()

    # ══════════════════════════════════════════════════════════════════════════
    # SHADOW CONE PROJECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _project_glare(self, track: dict, conf: float, glare_proj: list):
        """
        Paint a Gaussian shadow cone onto glare_proj using the SMOOTHED track position.

        glare_proj[r][c] = influence value 0.0–1.0 (1.0 = fully influenced/suppressed)
        Multiple overlapping cars → max() wins (darkest shadow).
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        col_f = track["cx_norm"] * (cols - 1)
        row_f = track["cy_norm"] * (rows - 1)

        conf_above_min = max(0.0, (conf - config.MIN_CONFIDENCE_GLARE)
                             / (1.0 - config.MIN_CONFIDENCE_GLARE + 1e-6))
        distance_factor = 1.0 - track["cy_norm"]

        col_radius = (config.SHADOW_COL_RADIUS_BASE
                      + conf_above_min * config.SHADOW_COL_CONF_SCALE) * cols
        row_radius = (config.SHADOW_ROW_RADIUS_BASE
                      + distance_factor * config.SHADOW_ROW_DIST_SCALE) * rows

        c_lo = max(0, int(col_f - col_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        c_hi = min(cols - 1, int(col_f + col_radius * config.SHADOW_MAX_DIST_NORM) + 1)
        r_lo = max(0, int(row_f - row_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        r_hi = min(rows - 1, int(row_f + row_radius * config.SHADOW_MAX_DIST_NORM) + 1)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                dc = (c - col_f) / (col_radius + 1e-6)
                dr = (r - row_f) / (row_radius + 1e-6)
                dist_norm = math.sqrt(dc * dc + dr * dr)
                if dist_norm > config.SHADOW_MAX_DIST_NORM:
                    continue

                # Gaussian influence: 1.0 at centre → 0.0 at edge
                gauss = math.exp(-config.SHADOW_PENUMBRA_FALLOFF * dist_norm ** 2)
                influence = gauss  # 1 = fully suppressed, 0 = untouched

                if influence > glare_proj[r][c]:
                    glare_proj[r][c] = influence

    def _project_hazard(self, track: dict, boost_proj: list):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        col_f = track["cx_norm"] * (cols - 1)
        row_f = track["cy_norm"] * (rows - 1)

        col_radius = config.BOOST_COL_RADIUS * cols
        row_radius = config.BOOST_ROW_RADIUS * rows

        c_lo = max(0, int(col_f - col_radius) - 1)
        c_hi = min(cols - 1, int(col_f + col_radius) + 1)
        r_lo = max(0, int(row_f - row_radius) - 1)
        r_hi = min(rows - 1, int(row_f + row_radius) + 1)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                dc = (c - col_f) / (col_radius + 1e-6)
                dr = (r - row_f) / (row_radius + 1e-6)
                if (dc * dc + dr * dr) <= 1.0:
                    boost_proj[r][c] = True

    # ══════════════════════════════════════════════════════════════════════════
    # COLUMN SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    def _update_zone_states(self):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS
        for c in range(cols):
            col_states = [self.led_states[r][c] for r in range(rows)]
            if "suppress" in col_states:
                self.zone_states[c] = "suppress"
                avg = sum(self.led_brightness[r][c] for r in range(rows)) / rows
                self.zone_conf[c] = avg
            elif "boost" in col_states:
                self.zone_states[c] = "boost"
                self.zone_conf[c] = config.BRIGHTNESS_FULL
            else:
                self.zone_states[c] = "on"
                self.zone_conf[c] = config.BRIGHTNESS_FULL

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC QUERY
    # ══════════════════════════════════════════════════════════════════════════

    def led_color(self, col: int, row: int) -> tuple:
        state      = self.led_states[row][col]
        brightness = self.led_brightness[row][col]

        if state == "suppress":
            if brightness <= 0.05:
                return config.C_LED_SUPPRESS, 0.0
            elif brightness < config.BRIGHTNESS_PENUMBRA:
                return config.C_LED_SUPPRESS, brightness
            else:
                return config.C_LED_PENUMBRA, brightness

        elif state == "boost":
            if self._flash_on:
                return config.C_LED_BOOST, 1.0
            else:
                return config.C_LED_ON, 0.15

        else:
            return config.C_LED_ON, brightness

    def zone_byte(self) -> int:
        result = 0
        for c in range(config.ZONE_COUNT):
            if self.zone_states[c] != "suppress":
                result |= (1 << c)
        return result

    def tick_flash(self):
        self._flash_tick += 1
        self._flash_on    = (self._flash_tick % 2 == 0)

    @staticmethod
    def _cx_to_col(cx_norm: float) -> int:
        col = int(cx_norm * config.ZONE_COUNT)
        return max(0, min(config.ZONE_COUNT - 1, col))

    @staticmethod
    def _cy_to_row(cy_norm: float) -> int:
        row = int(cy_norm * config.ROW_LEDS)
        return max(0, min(config.ROW_LEDS - 1, row))

    def summary(self) -> str:
        syms = {"on": "▪", "suppress": "✕", "boost": "★"}
        parts = [syms.get(s, "?") for s in self.zone_states]
        return f"[{''.join(parts)}]  0x{self.zone_byte():0{(config.ZONE_COUNT+3)//4}X}"
