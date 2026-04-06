"""
led_controller.py
-----------------
Per-LED beam control with drive mode awareness and city beam shaping.

New in this version
-------------------
Drive mode beam shaper
  set_drive_mode(mode) is called by the UI whenever the scene density
  estimator decides to change mode.

  HIGHWAY:     Full matrix, all rows active. Full adaptive pixel control.
  EXPRESSWAY:  Top EXPRESSWAY_DARK_ROW_FRACTION of rows forced off.
               Remaining rows: full pixel control. (Handles flyovers,
               bright motorway overhead lighting.)
  CITY:        Top CITY_DARK_ROW_FRACTION of rows forced off.
               This physically limits the beam to the low-beam cutoff
               (ECE R112 class C behaviour). On an 8-row matrix with
               CITY_DARK_ROW_FRACTION=0.50, rows 0–3 are dark, rows 4–7
               have full pixel-wise glare suppression and hazard boost.
               Pedestrians in the active rows are still highlighted.

Why force rows rather than just not drawing detections in them?
  Because the beam rows represent physical LED groups. In CITY mode you
  want the physical LEDs in the top rows to be literally off — not just
  "no shadow computed for them". This is the correct hardware behaviour.
  The beam shaper writes directly to led_brightness after all other logic,
  so it overrides suppression, boost, everything. Safety takes precedence.

The `led_states` grid still records the logical state for the UI overlay.
Forced-off rows show state "off" (not "on", not "suppress") so the matrix
display shows them visually distinct from active-but-full-beam rows.
"""

import math
import config


class ZoneController:

    def __init__(self):
        self._drive_mode = config.DRIVE_MODE_HIGHWAY
        self._init_state()

    def _init_state(self):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        self.led_states     = [["on"] * cols for _ in range(rows)]
        self.led_brightness = [[config.BRIGHTNESS_FULL] * cols for _ in range(rows)]

        self._glare_ema  = [[0.0] * cols for _ in range(rows)]
        self._hazard_ema = [[0.0] * cols for _ in range(rows)]

        self._glare_active  = [[False] * cols for _ in range(rows)]
        self._hazard_active = [[False] * cols for _ in range(rows)]

        self._brightness_target = [[config.BRIGHTNESS_FULL] * cols for _ in range(rows)]

        self._flash_tick = 0
        self._flash_on   = True

        self.zone_states = ["on"] * cols
        self.zone_conf   = [config.BRIGHTNESS_FULL] * cols

    def reset(self):
        self._init_state()

    def set_drive_mode(self, mode: str):
        """Called by UI when scene density estimator changes mode."""
        self._drive_mode = mode

    @property
    def drive_mode(self) -> str:
        return self._drive_mode

    def _forced_off_rows(self) -> int:
        """How many top rows to force off in the current drive mode."""
        rows = config.ROW_LEDS
        if self._drive_mode == config.DRIVE_MODE_CITY:
            return max(0, int(config.CITY_DARK_ROW_FRACTION * rows))
        elif self._drive_mode == config.DRIVE_MODE_EXPRESSWAY:
            return max(0, int(config.EXPRESSWAY_DARK_ROW_FRACTION * rows))
        return 0   # HIGHWAY: no forced-off rows

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC — called once per frame
    # ══════════════════════════════════════════════════════════════════════════

    def update(self, tracks: list):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS
        dark_rows = self._forced_off_rows()

        # ── 1. Project track influences onto LED grid ─────────────────────────
        glare_proj = [[0.0]   * cols for _ in range(rows)]
        boost_proj = [[False] * cols for _ in range(rows)]

        for track in tracks:
            cat  = track["category"]
            conf = track["conf"]
            if cat == "glare":
                self._project_glare(track, conf, glare_proj)
            elif cat == "hazard":
                self._project_hazard(track, boost_proj)

        # ── 2. EMA update ─────────────────────────────────────────────────────
        alpha = config.CONF_EMA_ALPHA
        for r in range(rows):
            for c in range(cols):
                self._glare_ema[r][c]  = alpha * glare_proj[r][c]  + (1-alpha) * self._glare_ema[r][c]
                target_h = 1.0 if boost_proj[r][c] else 0.0
                self._hazard_ema[r][c] = alpha * target_h           + (1-alpha) * self._hazard_ema[r][c]

        # ── 3. Hysteresis state transitions ───────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                g_inf = self._glare_ema[r][c]
                h_inf = self._hazard_ema[r][c]

                if not self._glare_active[r][c]:
                    if g_inf >= config.SUPPRESS_CONF_ON:
                        self._glare_active[r][c] = True
                else:
                    if g_inf < config.SUPPRESS_CONF_OFF:
                        self._glare_active[r][c] = False

                if not self._hazard_active[r][c]:
                    if h_inf >= config.BOOST_CONF_ON:
                        self._hazard_active[r][c] = True
                else:
                    if h_inf < config.BOOST_CONF_OFF:
                        self._hazard_active[r][c] = False

        # ── 4. Brightness targets ─────────────────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                if self._glare_active[r][c]:
                    g = self._glare_ema[r][c]
                    t_brightness = config.BRIGHTNESS_PENUMBRA * (1.0 - g)
                    self._brightness_target[r][c] = max(config.BRIGHTNESS_SUPPRESS, t_brightness)
                    self.led_states[r][c] = "suppress"
                elif self._hazard_active[r][c]:
                    self._brightness_target[r][c] = config.BRIGHTNESS_FULL
                    self.led_states[r][c] = "boost"
                else:
                    t = r / max(1, rows - 1)
                    b = config.BRIGHTNESS_FAR_ROW * (1-t) + config.BRIGHTNESS_NEAR_ROW * t
                    self._brightness_target[r][c] = b
                    self.led_states[r][c] = "on"

        # ── 5. EMA-smooth actual brightness ───────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                current = self.led_brightness[r][c]
                target  = self._brightness_target[r][c]
                alpha_b = config.LED_BRIGHTNESS_ALPHA_DOWN if target < current else config.LED_BRIGHTNESS_ALPHA_UP
                self.led_brightness[r][c] = alpha_b * target + (1 - alpha_b) * current

        # ── 6. CITY / EXPRESSWAY beam shaper ──────────────────────────────────
        # Override top N rows to fully off, regardless of detection state.
        # This is a hard physical override — these LEDs are simply switched off.
        # Applied AFTER all other logic so it takes absolute priority.
        for r in range(dark_rows):
            for c in range(cols):
                self.led_brightness[r][c] = 0.0
                self.led_states[r][c]     = "off"   # distinct from "suppress" in UI

        # ── 7. Column summary ─────────────────────────────────────────────────
        self._update_zone_states()

    # ══════════════════════════════════════════════════════════════════════════
    # SHADOW / BOOST PROJECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _project_glare(self, track: dict, conf: float, glare_proj: list):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        col_f = track["cx_norm"] * (cols - 1)
        row_f = track["cy_norm"] * (rows - 1)

        conf_above_min  = max(0.0, (conf - config.MIN_CONFIDENCE_GLARE)
                              / (1.0 - config.MIN_CONFIDENCE_GLARE + 1e-6))
        distance_factor = 1.0 - track["cy_norm"]

        col_radius = (config.SHADOW_COL_RADIUS_BASE + conf_above_min * config.SHADOW_COL_CONF_SCALE) * cols
        row_radius = (config.SHADOW_ROW_RADIUS_BASE + distance_factor * config.SHADOW_ROW_DIST_SCALE) * rows

        c_lo = max(0, int(col_f - col_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        c_hi = min(cols-1, int(col_f + col_radius * config.SHADOW_MAX_DIST_NORM) + 1)
        r_lo = max(0, int(row_f - row_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        r_hi = min(rows-1, int(row_f + row_radius * config.SHADOW_MAX_DIST_NORM) + 1)

        for r in range(r_lo, r_hi+1):
            for c in range(c_lo, c_hi+1):
                dc = (c - col_f) / (col_radius + 1e-6)
                dr = (r - row_f) / (row_radius + 1e-6)
                dist_norm = math.sqrt(dc*dc + dr*dr)
                if dist_norm > config.SHADOW_MAX_DIST_NORM:
                    continue
                gauss = math.exp(-config.SHADOW_PENUMBRA_FALLOFF * dist_norm**2)
                if gauss > glare_proj[r][c]:
                    glare_proj[r][c] = gauss

    def _project_hazard(self, track: dict, boost_proj: list):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS
        col_f = track["cx_norm"] * (cols - 1)
        row_f = track["cy_norm"] * (rows - 1)
        col_radius = config.BOOST_COL_RADIUS * cols
        row_radius = config.BOOST_ROW_RADIUS * rows
        c_lo = max(0, int(col_f - col_radius) - 1)
        c_hi = min(cols-1, int(col_f + col_radius) + 1)
        r_lo = max(0, int(row_f - row_radius) - 1)
        r_hi = min(rows-1, int(row_f + row_radius) + 1)
        for r in range(r_lo, r_hi+1):
            for c in range(c_lo, c_hi+1):
                dc = (c - col_f) / (col_radius + 1e-6)
                dr = (r - row_f) / (row_radius + 1e-6)
                if (dc*dc + dr*dr) <= 1.0:
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
                self.zone_conf[c] = sum(self.led_brightness[r][c] for r in range(rows)) / rows
            elif "boost" in col_states:
                self.zone_states[c] = "boost"
                self.zone_conf[c] = config.BRIGHTNESS_FULL
            elif all(s == "off" for s in col_states):
                self.zone_states[c] = "off"
                self.zone_conf[c] = 0.0
            else:
                self.zone_states[c] = "on"
                self.zone_conf[c] = config.BRIGHTNESS_FULL

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC QUERY
    # ══════════════════════════════════════════════════════════════════════════

    def led_color(self, col: int, row: int) -> tuple:
        state      = self.led_states[row][col]
        brightness = self.led_brightness[row][col]

        if state == "off":
            # Forced off by city/expressway beam shaper
            return config.C_LED_OFF, 0.0

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
            if self.zone_states[c] not in ("suppress", "off"):
                result |= (1 << c)
        return result

    def tick_flash(self):
        self._flash_tick += 1
        self._flash_on    = (self._flash_tick % 2 == 0)

    @staticmethod
    def _cx_to_col(cx_norm: float) -> int:
        return max(0, min(config.ZONE_COUNT-1, int(cx_norm * config.ZONE_COUNT)))

    @staticmethod
    def _cy_to_row(cy_norm: float) -> int:
        return max(0, min(config.ROW_LEDS-1, int(cy_norm * config.ROW_LEDS)))

    def summary(self) -> str:
        syms = {"on": "▪", "suppress": "✕", "boost": "★", "off": "·"}
        parts = [syms.get(s, "?") for s in self.zone_states]
        return f"[{''.join(parts)}] 0x{self.zone_byte():0{(config.ZONE_COUNT+3)//4}X} [{self._drive_mode.upper()}]"