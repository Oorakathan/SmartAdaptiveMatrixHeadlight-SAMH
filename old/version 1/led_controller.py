"""
led_controller.py
-----------------
Per-LED beam control — the "brain" of the headlamp.

Key design
----------
* Matrix dimensions (ZONE_COUNT × ROW_LEDS) are read from config at
  runtime so they respond to the user changing the matrix preset.

* Every spatial calculation is NORMALISED to the matrix size before any
  arithmetic, so the physics are identical whether you use 4×4 or 16×16.

* Temporal hysteresis filter:
    - Glare  : must appear SUPPRESS_TEMPORAL_ON  frames to activate;
               held for SUPPRESS_TEMPORAL_OFF frames after disappearing.
    - Hazard : 1-frame activation; held for BOOST_TEMPORAL_OFF frames.
  This prevents single-frame noise from flickering the headlamp.

* Shadow cone model (Audi-style):
    - Centre LED at (col, row) of the detected object → fully off.
    - Penumbra ring around it: brightness rises with normalised distance
      using a Gaussian falloff so the edge is smooth, not abrupt.
    - Cone expands for distant objects (far row → wider penumbra) —
      matches real physics: a far car's headlight illuminates a wider
      angular region.
    - Confidence scales the cone size: low-conf detection → narrow shadow
      (conservative); high-conf → full penumbra.

* Hazard boost:
    - Small ring of LEDs around the detected object flashes amber.
    - Does NOT override an existing suppress — glare takes priority.

Public interface
----------------
  update(detections)       → call once per frame
  tick_flash()             → call every FLASH_INTERVAL_MS from UI timer
  led_color(col, row)      → (hex_color, brightness 0–1) for rendering
  zone_byte()              → 8/16-bit int for Arduino serial
  reset()                  → full beam, clear all state
  summary()                → one-line debug string
"""

import math
import config


class ZoneController:

    def __init__(self):
        self._init_state()

    def _init_state(self):
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        # Per-LED logical state and brightness
        self.led_states     = [["on"] * cols for _ in range(rows)]
        self.led_brightness = [[config.BRIGHTNESS_FULL] * cols for _ in range(rows)]

        # Temporal counters: positive = "seen for N frames", negative = "absent for N frames"
        # Shape: [rows][cols]  — separate for glare and hazard
        self._glare_streak  = [[0] * cols for _ in range(rows)]
        self._hazard_streak = [[0] * cols for _ in range(rows)]

        # Pending influence accumulators (filled each frame, applied at end)
        # Each LED accumulates the darkest shadow it receives from any threat
        self._shadow_acc    = [[1.0] * cols for _ in range(rows)]  # 1.0 = no shadow
        self._boost_acc     = [[False] * cols for _ in range(rows)]

        # Flash
        self._flash_tick = 0
        self._flash_on   = True

        # Column-level summary (backward compat for zone bar / zone byte)
        self.zone_states = ["on"] * cols
        self.zone_conf   = [config.BRIGHTNESS_FULL] * cols

    def reset(self):
        self._init_state()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC — called once per frame from UI poll loop
    # ══════════════════════════════════════════════════════════════════════════

    def update(self, detections: list):
        """
        Recompute LED states from the detection list for this frame.
        Applies temporal hysteresis so single-frame noise doesn't flicker.
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        # ── 1. Reset per-frame accumulators ──────────────────────────────────
        shadow_acc = [[1.0]   * cols for _ in range(rows)]  # 1.0 = full bright
        boost_acc  = [[False] * cols for _ in range(rows)]

        # ── 2. Project each detection onto the LED grid ───────────────────────
        glare_hits  = [[False] * cols for _ in range(rows)]
        hazard_hits = [[False] * cols for _ in range(rows)]

        for det in detections:
            if det["category"] == "glare" and det["conf"] >= config.MIN_CONFIDENCE_GLARE:
                self._project_glare(det, shadow_acc, glare_hits)
            elif det["category"] == "hazard" and det["conf"] >= config.MIN_CONFIDENCE_HAZARD:
                self._project_hazard(det, boost_acc, hazard_hits)

        # ── 3. Temporal filtering ─────────────────────────────────────────────
        for r in range(rows):
            for c in range(cols):
                # --- GLARE ---
                if glare_hits[r][c]:
                    # Object present: increment streak (cap at ON threshold)
                    self._glare_streak[r][c] = min(
                        config.SUPPRESS_TEMPORAL_ON,
                        self._glare_streak[r][c] + 1
                    )
                else:
                    # Object absent: decrement streak toward 0
                    self._glare_streak[r][c] = max(
                        0,
                        self._glare_streak[r][c] - 1
                    )

                # --- HAZARD ---
                if hazard_hits[r][c]:
                    self._hazard_streak[r][c] = min(
                        config.BOOST_TEMPORAL_ON,
                        self._hazard_streak[r][c] + 1
                    )
                else:
                    self._hazard_streak[r][c] = max(
                        0,
                        self._hazard_streak[r][c] - 1
                    )

        # ── 4. Commit LED states from temporal-filtered streaks ────────────────
        for r in range(rows):
            for c in range(cols):
                glare_active  = self._glare_streak[r][c]  >= config.SUPPRESS_TEMPORAL_ON
                hazard_active = self._hazard_streak[r][c] >= config.BOOST_TEMPORAL_ON

                if glare_active:
                    # Use this frame's accumulated shadow brightness
                    # If no shadow from this frame but streak still active → hold last
                    brightness = shadow_acc[r][c] if glare_hits[r][c] else config.BRIGHTNESS_PENUMBRA
                    self.led_states[r][c]     = "suppress"
                    self.led_brightness[r][c] = brightness

                elif hazard_active and not glare_active:
                    self.led_states[r][c]     = "boost"
                    self.led_brightness[r][c] = config.BRIGHTNESS_FULL

                else:
                    self.led_states[r][c]     = "on"
                    self.led_brightness[r][c] = config.BRIGHTNESS_FULL

        # ── 5. Update column summary ──────────────────────────────────────────
        self._update_zone_states()

    # ══════════════════════════════════════════════════════════════════════════
    # SHADOW CONE PROJECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _project_glare(self, det: dict, shadow_acc: list, hit_map: list):
        """
        Paint a Gaussian shadow cone centred on det's LED position.

        All radii are in NORMALISED units (fraction of matrix dimension)
        so geometry is independent of matrix size.
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        # Map normalised image coords → LED grid coords
        col_f = det["cx_norm"] * (cols - 1)   # float column  0 … cols-1
        row_f = det["cy_norm"] * (rows - 1)   # float row     0 … rows-1

        conf = max(0.0, min(1.0, det["conf"]))
        conf_above_min = max(0.0, (conf - config.MIN_CONFIDENCE_GLARE)
                                 / (1.0 - config.MIN_CONFIDENCE_GLARE + 1e-6))

        # Distance factor: cy_norm=0 → top/far object, cy_norm=1 → near
        # Far objects get a WIDER penumbra (light cone physics)
        distance_factor = 1.0 - det["cy_norm"]   # 0=near, 1=far

        # Radii in normalised grid units
        col_radius = (config.SHADOW_COL_RADIUS_BASE
                      + conf_above_min * config.SHADOW_COL_CONF_SCALE) * cols
        row_radius = (config.SHADOW_ROW_RADIUS_BASE
                      + distance_factor * config.SHADOW_ROW_DIST_SCALE) * rows

        # Iterate only over LEDs within bounding box of the cone
        c_lo = max(0, int(col_f - col_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        c_hi = min(cols - 1, int(col_f + col_radius * config.SHADOW_MAX_DIST_NORM) + 1)
        r_lo = max(0, int(row_f - row_radius * config.SHADOW_MAX_DIST_NORM) - 1)
        r_hi = min(rows - 1, int(row_f + row_radius * config.SHADOW_MAX_DIST_NORM) + 1)

        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                # Normalised distance from cone centre
                dc = (c - col_f) / (col_radius + 1e-6)
                dr = (r - row_f) / (row_radius + 1e-6)
                dist_norm = math.sqrt(dc * dc + dr * dr)

                if dist_norm > config.SHADOW_MAX_DIST_NORM:
                    continue

                # Gaussian brightness: 0 at centre → PENUMBRA at edge
                # brightness = SUPPRESS + (PENUMBRA - SUPPRESS) * (1 - gauss)
                gauss = math.exp(-config.SHADOW_PENUMBRA_FALLOFF * dist_norm ** 2)
                brightness = (config.BRIGHTNESS_PENUMBRA
                              + (config.BRIGHTNESS_SUPPRESS - config.BRIGHTNESS_PENUMBRA)
                              * gauss)
                brightness = max(config.BRIGHTNESS_SUPPRESS,
                                 min(config.BRIGHTNESS_PENUMBRA, brightness))

                # Accumulate darkest shadow (multiple threats → darkest wins)
                if brightness < shadow_acc[r][c]:
                    shadow_acc[r][c] = brightness

                hit_map[r][c] = True

    def _project_hazard(self, det: dict, boost_acc: list, hit_map: list):
        """
        Mark a small region of LEDs around the hazard for boost/flash.
        Does not touch LEDs that are already in a glare shadow.
        """
        cols = config.ZONE_COUNT
        rows = config.ROW_LEDS

        col_f = det["cx_norm"] * (cols - 1)
        row_f = det["cy_norm"] * (rows - 1)

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
                    boost_acc[r][c] = True
                    hit_map[r][c]   = True

    # ══════════════════════════════════════════════════════════════════════════
    # COLUMN SUMMARY (for zone bar and Arduino byte)
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
        """
        Return (hex_color, brightness 0–1) for LED at (col, row).

        Normal beam brightness varies by row:
          row 0 (far, 100 m+) → BRIGHTNESS_FAR_ROW
          last row (near, wide flood) → BRIGHTNESS_NEAR_ROW
        This simulates the real luminous intensity distribution of a headlamp.
        """
        state      = self.led_states[row][col]
        brightness = self.led_brightness[row][col]
        rows       = config.ROW_LEDS

        if state == "suppress":
            # Penumbra colour blends teal→red as brightness decreases
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
                return config.C_LED_ON, 0.15   # dim teal during off-phase

        else:
            # Normal beam: far rows brighter (narrow high beam) than near rows
            t = row / max(1, rows - 1)   # 0 = far, 1 = near
            b = config.BRIGHTNESS_FAR_ROW * (1 - t) + config.BRIGHTNESS_NEAR_ROW * t
            return config.C_LED_ON, b * brightness

    def zone_byte(self) -> int:
        """
        Returns a bitmask of active (non-suppressed) zones.
        Bit N = 1 → zone N is on.
        For >8 columns, returns an int (not limited to 8-bit).
        """
        result = 0
        for c in range(config.ZONE_COUNT):
            if self.zone_states[c] != "suppress":
                result |= (1 << c)
        return result

    def tick_flash(self):
        """Advance hazard flash phase. Call every FLASH_INTERVAL_MS."""
        self._flash_tick += 1
        self._flash_on    = (self._flash_tick % 2 == 0)

    # ── Helpers ───────────────────────────────────────────────────────────────

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
