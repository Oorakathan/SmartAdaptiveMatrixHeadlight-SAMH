"""
led_controller.py
-----------------
Converts a list of detections into per-LED states and brightness values for an 8×8 matrix.

Responsibilities:
  - Decide which individual LEDs (8 cols × 8 rows) are "on", "suppress", or "boost"
  - Map vehicle coordinates to LED positions and calculate region of influence
  - Manage the hazard flash timer (tick-based, called externally)
  - Compute the zone byte (8-bit value for Arduino serial output - backward compatible)
  - Provide led_color(col, row) for the UI matrix renderer
  - Temporal suppression filter per LED (avoids single-frame false positives)

Does NOT:
  - Touch any queue
  - Know anything about tkinter or OpenCV
  - Read from files or network

This is a pure logic class — easy to unit-test in isolation.
Swap in a different beam logic by subclassing or replacing ZoneController.
"""

import config


class ZoneController:
    """
    Core LED matrix logic with individual LED control.

    State per LED (col 0–7, row 0–7):
        "on"       → full beam, nothing detected
        "suppress" → shadow LED (glare vehicle in this region)
        "boost"    → hazard pre-alert (animal / pedestrian detected)

    Row semantics (0 = far, 7 = wide flood):
        Row 0 → 100 m+  (narrow high beam)
        Row 7 → wide flood (always partially on even in shadow zones)

    Column semantics (0 = left, 7 = right):
        Corresponds to angular zones (horizontal FOV divided into 8)

    Flash:
        Hazard LEDs flash at FLASH_HZ. Call tick_flash() from the UI
        timer (every FLASH_INTERVAL_MS milliseconds) to advance the phase.
    """

    def __init__(self):
        # 8x8 matrix: led_states[row][col]
        self.led_states = [["on"] * config.ZONE_COUNT for _ in range(8)]
        self.led_brightness = [[config.BRIGHTNESS_FULL] * config.ZONE_COUNT for _ in range(8)]

        # Flash state
        self._flash_tick = 0
        self._flash_on   = True

        # Temporal filter: track consecutive detection counts per LED
        # LED is only suppressed after SUPPRESS_TEMPORAL consecutive frames
        self._suppress_streak = [[0] * config.ZONE_COUNT for _ in range(8)]

        # Backward compatibility: zone states (column-based view for annotation)
        self.zone_states = ["on"] * config.ZONE_COUNT
        self.zone_conf   = [config.BRIGHTNESS_FULL] * config.ZONE_COUNT

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: list):
        """
        Recompute LED states from the latest detection list.
        Call once per frame after the AI detector produces results.

        detections: list of dicts with keys:
            label    : str
            conf     : float
            cx_norm  : float  (0.0 = left edge, 1.0 = right edge)
            cy_norm  : float  (0.0 = top edge, 1.0 = bottom edge)
            category : str    "glare" | "hazard" | "other"
        """
        # Start fresh each frame
        new_states = [["on"] * config.ZONE_COUNT for _ in range(8)]
        new_brightness = [[config.BRIGHTNESS_FULL] * config.ZONE_COUNT for _ in range(8)]

        # Track which LEDs have glare this frame (for temporal filter)
        glare_this_frame = [[False] * config.ZONE_COUNT for _ in range(8)]

        for det in detections:
            col = self._cx_to_col(det["cx_norm"])
            row = self._cy_to_row(det["cy_norm"])

            if det["category"] == "glare":
                # Mark the region around the detected vehicle
                self._mark_glare_region(
                    new_states, new_brightness, glare_this_frame,
                    col, row
                )

            elif det["category"] == "hazard":
                # Mark the region for hazard (flashing boost)
                self._mark_hazard_region(new_states, new_brightness, col, row)

        # Update temporal streaks and apply temporal filtering
        for row in range(8):
            for col in range(config.ZONE_COUNT):
                if glare_this_frame[row][col]:
                    self._suppress_streak[row][col] += 1
                else:
                    self._suppress_streak[row][col] = 0

                # Only suppress if streak is long enough
                if new_states[row][col] == "suppress":
                    if self._suppress_streak[row][col] < config.SUPPRESS_TEMPORAL:
                        # Not enough consecutive frames, revert to on
                        new_states[row][col] = "on"
                        new_brightness[row][col] = config.BRIGHTNESS_FULL

        self.led_states = new_states
        self.led_brightness = new_brightness

        # Update backward-compatible zone states (based on column worst-case)
        self._update_zone_states()

    def _mark_glare_region(self, states, brightness, glare_flags, center_col, center_row):
        """Mark LEDs in the glare region around a detected vehicle."""
        for row in range(8):
            for col in range(config.ZONE_COUNT):
                # Calculate distance from center
                col_dist = abs(col - center_col)
                row_dist = abs(row - center_row)

                # Check if this LED is in the influence region
                if col_dist == 0 and row_dist == 0:
                    # Center LED - full suppression
                    states[row][col] = "suppress"
                    brightness[row][col] = config.BRIGHTNESS_SUPPRESS
                    glare_flags[row][col] = True

                elif col_dist <= config.LED_INFLUENCE_COL_RADIUS and row_dist <= config.LED_INFLUENCE_ROW_RADIUS:
                    # Adjacent LEDs - partial suppression
                    if states[row][col] != "suppress":
                        states[row][col] = "suppress"
                        brightness[row][col] = config.LED_ADJACENT_BRIGHTNESS
                        glare_flags[row][col] = True

    def _mark_hazard_region(self, states, brightness, center_col, center_row):
        """Mark LEDs for hazard (boost/flash) around a detected hazard."""
        for row in range(8):
            for col in range(config.ZONE_COUNT):
                col_dist = abs(col - center_col)
                row_dist = abs(row - center_row)

                # Hazard affects a region similar to glare
                if col_dist <= config.LED_INFLUENCE_COL_RADIUS and row_dist <= config.LED_INFLUENCE_ROW_RADIUS:
                    # Only override "on" state, not suppress
                    if states[row][col] == "on":
                        states[row][col] = "boost"
                        brightness[row][col] = config.BRIGHTNESS_FULL

    def _update_zone_states(self):
        """Update backward-compatible zone states for annotation overlay."""
        new_zone_states = []
        new_zone_conf = []

        for col in range(config.ZONE_COUNT):
            # Check column state - if any LED in column is suppressed, mark column
            has_suppress = any(self.led_states[row][col] == "suppress" for row in range(8))
            has_boost = any(self.led_states[row][col] == "boost" for row in range(8))

            if has_suppress:
                new_zone_states.append("suppress")
                # Average brightness across column
                avg_brightness = sum(self.led_brightness[row][col] for row in range(8)) / 8
                new_zone_conf.append(avg_brightness)
            elif has_boost:
                new_zone_states.append("boost")
                new_zone_conf.append(config.BRIGHTNESS_FULL)
            else:
                new_zone_states.append("on")
                new_zone_conf.append(config.BRIGHTNESS_FULL)

        self.zone_states = new_zone_states
        self.zone_conf = new_zone_conf

    def tick_flash(self):
        """
        Advance the hazard flash phase.
        Should be called every FLASH_INTERVAL_MS milliseconds.
        """
        self._flash_tick += 1
        self._flash_on    = (self._flash_tick % 2 == 0)

    def led_color(self, col: int, row: int) -> tuple:
        """
        Return (hex_color, brightness) for a single LED at (col, row).

        col  : 0–7  (angular zone, left to right)
        row  : 0–7  (distance band, 0 = far, 7 = wide flood)

        Returns:
            color      : str  hex colour e.g. "#00C2A8"
            brightness : float  0.0–1.0  (used to blend with off-colour)
        """
        state = self.led_states[row][col]
        base_brightness = self.led_brightness[row][col]

        if state == "suppress":
            # Row 7 (wide flood) stays partially on — driver still needs near-field
            if row == 7:
                return config.C_LED_SUPPRESS, max(base_brightness, config.LED_MIN_ROW7_BRIGHTNESS)
            return config.C_LED_SUPPRESS, base_brightness

        elif state == "boost":
            if self._flash_on:
                return config.C_LED_BOOST, 1.0
            else:
                # Off phase of flash — dim teal so it's not completely black
                return config.C_LED_ON, 0.20

        else:
            # Normal full beam — far rows slightly brighter than near rows
            # row 0 → BRIGHTNESS_FAR_ROW, row 7 → BRIGHTNESS_NEAR_ROW
            t = row / 7
            brightness = (
                config.BRIGHTNESS_FAR_ROW * (1 - t)
                + config.BRIGHTNESS_NEAR_ROW * t
            )
            return config.C_LED_ON, brightness * base_brightness

    def zone_byte(self) -> int:
        """
        Return an 8-bit integer representing active zones (backward compatible).
        Bit N = 1 means zone N is ON (not suppressed).
        This is the byte value that would be sent to Arduino over serial.

        Example:
            0xFF = 11111111 → all zones on (full beam)
            0xFC = 11111100 → zones 0,1 suppressed (car on the left)
        """
        result = 0
        for col in range(config.ZONE_COUNT):
            if self.zone_states[col] != "suppress":
                result |= (1 << col)
        return result

    def reset(self):
        """Reset all LEDs to full-beam on state."""
        self.led_states = [["on"] * config.ZONE_COUNT for _ in range(8)]
        self.led_brightness = [[config.BRIGHTNESS_FULL] * config.ZONE_COUNT for _ in range(8)]
        self._suppress_streak = [[0] * config.ZONE_COUNT for _ in range(8)]
        self.zone_states = ["on"] * config.ZONE_COUNT
        self.zone_conf = [config.BRIGHTNESS_FULL] * config.ZONE_COUNT
        self._flash_tick = 0
        self._flash_on = True

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cx_to_col(cx_norm: float) -> int:
        """Map a normalised x-coordinate (0.0–1.0) to a zone column (0–7)."""
        col = int(cx_norm * config.ZONE_COUNT)
        return max(0, min(config.ZONE_COUNT - 1, col))

    @staticmethod
    def _cy_to_row(cy_norm: float) -> int:
        """Map a normalised y-coordinate (0.0–1.0) to a row (0–7).
        0.0 = top of frame (far/row 0), 1.0 = bottom of frame (near/row 7)"""
        row = int(cy_norm * 8)
        return max(0, min(7, row))

    def summary(self) -> str:
        """One-line human-readable zone state summary for logging/debug."""
        symbols = {"on": "▪", "suppress": "✕", "boost": "★"}
        parts = [symbols.get(s, "?") for s in self.zone_states]
        byte_val = self.zone_byte()
        return f"[{''.join(parts)}]  byte=0x{byte_val:02X} ({byte_val:08b})"
