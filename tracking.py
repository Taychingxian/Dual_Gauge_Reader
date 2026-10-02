"""Tracking state machine for gauge needle stabilization."""

from collections import deque
import time

from measurement import circular_distance, signed_angle_delta, circular_mean, weighted_circular_mean


class TrackingState:
    """Encapsulates all mutable state for the gauge tracking pipeline.

    Replaces the 20+ module-level variables and 4 duplicated reset blocks
    in the original monolithic script.
    """

    def __init__(self):
        self.history_1 = deque(maxlen=10)
        self.history_2 = deque(maxlen=10)
        self.angle_history = deque(maxlen=12)
        self.center_history = deque(maxlen=7)
        self.stable_angle = None
        self.locked_center = None
        self.ema_v1 = None
        self.ema_v2 = None
        self.state = "SEARCHING"
        self.confidence = 0
        self.lost_frames = 0
        self.frame_number = 0
        self.last_predictions = []
        self.last_gauge_predictions = []
        self.last_needle_predictions = []
        self.last_pointer_tip = None
        self.dial_radius = None
        self.consecutive_move_frames = 0
        self.pending_move_angle = None
        self.tip_miss_frames = 0
        self.manual_center_lock = False
        self.manual_detector_lock = False
        self.locked_angle = None
        self.pending_center = None
        self.last_accepted_time = None

    def reading_age(self):
        """Seconds since a fresh observation last updated the tracked angle."""
        return None if self.last_accepted_time is None else time.monotonic() - self.last_accepted_time

    def resolve_dial_radius(self, fallback_radius, dial_box=None):
        """Keep face size independent of synthetic pointer length."""
        if dial_box is not None:
            self.dial_radius = max(40, int(min(dial_box[2:4]) * 0.5))
        elif self.dial_radius is None:
            self.dial_radius = max(40, int(fallback_radius))
        return self.dial_radius

    def refresh_center(self, center):
        """Confirm center movement twice before discarding old angle history."""
        import math
        if self.manual_center_lock or self.manual_detector_lock or center is None:
            self.pending_center = None
            return
        if self.locked_center is None or math.dist(center, self.locked_center) <= 3:
            self.pending_center = None
            return
        if self.pending_center is None or math.dist(center, self.pending_center) > 4:
            self.pending_center = center
            return
        self.reset(clear_pointer=True)
        self.locked_center = tuple(int(round(v)) for v in center)
        self.center_history.append(self.locked_center)
        self.pending_center = None

    def reset(self, clear_pointer=False, clear_lost_frames=False, clear_manual_lock=False):
        """Reset tracking to SEARCHING state.

        Args:
            clear_pointer: Also clear last_pointer_tip (used when detection
                loses the needle entirely).
            clear_lost_frames: Also reset lost_frames counter (used on gauge
                profile switch).
            clear_manual_lock: If True, explicitly clears manual center and detector lock.
                If False, preserves user's locked detector across temporary occlusions.
        """
        self.pending_center = None
        if not (self.manual_center_lock or self.manual_detector_lock) or clear_manual_lock:
            self.dial_radius = None
            self.locked_center = None
            self.center_history.clear()
            self.manual_center_lock = False
            self.manual_detector_lock = False
            self.locked_angle = None

        if not self.manual_detector_lock or clear_manual_lock:
            self.stable_angle = None
            self.last_accepted_time = None
            self.angle_history.clear()
            self.history_1.clear()
            self.history_2.clear()
            self.ema_v1 = None
            self.ema_v2 = None
            self.state = "SEARCHING"
            self.confidence = 0
            self.consecutive_move_frames = 0
            self.pending_move_angle = None
            self.tip_miss_frames = 0

        if clear_pointer:
            self.last_pointer_tip = None
        if clear_lost_frames:
            self.lost_frames = 0

    def update_angle(self, raw_angle, pointer_confidence, *, fresh=True, lock_frames=2):
        """Update once per observed tip; cached/coasting tips are display-only."""
        if not fresh:
            return
        if raw_angle is None:
            self.consecutive_move_frames = 0
            self.pending_move_angle = None
            if self.stable_angle is None:
                self.angle_history.clear()
            else:
                self.state = "HOLDING"
            return
        self.lost_frames = 0
        # If detector is locked, reject sudden 180 deg flips or massive outliers
        if self.manual_detector_lock and self.locked_angle is not None:
            dist_to_lock = circular_distance(raw_angle, self.locked_angle)
            if dist_to_lock > 75.0:
                # Wrong direction / opposite tail detected: ignore raw_angle and preserve locked needle orientation
                raw_angle = self.locked_angle

        if self.stable_angle is None:
            self.angle_history.append(raw_angle)
            self.state = "LOCKING"
            self.confidence = min(95, len(self.angle_history) * 30)
            if len(self.angle_history) >= lock_frames and max(
                circular_distance(raw_angle, a) for a in self.angle_history
            ) <= 10.0:
                self.stable_angle = weighted_circular_mean(self.angle_history)
                self.last_accepted_time = time.monotonic()
                self.state = "TRACKED"
                self.confidence = min(99, int(pointer_confidence * 100))
                if self.manual_detector_lock:
                    self.locked_angle = self.stable_angle
        else:
            delta = signed_angle_delta(self.stable_angle, raw_angle)
            abs_delta = abs(delta)

            if abs_delta <= 12.0:
                # Normal minor tracking noise: smooth low-pass update
                self.consecutive_move_frames = 0
                self.pending_move_angle = None
                self.angle_history.append(raw_angle)
                # Short, recency-weighted window follows real movement without
                # retaining a dozen stale observations at low detection rates.
                target_angle = weighted_circular_mean(list(self.angle_history)[-5:])
                self.stable_angle = (
                    self.stable_angle + 0.35 * signed_angle_delta(self.stable_angle, target_angle)
                ) % 360.0
                self.state = "TRACKED"
                self.confidence = min(99, max(60, int(pointer_confidence * 100)))
                self.last_accepted_time = time.monotonic()
                if self.manual_detector_lock:
                    self.locked_angle = self.stable_angle
            else:
                # Large angular difference: real needle rotation vs single-frame outlier
                if self.pending_move_angle is not None and circular_distance(raw_angle, self.pending_move_angle) <= 10.0:
                    self.consecutive_move_frames += 1
                else:
                    self.consecutive_move_frames = 1
                previous_move_angle = self.pending_move_angle
                self.pending_move_angle = raw_angle
                # Opposite rays commonly indicate the needle tail. Require
                # stronger confirmation while still allowing real large moves.
                required_observations = 4 if abs_delta >= 150.0 else 2
                if self.consecutive_move_frames >= required_observations:
                    # Two fresh, agreeing observations confirm needle movement.
                    # Fast-follow without getting stuck in HOLDING
                    self.stable_angle = raw_angle
                    self.last_accepted_time = time.monotonic()
                    self.angle_history.clear()
                    self.angle_history.extend((previous_move_angle, raw_angle))
                    self.consecutive_move_frames = 0
                    self.pending_move_angle = None
                    self.state = "TRACKED"
                    self.confidence = min(99, max(60, int(pointer_confidence * 100)))
                    if self.manual_detector_lock:
                        self.locked_angle = self.stable_angle
                else:
                    # Single-frame outlier: gently hold previous angle
                    self.state = "HOLDING"
                    self.confidence = max(40, self.confidence - 5)

