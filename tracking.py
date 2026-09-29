"""Tracking state machine for gauge needle stabilization."""

from collections import deque


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
        self.consecutive_move_frames = 0
        self.tip_miss_frames = 0

    def reset(self, clear_pointer=False, clear_lost_frames=False):
        """Reset tracking to SEARCHING state.

        Args:
            clear_pointer: Also clear last_pointer_tip (used when detection
                loses the needle entirely).
            clear_lost_frames: Also reset lost_frames counter (used on gauge
                profile switch).
        """
        self.locked_center = None
        self.center_history.clear()
        self.stable_angle = None
        self.angle_history.clear()
        self.history_1.clear()
        self.history_2.clear()
        self.ema_v1 = None
        self.ema_v2 = None
        self.state = "SEARCHING"
        self.confidence = 0
        self.consecutive_move_frames = 0
        self.tip_miss_frames = 0
        if clear_pointer:
            self.last_pointer_tip = None
        if clear_lost_frames:
            self.lost_frames = 0
