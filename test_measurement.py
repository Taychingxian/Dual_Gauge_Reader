"""Unit tests for measurement.py — pure angle and value computation functions."""

import math
import pytest

from measurement import (
    circular_distance,
    signed_angle_delta,
    circular_mean,
    value_from_angle,
    angle_in_scale_arc,
    angle_on_needle_side,
    rolling_median,
    ema_filter,
)

# ── Gauge profiles used across tests ─────────────────────────────────────────

UNIJIN_PROFILE = {
    "MIN_ANGLE": 140.0,
    "MAX_ANGLE": 52.0,
    "MIN_VAL_1": 0.0,  "MAX_VAL_1": 150.0, "UNIT_1": "PSI",
    "MIN_VAL_2": 0.0,  "MAX_VAL_2": 10.0,  "UNIT_2": "kgf/cm2",
    "SHOW_SECONDARY": True,
}

BADOTHERM_PROFILE = {
    "MIN_ANGLE": 135.0,
    "MAX_ANGLE": 45.0,
    "MIN_VAL_1": -1.0, "MAX_VAL_1": 15.0, "UNIT_1": "bar",
    "MIN_VAL_2": -14.5, "MAX_VAL_2": 217.5, "UNIT_2": "PSI",
    "SHOW_SECONDARY": False,
}


# ═══════════════════════════════════════════════════════════════════════════════
# circular_distance
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircularDistance:
    """Shortest angular distance between two angles (always 0–180)."""

    def test_same_angle(self):
        assert circular_distance(90.0, 90.0) == 0.0

    def test_opposite(self):
        assert circular_distance(0.0, 180.0) == 180.0

    def test_simple_gap(self):
        assert circular_distance(10.0, 40.0) == 30.0

    def test_wraparound_short_path(self):
        """350° and 10° are 20° apart, not 340°."""
        assert circular_distance(350.0, 10.0) == 20.0

    def test_symmetry(self):
        assert circular_distance(30.0, 270.0) == circular_distance(270.0, 30.0)

    def test_full_circle(self):
        """0° and 360° are the same angle."""
        assert circular_distance(0.0, 360.0) == 0.0

    def test_negative_input(self):
        """Should handle negative angles gracefully."""
        assert circular_distance(-10.0, 10.0) == 20.0

    def test_large_angles(self):
        """Angles beyond 360° should still work."""
        assert circular_distance(720.0, 10.0) == 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# signed_angle_delta
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignedAngleDelta:
    """Signed shortest rotation from current to target (-180 to +180)."""

    def test_no_rotation(self):
        assert signed_angle_delta(90.0, 90.0) == 0.0

    def test_clockwise(self):
        """10° → 40° is a +30° rotation."""
        assert signed_angle_delta(10.0, 40.0) == pytest.approx(30.0)

    def test_counterclockwise(self):
        """40° → 10° is a -30° rotation."""
        assert signed_angle_delta(40.0, 10.0) == pytest.approx(-30.0)

    def test_wraparound_clockwise(self):
        """350° → 10° should be +20° (short way), not -340°."""
        assert signed_angle_delta(350.0, 10.0) == pytest.approx(20.0)

    def test_wraparound_counterclockwise(self):
        """10° → 350° should be -20° (short way), not +340°."""
        assert signed_angle_delta(10.0, 350.0) == pytest.approx(-20.0)

    def test_exactly_opposite(self):
        """Result at ±180° boundary."""
        result = signed_angle_delta(0.0, 180.0)
        assert abs(result) == pytest.approx(180.0)


# ═══════════════════════════════════════════════════════════════════════════════
# circular_mean
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircularMean:
    """Vector mean of angles — must handle wraparound correctly."""

    def test_single_angle(self):
        assert circular_mean([45.0]) == pytest.approx(45.0)

    def test_simple_average(self):
        """Mean of 80° and 100° is 90°."""
        assert circular_mean([80.0, 100.0]) == pytest.approx(90.0)

    def test_wraparound_mean(self):
        """Mean of 350° and 10° should be 0° (or 360°), not 180°."""
        result = circular_mean([350.0, 10.0])
        # Accept 0° or 360° (both represent the same direction)
        assert result == pytest.approx(0.0, abs=0.1) or result == pytest.approx(360.0, abs=0.1)

    def test_three_clustered_angles(self):
        """Mean of 358°, 0°, 2° should be near 0°."""
        result = circular_mean([358.0, 0.0, 2.0])
        assert circular_distance(result, 0.0) < 2.0

    def test_uniform_distribution_cancels(self):
        """Evenly spaced angles have undefined mean — magnitude is ~0."""
        # 0°, 120°, 240° — vectors cancel out, atan2 can return anything
        # We just verify it doesn't crash
        circular_mean([0.0, 120.0, 240.0])

    def test_south_direction(self):
        """Mean of angles clustering around 270°."""
        result = circular_mean([260.0, 270.0, 280.0])
        assert result == pytest.approx(270.0, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# value_from_angle
# ═══════════════════════════════════════════════════════════════════════════════

class TestValueFromAngle:
    """Convert pointer angle to gauge scale readings."""

    def test_unijin_at_minimum(self):
        """Angle at MIN_ANGLE should read 0 PSI / 0 kgf/cm²."""
        v1, v2, exceeds = value_from_angle(140.0, UNIJIN_PROFILE)
        assert v1 == pytest.approx(0.0)
        assert v2 == pytest.approx(0.0)
        assert exceeds is False

    def test_unijin_at_maximum(self):
        """Angle at MAX_ANGLE should read 150 PSI / 10 kgf/cm²."""
        v1, v2, exceeds = value_from_angle(52.0, UNIJIN_PROFILE)
        assert v1 == pytest.approx(150.0)
        assert v2 == pytest.approx(10.0)
        assert exceeds is False

    def test_unijin_midpoint(self):
        """Midpoint angle should give ~half-scale reading."""
        sweep = (52.0 - 140.0) % 360.0  # = 272°
        mid_angle = (140.0 + sweep / 2) % 360.0
        v1, v2, exceeds = value_from_angle(mid_angle, UNIJIN_PROFILE)
        assert v1 == pytest.approx(75.0)
        assert v2 == pytest.approx(5.0)
        assert exceeds is False

    def test_badotherm_at_minimum(self):
        """Angle at MIN_ANGLE should read -1 bar."""
        v1, v2, exceeds = value_from_angle(135.0, BADOTHERM_PROFILE)
        assert v1 == pytest.approx(-1.0)
        assert exceeds is False

    def test_badotherm_at_maximum(self):
        """Angle at MAX_ANGLE should read 15 bar."""
        v1, v2, exceeds = value_from_angle(45.0, BADOTHERM_PROFILE)
        assert v1 == pytest.approx(15.0)
        assert exceeds is False

    def test_exceeds_limit_flag(self):
        """Angle in the dead zone (past MAX, before MIN) should flag exceeds_limit."""
        # For UNIJIN: MIN=140°, MAX=52° — arc sweeps 272° clockwise through 0°
        # The dead zone is 52° → 140° (the short way). 90° sits in this zone.
        _, _, exceeds = value_from_angle(90.0, UNIJIN_PROFILE)
        assert exceeds is True

    def test_clamping_keeps_values_in_range(self):
        """Even when exceeds_limit is True, values are clamped to scale range."""
        v1, v2, exceeds = value_from_angle(200.0, UNIJIN_PROFILE)
        assert 0.0 <= v1 <= 150.0
        assert 0.0 <= v2 <= 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# angle_in_scale_arc
# ═══════════════════════════════════════════════════════════════════════════════

class TestMedianAndEmaFilter:
    """Median filtering should reject spikes; EMA should smooth the result."""

    def test_rolling_median_rejects_single_spike(self):
        values = [10.0, 10.0, 10.0, 100.0, 10.0]
        assert rolling_median(values, window=5) == pytest.approx(10.0)

    def test_ema_filter_smooths_new_measurement(self):
        assert ema_filter(10.0, 20.0, alpha=0.5) == pytest.approx(15.0)


class TestAngleInScaleArc:
    """Check if an angle falls within a gauge's active scale arc."""

    def test_at_min_angle(self):
        assert angle_in_scale_arc(140.0, UNIJIN_PROFILE) is True

    def test_at_max_angle(self):
        assert angle_in_scale_arc(52.0, UNIJIN_PROFILE) is True

    def test_midscale(self):
        """An angle in the middle of the arc should be valid."""
        assert angle_in_scale_arc(0.0, UNIJIN_PROFILE) is True

    def test_outside_arc(self):
        """An angle clearly outside (with no margin overlap) should be False."""
        # For UNIJIN: arc goes 140° → (through 0°) → 52°
        # 90° is between 52° and 140° on the "dead zone" side
        assert angle_in_scale_arc(96.0, UNIJIN_PROFILE, margin=0.0) is False

    def test_margin_allows_near_miss(self):
        """Angle just past the max should still pass with default margin."""
        # 52° + 15° = 67° — within 20° margin
        assert angle_in_scale_arc(67.0, UNIJIN_PROFILE, margin=20.0) is True

    def test_zero_margin_strict(self):
        """With margin=0, must be exactly within the arc."""
        assert angle_in_scale_arc(140.0, UNIJIN_PROFILE, margin=0.0) is True
        assert angle_in_scale_arc(52.0, UNIJIN_PROFILE, margin=0.0) is True

    def test_wraparound_near_min(self):
        """Angle just before MIN (wrapping around) should pass with margin."""
        # UNIJIN MIN=140°, so 125° is 15° before MIN → within 20° margin
        assert angle_in_scale_arc(125.0, UNIJIN_PROFILE, margin=20.0) is True

    def test_badotherm_midscale(self):
        """Badotherm arc: 135° → (through 0°) → 45°."""
        assert angle_in_scale_arc(0.0, BADOTHERM_PROFILE) is True
        assert angle_in_scale_arc(270.0, BADOTHERM_PROFILE) is True


# ═══════════════════════════════════════════════════════════════════════════════
# angle_on_needle_side
# ═══════════════════════════════════════════════════════════════════════════════

class TestAngleOnNeedleSide:
    """Filter angles to the expected needle side of the gauge."""

    def test_no_side_constraint_always_true(self):
        """Profiles without NEEDLE_SIDE_ANGLE should always return True."""
        assert angle_on_needle_side(0.0, UNIJIN_PROFILE) is True
        assert angle_on_needle_side(180.0, UNIJIN_PROFILE) is True
        assert angle_on_needle_side(999.0, UNIJIN_PROFILE) is True

    def test_within_tolerance(self):
        profile = {"NEEDLE_SIDE_ANGLE": 90.0, "NEEDLE_SIDE_TOLERANCE": 30.0}
        assert angle_on_needle_side(90.0, profile) is True
        assert angle_on_needle_side(100.0, profile) is True
        assert angle_on_needle_side(70.0, profile) is True

    def test_outside_tolerance(self):
        profile = {"NEEDLE_SIDE_ANGLE": 90.0, "NEEDLE_SIDE_TOLERANCE": 30.0}
        assert angle_on_needle_side(180.0, profile) is False
        assert angle_on_needle_side(0.0, profile) is False

    def test_wraparound_tolerance(self):
        """Needle side at 350° with 30° tolerance should include 10°."""
        profile = {"NEEDLE_SIDE_ANGLE": 350.0, "NEEDLE_SIDE_TOLERANCE": 30.0}
        assert angle_on_needle_side(10.0, profile) is True
        assert angle_on_needle_side(340.0, profile) is True

    def test_default_tolerance(self):
        """Without explicit tolerance, default is 70°."""
        profile = {"NEEDLE_SIDE_ANGLE": 180.0}
        assert angle_on_needle_side(180.0, profile) is True
        assert angle_on_needle_side(240.0, profile) is True   # 60° away < 70°
        assert angle_on_needle_side(260.0, profile) is False   # 80° away > 70°
