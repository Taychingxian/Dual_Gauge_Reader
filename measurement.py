"""Pure angle and value computation functions for gauge reading."""
# Testing and debugging functions for gauge reading, including circular distance, signed angle delta, circular mean, and value conversion from angle to gauge readings.
import math


def rolling_median(values, window=5):
    """Return the median of the last N values while ignoring isolated spikes."""
    if not values:
        raise ValueError("rolling_median requires at least one value")
    trimmed = list(values)[-window:]
    ordered = sorted(trimmed)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)
    return float(ordered[midpoint])


def ema_filter(previous_value, current_value, alpha=0.18):
    """Apply a simple exponential moving average to smooth one reading."""
    if previous_value is None:
        return float(current_value)
    return float(previous_value + alpha * (current_value - previous_value))


def circular_distance(first_angle, second_angle):
    """Return the shortest angular distance between two angles (0-180 degrees)."""
    difference = abs(first_angle - second_angle) % 360.0
    return min(difference, 360.0 - difference)


def signed_angle_delta(current_angle, target_angle):
    """Return the signed shortest rotation from current_angle to target_angle (-180 to +180)."""
    return (target_angle - current_angle + 180.0) % 360.0 - 180.0


def circular_mean(angles):
    """Compute the circular (vector) mean of a sequence of angles in degrees."""
    sine = sum(math.sin(math.radians(a)) for a in angles)
    cosine = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(sine, cosine)) % 360.0


def weighted_circular_mean(angles, decay=0.7):
    """Compute weighted circular mean with exponential recency bias.

    More recent values (later in the sequence) receive exponentially higher
    weight. This makes the tracker more responsive to real needle movement
    while still averaging out per-frame noise.

    Args:
        angles: Sequence of angles in degrees (oldest first).
        decay: Weight decay factor (0–1). Lower = stronger recency bias.
            0.7 means each older sample gets 70% the weight of its successor.
    """
    n = len(angles)
    if n == 0:
        return 0.0
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)
    sine = sum(w * math.sin(math.radians(a)) for w, a in zip(weights, angles))
    cosine = sum(w * math.cos(math.radians(a)) for w, a in zip(weights, angles))
    return math.degrees(math.atan2(sine / total_weight, cosine / total_weight)) % 360.0


def value_from_angle(angle, profile):
    """Convert a pointer angle to gauge readings (primary and secondary).

    Returns:
        (value_1, value_2, exceeds_limit): The two scale readings and whether
        the angle is outside the gauge's defined range.
    """
    min_angle = profile["MIN_ANGLE"]
    sweep = (profile["MAX_ANGLE"] - min_angle) % 360.0
    travel = (angle - min_angle) % 360.0
    # Handle needle resting slightly before MIN mark (e.g. resting against physical stop pin)
    margin = 8.0
    if travel > 180.0 and (360.0 - travel) <= margin:
        travel -= 360.0
    raw_ratio = travel / sweep
    ratio = min(max(raw_ratio, 0.0), 1.0)
    value_1 = profile["MIN_VAL_1"] + ratio * (profile["MAX_VAL_1"] - profile["MIN_VAL_1"])
    value_2 = profile["MIN_VAL_2"] + ratio * (profile["MAX_VAL_2"] - profile["MIN_VAL_2"])
    offset = profile.get("VALUE_OFFSET_1", 0.0)
    value_1 += offset
    value_2 += offset * (profile["MAX_VAL_2"] - profile["MIN_VAL_2"]) / (profile["MAX_VAL_1"] - profile["MIN_VAL_1"])
    if "SECONDARY_PER_PRIMARY" in profile:
        # Dual-unit dials need not put their last printed marks at the same angle.
        value_2 = value_1 * profile["SECONDARY_PER_PRIMARY"]
    at_resting_min = -margin <= travel <= 0.0
    outside = (not (0.0 <= raw_ratio <= 1.0) or not (profile["MIN_VAL_1"] <= value_1 <= profile["MAX_VAL_1"])) and not at_resting_min
    return value_1, value_2, outside


def angle_in_scale_arc(angle, profile, margin=20.0):
    """Check whether an angle falls within the gauge's scale arc (with margin).

    Handles circular wrapping for gauges whose arc crosses the 0/360 boundary.
    """
    sweep = (profile["MAX_ANGLE"] - profile["MIN_ANGLE"]) % 360.0
    travel = (angle - profile["MIN_ANGLE"]) % 360.0
    # Handles circular wrapping margin just before the 0/MIN mark
    if travel > 180.0 and (360.0 - travel) <= margin:
        return True
    return travel <= (sweep + margin)


def angle_on_needle_side(angle, profile):
    """Check whether an angle is on the expected side of the needle."""
    side_angle = profile.get("NEEDLE_SIDE_ANGLE")
    if side_angle is None:
        return True
    tolerance = profile.get("NEEDLE_SIDE_TOLERANCE", 70.0)
    return circular_distance(angle, side_angle) <= tolerance
