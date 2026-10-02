"""Polar Annulus Unwrapping module for industrial pressure gauge reading.

Transforms the circular dial face into a flat, 1D/2D unwrapped linear ribbon
using OpenCV warpPolar. Measures the needle position along the scale arc,
eliminating center-pivot sensitivity and perspective tilt distortions.
"""

import math
import cv2
import numpy as np

from measurement import (
    angle_in_scale_arc,
    angle_on_needle_side,
    circular_distance,
    value_from_angle,
)


def detect_radial_pointer(frame, center, dial_radius, profile):
    """Find a dark shaft spanning the inner dial, independently of boxes/history.

    Tick marks only occupy the outer ring; text interrupts a ray. Require
    dark support in every radial section to distinguish these from a needle.
    """
    if frame is None or center is None or dial_radius < 40:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Some dials have short pointers ending near 65% of the bezel radius.
    # Stay inside the shaft, before its taper and the outer scale ticks.
    radii = np.linspace(dial_radius * 0.18, dial_radius * 0.60, 96)
    angles = np.deg2rad(np.arange(720) * 0.5)
    xs = (center[0] + radii[:, None] * np.cos(angles)).astype(np.float32)
    ys = (center[1] + radii[:, None] * np.sin(angles)).astype(np.float32)
    samples = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # Adapt to lighting using the bright dial face at each radius.
    background = np.percentile(samples, 75, axis=1)[:, None]
    dark = samples < np.minimum(background - 45, background * 0.65)
    sections = np.array_split(dark, 6, axis=0)
    support = np.min([s.mean(axis=0) for s in sections], axis=0)
    valid = np.array([angle_on_needle_side(c * 0.5, profile)
                      for c in range(720)])
    support[~valid] = 0
    best = int(np.argmax(support))
    if support[best] < 0.70:
        return None
    # Among supported shafts prefer outward continuation: a thick, short
    # counterweight can otherwise be darker than the actual pointer.
    outer_radii = np.linspace(dial_radius * 0.60, dial_radius * 0.85, 48)
    outer_xs = (center[0] + outer_radii[:, None] * np.cos(angles)).astype(np.float32)
    outer_ys = (center[1] + outer_radii[:, None] * np.sin(angles)).astype(np.float32)
    outer_samples = cv2.remap(gray, outer_xs, outer_ys, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    outer_background = np.percentile(outer_samples, 75, axis=1)[:, None]
    outer_dark = outer_samples < np.minimum(outer_background - 45, outer_background * 0.65)
    score = support + 0.5 * outer_dark.mean(axis=0)
    score[(support < 0.70) | ~valid] = -1
    best = int(np.argmax(score))
    # Break broad-shaft ties with overall darkness.
    contenders = np.flatnonzero(score >= score[best] - 0.03)
    best = int(contenders[np.argmin(samples[:, contenders].mean(axis=0))])
    angle = best * 0.5
    radius = dial_radius * 0.78
    return {
        "angle": angle,
        "tip": (int(round(center[0] + radius * math.cos(math.radians(angle)))),
                int(round(center[1] + radius * math.sin(math.radians(angle))))),
        "confidence": float(support[best]),
    }


def polar_unwrap_reading(
    frame,
    center,
    dial_radius,
    profile,
    reference_angle=None,
    num_intervals=None,
):
    """Read a gauge value using polar coordinate annulus unwrapping.

    Args:
        frame: BGR input image.
        center: (gx, gy) tuple of gauge center pivot.
        dial_radius: Radius of the dial face in pixels.
        profile: Dict containing gauge configuration (MIN_ANGLE, MAX_ANGLE, etc.).
        reference_angle: Optional prior angle in degrees (e.g., from Hough or tracking)
                         to weight search against background artifacts.
        num_intervals: Number of segments for piecewise scale calibration.

    Returns:
        dict with keys:
            value_1: Primary scale value (PSI or bar)
            value_2: Secondary scale value (kgf/cm2 or PSI)
            needle_angle: Measured pointer angle in degrees (0..360)
            needle_col: Measured needle column in the 720-wide strip
            tick_cols: List of detected tick column coordinates
            method: 'polar_unwrap'
            confidence: Float between 0.0 and 1.0
            exceeds_limit: True if needle is outside profile range
            debug_strip: Cropped unwrapped grayscale annulus image (H x 720)
        or None if frame / center / radius are invalid.
    """
    if frame is None or center is None:
        return None

    if num_intervals is None:
        num_intervals = profile.get("POLAR_INTERVALS", 10)

    gx, gy = int(round(center[0])), int(round(center[1]))
    h_img, w_img = frame.shape[:2]
    if gx < 0 or gx >= w_img or gy < 0 or gy >= h_img:
        return None

    if dial_radius is not None and dial_radius <= 0:
        return None
    if dial_radius is None:
        dial_radius = int(min(h_img, w_img) * 0.42)
    dial_radius = int(round(dial_radius))

    num_cols = 720  # 0.5 degrees per column

    try:
        # cv2.warpPolar: width=dial_radius (radial axis), height=720 (angular axis 0..360 deg)
        polar = cv2.warpPolar(
            frame,
            (dial_radius, num_cols),
            (gx, gy),
            dial_radius,
            cv2.INTER_LINEAR | cv2.WARP_POLAR_LINEAR,
        )
    except Exception:
        return None

    if polar is None or polar.size == 0:
        return None

    # Transpose so rows = radius, columns = angle (0..360 deg, col c = c * 0.5 deg)
    polar = cv2.transpose(polar)

    # Annulus: sample ring where scale ticks and pointer tip interact (60% to 88% radius)
    r_min = max(0, int(dial_radius * 0.60))
    r_max = min(dial_radius, int(dial_radius * 0.88))
    if r_max - r_min < 5:
        return None

    annulus = polar[r_min:r_max, :]
    if len(annulus.shape) == 3:
        gray = cv2.cvtColor(annulus, cv2.COLOR_BGR2GRAY)
    else:
        gray = annulus

    # Column-wise intensity profile (needle appears as a dark vertical band / intensity valley)
    col_sums = np.mean(gray, axis=0).astype(np.float32)

    # Smooth profile to suppress noise while preserving needle trough
    ksize = 15
    smoothed = cv2.GaussianBlur(col_sums.reshape(1, -1), (ksize, 1), 0).flatten()

    # ── Constrain needle search to active scale arc ───────────────────────────
    # Exclude dead-zone (e.g. 6 o'clock mounting stem, casing bezel, brand text)
    masked_profile = smoothed.copy()
    valid_mask = np.zeros(num_cols, dtype=bool)

    for col in range(num_cols):
        deg = col * 360.0 / num_cols
        in_arc = angle_in_scale_arc(deg, profile, margin=18.0)
        on_side = angle_on_needle_side(deg, profile)
        if in_arc and on_side:
            valid_mask[col] = True
        else:
            masked_profile[col] = 1e9  # Mask out dead zone

    if not np.any(valid_mask):
        return None

    shaft = detect_radial_pointer(frame, center, dial_radius, profile)
    if shaft is not None:
        reference_angle = shaft["angle"]
        # Annulus features cannot override a verified center-connected shaft.
        for col in range(num_cols):
            if circular_distance(col * 0.5, reference_angle) <= 6.0:
                valid_mask[col] = True
                masked_profile[col] = smoothed[col]
            else:
                valid_mask[col] = False
                masked_profile[col] = 1e9

    # Optional reference angle weighting to avoid jumping to dial screws or logos
    if reference_angle is not None and math.isfinite(reference_angle):
        ref_norm = reference_angle % 360.0
        for col in range(num_cols):
            if valid_mask[col]:
                deg = col * 360.0 / num_cols
                dist = circular_distance(deg, ref_norm)
                # Soft penalty for large deviations from expected needle direction
                masked_profile[col] += float((dist / 35.0) ** 2 * 20.0)

    min_idx = int(np.argmin(masked_profile))
    if not valid_mask[min_idx]:
        return None

    # Sub-pixel parabolic refinement around the needle minimum
    step = 0.0
    if 0 < min_idx < num_cols - 1:
        y1 = float(smoothed[min_idx - 1])
        y2 = float(smoothed[min_idx])
        y3 = float(smoothed[min_idx + 1])
        denom = y1 - 2.0 * y2 + y3
        if denom > 1e-4:
            step = 0.5 * (y1 - y3) / denom
            step = max(-1.0, min(1.0, step))

    needle_col = float(min_idx) + step
    needle_angle = (needle_col * 360.0 / num_cols) % 360.0
    if shaft is not None:
        # Annulus minima include printed ticks and displaced needle shadows.
        # Keep the verified shaft direction rather than snapping onto these.
        needle_angle = float(shaft["angle"])
        needle_col = needle_angle * num_cols / 360.0

    # ── Detect major tick marks (with needle masked out) ───────────────────────
    # Mask out needle column so tick search NEVER snaps onto the needle itself!
    tick_search_profile = smoothed.copy()
    needle_c_int = int(round(needle_col))
    mask_radius = max(6, int(num_cols * 7.0 / 360.0))  # +/- 7 degrees
    for dc in range(-mask_radius, mask_radius + 1):
        tick_search_profile[(needle_c_int + dc) % num_cols] = 1e9

    min_a = profile["MIN_ANGLE"]
    max_a = profile["MAX_ANGLE"]
    sweep = (max_a - min_a) % 360.0
    min_v1 = profile["MIN_VAL_1"]
    max_v1 = profile["MAX_VAL_1"]

    expected_ticks = []
    for i in range(num_intervals + 1):
        a = (min_a + i * sweep / num_intervals) % 360.0
        v = min_v1 + i * (max_v1 - min_v1) / num_intervals
        expected_ticks.append((a, v))

    tick_cols = []
    actual_ticks = []
    interval_cols = (sweep / num_intervals) * (num_cols / 360.0)
    search_window = max(3, int(interval_cols * 0.30))

    for a, v in expected_ticks:
        exp_col = int(round((a % 360.0) * num_cols / 360.0))
        best_col = exp_col
        min_t_val = 1e9
        for c in range(exp_col - search_window, exp_col + search_window + 1):
            val = tick_search_profile[c % num_cols]
            if val < min_t_val:
                min_t_val = val
                best_col = c % num_cols

        # If a valid local tick dip was found (not masked by needle)
        if min_t_val < 1e8:
            actual_ticks.append((best_col, v))
            tick_cols.append(best_col)
        else:
            # Fallback to expected geometric tick column
            actual_ticks.append((exp_col % num_cols, v))
            tick_cols.append(exp_col % num_cols)

    # ── Piecewise scale interpolation ─────────────────────────────────────────
    # Sort ticks by expected value order and unwrap column coordinates relative to needle
    actual_ticks.sort(key=lambda t: t[1])

    # Unwrap columns monotonically
    unwrapped_ticks = []
    offset = 0.0
    for i, (col, v) in enumerate(actual_ticks):
        if i == 0:
            unwrapped_ticks.append((float(col), v))
        else:
            prev_col = unwrapped_ticks[-1][0]
            curr_col = float(col) + offset
            # Circular wrap detection across the 0/num_cols boundary
            while curr_col < prev_col - (num_cols * 0.4):
                offset += num_cols
                curr_col += num_cols
            while curr_col > prev_col + (num_cols * 0.6):
                offset -= num_cols
                curr_col -= num_cols
            unwrapped_ticks.append((curr_col, v))

    # Unwrap needle_col into the tick domain
    needle_unwrapped = needle_col
    t_start = unwrapped_ticks[0][0]
    t_end = unwrapped_ticks[-1][0]
    best_dist = float("inf")
    best_nc = needle_col
    for wrap_k in (-1, 0, 1):
        test_nc = needle_col + wrap_k * num_cols
        dist = 0.0
        if test_nc < t_start:
            dist = t_start - test_nc
        elif test_nc > t_end:
            dist = test_nc - t_end
        else:
            best_nc = test_nc
            break
        if dist < best_dist:
            best_dist = dist
            best_nc = test_nc
    needle_unwrapped = best_nc

    # Interpolate between the bounding ticks
    val1 = None
    if needle_unwrapped <= unwrapped_ticks[0][0]:
        val1 = unwrapped_ticks[0][1]
    elif needle_unwrapped >= unwrapped_ticks[-1][0]:
        val1 = unwrapped_ticks[-1][1]
    else:
        for i in range(len(unwrapped_ticks) - 1):
            c1, v1 = unwrapped_ticks[i]
            c2, v2 = unwrapped_ticks[i + 1]
            if c1 <= needle_unwrapped <= c2:
                if abs(c2 - c1) > 1e-4:
                    ratio = (needle_unwrapped - c1) / (c2 - c1)
                    val1 = v1 + ratio * (v2 - v1)
                else:
                    val1 = v1
                break

    if val1 is None:
        val1, val2, outside = value_from_angle(needle_angle, profile)
    else:
        # Compute secondary value
        if "SECONDARY_PER_PRIMARY" in profile:
            val2 = val1 * profile["SECONDARY_PER_PRIMARY"]
        else:
            span_1 = max_v1 - min_v1
            ratio_1 = (val1 - min_v1) / span_1 if span_1 > 0 else 0.0
            min_v2 = profile.get("MIN_VAL_2", 0.0)
            max_v2 = profile.get("MAX_VAL_2", 0.0)
            val2 = min_v2 + ratio_1 * (max_v2 - min_v2)

    # Check limits
    # Interpolation clamps to the end tick, so check angular travel as well.
    # Otherwise a pointer beyond MAX silently reports the maximum value.
    exceeds_limit = (val1 < min_v1 or val1 > max_v1)
    # Check resting pin tolerance at MIN
    travel_deg = (needle_angle - min_a) % 360.0
    exceeds_limit = exceeds_limit or travel_deg > sweep
    if travel_deg > 180.0 and (360.0 - travel_deg) <= 8.0:
        exceeds_limit = False

    confidence = 0.92 if np.std(col_sums) > 5.0 else 0.50

    return {
        "value_1": float(val1),
        "value_2": float(val2),
        "needle_angle": float(needle_angle),
        "needle_col": float(needle_col),
        "tick_cols": tick_cols,
        "method": "polar_unwrap",
        "confidence": confidence,
        "exceeds_limit": exceeds_limit,
        "debug_strip": annulus,
    }


def draw_polar_debug(display_frame, unwrapped_strip, needle_col, tick_cols, profile):
    """Draw a compact inset visualization of the unwrapped polar ribbon."""
    if display_frame is None or unwrapped_strip is None:
        return display_frame

    h, w = unwrapped_strip.shape[:2]
    if h <= 0 or w <= 0:
        return display_frame

    target_w = min(display_frame.shape[1] - 40, 560)
    target_h = max(24, int(target_w * h / float(w)))

    if len(unwrapped_strip.shape) == 2:
        strip_color = cv2.cvtColor(unwrapped_strip, cv2.COLOR_GRAY2BGR)
    else:
        strip_color = unwrapped_strip.copy()

    # Draw detected ticks in green
    if tick_cols:
        for tc in tick_cols:
            c = int(round(tc)) % w
            cv2.line(strip_color, (c, 0), (c, h), (0, 220, 0), 1)

    # Draw detected needle in bright red
    if needle_col is not None and math.isfinite(needle_col):
        nc = int(round(needle_col)) % w
        cv2.line(strip_color, (nc, 0), (nc, h), (0, 0, 255), 2)
        cv2.circle(strip_color, (nc, h // 2), 3, (0, 255, 255), -1)

    resized = cv2.resize(strip_color, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Semi-transparent backing banner
    x_offset = (display_frame.shape[1] - target_w) // 2
    y_offset = display_frame.shape[0] - target_h - 45

    if y_offset > 0 and x_offset > 0:
        cv2.rectangle(
            display_frame,
            (x_offset - 4, y_offset - 16),
            (x_offset + target_w + 4, y_offset + target_h + 4),
            (20, 20, 20),
            -1,
        )
        display_frame[y_offset : y_offset + target_h, x_offset : x_offset + target_w] = resized
        cv2.putText(
            display_frame,
            "POLAR ANNULUS (0..360 deg) | Green: Ticks, Red: Needle",
            (x_offset, y_offset - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 220, 220),
            1,
        )

    return display_frame
