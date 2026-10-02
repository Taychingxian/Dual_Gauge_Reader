"""Computer vision detection functions for gauge center and pointer tip.

Optimizations over baseline:
- CLAHE preprocessing for consistent contrast across lighting conditions
- Adaptive Canny thresholds based on image statistics
- Edge masking to limit Hough detection to gauge area (fewer false lines)
- Precomputed circle scores (3x → 1x computation reduction)
"""

import cv2
import numpy as np
import math

from measurement import angle_in_scale_arc, angle_on_needle_side, circular_distance

# Reusable CLAHE object — avoids per-call allocation
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _enhance_gray(frame):
    """Convert BGR to grayscale with CLAHE contrast enhancement.

    CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes
    local contrast, making edge detection robust to shadows, glare, and
    uneven industrial lighting.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(gray)


def _adaptive_canny(gray, sigma=0.33):
    """Canny edge detection with auto-computed thresholds.

    Instead of fixed thresholds that only work for specific lighting,
    thresholds are derived from the image's median intensity — adapting
    automatically to bright, dim, or mixed lighting conditions.
    """
    median_val = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median_val))
    upper = int(min(255, (1.0 + sigma) * median_val))
    return cv2.Canny(gray, lower, upper)


def detect_gauge_center(frame, anchor=None, needle_box=None):
    """Detect the gauge dial center using Hough circle detection.

    Constrained: Requires an anchor point (e.g. detected needle center) to prevent
    false-positive circle detection on background objects (user's nose, light fixtures, etc.).
    Includes filters to reject overexposed lights/glare and featureless skin/wall patches.

    Args:
        frame: BGR image.
        anchor: Required (x, y) anchor point of the detected pointer to constrain circle search.
        needle_box: Optional (x, y, w, h) bounding box of the needle to constrain radius search.

    Returns:
        (x, y) center coordinates, or None if no suitable circle is found.
    """
    if anchor is None:
        anchor = (frame.shape[1] // 2, frame.shape[0] // 2)

    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.Canny(gray, 80, 180)

    # Needle dimensions provide strong priors on dial radius
    min_dim = min(small_frame.shape[:2])
    max_dim = max(small_frame.shape[:2])
    if needle_box is not None:
        needle_len = max(needle_box[2], needle_box[3]) * scale
        min_r = max(int(min_dim * 0.08), int(needle_len * 0.25))
        max_r = min(int(max_dim * 0.65), int(needle_len * 2.20))
        if min_r >= max_r:
            min_r = int(min_dim * 0.08)
            max_r = int(max_dim * 0.65)
    else:
        min_r = int(min_dim * 0.08)
        max_r = int(max_dim * 0.65)

    # Multi-pass Hough Circles with progressive sensitivity for challenging dials (e.g. Badotherm/UNIJIN)
    circles = None
    for param1, param2 in ((100, 36), (80, 26), (70, 20)):
        c = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(50 * scale),
            param1=param1, param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if c is not None and len(c[0]) > 0:
            circles = c[0]
            break

    small_anchor = (anchor[0] * scale, anchor[1] * scale)
    image_center = (small_frame.shape[1] / 2, small_frame.shape[0] / 2)

    valid_circles = []
    if circles is not None:
        # 1. Proximity, brightness, and texture verification per circle
        for circle in circles:
            cx, cy, r = map(int, circle)
            margin_x = int(small_frame.shape[1] * 0.04)
            margin_y = int(small_frame.shape[0] * 0.04)
            if cx < margin_x or cx > small_frame.shape[1] - margin_x or cy < margin_y or cy > small_frame.shape[0] - margin_y:
                continue
            dist_to_anchor = math.hypot(circle[0] - small_anchor[0], circle[1] - small_anchor[1])
            # Needle anchor can be anywhere within the dial radius
            if dist_to_anchor > circle[2] * 0.98:
                continue

            cx, cy, r = map(int, circle)
            inner_r = max(6, int(r * 0.50))
            x1 = max(0, cx - inner_r)
            y1 = max(0, cy - inner_r)
            x2 = min(small_frame.shape[1], cx + inner_r)
            y2 = min(small_frame.shape[0], cy + inner_r)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            roi = gray[y1:y2, x1:x2]

            # 2. Light bulb / glare filter:
            if np.mean(roi > 248) > 0.60:
                continue

            # 3. Nose / skin / blank wall filter:
            if np.std(roi) < 8.0:
                continue

            valid_circles.append(circle)

    if not valid_circles:
        # Fallback: Contour detection around needle anchor to find dial boundary
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 15, 3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (min_dim * 0.20) ** 2:
                (cx, cy), r = cv2.minEnclosingCircle(cnt)
                if min_r <= r <= max_r and math.hypot(cx - small_anchor[0], cy - small_anchor[1]) <= r * 0.98:
                    valid_circles.append((cx, cy, r))
                    break

    if not valid_circles:
        return None

    def circle_score(circle):
        center_x, center_y, radius = map(int, circle)
        edge_hits = 0
        samples = 0
        for angle in range(0, 360, 4):
            radians = math.radians(angle)
            for radius_offset in (-2, -1, 0, 1, 2):
                x = int(round(center_x + math.cos(radians) * (radius + radius_offset)))
                y = int(round(center_y + math.sin(radians) * (radius + radius_offset)))
                if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                    samples += 1
                    edge_hits += int(edges[y, x] > 0)
        edge_score = edge_hits / max(samples, 1)
        anchor_score = math.hypot(circle[0] - small_anchor[0], circle[1] - small_anchor[1]) / max(circle[2], 1.0)
        center_score = math.hypot(
            circle[0] - image_center[0], circle[1] - image_center[1]
        ) / max(min(small_frame.shape[:2]), 1)
        return edge_score, anchor_score, center_score

    scored = [(circle_score(c), c) for c in valid_circles]
    best_score, candidate = max(
        scored,
        key=lambda s: (s[0][0], -s[0][1]),
    )
    if best_score[0] < 0.03:
        if candidate[2] * 0.98 < math.hypot(candidate[0] - small_anchor[0], candidate[1] - small_anchor[1]):
            return None
    inv_scale = 1.0 / scale
    return int(round(candidate[0] * inv_scale)), int(round(candidate[1] * inv_scale))


def detect_pointer_tip(frame, center, profile, pointer_predictions=None, reference_angle=None):
    """Find a center-connected line constrained to this gauge's scale arc.

    Uses CLAHE-enhanced edge detection with adaptive thresholds, masked
    to the gauge dial area to eliminate spurious lines from the background.

    Args:
        frame: BGR image.
        center: (x, y) gauge center coordinates.
        profile: Gauge profile dict with angle range parameters.
        pointer_predictions: Optional model predictions (unused, kept for API compat).
        reference_angle: Optional locked/expected angle in degrees to reject wrong direction.

    Returns:
        (x, y) tip coordinates, or None if no suitable line is found.
    """
    gx, gy = center
    gray = _enhance_gray(frame)
    dial_radius = int(min(gray.shape) * 0.52)

    edges = _adaptive_canny(gray)

    # SPEED+ACCURACY: mask edges outside the gauge dial area —
    # eliminates background lines that waste Hough computation and
    # can produce false positive needle detections
    mask = np.zeros_like(edges)
    cv2.circle(mask, (gx, gy), dial_radius + 10, 255, -1)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20,
        minLineLength=30, maxLineGap=15
    )
    if lines is None:
        return None

    candidates = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(int, line)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 30:
            continue

        distance_1 = math.hypot(x1 - gx, y1 - gy)
        distance_2 = math.hypot(x2 - gx, y2 - gy)
        near_distance = min(distance_1, distance_2)

        far_point = (x2, y2) if distance_1 < distance_2 else (x1, y1)
        far_distance = max(distance_1, distance_2)

        # Check perpendicular distance from center to line
        dx_l = x2 - x1
        dy_l = y2 - y1
        perp = abs(dy_l * gx - dx_l * gy + x2 * y1 - y2 * x1) / max(length, 1.0)
        if perp > 35.0:
            continue

        near_limit = max(55.0, dial_radius * 0.22)
        # Radial lines (passing near center within 25px) can have near_dist up to 0.55*R
        # because center dial text/hub can obscure the root of the needle
        max_allowed_near = dial_radius * 0.55 if perp <= 25.0 else max(near_limit, length * 0.55)
        if near_distance > max_allowed_near:
            continue

        if far_distance > dial_radius or far_distance < 28.0:
            continue
        angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0
        in_arc = angle_in_scale_arc(angle, profile, margin=35.0)
        on_side = angle_on_needle_side(angle, profile)
        if on_side:
            arc_weight = 1.0 if in_arc else 0.70
            score = (far_distance ** 1.35) * length * arc_weight
            candidates.append((score, far_distance, far_point, angle, in_arc))

    raw_tip = _resolve_longer_needle(candidates, reference_angle=reference_angle)
    return verify_and_resolve_pointer_orientation(frame, center, raw_tip, profile, dial_radius=dial_radius)


def _resolve_longer_needle(candidates, reference_angle=None):
    """Filter out shorter opposite counterweights/tails, ensuring we choose the longer needle.

    A physical gauge needle often has a short counterweight or tail extending in the opposite
    direction (180 deg away) from the central hub. The true measuring tip is ALWAYS the longer
    extension that reaches outwards towards the scale marks.

    Args:
        candidates: List of tuples (score, far_dist, far_point, angle, in_arc)
        reference_angle: Optional locked/expected angle in degrees to reject wrong direction.

    Returns:
        (x, y) tip coordinates of the longer needle, or None if candidates is empty.
    """
    if not candidates:
        return None

    # Maximum reach among all candidates
    max_reach = max(c[1] for c in candidates)

    # Reject opposite tails and shorter stubs BEFORE consulting history.
    # The true measuring needle is ALWAYS the longer one reaching towards the scale.
    filtered = []
    for cand in candidates:
        score, far_dist, far_point, angle, in_arc = cand
        is_short = False
        if max_reach >= 40.0 and far_dist < max_reach * 0.65:
            is_short = True
        else:
            for other_score, other_dist, _, other_angle, _ in candidates:
                if other_dist > far_dist * 1.15 and circular_distance(angle, other_angle) >= 90.0:
                    is_short = True
                    break
        if not is_short:
            filtered.append(cand)

    valid = filtered if filtered else candidates

    if reference_angle is not None:
        sector_candidates = [c for c in valid if circular_distance(c[3], reference_angle) <= 85.0]
        if sector_candidates:
            valid = sector_candidates

    def ranking_key(c):
        score, far_dist, far_point, angle, in_arc = c
        reach_weight = (far_dist / max(max_reach, 1.0)) ** 1.5
        base = score * reach_weight
        if reference_angle is not None:
            dist_to_ref = circular_distance(angle, reference_angle)
            alignment = 1.0 / (1.0 + dist_to_ref * 0.05)
            return base * alignment
        return base

    best = max(valid, key=ranking_key)
    return best[2]


def measure_radial_dark_extent(gray, center, angle_deg, max_radius):
    """Measure how far dark needle pixels extend continuously from center along ray."""
    gx, gy = center
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    perp_cos, perp_sin = -sin_a, cos_a

    radii = np.arange(15, max_radius, 2)
    if len(radii) == 0:
        return 0.0

    xs = np.clip(np.round(gx + radii * cos_a).astype(int), 0, gray.shape[1] - 1)
    ys = np.clip(np.round(gy + radii * sin_a).astype(int), 0, gray.shape[0] - 1)

    needle_vals = gray[ys, xs].astype(float)

    xs_p = np.clip(np.round(gx + radii * cos_a + 8 * perp_cos).astype(int), 0, gray.shape[1] - 1)
    ys_p = np.clip(np.round(gy + radii * sin_a + 8 * perp_sin).astype(int), 0, gray.shape[0] - 1)
    xs_m = np.clip(np.round(gx + radii * cos_a - 8 * perp_cos).astype(int), 0, gray.shape[1] - 1)
    ys_m = np.clip(np.round(gy + radii * sin_a - 8 * perp_sin).astype(int), 0, gray.shape[0] - 1)

    bg_vals = (gray[ys_p, xs_p].astype(float) + gray[ys_m, xs_m].astype(float)) / 2.0
    contrast = bg_vals - needle_vals

    reach = 0.0
    gap = 0
    for r, c in zip(radii, contrast):
        if c >= 12.0:
            reach = float(r)
            gap = 0
        else:
            gap += 1
            if gap > 7:
                break
    return reach


def find_best_radial_extent(gray, center, base_ang, max_radius, fan_deg=10, step=2):
    """Find maximum dark extent and peak angle in a small angular fan."""
    best_reach = 0.0
    best_ang = base_ang
    for offset in range(-fan_deg, fan_deg + 1, step):
        ang = (base_ang + offset) % 360.0
        r = measure_radial_dark_extent(gray, center, ang, max_radius)
        if r > best_reach:
            best_reach = r
            best_ang = ang
    return best_reach, best_ang


def verify_and_resolve_pointer_orientation(frame, center, tip, profile, dial_radius=None):
    """Verify candidate tip orientation against its 180-deg opposite counterweight.

    A physical gauge needle has an asymmetric structure: the true measuring pointer
    extends far outwards (typically 60-85% of dial radius) towards the scale, while
    the counterweight tail terminates at 20-35% radius. If the candidate tip has
    a shorter reach than its opposite ray, or if it points into the dead zone while
    the opposite points into the scale arc, this automatically flips the direction
    to the true long pointer tip.
    """
    if tip is None or center is None or frame is None:
        return tip

    gx, gy = center
    tx, ty = tip
    dx, dy = tx - gx, ty - gy
    curr_dist = math.hypot(dx, dy)
    if curr_dist < 15.0:
        return tip

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h_img, w_img = gray.shape[:2]
    if dial_radius is None or dial_radius <= 20:
        dial_radius = int(min(h_img, w_img) * 0.48)

    curr_angle = math.degrees(math.atan2(dy, dx)) % 360.0
    opp_angle = (curr_angle + 180.0) % 360.0

    reach_curr, best_ang_curr = find_best_radial_extent(gray, center, curr_angle, dial_radius, fan_deg=8, step=2)
    reach_opp, best_ang_opp = find_best_radial_extent(gray, center, opp_angle, dial_radius, fan_deg=12, step=2)

    in_arc_curr = angle_in_scale_arc(curr_angle, profile, margin=15.0)
    in_arc_opp = angle_in_scale_arc(best_ang_opp, profile, margin=15.0)

    should_flip = False

    # 1. Candidate is in the dead zone, but only flip if opposite is distinctly longer (true pointer vs short tail)
    if not in_arc_curr and in_arc_opp and reach_opp > max(40.0, reach_curr * 1.20):
        should_flip = True
    # 2. Opposite has a distinctly longer continuous reach (true tip vs short counterweight tail)
    elif reach_opp > max(40.0, reach_curr * 1.25) and in_arc_opp:
        should_flip = True
    # 3. Candidate terminates short (< 28% radius) while opposite extends deep into scale (>= 45% radius)
    elif reach_curr < dial_radius * 0.28 and reach_opp >= dial_radius * 0.45 and in_arc_opp:
        should_flip = True

    if should_flip:
        new_reach = max(reach_opp, curr_dist)
        rad_opp = math.radians(best_ang_opp)
        new_tx = int(round(gx + new_reach * math.cos(rad_opp)))
        new_ty = int(round(gy + new_reach * math.sin(rad_opp)))
        return (new_tx, new_ty)

    return tip


def refine_gauge_pivot(frame, center, needle_box):
    """Refine a coarse center using a nearby circular hub; abstain if ambiguous."""
    if center is None or needle_box is None:
        return center
    extent = max(needle_box[2:])
    radius = max(12, int(extent * 0.35))
    gx, gy = center
    left, top = max(0, gx - radius), max(0, gy - radius)
    roi = frame[top:min(frame.shape[0], gy + radius + 1),
                left:min(frame.shape[1], gx + radius + 1)]
    if min(roi.shape[:2]) < 12:
        return center
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 12:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bx <= 0 or by <= 0 or bx + bw >= roi.shape[1] or by + bh >= roi.shape[0]:
            continue
        (cx, cy), (w, h), _ = cv2.fitEllipse(contour)
        if min(w, h) < max(6, extent * 0.035) or max(w, h) > extent * 0.28:
            continue
        if min(w, h) / max(w, h) < 0.78:
            continue
        area = cv2.contourArea(contour)
        if area / max(math.pi * w * h / 4, 1) < 0.80:
            continue
        point = (cx + left, cy + top)
        distance = math.dist(point, center)
        if distance > radius * 0.70:
            continue
        if not (needle_box[0] - needle_box[2] / 2 - w / 2 <= point[0] <= needle_box[0] + needle_box[2] / 2 + w / 2
                and needle_box[1] - needle_box[3] / 2 - h / 2 <= point[1] <= needle_box[1] + needle_box[3] / 2 + h / 2):
            continue
        candidates.append((distance, point))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return center
    best = candidates[0][1]
    # Nested edges of the same hub are fine; competing nearby hubs are not.
    if any(math.dist(best, point) > max(4, extent * 0.025)
           and distance < candidates[0][0] + radius * 0.2 for distance, point in candidates[1:]):
        return center
    return tuple(int(round(v)) for v in best)


def refine_needle_centerline(frame, center, tip):
    """Fit dark cross-section midpoints around a detected ray, not its edge.

    The existing detector supplies direction. Reject weak/offset fits rather
    than allowing dial text or a displaced shadow to replace that direction.
    """
    gx, gy = center
    length = math.dist(center, tip)
    if length < 40:
        return tip
    axis = np.array([(tip[0] - gx) / length, (tip[1] - gy) / length])
    normal = np.array([-axis[1], axis[0]])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    half = max(4, min(16, int(length * 0.07)))
    offsets = np.arange(-half, half + 1)
    points = []
    for distance in np.linspace(length * 0.25, length * 0.85, 24):
        base = np.array(center) + distance * axis
        coords = base + offsets[:, None] * normal
        xs, ys = np.rint(coords).astype(int).T
        if (xs.min() < 0 or ys.min() < 0 or xs.max() >= gray.shape[1] or ys.max() >= gray.shape[0]):
            continue
        values = gray[ys, xs].astype(float)
        dark, light = np.percentile(values, [10, 85])
        if light - dark < 35:
            continue
        mask = values < dark + 0.35 * (light - dark)
        indices = np.flatnonzero(mask)
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        runs = [run for run in runs if len(run) >= 2 and run[0] > 0 and run[-1] < len(offsets) - 1]
        if len(runs) != 1:
            continue
        mid = (offsets[runs[0][0]] + offsets[runs[0][-1]]) / 2
        points.append(base + mid * normal)
    if len(points) < 14:
        return tip
    points = np.asarray(points, dtype=np.float32)
    vx, vy, x, y = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([vx, vy], dtype=float)
    if np.dot(direction, axis) < 0:
        direction = -direction
    perp = np.array([-direction[1], direction[0]])
    residuals = np.abs((points - np.array([x, y])) @ perp)
    if np.percentile(residuals, 90) > 2 or abs(np.dot(np.array(center) - [x, y], perp)) > max(2, length * 0.015):
        return tip
    if np.dot(direction, axis) < math.cos(math.radians(6)):
        return tip
    return tuple(float(v) for v in np.array(center) + length * direction)


def detect_pointer_from_boxes(frame, center, predictions, profile, reference_angle=None):
    tip = _detect_pointer_box_edge(frame, center, predictions, profile, reference_angle)
    if tip is not None:
        tip = refine_needle_centerline(frame, center, tip)
    return verify_and_resolve_pointer_orientation(frame, center, tip, profile)


def _detect_pointer_box_edge(frame, center, predictions, profile, reference_angle=None):
    """Extract the needle direction from model boxes instead of using box centers.

    Runs CLAHE-enhanced Hough line detection within each pointer bounding box,
    strictly selecting the longer needle reach over opposite shorter counterweights.

    Args:
        frame: BGR image.
        center: (x, y) gauge center coordinates.
        predictions: List of (confidence, (x, y, w, h)) pointer predictions.
        profile: Gauge profile dict with angle range parameters.
        reference_angle: Optional locked/expected angle in degrees to lock needle direction.

    Returns:
        (x, y) tip coordinates, or None if no suitable line is found.
    """
    gx, gy = center
    gray = _enhance_gray(frame)
    dial_radius = int(min(gray.shape) * 0.48)
    candidates = []

    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
        if max(right - left, bottom - top) < 20 or min(right - left, bottom - top) < 1:
            continue

        # Pad narrow boxes so Hough Lines captures thin vertical/horizontal needles
        if right - left < 24:
            pad_x = (24 - (right - left)) // 2 + 2
            left = max(0, left - pad_x)
            right = min(gray.shape[1], right + pad_x)
        if bottom - top < 24:
            pad_y = (24 - (bottom - top)) // 2 + 2
            top = max(0, top - pad_y)
            bottom = min(gray.shape[0], bottom + pad_y)

        roi = gray[top:bottom, left:right]
        lines = cv2.HoughLinesP(
            _adaptive_canny(roi), 1, np.pi / 180,
            threshold=12, minLineLength=20, maxLineGap=10,
        )
        if lines is None:
            continue

        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            x1, y1, x2, y2 = x1 + left, y1 + top, x2 + left, y2 + top
            length = math.hypot(x2 - x1, y2 - y1)
            dist1 = math.hypot(x1 - gx, y1 - gy)
            dist2 = math.hypot(x2 - gx, y2 - gy)

            # Physical reach determines the pointing end (always the longer one)
            far_point = (x2, y2) if dist1 < dist2 else (x1, y1)
            far_dist = max(dist1, dist2)

            # Needle tip must extend out from the hub, but remain inside the dial face
            if far_dist < 28.0 or far_dist > dial_radius:
                continue

            # Perpendicular distance from gauge center to line segment to reject non-radial lines
            dx = x2 - x1
            dy = y2 - y1
            perp = abs(dy * gx - dx * gy + x2 * y1 - y2 * x1) / max(length, 1.0)
            if perp > 40.0:
                continue

            angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0

            # Sense the actual tip by tracing dark needle pixels extending along the line's ray
            ray_dx = far_point[0] - gx
            ray_dy = far_point[1] - gy
            ray_dist = math.hypot(ray_dx, ray_dy)
            if ray_dist >= 15.0 and roi.size > 0:
                ux, uy = ray_dx / ray_dist, ray_dy / ray_dist
                dark_thresh = max(30, int(np.percentile(roi, 35)))
                y_idxs, x_idxs = np.where(roi <= dark_thresh)
                if len(x_idxs) >= 8:
                    gx_rel = x_idxs + left - gx
                    gy_rel = y_idxs + top - gy
                    projs = gx_rel * ux + gy_rel * uy
                    perps = np.abs(-gx_rel * uy + gy_rel * ux)
                    needle_mask = (perps <= 6.0) & (projs > 15.0)
                    if np.any(needle_mask):
                        max_proj = float(np.percentile(projs[needle_mask], 98))
                        if max_proj > far_dist and max_proj <= dial_radius:
                            far_point = (int(round(gx + ux * max_proj)), int(round(gy + uy * max_proj)))
                            far_dist = max_proj

            in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
            arc_weight = 1.0 if in_arc else 0.65

            # Score prioritizes reach (longer needle extension) and line length
            score = (confidence * (far_dist ** 1.35) * length / (1.0 + perp * 0.15)) * arc_weight
            candidates.append((score, far_dist, far_point, angle, in_arc))

    return _resolve_longer_needle(candidates, reference_angle=reference_angle)


def fit_pointer_pca(frame, center, predictions, profile, reference_angle=None):
    """Fit the pointer direction with PCA on dark pixels inside the Roboflow box.

    Identifies the needle's primary axis and selects the longer extension
    from the hub, rejecting the opposite shorter tail/counterweight.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx, gy = center
    dial_radius = int(min(gray.shape) * 0.48)
    candidates = []

    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
        if max(right - left, bottom - top) < 15 or min(right - left, bottom - top) < 1:
            continue

        if right - left < 16:
            pad_x = (16 - (right - left)) // 2 + 2
            left = max(0, left - pad_x)
            right = min(gray.shape[1], right + pad_x)
        if bottom - top < 16:
            pad_y = (16 - (bottom - top)) // 2 + 2
            top = max(0, top - pad_y)
            bottom = min(gray.shape[0], bottom + pad_y)

        roi = gray[top:bottom, left:right]
        dark_threshold = max(30, int(np.percentile(roi, 35)))
        mask = (roi <= dark_threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        ys, xs = np.where(mask > 0)
        if xs.size < 12:
            continue

        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        mean = points.mean(axis=0)
        centered = points - mean
        if len(centered) <= 1:
            continue
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        principal = eigenvectors[:, order[0]]
        if np.linalg.norm(principal) < 1e-6:
            continue
        principal = principal / np.linalg.norm(principal)
        projected = centered @ principal
        if projected.size == 0:
            continue

        start = mean + principal * projected.min()
        end = mean + principal * projected.max()
        start_global = (int(round(start[0] + left)), int(round(start[1] + top)))
        end_global = (int(round(end[0] + left)), int(round(end[1] + top)))
        dist_start = math.hypot(start_global[0] - gx, start_global[1] - gy)
        dist_end = math.hypot(end_global[0] - gx, end_global[1] - gy)

        # Always select the longer extension from the hub as the needle tip
        if dist_start >= dist_end:
            tip = start_global
            far_dist = dist_start
        else:
            tip = end_global
            far_dist = dist_end

        if far_dist < 28.0 or far_dist > dial_radius:
            continue

        angle = math.degrees(math.atan2(tip[1] - gy, tip[0] - gx)) % 360.0
        in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
        arc_weight = 1.0 if in_arc else 0.70

        score = confidence * (far_dist ** 1.35) * arc_weight
        candidates.append((score, far_dist, tip, angle, in_arc))

    raw_tip = _resolve_longer_needle(candidates, reference_angle=reference_angle)
    return verify_and_resolve_pointer_orientation(frame, center, raw_tip, profile, dial_radius=dial_radius)


def pointer_corner_fallback(frame, center, predictions, profile, reference_angle=None):
    """Backward-compatible alias for the PCA-based pointer fit."""
    return fit_pointer_pca(frame, center, predictions, profile, reference_angle=reference_angle)


def pointer_box_fallback(center, box, frame=None, profile=None):
    """Estimate needle tip directly from the model bounding box and gauge center.

    Guaranteed fallback: If computer vision line detection misses due to glare,
    blur, or lighting, this uses the model's bounding box geometry so the needle
    detector never disappears.
    """
    gx, gy = center
    bx, by, bw, bh = box
    dx, dy = bx - gx, by - gy
    dist = math.hypot(dx, dy)
    if dist < 6.0:
        if bw >= bh:
            raw_tip = (int(round(bx + (bw / 2 if dx >= 0 else -bw / 2))), int(round(by)))
        else:
            raw_tip = (int(round(bx)), int(round(by + (bh / 2 if dy >= 0 else -bh / 2))))
    else:
        dir_x, dir_y = dx / dist, dy / dist
        reach = max(dist + max(bw, bh) * 0.40, max(bw, bh) * 0.65)
        raw_tip = (int(round(gx + dir_x * reach)), int(round(gy + dir_y * reach)))

    if frame is not None and profile is not None:
        return verify_and_resolve_pointer_orientation(frame, center, raw_tip, profile)
    return raw_tip


def detect_pointer_geometry(frame, center, predictions, profile, reference_angle=None):
    """Return the model-box needle line as (hub_candidate, tip_candidate).

    Guarantees the tip candidate is the longer extension from the hub.

    Args:
        frame: BGR image.
        center: (gx, gy) gauge center coordinates.
        predictions: List of (confidence, (x, y, w, h)) pointer predictions.
        profile: Gauge profile dict with angle range parameters.
        reference_angle: Optional locked/expected angle in degrees.

    Returns:
        ((hub_x, hub_y), (tip_x, tip_y)) or None if no suitable line is found.
    """
    gx, gy = center
    gray = _enhance_gray(frame)
    dial_radius = int(min(gray.shape) * 0.48)
    candidates = []

    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
        if max(right - left, bottom - top) < 20 or min(right - left, bottom - top) < 1:
            continue

        if right - left < 24:
            pad_x = (24 - (right - left)) // 2 + 2
            left = max(0, left - pad_x)
            right = min(gray.shape[1], right + pad_x)
        if bottom - top < 24:
            pad_y = (24 - (bottom - top)) // 2 + 2
            top = max(0, top - pad_y)
            bottom = min(gray.shape[0], bottom + pad_y)

        roi = gray[top:bottom, left:right]
        lines = cv2.HoughLinesP(
            _adaptive_canny(roi), 1, np.pi / 180,
            threshold=12, minLineLength=20, maxLineGap=10,
        )
        if lines is None:
            continue
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            x1, y1, x2, y2 = x1 + left, y1 + top, x2 + left, y2 + top
            length = math.hypot(x2 - x1, y2 - y1)
            dist1 = math.hypot(x1 - gx, y1 - gy)
            dist2 = math.hypot(x2 - gx, y2 - gy)

            near_point, far_point = ((x1, y1), (x2, y2)) if dist1 < dist2 else ((x2, y2), (x1, y1))
            far_dist = max(dist1, dist2)

            if far_dist < 28.0 or far_dist > dial_radius:
                continue
            angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0
            in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
            arc_weight = 1.0 if in_arc else 0.70
            score = confidence * (far_dist ** 1.35) * length * arc_weight
            candidates.append((score, far_dist, far_point, angle, in_arc, near_point))

    if not candidates:
        return None

    if reference_angle is not None:
        sector_candidates = [c for c in candidates if circular_distance(c[3], reference_angle) <= 85.0]
        if sector_candidates:
            candidates = sector_candidates

    # Filter out shorter opposite counterweights and stubs
    max_reach = max(c[1] for c in candidates)
    filtered = []
    for cand in candidates:
        score, far_dist, far_point, angle, in_arc, near_point = cand
        is_short = False
        if max_reach >= 40.0 and far_dist < max_reach * 0.65:
            is_short = True
        else:
            for other_score, other_dist, _, other_angle, _, _ in candidates:
                if other_dist > far_dist * 1.15 and circular_distance(angle, other_angle) >= 90.0:
                    is_short = True
                    break
        if not is_short:
            filtered.append(cand)

    valid = filtered if filtered else candidates
    best = max(valid, key=lambda c: c[0] * (c[1] / max(max_reach, 1.0)) ** 1.5)
    verified_tip = verify_and_resolve_pointer_orientation(frame, center, best[2], profile, dial_radius=dial_radius)
    return (best[5], verified_tip) if verified_tip is not None else None


def landmark_tip(predictions, center, profile, reference_angle=None):
    """Use a detected endpoint only when it agrees with the acquired hub and arc."""
    import math
    from measurement import angle_in_scale_arc, circular_distance
    valid = []
    for p in predictions:
        if not hasattr(p, "landmark_tip") or p.confidence < 0.25:
            continue
        length = math.dist(p.landmark_center, p.landmark_tip)
        if math.dist(center, p.landmark_center) > max(15, length*0.15):
            continue
        angle = math.degrees(math.atan2(p.landmark_tip[1]-center[1], p.landmark_tip[0]-center[0])) % 360
        if not angle_in_scale_arc(angle, profile, margin=15):
            continue
        if reference_angle is not None and circular_distance(angle, reference_angle) > 85:
            continue
        valid.append(p)
    return max(valid, key=lambda p:p.confidence).landmark_tip if valid else None

