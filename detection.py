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

from measurement import angle_in_scale_arc, angle_on_needle_side

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


def detect_gauge_center(frame, anchor=None):
    """Detect the gauge dial center using Hough circle detection.

    Constrained: Requires an anchor point (e.g. detected needle center) to prevent
    false-positive circle detection on background objects (user's nose, light fixtures, etc.).
    Includes filters to reject overexposed lights/glare and featureless skin/wall patches.

    Args:
        frame: BGR image.
        anchor: Required (x, y) anchor point of the detected pointer to constrain circle search.

    Returns:
        (x, y) center coordinates, or None if no suitable circle is found.
    """
    if anchor is None:
        return None

    scale = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.Canny(gray, 80, 180)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(80 * scale),
        param1=100, param2=50,
        minRadius=int(min(small_frame.shape[:2]) * 0.12),
        maxRadius=int(min(small_frame.shape[:2]) * 0.48),
    )
    if circles is None:
        return None

    image_center = (small_frame.shape[1] / 2, small_frame.shape[0] / 2)
    circles = circles[0]
    small_anchor = (anchor[0] * scale, anchor[1] * scale)

    # 1. Proximity, brightness, and texture verification per circle
    valid_circles = []
    for circle in circles:
        dist_to_anchor = math.hypot(circle[0] - small_anchor[0], circle[1] - small_anchor[1])
        # Needle anchor can be anywhere within the dial radius
        if dist_to_anchor > circle[2] * 0.95:
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
        # Reject only if > 50% of interior pixels are saturated white (> 245)
        if np.mean(roi > 245) > 0.50:
            continue

        # 3. Nose / skin / blank wall filter:
        if np.std(roi) < 9.0:
            continue

        valid_circles.append(circle)

    if not valid_circles:
        return None

    def circle_score(circle):
        center_x, center_y, radius = map(int, circle)
        edge_hits = 0
        samples = 0
        for angle in range(0, 360, 4):
            radians = math.radians(angle)
            for radius_offset in (-1, 0, 1):
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
    if best_score[0] < 0.12:
        return None
    inv_scale = 1.0 / scale
    return int(round(candidate[0] * inv_scale)), int(round(candidate[1] * inv_scale))


def detect_pointer_tip(frame, center, profile, pointer_predictions=None):
    """Find a center-connected line constrained to this gauge's scale arc.

    Uses CLAHE-enhanced edge detection with adaptive thresholds, masked
    to the gauge dial area to eliminate spurious lines from the background.

    Args:
        frame: BGR image.
        center: (x, y) gauge center coordinates.
        profile: Gauge profile dict with angle range parameters.
        pointer_predictions: Optional model predictions (unused, kept for API compat).

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
        near_limit = 55.0 if profile.get("NEEDLE_SIDE_ANGLE") is not None else 45.0
        if near_distance > max(near_limit, length * 0.55):
            continue

        far_distance = max(distance_1, distance_2)
        if far_distance > dial_radius or far_distance < 28.0:
            continue
        angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0
        in_arc = angle_in_scale_arc(angle, profile, margin=35.0)
        on_side = angle_on_needle_side(angle, profile)
        if on_side:
            arc_weight = 1.0 if in_arc else 0.70
            candidates.append(((far_distance ** 1.5) * arc_weight, length, far_point))

    if candidates:
        _, _, best_tip = max(candidates, key=lambda candidate: (candidate[0], candidate[1]))
        return best_tip
    return None


def detect_pointer_from_boxes(frame, center, predictions, profile):
    """Extract the needle direction from model boxes instead of using box centers.

    Runs CLAHE-enhanced Hough line detection within each pointer bounding box
    and scores lines by confidence, length, and proximity to center.

    Args:
        frame: BGR image.
        center: (x, y) gauge center coordinates.
        predictions: List of (confidence, (x, y, w, h)) pointer predictions.
        profile: Gauge profile dict with angle range parameters.

    Returns:
        (x, y) tip coordinates, or None if no suitable line is found.
    """
    gx, gy = center
    gray = _enhance_gray(frame)
    best = None

    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
        if right - left < 20 or bottom - top < 20:
            continue

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
            distance_1 = math.hypot(x1 - gx, y1 - gy)
            distance_2 = math.hypot(x2 - gx, y2 - gy)
            near_distance = min(distance_1, distance_2)
            far_point = (x2, y2) if distance_1 < distance_2 else (x1, y1)
            far_distance = max(distance_1, distance_2)
            dial_radius = int(min(gray.shape) * 0.48)
            # Needle tip must extend out from the hub, but remain inside the dial face
            if far_distance < 28.0 or far_distance > dial_radius:
                continue
            angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0
            if near_distance <= max(55.0, length * 0.60):
                in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
                arc_weight = 1.0 if in_arc else 0.70
                score = (confidence * length / (1.0 + near_distance * 0.15)) * arc_weight
                if best is None or score > best[0]:
                    best = (score, far_point)

    return None if best is None else best[1]


def fit_pointer_pca(frame, center, predictions, profile):
    """Fit the pointer direction with PCA on dark pixels inside the Roboflow box.

    This avoids text or tick marks that can skew corner-based fallback logic.
    We threshold for dark pixels in the needle ROI, run PCA to find the main axis,
    then choose the endpoint farthest from the dial hub as the needle tip.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx, gy = center
    best = None

    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
        if right - left < 10 or bottom - top < 10:
            continue

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
        distances = [
            math.hypot(start_global[0] - gx, start_global[1] - gy),
            math.hypot(end_global[0] - gx, end_global[1] - gy),
        ]
        tip = start_global if distances[0] >= distances[1] else end_global
        far_dist = max(distances)
        dial_radius = int(min(gray.shape) * 0.48)
        # Needle tip must extend out from the hub, but remain inside the dial face
        if far_dist < 28.0 or far_dist > dial_radius:
            continue
        angle = math.degrees(math.atan2(tip[1] - gy, tip[0] - gx)) % 360.0
        in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
        arc_weight = 1.0 if in_arc else 0.75

        # Score is model confidence weighted by scale-arc alignment
        score = confidence * arc_weight
        if best is None or score > best[0]:
            best = (score, tip)

    return None if best is None else best[1]


def pointer_corner_fallback(frame, center, predictions, profile):
    """Backward-compatible alias for the PCA-based pointer fit."""
    return fit_pointer_pca(frame, center, predictions, profile)


def detect_pointer_geometry(frame, center, predictions, profile):
    """Return the model-box needle line as (hub_candidate, tip_candidate).

    Uses CLAHE-enhanced Hough line detection within each pointer bounding box,
    scoring by confidence and length, and returns the best hub-to-tip line segment.

    Args:
        frame: BGR image.
        center: (gx, gy) gauge center coordinates.
        predictions: List of (confidence, (x, y, w, h)) pointer predictions.
        profile: Gauge profile dict with angle range parameters.

    Returns:
        ((hub_x, hub_y), (tip_x, tip_y)) or None if no suitable line is found.
    """
    gx, gy = center
    gray = _enhance_gray(frame)
    best = None
    for confidence, box in predictions:
        box_x, box_y, box_width, box_height = box
        left = max(0, int(box_x - box_width / 2))
        top = max(0, int(box_y - box_height / 2))
        right = min(gray.shape[1], int(box_x + box_width / 2))
        bottom = min(gray.shape[0], int(box_y + box_height / 2))
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
            dial_radius = int(min(gray.shape) * 0.48)
            # Needle tip must reach outward from the hub, but remain inside the dial face
            if far_dist < 28.0 or far_dist > dial_radius:
                continue
            angle = math.degrees(math.atan2(far_point[1] - gy, far_point[0] - gx)) % 360.0
            in_arc = angle_in_scale_arc(angle, profile, margin=25.0)
            arc_weight = 1.0 if in_arc else 0.70
            score = confidence * length * arc_weight
            if best is None or score > best[0]:
                best = (score, near_point, far_point)
    if best is None:
        return None
    return best[1], best[2]
