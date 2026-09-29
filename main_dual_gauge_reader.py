"""Industrial Gauge Reader — real-time dual-scale pressure gauge reading."""

import os, warnings
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import math
import time
from inference import get_model

from config import (
    ROBOFLOW_API_KEY, NEEDLE_MODEL_ID, GAUGE_MODEL_ID,
    MODEL_CONFIDENCE, GAUGE_PROFILES,
)
from camera import open_camera
from detection import (
    detect_gauge_center, detect_pointer_tip, detect_pointer_from_boxes,
    fit_pointer_pca, detect_pointer_geometry,
)
from measurement import (
    value_from_angle, circular_distance, signed_angle_delta, circular_mean,
    weighted_circular_mean, rolling_median, ema_filter,
)
from tracking import TrackingState

# ── Model Setup ──────────────────────────────────────────────────────────────

print("Loading needle model...")
needle_model = get_model(model_id=NEEDLE_MODEL_ID, api_key=ROBOFLOW_API_KEY)
print("Loading gauge model...")
try:
    gauge_model = get_model(model_id=GAUGE_MODEL_ID, api_key=ROBOFLOW_API_KEY)
except Exception as error:
    gauge_model = None
    print(f"Gauge model unavailable ({GAUGE_MODEL_ID}): {error}")
    print("Use the deployed Roboflow model ID, normally project-name/version.")

# ── State Initialization ─────────────────────────────────────────────────────

CAMERA_INDICES = [0, 1]
gauge_keys = list(GAUGE_PROFILES.keys())
active_gauge_idx = 0
cam_idx = 0

CENTER_LOCK_FRAMES = 2
ANGLE_LOCK_FRAMES = 2
DETECTION_INTERVAL_FAST = 2   # frames between detections while searching/locking
DETECTION_INTERVAL_SLOW = 4   # frames between detections while stably tracked

cap = open_camera(CAMERA_INDICES[cam_idx])
if cap is None:
    raise RuntimeError(
        f"Could not open camera {CAMERA_INDICES[cam_idx]}. "
        "Check Windows camera permissions or connect the camera."
    )

window_name = "Industrial Gauge Reader"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1024, 768)

ts = TrackingState()
last_exceed_notify_time = 0.0

print("Controls:")
print("  'g' -> Toggle gauge type (UNIJIN <-> Badotherm)")
print("  's' -> Toggle camera (Built-in <-> External)")
print("  'c' -> Unlock center and re-calibrate the hub")
print("  'q' -> Exit")

# ── Main Loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ts.frame_number += 1
    profile_name = gauge_keys[active_gauge_idx]
    profile = GAUGE_PROFILES[profile_name]

    # Adaptive detection interval: fast when searching, slow when stable
    effective_interval = (
        DETECTION_INTERVAL_SLOW if ts.state == "TRACKED" and ts.confidence >= 70
        else DETECTION_INTERVAL_FAST
    )
    new_detection = ts.frame_number == 1 or ts.frame_number % effective_interval == 0
    if new_detection:
        ts.last_needle_predictions = needle_model.infer(
            frame, confidence=MODEL_CONFIDENCE
        )[0].predictions
        ts.last_gauge_predictions = []
        if gauge_model is not None:
            ts.last_gauge_predictions = gauge_model.infer(
                frame, confidence=MODEL_CONFIDENCE
            )[0].predictions
        ts.last_predictions = list(ts.last_needle_predictions) + list(ts.last_gauge_predictions)
        ts.last_pointer_tip = None

    predictions = ts.last_predictions
    needle_predictions = ts.last_needle_predictions
    gauge_predictions = ts.last_gauge_predictions
    gauge_center, pointer_tip = None, None
    candidate_center = None
    center_confidence = -1.0
    pointer_predictions = []
    angle_deg = None
    exceeds_limit = False

    # Adaptive confidence threshold: 0.28 initially, 0.18 when locked
    pointer_conf_thresh = 0.18 if ts.state in {"TRACKED", "HOLDING"} and ts.locked_center is not None else 0.28
    gauge_conf_thresh = 0.30

    for pred in needle_predictions:
        cls_name = pred.class_name.lower()
        x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
        if (cls_name in {"pointer", "needle"} or "needle" in cls_name) and pred.confidence >= pointer_conf_thresh:
            if max(w, h) >= 15 and min(w, h) >= 3:  # Plausible needle dimensions
                pointer_predictions.append((
                    pred.confidence,
                    (x, y, w, h),
                ))

    pointer_predictions.sort(key=lambda p: p[0], reverse=True)

    raw_gauge_center = None
    for pred in gauge_predictions:
        x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
        if pred.confidence >= gauge_conf_thresh and pred.confidence > center_confidence:
            if w >= 50 and h >= 50:  # Plausible dial dimensions
                raw_gauge_center = (x, y)
                center_confidence = pred.confidence

    if new_detection and not pointer_predictions:
        if ts.locked_center is not None:
            ts.state = "HOLDING"
            ts.confidence = max(20, ts.confidence - 8)
        else:
            ts.reset(clear_pointer=True)

    # ── Mutual Co-location Validation ──────────────────────────────────────────
    candidate_center = None

    if ts.locked_center is None:
        proposed_center = None
        if raw_gauge_center is not None:
            proposed_center = raw_gauge_center
        elif pointer_predictions:
            anchor = pointer_predictions[0][1][:2]
            proposed_center = detect_gauge_center(frame, anchor)

        # Accept proposed center when co-located with needle or when directly detected by gauge model
        if proposed_center is not None and pointer_predictions:
            ptr_x, ptr_y = pointer_predictions[0][1][:2]
            dist_to_needle = math.hypot(proposed_center[0] - ptr_x, proposed_center[1] - ptr_y)
            if 8.0 <= dist_to_needle <= 450.0:
                candidate_center = proposed_center
                ts.center_history.append(proposed_center)
            else:
                candidate_center = None
                ts.center_history.clear()
        elif raw_gauge_center is not None:
            candidate_center = raw_gauge_center
            ts.center_history.append(raw_gauge_center)
        else:
            candidate_center = None
            ts.center_history.clear()

        if len(ts.center_history) >= CENTER_LOCK_FRAMES:
            center_x = np.median([c[0] for c in ts.center_history])
            center_y = np.median([c[1] for c in ts.center_history])
            center_spread = max(
                math.hypot(c[0] - center_x, c[1] - center_y)
                for c in ts.center_history
            )
            if center_spread <= 20.0:
                ts.locked_center = (int(center_x), int(center_y))
            else:
                ts.center_history.clear()

    if ts.locked_center is not None:
        gauge_center = ts.locked_center
    else:
        gauge_center = candidate_center

    pointer_confidence = 0.0
    for _, box in pointer_predictions[:1]:
        box_x, box_y, box_width, box_height = box
        cv2.rectangle(
            frame,
            (int(box_x - box_width / 2), int(box_y - box_height / 2)),
            (int(box_x + box_width / 2), int(box_y + box_height / 2)),
            (0, 165, 255), 2,
        )

    if candidate_center is not None and gauge_center is None:
        cv2.circle(frame, candidate_center, 8, (0, 165, 255), 2)
        cv2.putText(
            frame, f"ACQUIRING CENTER {len(ts.center_history)}/{CENTER_LOCK_FRAMES}",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
        )

    if gauge_center:
        if new_detection:
            tip = None

            # Deterministic detection hierarchy (no 0.50 threshold oscillation):
            # When needle box is detected, prioritize box-focused detection (fast, consistent ROI)
            if pointer_predictions:
                # 1. Box Hough line
                tip = detect_pointer_from_boxes(frame, gauge_center, pointer_predictions, profile)
                # 2. PCA eigenvector fit on needle pixels
                if tip is None:
                    tip = fit_pointer_pca(frame, gauge_center, pointer_predictions, profile)
                # 3. Needle hub/tip geometry
                if tip is None:
                    geom = detect_pointer_geometry(frame, gauge_center, pointer_predictions, profile)
                    if geom is not None:
                        tip = geom[1]

            # 4. Fallback to full-gauge face Hough lines if box detection did not find tip
            if tip is None:
                tip = detect_pointer_tip(frame, gauge_center, profile, pointer_predictions)

            # Temporal persistence (coast mode):
            # If detection momentarily misses 1-2 frames while tracked, coast on last known tip
            if tip is not None:
                ts.last_pointer_tip = tip
                ts.tip_miss_frames = 0
            else:
                ts.tip_miss_frames += 1
                if ts.tip_miss_frames <= 3 and ts.last_pointer_tip is not None and ts.state in {"TRACKED", "HOLDING"}:
                    tip = ts.last_pointer_tip
                else:
                    ts.last_pointer_tip = None

        pointer_tip = ts.last_pointer_tip
        pointer_confidence = 0.50 if pointer_tip is not None else 0.0

    if predictions and not gauge_center:
        detected_classes = sorted({pred.class_name for pred in predictions})
        cv2.putText(frame, f"Model classes: {', '.join(detected_classes)}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    elif gauge_center is None:
        cv2.putText(frame, "Align gauge in view (seeking dial/needle)...",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif pointer_tip is None:
        cv2.putText(frame, "Needle not detected",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    measurement_ready = False
    if gauge_center and pointer_tip:
        gx, gy = gauge_center
        px, py = pointer_tip

        raw_angle = math.degrees(math.atan2(py - gy, px - gx)) % 360.0
        ts.lost_frames = 0
        if ts.stable_angle is None:
            ts.angle_history.append(raw_angle)
            ts.state = "LOCKING"
            ts.confidence = min(95, len(ts.angle_history) * 30)
            if len(ts.angle_history) >= ANGLE_LOCK_FRAMES and max(
                circular_distance(raw_angle, a) for a in ts.angle_history
            ) <= 10.0:
                ts.stable_angle = weighted_circular_mean(ts.angle_history)
                ts.state = "TRACKED"
                ts.confidence = min(99, int(pointer_confidence * 100))
        else:
            delta = signed_angle_delta(ts.stable_angle, raw_angle)
            abs_delta = abs(delta)

            if abs_delta <= 12.0:
                # Normal minor tracking noise: smooth low-pass update
                ts.consecutive_move_frames = 0
                ts.angle_history.append(raw_angle)
                target_angle = circular_mean(ts.angle_history)
                ts.stable_angle = (
                    ts.stable_angle + 0.10 * signed_angle_delta(ts.stable_angle, target_angle)
                ) % 360.0
                ts.state = "TRACKED"
                ts.confidence = min(99, max(60, int(pointer_confidence * 100)))
            else:
                # Large angular difference: real needle rotation vs single-frame outlier
                ts.consecutive_move_frames += 1
                if ts.consecutive_move_frames >= 2:
                    # 2 consecutive frames agree -> genuine needle movement!
                    # Fast-follow without getting stuck in HOLDING
                    ts.stable_angle = raw_angle
                    ts.angle_history.clear()
                    ts.angle_history.append(raw_angle)
                    ts.consecutive_move_frames = 0
                    ts.state = "TRACKED"
                    ts.confidence = min(99, max(60, int(pointer_confidence * 100)))
                else:
                    # Single-frame outlier: gently hold previous angle
                    ts.state = "HOLDING"
                    ts.confidence = max(40, ts.confidence - 5)

        angle_deg = ts.stable_angle if ts.stable_angle is not None else raw_angle
        measurement_ready = ts.stable_angle is not None and len(ts.angle_history) >= ANGLE_LOCK_FRAMES
        stable_radius = max(40, int(math.hypot(px - gx, py - gy)))
        stable_tip = (
            int(round(gx + math.cos(math.radians(angle_deg)) * stable_radius)),
            int(round(gy + math.sin(math.radians(angle_deg)) * stable_radius)),
        )

        cv2.circle(frame, (gx, gy), 6, (0, 0, 255), -1)
        cv2.circle(frame, stable_tip, 5, (0, 255, 0), -1)
        cv2.line(frame, (gx, gy), stable_tip, (255, 0, 0), 2)

        if angle_deg is not None:
            v1, v2, exceeds_limit = value_from_angle(angle_deg, profile)

            if measurement_ready:
                if not exceeds_limit:
                    ts.history_1.append(v1)
                    ts.history_2.append(v2)
            elif not ts.history_1:
                ts.history_1.append(v1)
                ts.history_2.append(v2)

            median_v1 = rolling_median(ts.history_1, window=min(7, len(ts.history_1))) if ts.history_1 else v1
            median_v2 = rolling_median(ts.history_2, window=min(7, len(ts.history_2))) if ts.history_2 else v2
            ts.ema_v1 = ema_filter(ts.ema_v1, median_v1)
            ts.ema_v2 = ema_filter(ts.ema_v2, median_v2)

            val1_color = (0, 0, 255) if exceeds_limit else (0, 255, 0)
            val2_color = (0, 0, 255) if exceeds_limit else (0, 255, 255)
            val_status = " (INVALID)" if exceeds_limit else ""

            if measurement_ready:
                cv2.putText(frame, f"{profile['UNIT_1']}: {ts.ema_v1:.2f}{val_status}", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, val1_color, 3)
                if profile["SHOW_SECONDARY"]:
                    cv2.putText(frame, f"{profile['UNIT_2']}: {ts.ema_v2:.2f}{val_status}", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, val2_color, 2)
            else:
                # Needle is acquiring stability — show single clean line without text collision
                cv2.putText(frame, f"{profile['UNIT_1']}: {ts.ema_v1:.2f} (LOCKING...)", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            if exceeds_limit:
                current_time = time.time()
                if current_time - last_exceed_notify_time > 2.0:
                    print(f"[NOTIFICATION] Exceed the gauge limit! ({profile_name}: Pointer at {angle_deg:.1f} deg is out of bounds)")
                    try:
                        import winsound
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    except Exception:
                        pass
                    last_exceed_notify_time = current_time

                # Draw high-visibility warning badge on frame
                notify_text = "Exceed the gauge limit"
                notify_y = 180 if profile["SHOW_SECONDARY"] else 140
                (nw, nh), _ = cv2.getTextSize(notify_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (26, notify_y - nh - 8), (34 + nw, notify_y + 8), (0, 0, 180), -1)
                cv2.rectangle(frame, (26, notify_y - nh - 8), (34 + nw, notify_y + 8), (0, 0, 255), 2)
                cv2.putText(frame, notify_text, (30, notify_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Pointer: {angle_deg:.1f} deg", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if not (gauge_center and pointer_tip):
        ts.lost_frames += 1
        if ts.lost_frames > 20:
            ts.reset()

    # Dynamic positioning so tracking diagnostic text never collides with readings/alerts
    info_y = 220 if (profile["SHOW_SECONDARY"] or (angle_deg is not None and exceeds_limit)) else 180
    cv2.putText(frame, f"Tracking: {ts.state} ({ts.confidence}%)",
                (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if ts.state == "TRACKED" else (0, 165, 255), 2)
    cv2.putText(frame, f"Model predictions: {len(predictions)}",
                (30, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(
        frame,
        f"Gauge: {len(gauge_predictions)}  Needle: {len(pointer_predictions)}",
        (30, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2,
    )

    # Status line
    cv2.putText(frame, f"Profile: {profile_name} (Press 'g' to change)",
                (30, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Camera: {CAMERA_INDICES[cam_idx]} (Press 's' to switch)",
                (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cam_idx = (cam_idx + 1) % len(CAMERA_INDICES)
        cap.release()
        next_cap = open_camera(CAMERA_INDICES[cam_idx])
        if next_cap is None:
            print(f"Camera {CAMERA_INDICES[cam_idx]} is unavailable; keeping current camera.")
            cam_idx = (cam_idx - 1) % len(CAMERA_INDICES)
            cap = open_camera(CAMERA_INDICES[cam_idx])
        else:
            cap = next_cap
    elif key == ord('g'):
        active_gauge_idx = (active_gauge_idx + 1) % len(gauge_keys)
        ts.reset(clear_lost_frames=True)
        last_exceed_notify_time = 0.0
    elif key == ord('c'):
        ts.reset(clear_lost_frames=True)
        ts.locked_center = None
        ts.center_history.clear()
    elif key == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()