"""Industrial Gauge Reader — real-time dual-scale pressure gauge reading."""

import os, warnings
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import math
import time
from model_loader import load_models

from config import (
    ROBOFLOW_API_KEY, NEEDLE_MODEL_ID, GAUGE_MODEL_ID,
    MODEL_CONFIDENCE, GAUGE_PROFILES,
)
from camera import open_camera
from dataset_snapshot import save_snapshot
from detection import (
    refine_gauge_pivot,
    detect_gauge_center, detect_pointer_tip, detect_pointer_from_boxes,
    fit_pointer_pca, detect_pointer_geometry, pointer_box_fallback,
)
from measurement import (
    value_from_angle, rolling_median, ema_filter,
)
from polar_reading import polar_unwrap_reading, detect_radial_pointer
from tracking import TrackingState

# ── Model Setup ──────────────────────────────────────────────────────────────

needle_model, gauge_model = load_models()

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
use_polar_mode = not gauge_keys[active_gauge_idx].startswith("UNIJIN")
last_exceed_notify_time = 0.0
snapshot_notice = ""
snapshot_time = 0.0
snapshot_color = (0, 255, 0)

print("Controls:")
print("  'p' -> Toggle Reading Mode (Polar Annulus Unwrap <-> Geometric Angle)")
print("  'g' -> Toggle gauge type (UNIJIN <-> Badotherm)")
print("  's' -> Save raw photo and append blank expected value to labels CSV")
print("  'v' -> Toggle camera (Built-in <-> External)")
print("  'l' -> Lock detector center and needle angle")
print("  'c' -> Unlock center and re-calibrate the hub")
print("  'q' -> Exit")

# ── Main Loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    if 'use_polar_mode' not in locals() and 'use_polar_mode' not in globals():
        use_polar_mode = False
    fresh_measurement = False
    raw_angle = None
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

    predictions = ts.last_predictions
    needle_predictions = ts.last_needle_predictions
    gauge_predictions = ts.last_gauge_predictions
    gauge_center, pointer_tip = None, None
    candidate_center = None
    center_confidence = -1.0
    pointer_predictions = []
    angle_deg = None
    exceeds_limit = False

    # Adaptive confidence threshold: 0.15 initially (picks up real needles on webcams), 0.10 when locked
    pointer_conf_thresh = 0.10 if ts.state in {"TRACKED", "HOLDING"} and ts.locked_center is not None else 0.15
    gauge_conf_thresh = 0.25

    for pred in needle_predictions:
        cls_name = pred.class_name.lower()
        x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
        if (cls_name in {"pointer", "needle"} or "needle" in cls_name) and pred.confidence >= pointer_conf_thresh:
            if max(w, h) >= 15 and min(w, h) >= 1:  # Allow narrow vertical or horizontal needles
                pointer_predictions.append((
                    pred.confidence,
                    (x, y, w, h),
                ))
    # Sort pointer predictions to ALWAYS prioritize the longer needle
    pointer_predictions.sort(
        key=lambda p: max(p[1][2], p[1][3]) * (p[0] ** 0.3),
        reverse=True,
    )

    raw_gauge_center = None
    dial_box = None
    for pred in gauge_predictions:
        cls_lower = pred.class_name.lower()
        if cls_lower in {"pointer", "needle"}:
            continue
        x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)
        # 1. Base / Hub detection (e.g. Badotherm pivot hub)
        if cls_lower in {"base", "hub", "pivot"} and pred.confidence >= 0.25:
            raw_gauge_center = (x, y)
            center_confidence = pred.confidence + 1.0  # Prioritize pivot hub
        # 2. Circle plate / Gauge dial detection (e.g. Badotherm circle plate or generic dial)
        elif cls_lower in {"circle_plate", "gauge", "dial"} or (w >= 50 and h >= 50 and cls_lower != "number"):
            if pred.confidence >= gauge_conf_thresh and pred.confidence > center_confidence:
                raw_gauge_center = (x, y)
                center_confidence = pred.confidence
                dial_box = (x, y, w, h, pred.confidence)

    # Determine the most reliable current gauge center
    known_center = ts.locked_center or (ts.center_history[-1] if ts.center_history else raw_gauge_center)

    # When the needle is oriented vertically in the middle (70-110 deg or 250-290 deg),
    # the neural network has a known blind spot (0 detections).
    # Synthesize the needle box via computer vision Hough lines immediately, so the orange
    # detector never disappears, tracking never drops, and center validation passes:
    if not pointer_predictions and known_center is not None:
        ref_angle = ts.locked_angle if ts.locked_angle is not None else ts.stable_angle
        cv_tip = detect_pointer_tip(frame, known_center, profile, reference_angle=ref_angle)
        if cv_tip is None and ts.last_pointer_tip is not None and ts.tip_miss_frames <= 5:
            cv_tip = ts.last_pointer_tip
        if cv_tip is not None:
            gx, gy = known_center
            tx, ty = cv_tip
            synth_bx = (gx + tx) / 2.0
            synth_by = (gy + ty) / 2.0
            synth_bw = max(24.0, abs(tx - gx) + 18.0)
            synth_bh = max(24.0, abs(ty - gy) + 18.0)
            pointer_predictions.append((0.75, (synth_bx, synth_by, synth_bw, synth_bh)))

    # Sort pointer predictions to ALWAYS prioritize the longer needle
    if pointer_predictions:
        pointer_predictions.sort(
            key=lambda p: max(p[1][2], p[1][3]) * (p[0] ** 0.3),
            reverse=True,
        )

    if new_detection and not pointer_predictions:
        if ts.locked_center is not None:
            ts.state = "HOLDING"
            ts.confidence = max(20, ts.confidence - 8)
        elif not ts.center_history:
            ts.reset(clear_pointer=True)

    # ── Mutual Co-location Validation ──────────────────────────────────────────
    candidate_center = None

    if ts.locked_center is None and new_detection:
        proposed_center = None
        needle_box = pointer_predictions[0][1] if pointer_predictions else None
        if raw_gauge_center is not None:
            proposed_center = raw_gauge_center
        elif pointer_predictions:
            anchor = pointer_predictions[0][1][:2]
            proposed_center = detect_gauge_center(frame, anchor, needle_box=needle_box)
        else:
            # Fallback when needle is vertical/unseen: use center of frame as anchor
            frame_mid = (frame.shape[1] // 2, frame.shape[0] // 2)
            proposed_center = detect_gauge_center(frame, frame_mid)

        proposed_center = refine_gauge_pivot(frame, proposed_center, needle_box)

        # Accept proposed center when co-located with needle or when directly detected by gauge model/circle
        if proposed_center is not None and pointer_predictions:
            ptr_x, ptr_y = pointer_predictions[0][1][:2]
            dist_to_needle = math.hypot(proposed_center[0] - ptr_x, proposed_center[1] - ptr_y)
            if dist_to_needle <= 450.0:
                candidate_center = proposed_center
                ts.center_history.append(proposed_center)
        elif raw_gauge_center is not None:
            candidate_center = raw_gauge_center
            ts.center_history.append(raw_gauge_center)
        elif proposed_center is not None:
            candidate_center = proposed_center
            ts.center_history.append(proposed_center)
        else:
            # Maintain running center if detection dropped for a frame, or use needle anchor as fallback
            if ts.center_history:
                candidate_center = (
                    int(round(np.median([c[0] for c in ts.center_history]))),
                    int(round(np.median([c[1] for c in ts.center_history]))),
                )
            elif pointer_predictions:
                candidate_center = (int(pointer_predictions[0][1][0]), int(pointer_predictions[0][1][1]))

        if len(ts.center_history) >= CENTER_LOCK_FRAMES:
            center_x = np.median([c[0] for c in ts.center_history])
            center_y = np.median([c[1] for c in ts.center_history])
            center_spread = max(
                math.hypot(c[0] - center_x, c[1] - center_y)
                for c in ts.center_history
            )
            if center_spread <= 35.0:
                ts.locked_center = (int(center_x), int(center_y))
            else:
                ts.center_history.popleft()

    if new_detection and ts.locked_center is not None and not (ts.manual_center_lock or ts.manual_detector_lock):
        needle_box = pointer_predictions[0][1] if pointer_predictions else None
        center_anchor = raw_gauge_center or ts.locked_center
        if needle_box is not None:
            bx, by, bw, bh = needle_box
            # Reacquire near the current needle after the gauge moves away.
            if (abs(ts.locked_center[0] - bx) > bw / 2 + 20
                    or abs(ts.locked_center[1] - by) > bh / 2 + 20):
                center_anchor = raw_gauge_center or (int(bx), int(by))
        observed_center = refine_gauge_pivot(frame, center_anchor, needle_box)
        # A model box alone is not evidence of a newly located physical hub.
        if observed_center != center_anchor:
            ts.refresh_center(observed_center)
        else:
            ts.refresh_center(None)

    if ts.locked_center is not None:
        gauge_center = ts.locked_center
    else:
        gauge_center = candidate_center
        if gauge_center is None and ts.center_history:
            gauge_center = ts.center_history[-1]

    # Secondary synthesis if center was just discovered without previous history
    if not pointer_predictions and gauge_center is not None:
        ref_angle = ts.locked_angle if ts.locked_angle is not None else ts.stable_angle
        cv_tip = detect_pointer_tip(frame, gauge_center, profile, reference_angle=ref_angle)
        if cv_tip is not None:
            gx, gy = gauge_center
            tx, ty = cv_tip
            synth_bx = (gx + tx) / 2.0
            synth_by = (gy + ty) / 2.0
            synth_bw = max(24.0, abs(tx - gx) + 18.0)
            synth_bh = max(24.0, abs(ty - gy) + 18.0)
            pointer_predictions.append((0.75, (synth_bx, synth_by, synth_bw, synth_bh)))

    pointer_confidence = 0.0
    for _, box in pointer_predictions[:1]:
        box_x, box_y, box_width, box_height = box
        cv2.rectangle(
            display_frame,
            (int(box_x - box_width / 2), int(box_y - box_height / 2)),
            (int(box_x + box_width / 2), int(box_y + box_height / 2)),
            (0, 165, 255), 2,
        )

    # ── Visualize Whole Gauge Dial (Model Box or Calibrated Dial Perimeter) ──
    if dial_box is not None:
        db_x, db_y, db_w, db_h, db_conf = dial_box
        cv2.rectangle(
            display_frame,
            (int(db_x - db_w / 2), int(db_y - db_h / 2)),
            (int(db_x + db_w / 2), int(db_y + db_h / 2)),
            (255, 200, 0), 2,
        )
        cv2.putText(
            display_frame, f"Dial: {int(db_conf * 100)}%",
            (max(10, int(db_x - db_w / 2)), max(25, int(db_y - db_h / 2) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2,
        )
    elif gauge_center is not None:
        if pointer_tip is not None:
            needle_dist = math.hypot(pointer_tip[0] - gauge_center[0], pointer_tip[1] - gauge_center[1])
            dial_r = max(60, int(needle_dist * 1.15))
        elif pointer_predictions:
            needle_len = max(pointer_predictions[0][1][2], pointer_predictions[0][1][3])
            dial_r = max(60, int(needle_len * 1.25))
        else:
            dial_r = int(min(frame.shape[:2]) * 0.38)

        dial_r = ts.resolve_dial_radius(dial_r)
        cv2.circle(display_frame, gauge_center, dial_r, (255, 200, 0), 2)
        cv2.putText(
            display_frame, f"Dial: {profile_name.split()[0]}",
            (max(10, gauge_center[0] - dial_r), max(25, gauge_center[1] - dial_r - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2,
        )

    if ts.manual_detector_lock:
        cv2.putText(
            display_frame, "DETECTOR: FULLY LOCKED [Press 'C' to unlock]",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
    elif ts.manual_center_lock:
        cv2.putText(
            display_frame, "CENTER: LOCKED [Press 'L' to lock detector]",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
    elif candidate_center is not None and gauge_center is None:
        cv2.circle(display_frame, candidate_center, 8, (0, 165, 255), 2)
        cv2.putText(
            display_frame, f"ACQUIRING CENTER {len(ts.center_history)}/{CENTER_LOCK_FRAMES} [Press 'L' to lock]",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
        )
    elif gauge_center is not None:
        cv2.putText(
            display_frame, "DETECTOR: ACTIVE [Press 'L' to lock detector]",
            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

    if gauge_center:
        if new_detection:
            tip = None
            ref_angle = ts.locked_angle if ts.locked_angle is not None else ts.stable_angle

            # Filter pointer predictions to those near the gauge center (rejects corner artifacts)
            valid_pointer_predictions = []
            for conf, box in pointer_predictions:
                bx, by, bw, bh = box
                dist_to_hub = math.hypot(bx - gauge_center[0], by - gauge_center[1])
                box_half_diag = math.hypot(bw, bh) / 2.0
                if dist_to_hub <= box_half_diag + 60.0:
                    valid_pointer_predictions.append((conf, box))

            # Prioritize the longer needle reaching from the gauge center
            valid_pointer_predictions.sort(
                key=lambda p: max(p[1][2], p[1][3]) * (p[0] ** 0.3),
                reverse=True,
            )

            # Deterministic detection hierarchy:
            face_radius = ts.resolve_dial_radius(
                int(min(frame.shape[:2]) * 0.38) if dial_box is not None else dial_r,
                dial_box)
            from detection import landmark_tip
            tip = landmark_tip(needle_predictions, gauge_center, profile, ref_angle)
            shaft = detect_radial_pointer(frame, gauge_center, face_radius, profile) if tip is None else None
            if shaft is not None:
                tip = shaft["tip"]

            if tip is None and valid_pointer_predictions:
                # 1. Box Hough line
                tip = detect_pointer_from_boxes(frame, gauge_center, valid_pointer_predictions, profile, reference_angle=ref_angle)
                # 2. PCA eigenvector fit on needle pixels
                if tip is None:
                    tip = fit_pointer_pca(frame, gauge_center, valid_pointer_predictions, profile, reference_angle=ref_angle)
                # 3. Needle hub/tip geometry
                if tip is None:
                    geom = detect_pointer_geometry(frame, gauge_center, valid_pointer_predictions, profile, reference_angle=ref_angle)
                    if geom is not None:
                        tip = geom[1]

            # 5. Fallback to full-gauge face Hough lines if box detection did not find tip
            if tip is None:
                tip = detect_pointer_tip(frame, gauge_center, profile, valid_pointer_predictions, reference_angle=ref_angle)

            # A tip must extend beyond the hub and stay within the detected dial.
            if tip is not None:
                radius = math.hypot(tip[0] - gauge_center[0], tip[1] - gauge_center[1])
                if radius < 15 or (dial_box is not None and radius > max(dial_box[2:4]) * 0.6):
                    tip = None

            # Temporal persistence (coast mode):
            # If detection momentarily misses 1-2 frames while tracked, coast on last known tip
            if tip is not None:
                fresh_measurement = True
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
        cv2.putText(display_frame, f"Model classes: {', '.join(detected_classes)}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    elif gauge_center is None:
        cv2.putText(display_frame, "Align gauge in view (seeking dial/needle)...",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif pointer_tip is None:
        cv2.putText(display_frame, "Needle not detected",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    measurement_ready = False
    angle_deg = ts.stable_angle
    if gauge_center and pointer_tip:
        gx, gy = gauge_center
        px, py = pointer_tip

        raw_angle = math.degrees(math.atan2(py - gy, px - gx)) % 360.0

        ts.update_angle(raw_angle, pointer_confidence, fresh=fresh_measurement,
                        lock_frames=ANGLE_LOCK_FRAMES)

        angle_deg = ts.stable_angle if ts.stable_angle is not None else raw_angle
        measurement_ready = ts.stable_angle is not None and len(ts.angle_history) >= ANGLE_LOCK_FRAMES
        stable_radius = max(40, int(math.hypot(px - gx, py - gy)))
        stable_tip = (
            int(round(gx + math.cos(math.radians(angle_deg)) * stable_radius)),
            int(round(gy + math.sin(math.radians(angle_deg)) * stable_radius)),
        )

        cv2.circle(display_frame, (gx, gy), 6, (0, 0, 255), -1)
        # Upright plumb guide (12 o'clock / 270 deg) for visual tilt alignment
        top_guide = (gx, max(0, gy - stable_radius))
        cv2.line(display_frame, (gx, gy), top_guide, (0, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(display_frame, top_guide, 3, (0, 220, 220), -1)
        # Thin magenta = observed direction; blue = filtered direction.
        cv2.line(display_frame, (gx, gy), (int(round(px)), int(round(py))), (255, 0, 255), 1)
        cv2.circle(display_frame, stable_tip, 5, (0, 255, 0), -1)
        cv2.line(display_frame, (gx, gy), stable_tip, (255, 0, 0), 2)

        if angle_deg is not None:
            dial_r = ts.resolve_dial_radius(max(40, int(stable_radius * 1.15)), dial_box)
            polar_fn = globals().get("polar_unwrap_reading")
            is_polar = globals().get("use_polar_mode", False)
            polar_res = (
                polar_fn(frame, (gx, gy), dial_r, profile, reference_angle=angle_deg)
                if (is_polar and polar_fn is not None)
                else None
            )

            if polar_res is not None:
                v1 = polar_res["value_1"]
                v2 = polar_res["value_2"]
                exceeds_limit = polar_res["exceeds_limit"]
                reading_mode_label = "POLAR"
            else:
                v1, v2, exceeds_limit = value_from_angle(angle_deg, profile)
                reading_mode_label = "ANGLE"

            if fresh_measurement and ts.state != "HOLDING":
                if measurement_ready:
                    if not exceeds_limit:
                        ts.history_1.append(v1)
                        ts.history_2.append(v2)
                elif not ts.history_1:
                    ts.history_1.append(v1)
                    ts.history_2.append(v2)

                # Angle tracking already smooths and rejects isolated jumps.
                # Do not add a median + EMA that lags behind the drawn angle.
                ts.ema_v1 = v1
                ts.ema_v2 = v2

            val1_color = (0, 0, 255) if exceeds_limit else (0, 255, 0)
            secondary_invalid = exceeds_limit or v2 > profile["MAX_VAL_2"] + 1e-9
            val2_color = (0, 0, 255) if secondary_invalid else (0, 255, 255)
            age = ts.reading_age()
            held = ts.state == "HOLDING" or (age is not None and age > 1.0)
            held_status = f" (HELD {age:.1f}s)" if held and age is not None else ""
            val_status = (" (INVALID)" if exceeds_limit else "") + held_status
            if held:
                val1_color = (0, 165, 255)
            secondary_status = (" (INVALID)" if secondary_invalid else "") + held_status

            if exceeds_limit:
                cv2.putText(display_frame, "Exceed gauge limit", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            elif measurement_ready:
                cv2.putText(display_frame, f"{profile['UNIT_1']}: {ts.ema_v1:.2f}{val_status} [{reading_mode_label}]", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, val1_color, 3)
                if profile["SHOW_SECONDARY"]:
                    cv2.putText(display_frame, f"{profile['UNIT_2']}: {ts.ema_v2:.2f}{secondary_status}", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, val2_color, 2)
            else:
                # Needle is acquiring stability — show single clean line without text collision
                cv2.putText(display_frame, f"{profile['UNIT_1']}: {ts.ema_v1:.2f} (LOCKING...) [{reading_mode_label}]", (30, 95),
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

        cv2.putText(display_frame, f"Pointer: {angle_deg:.1f} deg", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if new_detection and not fresh_measurement:
        ts.update_angle(None, 0.0)
        ts.lost_frames += 1
        if ts.lost_frames > 20:
            ts.reset(clear_pointer=True)

    # Dynamic positioning so tracking diagnostic text never collides with readings/alerts
    info_y = 180 if exceeds_limit else (220 if profile["SHOW_SECONDARY"] else 180)
    age = ts.reading_age()
    age_label = "no accepted reading" if age is None else f"last accepted {age:.1f}s ago"
    cv2.putText(display_frame, f"Tracking: {ts.state} ({ts.confidence}%) - {age_label}",
                (30, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if ts.state == "TRACKED" else (0, 165, 255), 2)
    cv2.putText(display_frame, f"Model predictions: {len(predictions)}",
                (30, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    cv2.putText(
        display_frame,
        f"Gauge: {len(gauge_predictions)}  Needle: {len(pointer_predictions)}",
        (30, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2,
    )

    if raw_angle is not None and gauge_center and pointer_tip:
        raw_v1, raw_v2, raw_invalid = value_from_angle(raw_angle, profile)
        sample_kind = "fresh" if fresh_measurement else "cached"
        diagnostic = (f"Raw {raw_angle:.1f} deg: Exceed gauge limit"
                      if raw_invalid
                      else f"Raw {raw_angle:.1f} deg: {raw_v1:.2f} {profile['UNIT_1']}")
        if profile["SHOW_SECONDARY"] and not raw_invalid:
            diagnostic += f" / {raw_v2:.2f} {profile['UNIT_2']}"
        diagnostic += f" ({sample_kind}{', INVALID' if raw_invalid else ''})"
        cv2.putText(display_frame, diagnostic, (30, info_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 0, 255), 1)

    mode_str = "POLAR" if use_polar_mode else "ANGLE"
    cv2.putText(display_frame, f"Profile: {profile_name} (Press 'g' to switch, 'p' for mode [{mode_str}])",
                (30, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    controls = "[P] Mode  [S] Snap  [Z] Zero  [ [ / ] ] Nudge  [L] Lock  [C] Unlock  [G] Gauge  [Q] Quit"
    controls_width = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
    controls_scale = 0.55 * min(1.0, max(1, frame.shape[1] - 60) / controls_width)
    cv2.putText(display_frame, controls,
                (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, controls_scale, (200, 200, 200), 2)
    if snapshot_notice and time.time() - snapshot_time < 2.0:
        notice_width = cv2.getTextSize(snapshot_notice, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
        notice_scale = 0.55 * min(1.0, max(1, frame.shape[1] - 60) / notice_width)
        cv2.putText(display_frame, snapshot_notice,
                    (30, frame.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX,
                    notice_scale, snapshot_color, 2)

    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('p'), ord('P')):
        use_polar_mode = not use_polar_mode
        mode_label = "POLAR UNWRAP (Tilt-Immune)" if use_polar_mode else "GEOMETRIC ANGLE"
        snapshot_notice = f"MODE: {mode_label}"
        snapshot_color = (0, 255, 255)
        snapshot_time = time.time()
        print(f"[MODE] Switched reading method to: {mode_label}")
    elif key in (ord('s'), ord('S')):
        try:
            displayed_value = ts.ema_v1 if (measurement_ready and gauge_center and pointer_tip
                                            and not exceeds_limit and ts.state == "TRACKED"
                                            and ts.reading_age() is not None and ts.reading_age() <= 1.0) else None
            image_path = save_snapshot(frame, profile_name, live_predicted=displayed_value,
                                       annotated_frame=display_frame,
                                       diagnostics=dict(raw_angle=raw_angle if fresh_measurement else None,
                                                        fresh_measurement=fresh_measurement,
                                                        filtered_angle=angle_deg, center=gauge_center,
                                                        tip=pointer_tip, profile=dict(profile),
                                                        tracking_state=ts.state,
                                                        reading_age_seconds=ts.reading_age()))
            snapshot_notice = f"SAVED: {image_path} + detector photo"
            snapshot_color = (0, 255, 0)
            print(f"[SNAPSHOT] {snapshot_notice}")
        except (OSError, ValueError, cv2.error) as error:
            snapshot_notice = "Snapshot failed - see console"
            snapshot_color = (0, 0, 255)
            print(f"[SNAPSHOT ERROR] {error}")
        snapshot_time = time.time()
    elif key in (ord('z'), ord('Z')):
        curr_angle = angle_deg if angle_deg is not None else (ts.stable_angle if ts.stable_angle is not None else (raw_angle if 'raw_angle' in locals() else None))
        if curr_angle is not None:
            old_sweep = (profile["MAX_ANGLE"] - profile["MIN_ANGLE"]) % 360.0
            min_val = profile.get("MIN_VAL_1", 0.0)
            max_val = profile.get("MAX_VAL_1", 100.0)
            val_span = max_val - min_val
            zero_ratio = (0.0 - min_val) / val_span if val_span > 0 else 0.0
            zero_offset_deg = zero_ratio * old_sweep
            new_min = (curr_angle - zero_offset_deg) % 360.0
            profile["MIN_ANGLE"] = round(new_min, 2)
            profile["MAX_ANGLE"] = round((new_min + old_sweep) % 360.0, 2)
            from scale_calibration import CALIBRATION_PATH
            import json
            saved = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
            saved[profile_name] = {
                "MIN_ANGLE": profile["MIN_ANGLE"],
                "MAX_ANGLE": profile["MAX_ANGLE"],
                "source": "Live Zero-Tare calibration",
            }
            CALIBRATION_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            snapshot_notice = f"ZERO CALIBRATED: 0-mark at {curr_angle:.1f} deg (Saved)"
            snapshot_color = (0, 255, 0)
            snapshot_time = time.time()
            print(f"[CALIBRATION] Zero mark calibrated to {curr_angle:.2f} deg (MIN_ANGLE={profile['MIN_ANGLE']:.2f}) for {profile_name}. Saved to {CALIBRATION_PATH.name}.")
    elif key in (ord('['), ord('{')):
        # Nudge calibration angle +0.5 deg (decreases reading slightly)
        profile["MIN_ANGLE"] = round((profile["MIN_ANGLE"] + 0.5) % 360.0, 2)
        profile["MAX_ANGLE"] = round((profile["MAX_ANGLE"] + 0.5) % 360.0, 2)
        from scale_calibration import CALIBRATION_PATH
        import json
        saved = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
        saved[profile_name] = {
            "MIN_ANGLE": profile["MIN_ANGLE"],
            "MAX_ANGLE": profile["MAX_ANGLE"],
            "source": "Live nudge calibration",
        }
        CALIBRATION_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        snapshot_notice = f"NUDGE -: Zero now {profile['MIN_ANGLE']:.1f} deg (Saved)"
        snapshot_color = (0, 255, 255)
        snapshot_time = time.time()
        print(f"[CALIBRATION] Nudged {profile_name} MIN_ANGLE to {profile['MIN_ANGLE']:.2f} deg.")
    elif key in (ord(']'), ord('}')):
        # Nudge calibration angle -0.5 deg (increases reading slightly)
        profile["MIN_ANGLE"] = round((profile["MIN_ANGLE"] - 0.5) % 360.0, 2)
        profile["MAX_ANGLE"] = round((profile["MAX_ANGLE"] - 0.5) % 360.0, 2)
        from scale_calibration import CALIBRATION_PATH
        import json
        saved = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
        saved[profile_name] = {
            "MIN_ANGLE": profile["MIN_ANGLE"],
            "MAX_ANGLE": profile["MAX_ANGLE"],
            "source": "Live nudge calibration",
        }
        CALIBRATION_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        snapshot_notice = f"NUDGE +: Zero now {profile['MIN_ANGLE']:.1f} deg (Saved)"
        snapshot_color = (0, 255, 255)
        snapshot_time = time.time()
        print(f"[CALIBRATION] Nudged {profile_name} MIN_ANGLE to {profile['MIN_ANGLE']:.2f} deg.")
    elif key in (ord('v'), ord('V')):
        previous_cam_idx = cam_idx
        cam_idx = (cam_idx + 1) % len(CAMERA_INDICES)
        cap.release()
        cap = open_camera(CAMERA_INDICES[cam_idx])
        if cap is None:
            print(f"Camera {CAMERA_INDICES[cam_idx]} is unavailable; reopening previous camera.")
            cam_idx = previous_cam_idx
            cap = open_camera(CAMERA_INDICES[cam_idx])
        if cap is None:
            print("Could not reopen the previous camera. Closing gauge reader.")
            break
        # Reopening either camera invalidates every cached observation and lock.
        ts = TrackingState()
        last_exceed_notify_time = 0.0
    elif key in (ord('g'), ord('G')):
        active_gauge_idx = (active_gauge_idx + 1) % len(gauge_keys)
        use_polar_mode = not gauge_keys[active_gauge_idx].startswith("UNIJIN")
        ts.reset(clear_lost_frames=True, clear_manual_lock=True)
        ts.locked_center = None
        ts.locked_angle = None
        ts.center_history.clear()
        ts.manual_center_lock = False
        ts.manual_detector_lock = False
        last_exceed_notify_time = 0.0
        print(f"[PROFILE] Switched to {gauge_keys[active_gauge_idx]}. Detector reset for new gauge.")
    elif key in (ord('l'), ord('L')):
        chosen_center = candidate_center if candidate_center is not None else gauge_center
        if chosen_center is None and pointer_predictions:
            chosen_center = pointer_predictions[0][1][:2]
        if chosen_center is not None:
            ts.locked_center = (int(chosen_center[0]), int(chosen_center[1]))
            ts.manual_center_lock = True

        curr_angle = angle_deg if angle_deg is not None else (ts.stable_angle if ts.stable_angle is not None else (raw_angle if 'raw_angle' in locals() else None))
        if curr_angle is not None:
            ts.locked_angle = curr_angle
            ts.stable_angle = curr_angle
            ts.manual_detector_lock = True
            ts.state = "TRACKED"
            ts.confidence = 99
            print(f"[DETECTOR] Fully locked at center {ts.locked_center}, angle {curr_angle:.1f} deg.")
        elif ts.locked_center is not None:
            ts.manual_detector_lock = True
            print(f"[DETECTOR] Center locked at {ts.locked_center}. Needle detector locked.")
    elif key in (ord('c'), ord('C')):
        ts.reset(clear_lost_frames=True, clear_manual_lock=True)
        ts.locked_center = None
        ts.locked_angle = None
        ts.center_history.clear()
        ts.manual_center_lock = False
        ts.manual_detector_lock = False
        print("[DETECTOR] Unlocked. Re-acquiring center and needle...")
    elif key in (ord('q'), ord('Q')):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
