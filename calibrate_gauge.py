"""Gauge Angle Calibrator — live angle readout for gauge profile calibration."""

import os
import warnings

os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
warnings.filterwarnings("ignore")

import cv2
import math
from inference import get_model

from config import ROBOFLOW_API_KEY, NEEDLE_MODEL_ID
from camera import open_camera

# 1. Model Setup
print("Loading model for gauge calibration...")
model = get_model(model_id=NEEDLE_MODEL_ID, api_key=ROBOFLOW_API_KEY)

# 2. Camera Setup
camera_indices = [0, 1]
cam_idx = 0  # 0: Built-in, 1: External USB

cap = open_camera(camera_indices[cam_idx])
if cap is None:
    raise RuntimeError(
        f"Could not open camera {camera_indices[cam_idx]}. "
        "Check Windows camera permissions or connect the camera."
    )

window_name = "Gauge Angle Calibrator"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1024, 768)

calib_min_angle = None
calib_max_angle = None

print("\nCalibrator Controls:")
print("  Hold gauge upright facing camera squarely.")
print("  Press '0' -> Record current angle as MIN_ANGLE (Zero mark)")
print("  Press 'm' -> Record current angle as MAX_ANGLE (Full scale)")
print("  Press 'p' -> Print ready-to-use profile configuration snippet")
print("  Press 'r' -> Reset recorded angles")
print("  Press 's' -> Switch camera (Built-in <-> External)")
print("  Press 'q' or click 'X' -> Exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not available.")
        break

    results = model.infer(frame, confidence=0.20)[0]
    gauge_center = None
    pointer_pos = None
    angle_deg = None

    for pred in results.predictions:
        cls_name = pred.class_name.lower()
        x, y, w, h = int(pred.x), int(pred.y), int(pred.width), int(pred.height)

        # Draw light bounding box for visual confirmation
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        if cls_name in {"pressure-gauge-objects", "gauge", "dial", "meter", "center"}:
            gauge_center = (x, y)
        elif cls_name in {"pointer", "needle"}:
            pointer_pos = (x, y)

    if not gauge_center or not pointer_pos:
        cv2.putText(frame, "Align gauge in view (seeking dial + pointer)...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        gx, gy = gauge_center
        px, py = pointer_pos

        # Draw markers & ray
        cv2.circle(frame, (gx, gy), 6, (0, 0, 255), -1)      # Dial Center (Red)
        cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)      # Pointer (Green)
        cv2.line(frame, (gx, gy), (px, py), (255, 255, 0), 2) # Ray (Cyan/Yellow)

        # Compute standard clockwise angle from 3 o'clock position (0° to 360°)
        dx = px - gx
        dy = py - gy
        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg < 0:
            angle_deg += 360.0

        # Calibration readouts
        cv2.putText(frame, f"CURRENT ANGLE: {angle_deg:.1f} deg", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Center: ({gx}, {gy}) | Pointer: ({px}, {py})", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Recorded calibration status
        min_str = f"{calib_min_angle:.1f} deg [SAVED]" if calib_min_angle is not None else "Not set (Press '0')"
        max_str = f"{calib_max_angle:.1f} deg [SAVED]" if calib_max_angle is not None else "Not set (Press 'm')"
        cv2.putText(frame, f"[0-Mark] MIN_ANGLE: {min_str}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if calib_min_angle else (180, 180, 180), 2)
        cv2.putText(frame, f"[Full]   MAX_ANGLE: {max_str}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if calib_max_angle else (180, 180, 180), 2)

        if calib_min_angle is not None and calib_max_angle is not None:
            sweep = (calib_max_angle - calib_min_angle) % 360.0
            cv2.putText(frame, f"Active Sweep: {sweep:.1f} deg — Press 'p' to print profile snippet!", (30, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Move needle to 0 -> Press '0' | Move to Full Scale -> Press 'm'", (30, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 160), 1)

    cv2.putText(frame, f"Camera: {camera_indices[cam_idx]} (Press 's' to switch)",
                (30, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('0') and angle_deg is not None:
        calib_min_angle = round(angle_deg, 1)
        print(f"\n[CALIBRATION] MIN_ANGLE (Zero mark) recorded: {calib_min_angle} deg")
    elif key == ord('m') and angle_deg is not None:
        calib_max_angle = round(angle_deg, 1)
        print(f"\n[CALIBRATION] MAX_ANGLE (Full scale) recorded: {calib_max_angle} deg")
    elif key == ord('r'):
        calib_min_angle = None
        calib_max_angle = None
        print("\n[CALIBRATION] Recorded calibration angles reset.")
    elif key == ord('p'):
        min_val = calib_min_angle if calib_min_angle is not None else 0.0
        max_val = calib_max_angle if calib_max_angle is not None else 0.0
        sweep_deg = (max_val - min_val) % 360.0
        print("\n" + "=" * 62)
        print("          GENERATED GAUGE PROFILE CONFIGURATION")
        print("=" * 62)
        print(f'''    "Custom Gauge (Edit Units)": {{
        "MIN_ANGLE": {min_val:.1f},       # Zero mark
        "MAX_ANGLE": {max_val:.1f},       # Full scale mark (Active sweep: {sweep_deg:.1f} deg)
        "MIN_VAL_1": 0.0, "MAX_VAL_1": 100.0, "UNIT_1": "PSI",
        "MIN_VAL_2": 0.0, "MAX_VAL_2": 7.0,   "UNIT_2": "bar",
        "SHOW_SECONDARY": True,
    }},''')
        print("=" * 62)
        print("Copy and paste the above dictionary block into GAUGE_PROFILES in config.py!\n")
    elif key == ord('s'):
        cam_idx = (cam_idx + 1) % len(camera_indices)
        cap.release()
        next_cap = open_camera(camera_indices[cam_idx])
        if next_cap is None:
            print(f"Camera {camera_indices[cam_idx]} is unavailable; keeping current camera.")
            cam_idx = (cam_idx - 1) % len(camera_indices)
            cap = open_camera(camera_indices[cam_idx])
        else:
            cap = next_cap
    elif key == ord('q'):
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()