"""Fit scale angles from independent reference readings; no model training."""

import argparse
import csv
import json
import math
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().with_name("gauge_calibration.json")


def fit_scale(samples, profile):
    """Fit clockwise angle = intercept + slope * primary pressure."""
    if len(samples) < 3:
        raise ValueError("At least three samples at distinct pressure levels are required")
    low, high = profile["MIN_VAL_1"], profile["MAX_VAL_1"]
    sweep = (profile["MAX_ANGLE"] - profile["MIN_ANGLE"]) % 360
    pressures, angles = [], []
    for angle, actual in samples:
        if not math.isfinite(angle) or not math.isfinite(actual) or not low <= actual <= high:
            raise ValueError("Angles must be finite and actual readings must be within the profile range")
        travel = (angle - profile["MIN_ANGLE"]) % 360
        if travel > (sweep + 360) / 2:
            travel -= 360
        pressures.append(actual)
        angles.append(travel)
    if len(set(pressures)) < 3 or max(pressures) - min(pressures) < 0.5 * (high - low):
        raise ValueError("Use at least three pressure levels covering at least half the gauge span")
    mean_p = sum(pressures) / len(pressures)
    mean_a = sum(angles) / len(angles)
    slope = sum((p - mean_p) * (a - mean_a) for p, a in zip(pressures, angles)) / sum((p - mean_p) ** 2 for p in pressures)
    if slope <= 0 or not 0 < slope * (high - low) < 360:
        raise ValueError("Reference points do not describe a valid clockwise scale")
    intercept = mean_a - slope * mean_p
    errors = [abs((a - intercept) / slope - p) for p, a in zip(pressures, angles)]
    if max(errors) > 0.02 * (high - low):
        raise ValueError("Calibration points disagree by more than 2% of span; inspect detection or perspective")
    return {
        "MIN_ANGLE": (profile["MIN_ANGLE"] + intercept + slope * low) % 360,
        "MAX_ANGLE": (profile["MIN_ANGLE"] + intercept + slope * high) % 360,
        "calibration_samples": len(samples),
        "calibration_mae": sum(errors) / len(errors),
        "calibration_max_error": max(errors),
    }


def load_calibration(profiles, path=CALIBRATION_PATH):
    if not path.exists():
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    for name, data in saved.items():
        if name not in profiles:
            raise ValueError(f"Unknown calibrated profile: {name}")
        start, end = data["MIN_ANGLE"], data["MAX_ANGLE"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v < 360 for v in (start, end)) or start == end:
            raise ValueError(f"Invalid calibration angles for {name}")
        profiles[name].update(MIN_ANGLE=start, MAX_ANGLE=end)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="CSV with raw_angle,expected columns")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--unit", choices=["primary", "kgf/cm2"], default="primary")
    parser.add_argument("--save", action="store_true", help="Apply fitted angles on the next app start")
    args = parser.parse_args()
    from config import GAUGE_PROFILES
    matches = [name for name in GAUGE_PROFILES if name.lower().startswith(args.profile.lower())]
    if len(matches) != 1:
        parser.error("Select a unique profile: UNIJIN or Badotherm")
    name = matches[0]
    profile = GAUGE_PROFILES[name]
    try:
        if args.unit == "kgf/cm2" and profile["UNIT_2"] != "kgf/cm2":
            raise ValueError("kgf/cm2 input is supported only for UNIJIN")
        factor = profile["SECONDARY_PER_PRIMARY"] if args.unit == "kgf/cm2" else 1.0
        with args.samples.open(encoding="utf-8-sig", newline="") as stream:
            samples = [(float(row["raw_angle"]), float(row["expected"]) / factor) for row in csv.DictReader(stream)]
        result = fit_scale(samples, profile)
        print(json.dumps(result, indent=2))
        print("Fit errors are calibration errors, not independent test accuracy.")
        if args.save:
            saved = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else {}
            saved[name] = result
            temporary = CALIBRATION_PATH.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            temporary.replace(CALIBRATION_PATH)
            print("Saved. Restart the reader. Verify on separate pressures with the camera fixed.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
