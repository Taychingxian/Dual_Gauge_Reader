#Offline photo accuracy evaluation. Does not import or run the live reader.

#python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN --live
#python evaluate_accuracy.py accuracy_data/badotherm_labels.csv --profile Badotherm --live

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path


def load_labels(path, profile):
    """Validate labels before model loading. Paths are relative to the CSV."""
    samples = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image", "expected"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have image,expected columns")
        for line, row in enumerate(reader, 2):
            name = (row.get("image") or "").strip()
            try:
                expected = float(row.get("expected", ""))
            except (TypeError, ValueError):
                raise ValueError(f"Line {line}: expected must be a number") from None
            if not name or not math.isfinite(expected):
                raise ValueError(f"Line {line}: image and finite expected value required")
            if not profile["MIN_VAL_1"] <= expected <= profile["MAX_VAL_1"]:
                raise ValueError(f"Line {line}: expected is outside the profile range")
            samples.append((name, (path.parent / name).resolve(), expected))
    if not samples:
        raise ValueError("CSV contains no labeled images")
    return samples


def detect_image(frame, needle_model, gauge_model, profile, confidence, use_polar=False):
    """Use the live reader's initial-acquisition thresholds and fallback order."""
    from detection import (
        refine_gauge_pivot,
        detect_gauge_center, detect_pointer_from_boxes, fit_pointer_pca,
        detect_pointer_geometry, pointer_box_fallback, detect_pointer_tip,
    )
    from measurement import value_from_angle

    needles = needle_model.infer(frame, confidence=confidence)[0].predictions
    gauges = gauge_model.infer(frame, confidence=confidence)[0].predictions if gauge_model else []
    pointers = []
    for pred in needles:
        cls = pred.class_name.lower()
        box = tuple(int(v) for v in (pred.x, pred.y, pred.width, pred.height))
        if (cls in {"pointer", "needle"} or "needle" in cls) and pred.confidence >= 0.15:
            if max(box[2:]) >= 15 and min(box[2:]) >= 3:
                pointers.append((pred.confidence, box))
    pointers.sort(key=lambda item: max(item[1][2], item[1][3]) * (item[0] ** 0.3), reverse=True)
    hubs = [p for p in gauges if p.class_name.lower() in {"base", "hub", "pivot"} and p.confidence >= 0.25]
    candidates = [p for p in gauges if p.class_name.lower() in {"gauge", "circle_plate", "dial"}
                  and p.confidence >= 0.25 and int(p.width) >= 50 and int(p.height) >= 50]
    center = None
    if hubs:
        best = max(hubs, key=lambda p: p.confidence)
        center = (int(best.x), int(best.y))
    elif candidates:
        best = max(candidates, key=lambda p: p.confidence)
        center = (int(best.x), int(best.y))
    elif pointers:
        center = detect_gauge_center(frame, pointers[0][1][:2], needle_box=pointers[0][1])
    if center is not None and pointers:
        if math.dist(center, pointers[0][1][:2]) > 450.0:
            center = None
    elif center is None and pointers:
        center = pointers[0][1][:2]
    if center is None:
        return {"status": "no_center"}, None, None

    center = refine_gauge_pivot(frame, center, pointers[0][1] if pointers else None)
    valid = [p for p in pointers if math.dist(p[1][:2], center) <= math.hypot(*p[1][2:]) / 2 + 60]
    valid.sort(key=lambda item: max(item[1][2], item[1][3]) * (item[0] ** 0.3), reverse=True)
    if not valid:
        valid = pointers[:1]
    from detection import landmark_tip
    tip = landmark_tip(needles, center, profile)
    method = "obb_landmarks" if tip is not None else ""
    if valid and tip is None:
        for method, detector in (("box_hough", detect_pointer_from_boxes), ("pca", fit_pointer_pca)):
            tip = detector(frame, center, valid, profile, reference_angle=None)
            if tip is not None:
                break
        if tip is None:
            geometry = detect_pointer_geometry(frame, center, valid, profile, reference_angle=None)
            if geometry is not None:
                tip, method = geometry[1], "geometry"
        if tip is None:
            tip, method = pointer_box_fallback(center, valid[0][1]), "box_fallback"
    if tip is None:
        tip = detect_pointer_tip(frame, center, profile, pointers, reference_angle=None)
        method = "face_hough"
    if tip is None or tip == center:
        return {"status": "no_tip"}, center, None
    angle = math.degrees(math.atan2(tip[1] - center[1], tip[0] - center[0])) % 360

    if use_polar:
        from polar_reading import polar_unwrap_reading
        radius = math.hypot(tip[0] - center[0], tip[1] - center[1]) * 1.15
        polar_res = polar_unwrap_reading(frame, center, radius, profile, reference_angle=angle)
        if polar_res is not None:
            return dict(status="out_of_range" if polar_res["exceeds_limit"] else "ok",
                        predicted=polar_res["value_1"],
                        angle=polar_res["needle_angle"],
                        method="polar_unwrap"), center, tip

    value, _, outside = value_from_angle(angle, profile)
    return dict(status="out_of_range" if outside else "ok", predicted=value,
                angle=angle, method=method), center, tip


def summarize(rows, tolerance):
    valid = [r for r in rows if r["status"] == "ok"]
    errors = [r["absolute_error"] for r in valid]
    signed_errors = [r["predicted"] - r["expected"] for r in valid
                     if "predicted" in r and "expected" in r]
    passed = sum(error <= tolerance for error in errors)
    return dict(total=len(rows), valid=len(valid), failed=len(rows) - len(valid),
                valid_reading_rate_percent=100 * len(valid) / len(rows),
                within_tolerance_rate_percent=100 * passed / len(rows),
                tolerance=tolerance,
                mean_absolute_error=sum(errors) / len(errors) if errors else None,
                root_mean_squared_error=math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
                mean_signed_error=sum(signed_errors) / len(signed_errors) if signed_errors else None,
                worst_absolute_error=max(errors) if errors else None)


def explain_summary(summary):
    """Put readable results first while retaining numeric fields for tooling."""
    unit = summary["unit"]
    total, valid = summary["total"], summary["valid"]
    passed = round(summary["within_tolerance_rate_percent"] * total / 100)
    average = summary["mean_absolute_error"]
    worst = summary["worst_absolute_error"]
    explanation = {
        "gauge": summary["profile"],
        "test": summary["evaluation"],
        "result": f"{passed} of {total} readings passed ({summary['within_tolerance_rate_percent']:.1f}%).",
        "pass_rule": f"Computer reading must be within {summary['tolerance']:g} {unit} of your actual reading (including the boundary).",
        "passed": f"{passed} readings were within the allowed error.",
        "outside_allowed_error": f"{valid - passed} readings were too far from your actual reading.",
        "could_not_be_checked": f"{total - valid} readings were missing, invalid, or could not be processed.",
        "average_error": f"{average:.2f} {unit} away from the actual reading." if average is not None else "Unavailable: no usable readings.",
        "biggest_error": f"{worst:.2f} {unit} away from the actual reading." if worst is not None else "Unavailable: no usable readings.",
        "note": "Average and biggest errors use only usable readings. Pass rate includes all samples. A usable reading is not necessarily accurate.",
    }
    def metric(value, metric_unit, meaning):
        return dict(value=round(value, 4) if value is not None else None,
                    unit=metric_unit, meaning=meaning)

    metrics = {
        "MAE": metric(average, unit, "Average absolute error. Lower is better; 0 is perfect."),
        "RMSE": metric(summary.get("root_mean_squared_error"), unit,
                       "Error measure that gives larger mistakes more weight. Lower is better."),
        "bias": metric(summary.get("mean_signed_error"), unit,
                       "Average of predicted minus actual. Negative means reading too low; positive means too high."),
        "maximum_absolute_error": metric(worst, unit, "Largest absolute error in this test."),
        "pass_rate": metric(summary["within_tolerance_rate_percent"], "%",
                            f"Percentage of all samples within {summary['tolerance']:g} {unit} of actual. Higher is better."),
        "usable_reading_rate": metric(summary["valid_reading_rate_percent"], "%",
                                      "Percentage of samples with usable readings; this is not accuracy."),
        "sample_count": total,
        "usable_sample_count": valid,
        "note": "Error metrics use usable readings only. Rates use all samples. Null means unavailable. These results describe only the tested samples.",
    }
    return {
        "accuracy_rate_percent": round(summary["within_tolerance_rate_percent"], 4),
        "accuracy_rate_explanation": (
            f"{passed} out of {total} readings were accurate within +/- {summary['tolerance']:g} {unit} "
            f"of your actual readings. Missing or invalid readings count as not passing."
        ),
        "easy_to_read": explanation, "accuracy_metrics": metrics,
        **{key: value for key, value in summary.items()
           if key not in {"easy_to_read", "accuracy_metrics", "accuracy_rate_percent", "accuracy_rate_explanation"}},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="?", help="CSV containing image,expected columns")
    parser.add_argument("--profile", help="Exact profile name or unique prefix, e.g. UNIJIN or Badotherm")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--tolerance", type=float, default=None, help="Override allowed error in primary units (default: 2%% of gauge span)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "accuracy_results")
    parser.add_argument("--check-only", action="store_true", help="Check labels/images without loading ML models")
    parser.add_argument("--live", action="store_true", help="Evaluate saved live_predicted readings without loading models")
    parser.add_argument("--polar", action="store_true", help="Use Polar Annulus Unwrapping instead of geometric angle")
    args = parser.parse_args(argv)
    if args.tolerance is not None and (not math.isfinite(args.tolerance) or args.tolerance < 0):
        parser.error("--tolerance must be finite and nonnegative")

    try:
        from config import GAUGE_PROFILES, ROBOFLOW_API_KEY, NEEDLE_MODEL_ID, GAUGE_MODEL_ID, MODEL_CONFIDENCE
    except (ImportError, EnvironmentError) as error:
        parser.error(str(error))
    if args.list_profiles:
        for name, profile in GAUGE_PROFILES.items():
            print(f"{name}: labels and tolerance in {profile['UNIT_1']}")
        return 0
    if args.labels is None or not args.profile:
        parser.error("labels CSV and --profile are required")
    matches = [name for name in GAUGE_PROFILES if name.lower().startswith(args.profile.lower())]
    if len(matches) != 1:
        parser.error("Profile is unknown or ambiguous; use --list-profiles")
    name = matches[0]
    profile = GAUGE_PROFILES[name]
    span = profile["MAX_VAL_1"] - profile["MIN_VAL_1"]
    args.tolerance_basis = "2% of gauge span" if args.tolerance is None else "explicit absolute tolerance"
    if args.tolerance is None:
        args.tolerance = 0.02 * span
    args.tolerance_percent_of_span = 100 * args.tolerance / span
    try:
        samples = load_labels(args.labels.resolve(), profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.live:
        try:
            return evaluate_live(args, samples, name, profile)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    import cv2
    import numpy as np

    def read_image(path):
        # imdecode handles Unicode Windows paths, including this workspace.
        frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Cannot decode image")
        return frame

    if args.check_only:
        failures = 0
        for image_name, path, _ in samples:
            try:
                read_image(path)
            except (OSError, ValueError, cv2.error) as error:
                failures += 1
                print(f"FAIL {image_name}: {error}")
        print(f"Checked {len(samples)} images; {failures} failed. Models were not loaded.")
        return 1 if failures else 0

    os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
    from model_loader import load_models
    print("Loading models for offline evaluation...")
    needle_model, gauge_model = load_models()
    import config as model_config
    local_path = getattr(model_config, "LOCAL_MODEL_PATH", None) if getattr(model_config, "MODEL_BACKEND", "roboflow") == "local" else None
    gauge_error = None

    output = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (image_name, path, expected) in enumerate(samples, 1):
        row = dict(image=image_name, expected=expected, status="error", detail="")
        frame, center, tip = None, None, None
        try:
            frame = read_image(path)
            result, center, tip = detect_image(
                frame, needle_model, gauge_model, profile, MODEL_CONFIDENCE, use_polar=args.polar
            )
            row.update(result)
            if row["status"] == "ok":
                row["absolute_error"] = abs(row["predicted"] - expected)
                row["within_tolerance"] = row["absolute_error"] <= args.tolerance
        except Exception as error:
            row["detail"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        if frame is not None:
            display = frame.copy()
            if center is not None:
                cv2.circle(display, center, 6, (0, 0, 255), -1)
            if tip is not None:
                cv2.line(display, center, tip, (0, 255, 0), 2)
            text = f"Expected: {expected:g} {profile['UNIT_1']} | {row['status']}"
            if "predicted" in row:
                text += f" | Read: {row['predicted']:.2f} ({row['method']})"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            ok, encoded = cv2.imencode(".jpg", display)
            if ok:
                (output / f"{index:04d}_overlay.jpg").write_bytes(encoded.tobytes())
        print(f"[{index}/{len(samples)}] {image_name}: {row['status']}")

    fields = ["image", "expected", "status", "predicted", "angle", "method", "absolute_error", "within_tolerance", "detail"]
    with (output / "readings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.tolerance)
    summary.update(profile=name, unit=profile["UNIT_1"], profile_settings=profile,
                   tolerance_basis=args.tolerance_basis, tolerance_percent_of_span=args.tolerance_percent_of_span,
                   evaluation="independent unsmoothed images; no live tracking or manual locks",
                   needle_model=str(local_path) if local_path else NEEDLE_MODEL_ID,
                   gauge_model=str(local_path) if local_path else GAUGE_MODEL_ID,
                   gauge_model_available=gauge_model is not None, gauge_model_error=gauge_error,
                   labels=str(args.labels.resolve()))
    summary = explain_summary(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")
    return 0 if summary["valid"] else 1


def evaluate_live(args, samples, name, profile):
    """Compare recorded display values; missing readings count as failed samples."""
    with args.labels.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "live_predicted" not in (reader.fieldnames or []):
            raise ValueError("CSV has no live_predicted column. Capture new photos with S first.")
        labels = list(reader)
    rows = []
    for (image_name, _, expected), label in zip(samples, labels):
        row = dict(image=image_name, expected=expected, status="missing_live_prediction",
                   predicted="", absolute_error="", within_tolerance="")
        value = (label.get("live_predicted") or "").strip()
        if value:
            try:
                predicted = float(value)
            except ValueError:
                predicted = float("nan")
            if not math.isfinite(predicted):
                row["status"] = "invalid_live_prediction"
            elif not profile["MIN_VAL_1"] <= predicted <= profile["MAX_VAL_1"]:
                row.update(status="out_of_range", predicted=predicted)
            else:
                error = abs(predicted - expected)
                row.update(status="ok", predicted=predicted, absolute_error=error,
                           within_tolerance=error <= args.tolerance)
        rows.append(row)
    if args.check_only:
        failures = sum(row["status"] != "ok" for row in rows)
        print(f"Checked {len(rows)} live readings; {failures} missing or invalid. Models were not loaded.")
        return 1 if failures else 0
    output = args.output.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "readings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.tolerance)
    summary.update(profile=name, unit=profile["UNIT_1"],
                   tolerance_basis=args.tolerance_basis, tolerance_percent_of_span=args.tolerance_percent_of_span,
                   evaluation="saved live displayed readings", labels=str(args.labels.resolve()))
    summary = explain_summary(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {output}")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
