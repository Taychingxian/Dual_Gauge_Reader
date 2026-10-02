"""Save raw frames and append labels without touching live tracking state."""

import csv
from datetime import datetime
import io
import json
import os
import tempfile
from pathlib import Path

import cv2


def save_snapshot(frame, profile_name, data_dir=None, *, live_predicted=None, diagnostics=None,
                  annotated_frame=None):
    """Return the CSV-relative image path; leave expected blank for manual entry."""
    gauge = profile_name.split(" (", 1)[0].lower()
    if gauge not in {"unijin", "badotherm"}:
        raise ValueError(f"Unsupported snapshot profile: {profile_name}")
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent / "accuracy_data"
    photo_dir = root / f"{gauge}_photos"
    os.makedirs(photo_dir, exist_ok=True)
    stem = f"{gauge}_{datetime.now():%Y%m%d_%H%M%S}"

    # imencode + Python file I/O supports Unicode Windows paths, unlike imwrite.
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise OSError("Could not encode snapshot as JPEG")
    suffix = 0
    while True:
        image_path = photo_dir / f"{stem}{'_' + str(suffix) if suffix else ''}.jpg"
        try:
            image_stream = image_path.open("xb")
            break
        except FileExistsError:
            suffix += 1
    try:
        with image_stream:
            image_stream.write(encoded.tobytes())
    except OSError:
        image_path.unlink(missing_ok=True)
        raise

    relative_path = image_path.relative_to(root).as_posix()
    if annotated_frame is not None:
        success, annotated = cv2.imencode(".jpg", annotated_frame)
        if not success:
            raise OSError(f"Clean photo saved at {image_path}, but detector photo encoding failed")
        image_path.with_name(f"{image_path.stem}_detector.jpg").write_bytes(annotated.tobytes())
    extra = {'raw_angle': ''}
    if gauge == 'unijin':
        extra.update(expected_kgf_cm2='', live_predicted_kgf_cm2=(
            '' if live_predicted is None else f'{live_predicted * (6894.757293168 / 98066.5):.4f}'))
    if diagnostics is not None:
        image_path.with_suffix('.json').write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
        extra.update({key: diagnostics.get(key, '') for key in ('raw_angle', 'fresh_measurement')})
    # Upgrade old CSV headers once, preserving every existing column and value.
    # Keep the photo if CSV access fails so the capture can be recovered.
    try:
        labels = root / f"{gauge}_labels.csv"
        required = ["image", "expected", "live_predicted", *extra]
        fields = required.copy()
        if labels.exists() and labels.stat().st_size:
            with labels.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = list(reader.fieldnames or [])
                if not {"image", "expected"}.issubset(fields):
                    raise ValueError("Labels CSV must contain image,expected columns")
                if any(key not in fields for key in required):
                    old_rows = list(reader)
                else:
                    old_rows = None
            if old_rows is not None:
                fields.extend(key for key in required if key not in fields)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                                     dir=root, delete=False) as stream:
                        temporary = Path(stream.name)
                        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(old_rows)
                    os.replace(temporary, labels)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        with labels.open("a+b") as stream:
            size = stream.tell()
            prefix = ""
            if size:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) not in (b"\n", b"\r"):
                    prefix = "\n"
            row = io.StringIO(newline="")
            writer = csv.DictWriter(row, fieldnames=fields, lineterminator="\n")
            if not size:
                writer.writeheader()
            writer.writerow(dict(image=relative_path, expected="",
                                 live_predicted="" if live_predicted is None else f"{live_predicted:.2f}", **extra))
            stream.write((prefix + row.getvalue()).encode("utf-8"))
    except OSError as error:
        raise OSError(f"Photo saved at {image_path}, but CSV append failed: {error}") from error
    return relative_path