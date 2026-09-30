# Accuracy Testing Guide

[Back to README](README.md)

Use `evaluate_accuracy.py` to compare computer predictions with independently
recorded actual readings. Run commands in the VS Code **Terminal**, from the
project directory, using the same Python environment as the live reader.

## 1. Capture samples

```powershell
python main_dual_gauge_reader.py
```

Select the profile with **G**, then press **S** to save a clean camera photo and
the valid displayed primary reading. **V** switches cameras; **L** locks the
detector; **C** unlocks it. Uppercase and lowercase work.

| Gauge | Photo folder | Labels CSV | Actual-reading unit |
|---|---|---|---|
| UNIJIN | `accuracy_data/unijin_photos/` | `accuracy_data/unijin_labels.csv` | PSI |
| Badotherm | `accuracy_data/badotherm_photos/` | `accuracy_data/badotherm_labels.csv` | bar |

Folders and CSV headers are created automatically. Photos have timestamped names;
repeated captures within a second get numeric suffixes to prevent overwriting.
Snapshots have no drawn overlays and do not change tracking or filter history.
A green saved message appears for two seconds. File saving is synchronous, so
its duration depends on the disk and image size.

## 2. Fill in the actual readings

Open the appropriate CSV in Excel or a text editor. A completed row looks like:

```csv
image,expected,live_predicted
unijin_photos/unijin_20260930_113527.jpg,120,116.25
```

- `image`: path relative to the CSV location.
- `expected`: actual reading entered manually, in the primary unit above.
- `live_predicted`: primary reading displayed when S was pressed, to two decimal places.

Fill every `expected` cell. Fill or remove unused example rows such as
`../Unijin.jpg,,`. Save as CSV and close Excel before taking more snapshots,
because Excel may lock the file. Never copy predictions into the actual-reading
column just to improve the score.

Existing two-column CSVs are upgraded on the next capture while preserving
labels. Older live predictions cannot be recovered.

### Why can live_predicted be blank?

The image still saves when no valid reading is available. A live value is
recorded only when the gauge center and needle tip are available, measurement
is ready, and the angle is within the valid range. A locking, absent, or invalid
reading leaves the cell blank.

Do not invent the missing value. Capture again when a valid reading is available,
or evaluate the photo independently. A new photo prediction is not a recovered
live reading. In live evaluation, missing predictions count as failed samples.

## 3. Choose what to evaluate

### Recorded live readings

```powershell
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN --live
python evaluate_accuracy.py accuracy_data/badotherm_labels.csv --profile Badotherm --live
```

This compares `live_predicted` with `expected`. It does not load detection models,
open a camera, or process photos. The current configuration still requires the
existing API key in `.env`.

### Independent predictions from photos

```powershell
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN
python evaluate_accuracy.py accuracy_data/badotherm_labels.csv --profile Badotherm
```

Without `--live`, each photo gets a fresh prediction using the existing detection
and measurement functions. This loads the configured models and may require an
initial model download. It does not use live smoothing, tracking history, or
manual locks, so its predictions may differ from the recorded live values.

### Validate before evaluation

```powershell
# Validate actual readings and recorded live predictions
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN --live --check-only

# Validate actual readings and decode images, without loading models
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN --check-only

python evaluate_accuracy.py --list-profiles
python evaluate_accuracy.py --help
```

Check-only mode does not create results. Use the full commands in the terminal;
VS Code's Run Code button alone does not supply the required arguments.

## 4. Default tolerance: 2% of span

```text
span = maximum value - minimum value
allowed error = 0.02 × span
pass when |predicted - expected| <= allowed error
```

| Gauge | Span | Default allowed error |
|---|---:|---:|
| UNIJIN: 0–150 PSI | 150 PSI | ±3 PSI |
| Badotherm: −1–15 bar | 16 bar | ±0.32 bar |

For example, a UNIJIN actual value of 100 PSI passes with predictions from
97 to 103 PSI. A Badotherm actual value of 1.30 bar passes from 0.98 to 1.62 bar.
The boundary is included. This is a project testing criterion, not certification.

Both evaluation modes use this default. An explicit `--tolerance` overrides it
in **primary units, not percent**. For example, `--tolerance 2` means ±2 PSI for
UNIJIN or ±2 bar for Badotherm. Omit the option for the 2%-of-span rule.

## 5. Read the results

Each evaluation creates a timestamped directory under `accuracy_results/`.
Use `--output folder_name` to change the parent directory.

| File | Contents |
|---|---|
| `summary.json` | Accuracy percentage, readable explanation, tolerance, and metrics |
| `readings.csv` | Actual/predicted values, absolute error, status, and pass/fail |
| `0001_overlay.jpg`, etc. | Detection overlays for decoded images in photo mode only |

### Where is the accuracy percentage?

Read **`accuracy_rate_percent`** in `summary.json`.

```text
accuracy rate = samples within tolerance / all samples × 100
```

For example, 4 passing samples out of 5 means **80% within the chosen tolerance**.
It does not mean each numerical prediction is 80% correct.

| JSON field | Meaning |
|---|---|
| `accuracy_rate_explanation` | Passed count and allowed error in plain English |
| `easy_to_read` | Short explanation of the results |
| `tolerance` | Allowed error in primary units |
| `tolerance_basis` | Default: `2% of gauge span` |
| `tolerance_percent_of_span` | Default: `2.0` |
| `accuracy_metrics.MAE` | Average absolute error; lower is better |
| `accuracy_metrics.RMSE` | Error measure emphasizing larger mistakes |
| `accuracy_metrics.bias` | Average prediction minus actual; negative means reading low |
| `accuracy_metrics.maximum_absolute_error` | Largest absolute error |
| `accuracy_metrics.pass_rate` | Same percentage as `accuracy_rate_percent` |
| `accuracy_metrics.usable_reading_rate` | Percentage that could be evaluated, not accuracy |

Error metrics use usable readings only. Rates use all samples. Missing or invalid
live predictions and photo-processing failures count as not passing. A usable
reading can still be outside tolerance. No usable readings means error metrics
are `null`, not zero. Invalid or missing `expected` values stop evaluation rather
than silently skipping unlabeled samples.

An evaluation returns code 0 if at least one reading is usable; this does not
mean every sample passed. All-invalid evaluations return code 1. Check-only
returns code 1 when it finds a failed check.

## 6. Collect meaningful evidence

Test across pressure levels, lighting, camera angles, and separate sessions.
Snapshots seconds apart can repeat the same tracked value and provide limited
independent evidence. Keep final evaluation samples separate from samples used
to tune calibration. Five similar captures do not establish general reliability.

Human dial labels measure agreement with that human reading and carry reading
uncertainty. Calibrator labels also include the physical gauge's error. Record
which reference method you used and keep it consistent when comparing results.

The evaluator does not automatically adjust calibration or detection thresholds.

## Troubleshooting

| Problem | Action |
|---|---|
| `labels CSV and --profile are required` | Paste a complete command into the terminal |
| `Line N: expected must be a number` | Enter that row's actual reading or remove an unused example row; save the CSV |
| `CSV has no live_predicted column` | Capture new samples with the updated live reader |
| Blank `live_predicted` | Capture with a valid displayed reading; do not fabricate old values |
| CSV append failed | Close Excel and inspect the console message; the photo may still be saved |
| Image cannot be loaded | Check the CSV-relative path and that the photo exists |

For software regression checks, run `python -m pytest -q`. Those tests verify
code behavior; use the evaluation commands above to measure your gauge readings.
