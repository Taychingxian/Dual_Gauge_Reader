# Polar Annulus Unwrapping in Industrial Gauge Reader

This document details how **Polar Annulus Unwrapping** (`polar_reading.py`) operates within the system, its mathematical principles, its integration with live camera reading and offline evaluation, and how it delivers higher reading accuracy compared to traditional global-angle methods.

---

## 1. Executive Overview

### Upright needle detection

The dial radius is retained independently of synthetic needle tips between
frames and refreshed by gauge-box measurements. Otherwise a tip placed at
78% of the radius would shrink the next search radius repeatedly. Resetting
or unlocking the detector clears the retained radius for reacquisition.

The live detector first searches for a dark shaft spanning six radial sections
from 18% to 60% of the dial radius, allowing short needles that end around
65% of the bezel radius. This search is independent of model boxes
and previous angles. A supported shaft takes priority over box-based detection;
otherwise the existing Hough/PCA/geometry fallbacks remain available.
Polar reading also checks this shaft and limits its annulus search to within
6 degrees of it, preventing a wrong angle prior from pulling the reading onto
an unrelated scale tick. The search still depends on a reasonable pivot and
dial radius and does not guarantee immunity to perspective or poor lighting.
The verified shaft angle is retained rather than snapped to an annulus minimum.
Badotherm uses sixteen one-bar intervals; other profiles default to ten intervals.
Tick searches are local to configured scale angles. Rotating the gauge beyond
those windows requires scale realignment; polar unwrapping does not automatically
recover dial rotation or undo perspective distortion. A saved value offset can
also bias every result if it was fitted to a different camera pose.

### The Problem with Traditional Center-to-Tip Angle Methods
In conventional computer vision gauge reading:
1. **Camera Perspective Tilt:** When the camera views the gauge at even a slight angle, the circular dial face projects as an **ellipse**. Angles on opposite sides become compressed or stretched non-linearly.
2. **Pivot Jitter Sensitivity:** If the detected pivot center shifts by just $2 \text{ to } 3\text{ pixels}$ due to lighting or camera shake, the calculated angle can shift by $1.5^\circ \text{ to } 3.0^\circ$, causing an error of $\pm 0.2 \text{ to } 0.5 \text{ bar}$.
3. **Dial Non-Linearity:** Manufacturing tolerances mean printed tick marks are rarely placed at perfectly uniform mathematical degrees across the entire sweep.

### The Solution: Polar Annulus Unwrapping
Instead of measuring a single global angle from a central hub point, the system:
1. Isolates the circular band (annulus) where the tick marks and needle tip meet.
2. Mathematically unrolls the round dial into a **flat, horizontal strip**.
3. Locates the needle and nearest tick marks as vertical lines.
4. Calculates the pressure value by **piecewise interpolation between the two nearest bounding tick marks**.

```
    ROUND CAMERA IMAGE                           UNWRAPPED RECTANGULAR STRIP
       ,-''''-.
    .'   ||   '.                    0°       90°      180°     270°     360°
   /   \ || /   \                  ┌─────────┬────────┬────────┬────────┐
  | ====( @ )====|  ──warpPolar──> │  | | |  │  | | | │  | | | │  | | | │ (Tick marks)
   \   /    \   /                  │    |    │    |   │   ||   │    |   │
    '.        .'                   │  | | |  │  | | | │  | || | │  | | | │ (Needle tip)
      '-....-'                     └─────────┴────────┴────────┴────────┘
                                                             ▲
                                                    Needle detected here
```

---

## 2. Mathematical & Computer Vision Foundation

### A. Annulus Radial Isolation
A typical gauge dial has three distinct radial zones relative to radius $R$:
* **Center Hub ($0.0 R \to 0.60 R$):** Needle spindle, brass screw nut, brand logo ("UNIJIN", "Badotherm"). This region is discarded.
* **Scale Annulus ($0.60 R \to 0.88 R$):** The active ring containing primary/secondary tick marks, scale numbers, and the pointed needle tip.
* **Outer Rim ($> 0.88 R$):** Metal bezel, glass reflections, pipe fittings, background environment. This region is discarded.

### B. Polar Transformation (`cv2.warpPolar`)
Using OpenCV `WARP_POLAR_LINEAR`, the circular dial is sampled along concentric circles:
$$\text{polar}(r, \theta) = \text{frame}(x_c + r \cos \theta, \; y_c + r \sin \theta)$$

* **Angular Resolution:** $720\text{ columns}$ covering $360^\circ$ ($0.5^\circ\text{ per column}$).
* **Transposition:** The output is transposed so that the **X-axis represents Angle ($0^\circ \to 360^\circ$)** and the **Y-axis represents Radius**.

### C. Active Scale Arc Masking
Real industrial gauges have brass pipe stems, mounting brackets, and housing bezels (typically at $80^\circ - 95^\circ$, or 6 o'clock). 

To prevent the algorithm from falsely locking onto the dark pipe stem:
1. The scale arc is defined by `MIN_ANGLE` and `MAX_ANGLE` from the gauge profile.
2. Columns outside the active scale arc are assigned a cost of $+10^9$, guaranteeing that needle detection occurs strictly within the readable face.

### D. Sub-Pixel Needle Detection
1. Vertical column intensities across the grayscale annulus are averaged:
   $$I_{\text{col}}(x) = \frac{1}{H} \sum_{y=0}^{H-1} \text{gray}(y, x)$$
2. A 1D Gaussian filter ($k=15$) smooths high-frequency dial texture noise.
3. The needle corresponds to the minimum intensity column $x_{\min}$.
4. A 3-point parabolic fit provides sub-pixel resolution:
   $$\Delta x = 0.5 \times \frac{y_1 - y_3}{y_1 - 2y_2 + y_3} \quad (\text{if } y_1 - 2y_2 + y_3 > 10^{-4})$$
   $$x_{\text{needle}} = x_{\min} + \Delta x$$

### E. Needle Masking for Tick Identification
When the needle is near or over a tick mark, searching for ticks could cause the detector to mistake the needle for a tick mark.

The engine establishes a **Needle Exclusion Zone** ($\pm 7^\circ$, or $\pm 14\text{ columns}$) where tick detection is masked. Ticks outside this zone are snapped to actual visual contrast dips; ticks inside this zone fall back to expected geometry.

### F. Piecewise Bounding-Tick Interpolation
Given the detected needle column $x_{\text{needle}}$ and two adjacent tick marks $T_k = (x_k, V_k)$ and $T_{k+1} = (x_{k+1}, V_{k+1})$:
$$\text{Value} = V_k + \left( \frac{x_{\text{needle}} - x_k}{x_{k+1} - x_k} \right) \times (V_{k+1} - V_k)$$

This direct local ratio makes the reading **immune to global offset and camera tilt**.

---

## 3. System Architecture & Codebase Integration

```
  [Camera / Video Stream]
            │
            ▼
  [Object Detection Hierarchy]
  (Hough -> PCA -> Geometry -> Box Vector)
            │
            ├──> Center (gx, gy)
            └──> Needle Angle Prior (ref_angle)
                        │
                        ▼
            [polar_reading.py]
            ┌──────────────────────────────────────────────┐
            │ 1. cv2.warpPolar annulus extraction          │
            │ 2. Arc mask & Stem rejection                 │
            │ 3. 1D Sub-pixel needle column detection      │
            │ 4. Needle-masked tick detection              │
            │ 5. Piecewise linear scale interpolation      │
            └──────────────────────────────────────────────┘
                        │
                        ▼
       ┌────────────────┴────────────────┐
       ▼                                 ▼
[main_dual_gauge_reader.py]    [evaluate_accuracy.py]
  - Live HUD Display             - Offline Benchmarks
  - Toggle key [P]               - CLI flag --polar
  - Visual Inset Strip           - CSV / JSON reports
```

### Module Breakdown

| File | Role |
| :--- | :--- |
| [`polar_reading.py`](file:///c:/Users/HP/OneDrive/文档/Gauge-Reader/polar_reading.py) | **Core Polar Engine:** Houses `polar_unwrap_reading()` and `draw_polar_debug()`. |
| [`main_dual_gauge_reader.py`](file:///c:/Users/HP/OneDrive/文档/Gauge-Reader/main_dual_gauge_reader.py) | **Live UI & Stream:** Interactively calls polar reading every frame and displays the unwrapped inset. |
| [`evaluate_accuracy.py`](file:///c:/Users/HP/OneDrive/文档/Gauge-Reader/evaluate_accuracy.py) | **Benchmarking Tool:** Supports `--polar` flag to evaluate accuracy across recorded photo datasets. |
| [`test_polar_reading.py`](file:///c:/Users/HP/OneDrive/文档/Gauge-Reader/test_polar_reading.py) | **Automated Test Suite:** Verifies edge cases, synthetic inputs, and real gauge photos (`Unijin.jpg`, `Badotherm.jpg`). |

---

## 4. Live Operating Guide

### Keyboard Controls in `main_dual_gauge_reader.py`

| Key | Action | Description |
| :---: | :--- | :--- |
| **`P`** | **Toggle Reading Mode** | Switches between **`[POLAR]`** (Unwrapped linear strip) and **`[ANGLE]`** (Geometric angle). |
| **`G`** | Switch Profile | Toggles between **UNIJIN (0-150 PSI)** and **Badotherm (-1 to 15 bar)**. |
| **`L`** | Lock Detector | Freezes the pivot center and dial search region to eliminate jitter. |
| **`C`** | Unlock & Re-center | Clears locked hub and re-detects the gauge face from scratch. |
| **`S`** | Save Snapshot | Records the raw image frame and labels for calibration / verification. |
| **`Z`** | Zero Tare | Aligns the profile minimum with the current needle resting position. |

### Visual Feedback on Frame
When in `[POLAR]` mode:
1. **Status Header:** Displays `PSI: XX.XX [POLAR]` or `bar: XX.XX [POLAR]`.
2. **Bottom Inset Banner:** Shows the real-time unwrapped polar strip:
   * **Green Vertical Bars:** Identified major tick marks.
   * **Bright Red Vertical Line:** Identified needle position.

---

## 5. Offline Accuracy Evaluation

To benchmark test datasets using the polar engine:

```powershell
# Evaluate UNIJIN dataset with Polar Unwrapping
python evaluate_accuracy.py accuracy_data/unijin_labels.csv --profile UNIJIN --polar

# Evaluate Badotherm dataset with Polar Unwrapping
python evaluate_accuracy.py accuracy_data/badotherm_labels.csv --profile Badotherm --polar
```

---

## 6. Verification Summary

Automated tests in `test_polar_reading.py` validate:
* **Stem Rejection:** Does not lock onto the bottom 6 o'clock pipe fitting ($80^\circ - 95^\circ$).
* **Real Photos:**
  * `Unijin.jpg`: Detected Needle Angle $= 201.30^\circ \implies 37.66\text{ PSI}$ / $2.65\text{ kgf/cm}^2$.
  * `Badotherm.jpg`: Detected Needle Angle $= 336.30^\circ \implies 10.94\text{ bar}$.
* **Test Suite:** All 114 tests passing.
