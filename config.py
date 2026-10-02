"""Configuration and gauge profiles for the Industrial Gauge Reader."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "local").strip().lower()
LOCAL_MODEL_PATH = Path(os.environ.get("LOCAL_MODEL_PATH", "best.pt"))
if not LOCAL_MODEL_PATH.is_absolute():
    LOCAL_MODEL_PATH = Path(__file__).resolve().parent / LOCAL_MODEL_PATH
LOCAL_OBB_PATHS = [
    Path(p.strip()) if Path(p.strip()).is_absolute() else Path(__file__).resolve().parent / p.strip()
    for p in os.environ.get("LOCAL_OBB_PATHS", "models/needle_obb_1.pt,models/needle_obb_2.pt").split(",")
    if p.strip()
]
if MODEL_BACKEND not in {"local", "roboflow"}:
    raise ValueError("MODEL_BACKEND must be local or roboflow")
if MODEL_BACKEND == "roboflow" and not ROBOFLOW_API_KEY:
    raise EnvironmentError(
        "ROBOFLOW_API_KEY not set. "
        "Create a .env file with ROBOFLOW_API_KEY=your_key "
        "or set it as a system environment variable."
    )

NEEDLE_MODEL_ID = "needle-gauge-mfp6n/1"
GAUGE_MODEL_ID = "pressure-gauge-pfjl2-wmzz4/1"

MODEL_CONFIDENCE = 0.05

GAUGE_PROFILES = {
    "UNIJIN (0-150 PSI / 0-10 kgf/cm2)": {
        "MIN_ANGLE": 135.0,       # 0 mark (0 PSI / 0 kgf/cm2) at 7:30 o'clock
        "MAX_ANGLE": 52.86,       # 150 PSI mark (~278 deg sweep; 10 kgf/cm2 mark at ~39 deg)
        "MIN_VAL_1": 0.0, "MAX_VAL_1": 150.0, "UNIT_1": "PSI",
        "MIN_VAL_2": 0.0, "MAX_VAL_2": 10.0,  "UNIT_2": "kgf/cm2",
        # Physical conversion: 1 PSI = 6894.757293168 Pa;
        # 1 kgf/cm2 = 98066.5 Pa. The printed 10 mark precedes 150 PSI.
        "SECONDARY_PER_PRIMARY": 6894.757293168 / 98066.5,
        "SHOW_SECONDARY": True,
    },
    "Badotherm (-1 to 15 bar)": {
        "POLAR_INTERVALS": 16,   # One interval per bar, including -1 to 0.
        "MIN_ANGLE": 123.66,      # Calibrated -1 bar mark
        "MAX_ANGLE": 29.78,       # 15 bar mark (266.12 deg sweep, 16.63 deg/bar)
        "MIN_VAL_1": -1.0, "MAX_VAL_1": 15.0, "UNIT_1": "bar",
        "VALUE_OFFSET_1": 0.0,
        "MIN_VAL_2": -14.5, "MAX_VAL_2": 217.5, "UNIT_2": "PSI",
        "SHOW_SECONDARY": False,
    },
}

# Optional measured scale alignment shared by live and photo evaluation.
from scale_calibration import load_calibration
load_calibration(GAUGE_PROFILES)
