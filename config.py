"""Configuration and gauge profiles for the Industrial Gauge Reader."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
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
        "MIN_ANGLE": 140.0,       # 0 PSI / 0 kgf/cm2 mark
        "MAX_ANGLE": 52.0,        # 150 PSI / 10 kgf/cm2 mark
        "MIN_VAL_1": 0.0, "MAX_VAL_1": 150.0, "UNIT_1": "PSI",
        "MIN_VAL_2": 0.0, "MAX_VAL_2": 10.0,  "UNIT_2": "kgf/cm2",
        "SHOW_SECONDARY": True,
    },
    "Badotherm (-1 to 15 bar)": {
        "MIN_ANGLE": 135.0,       # -1 bar mark
        "MAX_ANGLE": 45.0,        # 15 bar mark
        "MIN_VAL_1": -1.0, "MAX_VAL_1": 15.0, "UNIT_1": "bar",
        "MIN_VAL_2": -14.5, "MAX_VAL_2": 217.5, "UNIT_2": "PSI",
        "SHOW_SECONDARY": False,
    },
}
