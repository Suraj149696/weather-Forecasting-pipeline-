"""
=============================================================================
LSTM Temperature Forecasting — Inference
File location : crs/model/predict.py

Flow:
  1. Load last 168 rows  <- data/processed/final_weather.csv
  2. Load model          <- models/best_model.keras
  3. Load scaler         <- models/scaler_params.pkl
  4. Run 24-hour forecast
  5. Save result         -> data/Forecast/forecast.json
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import tensorflow as tf

# ─── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from crs.utils.logger import get_logger
from crs.utils.config import config

logger = get_logger("predict")
logger.info(f"Project root : {PROJECT_ROOT}")

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH      = os.path.join(PROJECT_ROOT, "data", "processed", "final_weather.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "best_model.keras")
SCALER_PATH   = os.path.join(PROJECT_ROOT, "models", "scaler_params.pkl")
FORECAST_DIR  = os.path.join(PROJECT_ROOT, "data", "Forecast")
FORECAST_FILE = os.path.join(FORECAST_DIR, "forecast.json")
DATETIME_COL  = "datetime"

# ─── Unified date format ──────────────────────────────────────────────────────
DATE_FORMAT = "%Y-%m-%d %H:%M"   # e.g.  2026-05-05 00:00

# ─── Constants ────────────────────────────────────────────────────────────────
N_STEPS   = 168
FUT_HOURS = 24

FEATURE_COLS = [
    "temperature_C",
    "humidity_pct",
    "pressure_hpa",
    "rainfall_mm",
    "wind_speed_kmh",
    "wind_dir_deg",
    "solar_rad_Wm2",
    "dew_point_C",
    "feels_like_C",
    "cloud_cover_pct",
    "hour",
    "month",
    "is_daytime",
]
TARGET_COL   = "temperature_C"
TARGET_INDEX = FEATURE_COLS.index(TARGET_COL)


# ─── 1. Load model & scaler ───────────────────────────────────────────────────
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n  Model not found  : {MODEL_PATH}"
            f"\n  >> Run train.py first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"\n  Scaler not found : {SCALER_PATH}"
            f"\n  >> Run train.py first."
        )

    logger.info(f"Loading model  -> {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    logger.info(f"Loading scaler -> {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ─── 2. Load last 168 rows ────────────────────────────────────────────────────
def load_recent_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"\n  CSV not found : {CSV_PATH}")

    logger.info(f"Reading CSV -> {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    df_clean = df.dropna(subset=FEATURE_COLS)

    if len(df_clean) < N_STEPS:
        raise ValueError(f"Need {N_STEPS} rows, only got {len(df_clean)}")

    last_168 = df_clean[FEATURE_COLS].iloc[-N_STEPS:].reset_index(drop=True)
    last_dt  = pd.to_datetime(df_clean[DATETIME_COL].iloc[-1]).to_pydatetime()

    logger.info(f"Last known timestamp : {last_dt.strftime(DATE_FORMAT)}")
    logger.info(f"Input window         : {N_STEPS} rows  ({N_STEPS // 24} days)")

    return last_168, last_dt


# ─── 3. Inverse-transform helper ─────────────────────────────────────────────
def _inverse_transform(norm_2d: np.ndarray, scaler) -> np.ndarray:
    samples, fut_hours = norm_2d.shape
    out   = np.zeros_like(norm_2d)
    dummy = np.zeros((samples, scaler.n_features_in_), dtype=np.float32)
    for h in range(fut_hours):
        dummy[:, TARGET_INDEX] = norm_2d[:, h]
        out[:, h] = scaler.inverse_transform(dummy)[:, TARGET_INDEX]
    return out


# ─── 4. Predict ───────────────────────────────────────────────────────────────
def predict_next_24h(recent_168h_df: pd.DataFrame, model, scaler) -> np.ndarray:
    if len(recent_168h_df) != N_STEPS:
        raise ValueError(f"Need exactly {N_STEPS} rows, got {len(recent_168h_df)}")

    x       = recent_168h_df[FEATURE_COLS].values.astype(np.float32)
    x_norm  = scaler.transform(x)
    x_input = x_norm[np.newaxis, ...]

    y_norm = model.predict(x_input, verbose=0)
    y_C    = _inverse_transform(y_norm, scaler)
    return y_C[0]   # shape (24,)


# ─── 5. Save forecast.json ────────────────────────────────────────────────────
def save_forecast(forecast_C: np.ndarray, base_dt: datetime) -> dict:
    """
    All datetimes use the same format: YYYY-MM-DD HH:MM

    Output example:
    {
        "generated_at"  : "2026-05-07 16:32",
        "base_datetime" : "2026-05-04 23:00",
        "unit"          : "C",
        "forecast_hours": 24,
        "forecast": [
            {"horizon": "h+01", "datetime": "2026-05-05 00:00", "temperature_C": 28.83},
            {"horizon": "h+02", "datetime": "2026-05-05 01:00", "temperature_C": 28.73},
            ...
        ]
    }
    """
    os.makedirs(FORECAST_DIR, exist_ok=True)

    forecast_list = []
    for h, temp in enumerate(forecast_C, 1):
        forecast_dt = base_dt + timedelta(hours=h)
        forecast_list.append({
            "horizon"       : f"h+{h:02d}",
            "datetime"      : forecast_dt.strftime(DATE_FORMAT),   # ✅ same format
            "temperature_C" : round(float(temp), 2),
        })

    output = {
        "generated_at"  : datetime.now().strftime(DATE_FORMAT),    # ✅ same format
        "base_datetime" : base_dt.strftime(DATE_FORMAT),           # ✅ same format
        "unit"          : "C",
        "forecast_hours": FUT_HOURS,
        "forecast"      : forecast_list,
    }

    with open(FORECAST_FILE, "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Forecast saved -> {FORECAST_FILE}")
    return output


# ─── 6. Main ──────────────────────────────────────────────────────────────────
def run_forecast() -> dict:
    model, scaler      = load_artifacts()
    recent_df, last_dt = load_recent_data()
    forecast_C         = predict_next_24h(recent_df, model, scaler)
    result             = save_forecast(forecast_C, last_dt)

    logger.info("── 24-Hour Temperature Forecast ──")
    for item in result["forecast"]:
        logger.info(f"  {item['horizon']}  {item['datetime']}  ->  {item['temperature_C']:.2f} C")

    return result


if __name__ == "__main__":
    run_forecast()