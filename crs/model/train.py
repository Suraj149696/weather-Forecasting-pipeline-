"""
=============================================================================
LSTM Temperature Forecasting Model — 24-Hour Ahead Forecast
File location : crs/model/train.py
=============================================================================
"""

# ─── 0. Imports ───────────────────────────────────────────────────────────────
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print(f"TensorFlow version: {tf.__version__}")

# ─── Resolve project root (2 levels up from crs/model/train.py) ───────────────
# train.py  →  crs/model/train.py
# ROOT      →  Weather_Forcastong_Project/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(f"Project root : {PROJECT_ROOT}")

# ─── 1. Configuration (all paths are ABSOLUTE) ────────────────────────────────
CSV_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "final_weather.csv")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")                          # models/
SCALER_PATH    = os.path.join(PROJECT_ROOT, "models", "scaler_params.pkl")     # ✅ FIXED
PLOT_PATH      = os.path.join(PROJECT_ROOT, "models", "lstm_24h_results.png")  # ✅ FIXED
DATETIME_COL   = "datetime"
TARGET_COL     = "temperature_C"

# LSTM hyper-parameters
N_STEPS     = 168   # look-back window: 7 days × 24 hours
FUT_HOURS   = 24    # predict next 24 hours
LSTM_UNITS  = 128
BATCH_SIZE  = 64
EPOCHS      = 50
SPLIT_TRAIN = 0.80
SPLIT_VAL   = 0.10  # remaining 0.10 → test

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── 2. Feature Selection ─────────────────────────────────────────────────────
FEATURE_COLS = [
    "temperature_C",    # target (index 0)
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

TARGET_INDEX = FEATURE_COLS.index(TARGET_COL)   # must stay 0

# ─── 3. Load & Clean Data ─────────────────────────────────────────────────────
print("\n[1/7] Loading data ...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, parse_dates=[DATETIME_COL])
df.sort_values(DATETIME_COL, inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"      Rows: {len(df):,}  |  Date range: {df[DATETIME_COL].min()} -> {df[DATETIME_COL].max()}")

df_clean = df[FEATURE_COLS].dropna()
print(f"      After dropna: {len(df_clean):,} rows")

data_array = df_clean[FEATURE_COLS].values   # shape (N, n_features)
n_features = data_array.shape[1]

# ─── 4. Train / Val / Test Split (chronological) ──────────────────────────────
print("\n[2/7] Splitting dataset ...")
n         = len(data_array)
train_end = int(n * SPLIT_TRAIN)
val_end   = int(n * (SPLIT_TRAIN + SPLIT_VAL))

data_train = data_array[:train_end]
data_val   = data_array[train_end:val_end]
data_test  = data_array[val_end:]

print(f"      Train: {len(data_train):,}  Val: {len(data_val):,}  Test: {len(data_test):,}")

# ─── 5. Normalisation (fit on train only) ─────────────────────────────────────
print("\n[3/7] Normalising ...")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)   # ensure models/ exists before saving

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_norm = scaler.fit_transform(data_train)
data_val_norm   = scaler.transform(data_val)
data_test_norm  = scaler.transform(data_test)

joblib.dump(scaler, SCALER_PATH)             # ✅ saves to models/scaler_params.pkl
print(f"      Scaler saved -> {SCALER_PATH}")

# ─── 6. Sequence Creation (Multi-Step) ────────────────────────────────────────
def create_sequences(data, n_steps, fut_hours, target_idx):
    """
    X shape: (samples, n_steps, n_features)
    y shape: (samples, fut_hours)
    """
    X, y = [], []
    for i in range(len(data) - n_steps - fut_hours + 1):
        end_ix = i + n_steps
        seq_x  = data[i:end_ix, :]
        seq_y  = data[end_ix:end_ix + fut_hours, target_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


print("\n[4/7] Creating sequences ...")
X_train, y_train = create_sequences(data_train_norm, N_STEPS, FUT_HOURS, TARGET_INDEX)
X_val,   y_val   = create_sequences(data_val_norm,   N_STEPS, FUT_HOURS, TARGET_INDEX)
X_test,  y_test  = create_sequences(data_test_norm,  N_STEPS, FUT_HOURS, TARGET_INDEX)

print(f"      X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"      X_val  : {X_val.shape}    y_val  : {y_val.shape}")
print(f"      X_test : {X_test.shape}   y_test : {y_test.shape}")

# ─── 7. Model Definition ──────────────────────────────────────────────────────
print("\n[5/7] Building LSTM model ...")

model = Sequential([
    LSTM(LSTM_UNITS,
         input_shape=(N_STEPS, n_features),
         return_sequences=False,
         name="lstm_1"),
    Dropout(0.2, name="dropout_1"),
    Dense(64, activation="relu", name="dense_hidden"),
    Dense(FUT_HOURS, name="output")   # 24 outputs
], name="TempForecast_LSTM_24h")

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# ─── 8. Callbacks ─────────────────────────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")   # ✅ models/best_model.keras

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        mode="min",
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=0
    ),
]

# ─── 9. Training ──────────────────────────────────────────────────────────────
print("\n[6/7] Training ...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

# ─── 10. Evaluation ───────────────────────────────────────────────────────────
print("\n[7/7] Evaluating on test set ...")

y_pred_norm = model.predict(X_test, verbose=0)   # (samples, 24)

def inverse_transform_target(norm_values_2d, scaler, target_idx):
    """Un-normalise (samples, fut_hours) -> degrees C."""
    samples, fut_hours = norm_values_2d.shape
    out   = np.zeros_like(norm_values_2d)
    dummy = np.zeros((samples, scaler.n_features_in_), dtype=np.float32)
    for h in range(fut_hours):
        dummy[:, target_idx] = norm_values_2d[:, h]
        out[:, h] = scaler.inverse_transform(dummy)[:, target_idx]
    return out

y_pred_C = inverse_transform_target(y_pred_norm, scaler, TARGET_INDEX)
y_true_C = inverse_transform_target(y_test,      scaler, TARGET_INDEX)

rmse_per_h   = np.sqrt(np.mean((y_true_C - y_pred_C) ** 2, axis=0))
mae_per_h    = np.mean(np.abs(y_true_C - y_pred_C), axis=0)
rmse_overall = np.sqrt(mean_squared_error(y_true_C.flatten(), y_pred_C.flatten()))
mae_overall  = mean_absolute_error(y_true_C.flatten(), y_pred_C.flatten())

print(f"\n  Overall Test RMSE : {rmse_overall:.3f} °C")
print(f"  Overall Test MAE  : {mae_overall:.3f} °C")
print(f"\n  Per-horizon RMSE (h+1 to h+24):")
for h, (r, m) in enumerate(zip(rmse_per_h, mae_per_h), 1):
    print(f"    h+{h:02d}: RMSE={r:.3f}°C  MAE={m:.3f}°C")

# ─── 11. Plots ────────────────────────────────────────────────────────────────
hours = np.arange(1, FUT_HOURS + 1)
fig, axes = plt.subplots(4, 1, figsize=(14, 18))

ax = axes[0]
ax.plot(history.history["loss"],     label="Train Loss (MSE)")
ax.plot(history.history["val_loss"], label="Val Loss (MSE)", linestyle="--")
ax.set_title("Training & Validation Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.bar(hours, rmse_per_h, alpha=0.7, label="RMSE per horizon")
ax.plot(hours, mae_per_h, color="red", marker="o", linewidth=1.5, label="MAE per horizon")
ax.set_title("Error by Forecast Horizon (h+1 to h+24)")
ax.set_xlabel("Hours ahead"); ax.set_ylabel("Error (°C)")
ax.set_xticks(hours); ax.legend(); ax.grid(alpha=0.3, axis="y")

ax = axes[2]
ax.plot(hours, y_true_C[-1], label="Actual",    marker="o",  linewidth=1.5)
ax.plot(hours, y_pred_C[-1], label="Predicted", marker="x",  linewidth=1.5, linestyle="--")
ax.set_title("Sample Forecast - Last Test Window (24-Hour Horizon)")
ax.set_xlabel("Hours ahead"); ax.set_ylabel("Temperature (°C)")
ax.set_xticks(hours); ax.legend(); ax.grid(alpha=0.3)

zoom_n = 7 * 24
ax = axes[3]
ax.plot(y_true_C[-zoom_n:, 0], label="Actual (h+1)",    alpha=0.85, linewidth=1.2)
ax.plot(y_pred_C[-zoom_n:, 0], label="Predicted (h+1)", alpha=0.85, linewidth=1.2, linestyle="--")
ax.set_title("Zoomed - Last 7 Days of Test Set (h+1 slice)")
ax.set_xlabel("Hour index"); ax.set_ylabel("Temperature (°C)")
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)   # ✅ saves to models/lstm_24h_results.png
plt.show()
print(f"\n  Plot saved -> {PLOT_PATH}")

# ─── 12. Save final model ─────────────────────────────────────────────────────
FINAL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.keras")  # ✅ models/best_model.keras
model.save(FINAL_MODEL_PATH)
print(f"  Model saved -> {FINAL_MODEL_PATH}")

# ─── 13. Summary ──────────────────────────────────────────────────────────────
print("\n[Done] Summary:")
print(f"   Project root     : {PROJECT_ROOT}")
print(f"   Features used    : {FEATURE_COLS}")
print(f"   Look-back        : {N_STEPS} hours  ({N_STEPS//24} days)")
print(f"   Forecast horizon : {FUT_HOURS} hours ahead")
print(f"   Overall Test RMSE: {rmse_overall:.3f} °C")
print(f"   Overall Test MAE : {mae_overall:.3f} °C")
print(f"\n   Saved files:")
print(f"     Scaler  -> {SCALER_PATH}")
print(f"     Model   -> {FINAL_MODEL_PATH}")
print(f"     Plot    -> {PLOT_PATH}")