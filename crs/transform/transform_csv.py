import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import os

RAW_PATH = "data/raw/Mumbai_Weather_Data_2010_to_2026.csv"
PROCESSED_PATH = "data/processed/final_weather.csv"


def run_transform():
    print("🔄 Starting data transformation...")

    # ✅ Load data
    df = pd.read_csv(RAW_PATH)

    # -----------------------------
    # 🧹 1. Basic Cleaning
    # -----------------------------
    df.columns = df.columns.str.strip()

    # ✅ Convert datetime & remove timezone
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['datetime'] = df['datetime'].dt.tz_localize(None)

    # Drop invalid datetime rows
    df = df.dropna(subset=['datetime'])

    # -----------------------------
    # 🧼 2. Handle Missing Values
    # -----------------------------
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # -----------------------------
    # 🔢 3. Round Numeric Values (1 decimal)
    # -----------------------------
    df[numeric_cols] = df[numeric_cols].round(1)

    # -----------------------------
    # 🧠 4. Feature Engineering
    # -----------------------------
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Weekend flag (useful feature)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Temperature difference
    df['temp_diff'] = (df['temperature_C'] - df['feels_like_C']).round(1)

    # Wind category
    df['wind_category'] = pd.cut(
        df['wind_speed_kmh'],
        bins=[0, 5, 15, 30, 100],
        labels=['calm', 'breeze', 'windy', 'storm']
    )

    # -----------------------------
    # 📊 5. Extra Useful Features (🔥)
    # -----------------------------
    # Feels hotter or colder
    df['feels_hotter'] = (df['feels_like_C'] > df['temperature_C']).astype(int)

    # High humidity flag
    df['high_humidity'] = (df['humidity_pct'] > 70).astype(int)

    # Rain flag
    df['is_rain'] = (df['rainfall_mm'] > 0).astype(int)

    # -----------------------------
    # 🧾 6. Sort Data
    # -----------------------------
    df = df.sort_values(by='datetime')

    # -----------------------------
    # 💾 7. Save Processed Data
    # -----------------------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"✅ Transformed data saved at: {PROCESSED_PATH}")


if __name__ == "__main__":
    run_transform()