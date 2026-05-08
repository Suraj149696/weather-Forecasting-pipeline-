import sys
sys.stdout.reconfigure(encoding='utf-8')

import openmeteo_requests
import pandas as pd
import os
from datetime import datetime, timedelta

# Paths (use relative for pipeline)
RAW_API_PATH = "data/raw/api_weather.csv"
FINAL_PATH = "data/processed/final_weather.csv"


# --------------------------------------------------
# 🔍 Get last available date from processed data
# --------------------------------------------------
def get_last_date():
    print("🔍 Checking existing processed data...")

    if not os.path.exists(FINAL_PATH):
        print("❌ No processed file found")
        return None

    df = pd.read_csv(FINAL_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])

    last_date = df['datetime'].max().date()
    print(f"✅ Last available date: {last_date}")

    return last_date


# --------------------------------------------------
# 🌐 Extract Incremental API Data
# --------------------------------------------------
def run_extract_api():
    print("\n🌐 STEP: Extract API Data")

    last_date = get_last_date()

    if last_date is None:
        print("⚠️ No historical data → Run full load separately")
        return

    # Incremental range
    start_date = last_date + timedelta(days=1)
    end_date = datetime.now().date() - timedelta(days=1)

    print(f"📅 Start Date: {start_date}")
    print(f"📅 End Date: {end_date}")

    # No new data case
    if start_date > end_date:
        print("✅ Data already up-to-date")
        return

    # --------------------------------------------------
    # 📡 API Call
    # --------------------------------------------------
    print("📡 Calling Open-Meteo API...")

    om = openmeteo_requests.Client()

    params = {
        "latitude": 19.076,
        "longitude": 72.8777,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "wind_speed_10m", "wind_direction_10m",
            "surface_pressure", "cloud_cover", "visibility",
            "dew_point_2m", "shortwave_radiation", "is_day"
        ],
        "timezone": "Asia/Kolkata"
    }

    try:
        r = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
        h = r.Hourly()
        print("✅ API response received")
    except Exception as e:
        print("❌ API Error:", e)
        return

    # --------------------------------------------------
    # 📊 Convert API → DataFrame
    # --------------------------------------------------
    try:
        time_index = pd.date_range(
            start=pd.to_datetime(h.Time(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
            periods=h.Variables(0).ValuesAsNumpy().shape[0],
            freq="h"
        )

        df = pd.DataFrame({
            "datetime":        time_index,
            "temperature_C":   h.Variables(0).ValuesAsNumpy(),
            "humidity_pct":    h.Variables(1).ValuesAsNumpy(),
            "feels_like_C":    h.Variables(2).ValuesAsNumpy(),
            "rainfall_mm":     h.Variables(3).ValuesAsNumpy(),
            "wind_speed_kmh":  h.Variables(4).ValuesAsNumpy(),
            "wind_dir_deg":    h.Variables(5).ValuesAsNumpy(),
            "pressure_hpa":    h.Variables(6).ValuesAsNumpy(),
            "cloud_cover_pct": h.Variables(7).ValuesAsNumpy(),
            "visibility_km":   h.Variables(8).ValuesAsNumpy() / 1000,  # meters → km
            "dew_point_C":     h.Variables(9).ValuesAsNumpy(),
            "solar_rad_Wm2":   h.Variables(10).ValuesAsNumpy(),
            "is_daytime":      h.Variables(11).ValuesAsNumpy(),
        })

        print(f"📊 Extracted rows: {len(df)}")
        print("📌 Sample data:")
        print(df.head())

    except Exception as e:
        print("❌ Data conversion error:", e)
        return

    # --------------------------------------------------
    # 💾 Save Raw Data
    # --------------------------------------------------
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(RAW_API_PATH, index=False)

    print(f"💾 Saved API data → {RAW_API_PATH}")


# --------------------------------------------------
if __name__ == "__main__":
    run_extract_api()