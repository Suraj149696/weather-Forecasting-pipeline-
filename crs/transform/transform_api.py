import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import pandas as pd
import logging

# =========================================================
# PROJECT ROOT PATH
# =========================================================
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# =========================================================
# FILE PATHS
# =========================================================
RAW_API_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "api_weather.csv"
)

FINAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_weather.csv"
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# MAIN TRANSFORMATION FUNCTION
# =========================================================
def run_transform_incremental():

    try:

        logger.info("Starting incremental transformation...")

        # =================================================
        # 1. CHECK API FILE
        # =================================================
        if not os.path.exists(RAW_API_PATH):
            logger.error(f"API file not found: {RAW_API_PATH}")
            return

        # =================================================
        # 2. LOAD API DATA
        # =================================================
        df = pd.read_csv(RAW_API_PATH)

        logger.info(f"New API rows: {len(df)}")

        # =================================================
        # 3. CLEAN COLUMN NAMES
        # =================================================
        df.columns = df.columns.str.strip()

        # =================================================
        # 4. DATETIME CLEANING
        # =================================================
        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        df["datetime"] = df["datetime"].dt.tz_localize(None)

        df = df.dropna(subset=["datetime"])

        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=["datetime"])

        logger.info(f"Rows after datetime cleaning: {len(df)}")

        # =================================================
        # 5. HANDLE MISSING VALUES
        # =================================================
        numeric_cols = df.select_dtypes(
            include=["float64", "int64"]
        ).columns

        for col in numeric_cols:
            df.loc[:, col] = df[col].fillna(
                df[col].median()
            )

        # =================================================
        # 6. ROUND VALUES
        # =================================================
        df[numeric_cols] = df[numeric_cols].round(1)

        # =================================================
        # 7. FEATURE ENGINEERING
        # =================================================
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["hour"] = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek

        df["is_weekend"] = (
            df["day_of_week"].isin([5, 6]).astype(int)
        )

        df["temp_diff"] = (
            df["temperature_C"] - df["feels_like_C"]
        ).round(1)

        df["wind_category"] = pd.cut(
            df["wind_speed_kmh"],
            bins=[-1, 5, 15, 30, float("inf")],
            labels=["calm", "breeze", "windy", "storm"]
        )

        df["feels_hotter"] = (
            df["feels_like_C"] > df["temperature_C"]
        ).astype(int)

        df["high_humidity"] = (
            df["humidity_pct"] > 70
        ).astype(int)

        df["is_rain"] = (
            df["rainfall_mm"] > 0
        ).astype(int)

        logger.info("Feature engineering completed")

        # =================================================
        # 8. LOAD EXISTING FILE
        # =================================================
        if os.path.exists(FINAL_PATH):

            old_df = pd.read_csv(FINAL_PATH)

            old_df["datetime"] = pd.to_datetime(
                old_df["datetime"],
                errors="coerce"
            )

            logger.info(f"Existing rows: {len(old_df)}")

        else:

            logger.warning(
                "No existing processed file found. Creating new."
            )

            old_df = pd.DataFrame()

        # =================================================
        # 9. MERGE DATA
        # =================================================
        combined_df = pd.concat(
            [old_df, df],
            ignore_index=True
        )

        before = len(combined_df)

        combined_df = combined_df.drop_duplicates(
            subset=["datetime"],
            keep="last"
        )

        after = len(combined_df)

        logger.info(
            f"Removed duplicates: {before - after}"
        )

        # =================================================
        # 10. SORT DATA
        # =================================================
        combined_df = combined_df.sort_values(
            by="datetime"
        )

        combined_df = combined_df.reset_index(drop=True)

        # =================================================
        # 11. CREATE DIRECTORY
        # =================================================
        os.makedirs(
            os.path.dirname(FINAL_PATH),
            exist_ok=True
        )

        # =================================================
        # 12. SAVE FILE
        # =================================================
        combined_df.to_csv(
            FINAL_PATH,
            index=False,
            encoding="utf-8"
        )

        logger.info(f"Final rows: {len(combined_df)}")
        logger.info(f"File saved successfully!")
        logger.info(f"Saved path → {FINAL_PATH}")

    except Exception as e:

        logger.exception(
            f"Transformation failed: {e}"
        )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    run_transform_incremental()