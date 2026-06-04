# 🌦️ Weather Data Forecasting Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white)

**End-to-End Automated ML Pipeline · LSTM Deep Learning · 24-Hour Temperature Forecasting**  
*Time Series · Feature Engineering · API Integration · CI/CD Automation*

</div>

---

## 📌 Project Summary

> An **end-to-end automated weather forecasting pipeline** that extracts real-time weather data from an API, processes historical records, engineers time-series features, trains an **LSTM deep learning model**, and generates **24-hour temperature forecasts** — all running automatically every day via **GitHub Actions CI/CD**.

No manual execution. No manual updates. The pipeline runs, trains, predicts, and pushes results to the repository on its own — every single day.

---

## 🚀 Project Highlights

| Feature | Detail |
|---|---|
| 🌐 Data Source | Open-Meteo API + Historical CSV (2010–2025) |
| 🧠 Model | LSTM Neural Network (Deep Learning) |
| 📅 Forecast Horizon | **Next 24 Hours** (hourly predictions) |
| 🔁 Input Window | Last **168 hours (7 days)** of weather data |
| ⚙️ Automation | **GitHub Actions CI/CD** — runs daily at 6:00 AM IST |
| 📤 Output | `forecast.json` auto-pushed to repository |

---

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   GitHub Actions (Daily Trigger)            │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         1. DATA EXTRACTION        │
          │  Open-Meteo API + Historical CSV  │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │       2. DATA TRANSFORMATION      │
          │  Cleaning · Feature Engineering   │
          │  Hour / Month / Day / Wind / Temp │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         3. MODEL TRAINING         │
          │   LSTM · MinMaxScaler · Keras     │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │         4. FORECASTING            │
          │  Last 168hrs → Next 24hrs Temp    │
          └────────────────┬─────────────────┘
                           │
          ┌────────────────▼─────────────────┐
          │     5. OUTPUT & AUTO-PUSH         │
          │   forecast.json → GitHub Repo     │
          └──────────────────────────────────┘
```

---

## 📋 Pipeline Steps in Detail

### Step 1 — Data Extraction
- Fetches real-time weather data from **Open-Meteo API** (`extract_api.py`)
- Loads historical weather records from CSV — **2010 to 2025** (`extract_csv.py`)
- Both sources merged to form a comprehensive time-series dataset

### Step 2 — Data Transformation & Feature Engineering
- Handles missing values and cleans raw records
- Engineers time-based and weather-based features:

| Feature | Description |
|---|---|
| `hour` | Hour of the day (0–23) |
| `month` | Month of the year (1–12) |
| `day` | Day of the week |
| `temp_difference` | Temperature delta from previous hour |
| `wind_category` | Categorical wind speed classification |
| `weekend_flag` | Binary flag for weekends |

- Final processed dataset saved to: `data/processed/final_weather.csv`

### Step 3 — LSTM Model Training
- Time-series data scaled using **MinMaxScaler**
- **LSTM neural network** trained on sequential temperature patterns
- Model artefacts saved for reuse:
  - `models/best_model.keras` — trained LSTM model
  - `models/scaler_params.pkl` — scaler parameters for inverse transformation

### Step 4 — 24-Hour Forecasting
- Takes the **last 168 hours (7 days)** of processed data as input
- LSTM model predicts **next 24 hours** of hourly temperature
- Forecast output saved to: `data/Forecast/forecast.json`

### Step 5 — CI/CD Auto-Push
- GitHub Actions workflow commits and pushes `forecast.json` back to the repository automatically
- Every morning the repo contains fresh, up-to-date predictions

---

## 📊 Forecast Output Sample

```json
{
  "generated_at": "2026-05-08 06:00",
  "base_datetime": "2026-05-07 23:00",
  "unit": "C",
  "forecast_hours": 24,
  "forecast": [
    { "horizon": "h+01", "datetime": "2026-05-08 00:00", "temperature_C": 27.9 },
    { "horizon": "h+02", "datetime": "2026-05-08 01:00", "temperature_C": 27.4 },
    { "horizon": "h+03", "datetime": "2026-05-08 02:00", "temperature_C": 26.8 }
  ]
}
```

---

## ⚙️ GitHub Actions — CI/CD Automation

The entire pipeline is automated using a **GitHub Actions workflow** (`forecast.yml`).

```yaml
Trigger:   Scheduled daily (cron) — 6:00 AM IST
Steps:
  1. Checkout repository
  2. Set up Python environment
  3. Install dependencies (requirements.txt)
  4. Run extract → transform → train → predict
  5. Commit & push forecast.json to repository
```

**Zero manual intervention required.** The pipeline is fully self-sustaining.

---

## 📁 Project Structure

```
weather-data-pipeline/
│
├── .github/
│   └── workflows/
│       └── forecast.yml          ← GitHub Actions CI/CD workflow
│
├── crs/                          ← Core pipeline logic
│   ├── extract/
│   │   ├── extract_csv.py        ← Load historical CSV data
│   │   └── extract_api.py        ← Fetch from Open-Meteo API
│   │
│   ├── transform/
│   │   ├── transform_csv.py      ← Clean & engineer CSV features
│   │   └── transform_api.py      ← Clean & engineer API features
│   │
│   ├── model/
│   │   ├── train.py              ← LSTM model training
│   │   └── predict.py            ← 24-hour forecast generation
│   │
│   └── utils/
│       ├── config.py             ← Configuration & constants
│       └── logger.py             ← Logging system
│
├── data/
│   ├── raw/
│   │   ├── weather_2010_2025.csv ← Historical weather data
│   │   └── api_weather.csv       ← Latest API pull
│   │
│   └── processed/
│       └── final_weather.csv     ← Cleaned & engineered dataset
│
├── models/
│   ├── best_model.keras          ← Trained LSTM model
│   └── scaler_params.pkl         ← MinMaxScaler parameters
│
├── data/Forecast/
│   └── forecast.json             ← Latest 24-hr prediction (auto-updated)
│
├── notebooks/
│   └── eda.ipynb                 ← Exploratory data analysis
│
├── logs/
│   └── pipeline.log              ← Pipeline execution logs
│
├── .env                          ← API keys (not committed)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Data Processing | Pandas, NumPy |
| ML Utilities | Scikit-learn (MinMaxScaler) |
| Data Source | Open-Meteo API |
| Automation | GitHub Actions (CI/CD) |
| Logging | Python Logging Module |
| Serialisation | Pickle (scaler), Keras (model) |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Suraj149696/weather-Forecasting-pipeline-.git

# 2. Navigate into the project
cd weather-Forecasting-pipeline-

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key to .env
echo "API_KEY=your_open_meteo_key" > .env

# 5. Run the full pipeline manually
python crs/extract/extract_api.py
python crs/transform/transform_api.py
python crs/model/train.py
python crs/model/predict.py
```

> The GitHub Actions workflow runs this automatically every day — manual execution is only needed for local testing.

---

## 💡 Key Learnings & Skills Demonstrated

- ✅ **End-to-end ML pipeline design** — from raw data to automated predictions
- ✅ **LSTM time-series modelling** — sequential pattern learning for forecasting
- ✅ **Feature engineering** for temporal and weather data
- ✅ **API integration** — real-time data ingestion from Open-Meteo
- ✅ **CI/CD with GitHub Actions** — fully automated, production-style workflow
- ✅ **Modular code architecture** — clean separation of extract, transform, model, utils
- ✅ **Logging system** — production-grade pipeline observability

---

## 📬 Connect with Me

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR-LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Suraj149696)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:YOUR-EMAIL)

*Open to Data Analyst | Business Analyst | MIS Analyst | Supply Chain Analyst roles*

</div>

---

<div align="center">
  <sub>Built with Python · TensorFlow · GitHub Actions · Open-Meteo API · End-to-End ML Pipeline</sub>
</div>
