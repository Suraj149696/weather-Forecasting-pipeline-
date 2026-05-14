# 🌦️ Weather Data Forecasting Pipeline (End-to-End ML Project)

This project is an **end-to-end automated weather forecasting pipeline** that collects, processes, and predicts weather data using Machine Learning (LSTM).

It runs automatically using **GitHub Actions CI/CD workflow**.

- Collects weather data from API + CSV
- Cleans and transforms data
- Engineers features
- Trains an LSTM deep learning model
- Generates 24-hour weather forecasts
- Automatically runs using **GitHub Actions (CI/CD pipeline)**

---

## 🚀 Project Workflow

### 1. Data Extraction
- Fetch weather data using API (`Open-Meteo`)
- Load historical CSV data

### 2. Data Transformation
- Clean missing values
- Feature engineering:
  - Hour, month, day
  - Temperature difference
  - Wind categories
  - Weekend flag
- Save final dataset:data/processed/final_weather.csv


### 3. Model Training
- LSTM neural network trained on time-series data
- Scaled using `MinMaxScaler`
- Model saved as:
models/best_model.keras
models/scaler_params.pkl



### 4. Forecasting
- Uses last 168 hours (7 days) of data
- Predicts next 24 hours temperature
- Output saved as:data/Forecast/forecast.json


---

## ⚙️ Tech Stack

- Python 🐍
- Pandas / NumPy
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Open-Meteo API
- GitHub Actions (CI/CD)
- Logging system

---

## 📁 Project Structure

weather-data-pipeline/
│
├── .env
│
├── .github/
│ └── workflows/
│ └── forecast.yml
│
├── crs/ # Core pipeline logic
│ ├── extract/ # Data extraction
│ │ ├── extract_csv.py
│ │ └── extract_api.py
│
│ ├── transform/ # Data cleaning & feature engineering
│ │ ├── transform_csv.py
│ │ └── transform_api.py
│
│ ├── model/ # ML model
│ │ ├── train.py
│ │ └── predict.py
│
│ ├── utils/ # Common utilities
│ │ ├── config.py
│ │ └── logger.py
│
├── data/ # Data storage
│ ├── raw/
│ │ ├── weather_2010_2025.csv
│ │ └── api_weather.csv
│ │
│ └── processed/
│ └── final_weather.csv
│
├── models/ # Saved ML model
│ ├── best_model.keras
│ └── scaler_params.pkl
│
├── notebooks/ # EDA & experimentation
│ └── eda.ipynb
│
├── logs/
│ └── pipeline.log
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Tech Stack

- Python 🐍
- Pandas, NumPy
- TensorFlow / Keras (LSTM)
- Scikit-learn
- Open-Meteo API
- Logging system
- GitHub Actions (CI/CD)

---

## 🔁 GitHub Actions Automation

This project runs automatically using GitHub Actions.

### ⏰ Schedule
- Daily execution (cron-based)
- Example: 6:00 AM IST

### Workflow Steps:
1. Extract API data
2. Transform dataset
3. Run LSTM model prediction
4. Generate forecast JSON
5. Push updates to GitHub repository

---

## 📊 Output Example

```json
{
  "generated_at": "2026-05-08 06:00",
  "base_datetime": "2026-05-07 23:00",
  "unit": "C",
  "forecast_hours": 24,
  "forecast": [
    {
      "horizon": "h+01",
      "datetime": "2026-05-08 00:00",
      "temperature_C": 27.9
    }
  ]
}
