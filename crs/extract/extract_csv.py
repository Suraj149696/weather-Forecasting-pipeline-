import pandas as pd
import os

def load_historical_data(filepath="data/raw/Mumbai_Weather_Data_2010_to_2026.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    print("CSV Loaded Successfully ✅")
    return df


# 👇 ADD THIS
if __name__ == "__main__":
    df = load_historical_data()
    print(df.head())