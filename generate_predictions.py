import pandas as pd
import numpy as np
import os
from pathlib import Path

# Absolute path to the data folder
DATA_DIR = Path(r"f:\stock-prediction-project\data")

def generate_predictions():
    print(f"Creating folder: {DATA_DIR}")
    DATA_DIR.mkdir(exist_ok=True)  # Force-create the folder
    
    dates = pd.date_range(start="2023-01-01", periods=100)
    models = {
        "arima": 0,
        "sarima": 5,
        "prophet": 10,
        "lstm": 15
    }
    
    for model, offset in models.items():
        file_path = DATA_DIR / f"{model}_predictions.csv"
        print(f"Creating: {file_path}")
        pd.DataFrame({
            'Date': dates,
            'Prediction': np.random.randn(100).cumsum() + offset
        }).to_csv(file_path, index=False)
    
    print("\nFiles created:")
    print(*os.listdir(DATA_DIR), sep="\n")

if __name__ == "__main__":
    generate_predictions()