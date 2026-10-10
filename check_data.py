
import pandas as pd
from pathlib import Path

file_path = Path("data/Churn_Modelling.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Dataset not found inside the data folder."
    )

df = pd.read_csv(file_path)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- CHURN DISTRIBUTION ---")
print(df["Exited"].value_counts())

print("\n--- CHURN PERCENTAGES ---")
print((df["Exited"].value_counts(normalize=True) * 100).round(2))
