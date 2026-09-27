from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.csv.gz"))

if not files:
    raise FileNotFoundError(
        "No CSV or CSV.GZ file found in data/raw"
    )

file_path = files[0]

print(f"Reading: {file_path}")

df = pd.read_csv(file_path)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())