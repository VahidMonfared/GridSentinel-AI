from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

files = list(RAW_DIR.glob("*StormEvents*details*.csv")) + \
        list(RAW_DIR.glob("*StormEvents*details*.csv.gz"))

print("\nNOAA files found:")
for f in files:
    print(f)

if not files:
    raise FileNotFoundError("NOAA details file not found.")

file_path = files[0]

print(f"\nUsing file: {file_path}")

df = pd.read_csv(file_path, low_memory=False)

print("\nRows:")
print(len(df))

print("\nDate examples:")
print(df["BEGIN_DATE_TIME"].head())

df["BEGIN_DATE_TIME"] = pd.to_datetime(
    df["BEGIN_DATE_TIME"],
    errors="coerce"
)

print("\nDate range:")
print(df["BEGIN_DATE_TIME"].min())
print(df["BEGIN_DATE_TIME"].max())

texas = df[
    df["STATE"]
    .astype(str)
    .str.strip()
    .str.upper()
    .eq("TEXAS")
].copy()

print("\nTexas rows:")
print(len(texas))

print("\nWFO values containing HGX:")
print(
    texas[
        texas["WFO"]
        .astype(str)
        .str.upper()
        .str.contains("HGX", na=False)
    ]["WFO"].value_counts()
)

print("\nNames containing HARR:")
print(
    texas[
        texas["CZ_NAME"]
        .astype(str)
        .str.upper()
        .str.contains("HARR", na=False)
    ][
        ["CZ_NAME", "CZ_TYPE", "CZ_FIPS", "WFO"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nTexas events July 8, 2024:")
july8 = texas[
    texas["BEGIN_DATE_TIME"].dt.date
    == pd.Timestamp("2024-07-08").date()
]

print(
    july8[
        [
            "BEGIN_DATE_TIME",
            "EVENT_TYPE",
            "CZ_TYPE",
            "CZ_FIPS",
            "CZ_NAME",
            "WFO",
        ]
    ]
    .sort_values("BEGIN_DATE_TIME")
    .to_string(index=False)
)