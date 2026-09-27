from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/harris_weather_2024.csv")

files = list(RAW_DIR.glob("*StormEvents*details*.csv"))

if not files:
    raise FileNotFoundError("NOAA Storm Events file not found.")

file_path = files[0]

df = pd.read_csv(file_path, low_memory=False)

DATE_FORMAT = "%d-%b-%y %H:%M:%S"

df["BEGIN_DATE_TIME"] = pd.to_datetime(
    df["BEGIN_DATE_TIME"],
    format=DATE_FORMAT,
    errors="coerce"
)

df["END_DATE_TIME"] = pd.to_datetime(
    df["END_DATE_TIME"],
    format=DATE_FORMAT,
    errors="coerce"
)

texas = df[
    df["STATE"].astype(str).str.strip().str.upper().eq("TEXAS")
].copy()

# Exact Harris County + Coastal Harris NOAA zone
harris = texas[
    (
        (texas["CZ_TYPE"].astype(str).str.strip().eq("C"))
        & (pd.to_numeric(texas["CZ_FIPS"], errors="coerce") == 201)
    )
    |
    (
        texas["CZ_NAME"]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("COASTAL HARRIS")
    )
].copy()

harris = harris.dropna(
    subset=["BEGIN_DATE_TIME", "END_DATE_TIME"]
)

rows = []

for _, row in harris.iterrows():

    dates = pd.date_range(
        row["BEGIN_DATE_TIME"].normalize(),
        row["END_DATE_TIME"].normalize(),
        freq="D"
    )

    for date in dates:
        rows.append(
            {
                "date": date,
                "event_type": row["EVENT_TYPE"],
                "cz_name": row["CZ_NAME"],
            }
        )

expanded = pd.DataFrame(rows)

if expanded.empty:
    raise ValueError("No Harris County NOAA events found.")

daily_weather = (
    expanded.groupby("date", as_index=False)
    .agg(
        storm_count=("event_type", "count"),
        storm_types=(
            "event_type",
            lambda x: " | ".join(sorted(set(x.astype(str))))
        ),
    )
)

daily_weather["storm_event"] = 1

official_events = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2024-05-16",
                "2024-07-08",
            ]
        ),
        "storm_count": [1, 1],
        "storm_types": [
            "Houston Derecho",
            "Hurricane Beryl",
        ],
        "storm_event": [1, 1],
    }
)

daily_weather = pd.concat(
    [daily_weather, official_events],
    ignore_index=True,
)

daily_weather = (
    daily_weather
    .groupby("date", as_index=False)
    .agg(
        storm_count=("storm_count", "sum"),
        storm_types=(
            "storm_types",
            lambda x: " | ".join(sorted(set(x.astype(str))))
        ),
        storm_event=("storm_event", "max"),
    )
    .sort_values("date")
)

daily_weather.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nHarris NOAA source events:")
print(len(harris))

print("\nDaily weather rows:")
print(len(daily_weather))

print("\nEvents around Beryl:")
print(
    daily_weather[
        daily_weather["date"].between(
            "2024-07-06",
            "2024-07-10"
        )
    ].to_string(index=False)
)

print("\nTop event types:")
print(
    harris["EVENT_TYPE"]
    .value_counts()
    .head(15)
)