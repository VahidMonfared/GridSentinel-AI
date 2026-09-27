from pathlib import Path
import pandas as pd

OUTAGE_FILE = Path("data/processed/harris_outages_daily_2024.csv")
WEATHER_FILE = Path("data/processed/harris_weather_2024.csv")
OUTPUT_FILE = Path("data/processed/harris_model_dataset_2024.csv")

outage = pd.read_csv(OUTAGE_FILE)
weather = pd.read_csv(WEATHER_FILE)

outage["date"] = pd.to_datetime(outage["date"])
weather["date"] = pd.to_datetime(weather["date"])

df = outage.merge(
    weather,
    on="date",
    how="left"
)

df["storm_event"] = df["storm_event"].fillna(0).astype(int)
df["storm_count"] = df["storm_count"].fillna(0).astype(int)
df["storm_types"] = df["storm_types"].fillna("None")

df["is_hurricane"] = (
    df["storm_types"]
    .str.contains(
        "Hurricane",
        case=False,
        na=False
    )
).astype(int)

df["is_derecho"] = (
    df["storm_types"]
    .str.contains(
        "Derecho",
        case=False,
        na=False
    )
).astype(int)

df["is_tornado"] = (
    df["storm_types"]
    .str.contains(
        "Tornado",
        case=False,
        na=False
    )
).astype(int)

df["is_thunderstorm_wind"] = (
    df["storm_types"]
    .str.contains(
        "Thunderstorm Wind",
        case=False,
        na=False
    )
).astype(int)

df["is_flood"] = (
    df["storm_types"]
    .str.contains(
        "Flood",
        case=False,
        na=False
    )
).astype(int)

df = df.sort_values("date").reset_index(drop=True)

df["storm_recent_7d"] = (
    df["storm_event"]
    .rolling(window=7, min_periods=1)
    .max()
    .astype(int)
)

df["storm_count_7d"] = (
    df["storm_count"]
    .rolling(window=7, min_periods=1)
    .sum()
)

for col in [
    "is_hurricane",
    "is_derecho",
    "is_tornado",
    "is_thunderstorm_wind",
    "is_flood",
]:
    df[f"{col}_recent_7d"] = (
        df[col]
        .rolling(
            window=7,
            min_periods=1
        )
        .max()
        .astype(int)
    )

df = df.sort_values("date").reset_index(drop=True)

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["day_of_week"] = df["date"].dt.dayofweek

df["outage_lag_1d"] = df["daily_peak_outage"].shift(1)

df["outage_mean_7d"] = (
    df["daily_peak_outage"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

df["outage_max_7d"] = (
    df["daily_peak_outage"]
    .shift(1)
    .rolling(window=7)
    .max()
)

df = df.dropna(
    subset=[
        "outage_lag_1d",
        "outage_mean_7d",
        "outage_max_7d",
    ]
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nStorm days:")
print(df["storm_event"].value_counts())

print("\nTop outage days:")
print(
    df.sort_values(
        "daily_peak_outage",
        ascending=False
    )[
        [
            "date",
            "daily_peak_outage",
            "daily_peak_outage_rate",
            "storm_event",
            "storm_count",
            "storm_types",
        ]
    ].head(10)
)