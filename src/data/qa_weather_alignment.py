from pathlib import Path
import pandas as pd

MODEL_FILE = Path("data/processed/harris_model_dataset_2024.csv")

df = pd.read_csv(MODEL_FILE, parse_dates=["date"])

top_days = (
    df.sort_values("daily_peak_outage", ascending=False)
    .head(10)[
        [
            "date",
            "daily_peak_outage",
            "storm_event",
            "storm_types",
        ]
    ]
)

print("\nTop 10 outage days with weather alignment:")
print(top_days.to_string(index=False))

print("\nDays with very high outage but no weather event:")

q90 = df["daily_peak_outage"].quantile(0.90)

suspect = df[
    (df["daily_peak_outage"] >= q90)
    & (df["storm_event"] == 0)
][
    [
        "date",
        "daily_peak_outage",
        "storm_types",
    ]
]

print(suspect.to_string(index=False))