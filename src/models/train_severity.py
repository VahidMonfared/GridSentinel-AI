from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_DIR = Path(
    "src/models/artifacts"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


# -----------------------------
# Temporal split
# -----------------------------

train = df[
    df["date"] < pd.Timestamp("2024-07-01")
].copy()

test = df[
    df["date"] >= pd.Timestamp("2024-07-01")
].copy()


# -----------------------------
# Features
# -----------------------------

features = [
    "storm_event",
    "storm_count",
    "storm_recent_7d",
    "storm_count_7d",
    "month",
    "day_of_year",
    "day_of_week",
    "outage_lag_1d",
    "outage_mean_7d",
    "outage_max_7d",
]


X_train = train[features]
X_test = test[features]


# -----------------------------
# Log target
# Reduces domination by extreme events
# -----------------------------

y_train_log = np.log1p(
    train["daily_peak_outage"]
)


# -----------------------------
# Severity model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
)

model.fit(
    X_train,
    y_train_log
)


# -----------------------------
# Predictions
# -----------------------------

pred_log = model.predict(
    X_test
)

pred = np.expm1(
    pred_log
)

actual = test[
    "daily_peak_outage"
].values


mae = mean_absolute_error(
    actual,
    pred
)

r2 = r2_score(
    actual,
    pred
)


print("\nSeverity model MAE:")
print(round(mae, 2))

print("\nSeverity model R2:")
print(round(r2, 4))


# -----------------------------
# Training-based severity bands
# -----------------------------

q50 = train[
    "daily_peak_outage"
].quantile(0.50)

q75 = train[
    "daily_peak_outage"
].quantile(0.75)

q90 = train[
    "daily_peak_outage"
].quantile(0.90)


def severity_label(value):

    if value < q50:
        return "Low"

    if value < q75:
        return "Moderate"

    if value < q90:
        return "High"

    return "Critical"


# -----------------------------
# Beryl case
# -----------------------------

beryl = test[
    test["date"]
    == pd.Timestamp("2024-07-08")
].copy()

beryl_pred_log = model.predict(
    beryl[features]
)[0]

beryl_pred = np.expm1(
    beryl_pred_log
)

beryl_actual = beryl[
    "daily_peak_outage"
].iloc[0]


print("\nBeryl actual outage:")
print(int(beryl_actual))

print("\nBeryl predicted outage:")
print(int(beryl_pred))

print("\nBeryl predicted severity:")
print(
    severity_label(
        beryl_pred
    )
)

print("\nBeryl actual severity:")
print(
    severity_label(
        beryl_actual
    )
)


# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    {
        "model": model,
        "features": features,
        "severity_thresholds": {
            "q50": q50,
            "q75": q75,
            "q90": q90,
        },
    },
    MODEL_DIR
    / "severity_model.joblib"
)

print("\nSaved severity model.")