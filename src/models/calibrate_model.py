from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

BASE_MODEL_FILE = Path(
    "src/models/artifacts/best_model.joblib"
)

META_FILE = Path(
    "src/models/artifacts/model_metadata.joblib"
)

OUTPUT_FILE = Path(
    "src/models/artifacts/calibrated_model.joblib"
)


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

base_model_template = joblib.load(
    BASE_MODEL_FILE
)

metadata = joblib.load(
    META_FILE
)

features = metadata["features"]


# -----------------------------
# Temporal split
# Jan-Apr = train
# May-Jun = calibration
# Jul-Dec = final test
# -----------------------------

train = df[
    df["date"] < pd.Timestamp("2024-05-01")
].copy()

calibration = df[
    (df["date"] >= pd.Timestamp("2024-05-01"))
    & (df["date"] < pd.Timestamp("2024-07-01"))
].copy()

test = df[
    df["date"] >= pd.Timestamp("2024-07-01")
].copy()


# -----------------------------
# Define target only from train
# -----------------------------

outage_threshold = train[
    "daily_peak_outage"
].quantile(0.90)

for subset in [train, calibration, test]:

    subset["major_outage"] = (
        subset["daily_peak_outage"]
        >= outage_threshold
    ).astype(int)


# -----------------------------
# Fit base model
# -----------------------------

base_model = clone(
    base_model_template
)

base_model.fit(
    train[features],
    train["major_outage"]
)


# -----------------------------
# Platt calibration
# -----------------------------

cal_scores = base_model.decision_function(
    calibration[features]
).reshape(-1, 1)

calibrator = LogisticRegression(
    random_state=42
)

calibrator.fit(
    cal_scores,
    calibration["major_outage"]
)


# -----------------------------
# Test probabilities
# -----------------------------

test_scores = base_model.decision_function(
    test[features]
).reshape(-1, 1)

raw_prob = base_model.predict_proba(
    test[features]
)[:, 1]

calibrated_prob = calibrator.predict_proba(
    test_scores
)[:, 1]


raw_brier = brier_score_loss(
    test["major_outage"],
    raw_prob
)

calibrated_brier = brier_score_loss(
    test["major_outage"],
    calibrated_prob
)


print("\nRaw Brier score:")
print(round(raw_brier, 4))

print("\nCalibrated Brier score:")
print(round(calibrated_brier, 4))


# -----------------------------
# Beryl case
# -----------------------------

beryl = test[
    test["date"] == pd.Timestamp("2024-07-08")
]

beryl_score = base_model.decision_function(
    beryl[features]
).reshape(-1, 1)

beryl_raw = base_model.predict_proba(
    beryl[features]
)[0, 1]

beryl_calibrated = calibrator.predict_proba(
    beryl_score
)[0, 1]


print("\nBeryl raw probability:")
print(round(beryl_raw, 4))

print("\nBeryl calibrated probability:")
print(round(beryl_calibrated, 4))


# -----------------------------
# Save calibrated system
# -----------------------------

joblib.dump(
    {
        "base_model": base_model,
        "calibrator": calibrator,
        "features": features,
        "outage_threshold": outage_threshold,
    },
    OUTPUT_FILE
)

print("\nSaved calibrated model:")
print(OUTPUT_FILE)