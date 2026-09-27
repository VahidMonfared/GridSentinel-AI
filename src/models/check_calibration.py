from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model_tuned.joblib"
)

META_FILE = Path(
    "src/models/artifacts/tuned_metadata.joblib"
)

FIGURE_DIR = Path("docs/figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Load data and model
# -----------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"],
)

model = joblib.load(
    MODEL_FILE
)

metadata = joblib.load(
    META_FILE
)

features = metadata["features"]
outage_threshold = metadata["outage_threshold"]


# -----------------------------
# Final test period
# -----------------------------

test = df[
    df["date"] >= pd.Timestamp("2024-07-01")
].copy()

test["major_outage"] = (
    test["daily_peak_outage"]
    >= outage_threshold
).astype(int)


X_test = test[features]
y_test = test["major_outage"]


# -----------------------------
# Predicted probabilities
# -----------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]


# -----------------------------
# Brier score
# -----------------------------

brier = brier_score_loss(
    y_test,
    probabilities
)

print("\nBrier score:")
print(round(brier, 4))


# -----------------------------
# Calibration curve
# -----------------------------

fraction_positive, mean_predicted = (
    calibration_curve(
        y_test,
        probabilities,
        n_bins=5,
        strategy="quantile",
    )
)


results = pd.DataFrame(
    {
        "mean_predicted_probability":
            mean_predicted,
        "observed_positive_rate":
            fraction_positive,
    }
)

print("\nCalibration table:")
print(results.to_string(index=False))


# -----------------------------
# Save plot
# -----------------------------

plt.figure(figsize=(6, 6))

plt.plot(
    mean_predicted,
    fraction_positive,
    marker="o",
    label="GridSentinel",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Predicted probability")
plt.ylabel("Observed major-outage rate")
plt.title("GridSentinel Probability Calibration")
plt.legend()
plt.tight_layout()

OUTPUT_FILE = (
    FIGURE_DIR
    / "calibration_curve.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
)

plt.close()

print("\nSaved calibration plot:")
print(OUTPUT_FILE)


# -----------------------------
# Beryl check
# -----------------------------

beryl = test[
    test["date"]
    == pd.Timestamp("2024-07-08")
]

beryl_probability = model.predict_proba(
    beryl[features]
)[0, 1]

print("\nBeryl probability:")
print(round(beryl_probability, 4))