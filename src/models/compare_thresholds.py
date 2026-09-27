from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model_tuned.joblib"
)

META_FILE = Path(
    "src/models/artifacts/tuned_metadata.joblib"
)

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"],
)

model = joblib.load(MODEL_FILE)
metadata = joblib.load(META_FILE)

features = metadata["features"]
outage_threshold = metadata["outage_threshold"]

test = df[
    df["date"] >= pd.Timestamp("2024-07-01")
].copy()

test["major_outage"] = (
    test["daily_peak_outage"] >= outage_threshold
).astype(int)

probabilities = model.predict_proba(
    test[features]
)[:, 1]

for threshold in [0.25, 0.50, 0.65]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        test["major_outage"],
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        test["major_outage"],
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        test["major_outage"],
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        test["major_outage"],
        predictions,
    )

    print(f"\nThreshold = {threshold}")

    print(
        f"Precision = {precision:.3f}"
    )

    print(
        f"Recall = {recall:.3f}"
    )

    print(
        f"F1 = {f1:.3f}"
    )

    print("Confusion matrix:")
    print(cm)


beryl = test[
    test["date"] == pd.Timestamp("2024-07-08")
]

beryl_probability = model.predict_proba(
    beryl[features]
)[0, 1]

print("\nBeryl probability:")
print(round(beryl_probability, 4))