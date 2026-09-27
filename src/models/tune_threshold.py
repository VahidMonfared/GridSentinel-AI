from pathlib import Path
from sklearn.metrics import fbeta_score
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
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
    "src/models/artifacts/best_model.joblib"
)

META_FILE = Path(
    "src/models/artifacts/model_metadata.joblib"
)

OUTPUT_FILE = Path(
    "src/models/artifacts/threshold_results.csv"
)


# ----------------------------------
# Load data and model
# ----------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values("date").reset_index(drop=True)

base_model = joblib.load(
    MODEL_FILE
)

metadata = joblib.load(
    META_FILE
)

features = metadata["features"]


# ----------------------------------
# Temporal windows
# ----------------------------------

train_end = pd.Timestamp("2024-05-01")
validation_end = pd.Timestamp("2024-07-01")

train_df = df[
    df["date"] < train_end
].copy()

validation_df = df[
    (df["date"] >= train_end)
    & (df["date"] < validation_end)
].copy()

test_df = df[
    df["date"] >= validation_end
].copy()


# ----------------------------------
# Target threshold
# Learned only from early training
# ----------------------------------

outage_threshold = train_df[
    "daily_peak_outage"
].quantile(0.90)

for subset in [
    train_df,
    validation_df,
    test_df,
]:
    subset["major_outage"] = (
        subset["daily_peak_outage"]
        >= outage_threshold
    ).astype(int)


# ----------------------------------
# Train temporary model
# ----------------------------------

model = clone(base_model)

model.fit(
    train_df[features],
    train_df["major_outage"],
)


# ----------------------------------
# Validation probabilities
# ----------------------------------

val_prob = model.predict_proba(
    validation_df[features]
)[:, 1]

y_val = validation_df[
    "major_outage"
].values

print("\nValidation class distribution:")
print(pd.Series(y_val).value_counts())

# ----------------------------------
# Threshold search
# F2 emphasizes recall
# ----------------------------------

results = []

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

for threshold in thresholds:

    pred = (
        val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_val,
        pred,
        beta=2,
        zero_division=0,
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f2": f2,
        }
    )


results_df = pd.DataFrame(results)

# ----------------------------------
# Risk-based threshold policy
# Require high recall first
# ----------------------------------

MIN_RECALL = 0.90

eligible = results_df[
    results_df["recall"] >= MIN_RECALL
].copy()

if not eligible.empty:

    best_row = (
        eligible
        .sort_values(
            ["precision", "f2"],
            ascending=False,
        )
        .iloc[0]
    )

    selection_policy = (
        "Recall >= 0.90, then maximize precision"
    )

else:

    best_row = (
        results_df
        .sort_values(
            ["f2", "recall"],
            ascending=False,
        )
        .iloc[0]
    )

    selection_policy = (
        "Fallback: maximize F2"
    )

best_threshold = float(
    best_row["threshold"]
)

print("\nThreshold selection policy:")
print(selection_policy)

print("\nSelected threshold:")
print(best_threshold)

print("\nSelected validation metrics:")
print(best_row)


print("\nValidation threshold results:")
print(results_df.to_string(index=False))

print("\nSelected threshold:")
print(best_threshold)

print("\nSelected validation metrics:")
print(best_row)


# ----------------------------------
# Retrain on Jan-Jun
# ----------------------------------

final_train = df[
    df["date"] < validation_end
].copy()

final_train["major_outage"] = (
    final_train["daily_peak_outage"]
    >= outage_threshold
).astype(int)

final_model = clone(base_model)

final_model.fit(
    final_train[features],
    final_train["major_outage"],
)


# ----------------------------------
# Final test evaluation
# ----------------------------------

test_prob = final_model.predict_proba(
    test_df[features]
)[:, 1]

test_pred = (
    test_prob >= best_threshold
).astype(int)

y_test = test_df[
    "major_outage"
].values

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    test_pred
)


print("\nFinal test metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))

print("\nConfusion matrix:")
print(cm)


# ----------------------------------
# Beryl check
# ----------------------------------

beryl = test_df[
    test_df["date"]
    == pd.Timestamp("2024-07-08")
].copy()

beryl_prob = final_model.predict_proba(
    beryl[features]
)[0, 1]

beryl_pred = int(
    beryl_prob >= best_threshold
)

print("\nBeryl probability:")
print(round(beryl_prob, 4))

print("\nBeryl classification:")
print(beryl_pred)


# ----------------------------------
# Save threshold + model
# ----------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

joblib.dump(
    final_model,
    "src/models/artifacts/best_model_tuned.joblib"
)

joblib.dump(
    {
        "features": features,
        "outage_threshold": outage_threshold,
        "decision_threshold": best_threshold,
    },
    "src/models/artifacts/tuned_metadata.joblib"
)