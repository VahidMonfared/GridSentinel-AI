from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


DATA_FILE = Path(
    "data/processed/harris_model_dataset_2024.csv"
)

MODEL_FILE = Path(
    "src/models/artifacts/best_model.joblib"
)

META_FILE = Path(
    "src/models/artifacts/model_metadata.joblib"
)

FIGURE_DIR = Path("docs/figures")

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------
# Load data and trained model
# -----------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

model = joblib.load(
    MODEL_FILE
)

metadata = joblib.load(
    META_FILE
)

features = metadata["features"]


# -----------------------------
# Select real Beryl case
# -----------------------------

case = df[
    df["date"] == pd.Timestamp("2024-07-08")
].copy()

if case.empty:
    raise ValueError(
        "July 8, 2024 case not found."
    )

X = df[features].copy()
X_case = case[features].copy()


# -----------------------------
# Prediction
# -----------------------------

probability = model.predict_proba(
    X_case
)[0, 1]

print("\nDate:")
print(case["date"].iloc[0])

print("\nActual peak outage:")
print(
    int(
        case["daily_peak_outage"].iloc[0]
    )
)

print("\nPredicted major-outage probability:")
print(
    round(probability, 4)
)


# -----------------------------
# Access logistic regression
# pipeline components
# -----------------------------

preprocessor = model[:-1]

classifier = model.named_steps[
    "model"
]

X_transformed = (
    preprocessor.transform(X)
)

X_case_transformed = (
    preprocessor.transform(X_case)
)


# -----------------------------
# SHAP explanation
# -----------------------------

explainer = shap.LinearExplainer(
    classifier,
    X_transformed
)

shap_values = explainer(
    X_case_transformed
)

values = shap_values.values[0]


explanation = pd.DataFrame(
    {
        "feature": features,
        "feature_value":
            X_case.iloc[0].values,
        "shap_value": values,
    }
)

explanation["importance"] = (
    explanation["shap_value"].abs()
)

explanation = explanation.sort_values(
    "importance",
    ascending=False
)


print("\nTop explanation features:")

print(
    explanation[
        [
            "feature",
            "feature_value",
            "shap_value",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# -----------------------------
# Save SHAP plot
# -----------------------------

shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    show=False
)

plt.tight_layout()

OUTPUT_FILE = (
    FIGURE_DIR
    / "beryl_local_shap_explanation.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nSaved explanation figure:")
print(OUTPUT_FILE)