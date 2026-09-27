from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
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


# ---------------------------------
# Load data
# ---------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ---------------------------------
# Temporal split
# ---------------------------------

train = df[
    df["date"] < pd.Timestamp("2024-07-01")
].copy()

test = df[
    df["date"] >= pd.Timestamp("2024-07-01")
].copy()


# ---------------------------------
# Severity thresholds from TRAIN only
# ---------------------------------

q50 = train[
    "daily_peak_outage"
].quantile(0.50)

q75 = train[
    "daily_peak_outage"
].quantile(0.75)

q90 = train[
    "daily_peak_outage"
].quantile(0.90)


print("\nSeverity thresholds:")
print("Low < ", round(q50, 2))
print("Moderate < ", round(q75, 2))
print("High < ", round(q90, 2))
print("Critical >= ", round(q90, 2))


def make_severity(value):

    if value < q50:
        return "Low"

    elif value < q75:
        return "Moderate"

    elif value < q90:
        return "High"

    else:
        return "Critical"


train["severity"] = train[
    "daily_peak_outage"
].apply(make_severity)

test["severity"] = test[
    "daily_peak_outage"
].apply(make_severity)


print("\nTraining severity distribution:")
print(
    train["severity"].value_counts()
)

print("\nTest severity distribution:")
print(
    test["severity"].value_counts()
)


# ---------------------------------
# Features
# ---------------------------------

features = [
    "storm_event",
    "storm_count",

    "is_hurricane",
    "is_derecho",
    "is_tornado",
    "is_thunderstorm_wind",
    "is_flood",

    "storm_recent_7d",
    "storm_count_7d",

    "is_hurricane_recent_7d",
    "is_derecho_recent_7d",
    "is_tornado_recent_7d",
    "is_thunderstorm_wind_recent_7d",
    "is_flood_recent_7d",

    "month",
    "day_of_year",
    "day_of_week",

    "outage_lag_1d",
    "outage_mean_7d",
    "outage_max_7d",
]


X_train = train[features]
y_train = train["severity"]

X_test = test[features]
y_test = test["severity"]


# ---------------------------------
# Candidate models
# ---------------------------------

models = {

    "logistic_regression": Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                )
            ),
        ]
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    ),
}


results = []
trained_models = {}


# ---------------------------------
# Train and evaluate
# ---------------------------------

for name, model in models.items():

    print(
        f"\nTraining severity model: {name}"
    )

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )

    results.append(
        {
            "model": name,
            "balanced_accuracy":
                balanced_acc,
            "macro_f1":
                macro_f1,
        }
    )

    trained_models[name] = model

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y_test,
            pred,
            zero_division=0
        )
    )

    print(
        "Confusion matrix:"
    )

    labels = [
        "Low",
        "Moderate",
        "High",
        "Critical",
    ]

    print(
        confusion_matrix(
            y_test,
            pred,
            labels=labels
        )
    )


# ---------------------------------
# Compare models
# ---------------------------------

results_df = pd.DataFrame(
    results
)

print(
    "\nSeverity model comparison:"
)

print(
    results_df.to_string(
        index=False
    )
)


best_name = (
    results_df
    .sort_values(
        "macro_f1",
        ascending=False
    )
    .iloc[0]["model"]
)

best_model = trained_models[
    best_name
]


print(
    "\nBest severity model:"
)

print(
    best_name
)


# ---------------------------------
# Beryl evaluation
# ---------------------------------

beryl = test[
    test["date"]
    == pd.Timestamp("2024-07-08")
].copy()

beryl_pred = best_model.predict(
    beryl[features]
)[0]

beryl_actual = beryl[
    "severity"
].iloc[0]


print(
    "\nBeryl actual severity:"
)

print(
    beryl_actual
)

print(
    "\nBeryl predicted severity:"
)

print(
    beryl_pred
)


# ---------------------------------
# Save model
# ---------------------------------

joblib.dump(
    {
        "model":
            best_model,

        "features":
            features,

        "severity_thresholds":
            {
                "q50": q50,
                "q75": q75,
                "q90": q90,
            },

        "labels":
            [
                "Low",
                "Moderate",
                "High",
                "Critical",
            ],
    },
    MODEL_DIR
    / "severity_classifier.joblib"
)


print(
    "\nSaved severity classifier."
)