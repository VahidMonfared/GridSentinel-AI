from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


DATA_FILE = Path("data/processed/harris_model_dataset_2024.csv")
MODEL_DIR = Path("src/models/artifacts")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"],
)

df = df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# 2. Time-based split
# Train = Jan-Jun
# Test  = Jul-Dec
# --------------------------------------------------

split_date = pd.Timestamp("2024-07-01")

train_df = df[df["date"] < split_date].copy()
test_df = df[df["date"] >= split_date].copy()


# --------------------------------------------------
# 3. Target definition
# Threshold learned ONLY from training data
# --------------------------------------------------

threshold = train_df[
    "daily_peak_outage"
].quantile(0.90)

train_df["major_outage"] = (
    train_df["daily_peak_outage"] >= threshold
).astype(int)

test_df["major_outage"] = (
    test_df["daily_peak_outage"] >= threshold
).astype(int)


# --------------------------------------------------
# 4. Features
# --------------------------------------------------

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
X_train = train_df[features]
y_train = train_df["major_outage"]

X_test = test_df[features]
y_test = test_df["major_outage"]


print("\nTraining rows:")
print(len(train_df))

print("\nTest rows:")
print(len(test_df))

print("\nMajor outage threshold:")
print(round(threshold, 2))

print("\nTrain class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# --------------------------------------------------
# 5. Models
# --------------------------------------------------

models = {

    "logistic_regression": Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    ),

    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    ),
}


# --------------------------------------------------
# 6. MLflow
# --------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(
    "GridSentinel_Outage_Risk"
)

results = []


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    with mlflow.start_run(
        run_name=model_name
    ):

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)


        if y_test.nunique() == 2:
            auc = roc_auc_score(
                y_test,
                probabilities,
            )
        else:
            auc = float("nan")


        pr_auc = average_precision_score(
            y_test,
            probabilities,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        brier = brier_score_loss(
            y_test,
            probabilities,
        )


        mlflow.log_param(
            "split_date",
            str(split_date.date()),
        )

        mlflow.log_param(
            "major_outage_threshold",
            float(threshold),
        )

        mlflow.log_param(
            "features",
            ",".join(features),
        )

        mlflow.log_metrics(
            {
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "brier_score": brier,
            }
        )


        # Save model safely with joblib
        model_path = (
            MODEL_DIR
            / f"{model_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        # Log saved file as MLflow artifact
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )


        results.append(
            {
                "model": model_name,
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "brier_score": brier,
            }
        )


# --------------------------------------------------
# 7. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False,
)

print("\nModel comparison:")
print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# 8. Save best model
# --------------------------------------------------

best_model_name = (
    results_df.iloc[0]["model"]
)

best_model = models[
    best_model_name
]

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib",
)

joblib.dump(
    {
        "features": features,
        "threshold": threshold,
        "best_model_name": best_model_name,
        "split_date": split_date,
    },
    MODEL_DIR / "model_metadata.joblib",
)


results_df.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False,
)


print("\nBest model:")
print(best_model_name)

print("\nSaved best model:")
print(
    MODEL_DIR / "best_model.joblib"
)